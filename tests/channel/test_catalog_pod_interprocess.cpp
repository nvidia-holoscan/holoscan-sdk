/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sys/wait.h>
#include <unistd.h>

#include <csignal>

#include <gtest/gtest.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>

#include <holomq/context_builder.hpp>
#include <holomq/errors.hpp>
#include <holomq/publisher.hpp>
#include <holomq/pubsub_options.hpp>
#include <holoscan/channel/endpoint_catalog_builder.hpp>

#include "catalog_pod_test_types.hpp"
#include "channel_test_support.hpp"

namespace {

#ifndef HOLOSCAN_CHANNEL_CATALOG_POD_INTERPROCESS_CHILD_EXE
#define HOLOSCAN_CHANNEL_CATALOG_POD_INTERPROCESS_CHILD_EXE ""
#endif

using holoscan::channel_test::ChildProcessCleanup;
using holoscan::channel_test::read_exact;
using holoscan::channel_test::write_exact;

// Builds the parent-side import catalog for a single CatalogProcessPod endpoint.
// Sets max_loaned_samples=1 to match the bounded-credit test configuration.
[[nodiscard]] holoscan::expected<holoscan::ChannelEndpointCatalog, holoscan::Error>
make_import_catalog(const std::string& endpoint_name) {
  auto builder = holoscan::ChannelEndpointCatalog::builder("catalog-pod-parent");
  holoscan::TopicOptions options;
  options.bounds.publisher = holoscan::PublisherEndpointBounds{.max_loaned_samples = 1U};
  auto added = builder.import_topic<holoscan::channel_test::CatalogProcessPod>(
      endpoint_name, "holomq://" + endpoint_name, options);
  if (!added) {
    return holoscan::make_unexpected(std::move(added.error()));
  }
  return std::move(builder).build();
}

// A typed HoloMQ loan is the nonblocking pressure boundary beneath the Channel POD adapter. Hold
// one unsent custom-POD loan to exhaust the explicit finite credit, verify the exact typed error,
// then release it and prove that the same publisher recovers without changing service identity.
// REQ: CHAN-066
TEST(CatalogPodBoundaryTest, TypedCustomPodLoanPressureIsBoundedAndRecoverable) {
  // Key setup: publisher created with max_loaned_samples=1; first loan held without
  //   sending so the finite credit pool is fully exhausted.
  auto context = holomq::ContextBuilder().create();
  ASSERT_TRUE(context) << context.error().message;

  holomq::PublisherPubSubOptions publisher_options;
  publisher_options.max_loaned_samples = 1U;
  auto publisher = context->publisher_builder(holoscan::channel_test::make_unique_channel_name())
                       .publisher_options(publisher_options)
                       .payload_type<holoscan::channel_test::CatalogProcessPod>()
                       .create();
  ASSERT_TRUE(publisher) << publisher.error().message;

  {
    auto held = publisher->loan();
    ASSERT_TRUE(held) << held.error().message;
    held->payload() = holoscan::channel_test::CatalogProcessPod{.frame = 1U, .value = -1};

    // Key assertion: second loan returns kBackpressured while first is still held.
    auto pressured = publisher->loan();
    ASSERT_FALSE(pressured);
    EXPECT_EQ(pressured.error().code, holomq::ErrorCode::kBackpressured);
  }

  auto recovered = publisher->loan();
  ASSERT_TRUE(recovered) << recovered.error().message;
  recovered->payload() = holoscan::channel_test::CatalogProcessPod{.frame = 2U, .value = -2};
}

// The child independently builds the matching export catalog after exec(). Successful attach and
// decode therefore prove the canonical POD name/version/layout contract rather than shared C++
// object state, an inherited catalog instance, or a process-local registry.
//
// Process topology:
//   parent (publisher) --[CatalogProcessPod frame=301]--> child (subscriber, exec'd)
//   start_pipe:  parent --[1-byte go signal]--> child
//   result_pipe: child  --[ready=0x01, then result=0x00]--> parent
// REQ: CHAN-068
// REQ: CHAN-074
TEST(CatalogPodBoundaryTest, IndependentCatalogClientDecodesCustomPodAcrossProcess) {
  // Key setup: child is exec'd independently and builds its own export catalog from
  //   scratch; no C++ state is shared across the exec boundary.
  const char* const child_exe = HOLOSCAN_CHANNEL_CATALOG_POD_INTERPROCESS_CHILD_EXE;
  ASSERT_NE(child_exe[0], '\0')
      << "HOLOSCAN_CHANNEL_CATALOG_POD_INTERPROCESS_CHILD_EXE not configured";

  const std::string endpoint_name = holoscan::channel_test::make_unique_channel_name();
  auto catalog = make_import_catalog(endpoint_name);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());

  std::array<int, 2> result_pipe{};
  std::array<int, 2> start_pipe{};
  ASSERT_EQ(::pipe(result_pipe.data()), 0);
  ASSERT_EQ(::pipe(start_pipe.data()), 0);
  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    ::close(result_pipe[0]);
    ::close(start_pipe[1]);
    const std::string result_fd = std::to_string(result_pipe[1]);
    const std::string start_fd = std::to_string(start_pipe[0]);
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_RESULT_FD", result_fd.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_START_FD", start_fd.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_ENDPOINT_NAME", endpoint_name.c_str(), 1));
    ::execl(child_exe, "channel_catalog_pod_interprocess_child", static_cast<char*>(nullptr));
    _exit(127);
  }
  ::close(result_pipe[1]);
  ::close(start_pipe[0]);
  ChildProcessCleanup child_cleanup{child, result_pipe[0], start_pipe[1]};

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  auto publisher =
      runtime->create_publisher<holoscan::channel_test::CatalogProcessPod>(*catalog, endpoint_name);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());

  constexpr std::uint8_t kStartChild = 1U;
  ASSERT_TRUE(write_exact(child_cleanup.start_fd(), &kStartChild, sizeof(kStartChild)));
  child_cleanup.close_start();

  std::uint8_t ready_code = 0xffU;
  ASSERT_TRUE(read_exact(child_cleanup.result_fd(), &ready_code, sizeof(ready_code)));
  if (ready_code != 1U) {
    int status = 0;
    ASSERT_EQ(child_cleanup.wait(&status), child);
    ASSERT_TRUE(WIFEXITED(status));
    FAIL() << "catalog child failed before subscriber-ready handshake; exit=" << WEXITSTATUS(status)
           << " validation code " << static_cast<unsigned>(ready_code);
  }

  ASSERT_TRUE(holoscan::channel_test::wait_for_subscribers(*publisher, std::chrono::seconds{2}));
  auto emitted = publisher->emit_sample(holoscan::Sample<holoscan::channel_test::CatalogProcessPod>{
      holoscan::channel_test::CatalogProcessPod{.frame = 301U, .value = -33},
      holoscan::SampleMetadata{
          .trace_id = 0x1234U,
          .frame_id = 0x5678U,
          .flags = holoscan::SampleFlags::kDegraded,
      },
  });
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  EXPECT_EQ(*emitted, 1U);

  std::uint8_t child_code = 0xffU;
  ASSERT_TRUE(read_exact(child_cleanup.result_fd(), &child_code, sizeof(child_code)));
  child_cleanup.close_result();
  int status = 0;
  ASSERT_EQ(child_cleanup.wait(&status), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
  // Key assertion: child_code==0 and exit status==0 confirm the child decoded the
  //   custom POD sample with correct field values and sample metadata.
  EXPECT_EQ(child_code, 0U) << "child validation stage/error code "
                            << static_cast<unsigned>(child_code);
}

}  // namespace
