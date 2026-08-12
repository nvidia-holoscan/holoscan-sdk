/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/context.hpp>
#include <holomq/context_builder.hpp>

#include <gtest/gtest.h>

#include "fd_pass/client.hpp"
#include "fd_pass/sockaddr.hpp"

#include <cstdint>
#include <string>
#include <utility>

#include <unistd.h>

namespace external_api {

struct Payload {};

// Models an application namespace that defines the same helper name. HoloMQ templates must qualify
// their helper calls so argument-dependent lookup cannot make them ambiguous for external payloads.
template <typename T, typename E>
holomq::unexpected<E> forward_error(const holomq::expected<T, E>& expected_value);

}  // namespace external_api

namespace {

struct EmptyPayload {};

[[nodiscard]] std::string make_fd_pass_socket_path() {
  return "/tmp/holomq_context_lifecycle_" + std::to_string(static_cast<long>(::getpid())) + ".sock";
}

void cleanup_fd_pass_socket_path(const std::string& socket_path) {
  if (holomq::fd_pass::endpoint_uses_filesystem_path()) {
    static_cast<void>(::unlink(socket_path.c_str()));
  }
}

}  // namespace

TEST(ContextTest, DestroyingContextStopsFdPassServer) {
  const std::string socket_path = make_fd_pass_socket_path();
  struct SocketPathCleanup {
    const std::string& path;
    ~SocketPathCleanup() { cleanup_fd_pass_socket_path(path); }
  } cleanup{socket_path};

  ASSERT_LE(socket_path.size(), holomq::fd_pass::endpoint_max_length());

  {
    auto context_or =
        holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
    ASSERT_TRUE(context_or.has_value());
  }

  EXPECT_FALSE(holomq::fd_pass::Client::create(socket_path).has_value());
}

TEST(ContextTest, SubscriberSupportsPayloadNamespaceWithForwardErrorHelper) {
  auto context_or = holomq::ContextBuilder().with_vmm_fd_export(false).create();
  ASSERT_TRUE(context_or.has_value());
  auto& context = context_or.value();

  auto subscriber_or = context.subscriber_builder("external_payload_adl_test")
                           .payload_type<external_api::Payload>()
                           .create();
  ASSERT_TRUE(subscriber_or.has_value());

  const auto sample_or = subscriber_or->try_take();
  ASSERT_FALSE(sample_or.has_value());
  EXPECT_EQ(sample_or.error().code, holomq::ErrorCode::kNotReady);
}

TEST(ContextTest, EndpointBuildersPreserveInvalidServiceNameError) {
  auto context_or = holomq::ContextBuilder().with_vmm_fd_export(false).create();
  ASSERT_TRUE(context_or.has_value());
  auto& context = context_or.value();

  auto publisher_or = context.publisher_builder("").payload_type<EmptyPayload>().create();
  ASSERT_FALSE(publisher_or.has_value());
  EXPECT_EQ(publisher_or.error().code, holomq::ErrorCode::kInvalidArgument);
  EXPECT_NE(publisher_or.error().message.find("service name"), std::string::npos);

  auto subscriber_or = context.subscriber_builder("").payload_type<EmptyPayload>().create();
  ASSERT_FALSE(subscriber_or.has_value());
  EXPECT_EQ(subscriber_or.error().code, holomq::ErrorCode::kInvalidArgument);
  EXPECT_NE(subscriber_or.error().message.find("service name"), std::string::npos);
}

TEST(ContextTest, MovedFromContextQueriesDoNotCrash) {
  auto context_or = holomq::ContextBuilder().with_vmm_fd_export(false).create();
  ASSERT_TRUE(context_or.has_value());

  holomq::Context active = std::move(context_or).value();
  ASSERT_TRUE(context_or.has_value());

  holomq::Context& moved_from = context_or.value();
  EXPECT_FALSE(moved_from.vmm_fd_export_enabled());
  EXPECT_TRUE(moved_from.fd_pass_socket_path().empty());
  EXPECT_FALSE(moved_from.import_buffer_unsynchronized(holomq::BufferHandle{}).has_value());
  EXPECT_FALSE(moved_from.create_waitset().has_value());
  EXPECT_FALSE(moved_from.create_device_buffer_pool(256U, 1U).has_value());

  const auto publisher_or =
      moved_from.publisher_builder("moved_from_test").payload_type<EmptyPayload>().create();
  EXPECT_FALSE(publisher_or.has_value());

  const auto subscriber_or =
      moved_from.subscriber_builder("moved_from_test").payload_type<EmptyPayload>().create();
  EXPECT_FALSE(subscriber_or.has_value());

  // Keep the moved-to context alive so the test only exercises the moved-from shell.
  EXPECT_FALSE(active.vmm_fd_export_enabled());
}
