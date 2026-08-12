/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone catalog client used to prove that custom-POD identity and fieldwise codec evidence
 * survive an exec boundary. The child independently authors the export-side catalog rather than
 * borrowing the parent's in-memory catalog or native object representation.
 */

#include <unistd.h>

#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <holoscan/channel/endpoint_catalog_builder.hpp>
#include <holoscan/core/errors.hpp>

#include "catalog_pod_test_types.hpp"
#include "channel_test_support.hpp"

namespace {

using holoscan::channel_test::read_exact;
using holoscan::channel_test::write_exact;

[[nodiscard]] int env_fd(const char* const name) {
  const char* const value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return -1;
  }
  const std::string_view text(value);
  int fd = -1;
  const auto parsed = std::from_chars(text.begin(), text.end(), fd);
  return parsed.ec == std::errc() && parsed.ptr == text.end() && fd >= 0 ? fd : -1;
}

[[nodiscard]] std::string env_string(const char* const name) {
  const char* const value = std::getenv(name);
  return value == nullptr ? std::string{} : std::string{value};
}

[[nodiscard]] int finish(const int result_fd, const std::uint8_t code) {
  static_cast<void>(write_exact(result_fd, &code, sizeof(code)));
  ::close(result_fd);
  return code == 0U ? 0 : 1;
}

}  // namespace

int main() {
  const int result_fd = env_fd("HOLOSCAN_CHANNEL_RESULT_FD");
  const int start_fd = env_fd("HOLOSCAN_CHANNEL_START_FD");
  const std::string endpoint_name = env_string("HOLOSCAN_CHANNEL_ENDPOINT_NAME");
  if (result_fd < 0 || start_fd < 0 || endpoint_name.empty()) {
    return 1;
  }

  std::uint8_t start_code = 0U;
  const bool started = read_exact(start_fd, &start_code, sizeof(start_code)) && start_code == 1U;
  ::close(start_fd);
  if (!started) {
    return finish(result_fd, 9U);
  }

  auto builder = holoscan::ChannelEndpointCatalog::builder("catalog-pod-child");
  auto added = builder.export_topic<holoscan::channel_test::CatalogProcessPod>(
      endpoint_name, "holomq://" + endpoint_name);
  if (!added) {
    return finish(result_fd, 10U);
  }
  auto catalog = std::move(builder).build();
  if (!catalog) {
    return finish(result_fd, 11U);
  }
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    return finish(result_fd, 12U);
  }
  auto subscriber = runtime->create_subscriber<holoscan::channel_test::CatalogProcessPod>(
      *catalog, endpoint_name);
  if (!subscriber) {
    return finish(result_fd, 13U);
  }

  constexpr std::uint8_t kSubscriberReady = 1U;
  if (!write_exact(result_fd, &kSubscriberReady, sizeof(kSubscriberReady))) {
    return 1;
  }

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (std::chrono::steady_clock::now() < deadline) {
    auto received = subscriber->receive_sample();
    if (!received) {
      if (received.error().code != holoscan::ErrorCode::kNotReady) {
        return finish(result_fd, 14U);
      }
      std::this_thread::sleep_for(std::chrono::microseconds{100});
      continue;
    }

    constexpr holoscan::channel_test::CatalogProcessPod kExpected{.frame = 301U, .value = -33};
    if (received->data != kExpected || received->metadata.sequence == 0U ||
        received->metadata.trace_id != 0x1234U || received->metadata.frame_id != 0x5678U ||
        received->metadata.flags != holoscan::SampleFlags::kDegraded ||
        received->metadata.byte_size != sizeof(kExpected) ||
        received->metadata.payload_kind != holoscan::PayloadKind::kInlineTyped) {
      return finish(result_fd, 15U);
    }
    return finish(result_fd, 0U);
  }

  return finish(result_fd, 16U);
}
