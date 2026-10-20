/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone child process for Channel interprocess tests that do not involve schema payloads.
 */

#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <unistd.h>

#include <holoscan/core/errors.hpp>

#include "channel_test_support.hpp"

namespace {

struct ControlPayload {
  std::uint64_t sequence{};
  std::int32_t value{};
  double timestamp{};
};

static_assert(holoscan::ChannelInlinePayload<ControlPayload>);

using holoscan::channel_test::read_exact;
using holoscan::channel_test::write_exact;

[[nodiscard]] int env_fd(const char* const name) {
  const char* const value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return -1;
  }
  const std::string_view text(value);
  int fd = -1;
  const auto result = std::from_chars(text.begin(), text.end(), fd);
  return result.ec == std::errc() && result.ptr == text.end() && fd >= 0 ? fd : -1;
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

[[nodiscard]] bool wait_for_start(const int start_fd) {
  std::uint8_t start_code = 0U;
  const bool started = read_exact(start_fd, &start_code, sizeof(start_code)) && start_code == 1U;
  ::close(start_fd);
  return started;
}

}  // namespace

int main() {
  const int result_fd = env_fd("HOLOSCAN_CHANNEL_RESULT_FD");
  const int start_fd = env_fd("HOLOSCAN_CHANNEL_START_FD");
  const std::string channel_name = env_string("HOLOSCAN_CHANNEL_NAME");
  if (result_fd < 0 || start_fd < 0 || channel_name.empty()) {
    return 1;
  }

  if (!wait_for_start(start_fd)) {
    return finish(result_fd, 9U);
  }

  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    return finish(result_fd, 10U);
  }

  holoscan::ChannelOptions options{};
  options.queue_policy = holoscan::ChannelQueuePolicy::kBlocking;
  options.subscriber_queue_depth = 8U;
  options.expected_recipient_count = 1U;

  auto subscriber = runtime->create_subscriber<ControlPayload>(channel_name, options);
  if (!subscriber) {
    return finish(result_fd, 11U);
  }
  constexpr std::uint8_t kSubscriberReady = 1U;
  if (!write_exact(result_fd, &kSubscriberReady, sizeof(kSubscriberReady))) {
    return 1;
  }

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    auto received = subscriber->receive();
    if (!received) {
      if (received.error().code != holoscan::ErrorCode::kNotReady) {
        return finish(result_fd, 12U);
      }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }

    const ControlPayload payload = std::move(received).value();
    if (payload.sequence != 301U || payload.value != -33 || payload.timestamp != 99.25) {
      return finish(result_fd, 13U);
    }
    return finish(result_fd, 0U);
  }

  return finish(result_fd, 14U);
}
