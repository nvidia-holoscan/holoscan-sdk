/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone child process for the Channel event-driven wait tests. Unlike
 * channel_interprocess_child.cpp, which polls receive() with a sleep, this child makes exactly one
 * blocking wait_and_receive_sample call. That single call is the assertion: it proves a typed
 * subscriber can wait across a process boundary with no periodic polling.
 */

#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>

#include <sys/resource.h>
#include <unistd.h>

#include <holoscan/core/errors.hpp>

#include "channel_test_support.hpp"

namespace {

// The published scalar the parent emits. A bare struct is a valid inline payload but not a
// PublishedPayload, which the sample-returning API is constrained on.
using WaitPayload = std::uint64_t;

constexpr WaitPayload kExpectedValue = 4242U;

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

[[nodiscard]] bool env_flag(const char* const name, const bool fallback) {
  const std::string value = env_string(name);
  return value.empty() ? fallback : value != "0";
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

// Total CPU this process has consumed. Measured here rather than with RUSAGE_CHILDREN in the
// parent, because the parent's figure accumulates over every child its test binary has ever
// reaped, and because this isolates the wait from process startup.
[[nodiscard]] std::chrono::microseconds process_cpu_time() {
  ::rusage usage{};
  if (::getrusage(RUSAGE_SELF, &usage) != 0) {
    return std::chrono::microseconds::zero();
  }
  const auto to_microseconds = [](const ::timeval& value) {
    return std::chrono::microseconds(static_cast<std::int64_t>(value.tv_sec) * 1000000 +
                                     static_cast<std::int64_t>(value.tv_usec));
  };
  return to_microseconds(usage.ru_utime) + to_microseconds(usage.ru_stime);
}

}  // namespace

int main() {
  const int result_fd = env_fd("HOLOSCAN_CHANNEL_EVENT_WAIT_RESULT_FD");
  const int start_fd = env_fd("HOLOSCAN_CHANNEL_EVENT_WAIT_START_FD");
  const std::string channel_name = env_string("HOLOSCAN_CHANNEL_EVENT_WAIT_NAME");
  const bool expect_sample = env_flag("HOLOSCAN_CHANNEL_EVENT_WAIT_EXPECT_SAMPLE", true);
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
  options.enable_event_wake = true;

  auto subscriber = runtime->create_subscriber<WaitPayload>(channel_name, options);
  if (!subscriber) {
    return finish(result_fd, 11U);
  }
  constexpr std::uint8_t kSubscriberReady = 1U;
  if (!write_exact(result_fd, &kSubscriberReady, sizeof(kSubscriberReady))) {
    return 1;
  }

  // Tell the parent the wait is about to begin, so it can time its emit instead of guessing.
  constexpr std::uint8_t kAboutToBlock = 2U;
  if (!write_exact(result_fd, &kAboutToBlock, sizeof(kAboutToBlock))) {
    return 1;
  }

  const auto cpu_before = process_cpu_time();
  const auto wall_before = std::chrono::steady_clock::now();

  // Exactly one call, with no surrounding loop and no sleep anywhere in this process.
  auto received = subscriber->wait_and_receive_sample(
      wall_before + (expect_sample ? std::chrono::seconds(10) : std::chrono::seconds(1)));

  const auto wall = std::chrono::steady_clock::now() - wall_before;
  const auto cpu = process_cpu_time() - cpu_before;

  if (expect_sample) {
    if (!received) {
      return finish(result_fd, 12U);
    }
    if (received->data != kExpectedValue) {
      return finish(result_fd, 13U);
    }
  } else {
    if (received) {
      return finish(result_fd, 14U);
    }
    if (received.error().code != holoscan::ErrorCode::kTimeout) {
      return finish(result_fd, 15U);
    }
  }

  // A level-triggered attachment that is never drained, or any poll fallback, would burn a core
  // for the whole block. A genuine kernel wait costs almost nothing. Only meaningful once the wait
  // actually blocked for a while, so short waits skip the check rather than assert on noise.
  const auto wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(wall).count();
  const auto cpu_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cpu).count();
  if (wall_ms >= 200 && cpu_ms > wall_ms / 4) {
    return finish(result_fd, 16U);
  }

  return finish(result_fd, 0U);
}
