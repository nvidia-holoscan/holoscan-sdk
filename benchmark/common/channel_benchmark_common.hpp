/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

#include <fmt/format.h>

#include <holoscan/channel/channel.hpp>
#include <holoscan/core/errors.hpp>

namespace holoscan::benchmarking {

class ScopedBenchmarkCPUAffinity {
 public:
  ScopedBenchmarkCPUAffinity() = default;
  ScopedBenchmarkCPUAffinity(const ScopedBenchmarkCPUAffinity&) = delete;
  ScopedBenchmarkCPUAffinity& operator=(const ScopedBenchmarkCPUAffinity&) = delete;

  ~ScopedBenchmarkCPUAffinity() {
#if defined(__linux__)
    if (restore_) {
      static_cast<void>(pthread_setaffinity_np(pthread_self(), sizeof(original_), &original_));
    }
#endif
  }

  [[nodiscard]] bool pin(const int cpu) noexcept {
    if (cpu < 0) {
      return true;
    }
#if defined(__linux__)
    if (cpu >= CPU_SETSIZE) {
      return false;
    }
    cpu_set_t original{};
    if (!restore_ && pthread_getaffinity_np(pthread_self(), sizeof(original), &original) != 0) {
      return false;
    }
    cpu_set_t requested{};
    CPU_ZERO(&requested);
    CPU_SET(cpu, &requested);
    if (pthread_setaffinity_np(pthread_self(), sizeof(requested), &requested) != 0) {
      return false;
    }
    if (!restore_) {
      original_ = original;
      restore_ = true;
    }
    return true;
#else
    return false;
#endif
  }

 private:
#if defined(__linux__)
  cpu_set_t original_{};
  bool restore_{};
#endif
};

[[nodiscard]] inline bool pin_current_benchmark_thread(const int cpu) noexcept {
  if (cpu < 0) {
    return true;
  }
#if defined(__linux__)
  if (cpu >= CPU_SETSIZE) {
    return false;
  }
  cpu_set_t requested{};
  CPU_ZERO(&requested);
  CPU_SET(cpu, &requested);
  return pthread_setaffinity_np(pthread_self(), sizeof(requested), &requested) == 0;
#else
  return false;
#endif
}

[[nodiscard]] inline std::optional<int> benchmark_cpu(const char* const name) noexcept {
  const char* const value = std::getenv(name);
  if (value == nullptr) {
    return -1;
  }
  int cpu = -1;
  const std::string_view text{value};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), cpu);
  if (error != std::errc() || end != text.data() + text.size() || cpu < -1) {
    return std::nullopt;
  }
  return cpu;
}

[[nodiscard]] inline std::int64_t benchmark_now_ns() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

[[nodiscard]] inline bool wait_for_benchmark_count(const std::atomic<std::uint64_t>& count,
                                                   const std::uint64_t expected_count) {
  constexpr auto kTimeout = std::chrono::seconds{3};
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  while (count.load(std::memory_order_acquire) < expected_count) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::yield();
  }
  return true;
}

[[nodiscard]] inline std::uint64_t benchmark_process_id() noexcept {
#if defined(__linux__)
  return static_cast<std::uint64_t>(::getpid());
#else
  return 0U;
#endif
}

[[nodiscard]] inline std::string make_unique_channel_benchmark_name(const std::string_view prefix,
                                                                    const std::string_view suffix) {
  static std::atomic<std::uint64_t> next_id{0U};
  return fmt::format("{}{}_{}_{}",
                     prefix,
                     benchmark_process_id(),
                     next_id.fetch_add(1U, std::memory_order_relaxed),
                     suffix);
}

[[nodiscard]] inline std::string make_unique_holomq_benchmark_socket_path(
    const std::string_view prefix) {
  static std::atomic<std::uint64_t> next_id{0U};
  return fmt::format("/tmp/{}{}_{}.sock",
                     prefix,
                     benchmark_process_id(),
                     next_id.fetch_add(1U, std::memory_order_relaxed));
}

inline void cleanup_holomq_benchmark_socket_path(const std::string& socket_path) {
  static_cast<void>(std::remove(socket_path.c_str()));
}

[[nodiscard]] inline ChannelOptions make_channel_benchmark_options(
    const ChannelQueuePolicy policy, const std::uint32_t expected_recipient_count = 1U,
    const std::uint32_t subscriber_queue_depth = 64U) {
  ChannelOptions options{};
  options.queue_policy = policy;
  options.subscriber_queue_depth = subscriber_queue_depth;
  options.publisher_history_depth = 0U;
  options.expected_recipient_count = expected_recipient_count;
  return options;
}

template <typename Publisher, typename EmitProbe, typename DrainProbe>
[[nodiscard]] bool wait_for_channel_subscribers(
    Publisher& publisher, const std::uint32_t expected_recipients,
    // Retry the same probes as lvalues during startup.
    // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    EmitProbe&& emit_probe, DrainProbe&& drain_probe) {
  constexpr auto kTimeout = std::chrono::seconds{5};
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  std::uint32_t consecutive_ready_refreshes = 0U;
  while (std::chrono::steady_clock::now() < deadline) {
    auto updated = publisher.update_connections();
    if (!updated.has_value()) {
      std::cerr << "update_connections failed while waiting for subscribers: "
                << runtime_error_message(updated.error()) << '\n';
      return false;
    }
    if (!publisher.has_subscribers()) {
      consecutive_ready_refreshes = 0U;
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
      continue;
    }
    ++consecutive_ready_refreshes;
    // The advisory connection state can lead the publisher's send-recipient snapshot.
    if (consecutive_ready_refreshes < 2U) {
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
      continue;
    }

    auto emitted = emit_probe();
    drain_probe();
    if (emitted.has_value() && emitted.value() == expected_recipients) {
      return true;
    }
    if (!emitted.has_value() && emitted.error().code != ErrorCode::kNotReady &&
        emitted.error().code != ErrorCode::kBackpressured &&
        emitted.error().code != ErrorCode::kPublicationPartial &&
        emitted.error().code != ErrorCode::kPublicationIndeterminate) {
      std::cerr << "startup probe failed: " << runtime_error_message(emitted.error()) << '\n';
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  std::cerr << "timed out waiting for " << expected_recipients << " subscriber(s)\n";
  return false;
}

template <typename Publisher, typename EmitProbe>
[[nodiscard]] bool wait_for_channel_subscribers(Publisher& publisher,
                                                const std::uint32_t expected_recipients,
                                                EmitProbe&& emit_probe) {
  return wait_for_channel_subscribers(
      publisher, expected_recipients, std::forward<EmitProbe>(emit_probe), []() noexcept {});
}

}  // namespace holoscan::benchmarking
