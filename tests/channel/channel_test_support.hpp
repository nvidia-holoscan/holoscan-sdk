/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/channel/channel.hpp>
#include <holoscan/core/errors.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <utility>

#include <sys/types.h>
#include <unistd.h>

namespace holoscan::channel_test {

/// Build standard test Channel options with bounded queues and recipient evidence.
[[nodiscard]] inline holoscan::ChannelOptions make_options(
    const holoscan::ChannelQueuePolicy queue_policy,
    const std::uint32_t expected_recipient_count = 1U) {
  holoscan::ChannelOptions options{};
  options.queue_policy = queue_policy;
  options.subscriber_queue_depth = 8U;
  options.expected_recipient_count = expected_recipient_count;
  return options;
}

/// Read exactly `length` bytes unless EOF or a non-retryable error occurs.
[[nodiscard]] bool read_exact(int fd, void* buffer, std::size_t length) noexcept;

/// Write exactly `length` bytes unless a non-retryable error occurs.
[[nodiscard]] bool write_exact(int fd, const void* buffer, std::size_t length) noexcept;

/// Own a child process and its parent-side handshake file descriptors.
class ChildProcessCleanup {
 public:
  explicit ChildProcessCleanup(pid_t child, int result_fd = -1, int start_fd = -1) noexcept;
  ChildProcessCleanup(const ChildProcessCleanup&) = delete;
  ChildProcessCleanup& operator=(const ChildProcessCleanup&) = delete;
  ChildProcessCleanup(ChildProcessCleanup&&) = delete;
  ChildProcessCleanup& operator=(ChildProcessCleanup&&) = delete;
  ~ChildProcessCleanup() noexcept;

  [[nodiscard]] pid_t child() const noexcept { return child_; }
  [[nodiscard]] int result_fd() const noexcept { return result_fd_; }
  [[nodiscard]] int start_fd() const noexcept { return start_fd_; }

  void close_result() noexcept;
  void close_start() noexcept;
  [[nodiscard]] pid_t wait(int* status) noexcept;

 private:
  static void close_fd(int& fd) noexcept;
  void terminate_and_wait() noexcept;

  pid_t child_{-1};
  int result_fd_{-1};
  int start_fd_{-1};
};

inline std::atomic<std::uint64_t> g_topic_counter{0U};

[[nodiscard]] inline std::string make_unique_channel_name() {
  const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
  return "holoscan_channel_test_" + std::to_string(static_cast<long>(::getpid())) + "_" +
         std::to_string(timestamp) + "_" +
         std::to_string(g_topic_counter.fetch_add(1U, std::memory_order_relaxed));
}

[[nodiscard]] constexpr holoscan::SchemaId make_descriptor_schema_id(
    const std::uint64_t sequence) noexcept {
  holoscan::SchemaId id;
  for (std::size_t index = 0U; index < id.value.size(); ++index) {
    const auto sequence_byte =
        static_cast<std::uint8_t>(sequence >> static_cast<unsigned>((index % 8U) * 8U));
    id.value[index] =
        static_cast<std::byte>(sequence_byte ^ static_cast<std::uint8_t>(0x5DU + 17U * index));
  }
  return id;
}

[[nodiscard]] inline holoscan::ChannelSampleDescriptor make_descriptor(
    const std::uint64_t sequence) {
  return holoscan::ChannelSampleDescriptor{
      .magic = holoscan::kChannelSampleDescriptorMagic,
      .abi_version = holoscan::kChannelSampleDescriptorAbiVersion,
      .payload_kind = holoscan::ChannelPayloadKind::kLocalBufferDescriptor,
      .flags = static_cast<std::uint32_t>(sequence & 0xffU),
      .expected_recipient_count = 1U,
      .process_id = static_cast<std::uint64_t>(::getpid()),
      .sidecar_token = 0x1000U + sequence,
      .generation = 0x2000U + sequence,
      .schema_id = make_descriptor_schema_id(sequence),
      .sequence = sequence,
      .byte_size = 4096U + sequence,
  };
}

template <typename Publisher>
[[nodiscard]] bool wait_for_subscribers(Publisher& publisher,
                                        const std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (publisher.has_subscribers()) {
      return true;
    }
    static_cast<void>(publisher.update_connections());
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return publisher.has_subscribers();
}

[[nodiscard]] inline holoscan::expected<void, holoscan::Error> publish_descriptor(
    holoscan::ChannelDescriptorPublisher& publisher,
    const holoscan::ChannelSampleDescriptor& descriptor) {
  auto published = publisher.publish(descriptor);
  if (!published) {
    return holoscan::make_unexpected(std::move(published).error());
  }
  if (published.value() == 0U) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kNotReady, "descriptor was published before subscribers connected"});
  }
  return {};
}

[[nodiscard]] inline holoscan::expected<void, holoscan::Error> publish_descriptor(
    holoscan::ChannelDescriptorPublisher& publisher, const std::uint64_t sequence) {
  return publish_descriptor(publisher, make_descriptor(sequence));
}

template <typename Predicate>
[[nodiscard]] bool wait_until(const Predicate& predicate, const std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return predicate();
}

}  // namespace holoscan::channel_test
