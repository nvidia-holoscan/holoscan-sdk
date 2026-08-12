/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone child process for direct Tensor Channel interprocess tests.
 */

#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <unistd.h>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>

#include "channel_test_support.hpp"

namespace {

[[nodiscard]] bool read_host_pattern(const void* const ptr, const std::size_t size_bytes,
                                     const std::uint32_t seed) {
  const auto* words = static_cast<const std::uint32_t*>(ptr);
  const std::size_t word_count = size_bytes / sizeof(std::uint32_t);
  for (std::size_t i = 0; i < word_count; ++i) {
    if (words[i] != seed + static_cast<std::uint32_t>(i)) {
      return false;
    }
  }
  return true;
}

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

[[nodiscard]] std::uint32_t env_u32(const char* const name, const std::uint32_t fallback) {
  const char* const value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  return static_cast<std::uint32_t>(std::strtoul(value, nullptr, 10));
}

[[nodiscard]] std::uint64_t env_u64(const char* const name, const std::uint64_t fallback) {
  const char* const value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  return static_cast<std::uint64_t>(std::strtoull(value, nullptr, 10));
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
  const int result_fd = env_fd("HOLOSCAN_CHANNEL_TENSOR_RESULT_FD");
  const int start_fd = env_fd("HOLOSCAN_CHANNEL_TENSOR_START_FD");
  const std::string channel_name = env_string("HOLOSCAN_CHANNEL_TENSOR_NAME");
  const std::uint64_t expected_bytes = env_u64("HOLOSCAN_CHANNEL_TENSOR_EXPECTED_BYTES", 0U);
  const std::uint32_t expected_seed = env_u32("HOLOSCAN_CHANNEL_TENSOR_EXPECTED_SEED", 0U);
  const bool device_async = env_u32("HOLOSCAN_CHANNEL_TENSOR_DEVICE_ASYNC", 0U) != 0U;
  if (result_fd < 0 || start_fd < 0 || channel_name.empty() || expected_bytes == 0U) {
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
  options.max_subscriber_queue_depth = env_u32("HOLOSCAN_CHANNEL_TENSOR_MAX_QUEUE_DEPTH", 8U);
  options.subscriber_queue_depth = options.max_subscriber_queue_depth;
  options.max_subscriber_borrowed_samples = env_u32("HOLOSCAN_CHANNEL_TENSOR_MAX_BORROWED", 0U);
  options.max_borrowed_samples = options.max_subscriber_borrowed_samples;
  options.expected_recipient_count = 1U;
  options.enable_event_wake = env_u32("HOLOSCAN_CHANNEL_TENSOR_EVENT_WAKE", 0U) != 0U;
  options.tensor_transfer_mode = holoscan::ChannelTensorTransferMode::kBufferHandle;

  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  if (!subscriber) {
    return finish(result_fd, 11U);
  }
  constexpr std::uint8_t kSubscriberReady = 1U;
  if (!write_exact(result_fd, &kSubscriberReady, sizeof(kSubscriberReady))) {
    return 1;
  }

  // BufferHandle import includes a CUDA VMM generation sidecar even when payload bytes live in
  // host shared memory. Initialize CUDA only after the subscriber attaches so service matching is
  // independent of device setup.
  if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
    return finish(result_fd, 12U);
  }

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    auto received = subscriber->receive_sample();
    if (!received) {
      if (received.error().code != holoscan::ErrorCode::kNotReady) {
        return finish(result_fd, 13U);
      }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }

    holoscan::Sample<holoscan::Tensor> sample = std::move(received).value();
    if (sample.metadata.payload_kind != holoscan::PayloadKind::kDescriptorBacked ||
        sample.metadata.byte_size != expected_bytes) {
      return finish(result_fd, 14U);
    }
    if (device_async) {
      if (!sample.data.is_read_only() || sample.data.device().device_type != kDLCUDA ||
          static_cast<std::uint64_t>(sample.data.nbytes()) != expected_bytes) {
        return finish(result_fd, 15U);
      }

      cudaStream_t consumer_stream = nullptr;
      if (cudaStreamCreateWithFlags(&consumer_stream, cudaStreamNonBlocking) != cudaSuccess) {
        return finish(result_fd, 17U);
      }
      void* host_data = nullptr;
      if (cudaMallocHost(&host_data, expected_bytes) != cudaSuccess) {
        static_cast<void>(cudaStreamDestroy(consumer_stream));
        return finish(result_fd, 18U);
      }
      auto synchronized = sample.data.synchronize_for_consumer_stream(consumer_stream);
      if (!synchronized ||
          cudaMemcpyAsync(host_data,
                          sample.data.data(),
                          expected_bytes,
                          cudaMemcpyDeviceToHost,
                          consumer_stream) != cudaSuccess ||
          !sample.data.set_deallocation_stream(consumer_stream)) {
        static_cast<void>(cudaStreamSynchronize(consumer_stream));
        sample.data = holoscan::Tensor{};
        static_cast<void>(cudaFreeHost(host_data));
        static_cast<void>(cudaStreamDestroy(consumer_stream));
        return finish(result_fd, 19U);
      }
      constexpr std::uint8_t kConsumerWaitEnqueued = 2U;
      if (!write_exact(result_fd, &kConsumerWaitEnqueued, sizeof(kConsumerWaitEnqueued))) {
        static_cast<void>(cudaStreamSynchronize(consumer_stream));
        sample.data = holoscan::Tensor{};
        static_cast<void>(cudaFreeHost(host_data));
        static_cast<void>(cudaStreamDestroy(consumer_stream));
        return 1;
      }
      if (cudaStreamSynchronize(consumer_stream) != cudaSuccess ||
          !read_host_pattern(host_data, expected_bytes, expected_seed)) {
        sample.data = holoscan::Tensor{};
        static_cast<void>(cudaFreeHost(host_data));
        static_cast<void>(cudaStreamDestroy(consumer_stream));
        return finish(result_fd, 20U);
      }
      sample.data = holoscan::Tensor{};
      static_cast<void>(cudaFreeHost(host_data));
      static_cast<void>(cudaStreamDestroy(consumer_stream));
      return finish(result_fd, 0U);
    }
    if (!sample.data.is_read_only() ||
        static_cast<std::uint64_t>(sample.data.nbytes()) != expected_bytes ||
        !read_host_pattern(sample.data.data(), expected_bytes, expected_seed)) {
      return finish(result_fd, 15U);
    }
    return finish(result_fd, 0U);
  }

  return finish(result_fd, 16U);
}
