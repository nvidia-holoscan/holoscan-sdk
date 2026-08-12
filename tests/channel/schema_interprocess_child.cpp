/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Standalone child process for Channel schema interprocess tests. The parent can publish either a
 * small inline String schema payload or a descriptor-backed Image payload whose descriptor carries
 * a HoloMQ BufferHandle. This process receives through the public Channel schema subscriber API.
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

#include <cuda_runtime_api.h>

#include <holoscan/channel/channel.hpp>
#include <holoscan/channel/endpoint_catalog_builder.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/schema/descriptor_flatbuffer_payload.hpp>
// Provides generated owning types and schema-trait specializations.
#include <holoscan/schema/flatbuffer_schema_package.hpp>  // NOLINT(misc-include-cleaner)
#include <holoscan/schema/image_metadata.hpp>
#include <holoscan/schema/inline_flatbuffer_payload.hpp>

#include "channel_test_support.hpp"
#include "custom_control_generated.h"
// Registers the generated CustomControl owning type with schema concepts.
#include "custom_control_schema_traits.hpp"  // NOLINT(misc-include-cleaner)
#include "string_generated.h"

namespace {

using DescriptorBackedImagePayload =
    holoscan::schema::DescriptorFlatBufferPayloadFor<holoscan::schema::Image>;
using StringPayload = holoscan::schema::InlineFlatBufferPayloadFor<holoscan::schema::String>;
using CustomControlPayload =
    holoscan::schema::InlineFlatBufferPayloadFor<holoscan::tests::channel::CustomControl>;

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

[[nodiscard]] int run_string_child(const int result_fd, const int start_fd,
                                   const std::string& channel_name) {
  if (!wait_for_start(start_fd)) {
    return finish(result_fd, 9U);
  }

  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    return finish(result_fd, 20U);
  }

  holoscan::ChannelOptions options{};
  options.queue_policy = holoscan::ChannelQueuePolicy::kBlocking;
  options.subscriber_queue_depth = 8U;
  options.expected_recipient_count = 1U;

  auto subscriber = runtime->create_subscriber<StringPayload>(channel_name, options);
  if (!subscriber) {
    return finish(result_fd, 21U);
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
        return finish(result_fd, 22U);
      }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }

    const StringPayload payload = std::move(received).value();
    const auto valid = payload.validate();
    if (!valid) {
      return finish(result_fd, 23U);
    }
    const holoscan::schema::String* parsed = payload.root();
    if (parsed == nullptr || parsed->data() == nullptr ||
        parsed->data()->string_view() != "ipc-string") {
      return finish(result_fd, 24U);
    }
    return finish(result_fd, 0U);
  }

  return finish(result_fd, 25U);
}

[[nodiscard]] int run_custom_child(const int result_fd, const int start_fd,
                                   const std::string& channel_name) {
  if (!wait_for_start(start_fd)) {
    return finish(result_fd, 9U);
  }

  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    return finish(result_fd, 50U);
  }

  holoscan::ChannelOptions options{};
  options.queue_policy = holoscan::ChannelQueuePolicy::kBlocking;
  options.subscriber_queue_depth = 8U;
  options.expected_recipient_count = 1U;

  auto subscriber = runtime->create_subscriber<CustomControlPayload>(channel_name, options);
  if (!subscriber) {
    return finish(result_fd, 51U);
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
        return finish(result_fd, 52U);
      }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }

    const CustomControlPayload payload = std::move(received).value();
    const auto valid = payload.validate();
    if (!valid) {
      return finish(result_fd, 53U);
    }
    const holoscan::tests::channel::CustomControl* parsed = payload.root();
    if (parsed == nullptr || parsed->sequence() != 42U || parsed->gain() != 1.25F ||
        parsed->label() == nullptr || parsed->label()->string_view() != "ipc-custom") {
      return finish(result_fd, 54U);
    }
    return finish(result_fd, 0U);
  }

  return finish(result_fd, 55U);
}

[[nodiscard]] int run_owning_string_child(const int result_fd, const int start_fd,
                                          const std::string& endpoint_name) {
  if (!wait_for_start(start_fd)) {
    return finish(result_fd, 9U);
  }

  auto builder = holoscan::ChannelEndpointCatalog::builder("owning-schema-child");
  auto added =
      builder.export_topic<holoscan::schema::StringT>(endpoint_name, "holomq://" + endpoint_name);
  if (!added) {
    return finish(result_fd, 40U);
  }
  auto catalog = std::move(builder).build();
  if (!catalog) {
    return finish(result_fd, 41U);
  }
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    return finish(result_fd, 42U);
  }
  auto subscriber = runtime->create_subscriber<holoscan::schema::StringT>(*catalog, endpoint_name);
  if (!subscriber) {
    return finish(result_fd, 43U);
  }

  constexpr std::uint8_t kSubscriberReady = 1U;
  if (!write_exact(result_fd, &kSubscriberReady, sizeof(kSubscriberReady))) {
    return 1;
  }

  const auto data_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  bool received_data = false;
  while (std::chrono::steady_clock::now() < data_deadline) {
    auto received = subscriber->receive_sample();
    if (!received) {
      if (received.error().code != holoscan::ErrorCode::kNotReady) {
        return finish(result_fd, 44U);
      }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }
    if (received->data.data != "owning schema across exec" || received->metadata.sequence == 0U ||
        received->metadata.trace_id != 0x1234U || received->metadata.frame_id != 0x5678U ||
        received->metadata.flags != holoscan::SampleFlags::kDegraded ||
        received->metadata.byte_size <= received->data.data.size() ||
        received->metadata.payload_kind != holoscan::PayloadKind::kSerializedBytes) {
      return finish(result_fd, 45U);
    }
    received_data = true;
    break;
  }
  if (!received_data) {
    return finish(result_fd, 46U);
  }

  constexpr std::uint8_t kDataReceived = 2U;
  if (!write_exact(result_fd, &kDataReceived, sizeof(kDataReceived))) {
    return 1;
  }

  const auto eos_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < eos_deadline) {
    auto received = subscriber->receive_sample();
    if (!received) {
      if (received.error().code == holoscan::ErrorCode::kClosed) {
        return finish(result_fd, 0U);
      }
      if (received.error().code != holoscan::ErrorCode::kNotReady) {
        return finish(result_fd, 47U);
      }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }
    return finish(result_fd, 48U);
  }

  return finish(result_fd, 49U);
}

[[nodiscard]] int run_descriptor_child(const int result_fd, const int start_fd,
                                       const std::string& channel_name) {
  if (!wait_for_start(start_fd)) {
    return finish(result_fd, 9U);
  }

  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    return finish(result_fd, 30U);
  }

  holoscan::ChannelOptions options{};
  options.queue_policy = holoscan::ChannelQueuePolicy::kBlocking;
  options.subscriber_queue_depth = 8U;
  options.expected_recipient_count = 1U;

  auto subscriber = runtime->create_subscriber<DescriptorBackedImagePayload>(channel_name, options);
  if (!subscriber) {
    return finish(result_fd, 31U);
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
        return finish(result_fd, 32U);
      }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }

    const DescriptorBackedImagePayload payload = std::move(received).value();
    const auto valid = payload.validate();
    if (!valid) {
      return finish(result_fd, 33U);
    }
    const holoscan::PayloadDescriptor* descriptor = payload.descriptor_at(0U);
    if (descriptor == nullptr ||
        descriptor->payload_kind != holoscan::ChannelPayloadKind::kBufferHandle) {
      return finish(result_fd, 34U);
    }
    if (payload.root() == nullptr) {
      return finish(result_fd, 35U);
    }
    return finish(result_fd, 0U);
  }

  return finish(result_fd, 36U);
}

[[nodiscard]] int run_image_child(const int result_fd, const int start_fd,
                                  const std::string& channel_name,
                                  const std::uint64_t expected_bytes,
                                  const std::uint32_t expected_seed) {
  if (!wait_for_start(start_fd)) {
    return finish(result_fd, 9U);
  }

  if (expected_bytes == 0U) {
    return finish(result_fd, 8U);
  }

  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    return finish(result_fd, 11U);
  }

  holoscan::ChannelOptions options{};
  options.queue_policy = holoscan::ChannelQueuePolicy::kBlocking;
  options.subscriber_queue_depth = 8U;
  options.expected_recipient_count = 1U;

  auto subscriber =
      runtime->create_tensor_schema_subscriber<DescriptorBackedImagePayload>(channel_name, options);
  if (!subscriber) {
    return finish(result_fd, 12U);
  }
  constexpr std::uint8_t kSubscriberReady = 1U;
  if (!write_exact(result_fd, &kSubscriberReady, sizeof(kSubscriberReady))) {
    return 1;
  }

  // Initialize CUDA after the IPC subscriber is constructed so service discovery for the
  // transport-control plane is independent of CUDA context setup.
  if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
    return finish(result_fd, 10U);
  }

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    auto received = subscriber->receive();
    if (!received) {
      if (received.error().code != holoscan::ErrorCode::kNotReady) {
        return finish(result_fd, 13U);
      }
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }

    auto sample = std::move(received).value();
    if (sample.payload.payload_kind != holoscan::ChannelPayloadKind::kBufferHandle) {
      return finish(result_fd, 14U);
    }
    if (!sample.tensor.is_read_only()) {
      return finish(result_fd, 15U);
    }
    if (sample.tensor.device().device_type != kDLCUDAHost) {
      return finish(result_fd, 16U);
    }
    if (sample.tensor.nbytes() != static_cast<int64_t>(expected_bytes)) {
      return finish(result_fd, 17U);
    }
    if (!read_host_pattern(
            sample.tensor.data(), static_cast<std::size_t>(expected_bytes), expected_seed)) {
      return finish(result_fd, 18U);
    }
    sample.tensor = holoscan::Tensor{};
    return finish(result_fd, 0U);
  }

  return finish(result_fd, 19U);
}

}  // namespace

int main() {
  const int result_fd = env_fd("HOLOSCAN_CHANNEL_SCHEMA_RESULT_FD");
  const int start_fd = env_fd("HOLOSCAN_CHANNEL_SCHEMA_START_FD");
  const std::string channel_name = env_string("HOLOSCAN_CHANNEL_SCHEMA_NAME");
  const std::string mode = env_string("HOLOSCAN_CHANNEL_SCHEMA_MODE");
  const std::uint64_t expected_bytes =
      env_u64("HOLOSCAN_CHANNEL_SCHEMA_EXPECTED_BYTES", /*fallback=*/0U);
  const std::uint32_t expected_seed =
      env_u32("HOLOSCAN_CHANNEL_SCHEMA_EXPECTED_SEED", /*fallback=*/0U);
  if (result_fd < 0 || start_fd < 0 || channel_name.empty() || mode.empty()) {
    return 1;
  }
  if (mode == "string") {
    return run_string_child(result_fd, start_fd, channel_name);
  }
  if (mode == "custom") {
    return run_custom_child(result_fd, start_fd, channel_name);
  }
  if (mode == "owning_string") {
    return run_owning_string_child(result_fd, start_fd, channel_name);
  }
  if (mode == "descriptor") {
    return run_descriptor_child(result_fd, start_fd, channel_name);
  }
  if (mode == "image") {
    return run_image_child(result_fd, start_fd, channel_name, expected_bytes, expected_seed);
  }

  return finish(result_fd, 7U);
}
