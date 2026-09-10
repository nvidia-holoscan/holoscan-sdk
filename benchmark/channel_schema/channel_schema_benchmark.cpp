/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <flatbuffers/flatbuffer_builder.h>

#include <holomq/context_builder.hpp>
#include <holoscan/channel/channel.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/memory/allocator.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/schema/descriptor_flatbuffer_payload.hpp>
#include <holoscan/schema/flatbuffer_schema_package.hpp>
#include <holoscan/schema/inline_flatbuffer_payload.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "../../src/channel/channel_inline_payload_test_access.hpp"
#include "../../src/channel/holomq_tensor.hpp"
#include "../../src/channel/schema_envelope_codec.hpp"
#include "../common/channel_benchmark_common.hpp"
#include "header_generated.h"
#include "image_generated.h"
#include "imu_generated.h"
#include "time_generated.h"

namespace {

using ImuPayload = holoscan::schema::InlineFlatBufferPayloadFor<holoscan::schema::Imu>;
using ImagePayload = holoscan::schema::InlineFlatBufferPayloadFor<holoscan::schema::Image>;
using DescriptorBackedImagePayload =
    holoscan::schema::DescriptorFlatBufferPayloadFor<holoscan::schema::Image>;
using holoscan::benchmarking::benchmark_cpu;
using holoscan::benchmarking::benchmark_now_ns;
using holoscan::benchmarking::benchmark_process_id;
using holoscan::benchmarking::cleanup_holomq_benchmark_socket_path;
using holoscan::benchmarking::make_channel_benchmark_options;
using holoscan::benchmarking::make_unique_channel_benchmark_name;
using holoscan::benchmarking::make_unique_holomq_benchmark_socket_path;
using holoscan::benchmarking::pin_current_benchmark_thread;
using holoscan::benchmarking::ScopedBenchmarkCPUAffinity;
using holoscan::benchmarking::wait_for_benchmark_count;
using holoscan::benchmarking::wait_for_channel_subscribers;

constexpr std::size_t kTensorSchemaPoolSize = 128U;
constexpr std::size_t kHoloMQSchemaTensorBytes = 64U;
constexpr std::uint64_t kBenchmarkEndpointGeneration = 1U;

[[nodiscard]] bool cuda_unavailable_for_holomq_pool(benchmark::State& state) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    state.SkipWithError("no CUDA device available for HoloMQ schema buffer benchmark");
    return true;
  }
  if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
    state.SkipWithError("failed to initialize CUDA for HoloMQ schema buffer benchmark");
    return true;
  }
  return false;
}

struct HoloMQSchemaBufferFixture {
  HoloMQSchemaBufferFixture(std::string socket_path_arg, holomq::Context context_arg,
                            holomq::BufferPool pool_arg, holomq::Buffer buffer_arg,
                            DescriptorBackedImagePayload payload_arg)
      : socket_path(std::move(socket_path_arg)),
        context(std::move(context_arg)),
        pool(std::move(pool_arg)),
        buffer(std::move(buffer_arg)),
        payload(std::move(payload_arg)) {}

  ~HoloMQSchemaBufferFixture() { cleanup_holomq_benchmark_socket_path(socket_path); }

  HoloMQSchemaBufferFixture(const HoloMQSchemaBufferFixture&) = delete;
  HoloMQSchemaBufferFixture& operator=(const HoloMQSchemaBufferFixture&) = delete;

  std::string socket_path;
  holomq::Context context;
  holomq::BufferPool pool;
  holomq::Buffer buffer;
  DescriptorBackedImagePayload payload;
};

[[nodiscard]] constexpr std::size_t transport_bucket_size_for_payload(
    const std::size_t payload_size) noexcept {
  if (payload_size <= 8U) {
    return 8U;
  }
  if (payload_size <= 16U) {
    return 16U;
  }
  if (payload_size <= 32U) {
    return 32U;
  }
  if (payload_size <= 64U) {
    return 64U;
  }
  if (payload_size <= 128U) {
    return 128U;
  }
  if (payload_size <= 256U) {
    return 256U;
  }
  if (payload_size <= 512U) {
    return 512U;
  }
  if (payload_size <= 1024U) {
    return 1024U;
  }
  if (payload_size <= 2048U) {
    return 2048U;
  }
  if (payload_size <= 4096U) {
    return 4096U;
  }
  return 0U;
}

[[nodiscard]] flatbuffers::FlatBufferBuilder make_imu_builder() {
  flatbuffers::FlatBufferBuilder builder;
  const auto stamp = holoscan::schema::CreateTime(builder, 123456789, 123456799);
  const auto frame = builder.CreateString("imu_link");
  const auto header = holoscan::schema::CreateHeader(builder, stamp, frame);
  const holoscan::schema::Quaternionf orientation{0.0F, 0.0F, 0.0F, 1.0F};
  const holoscan::schema::Vector3f angular_velocity{0.1F, 0.2F, 0.3F};
  const holoscan::schema::Vector3f linear_acceleration{9.8F, 0.0F, -0.1F};
  builder.Finish(holoscan::schema::CreateImu(
      builder, header, &orientation, &angular_velocity, &linear_acceleration));
  return builder;
}

[[nodiscard]] flatbuffers::FlatBufferBuilder make_time_builder() {
  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(holoscan::schema::CreateTime(builder, 123456789, 123456799));
  return builder;
}

struct TimeProcessLocalFrame {
  static constexpr std::size_t kCanonicalCapacity =
      holoscan::generated_schema_max_serialized_size_v<holoscan::schema::TimeT>;
  static constexpr std::size_t kTransportCapacity =
      holoscan::kSchemaEnvelopeHeaderSize + kCanonicalCapacity;

  std::array<std::byte, kTransportCapacity> bytes{};
  std::size_t canonical_size{};

  [[nodiscard]] std::span<const std::byte> frame() const noexcept {
    return std::span<const std::byte>{bytes}.first(holoscan::kSchemaEnvelopeHeaderSize +
                                                   canonical_size);
  }
};

[[nodiscard]] holoscan::SchemaIdentityRecord time_schema_identity_record() {
  constexpr holoscan::SchemaIdentity kIdentity =
      holoscan::schema_identity<holoscan::schema::TimeT>();
  return {
      .canonical_name = std::string{kIdentity.canonical_name},
      .version = kIdentity.version,
      .schema_sha256 = kIdentity.schema_sha256,
      .id = kIdentity.id,
  };
}

[[nodiscard]] std::optional<TimeProcessLocalFrame> make_time_process_local_frame() {
  const auto builder = make_time_builder();
  if (builder.GetSize() == 0U || builder.GetSize() > TimeProcessLocalFrame::kCanonicalCapacity) {
    return std::nullopt;
  }
  TimeProcessLocalFrame frame{};
  frame.canonical_size = builder.GetSize();
  const holoscan::SchemaIdentityRecord identity = time_schema_identity_record();
  auto encoded = holoscan::channel_internal::encode_schema_envelope(
      std::span<std::byte>{frame.bytes}.first(holoscan::kSchemaEnvelopeHeaderSize),
      identity,
      frame.canonical_size);
  if (!encoded) {
    return std::nullopt;
  }
  const auto* const canonical = reinterpret_cast<const std::byte*>(builder.GetBufferPointer());
  std::copy_n(
      canonical, frame.canonical_size, frame.bytes.begin() + holoscan::kSchemaEnvelopeHeaderSize);
  return frame;
}

enum class ConcurrentChannelReceiveMode : std::uint8_t { kPolling, kKernelWaitSet };

// Keep both receive modes in one benchmark implementation so their measured publication paths
// remain identical.
// NOLINTBEGIN(readability-function-cognitive-complexity)
template <ConcurrentChannelReceiveMode ReceiveMode>
void run_channel_schema_concurrent_publisher_send(benchmark::State& state) {
  const auto batch_size = static_cast<std::uint64_t>(state.range(0));
  if (batch_size == 0U || batch_size > std::numeric_limits<std::uint32_t>::max()) {
    state.SkipWithError("Channel schema publisher batch is outside the supported range");
    return;
  }
  const std::optional<int> publisher_cpu = benchmark_cpu("HOLOSCAN_BENCHMARK_PUBLISHER_CPU");
  const std::optional<int> subscriber_cpu = benchmark_cpu("HOLOSCAN_BENCHMARK_SUBSCRIBER_CPU");
  if (!publisher_cpu.has_value() || !subscriber_cpu.has_value()) {
    state.SkipWithError("invalid Channel benchmark CPU environment");
    return;
  }
  ScopedBenchmarkCPUAffinity publisher_affinity;
  if (!publisher_affinity.pin(*publisher_cpu)) {
    state.SkipWithError("failed to pin Channel benchmark publisher");
    return;
  }

  const auto frame = make_time_process_local_frame();
  if (!frame.has_value()) {
    state.SkipWithError("failed to build Time process-local transport frame");
    return;
  }
  const holoscan::SchemaIdentityRecord identity = time_schema_identity_record();
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    state.SkipWithError("failed to create Channel runtime");
    return;
  }
  const std::string channel_name = make_unique_channel_benchmark_name(
      "holoscan_schema_channel_benchmark_",
      ReceiveMode == ConcurrentChannelReceiveMode::kPolling ? "time_publisher_polling"
                                                            : "time_publisher_waitset");
  holoscan::ChannelOptions options =
      make_channel_benchmark_options(holoscan::ChannelQueuePolicy::kBlocking);
  options.subscriber_queue_depth = static_cast<std::uint32_t>(batch_size);
  options.enable_event_wake = ReceiveMode == ConcurrentChannelReceiveMode::kKernelWaitSet;
  auto publisher = holoscan::channel_internal::InlinePayloadTestAccess::create_publisher(
      *runtime, channel_name, TimeProcessLocalFrame::kTransportCapacity, options);
  auto subscriber = holoscan::channel_internal::InlinePayloadTestAccess::create_subscriber(
      *runtime, channel_name, TimeProcessLocalFrame::kTransportCapacity, options);
  if (!publisher || !subscriber) {
    state.SkipWithError("failed to create Channel schema transport endpoints");
    return;
  }

  const holoscan::SampleMetadata metadata{
      .byte_size = static_cast<std::uint32_t>(frame->canonical_size),
      .payload_kind = holoscan::PayloadKind::kSerializedBytes,
  };
  std::array<std::byte, TimeProcessLocalFrame::kTransportCapacity> startup_scratch{};
  if (!wait_for_channel_subscribers(
          **publisher,
          1U,
          [&]() { return (*publisher)->emit(frame->frame(), metadata); },
          [&]() { static_cast<void>((*subscriber)->receive(startup_scratch)); })) {
    state.SkipWithError("timed out waiting for Channel schema transport subscriber");
    return;
  }

  // Only the wait-set template instantiation mutates these after construction.
  // NOLINTNEXTLINE(misc-const-correctness)
  std::unique_ptr<holoscan::channel_internal::InlinePayloadWaitSet> wait_set;
  std::uint64_t wait_handle = 0U;  // NOLINT(misc-const-correctness)
  if constexpr (ReceiveMode == ConcurrentChannelReceiveMode::kKernelWaitSet) {
    auto created = holoscan::channel_internal::InlinePayloadTestAccess::create_wait_set(*runtime);
    if (!created) {
      state.SkipWithError("failed to create Channel schema transport wait set");
      return;
    }
    wait_set = std::move(*created);
    auto attached = wait_set->attach(**subscriber);
    if (!attached) {
      state.SkipWithError("failed to attach Channel schema transport subscriber");
      return;
    }
    wait_handle = *attached;
  }

  std::barrier startup{2};
  std::atomic<bool> stop{false};
  std::atomic<bool> subscriber_affinity_ok{true};
  std::atomic<bool> subscriber_failed{false};
  std::atomic<std::uint64_t> receive_count{0U};
  std::uint64_t checksum = 0U;
  std::thread subscriber_thread([&]() {
    subscriber_affinity_ok.store(pin_current_benchmark_thread(*subscriber_cpu),
                                 std::memory_order_release);
    startup.arrive_and_wait();
    if (!subscriber_affinity_ok.load(std::memory_order_acquire)) {
      return;
    }

    std::array<std::byte, TimeProcessLocalFrame::kTransportCapacity> receive_scratch{};
    auto drain_ready_samples = [&]() {
      bool received_any = false;
      while (true) {
        auto received = (*subscriber)->receive(receive_scratch);
        if (!received) {
          if (received.error().code != holoscan::ErrorCode::kNotReady) {
            subscriber_failed.store(true, std::memory_order_release);
          }
          break;
        }
        received_any = true;
        if (received->payload_size != frame->canonical_size ||
            received->metadata.byte_size != frame->canonical_size ||
            received->metadata.payload_kind != holoscan::PayloadKind::kSerializedBytes ||
            !holoscan::channel_internal::decode_schema_envelope(
                receive_scratch, identity, TimeProcessLocalFrame::kCanonicalCapacity)) {
          subscriber_failed.store(true, std::memory_order_release);
          return false;
        }
        checksum ^=
            std::to_integer<std::uint8_t>(receive_scratch[holoscan::kSchemaEnvelopeHeaderSize]);
        receive_count.fetch_add(1U, std::memory_order_release);
      }
      return received_any;
    };
    while (!stop.load(std::memory_order_acquire)) {
      if constexpr (ReceiveMode == ConcurrentChannelReceiveMode::kKernelWaitSet) {
        auto waited = wait_set->wait_for(std::chrono::milliseconds{10});
        if (!waited) {
          subscriber_failed.store(true, std::memory_order_release);
          return;
        }
        if (!*waited) {
          continue;
        }
        std::array<std::uint64_t, 1U> ready{};
        auto ready_count = wait_set->ready_handles(ready);
        if (!ready_count) {
          subscriber_failed.store(true, std::memory_order_release);
          return;
        }
        if (std::find(ready.begin(), ready.begin() + *ready_count, wait_handle) ==
            ready.begin() + *ready_count) {
          continue;
        }
      }
      if (!drain_ready_samples() && ReceiveMode == ConcurrentChannelReceiveMode::kPolling) {
        std::this_thread::yield();
      }
    }
  });
  startup.arrive_and_wait();
  if (!subscriber_affinity_ok.load(std::memory_order_acquire)) {
    stop.store(true, std::memory_order_release);
    subscriber_thread.join();
    state.SkipWithError("failed to pin Channel benchmark subscriber");
    return;
  }

  std::int64_t completed_iterations = 0;
  std::uint64_t published_count = 0U;
  for (auto _ : state) {
    const std::int64_t start_ns = benchmark_now_ns();
    bool publish_failed = false;
    for (std::uint64_t index = 0U; index < batch_size; ++index) {
      const auto published = (*publisher)->emit_phase(frame->frame(), metadata);
      if (published.phase != holoscan::channel_internal::InlinePublicationPhase::kAccepted ||
          published.recipient_count != 1U) {
        publish_failed = true;
        break;
      }
    }
    const std::int64_t end_ns = benchmark_now_ns();
    const std::uint64_t expected_count = published_count + batch_size;
    if (publish_failed || !wait_for_benchmark_count(receive_count, expected_count)) {
      state.SkipWithError(publish_failed ? "Channel schema transport publication failed"
                                         : "Channel schema transport receive timed out");
      break;
    }
    published_count = expected_count;
    state.SetIterationTime(static_cast<double>(end_ns - start_ns) * 1.0e-9);
    ++completed_iterations;
  }

  stop.store(true, std::memory_order_release);
  subscriber_thread.join();
  if (subscriber_failed.load(std::memory_order_acquire)) {
    state.SkipWithError("Channel schema transport receive failed");
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(completed_iterations * static_cast<std::int64_t>(batch_size));
  state.counters["batch_size"] = static_cast<double>(batch_size);
  state.counters["canonical_bytes"] = static_cast<double>(frame->canonical_size);
  state.counters["concurrent"] = 1.0;
  state.counters["event_wake"] =
      ReceiveMode == ConcurrentChannelReceiveMode::kKernelWaitSet ? 1.0 : 0.0;
  state.counters["publisher_cpu"] = static_cast<double>(*publisher_cpu);
  state.counters["publisher_send_stage"] = 1.0;
  state.counters["receive_polling"] =
      ReceiveMode == ConcurrentChannelReceiveMode::kPolling ? 1.0 : 0.0;
  state.counters["subscriber_cpu"] = static_cast<double>(*subscriber_cpu);
  state.counters["subscribers"] = 1.0;
  state.counters["transport_bytes"] = static_cast<double>(frame->frame().size());
}
// NOLINTEND(readability-function-cognitive-complexity)

[[nodiscard]] holoscan::Tensor make_strided_cpu_tensor_3x2() {
  static std::vector<float> data{1.0F, 2.0F, 0.0F, 0.0F, 3.0F, 4.0F, 0.0F, 0.0F, 5.0F, 6.0F};
  auto context = std::make_shared<holoscan::DLManagedTensorContext>();
  context->dl_shape = {3, 2};
  context->dl_strides = {4, 1};
  context->tensor.dl_tensor.data = data.data();
  context->tensor.dl_tensor.device = DLDevice{.device_type = kDLCPU, .device_id = 0};
  context->tensor.dl_tensor.ndim = 2;
  context->tensor.dl_tensor.dtype = DLDataType{.code = kDLFloat, .bits = 32, .lanes = 1};
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = context->dl_strides.data();
  context->tensor.dl_tensor.byte_offset = 0;
  return holoscan::Tensor(std::move(context));
}

[[nodiscard]] ImagePayload make_image_metadata_payload() {
  const holoscan::Tensor tensor = make_strided_cpu_tensor_3x2();
  const holoscan::schema::ImageMetadataOptions metadata_options{
      .height = 3,
      .width = 2,
      .encoding = holoscan::schema::ImageEncoding_MONO32F,
      .capture_timestamp_ns = 987654321,
      .frame_id = "camera_frame"};
  auto metadata = holoscan::schema::build_image_metadata(tensor, metadata_options);
  if (!metadata) {
    return {};
  }
  auto payload = ImagePayload::copy_from(
      std::span<const std::byte>(metadata->data(), metadata->size()), kBenchmarkEndpointGeneration);
  if (!payload) {
    return {};
  }
  return payload.value();
}

[[nodiscard]] std::vector<std::byte> make_image_metadata_bytes(
    const holoscan::Tensor& tensor, const char* const frame_id, const std::int32_t height = 3,
    const std::int32_t width = 2,
    const holoscan::schema::ImageEncoding encoding = holoscan::schema::ImageEncoding_MONO32F) {
  const holoscan::schema::ImageMetadataOptions metadata_options{.height = height,
                                                                .width = width,
                                                                .encoding = encoding,
                                                                .capture_timestamp_ns = 987654321,
                                                                .frame_id = frame_id};
  auto metadata = holoscan::schema::build_image_metadata(tensor, metadata_options);
  if (!metadata) {
    return {};
  }
  return metadata.value();
}

template <typename Publisher>
[[nodiscard]] std::vector<holoscan::Tensor> make_tensor_schema_pool(
    const Publisher& publisher, const std::size_t pool_size = kTensorSchemaPoolSize) {
  auto memory_options = publisher.tensor_memory_options();
  if (!memory_options.has_value()) {
    return {};
  }

  auto allocator = std::make_shared<holoscan::UnboundedAllocator>();
  const std::array<int64_t, 2U> shape{3, 2};
  const auto dtype = DLDataType{.code = kDLFloat, .bits = 32U, .lanes = 1U};

  std::vector<holoscan::Tensor> tensors;
  tensors.reserve(pool_size);
  for (std::size_t i = 0U; i < pool_size; ++i) {
    auto tensor = holoscan::make_tensor(shape,
                                        dtype,
                                        std::nullopt,
                                        holoscan::MemoryStorageType::kHost,
                                        allocator,
                                        memory_options.value());
    if (!tensor.has_value()) {
      return {};
    }
    tensors.emplace_back(std::move(tensor).value());
  }
  return tensors;
}

[[nodiscard]] holoscan::ChannelSampleDescriptor make_tensor_data_descriptor(
    const holoscan::Tensor& tensor, const std::uint64_t sequence = 1U) {
  return holoscan::ChannelSampleDescriptor{
      .magic = holoscan::kChannelSampleDescriptorMagic,
      .abi_version = holoscan::kChannelSampleDescriptorAbiVersion,
      .payload_kind = holoscan::ChannelPayloadKind::kLocalBufferDescriptor,
      .flags = 0U,
      .expected_recipient_count = 1U,
      .process_id = benchmark_process_id(),
      .sidecar_token = 0x7000U + sequence,
      .generation = 0x9000U + sequence,
      .schema_id = holoscan::schema_identity<holoscan::schema::Tensor>().id,
      .sequence = sequence,
      .byte_size = static_cast<std::uint64_t>(tensor.nbytes()),
  };
}

[[nodiscard]] DescriptorBackedImagePayload make_descriptor_backed_image_payload() {
  const holoscan::Tensor tensor = make_strided_cpu_tensor_3x2();
  const holoscan::schema::ImageMetadataOptions metadata_options{
      .height = 3,
      .width = 2,
      .encoding = holoscan::schema::ImageEncoding_MONO32F,
      .capture_timestamp_ns = 987654321,
      .frame_id = "camera_frame"};
  auto metadata = holoscan::schema::build_image_metadata(tensor, metadata_options);
  if (!metadata) {
    return {};
  }
  // The descriptor is fabricated for the current benchmark slice. It measures metadata + descriptor
  // transport only; no TensorSidecarStore record is minted or resolved yet.
  auto payload = DescriptorBackedImagePayload::copy_from(
      std::span<const std::byte>(metadata->data(), metadata->size()),
      make_tensor_data_descriptor(tensor),
      kBenchmarkEndpointGeneration);
  if (!payload) {
    return {};
  }
  return payload.value();
}

[[nodiscard]] std::unique_ptr<HoloMQSchemaBufferFixture> make_holomq_schema_buffer_fixture() {
  const std::string socket_path =
      make_unique_holomq_benchmark_socket_path("hsdk_channel_schema_benchmark_");
  auto context =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  if (!context.has_value()) {
    cleanup_holomq_benchmark_socket_path(socket_path);
    return nullptr;
  }

  auto pool = context->create_host_pinned_buffer_pool(
      kHoloMQSchemaTensorBytes, /*buffer_count=*/1U, CU_MEMHOSTREGISTER_PORTABLE);
  if (!pool.has_value()) {
    cleanup_holomq_benchmark_socket_path(socket_path);
    return nullptr;
  }

  auto buffer = pool->acquire();
  if (!buffer.has_value()) {
    cleanup_holomq_benchmark_socket_path(socket_path);
    return nullptr;
  }
  auto handle = buffer->export_handle_unsynchronized();
  if (!handle.has_value()) {
    cleanup_holomq_benchmark_socket_path(socket_path);
    return nullptr;
  }

  const std::array<int64_t, 2U> shape{4, 4};
  auto source_tensor = holoscan::make_tensor(shape,
                                             DLDataType{.code = kDLUInt, .bits = 32U, .lanes = 1U},
                                             std::nullopt,
                                             holoscan::MemoryStorageType::kHost,
                                             buffer->ptr());
  if (!source_tensor.has_value()) {
    cleanup_holomq_benchmark_socket_path(socket_path);
    return nullptr;
  }

  const auto metadata = make_image_metadata_bytes(source_tensor.value(),
                                                  "holomq_schema_camera",
                                                  /*height=*/4,
                                                  /*width=*/4,
                                                  holoscan::schema::ImageEncoding_MONO32);
  if (metadata.empty()) {
    cleanup_holomq_benchmark_socket_path(socket_path);
    return nullptr;
  }

  const holoscan::PayloadDescriptor descriptor{
      .payload_kind = holoscan::ChannelPayloadKind::kBufferHandle,
      .schema_id = holoscan::schema_identity<holoscan::schema::Image>().id,
      .sequence = 1U,
      .byte_size = kHoloMQSchemaTensorBytes,
      .buffer = holoscan::channel_internal::make_holomq_payload_reference(*handle)};
  auto payload = DescriptorBackedImagePayload::copy_from(
      std::span<const std::byte>(metadata.data(), metadata.size()),
      descriptor,
      kBenchmarkEndpointGeneration);
  if (!payload.has_value()) {
    cleanup_holomq_benchmark_socket_path(socket_path);
    return nullptr;
  }

  return std::make_unique<HoloMQSchemaBufferFixture>(socket_path,
                                                     std::move(context).value(),
                                                     std::move(pool).value(),
                                                     std::move(buffer).value(),
                                                     std::move(payload).value());
}

template <typename Payload>
[[nodiscard]] bool verify_payload(const Payload& payload) {
  auto valid = payload.validate();
  if (!valid) {
    return false;
  }
  if constexpr (std::is_same_v<typename Payload::root_type, holoscan::schema::Image>) {
    return holoscan::schema::parse_image_metadata(payload.bytes()).has_value();
  }
  return true;
}

[[nodiscard]] std::uint64_t checksum_schema_id(const holoscan::SchemaId& schema_id) noexcept {
  std::uint64_t checksum = 14695981039346656037ULL;
  for (const std::byte byte : schema_id.value) {
    checksum ^= std::to_integer<std::uint8_t>(byte);
    checksum *= 1099511628211ULL;
  }
  return checksum;
}

template <typename Payload>
[[nodiscard]] std::uint64_t checksum_payload(const Payload& payload) noexcept {
  const auto bytes = payload.bytes();
  std::uint64_t checksum =
      checksum_schema_id(payload.schema_id) ^ static_cast<std::uint64_t>(bytes.size());
  if constexpr (requires { payload.endpoint_generation; }) {
    checksum ^= payload.endpoint_generation;
  }
  const std::size_t sampled_bytes = std::min<std::size_t>(bytes.size(), 16U);
  for (std::size_t i = 0; i < sampled_bytes; ++i) {
    checksum ^= std::to_integer<std::uint64_t>(bytes[i]) << ((i % 8U) * 8U);
  }
  if constexpr (requires { payload.descriptor_at(0U); }) {
    const holoscan::PayloadDescriptor* descriptor = payload.descriptor_at(0U);
    if (descriptor != nullptr) {
      checksum ^= descriptor->byte_size;
      checksum ^= checksum_schema_id(descriptor->schema_id);
      if (descriptor->payload_kind == holoscan::ChannelPayloadKind::kLocalBufferDescriptor) {
        checksum ^= descriptor->local.token;
        checksum ^= descriptor->local.generation;
      }
    }
  }
  return checksum;
}

template <typename Payload>
[[nodiscard]] std::uint64_t checksum_resolved_tensor_sample(
    const holoscan::ChannelResolvedTensorSample<Payload>& sample) noexcept {
  std::uint64_t checksum = checksum_payload(sample.metadata);
  checksum ^= reinterpret_cast<std::uintptr_t>(sample.tensor.data());
  checksum ^= static_cast<std::uint64_t>(sample.tensor.nbytes());
  if (sample.payload.payload_kind == holoscan::ChannelPayloadKind::kLocalBufferDescriptor) {
    checksum ^= sample.payload.local.token;
    checksum ^= sample.payload.local.generation;
  } else if (sample.payload.payload_kind == holoscan::ChannelPayloadKind::kBufferHandle) {
    checksum ^= sample.payload.byte_size;
    checksum ^= checksum_schema_id(sample.payload.schema_id);
  }
  return checksum;
}

template <typename SubscriberContainer>
void drain_subscribers(SubscriberContainer& subscribers, std::uint64_t& checksum) {
  for (auto& subscriber : subscribers) {
    while (true) {
      auto sample = subscriber.receive();
      if (!sample.has_value()) {
        break;
      }
      checksum ^= checksum_payload(sample.value());
    }
  }
}

template <typename SubscriberContainer>
void drain_tensor_schema_subscribers(SubscriberContainer& subscribers, std::uint64_t& checksum) {
  for (auto& subscriber : subscribers) {
    while (true) {
      auto sample = subscriber.receive();
      if (!sample.has_value()) {
        break;
      }
      checksum ^= checksum_resolved_tensor_sample(sample.value());
    }
  }
}

template <typename Payload>
[[nodiscard]] bool receive_tensor_schema_sample_or_skip(
    benchmark::State& state, holoscan::ChannelTensorSchemaSubscriber<Payload>& subscriber,
    holoscan::ChannelResolvedTensorSample<Payload>& sample) {
  while (true) {
    auto received = subscriber.receive();
    if (received.has_value()) {
      sample = std::move(received).value();
      return true;
    }
    if (received.error().code != holoscan::ErrorCode::kNotReady) {
      const std::string message = "Tensor schema Channel receive failed: " +
                                  holoscan::runtime_error_message(received.error());
      state.SkipWithError(message.c_str());
      return false;
    }
  }
}

template <typename Payload, typename SubscriberContainer>
[[nodiscard]] bool wait_for_subscriber_count(holoscan::ChannelPublisher<Payload>& publisher,
                                             SubscriberContainer& subscribers,
                                             const Payload& payload,
                                             const std::uint32_t expected_recipients) {
  std::uint64_t ignored_checksum = 0U;
  return wait_for_channel_subscribers(
      publisher,
      expected_recipients,
      [&publisher, &payload]() { return publisher.emit(payload); },
      [&subscribers, &ignored_checksum]() { drain_subscribers(subscribers, ignored_checksum); });
}

template <typename Payload, typename SubscriberContainer>
[[nodiscard]] bool wait_for_tensor_schema_subscriber_count(
    holoscan::ChannelTensorSchemaPublisher<Payload>& publisher, SubscriberContainer& subscribers,
    const std::span<const std::byte> metadata, const std::uint32_t expected_recipients,
    const std::vector<holoscan::Tensor>& tensors) {
  std::uint64_t ignored_checksum = 0U;
  std::size_t index = 0U;
  return wait_for_channel_subscribers(
      publisher,
      expected_recipients,
      [&publisher, &metadata, &tensors, &index]() {
        return publisher.emit(metadata, holoscan::Tensor(tensors[index++ % tensors.size()]));
      },
      [&subscribers, &ignored_checksum]() {
        drain_tensor_schema_subscribers(subscribers, ignored_checksum);
      });
}

template <typename Payload, typename SubscriberContainer>
[[nodiscard]] bool wait_for_tensor_schema_payload_subscriber_count(
    holoscan::ChannelTensorSchemaPublisher<Payload>& publisher, SubscriberContainer& subscribers,
    const Payload& payload, const std::uint32_t expected_recipients) {
  std::uint64_t ignored_checksum = 0U;
  return wait_for_channel_subscribers(
      publisher,
      expected_recipients,
      [&publisher, &payload]() { return publisher.emit_payload(payload); },
      [&subscribers, &ignored_checksum]() {
        drain_tensor_schema_subscribers(subscribers, ignored_checksum);
      });
}

template <typename Payload, std::size_t SubscriberCount>
void run_channel_schema_pubsub_polling(benchmark::State& state,
                                       const holoscan::ChannelQueuePolicy policy,
                                       const char* const channel_suffix, const Payload& payload,
                                       const std::uint32_t queue_depth = 64U) {
  if (!verify_payload(payload)) {
    state.SkipWithError("invalid schema payload");
    return;
  }

  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime.has_value()) {
    const std::string message =
        "failed to create ChannelRuntime: " + holoscan::runtime_error_message(runtime.error());
    state.SkipWithError(message.c_str());
    return;
  }

  const std::string channel_name =
      make_unique_channel_benchmark_name("holoscan_schema_channel_benchmark_", channel_suffix);
  const holoscan::ChannelOptions options = make_channel_benchmark_options(
      policy, static_cast<std::uint32_t>(SubscriberCount), queue_depth);
  auto publisher = runtime->create_publisher<Payload>(channel_name, options);
  if (!publisher.has_value()) {
    const std::string message =
        "failed to create Channel publisher: " + holoscan::runtime_error_message(publisher.error());
    state.SkipWithError(message.c_str());
    return;
  }

  std::vector<holoscan::ChannelSubscriber<Payload>> subscribers;
  subscribers.reserve(SubscriberCount);
  for (std::size_t i = 0; i < SubscriberCount; ++i) {
    auto subscriber = runtime->create_subscriber<Payload>(channel_name, options);
    if (!subscriber.has_value()) {
      const std::string message = "failed to create Channel subscriber: " +
                                  holoscan::runtime_error_message(subscriber.error());
      state.SkipWithError(message.c_str());
      return;
    }
    subscribers.emplace_back(std::move(subscriber).value());
  }

  if (!wait_for_subscriber_count(
          publisher.value(), subscribers, payload, static_cast<std::uint32_t>(SubscriberCount))) {
    state.SkipWithError("timed out waiting for Channel subscribers");
    return;
  }

  std::uint64_t checksum = 0U;
  for (auto _ : state) {
    auto emitted = publisher->emit(payload);
    if (!emitted.has_value()) {
      const std::string message =
          "Channel publish failed: " + holoscan::runtime_error_message(emitted.error());
      state.SkipWithError(message.c_str());
      break;
    }
    if (emitted.value() < SubscriberCount) {
      state.SkipWithError("Channel publish did not reach all subscribers");
      break;
    }

    for (auto& subscriber : subscribers) {
      std::optional<Payload> sample;
      while (!sample.has_value()) {
        auto received = subscriber.receive();
        if (received.has_value()) {
          sample = received.value();
        } else if (received.error().code != holoscan::ErrorCode::kNotReady) {
          const std::string message =
              "Channel receive failed: " + holoscan::runtime_error_message(received.error());
          state.SkipWithError(message.c_str());
          break;
        }
      }
      if (!sample.has_value()) {
        break;
      }
      checksum ^= checksum_payload(sample.value());
    }
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(SubscriberCount));
  state.counters["subscribers"] = static_cast<double>(SubscriberCount);
  state.counters["payload_bytes"] = static_cast<double>(sizeof(Payload));
  state.counters["transport_bucket_bytes"] =
      static_cast<double>(transport_bucket_size_for_payload(sizeof(Payload)));
  state.counters["flatbuffer_bytes"] = static_cast<double>(payload.bytes().size());
}

template <typename Payload, std::size_t SubscriberCount>
void run_channel_tensor_schema_pubsub_polling(benchmark::State& state,
                                              const holoscan::ChannelQueuePolicy policy,
                                              const char* const channel_suffix,
                                              const std::uint32_t queue_depth = 64U) {
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime.has_value()) {
    const std::string message =
        "failed to create ChannelRuntime: " + holoscan::runtime_error_message(runtime.error());
    state.SkipWithError(message.c_str());
    return;
  }

  const std::string channel_name =
      make_unique_channel_benchmark_name("holoscan_schema_channel_benchmark_", channel_suffix);
  const holoscan::ChannelOptions options = make_channel_benchmark_options(
      policy, static_cast<std::uint32_t>(SubscriberCount), queue_depth);
  auto publisher = runtime->create_tensor_schema_publisher<Payload>(channel_name, options);
  if (!publisher.has_value()) {
    const std::string message = "failed to create Tensor schema Channel publisher: " +
                                holoscan::runtime_error_message(publisher.error());
    state.SkipWithError(message.c_str());
    return;
  }

  std::vector<holoscan::ChannelTensorSchemaSubscriber<Payload>> subscribers;
  subscribers.reserve(SubscriberCount);
  for (std::size_t i = 0U; i < SubscriberCount; ++i) {
    auto subscriber = runtime->create_tensor_schema_subscriber<Payload>(channel_name, options);
    if (!subscriber.has_value()) {
      const std::string message = "failed to create Tensor schema Channel subscriber: " +
                                  holoscan::runtime_error_message(subscriber.error());
      state.SkipWithError(message.c_str());
      return;
    }
    subscribers.emplace_back(std::move(subscriber).value());
  }

  const std::vector<holoscan::Tensor> tensors = make_tensor_schema_pool(publisher.value());
  if (tensors.empty()) {
    state.SkipWithError("failed to allocate benchmark Tensor schema pool");
    return;
  }

  const auto metadata = make_image_metadata_bytes(tensors.front(), "schema_materialized_camera");
  if (metadata.empty()) {
    state.SkipWithError("failed to build benchmark Image metadata");
    return;
  }
  const auto metadata_view = std::span<const std::byte>(metadata.data(), metadata.size());
  if (!wait_for_tensor_schema_subscriber_count(publisher.value(),
                                               subscribers,
                                               metadata_view,
                                               static_cast<std::uint32_t>(SubscriberCount),
                                               tensors)) {
    state.SkipWithError("timed out waiting for Tensor schema Channel subscribers");
    return;
  }

  std::uint64_t checksum = 0U;
  std::size_t tensor_index = 0U;
  for (auto _ : state) {
    auto emitted =
        publisher->emit(metadata_view, holoscan::Tensor(tensors[tensor_index++ % tensors.size()]));
    if (!emitted.has_value()) {
      const std::string message = "Tensor schema Channel publish failed: " +
                                  holoscan::runtime_error_message(emitted.error());
      state.SkipWithError(message.c_str());
      break;
    }
    if (emitted.value() < SubscriberCount) {
      state.SkipWithError("Tensor schema Channel publish did not reach all subscribers");
      break;
    }

    for (auto& subscriber : subscribers) {
      holoscan::ChannelResolvedTensorSample<Payload> sample{};
      if (!receive_tensor_schema_sample_or_skip(state, subscriber, sample)) {
        break;
      }
      checksum ^= checksum_resolved_tensor_sample(sample);
    }
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(SubscriberCount));
  state.counters["subscribers"] = static_cast<double>(SubscriberCount);
  state.counters["payload_bytes"] = static_cast<double>(sizeof(Payload));
  state.counters["transport_bucket_bytes"] =
      static_cast<double>(transport_bucket_size_for_payload(sizeof(Payload)));
  state.counters["flatbuffer_bytes"] = static_cast<double>(metadata.size());
  state.counters["tensor_bytes"] = static_cast<double>(tensors.front().nbytes());
  state.counters["materializes_tensor"] = 1.0;
}

void run_channel_holomq_image_tensor_schema_pubsub_polling(benchmark::State& state) {
  if (cuda_unavailable_for_holomq_pool(state)) {
    return;
  }

  auto fixture = make_holomq_schema_buffer_fixture();
  if (!fixture) {
    state.SkipWithError("failed to create HoloMQ schema buffer fixture");
    return;
  }

  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime.has_value()) {
    const std::string message =
        "failed to create ChannelRuntime: " + holoscan::runtime_error_message(runtime.error());
    state.SkipWithError(message.c_str());
    return;
  }

  const std::string channel_name = make_unique_channel_benchmark_name(
      "holoscan_schema_channel_benchmark_", "schema_holomq_image_1to1_blocking");
  const holoscan::ChannelOptions options =
      make_channel_benchmark_options(holoscan::ChannelQueuePolicy::kBlocking);
  auto publisher =
      runtime->create_tensor_schema_publisher<DescriptorBackedImagePayload>(channel_name, options);
  if (!publisher.has_value()) {
    const std::string message = "failed to create HoloMQ Image schema publisher: " +
                                holoscan::runtime_error_message(publisher.error());
    state.SkipWithError(message.c_str());
    return;
  }
  auto subscriber =
      runtime->create_tensor_schema_subscriber<DescriptorBackedImagePayload>(channel_name, options);
  if (!subscriber.has_value()) {
    const std::string message = "failed to create HoloMQ Image schema subscriber: " +
                                holoscan::runtime_error_message(subscriber.error());
    state.SkipWithError(message.c_str());
    return;
  }

  std::array<holoscan::ChannelTensorSchemaSubscriber<DescriptorBackedImagePayload>, 1U> subscribers{
      std::move(subscriber).value()};
  if (!wait_for_tensor_schema_payload_subscriber_count(
          publisher.value(), subscribers, fixture->payload, /*expected_recipients=*/1U)) {
    state.SkipWithError("timed out waiting for HoloMQ Image schema subscriber");
    return;
  }

  std::uint64_t checksum = 0U;
  for (auto _ : state) {
    auto emitted = publisher->emit_payload(fixture->payload);
    if (!emitted.has_value()) {
      const std::string message =
          "HoloMQ Image schema publish failed: " + holoscan::runtime_error_message(emitted.error());
      state.SkipWithError(message.c_str());
      break;
    }
    if (emitted.value() < 1U) {
      state.SkipWithError("HoloMQ Image schema publish did not reach the subscriber");
      break;
    }

    holoscan::ChannelResolvedTensorSample<DescriptorBackedImagePayload> sample{};
    if (!receive_tensor_schema_sample_or_skip(state, subscribers[0], sample)) {
      break;
    }
    checksum ^= checksum_resolved_tensor_sample(sample);
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(state.iterations());
  state.counters["subscribers"] = 1.0;
  state.counters["payload_bytes"] = static_cast<double>(sizeof(DescriptorBackedImagePayload));
  state.counters["transport_bucket_bytes"] =
      static_cast<double>(transport_bucket_size_for_payload(sizeof(DescriptorBackedImagePayload)));
  state.counters["flatbuffer_bytes"] = static_cast<double>(fixture->payload.bytes().size());
  state.counters["tensor_bytes"] = static_cast<double>(kHoloMQSchemaTensorBytes);
  state.counters["materializes_tensor"] = 1.0;
  state.counters["holomq_buffer_handle"] = 1.0;
}

[[nodiscard]] ImuPayload make_imu_payload() {
  const auto builder = make_imu_builder();
  auto payload = ImuPayload::copy_from_builder(builder, kBenchmarkEndpointGeneration);
  if (!payload) {
    return {};
  }
  return payload.value();
}

void BM_ChannelSchema_ImuFlatBuffer_OneToOne_Blocking(benchmark::State& state) {
  run_channel_schema_pubsub_polling<ImuPayload, 1>(state,
                                                   holoscan::ChannelQueuePolicy::kBlocking,
                                                   "schema_imu_1to1_blocking",
                                                   make_imu_payload());
}

void BM_ChannelSchema_TimeTransport_ConcurrentPollingPublisherSend(benchmark::State& state) {
  run_channel_schema_concurrent_publisher_send<ConcurrentChannelReceiveMode::kPolling>(state);
}

void BM_ChannelSchema_TimeTransport_ConcurrentWaitSetPublisherSend(benchmark::State& state) {
  run_channel_schema_concurrent_publisher_send<ConcurrentChannelReceiveMode::kKernelWaitSet>(state);
}

void BM_ChannelSchema_ImuFlatBuffer_OneToOne_DropOldestDepthOne(benchmark::State& state) {
  run_channel_schema_pubsub_polling<ImuPayload, 1>(state,
                                                   holoscan::ChannelQueuePolicy::kDropOldest,
                                                   "schema_imu_1to1_drop_oldest_depth_1",
                                                   make_imu_payload(),
                                                   1U);
}

template <std::size_t SubscriberCount>
void BM_ChannelSchema_ImuFlatBuffer_Fanout_Blocking(benchmark::State& state) {
  run_channel_schema_pubsub_polling<ImuPayload, SubscriberCount>(
      state,
      holoscan::ChannelQueuePolicy::kBlocking,
      "schema_imu_fanout_blocking",
      make_imu_payload());
}

template <std::size_t SubscriberCount>
void BM_ChannelSchema_ImuFlatBuffer_Fanout_DropOldestDepthOne(benchmark::State& state) {
  run_channel_schema_pubsub_polling<ImuPayload, SubscriberCount>(
      state,
      holoscan::ChannelQueuePolicy::kDropOldest,
      "schema_imu_fanout_drop_oldest_depth_1",
      make_imu_payload(),
      1U);
}

void BM_ChannelSchema_ImageMetadataFlatBuffer_OneToOne_Blocking(benchmark::State& state) {
  run_channel_schema_pubsub_polling<ImagePayload, 1>(state,
                                                     holoscan::ChannelQueuePolicy::kBlocking,
                                                     "schema_image_metadata_1to1_blocking",
                                                     make_image_metadata_payload());
}

void BM_ChannelSchema_DescriptorBackedImageFlatBuffer_OneToOne_Blocking(benchmark::State& state) {
  run_channel_schema_pubsub_polling<DescriptorBackedImagePayload, 1>(
      state,
      holoscan::ChannelQueuePolicy::kBlocking,
      "schema_descriptor_image_1to1_blocking",
      make_descriptor_backed_image_payload());
}

void BM_ChannelSchema_DescriptorBackedImageFlatBuffer_OneToOne_DropOldestDepthOne(
    benchmark::State& state) {
  run_channel_schema_pubsub_polling<DescriptorBackedImagePayload, 1>(
      state,
      holoscan::ChannelQueuePolicy::kDropOldest,
      "schema_descriptor_image_1to1_drop_oldest_depth_1",
      make_descriptor_backed_image_payload(),
      1U);
}

void BM_ChannelSchema_DescriptorBackedImageTensorSchema_OneToOne_Blocking(benchmark::State& state) {
  run_channel_tensor_schema_pubsub_polling<DescriptorBackedImagePayload, 1>(
      state, holoscan::ChannelQueuePolicy::kBlocking, "schema_tensor_image_1to1_blocking");
}

void BM_ChannelSchema_DescriptorBackedImageTensorSchema_OneToOne_DropOldestDepthOne(
    benchmark::State& state) {
  run_channel_tensor_schema_pubsub_polling<DescriptorBackedImagePayload, 1>(
      state,
      holoscan::ChannelQueuePolicy::kDropOldest,
      "schema_tensor_image_1to1_drop_oldest_depth_1",
      1U);
}

void BM_ChannelSchema_DescriptorBackedImageHoloMQBuffer_OneToOne_Blocking(benchmark::State& state) {
  run_channel_holomq_image_tensor_schema_pubsub_polling(state);
}

template <std::size_t SubscriberCount>
void BM_ChannelSchema_DescriptorBackedImageTensorSchema_Fanout_Blocking(benchmark::State& state) {
  run_channel_tensor_schema_pubsub_polling<DescriptorBackedImagePayload, SubscriberCount>(
      state, holoscan::ChannelQueuePolicy::kBlocking, "schema_tensor_image_fanout_blocking");
}

template <std::size_t SubscriberCount>
void BM_ChannelSchema_DescriptorBackedImageTensorSchema_Fanout_DropOldestDepthOne(
    benchmark::State& state) {
  run_channel_tensor_schema_pubsub_polling<DescriptorBackedImagePayload, SubscriberCount>(
      state,
      holoscan::ChannelQueuePolicy::kDropOldest,
      "schema_tensor_image_fanout_drop_oldest_depth_1",
      1U);
}

BENCHMARK(BM_ChannelSchema_ImuFlatBuffer_OneToOne_Blocking)->UseRealTime();
BENCHMARK(BM_ChannelSchema_ImuFlatBuffer_OneToOne_DropOldestDepthOne)->UseRealTime();
BENCHMARK(BM_ChannelSchema_TimeTransport_ConcurrentPollingPublisherSend)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK(BM_ChannelSchema_TimeTransport_ConcurrentWaitSetPublisherSend)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_ChannelSchema_ImuFlatBuffer_Fanout_Blocking, 2)->UseRealTime();
BENCHMARK_TEMPLATE(BM_ChannelSchema_ImuFlatBuffer_Fanout_Blocking, 4)->UseRealTime();
BENCHMARK_TEMPLATE(BM_ChannelSchema_ImuFlatBuffer_Fanout_Blocking, 8)->UseRealTime();
BENCHMARK_TEMPLATE(BM_ChannelSchema_ImuFlatBuffer_Fanout_DropOldestDepthOne, 2)->UseRealTime();
BENCHMARK_TEMPLATE(BM_ChannelSchema_ImuFlatBuffer_Fanout_DropOldestDepthOne, 4)->UseRealTime();
BENCHMARK_TEMPLATE(BM_ChannelSchema_ImuFlatBuffer_Fanout_DropOldestDepthOne, 8)->UseRealTime();
BENCHMARK(BM_ChannelSchema_ImageMetadataFlatBuffer_OneToOne_Blocking)->UseRealTime();
BENCHMARK(BM_ChannelSchema_DescriptorBackedImageFlatBuffer_OneToOne_Blocking)->UseRealTime();
BENCHMARK(BM_ChannelSchema_DescriptorBackedImageFlatBuffer_OneToOne_DropOldestDepthOne)
    ->UseRealTime();
BENCHMARK(BM_ChannelSchema_DescriptorBackedImageTensorSchema_OneToOne_Blocking)->UseRealTime();
BENCHMARK(BM_ChannelSchema_DescriptorBackedImageTensorSchema_OneToOne_DropOldestDepthOne)
    ->UseRealTime();
BENCHMARK(BM_ChannelSchema_DescriptorBackedImageHoloMQBuffer_OneToOne_Blocking)->UseRealTime();
BENCHMARK_TEMPLATE(BM_ChannelSchema_DescriptorBackedImageTensorSchema_Fanout_Blocking, 2)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_ChannelSchema_DescriptorBackedImageTensorSchema_Fanout_Blocking, 4)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_ChannelSchema_DescriptorBackedImageTensorSchema_Fanout_Blocking, 8)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_ChannelSchema_DescriptorBackedImageTensorSchema_Fanout_DropOldestDepthOne, 2)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_ChannelSchema_DescriptorBackedImageTensorSchema_Fanout_DropOldestDepthOne, 4)
    ->UseRealTime();
BENCHMARK_TEMPLATE(BM_ChannelSchema_DescriptorBackedImageTensorSchema_Fanout_DropOldestDepthOne, 8)
    ->UseRealTime();

}  // namespace
