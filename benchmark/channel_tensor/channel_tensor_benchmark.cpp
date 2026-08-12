/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <holomq/context_builder.hpp>
#include <holomq/inbound_message.hpp>
#include <holomq/publisher.hpp>
#include <holomq/pubsub_options.hpp>
#include <holomq/subscriber.hpp>
#include <holoscan/channel/channel.hpp>
#include <holoscan/channel/tensor_endpoints.hpp>
#include <holoscan/core/domain/dlpack_utils.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/memory/allocator.hpp>
#include <holoscan/memory/block_memory_pool.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>

#include "../../src/channel/channel_tensor_sidecar_store.hpp"
#include "../../src/channel/channel_tensor_test_access.hpp"
#include "../../src/channel/channel_tensor_transport_payload.hpp"
#include "../../src/channel/endpoint_catalog_state.hpp"
#include "../../src/channel/holomq_tensor.hpp"
#include "../common/channel_benchmark_common.hpp"

namespace {

using holoscan::benchmarking::make_channel_benchmark_options;
using holoscan::benchmarking::wait_for_channel_subscribers;

constexpr std::size_t kTensorBytes = 64U;
constexpr std::size_t kTensorPoolSize = 128U;
constexpr std::uint32_t kMinOutstandingSweepQueueDepth = 64U;
constexpr std::uint64_t kBenchmarkProducerGeneration = 1U;

enum class ReplacementCleanupMode { kTransportCallback, kPublisherMirror, kBlockingBaseline };
enum class LocalPayloadWork { kTransportOnly, kCallbackOnly, kSidecar };
enum class LocalTransportStage { kPublish, kTake };
enum class BufferHandleEmitStage { kExportDescriptor, kConstructDescriptor };
// BufferHandle receive stages after the typed transport has yielded its descriptor.
enum class BufferHandleReceiveStage { kImport, kMaterialize, kRelease };

[[nodiscard]] std::string make_unique_channel_name(const char* const suffix) {
  return holoscan::benchmarking::make_unique_channel_benchmark_name(
      "holoscan_tensor_channel_benchmark_", suffix);
}

[[nodiscard]] holoscan::expected<holoscan::ChannelEndpointCatalog, holoscan::Error>
make_catalog_tensor_benchmark(const std::string& endpoint_name, const std::uint32_t slot_capacity) {
  holoscan::detail::ChannelEndpointRecord endpoint{
      .name = endpoint_name,
      .uri = "holomq://" + endpoint_name,
      .direction = holoscan::TopicDirection::kImport,
      .name_origin = holoscan::EndpointNameOrigin::kExplicit,
      .queue_policy = holoscan::QueuePolicy::kBlocking,
      .payload_identity =
          holoscan::PayloadIdentityRecord{
              .kind = holoscan::PayloadIdentityKind::kTensor,
              .payload_kind = holoscan::PayloadKind::kDescriptorBacked,
          },
      .tensor =
          holoscan::LogicalTensorRequirementsRecord{
              .memory_kind = holoscan::MemoryKind::kHost,
              .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
              .max_rank = 1U,
              .max_byte_span = kTensorBytes,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
              .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded,
          },
      .bounds =
          holoscan::EndpointBounds{
              .publisher = holoscan::PublisherEndpointBounds{.max_loaned_samples = slot_capacity},
          },
  };
  return holoscan::detail::ChannelEndpointCatalogAccess::normalize("catalog-tensor-loan-benchmark",
                                                                   {std::move(endpoint)});
}

[[nodiscard]] holomq::PubSubServiceOptions make_holomq_service_options(
    const holomq::QueuePolicy policy, const std::uint32_t max_subscribers,
    const std::uint32_t max_subscriber_queue_depth) {
  holomq::PubSubServiceOptions options{};
  options.queue_policy = policy;
  options.max_subscribers = max_subscribers;
  options.max_subscriber_queue_depth = max_subscriber_queue_depth;
  options.publisher_history_depth = 0U;
  options.enable_event_wake = false;
  return options;
}

[[nodiscard]] std::shared_ptr<holoscan::Allocator> make_pool_allocator(
    const holoscan::MemoryStorageType storage_type, const std::size_t pool_size) {
  if (storage_type != holoscan::MemoryStorageType::kDevice) {
    return std::make_shared<holoscan::UnboundedAllocator>();
  }
  auto pool = std::make_shared<holoscan::BlockMemoryPool>(
      holoscan::MemoryStorageType::kDevice, kTensorBytes, pool_size);
  if (!pool->initialize_pool()) {
    return nullptr;
  }
  return pool;
}

[[nodiscard]] bool device_unavailable_for(benchmark::State& state,
                                          const holoscan::MemoryStorageType storage_type) {
  if (storage_type != holoscan::MemoryStorageType::kDevice) {
    return false;
  }
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
    return false;
  }
  state.SkipWithError("no CUDA device available for device Tensor benchmark");
  return true;
}

[[nodiscard]] bool cuda_unavailable_for_holomq_pool(benchmark::State& state) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    state.SkipWithError("no CUDA device available for HoloMQ Tensor pool benchmark");
    return true;
  }
  if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
    state.SkipWithError("failed to initialize CUDA for HoloMQ Tensor pool benchmark");
    return true;
  }
  return false;
}

[[nodiscard]] std::optional<holoscan::Tensor> make_sidecar_benchmark_tensor(
    std::array<std::byte, kTensorBytes>& storage) {
  const std::array<int64_t, 1U> shape{static_cast<int64_t>(storage.size())};
  auto tensor = holoscan::make_tensor(shape,
                                      DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
                                      std::nullopt,
                                      holoscan::MemoryStorageType::kSystem,
                                      storage.data());
  if (!tensor.has_value()) {
    return std::nullopt;
  }
  return std::move(tensor).value();
}

[[nodiscard]] bool register_sidecar_benchmark_recipients(holoscan::TensorSidecarStore& store,
                                                         const std::size_t recipient_count) {
  for (std::size_t index = 0U; index < recipient_count; ++index) {
    holomq::DeliveryRecipientId recipient_id{};
    recipient_id.bytes[0] = static_cast<std::uint8_t>(index + 1U);
    if (!store.register_recipient(recipient_id).has_value()) {
      return false;
    }
  }
  return true;
}

struct HoloMQTensorPoolFixture {
  HoloMQTensorPoolFixture(std::string socket_path_arg, holomq::Context context_arg,
                          holomq::BufferPool pool_arg, std::vector<holoscan::Tensor> tensors_arg)
      : socket_path(std::move(socket_path_arg)),
        context(std::move(context_arg)),
        pool(std::move(pool_arg)),
        tensors(std::move(tensors_arg)) {}

  ~HoloMQTensorPoolFixture() {
    holoscan::benchmarking::cleanup_holomq_benchmark_socket_path(socket_path);
  }

  HoloMQTensorPoolFixture(const HoloMQTensorPoolFixture&) = delete;
  HoloMQTensorPoolFixture& operator=(const HoloMQTensorPoolFixture&) = delete;

  std::string socket_path;
  holomq::Context context;
  holomq::BufferPool pool;
  std::vector<holoscan::Tensor> tensors;
};

[[nodiscard]] std::unique_ptr<HoloMQTensorPoolFixture> make_holomq_tensor_pool_fixture(
    const std::size_t pool_size = kTensorPoolSize,
    const holoscan::MemoryStorageType storage_type = holoscan::MemoryStorageType::kHost) {
  const std::string socket_path = holoscan::benchmarking::make_unique_holomq_benchmark_socket_path(
      "hsdk_channel_tensor_benchmark_");
  auto context =
      holomq::ContextBuilder().with_vmm_fd_export(true).with_fd_pass_socket(socket_path).create();
  if (!context.has_value()) {
    holoscan::benchmarking::cleanup_holomq_benchmark_socket_path(socket_path);
    return nullptr;
  }

  auto pool = [&]() {
    if (storage_type == holoscan::MemoryStorageType::kDevice) {
      return context->create_device_buffer_pool(kTensorBytes, pool_size);
    }
    if (storage_type == holoscan::MemoryStorageType::kSystem) {
      return context->create_host_buffer_pool(kTensorBytes, pool_size);
    }
    return context->create_host_pinned_buffer_pool(
        kTensorBytes, pool_size, CU_MEMHOSTREGISTER_PORTABLE);
  }();
  if (!pool.has_value()) {
    holoscan::benchmarking::cleanup_holomq_benchmark_socket_path(socket_path);
    return nullptr;
  }

  const std::array<int64_t, 1U> shape{static_cast<int64_t>(kTensorBytes)};
  const auto dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U};

  std::vector<holoscan::Tensor> tensors;
  tensors.reserve(pool_size);
  for (std::size_t i = 0U; i < pool_size; ++i) {
    auto buffer = pool->acquire();
    if (!buffer.has_value()) {
      holoscan::benchmarking::cleanup_holomq_benchmark_socket_path(socket_path);
      return nullptr;
    }
    auto tensor = holoscan::channel_internal::make_tensor_from_holomq_buffer(
        shape, dtype, std::nullopt, std::move(buffer).value());
    if (!tensor.has_value()) {
      holoscan::benchmarking::cleanup_holomq_benchmark_socket_path(socket_path);
      return nullptr;
    }
    tensors.emplace_back(std::move(tensor).value());
  }

  return std::make_unique<HoloMQTensorPoolFixture>(
      socket_path, std::move(*context), std::move(*pool), std::move(tensors));
}

[[nodiscard]] std::vector<holoscan::Tensor> make_tensor_pool(
    const std::uint32_t expected_recipients, const std::size_t pool_size = kTensorPoolSize,
    const holoscan::MemoryStorageType storage_type = holoscan::MemoryStorageType::kHost) {
  const std::array<int64_t, 1U> shape{static_cast<int64_t>(kTensorBytes)};
  const auto dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U};

  auto allocator = make_pool_allocator(storage_type, pool_size);
  if (!allocator) {
    return {};
  }

  std::vector<holoscan::Tensor> tensors;
  tensors.reserve(pool_size);
  for (std::size_t i = 0U; i < pool_size; ++i) {
    auto tensor = holoscan::make_tensor(
        shape,
        dtype,
        std::nullopt,
        storage_type,
        allocator,
        holoscan::MemoryBuffer::Options{.max_fanout_release_streams = expected_recipients});
    if (!tensor.has_value()) {
      return {};
    }
    tensors.emplace_back(std::move(tensor).value());
  }
  return tensors;
}

template <typename SubscriberContainer>
[[nodiscard]] bool wait_for_subscriber_count(holoscan::ManualTensorPublisher& publisher,
                                             SubscriberContainer& subscribers,
                                             std::uint32_t expected_recipients,
                                             const std::vector<holoscan::Tensor>& tensors);

template <typename SubscriberContainer>
[[nodiscard]] bool receive_from_all_subscribers_or_skip(benchmark::State& state,
                                                        SubscriberContainer& subscribers,
                                                        std::uint64_t& checksum);

[[nodiscard]] bool wait_for_catalog_tensor_subscriber(
    holoscan::ChannelPublisher<holoscan::Tensor>& publisher,
    holoscan::ChannelSubscriber<holoscan::Tensor>& subscriber) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  std::uint32_t consecutive_ready_refreshes = 0U;
  while (std::chrono::steady_clock::now() < deadline) {
    auto subscriber_refreshed = subscriber.has_samples();
    auto publisher_refreshed = publisher.update_connections();
    if (!subscriber_refreshed.has_value() || !publisher_refreshed.has_value()) {
      return false;
    }
    if (publisher.has_subscribers()) {
      ++consecutive_ready_refreshes;
      if (consecutive_ready_refreshes == 2U) {
        return true;
      }
    } else {
      consecutive_ready_refreshes = 0U;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return false;
}

enum class ChannelTensorMeasuredStage : std::uint8_t {
  kRoundTrip,
  kEmit,
  kReceive,
};

template <std::size_t SubscriberCount,
          ChannelTensorMeasuredStage MeasuredStage = ChannelTensorMeasuredStage::kRoundTrip>
void run_channel_tensor_holomq_pool_polling(
    benchmark::State& state, const char* const channel_suffix,
    const holoscan::ChannelTensorTransferMode transfer_mode =
        holoscan::ChannelTensorTransferMode::kSameProcessSidecar,
    const holoscan::MemoryStorageType source_storage_type = holoscan::MemoryStorageType::kHost,
    const bool enable_event_wake = false) {
  if (cuda_unavailable_for_holomq_pool(state)) {
    return;
  }

  auto tensor_pool = make_holomq_tensor_pool_fixture(kTensorPoolSize, source_storage_type);
  if (!tensor_pool || tensor_pool->tensors.empty()) {
    state.SkipWithError("failed to create HoloMQ Tensor benchmark pool");
    return;
  }

  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime.has_value()) {
    const std::string message =
        "failed to create ChannelRuntime: " + holoscan::runtime_error_message(runtime.error());
    state.SkipWithError(message.c_str());
    return;
  }

  const std::string channel_name = make_unique_channel_name(channel_suffix);
  holoscan::ChannelOptions options = make_channel_benchmark_options(
      holoscan::ChannelQueuePolicy::kBlocking, static_cast<std::uint32_t>(SubscriberCount));
  options.tensor_transfer_mode = transfer_mode;
  options.enable_event_wake = enable_event_wake;
  auto publisher = holoscan::channel_internal::TensorPublisherTestAccess::create(
      *runtime, channel_name, options);
  if (!publisher.has_value()) {
    const std::string message = "failed to create Tensor Channel publisher: " +
                                holoscan::runtime_error_message(publisher.error());
    state.SkipWithError(message.c_str());
    return;
  }

  std::vector<holoscan::ChannelSubscriber<holoscan::Tensor>> subscribers;
  subscribers.reserve(SubscriberCount);
  for (std::size_t i = 0U; i < SubscriberCount; ++i) {
    auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
    if (!subscriber.has_value()) {
      const std::string message = "failed to create Tensor Channel subscriber: " +
                                  holoscan::runtime_error_message(subscriber.error());
      state.SkipWithError(message.c_str());
      return;
    }
    subscribers.emplace_back(std::move(subscriber).value());
  }

  if (!wait_for_subscriber_count(publisher.value(),
                                 subscribers,
                                 static_cast<std::uint32_t>(SubscriberCount),
                                 tensor_pool->tensors)) {
    state.SkipWithError("timed out waiting for Tensor Channel subscribers");
    return;
  }

  std::uint64_t checksum = 0U;
  std::size_t tensor_index = 0U;
  for (auto _ : state) {
    std::chrono::steady_clock::time_point measured_start;  // NOLINT(misc-const-correctness)
    if constexpr (MeasuredStage == ChannelTensorMeasuredStage::kEmit) {
      measured_start = std::chrono::steady_clock::now();
    }
    auto emitted = publisher->emit(
        holoscan::Tensor(tensor_pool->tensors[tensor_index++ % tensor_pool->tensors.size()]));
    if (!emitted.has_value()) {
      const std::string message =
          "Tensor Channel emit failed: " + holoscan::runtime_error_message(emitted.error());
      state.SkipWithError(message.c_str());
      break;
    }
    if (emitted.value() < SubscriberCount) {
      state.SkipWithError("Tensor Channel emit did not reach all subscribers");
      break;
    }
    if constexpr (MeasuredStage == ChannelTensorMeasuredStage::kEmit) {
      const auto measured_end = std::chrono::steady_clock::now();
      state.SetIterationTime(std::chrono::duration<double>(measured_end - measured_start).count());
    } else if constexpr (MeasuredStage == ChannelTensorMeasuredStage::kReceive) {
      measured_start = std::chrono::steady_clock::now();
    }

    const bool received = receive_from_all_subscribers_or_skip(state, subscribers, checksum);
    if constexpr (MeasuredStage == ChannelTensorMeasuredStage::kReceive) {
      const auto measured_end = std::chrono::steady_clock::now();
      state.SetIterationTime(std::chrono::duration<double>(measured_end - measured_start).count());
    }
    if (!received) {
      break;
    }
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(SubscriberCount));
  state.counters["subscribers"] = static_cast<double>(SubscriberCount);
  state.counters["tensor_bytes"] = static_cast<double>(kTensorBytes);
  state.counters["holomq_pool_backed"] = 1.0;
  state.counters["buffer_handle_mode"] =
      static_cast<double>(transfer_mode == holoscan::ChannelTensorTransferMode::kBufferHandle);
  state.counters["device_storage"] =
      static_cast<double>(source_storage_type == holoscan::MemoryStorageType::kDevice);
  state.counters["event_wake"] = static_cast<double>(enable_event_wake);
  state.counters["emit_only"] =
      static_cast<double>(MeasuredStage == ChannelTensorMeasuredStage::kEmit);
  state.counters["receive_only"] =
      static_cast<double>(MeasuredStage == ChannelTensorMeasuredStage::kReceive);
}

[[nodiscard]] bool emit_tensor_or_skip(benchmark::State& state,
                                       holoscan::ManualTensorPublisher& publisher,
                                       const holoscan::Tensor& tensor,
                                       const char* const operation) {
  auto emitted = publisher.emit(holoscan::Tensor(tensor));
  if (!emitted.has_value()) {
    const std::string message =
        std::string(operation) + " failed: " + holoscan::runtime_error_message(emitted.error());
    state.SkipWithError(message.c_str());
    return false;
  }
  if (emitted.value() < 1U) {
    const std::string message = std::string(operation) + " did not reach the subscriber";
    state.SkipWithError(message.c_str());
    return false;
  }
  benchmark::DoNotOptimize(emitted.value());
  return true;
}

template <typename SubscriberContainer>
void drain_subscribers(SubscriberContainer& subscribers, std::uint64_t& checksum) {
  for (auto& subscriber : subscribers) {
    while (true) {
      auto received = subscriber.receive();
      if (!received.has_value()) {
        break;
      }
      const holoscan::Tensor tensor = std::move(received).value();
      checksum ^= reinterpret_cast<std::uintptr_t>(tensor.data());
      checksum ^= static_cast<std::uint64_t>(tensor.nbytes());
    }
  }
}

[[nodiscard]] bool drain_expected_samples_or_skip(
    benchmark::State& state, holoscan::ChannelSubscriber<holoscan::Tensor>& subscriber,
    const std::size_t expected_samples, std::uint64_t& checksum) {
  for (std::size_t i = 0U; i < expected_samples; ++i) {
    auto received = subscriber.receive();
    if (!received.has_value()) {
      const std::string message = "Tensor Channel drain failed after prefill: " +
                                  holoscan::runtime_error_message(received.error());
      state.SkipWithError(message.c_str());
      return false;
    }
    const holoscan::Tensor tensor = std::move(received).value();
    checksum ^= reinterpret_cast<std::uintptr_t>(tensor.data());
    checksum ^= static_cast<std::uint64_t>(tensor.nbytes());
  }
  return true;
}

template <typename SubscriberContainer>
[[nodiscard]] bool receive_from_all_subscribers_or_skip(benchmark::State& state,
                                                        SubscriberContainer& subscribers,
                                                        std::uint64_t& checksum) {
  for (auto& subscriber : subscribers) {
    std::optional<holoscan::Tensor> tensor;
    while (!tensor.has_value()) {
      auto received = subscriber.receive();
      if (received.has_value()) {
        tensor = std::move(received).value();
      } else if (received.error().code != holoscan::ErrorCode::kNotReady) {
        const std::string message =
            "Tensor Channel receive failed: " + holoscan::runtime_error_message(received.error());
        state.SkipWithError(message.c_str());
        return false;
      }
    }
    checksum ^= reinterpret_cast<std::uintptr_t>(tensor->data());
    checksum ^= static_cast<std::uint64_t>(tensor->nbytes());
  }
  return true;
}

template <typename SubscriberContainer>
[[nodiscard]] bool wait_for_subscriber_count(holoscan::ManualTensorPublisher& publisher,
                                             SubscriberContainer& subscribers,
                                             const std::uint32_t expected_recipients,
                                             const std::vector<holoscan::Tensor>& tensors) {
  std::uint64_t ignored_checksum = 0U;
  std::size_t index = 0U;
  return wait_for_channel_subscribers(
      publisher,
      expected_recipients,
      [&publisher, &tensors, &index]() {
        return publisher.emit(holoscan::Tensor(tensors[index++ % tensors.size()]));
      },
      [&subscribers, &ignored_checksum]() { drain_subscribers(subscribers, ignored_checksum); });
}

using TensorTransportPayload = holoscan::channel_internal::TensorTransportPayload;

struct LocalTensorTransportPayload {
  holoscan::SampleMetadata metadata{};
  holoscan::ChannelSampleDescriptor descriptor{};
};

static_assert(std::is_standard_layout_v<LocalTensorTransportPayload>);
static_assert(std::is_trivially_copyable_v<LocalTensorTransportPayload>);

void set_local_descriptor(TensorTransportPayload& payload,
                          const holoscan::ChannelSampleDescriptor& descriptor) {
  payload.metadata = holoscan::sample_metadata_from_descriptor(descriptor).value();
  payload.producer_generation = kBenchmarkProducerGeneration;
  payload.operation_kind = holoscan::channel_internal::TensorTransportOperationKind::kData;
  payload.payload_kind = holoscan::ChannelPayloadKind::kLocalBufferDescriptor;
  payload.reference.local = descriptor;
}

void set_local_descriptor(LocalTensorTransportPayload& payload,
                          const holoscan::ChannelSampleDescriptor& descriptor) {
  payload.metadata = holoscan::sample_metadata_from_descriptor(descriptor).value();
  payload.descriptor = descriptor;
}

[[nodiscard]] const holoscan::ChannelSampleDescriptor& local_descriptor(
    const TensorTransportPayload& payload) {
  return payload.reference.local;
}

[[nodiscard]] const holoscan::ChannelSampleDescriptor& local_descriptor(
    const LocalTensorTransportPayload& payload) {
  return payload.descriptor;
}

template <typename Payload>
[[nodiscard]] std::optional<holomq::InboundMessage<Payload>> wait_for_holomq_sample(
    holomq::Subscriber<Payload>& subscriber) {
  constexpr auto kTimeout = std::chrono::seconds(3);
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto sample = subscriber.try_take();
    if (sample.has_value()) {
      return std::move(sample).value();
    }
    std::this_thread::yield();
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<holomq::InboundMessage<TensorTransportPayload>>
wait_for_holomq_tensor_sample(holomq::Subscriber<TensorTransportPayload>& subscriber) {
  return wait_for_holomq_sample(subscriber);
}

template <typename Payload, LocalPayloadWork Work = LocalPayloadWork::kSidecar>
void run_tensor_local_payload_round_trip(  // NOLINT(readability-function-cognitive-complexity)
    benchmark::State& state, const char* const suffix) {
  std::array<std::byte, kTensorBytes> storage{};
  auto tensor = make_sidecar_benchmark_tensor(storage);
  if (!tensor.has_value()) {
    state.SkipWithError("failed to create local-payload Tensor");
    return;
  }

  auto context = holomq::ContextBuilder().with_vmm_fd_export(false).create();
  if (!context.has_value()) {
    state.SkipWithError("failed to create local-payload HoloMQ context");
    return;
  }

  const auto service_options = make_holomq_service_options(holomq::QueuePolicy::kBlocking, 1U, 1U);
  holomq::PublisherPubSubOptions publisher_options{};
  publisher_options.enable_sample_release_callbacks = Work != LocalPayloadWork::kTransportOnly;
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = 1U;

  const std::string service_name = make_unique_channel_name(suffix);
  auto publisher = context->publisher_builder(service_name)
                       .service_options(service_options)
                       .enable_event_wake(false)
                       .publisher_options(publisher_options)
                       .payload_type<Payload>()
                       .create();
  if (!publisher.has_value()) {
    state.SkipWithError("failed to create local-payload publisher");
    return;
  }
  auto subscriber = context->subscriber_builder(service_name)
                        .service_options(service_options)
                        .enable_event_wake(false)
                        .subscriber_options(subscriber_options)
                        .payload_type<Payload>()
                        .create();
  if (!subscriber.has_value()) {
    state.SkipWithError("failed to create local-payload subscriber");
    return;
  }

  const auto store = std::make_shared<holoscan::TensorSidecarStore>(4U, 1U);
  if (!store->register_recipient(subscriber->recipient_id()).has_value()) {
    state.SkipWithError("failed to register local-payload recipient");
    return;
  }

  constexpr auto kConnectionTimeout = std::chrono::seconds(3);
  const auto connection_deadline = std::chrono::steady_clock::now() + kConnectionTimeout;
  bool connected = false;
  while (std::chrono::steady_clock::now() < connection_deadline) {
    static_cast<void>(publisher->update_connections());
    auto probe = publisher->loan();
    if (!probe.has_value()) {
      std::this_thread::yield();
      continue;
    }
    auto sent = publisher->send(std::move(probe).value());
    if (sent.has_value() && sent.value() == 1) {
      connected = wait_for_holomq_sample(subscriber.value()).has_value();
      break;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  if (!connected) {
    state.SkipWithError("timed out waiting for local-payload subscriber");
    return;
  }

  std::uint64_t checksum = 0U;
  std::uint64_t sequence = 0U;
  const auto callback_count = std::make_shared<std::atomic<std::uint64_t>>(0U);
  for (auto _ : state) {
    auto message = publisher->loan();
    if (!message.has_value()) {
      state.SkipWithError("failed to loan local-payload sample");
      break;
    }
    holoscan::ChannelSampleDescriptor descriptor{};
    if constexpr (Work == LocalPayloadWork::kSidecar) {
      auto inserted = store->insert(holoscan::Tensor(tensor.value()),
                                    holoscan::SampleMetadata{.sequence = ++sequence});
      if (!inserted.has_value()) {
        state.SkipWithError("failed to insert local-payload sidecar record");
        break;
      }
      descriptor = inserted.value();
    } else {
      descriptor.sidecar_token = 1U;
      descriptor.expected_recipient_count = 1U;
      descriptor.sequence = ++sequence;
      descriptor.byte_size = kTensorBytes;
    }
    Payload payload{};
    set_local_descriptor(payload, descriptor);
    message->payload() = payload;
    auto sent = [&]() {
      if constexpr (Work == LocalPayloadWork::kTransportOnly) {
        return publisher->send(std::move(message).value());
      } else if constexpr (Work == LocalPayloadWork::kCallbackOnly) {
        return publisher->send(std::move(message).value(), [callback_count]() noexcept {
          callback_count->fetch_add(1U, std::memory_order_relaxed);
        });
      } else {
        return publisher->send(std::move(message).value(),
                               [store, descriptor]() noexcept { store->release_all(descriptor); });
      }
    }();
    if (!sent.has_value() || sent.value() != 1) {
      state.SkipWithError("failed to send local-payload sample");
      break;
    }
    auto inbound = wait_for_holomq_sample(subscriber.value());
    if (!inbound.has_value()) {
      state.SkipWithError("failed to receive local-payload sample");
      break;
    }
    const Payload received_payload = inbound->payload();
    if constexpr (Work == LocalPayloadWork::kSidecar) {
      auto received = store->acquire_sample(local_descriptor(received_payload), 0U);
      if (!received.has_value()) {
        state.SkipWithError("failed to acquire local-payload Tensor");
        break;
      }
      checksum ^= reinterpret_cast<std::uintptr_t>(received->data.data());
      checksum ^= received->metadata.sequence;
    } else {
      checksum ^= local_descriptor(received_payload).sequence;
      checksum ^= received_payload.metadata.sequence;
    }
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(state.iterations());
  state.counters["payload_bytes"] = static_cast<double>(sizeof(Payload));
  state.counters["release_callbacks"] =
      static_cast<double>(callback_count->load(std::memory_order_relaxed));
  state.counters["sidecar"] = static_cast<double>(Work == LocalPayloadWork::kSidecar);
  state.counters["tensor_bytes"] = static_cast<double>(kTensorBytes);
}

template <typename Payload, LocalTransportStage Stage>
void run_tensor_local_transport_stage(  // NOLINT(readability-function-cognitive-complexity)
    benchmark::State& state, const char* const suffix) {
  const auto batch_size = static_cast<std::uint32_t>(state.range(0));
  auto context = holomq::ContextBuilder().with_vmm_fd_export(false).create();
  if (!context.has_value()) {
    state.SkipWithError("failed to create local-transport-stage HoloMQ context");
    return;
  }

  const auto service_options =
      make_holomq_service_options(holomq::QueuePolicy::kBlocking, 1U, batch_size);
  holomq::PublisherPubSubOptions publisher_options{};
  publisher_options.enable_sample_release_callbacks = false;
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = batch_size;

  const std::string service_name = make_unique_channel_name(suffix);
  auto publisher = context->publisher_builder(service_name)
                       .service_options(service_options)
                       .enable_event_wake(false)
                       .publisher_options(publisher_options)
                       .payload_type<Payload>()
                       .create();
  auto subscriber = context->subscriber_builder(service_name)
                        .service_options(service_options)
                        .enable_event_wake(false)
                        .subscriber_options(subscriber_options)
                        .payload_type<Payload>()
                        .create();
  if (!publisher.has_value() || !subscriber.has_value()) {
    state.SkipWithError("failed to create local-transport-stage endpoints");
    return;
  }

  constexpr auto kConnectionTimeout = std::chrono::seconds(3);
  const auto connection_deadline = std::chrono::steady_clock::now() + kConnectionTimeout;
  bool connected = false;
  while (std::chrono::steady_clock::now() < connection_deadline) {
    static_cast<void>(publisher->update_connections());
    auto probe = publisher->loan();
    if (!probe.has_value()) {
      std::this_thread::yield();
      continue;
    }
    auto sent = publisher->send(std::move(probe).value());
    if (sent.has_value() && sent.value() == 1) {
      connected = wait_for_holomq_sample(subscriber.value()).has_value();
      break;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  if (!connected) {
    state.SkipWithError("timed out waiting for local-transport-stage subscriber");
    return;
  }

  std::uint64_t checksum = 0U;
  std::uint64_t sequence = 0U;
  std::int64_t completed_batches = 0;
  auto publish_batch = [&]() {
    for (std::uint32_t index = 0U; index < batch_size; ++index) {
      auto message = publisher->loan();
      if (!message.has_value()) {
        return false;
      }
      holoscan::ChannelSampleDescriptor descriptor{};
      descriptor.sidecar_token = 1U;
      descriptor.expected_recipient_count = 1U;
      descriptor.sequence = ++sequence;
      descriptor.byte_size = kTensorBytes;
      Payload payload{};
      set_local_descriptor(payload, descriptor);
      message->payload() = payload;
      auto sent = publisher->send(std::move(message).value());
      if (!sent.has_value() || sent.value() != 1) {
        return false;
      }
    }
    return true;
  };
  auto take_batch = [&]() {
    for (std::uint32_t index = 0U; index < batch_size; ++index) {
      auto inbound = subscriber->try_take();
      if (!inbound.has_value()) {
        return false;
      }
      const Payload received_payload = inbound->payload();
      checksum ^= local_descriptor(received_payload).sequence;
      checksum ^= received_payload.metadata.sequence;
    }
    return true;
  };

  for (auto _ : state) {
    bool succeeded = false;
    if constexpr (Stage == LocalTransportStage::kPublish) {
      const auto measured_start = std::chrono::steady_clock::now();
      succeeded = publish_batch();
      const auto measured_end = std::chrono::steady_clock::now();
      state.SetIterationTime(std::chrono::duration<double>(measured_end - measured_start).count());
      if (succeeded) {
        for (std::uint32_t index = 0U; index < batch_size; ++index) {
          auto inbound = wait_for_holomq_sample(subscriber.value());
          if (!inbound.has_value()) {
            succeeded = false;
            break;
          }
          const Payload received_payload = inbound->payload();
          checksum ^= local_descriptor(received_payload).sequence;
        }
      }
    } else {
      if (!publish_batch()) {
        state.SkipWithError("failed to prefill local-transport-stage batch");
        break;
      }
      const auto measured_start = std::chrono::steady_clock::now();
      succeeded = take_batch();
      const auto measured_end = std::chrono::steady_clock::now();
      state.SetIterationTime(std::chrono::duration<double>(measured_end - measured_start).count());
    }
    if (!succeeded) {
      state.SkipWithError("failed to process local-transport-stage batch");
      break;
    }
    ++completed_batches;
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(completed_batches * batch_size);
  state.counters["batch_size"] = static_cast<double>(batch_size);
  state.counters["payload_bytes"] = static_cast<double>(sizeof(Payload));
  state.counters["publish_stage"] = static_cast<double>(Stage == LocalTransportStage::kPublish);
}

template <std::size_t SubscriberCount, ReplacementCleanupMode CleanupMode>
void run_channel_tensor_drop_oldest_overwrite(  // NOLINT(readability-function-cognitive-complexity)
    benchmark::State& state) {
  std::array<std::byte, kTensorBytes> storage{};
  auto tensor = make_sidecar_benchmark_tensor(storage);
  if (!tensor.has_value()) {
    state.SkipWithError("failed to create drop-oldest-overwrite Tensor");
    return;
  }

  auto context = holomq::ContextBuilder().with_vmm_fd_export(false).create();
  if (!context.has_value()) {
    state.SkipWithError("failed to create drop-oldest-overwrite HoloMQ context");
    return;
  }

  constexpr bool kUsesReplacement = CleanupMode != ReplacementCleanupMode::kBlockingBaseline;
  const holomq::QueuePolicy queue_policy =
      kUsesReplacement ? holomq::QueuePolicy::kDropOldest : holomq::QueuePolicy::kBlocking;
  const auto service_options = make_holomq_service_options(queue_policy, SubscriberCount, 1U);
  const holomq::PublisherPubSubOptions publisher_options{};
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = 1U;

  const auto store = std::make_shared<holoscan::TensorSidecarStore>(
      /*capacity=*/4U, static_cast<std::uint32_t>(SubscriberCount));
  std::atomic<std::uint64_t> callback_replacements{0U};  // NOLINT(misc-const-correctness)
  const std::string service_name = make_unique_channel_name(
      kUsesReplacement ? "tensor_drop_oldest_overwrite" : "tensor_blocking_no_replacement");
  auto publisher_builder = context->publisher_builder(service_name)
                               .service_options(service_options)
                               .enable_event_wake(false)
                               .publisher_options(publisher_options)
                               .payload_type<TensorTransportPayload>();
  if constexpr (CleanupMode == ReplacementCleanupMode::kTransportCallback) {
    publisher_builder =
        std::move(publisher_builder)
            .on_sample_replaced([store, &callback_replacements](
                                    const TensorTransportPayload& payload,
                                    const holomq::SampleReplacedInfo& info) noexcept {
              if (payload.payload_kind == holoscan::ChannelPayloadKind::kLocalBufferDescriptor) {
                store->release_replaced(payload.reference.local, info.receiver_id);
                callback_replacements.fetch_add(1U, std::memory_order_relaxed);
              }
            });
  }
  auto publisher = std::move(publisher_builder).create();
  if (!publisher.has_value()) {
    state.SkipWithError("failed to create drop-oldest-overwrite HoloMQ publisher");
    return;
  }

  std::vector<holomq::Subscriber<TensorTransportPayload>> subscribers;
  subscribers.reserve(SubscriberCount);
  std::array<holomq::DeliveryRecipientId, SubscriberCount> recipient_ids{};
  for (std::size_t index = 0U; index < SubscriberCount; ++index) {
    auto subscriber = context->subscriber_builder(service_name)
                          .service_options(service_options)
                          .enable_event_wake(false)
                          .subscriber_options(subscriber_options)
                          .payload_type<TensorTransportPayload>()
                          .create();
    if (!subscriber.has_value()) {
      state.SkipWithError("failed to create drop-oldest-overwrite HoloMQ subscriber");
      return;
    }
    recipient_ids[index] = subscriber->recipient_id();
    if (!store->register_recipient(recipient_ids[index]).has_value()) {
      state.SkipWithError("failed to register drop-oldest-overwrite recipient");
      return;
    }
    subscribers.emplace_back(std::move(subscriber).value());
  }

  constexpr auto kConnectionTimeout = std::chrono::seconds(3);
  const auto connection_deadline = std::chrono::steady_clock::now() + kConnectionTimeout;
  bool connected = false;
  while (std::chrono::steady_clock::now() < connection_deadline) {
    static_cast<void>(publisher->update_connections());
    auto probe = publisher->loan();
    if (!probe.has_value()) {
      std::this_thread::yield();
      continue;
    }
    probe->payload().payload_kind = holoscan::ChannelPayloadKind::kBufferHandle;
    auto sent = publisher->send(std::move(probe).value());
    if (sent.has_value() && sent.value() == static_cast<int>(SubscriberCount)) {
      connected = true;
      break;
    }
    for (auto& subscriber : subscribers) {
      while (subscriber.try_take().has_value()) {
      }
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  if (!connected) {
    state.SkipWithError("timed out waiting for drop-oldest-overwrite subscribers");
    return;
  }
  for (auto& subscriber : subscribers) {
    if (!wait_for_holomq_tensor_sample(subscriber).has_value()) {
      state.SkipWithError("failed to drain drop-oldest-overwrite connection probe");
      return;
    }
  }

  struct PublishedDescriptor {
    holoscan::ChannelSampleDescriptor descriptor{};
    int recipient_count{};
  };
  auto publish_tensor = [&](const std::uint64_t sequence) -> std::optional<PublishedDescriptor> {
    auto message = publisher->loan();
    if (!message.has_value()) {
      state.SkipWithError("failed to loan drop-oldest-overwrite transport sample");
      return std::nullopt;
    }
    auto descriptor = store->insert(holoscan::Tensor(tensor.value()),
                                    holoscan::SampleMetadata{.sequence = sequence});
    if (!descriptor.has_value()) {
      state.SkipWithError("failed to insert drop-oldest-overwrite sidecar record");
      return std::nullopt;
    }

    TensorTransportPayload payload{};
    payload.metadata =
        holoscan::SampleMetadata{.sequence = descriptor->sequence,
                                 .byte_size = descriptor->byte_size,
                                 .payload_kind = holoscan::PayloadKind::kDescriptorBacked};
    payload.producer_generation = kBenchmarkProducerGeneration;
    payload.operation_kind = holoscan::channel_internal::TensorTransportOperationKind::kData;
    payload.payload_kind = holoscan::ChannelPayloadKind::kLocalBufferDescriptor;
    payload.reference.local = descriptor.value();
    message->payload() = payload;
    auto sent = publisher->send(std::move(message).value(),
                                [store, published_descriptor = descriptor.value()]() noexcept {
                                  store->release_all(published_descriptor);
                                });
    if (!sent.has_value()) {
      state.SkipWithError("failed to send drop-oldest-overwrite transport sample");
      return std::nullopt;
    }
    return PublishedDescriptor{.descriptor = descriptor.value(), .recipient_count = sent.value()};
  };

  std::uint64_t checksum = 0U;
  auto receive_all = [&]() {
    for (std::size_t index = 0U; index < SubscriberCount; ++index) {
      auto message = wait_for_holomq_tensor_sample(subscribers[index]);
      if (!message.has_value() ||
          message->payload().payload_kind != holoscan::ChannelPayloadKind::kLocalBufferDescriptor) {
        state.SkipWithError("failed to receive drop-oldest-overwrite sidecar descriptor");
        return false;
      }
      auto received = store->acquire(message->payload().reference.local, index);
      if (!received.has_value()) {
        state.SkipWithError("failed to acquire drop-oldest-overwrite sidecar Tensor");
        return false;
      }
      checksum ^= reinterpret_cast<std::uintptr_t>(received->data());
      checksum ^= message->payload().metadata.sequence;
    }
    return true;
  };

  std::array<holoscan::ChannelSampleDescriptor, SubscriberCount> replacement_mirror{};
  std::uint64_t mirror_replacements = 0U;
  auto update_replacement_mirror = [&](const holoscan::ChannelSampleDescriptor& descriptor) {
    for (std::size_t index = 0U; index < SubscriberCount; ++index) {
      if (replacement_mirror[index].sidecar_token != 0U) {
        store->release_replaced(replacement_mirror[index], recipient_ids[index]);
        ++mirror_replacements;
      }
      replacement_mirror[index] = descriptor;
    }
  };

  std::uint64_t sequence = 0U;
  if constexpr (kUsesReplacement) {
    auto initial = publish_tensor(++sequence);
    if (!initial.has_value() || initial->recipient_count != static_cast<int>(SubscriberCount)) {
      state.SkipWithError("failed to prime drop-oldest-overwrite sample");
      return;
    }
    if constexpr (CleanupMode == ReplacementCleanupMode::kPublisherMirror) {
      update_replacement_mirror(initial->descriptor);
    }
  }

  std::int64_t completed_iterations = 0;
  for (auto _ : state) {
    const auto measured_start = std::chrono::steady_clock::now();
    auto published = publish_tensor(++sequence);
    if (published.has_value() && published->recipient_count == static_cast<int>(SubscriberCount)) {
      if constexpr (CleanupMode == ReplacementCleanupMode::kPublisherMirror) {
        update_replacement_mirror(published->descriptor);
      }
    }
    const auto measured_end = std::chrono::steady_clock::now();
    if (!published.has_value() || published->recipient_count != static_cast<int>(SubscriberCount)) {
      state.SkipWithError("drop-oldest-overwrite publish did not reach every subscriber");
      break;
    }
    state.SetIterationTime(std::chrono::duration<double>(measured_end - measured_start).count());
    ++completed_iterations;

    if constexpr (CleanupMode == ReplacementCleanupMode::kBlockingBaseline) {
      if (!receive_all()) {
        break;
      }
    }
  }

  if constexpr (kUsesReplacement) {
    if (!receive_all()) {
      return;
    }
  }

  const std::uint64_t expected_replacements =
      static_cast<std::uint64_t>(completed_iterations) * SubscriberCount;
  if constexpr (CleanupMode == ReplacementCleanupMode::kTransportCallback) {
    if (callback_replacements.load(std::memory_order_relaxed) != expected_replacements) {
      state.SkipWithError("drop-oldest-overwrite callback count mismatch");
      return;
    }
  } else if constexpr (CleanupMode == ReplacementCleanupMode::kPublisherMirror) {
    if (mirror_replacements != expected_replacements) {
      state.SkipWithError("drop-oldest-overwrite mirror count mismatch");
      return;
    }
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(completed_iterations);
  state.counters["subscribers"] = static_cast<double>(SubscriberCount);
  state.counters["tensor_bytes"] = static_cast<double>(kTensorBytes);
  state.counters["replacement_callbacks"] =
      static_cast<double>(callback_replacements.load(std::memory_order_relaxed));
  state.counters["mirror_replacements"] = static_cast<double>(mirror_replacements);
}

template <std::size_t SubscriberCount,
          ChannelTensorMeasuredStage MeasuredStage = ChannelTensorMeasuredStage::kRoundTrip>
void run_channel_tensor_polling(
    benchmark::State& state, const holoscan::ChannelQueuePolicy policy,
    const char* const channel_suffix,
    const holoscan::MemoryStorageType storage_type = holoscan::MemoryStorageType::kHost,
    const std::uint32_t queue_depth = 64U) {
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime.has_value()) {
    const std::string message =
        "failed to create ChannelRuntime: " + holoscan::runtime_error_message(runtime.error());
    state.SkipWithError(message.c_str());
    return;
  }

  const std::string channel_name = make_unique_channel_name(channel_suffix);
  const holoscan::ChannelOptions options = make_channel_benchmark_options(
      policy, static_cast<std::uint32_t>(SubscriberCount), queue_depth);
  auto publisher = holoscan::channel_internal::TensorPublisherTestAccess::create(
      *runtime, channel_name, options);
  if (!publisher.has_value()) {
    const std::string message = "failed to create Tensor Channel publisher: " +
                                holoscan::runtime_error_message(publisher.error());
    state.SkipWithError(message.c_str());
    return;
  }

  std::vector<holoscan::ChannelSubscriber<holoscan::Tensor>> subscribers;
  subscribers.reserve(SubscriberCount);
  for (std::size_t i = 0U; i < SubscriberCount; ++i) {
    auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
    if (!subscriber.has_value()) {
      const std::string message = "failed to create Tensor Channel subscriber: " +
                                  holoscan::runtime_error_message(subscriber.error());
      state.SkipWithError(message.c_str());
      return;
    }
    subscribers.emplace_back(std::move(subscriber).value());
  }

  const std::vector<holoscan::Tensor> tensors =
      make_tensor_pool(static_cast<std::uint32_t>(SubscriberCount), kTensorPoolSize, storage_type);
  if (tensors.empty()) {
    state.SkipWithError("failed to allocate benchmark Tensor pool");
    return;
  }

  if (!wait_for_subscriber_count(
          publisher.value(), subscribers, static_cast<std::uint32_t>(SubscriberCount), tensors)) {
    state.SkipWithError("timed out waiting for Tensor Channel subscribers");
    return;
  }

  std::uint64_t checksum = 0U;
  std::size_t tensor_index = 0U;
  for (auto _ : state) {
    std::chrono::steady_clock::time_point measured_start;  // NOLINT(misc-const-correctness)
    if constexpr (MeasuredStage == ChannelTensorMeasuredStage::kEmit) {
      measured_start = std::chrono::steady_clock::now();
    }
    auto emitted = publisher->emit(holoscan::Tensor(tensors[tensor_index++ % tensors.size()]));
    if (!emitted.has_value()) {
      const std::string message =
          "Tensor Channel emit failed: " + holoscan::runtime_error_message(emitted.error());
      state.SkipWithError(message.c_str());
      break;
    }
    if (emitted.value() < SubscriberCount) {
      state.SkipWithError("Tensor Channel emit did not reach all subscribers");
      break;
    }

    if constexpr (MeasuredStage == ChannelTensorMeasuredStage::kEmit) {
      const auto measured_end = std::chrono::steady_clock::now();
      state.SetIterationTime(std::chrono::duration<double>(measured_end - measured_start).count());
    } else if constexpr (MeasuredStage == ChannelTensorMeasuredStage::kReceive) {
      measured_start = std::chrono::steady_clock::now();
    }

    const bool received = receive_from_all_subscribers_or_skip(state, subscribers, checksum);
    if constexpr (MeasuredStage == ChannelTensorMeasuredStage::kReceive) {
      const auto measured_end = std::chrono::steady_clock::now();
      state.SetIterationTime(std::chrono::duration<double>(measured_end - measured_start).count());
    }
    if (!received) {
      break;
    }
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(SubscriberCount));
  state.counters["subscribers"] = static_cast<double>(SubscriberCount);
  state.counters["tensor_bytes"] = static_cast<double>(kTensorBytes);
  state.counters["device_storage"] =
      static_cast<double>(storage_type == holoscan::MemoryStorageType::kDevice);
  state.counters["emit_only"] =
      static_cast<double>(MeasuredStage == ChannelTensorMeasuredStage::kEmit);
  state.counters["receive_only"] =
      static_cast<double>(MeasuredStage == ChannelTensorMeasuredStage::kReceive);
}

template <std::size_t SubscriberCount>
void BM_TensorSidecarStore_InsertAcquire(benchmark::State& state) {
  std::array<std::byte, kTensorBytes> storage{};
  auto tensor = make_sidecar_benchmark_tensor(storage);
  if (!tensor.has_value()) {
    state.SkipWithError("failed to create sidecar benchmark Tensor");
    return;
  }

  holoscan::TensorSidecarStore store(/*capacity=*/1U,
                                     /*expected_recipient_count=*/SubscriberCount);
  if (!register_sidecar_benchmark_recipients(store, SubscriberCount)) {
    state.SkipWithError("failed to register sidecar benchmark recipients");
    return;
  }

  std::uint64_t checksum = 0U;
  for (auto _ : state) {
    auto descriptor = store.insert(holoscan::Tensor(tensor.value()));
    if (!descriptor.has_value()) {
      state.SkipWithError("failed to insert sidecar benchmark Tensor");
      break;
    }
    for (std::size_t index = 0U; index < SubscriberCount; ++index) {
      auto received = store.acquire(descriptor.value(), index);
      if (!received.has_value()) {
        state.SkipWithError("failed to acquire sidecar benchmark Tensor");
        return;
      }
      checksum ^= reinterpret_cast<std::uintptr_t>(received->data());
    }
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(SubscriberCount));
  state.counters["subscribers"] = static_cast<double>(SubscriberCount);
  state.counters["tensor_bytes"] = static_cast<double>(kTensorBytes);
}

template <holoscan::MemoryStorageType StorageType, BufferHandleEmitStage Stage>
void run_buffer_handle_emit_stage(benchmark::State& state) {
  if (cuda_unavailable_for_holomq_pool(state)) {
    return;
  }

  const auto batch_size = static_cast<std::size_t>(state.range(0));
  auto fixture = make_holomq_tensor_pool_fixture(kTensorPoolSize, StorageType);
  if (!fixture || fixture->tensors.empty()) {
    state.SkipWithError("failed to create HoloMQ Tensor benchmark pool");
    return;
  }
  const auto& tensor = fixture->tensors.front();
  auto source_descriptor = holoscan::channel_internal::export_holomq_tensor_descriptor(tensor);
  if (!source_descriptor.has_value()) {
    state.SkipWithError("failed to export source HoloMQ Tensor descriptor");
    return;
  }

  std::vector<holoscan::channel_internal::HoloMQTensorDescriptor> descriptors;
  descriptors.reserve(batch_size);
  std::uint64_t checksum = 0U;
  std::int64_t completed_batches = 0;
  for (auto _ : state) {
    descriptors.clear();
    bool succeeded = true;
    const auto measured_start = std::chrono::steady_clock::now();
    for (std::size_t index = 0U; index < batch_size; ++index) {
      auto descriptor = [&]() {
        if constexpr (Stage == BufferHandleEmitStage::kExportDescriptor) {
          return holoscan::channel_internal::export_holomq_tensor_descriptor(tensor);
        } else {
          return holoscan::channel_internal::make_holomq_tensor_descriptor(
              tensor, source_descriptor->buffer_handle);
        }
      }();
      if (!descriptor.has_value()) {
        succeeded = false;
        break;
      }
      descriptors.emplace_back(std::move(descriptor).value());
    }
    const auto measured_end = std::chrono::steady_clock::now();
    if (!succeeded) {
      state.SkipWithError("failed to process HoloMQ BufferHandle emit stage batch");
      break;
    }
    state.SetIterationTime(std::chrono::duration<double>(measured_end - measured_start).count());
    for (const auto& descriptor : descriptors) {
      checksum ^= descriptor.byte_size;
      checksum ^= descriptor.buffer_handle ? 1U : 0U;
    }
    ++completed_batches;
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(completed_batches * static_cast<std::int64_t>(batch_size));
  state.counters["batch_size"] = static_cast<double>(batch_size);
  state.counters["construct_descriptor_only"] =
      static_cast<double>(Stage == BufferHandleEmitStage::kConstructDescriptor);
  state.counters["device_storage"] =
      static_cast<double>(StorageType == holoscan::MemoryStorageType::kDevice);
  state.counters["export_descriptor"] =
      static_cast<double>(Stage == BufferHandleEmitStage::kExportDescriptor);
  state.counters["tensor_bytes"] = static_cast<double>(kTensorBytes);
}

void BM_ChannelTensor_HoloMQBufferHandleExportDescriptor(benchmark::State& state) {
  run_buffer_handle_emit_stage<holoscan::MemoryStorageType::kHost,
                               BufferHandleEmitStage::kExportDescriptor>(state);
}

void BM_ChannelTensor_HoloMQBufferHandleConstructDescriptor(benchmark::State& state) {
  run_buffer_handle_emit_stage<holoscan::MemoryStorageType::kHost,
                               BufferHandleEmitStage::kConstructDescriptor>(state);
}

void BM_ChannelTensor_HoloMQDeviceBufferHandleExportDescriptor(benchmark::State& state) {
  run_buffer_handle_emit_stage<holoscan::MemoryStorageType::kDevice,
                               BufferHandleEmitStage::kExportDescriptor>(state);
}

void BM_ChannelTensor_HoloMQDeviceBufferHandleConstructDescriptor(benchmark::State& state) {
  run_buffer_handle_emit_stage<holoscan::MemoryStorageType::kDevice,
                               BufferHandleEmitStage::kConstructDescriptor>(state);
}

template <holoscan::MemoryStorageType StorageType, BufferHandleReceiveStage Stage>
void run_buffer_handle_receive_stage(  // NOLINT(readability-function-cognitive-complexity)
    benchmark::State& state) {
  if (cuda_unavailable_for_holomq_pool(state)) {
    return;
  }

  const auto batch_size = static_cast<std::size_t>(state.range(0));
  auto fixture = make_holomq_tensor_pool_fixture(kTensorPoolSize, StorageType);
  if (!fixture || fixture->tensors.empty()) {
    state.SkipWithError("failed to create HoloMQ Tensor benchmark pool");
    return;
  }

  auto descriptor =
      holoscan::channel_internal::export_holomq_tensor_descriptor(fixture->tensors.front());
  if (!descriptor.has_value()) {
    state.SkipWithError("failed to export HoloMQ Tensor descriptor");
    return;
  }
  auto warm_import = fixture->context.import_buffer_unsynchronized(descriptor->buffer_handle);
  if (!warm_import.has_value()) {
    state.SkipWithError("failed to warm HoloMQ buffer import cache");
    return;
  }
  // Measure the steady-state cache-hit path. The first import can establish CUDA VMM or pinned-host
  // mappings and is intentionally outside every timed interval.
  warm_import = holomq::ConstBuffer{};

  std::vector<holomq::ConstBuffer> imported_buffers;
  std::vector<holoscan::Tensor> materialized_tensors;
  imported_buffers.reserve(batch_size);
  materialized_tensors.reserve(batch_size);

  auto import_batch = [&]() {
    imported_buffers.clear();
    for (std::size_t index = 0U; index < batch_size; ++index) {
      auto imported = fixture->context.import_buffer_unsynchronized(descriptor->buffer_handle);
      if (!imported.has_value()) {
        return false;
      }
      imported_buffers.emplace_back(std::move(imported).value());
    }
    return true;
  };
  auto materialize_batch = [&]() {
    materialized_tensors.clear();
    for (auto& imported : imported_buffers) {
      auto tensor = holoscan::channel_internal::make_tensor_from_holomq_const_buffer(
          descriptor.value(), std::move(imported));
      if (!tensor.has_value()) {
        return false;
      }
      materialized_tensors.emplace_back(std::move(tensor).value());
    }
    return true;
  };

  std::uint64_t checksum = 0U;
  std::int64_t completed_batches = 0;
  for (auto _ : state) {
    bool succeeded = true;
    if constexpr (Stage == BufferHandleReceiveStage::kImport) {
      const auto measured_start = std::chrono::steady_clock::now();
      succeeded = import_batch();
      const auto measured_end = std::chrono::steady_clock::now();
      if (succeeded) {
        state.SetIterationTime(
            std::chrono::duration<double>(measured_end - measured_start).count());
        for (const auto& imported : imported_buffers) {
          checksum ^= reinterpret_cast<std::uintptr_t>(imported.ptr());
        }
        imported_buffers.clear();
      }
    } else if constexpr (Stage == BufferHandleReceiveStage::kMaterialize) {
      succeeded = import_batch();
      if (succeeded) {
        const auto measured_start = std::chrono::steady_clock::now();
        succeeded = materialize_batch();
        const auto measured_end = std::chrono::steady_clock::now();
        if (succeeded) {
          state.SetIterationTime(
              std::chrono::duration<double>(measured_end - measured_start).count());
          for (const auto& tensor : materialized_tensors) {
            checksum ^= reinterpret_cast<std::uintptr_t>(tensor.data());
          }
          imported_buffers.clear();
          materialized_tensors.clear();
        }
      }
    } else {
      succeeded = import_batch() && materialize_batch();
      if (succeeded) {
        const auto measured_start = std::chrono::steady_clock::now();
        materialized_tensors.clear();
        const auto measured_end = std::chrono::steady_clock::now();
        state.SetIterationTime(
            std::chrono::duration<double>(measured_end - measured_start).count());
        imported_buffers.clear();
      }
    }
    if (!succeeded) {
      state.SkipWithError("failed to process HoloMQ BufferHandle receive stage batch");
      break;
    }
    ++completed_batches;
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(completed_batches * static_cast<std::int64_t>(batch_size));
  state.counters["batch_size"] = static_cast<double>(batch_size);
  state.counters["device_storage"] =
      static_cast<double>(StorageType == holoscan::MemoryStorageType::kDevice);
  state.counters["import_only"] = static_cast<double>(Stage == BufferHandleReceiveStage::kImport);
  state.counters["materialize_only"] =
      static_cast<double>(Stage == BufferHandleReceiveStage::kMaterialize);
  state.counters["release_only"] = static_cast<double>(Stage == BufferHandleReceiveStage::kRelease);
  state.counters["tensor_bytes"] = static_cast<double>(kTensorBytes);
}

void BM_ChannelTensor_HoloMQBufferHandleImport(benchmark::State& state) {
  run_buffer_handle_receive_stage<holoscan::MemoryStorageType::kHost,
                                  BufferHandleReceiveStage::kImport>(state);
}

void BM_ChannelTensor_HoloMQBufferHandleMaterialize(benchmark::State& state) {
  run_buffer_handle_receive_stage<holoscan::MemoryStorageType::kHost,
                                  BufferHandleReceiveStage::kMaterialize>(state);
}

void BM_ChannelTensor_HoloMQBufferHandleRelease(benchmark::State& state) {
  run_buffer_handle_receive_stage<holoscan::MemoryStorageType::kHost,
                                  BufferHandleReceiveStage::kRelease>(state);
}

void BM_ChannelTensor_HoloMQDeviceBufferHandleImport(benchmark::State& state) {
  run_buffer_handle_receive_stage<holoscan::MemoryStorageType::kDevice,
                                  BufferHandleReceiveStage::kImport>(state);
}

void BM_ChannelTensor_HoloMQDeviceBufferHandleMaterialize(benchmark::State& state) {
  run_buffer_handle_receive_stage<holoscan::MemoryStorageType::kDevice,
                                  BufferHandleReceiveStage::kMaterialize>(state);
}

void BM_ChannelTensor_HoloMQDeviceBufferHandleRelease(benchmark::State& state) {
  run_buffer_handle_receive_stage<holoscan::MemoryStorageType::kDevice,
                                  BufferHandleReceiveStage::kRelease>(state);
}

void BM_ChannelTensor_HoloMQDevicePointerClassification(benchmark::State& state) {
  if (cuda_unavailable_for_holomq_pool(state)) {
    return;
  }

  const auto batch_size = static_cast<std::size_t>(state.range(0));
  auto fixture =
      make_holomq_tensor_pool_fixture(kTensorPoolSize, holoscan::MemoryStorageType::kDevice);
  if (!fixture || fixture->tensors.empty()) {
    state.SkipWithError("failed to create HoloMQ device Tensor benchmark pool");
    return;
  }
  void* const pointer = const_cast<void*>(fixture->tensors.front().data());

  DLDevice last_device{};
  std::int64_t completed_batches = 0;
  for (auto _ : state) {
    bool succeeded = true;
    const auto measured_start = std::chrono::steady_clock::now();
    for (std::size_t index = 0U; index < batch_size; ++index) {
      auto device = holoscan::dldevice_from_pointer(pointer);
      if (!device.has_value()) {
        succeeded = false;
        break;
      }
      last_device = device.value();
    }
    const auto measured_end = std::chrono::steady_clock::now();
    if (!succeeded) {
      state.SkipWithError("failed to classify HoloMQ device Tensor pointer");
      break;
    }
    state.SetIterationTime(std::chrono::duration<double>(measured_end - measured_start).count());
    ++completed_batches;
  }

  benchmark::DoNotOptimize(last_device);
  state.SetItemsProcessed(completed_batches * static_cast<std::int64_t>(batch_size));
  state.counters["batch_size"] = static_cast<double>(batch_size);
  state.counters["tensor_bytes"] = static_cast<double>(kTensorBytes);
}

template <std::size_t SubscriberCount>
void BM_TensorSidecarStore_InsertOnly(benchmark::State& state) {
  const auto batch_size = static_cast<std::size_t>(state.range(0));
  std::array<std::byte, kTensorBytes> storage{};
  auto tensor = make_sidecar_benchmark_tensor(storage);
  if (!tensor.has_value()) {
    state.SkipWithError("failed to create sidecar benchmark Tensor");
    return;
  }

  holoscan::TensorSidecarStore store(/*capacity=*/batch_size,
                                     /*expected_recipient_count=*/SubscriberCount);
  if (!register_sidecar_benchmark_recipients(store, SubscriberCount)) {
    state.SkipWithError("failed to register sidecar benchmark recipients");
    return;
  }

  std::vector<holoscan::ChannelSampleDescriptor> descriptors(batch_size);
  std::int64_t completed_batches = 0;
  for (auto _ : state) {
    bool succeeded = true;
    const auto measured_start = std::chrono::steady_clock::now();
    for (std::size_t index = 0U; index < batch_size; ++index) {
      auto descriptor = store.insert(holoscan::Tensor(tensor.value()));
      if (!descriptor.has_value()) {
        succeeded = false;
        break;
      }
      descriptors[index] = descriptor.value();
    }
    const auto measured_end = std::chrono::steady_clock::now();
    if (!succeeded) {
      state.SkipWithError("failed to insert sidecar benchmark Tensor");
      break;
    }
    state.SetIterationTime(std::chrono::duration<double>(measured_end - measured_start).count());
    for (const auto& descriptor : descriptors) {
      store.release_all(descriptor);
    }
    ++completed_batches;
  }

  state.SetItemsProcessed(completed_batches * static_cast<std::int64_t>(batch_size));
  state.counters["batch_size"] = static_cast<double>(batch_size);
  state.counters["subscribers"] = static_cast<double>(SubscriberCount);
  state.counters["tensor_bytes"] = static_cast<double>(kTensorBytes);
}

template <std::size_t SubscriberCount>
void BM_TensorSidecarStore_AcquireOnly(benchmark::State& state) {
  const auto batch_size = static_cast<std::size_t>(state.range(0));
  std::array<std::byte, kTensorBytes> storage{};
  auto tensor = make_sidecar_benchmark_tensor(storage);
  if (!tensor.has_value()) {
    state.SkipWithError("failed to create sidecar benchmark Tensor");
    return;
  }

  holoscan::TensorSidecarStore store(/*capacity=*/batch_size,
                                     /*expected_recipient_count=*/SubscriberCount);
  if (!register_sidecar_benchmark_recipients(store, SubscriberCount)) {
    state.SkipWithError("failed to register sidecar benchmark recipients");
    return;
  }

  std::uint64_t checksum = 0U;
  std::vector<holoscan::ChannelSampleDescriptor> descriptors(batch_size);
  std::int64_t completed_batches = 0;
  for (auto _ : state) {
    bool succeeded = true;
    for (std::size_t index = 0U; index < batch_size; ++index) {
      auto descriptor = store.insert(holoscan::Tensor(tensor.value()));
      if (!descriptor.has_value()) {
        succeeded = false;
        break;
      }
      descriptors[index] = descriptor.value();
    }
    if (!succeeded) {
      state.SkipWithError("failed to prefill sidecar acquire batch");
      break;
    }

    const auto measured_start = std::chrono::steady_clock::now();
    for (const auto& descriptor : descriptors) {
      for (std::size_t index = 0U; index < SubscriberCount; ++index) {
        auto received = store.acquire(descriptor, index);
        if (!received.has_value()) {
          succeeded = false;
          break;
        }
        checksum ^= reinterpret_cast<std::uintptr_t>(received->data());
      }
      if (!succeeded) {
        break;
      }
    }
    const auto measured_end = std::chrono::steady_clock::now();
    if (!succeeded) {
      state.SkipWithError("failed to acquire sidecar benchmark Tensor");
      break;
    }
    state.SetIterationTime(std::chrono::duration<double>(measured_end - measured_start).count());
    ++completed_batches;
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(completed_batches * static_cast<std::int64_t>(batch_size) *
                          static_cast<std::int64_t>(SubscriberCount));
  state.counters["batch_size"] = static_cast<double>(batch_size);
  state.counters["subscribers"] = static_cast<double>(SubscriberCount);
  state.counters["tensor_bytes"] = static_cast<double>(kTensorBytes);
}

template <std::size_t SubscriberCount>
void BM_TensorSidecarStore_InsertForceRelease(benchmark::State& state) {
  std::array<std::byte, kTensorBytes> storage{};
  auto tensor = make_sidecar_benchmark_tensor(storage);
  if (!tensor.has_value()) {
    state.SkipWithError("failed to create sidecar benchmark Tensor");
    return;
  }

  holoscan::TensorSidecarStore store(/*capacity=*/1U,
                                     /*expected_recipient_count=*/SubscriberCount);
  if (!register_sidecar_benchmark_recipients(store, SubscriberCount)) {
    state.SkipWithError("failed to register sidecar benchmark recipients");
    return;
  }

  for (auto _ : state) {
    auto descriptor = store.insert(holoscan::Tensor(tensor.value()));
    if (!descriptor.has_value()) {
      state.SkipWithError("failed to insert sidecar benchmark Tensor");
      break;
    }
    store.release_all(descriptor.value());
  }

  state.SetItemsProcessed(state.iterations());
  state.counters["subscribers"] = static_cast<double>(SubscriberCount);
  state.counters["tensor_bytes"] = static_cast<double>(kTensorBytes);
}

struct alignas(64) SpscDescriptorSlot {
  holoscan::ChannelSampleDescriptor descriptor{};
  std::atomic<bool> ready{false};
};

template <bool UseSidecarStore>
void run_spsc_descriptor_pipeline(  // NOLINT(readability-function-cognitive-complexity)
    benchmark::State& state) {
  const auto ring_depth = static_cast<std::size_t>(state.range(0));
  std::array<std::byte, kTensorBytes> storage{};
  auto tensor = make_sidecar_benchmark_tensor(storage);
  if (!tensor.has_value()) {
    state.SkipWithError("failed to create sidecar pipeline Tensor");
    return;
  }

  holoscan::TensorSidecarStore store(ring_depth, /*expected_recipient_count=*/1U);
  if (!register_sidecar_benchmark_recipients(store, 1U)) {
    state.SkipWithError("failed to register sidecar pipeline recipient");
    return;
  }

  // The runtime-sized ring owns non-movable atomic-bearing slots.
  auto slots = std::make_unique<SpscDescriptorSlot[]>(  // NOLINT(cppcoreguidelines-avoid-c-arrays)
      ring_depth);
  std::atomic<bool> producer_done{false};
  std::atomic<bool> pipeline_failed{false};  // NOLINT(misc-const-correctness)
  std::uint64_t checksum = 0U;
  std::thread consumer([&]() {
    std::size_t read_index = 0U;
    while (true) {
      SpscDescriptorSlot& slot = slots[read_index];
      if (!slot.ready.load(std::memory_order_acquire)) {
        if (producer_done.load(std::memory_order_acquire)) {
          break;
        }
        std::this_thread::yield();
        continue;
      }

      if constexpr (UseSidecarStore) {
        auto acquired = store.acquire(slot.descriptor, /*recipient_index=*/0U);
        if (!acquired.has_value()) {
          pipeline_failed.store(true, std::memory_order_release);
          return;
        }
        checksum ^= reinterpret_cast<std::uintptr_t>(acquired->data());
        checksum ^= static_cast<std::uint64_t>(acquired->nbytes());
      } else {
        checksum ^= slot.descriptor.sequence;
      }
      slot.ready.store(false, std::memory_order_release);
      read_index = (read_index + 1U) % ring_depth;
    }
  });

  std::size_t write_index = 0U;
  std::uint64_t baseline_sequence = 1U;  // NOLINT(misc-const-correctness)
  for (auto _ : state) {
    SpscDescriptorSlot& slot = slots[write_index];
    while (slot.ready.load(std::memory_order_acquire) &&
           !pipeline_failed.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    if (pipeline_failed.load(std::memory_order_acquire)) {
      break;
    }

    if constexpr (UseSidecarStore) {
      auto descriptor = store.insert(holoscan::Tensor(tensor.value()));
      if (!descriptor.has_value()) {
        pipeline_failed.store(true, std::memory_order_release);
        break;
      }
      slot.descriptor = descriptor.value();
    } else {
      slot.descriptor.sequence = baseline_sequence++;
    }
    slot.ready.store(true, std::memory_order_release);
    write_index = (write_index + 1U) % ring_depth;
  }

  producer_done.store(true, std::memory_order_release);
  consumer.join();
  if (pipeline_failed.load(std::memory_order_acquire)) {
    state.SkipWithError("sidecar descriptor pipeline failed");
    return;
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(state.iterations());
  state.counters["ring_depth"] = static_cast<double>(ring_depth);
  state.counters["sidecar_store"] = UseSidecarStore ? 1.0 : 0.0;
  state.counters["tensor_bytes"] = static_cast<double>(kTensorBytes);
}

void BM_DescriptorSpscRing(benchmark::State& state) {
  run_spsc_descriptor_pipeline<false>(state);
}

void BM_TensorSidecarStore_ConcurrentInsertAcquire(benchmark::State& state) {
  run_spsc_descriptor_pipeline<true>(state);
}

void BM_ChannelTensor_EmitBlockingWithOutstandingRecords(benchmark::State& state) {
  const auto outstanding_records = static_cast<std::size_t>(state.range(0));
  const auto required_records = outstanding_records + 1U;

  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime.has_value()) {
    const std::string message =
        "failed to create ChannelRuntime: " + holoscan::runtime_error_message(runtime.error());
    state.SkipWithError(message.c_str());
    return;
  }

  holoscan::ChannelOptions options =
      make_channel_benchmark_options(holoscan::ChannelQueuePolicy::kBlocking, 1U);
  options.max_subscriber_queue_depth = std::max<std::uint32_t>(
      kMinOutstandingSweepQueueDepth, static_cast<std::uint32_t>(required_records + 1U));
  options.subscriber_queue_depth = options.max_subscriber_queue_depth;

  const std::string channel_name = make_unique_channel_name("tensor_emit_occupancy_blocking");
  auto publisher = holoscan::channel_internal::TensorPublisherTestAccess::create(
      *runtime, channel_name, options);
  if (!publisher.has_value()) {
    const std::string message = "failed to create Tensor Channel publisher: " +
                                holoscan::runtime_error_message(publisher.error());
    state.SkipWithError(message.c_str());
    return;
  }

  auto subscriber = runtime->create_subscriber<holoscan::Tensor>(channel_name, options);
  if (!subscriber.has_value()) {
    const std::string message = "failed to create Tensor Channel subscriber: " +
                                holoscan::runtime_error_message(subscriber.error());
    state.SkipWithError(message.c_str());
    return;
  }

  const std::vector<holoscan::Tensor> tensors =
      make_tensor_pool(1U, std::max<std::size_t>(required_records, 1U));
  if (tensors.empty()) {
    state.SkipWithError("failed to allocate benchmark Tensor pool");
    return;
  }

  std::array<holoscan::ChannelSubscriber<holoscan::Tensor>, 1U> subscribers{
      std::move(subscriber).value()};
  if (!wait_for_subscriber_count(publisher.value(), subscribers, 1U, tensors)) {
    state.SkipWithError("timed out waiting for Tensor Channel subscriber");
    return;
  }

  std::uint64_t checksum = 0U;
  std::size_t tensor_index = 0U;
  for (auto _ : state) {
    state.PauseTiming();
    for (std::size_t i = 0U; i < outstanding_records; ++i) {
      if (!emit_tensor_or_skip(
              state, publisher.value(), tensors[tensor_index++ % tensors.size()], "prefill emit")) {
        state.ResumeTiming();
        return;
      }
    }

    // Only this additional emit is measured. The subscriber is intentionally not drained until
    // after the measured emit, so both the sidecar store and HoloMQ subscriber queue already hold
    // the requested number of unread descriptors.
    state.ResumeTiming();
    const bool measured_emit_ok = emit_tensor_or_skip(
        state, publisher.value(), tensors[tensor_index++ % tensors.size()], "measured emit");
    state.PauseTiming();
    if (!measured_emit_ok || !drain_expected_samples_or_skip(
                                 state, subscribers[0], outstanding_records + 1U, checksum)) {
      state.ResumeTiming();
      return;
    }
    state.ResumeTiming();
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(state.iterations());
  state.counters["outstanding_records"] = static_cast<double>(outstanding_records);
  state.counters["sidecar_record_capacity"] =
      static_cast<double>(options.max_subscriber_queue_depth) *
      static_cast<double>(options.expected_recipient_count);
  state.counters["subscriber_queue_depth"] = static_cast<double>(options.subscriber_queue_depth);
  state.counters["subscribers"] = 1.0;
  state.counters["tensor_bytes"] = static_cast<double>(kTensorBytes);
}

[[nodiscard]] std::optional<holoscan::Tensor> receive_catalog_tensor_or_skip(
    benchmark::State& state, holoscan::ChannelSubscriber<holoscan::Tensor>& subscriber) {
  // The caller pauses benchmark timing before entering. Resume only on failure so successful
  // transport polling and recipient retirement remain outside the measured loan reacquisition.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (std::chrono::steady_clock::now() < deadline) {
    auto received = subscriber.receive();
    if (received.has_value()) {
      return std::move(*received);
    }
    if (received.error().code != holoscan::ErrorCode::kNotReady) {
      state.ResumeTiming();
      state.SkipWithError("failed to receive catalog Tensor loan");
      return std::nullopt;
    }
    std::this_thread::yield();
  }
  state.ResumeTiming();
  state.SkipWithError("timed out receiving catalog Tensor loan");
  return std::nullopt;
}

void BM_CatalogTensorLoanAfterPublicationByCapacity(benchmark::State& state) {
  const auto slot_capacity = static_cast<std::uint32_t>(state.range(0));
  const std::string endpoint_name = make_unique_channel_name("catalog_tensor_loan_capacity");
  auto catalog = make_catalog_tensor_benchmark(endpoint_name, slot_capacity);
  if (!catalog.has_value()) {
    const std::string message =
        "failed to create catalog: " + holoscan::runtime_error_message(catalog.error());
    state.SkipWithError(message.c_str());
    return;
  }
  auto endpoint = catalog->endpoint(endpoint_name);
  auto runtime = holoscan::ChannelRuntime::create();
  if (!endpoint.has_value() || !runtime.has_value()) {
    state.SkipWithError("failed to create catalog endpoint or ChannelRuntime");
    return;
  }
  auto publisher = runtime->create_publisher<holoscan::Tensor>(*endpoint);
  auto subscriber =
      holoscan::channel_internal::TensorPublisherTestAccess::create_catalog_subscriber(*runtime,
                                                                                       *endpoint);
  if (!publisher.has_value() || !subscriber.has_value() ||
      !wait_for_catalog_tensor_subscriber(*publisher, *subscriber)) {
    state.SkipWithError("failed to connect catalog Tensor endpoints");
    return;
  }

  const std::array<std::int64_t, 1U> shape{static_cast<std::int64_t>(kTensorBytes)};
  const holoscan::TensorLoanRequest request{
      .shape = shape,
      .dtype = DLDataType{.code = kDLUInt, .bits = 8U, .lanes = 1U},
  };
  auto initial_loan = publisher->loan_tensor(request);
  if (!initial_loan.has_value()) {
    state.SkipWithError("failed to acquire initial catalog Tensor loan");
    return;
  }
  std::optional<holoscan::TensorPublishLoan> loan;
  loan.emplace(std::move(*initial_loan));

  std::uint64_t checksum = 0U;
  for (auto _ : state) {
    state.PauseTiming();
    auto writer = loan->write_host();
    if (!writer.has_value() || !std::move(*writer).commit().has_value()) {
      state.ResumeTiming();
      state.SkipWithError("failed to commit catalog Tensor loan");
      return;
    }
    auto emitted = publisher->emit(std::move(*loan));
    loan.reset();
    if (!emitted.has_value() || *emitted != 1U) {
      state.ResumeTiming();
      state.SkipWithError("failed to publish catalog Tensor loan");
      return;
    }

    auto received_tensor = receive_catalog_tensor_or_skip(state, *subscriber);
    if (!received_tensor.has_value()) {
      return;
    }
    checksum += received_tensor->nbytes();
    received_tensor.reset();
    if (!publisher->update_connections().has_value()) {
      state.ResumeTiming();
      state.SkipWithError("failed to reclaim catalog Tensor publication");
      return;
    }

    state.ResumeTiming();
    auto next_loan = publisher->loan_tensor(request);
    state.PauseTiming();
    if (!next_loan.has_value()) {
      state.ResumeTiming();
      state.SkipWithError("failed to reacquire catalog Tensor loan");
      return;
    }
    loan.emplace(std::move(*next_loan));
    state.ResumeTiming();
  }

  benchmark::DoNotOptimize(checksum);
  state.SetItemsProcessed(state.iterations());
  state.counters["active_loans"] = 1.0;
  state.counters["slot_capacity"] = static_cast<double>(slot_capacity);
  state.counters["tensor_bytes"] = static_cast<double>(kTensorBytes);
}

void BM_ChannelTensor_OneToOne_Blocking(benchmark::State& state) {
  run_channel_tensor_polling<1>(
      state, holoscan::ChannelQueuePolicy::kBlocking, "tensor_1to1_blocking");
}

void BM_ChannelTensor_OneToOne_DropOldestDepthOne(benchmark::State& state) {
  run_channel_tensor_polling<1>(state,
                                holoscan::ChannelQueuePolicy::kDropOldest,
                                "tensor_1to1_drop_oldest_depth_1",
                                holoscan::MemoryStorageType::kHost,
                                1U);
}

void BM_ChannelTensor_LocalPayload_Unified(benchmark::State& state) {
  run_tensor_local_payload_round_trip<TensorTransportPayload>(state,
                                                              "tensor_local_payload_unified");
}

void BM_ChannelTensor_LocalPayload_Slim(benchmark::State& state) {
  run_tensor_local_payload_round_trip<LocalTensorTransportPayload>(state,
                                                                   "tensor_local_payload_slim");
}

void BM_ChannelTensor_LocalPayload_UnifiedTransportOnly(benchmark::State& state) {
  run_tensor_local_payload_round_trip<TensorTransportPayload, LocalPayloadWork::kTransportOnly>(
      state, "tensor_local_payload_unified_transport_only");
}

void BM_ChannelTensor_LocalPayload_SlimTransportOnly(benchmark::State& state) {
  run_tensor_local_payload_round_trip<LocalTensorTransportPayload,
                                      LocalPayloadWork::kTransportOnly>(
      state, "tensor_local_payload_slim_transport_only");
}

void BM_ChannelTensor_LocalPayload_UnifiedCallbackOnly(benchmark::State& state) {
  run_tensor_local_payload_round_trip<TensorTransportPayload, LocalPayloadWork::kCallbackOnly>(
      state, "tensor_local_payload_unified_callback_only");
}

void BM_ChannelTensor_LocalPayload_SlimCallbackOnly(benchmark::State& state) {
  run_tensor_local_payload_round_trip<LocalTensorTransportPayload, LocalPayloadWork::kCallbackOnly>(
      state, "tensor_local_payload_slim_callback_only");
}

void BM_ChannelTensor_LocalTransport_UnifiedPublish(benchmark::State& state) {
  run_tensor_local_transport_stage<TensorTransportPayload, LocalTransportStage::kPublish>(
      state, "tensor_local_transport_unified_publish");
}

void BM_ChannelTensor_LocalTransport_SlimPublish(benchmark::State& state) {
  run_tensor_local_transport_stage<LocalTensorTransportPayload, LocalTransportStage::kPublish>(
      state, "tensor_local_transport_slim_publish");
}

void BM_ChannelTensor_LocalTransport_UnifiedTake(benchmark::State& state) {
  run_tensor_local_transport_stage<TensorTransportPayload, LocalTransportStage::kTake>(
      state, "tensor_local_transport_unified_take");
}

void BM_ChannelTensor_LocalTransport_SlimTake(benchmark::State& state) {
  run_tensor_local_transport_stage<LocalTensorTransportPayload, LocalTransportStage::kTake>(
      state, "tensor_local_transport_slim_take");
}

template <std::size_t SubscriberCount>
void BM_ChannelTensor_DropOldestOverwrite_TransportCallback(benchmark::State& state) {
  run_channel_tensor_drop_oldest_overwrite<SubscriberCount,
                                           ReplacementCleanupMode::kTransportCallback>(state);
}

template <std::size_t SubscriberCount>
void BM_ChannelTensor_DropOldestOverwrite_PublisherMirror(benchmark::State& state) {
  run_channel_tensor_drop_oldest_overwrite<SubscriberCount,
                                           ReplacementCleanupMode::kPublisherMirror>(state);
}

template <std::size_t SubscriberCount>
void BM_ChannelTensor_DropOldestOverwrite_BlockingBaseline(benchmark::State& state) {
  run_channel_tensor_drop_oldest_overwrite<SubscriberCount,
                                           ReplacementCleanupMode::kBlockingBaseline>(state);
}

template <std::size_t SubscriberCount>
void BM_ChannelTensor_Fanout_Blocking(benchmark::State& state) {
  run_channel_tensor_polling<SubscriberCount>(
      state, holoscan::ChannelQueuePolicy::kBlocking, "tensor_fanout_blocking");
}

template <std::size_t SubscriberCount>
void BM_ChannelTensor_Fanout_DropOldestDepthOne(benchmark::State& state) {
  run_channel_tensor_polling<SubscriberCount>(state,
                                              holoscan::ChannelQueuePolicy::kDropOldest,
                                              "tensor_fanout_drop_oldest_depth_1",
                                              holoscan::MemoryStorageType::kHost,
                                              1U);
}

// Device-backed variants exercise the same descriptor/handle transport hot path while carrying
// real CUDA device Tensors (allocated from a BlockMemoryPool) instead of host-array handles. They
// deliberately do NOT launch CUDA work, set readiness/deallocation streams, or exercise DLPack
// lifetime extension, so they measure device-backed Tensor *handle transport* overhead -- not full
// stream-safe production Tensor handoff (which is separate stream/event pipeline benchmark work).
// They skip when no CUDA device is present.
void BM_ChannelTensor_OneToOne_DeviceBlocking(benchmark::State& state) {
  if (device_unavailable_for(state, holoscan::MemoryStorageType::kDevice)) {
    return;
  }
  run_channel_tensor_polling<1>(state,
                                holoscan::ChannelQueuePolicy::kBlocking,
                                "tensor_1to1_device_blocking",
                                holoscan::MemoryStorageType::kDevice);
}

void BM_ChannelTensor_OneToOne_DeviceEmitBlocking(benchmark::State& state) {
  if (device_unavailable_for(state, holoscan::MemoryStorageType::kDevice)) {
    return;
  }
  run_channel_tensor_polling<1, ChannelTensorMeasuredStage::kEmit>(
      state,
      holoscan::ChannelQueuePolicy::kBlocking,
      "tensor_1to1_device_emit_blocking",
      holoscan::MemoryStorageType::kDevice);
}

void BM_ChannelTensor_OneToOne_DeviceReceiveBlocking(benchmark::State& state) {
  if (device_unavailable_for(state, holoscan::MemoryStorageType::kDevice)) {
    return;
  }
  run_channel_tensor_polling<1, ChannelTensorMeasuredStage::kReceive>(
      state,
      holoscan::ChannelQueuePolicy::kBlocking,
      "tensor_1to1_device_receive_blocking",
      holoscan::MemoryStorageType::kDevice);
}

void BM_ChannelTensor_OneToOne_HoloMQPoolBlocking(benchmark::State& state) {
  run_channel_tensor_holomq_pool_polling<1>(state, "tensor_1to1_holomq_pool_blocking");
}

void BM_ChannelTensor_OneToOne_HoloMQBufferHandleBlocking(benchmark::State& state) {
  run_channel_tensor_holomq_pool_polling<1>(state,
                                            "tensor_1to1_holomq_buffer_handle_blocking",
                                            holoscan::ChannelTensorTransferMode::kBufferHandle);
}

void BM_ChannelTensor_OneToOne_HoloMQPoolEmitBlocking(benchmark::State& state) {
  run_channel_tensor_holomq_pool_polling<1, ChannelTensorMeasuredStage::kEmit>(
      state, "tensor_1to1_holomq_pool_emit_blocking");
}

void BM_ChannelTensor_OneToOne_HoloMQPoolEventWakeEmitBlocking(benchmark::State& state) {
  run_channel_tensor_holomq_pool_polling<1, ChannelTensorMeasuredStage::kEmit>(
      state,
      "tensor_1to1_holomq_pool_event_wake_emit_blocking",
      holoscan::ChannelTensorTransferMode::kSameProcessSidecar,
      holoscan::MemoryStorageType::kHost,
      true);
}

void BM_ChannelTensor_OneToOne_HoloMQBufferHandleEmitBlocking(benchmark::State& state) {
  run_channel_tensor_holomq_pool_polling<1, ChannelTensorMeasuredStage::kEmit>(
      state,
      "tensor_1to1_holomq_buffer_handle_emit_blocking",
      holoscan::ChannelTensorTransferMode::kBufferHandle);
}

void BM_ChannelTensor_OneToOne_HoloMQPoolReceiveBlocking(benchmark::State& state) {
  run_channel_tensor_holomq_pool_polling<1, ChannelTensorMeasuredStage::kReceive>(
      state, "tensor_1to1_holomq_pool_receive_blocking");
}

void BM_ChannelTensor_OneToOne_HoloMQBufferHandleReceiveBlocking(benchmark::State& state) {
  run_channel_tensor_holomq_pool_polling<1, ChannelTensorMeasuredStage::kReceive>(
      state,
      "tensor_1to1_holomq_buffer_handle_receive_blocking",
      holoscan::ChannelTensorTransferMode::kBufferHandle);
}

void BM_ChannelTensor_OneToOne_HoloMQDevicePoolBlocking(benchmark::State& state) {
  run_channel_tensor_holomq_pool_polling<1>(
      state,
      "tensor_1to1_holomq_device_pool_blocking",
      holoscan::ChannelTensorTransferMode::kSameProcessSidecar,
      holoscan::MemoryStorageType::kDevice);
}

void BM_ChannelTensor_OneToOne_HoloMQDeviceBufferHandleBlocking(benchmark::State& state) {
  run_channel_tensor_holomq_pool_polling<1>(state,
                                            "tensor_1to1_holomq_device_buffer_handle_blocking",
                                            holoscan::ChannelTensorTransferMode::kBufferHandle,
                                            holoscan::MemoryStorageType::kDevice);
}

void BM_ChannelTensor_OneToOne_HoloMQDevicePoolEmitBlocking(benchmark::State& state) {
  run_channel_tensor_holomq_pool_polling<1, ChannelTensorMeasuredStage::kEmit>(
      state,
      "tensor_1to1_holomq_device_pool_emit_blocking",
      holoscan::ChannelTensorTransferMode::kSameProcessSidecar,
      holoscan::MemoryStorageType::kDevice);
}

void BM_ChannelTensor_OneToOne_HoloMQDeviceBufferHandleEmitBlocking(benchmark::State& state) {
  run_channel_tensor_holomq_pool_polling<1, ChannelTensorMeasuredStage::kEmit>(
      state,
      "tensor_1to1_holomq_device_buffer_handle_emit_blocking",
      holoscan::ChannelTensorTransferMode::kBufferHandle,
      holoscan::MemoryStorageType::kDevice);
}

void BM_ChannelTensor_OneToOne_HoloMQDevicePoolReceiveBlocking(benchmark::State& state) {
  run_channel_tensor_holomq_pool_polling<1, ChannelTensorMeasuredStage::kReceive>(
      state,
      "tensor_1to1_holomq_device_pool_receive_blocking",
      holoscan::ChannelTensorTransferMode::kSameProcessSidecar,
      holoscan::MemoryStorageType::kDevice);
}

void BM_ChannelTensor_OneToOne_HoloMQDeviceBufferHandleReceiveBlocking(benchmark::State& state) {
  run_channel_tensor_holomq_pool_polling<1, ChannelTensorMeasuredStage::kReceive>(
      state,
      "tensor_1to1_holomq_device_buffer_handle_receive_blocking",
      holoscan::ChannelTensorTransferMode::kBufferHandle,
      holoscan::MemoryStorageType::kDevice);
}

template <std::size_t SubscriberCount>
void BM_ChannelTensor_Fanout_DeviceBlocking(benchmark::State& state) {
  if (device_unavailable_for(state, holoscan::MemoryStorageType::kDevice)) {
    return;
  }
  run_channel_tensor_polling<SubscriberCount>(state,
                                              holoscan::ChannelQueuePolicy::kBlocking,
                                              "tensor_fanout_device_blocking",
                                              holoscan::MemoryStorageType::kDevice);
}

template <std::size_t SubscriberCount>
void BM_ChannelTensor_Fanout_HoloMQPoolBlocking(benchmark::State& state) {
  run_channel_tensor_holomq_pool_polling<SubscriberCount>(state,
                                                          "tensor_fanout_holomq_pool_blocking");
}

BENCHMARK(BM_ChannelTensor_OneToOne_Blocking)->UseRealTime();
BENCHMARK(BM_ChannelTensor_OneToOne_DropOldestDepthOne)->UseRealTime();
BENCHMARK(BM_ChannelTensor_LocalPayload_Unified)->UseRealTime();
BENCHMARK(BM_ChannelTensor_LocalPayload_Slim)->UseRealTime();
BENCHMARK(BM_ChannelTensor_LocalPayload_UnifiedTransportOnly)->UseRealTime();
BENCHMARK(BM_ChannelTensor_LocalPayload_SlimTransportOnly)->UseRealTime();
BENCHMARK(BM_ChannelTensor_LocalPayload_UnifiedCallbackOnly)->UseRealTime();
BENCHMARK(BM_ChannelTensor_LocalPayload_SlimCallbackOnly)->UseRealTime();
BENCHMARK(BM_ChannelTensor_LocalTransport_UnifiedPublish)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK(BM_ChannelTensor_LocalTransport_SlimPublish)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK(BM_ChannelTensor_LocalTransport_UnifiedTake)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK(BM_ChannelTensor_LocalTransport_SlimTake)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK_TEMPLATE(BM_ChannelTensor_Fanout_Blocking, 2)->UseRealTime();
BENCHMARK_TEMPLATE(BM_ChannelTensor_Fanout_Blocking, 4)->UseRealTime();
BENCHMARK_TEMPLATE(BM_ChannelTensor_Fanout_Blocking, 8)->UseRealTime();
BENCHMARK_TEMPLATE(BM_ChannelTensor_Fanout_DropOldestDepthOne, 2)->UseRealTime();
BENCHMARK_TEMPLATE(BM_ChannelTensor_Fanout_DropOldestDepthOne, 4)->UseRealTime();
BENCHMARK_TEMPLATE(BM_ChannelTensor_Fanout_DropOldestDepthOne, 8)->UseRealTime();
BENCHMARK_TEMPLATE(BM_ChannelTensor_DropOldestOverwrite_TransportCallback, 1)->UseManualTime();
BENCHMARK_TEMPLATE(BM_ChannelTensor_DropOldestOverwrite_TransportCallback, 2)->UseManualTime();
BENCHMARK_TEMPLATE(BM_ChannelTensor_DropOldestOverwrite_TransportCallback, 4)->UseManualTime();
BENCHMARK_TEMPLATE(BM_ChannelTensor_DropOldestOverwrite_TransportCallback, 8)->UseManualTime();
BENCHMARK_TEMPLATE(BM_ChannelTensor_DropOldestOverwrite_PublisherMirror, 1)->UseManualTime();
BENCHMARK_TEMPLATE(BM_ChannelTensor_DropOldestOverwrite_PublisherMirror, 2)->UseManualTime();
BENCHMARK_TEMPLATE(BM_ChannelTensor_DropOldestOverwrite_PublisherMirror, 4)->UseManualTime();
BENCHMARK_TEMPLATE(BM_ChannelTensor_DropOldestOverwrite_PublisherMirror, 8)->UseManualTime();
BENCHMARK_TEMPLATE(BM_ChannelTensor_DropOldestOverwrite_BlockingBaseline, 1)->UseManualTime();
BENCHMARK_TEMPLATE(BM_ChannelTensor_DropOldestOverwrite_BlockingBaseline, 2)->UseManualTime();
BENCHMARK_TEMPLATE(BM_ChannelTensor_DropOldestOverwrite_BlockingBaseline, 4)->UseManualTime();
BENCHMARK_TEMPLATE(BM_ChannelTensor_DropOldestOverwrite_BlockingBaseline, 8)->UseManualTime();
BENCHMARK_TEMPLATE(BM_TensorSidecarStore_InsertAcquire, 1)->UseRealTime();
BENCHMARK_TEMPLATE(BM_TensorSidecarStore_InsertAcquire, 2)->UseRealTime();
BENCHMARK_TEMPLATE(BM_TensorSidecarStore_InsertAcquire, 4)->UseRealTime();
BENCHMARK_TEMPLATE(BM_TensorSidecarStore_InsertOnly, 1)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK_TEMPLATE(BM_TensorSidecarStore_InsertOnly, 2)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK_TEMPLATE(BM_TensorSidecarStore_InsertOnly, 4)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK_TEMPLATE(BM_TensorSidecarStore_AcquireOnly, 1)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK_TEMPLATE(BM_TensorSidecarStore_AcquireOnly, 2)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK_TEMPLATE(BM_TensorSidecarStore_AcquireOnly, 4)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK_TEMPLATE(BM_TensorSidecarStore_InsertForceRelease, 1)->UseRealTime();
BENCHMARK_TEMPLATE(BM_TensorSidecarStore_InsertForceRelease, 2)->UseRealTime();
BENCHMARK_TEMPLATE(BM_TensorSidecarStore_InsertForceRelease, 4)->UseRealTime();
BENCHMARK(BM_ChannelTensor_HoloMQBufferHandleExportDescriptor)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK(BM_ChannelTensor_HoloMQBufferHandleConstructDescriptor)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK(BM_ChannelTensor_HoloMQDeviceBufferHandleExportDescriptor)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK(BM_ChannelTensor_HoloMQDeviceBufferHandleConstructDescriptor)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK(BM_ChannelTensor_HoloMQBufferHandleImport)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK(BM_ChannelTensor_HoloMQBufferHandleMaterialize)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK(BM_ChannelTensor_HoloMQBufferHandleRelease)->Arg(1)->Arg(8)->Arg(64)->UseManualTime();
BENCHMARK(BM_ChannelTensor_HoloMQDeviceBufferHandleImport)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK(BM_ChannelTensor_HoloMQDeviceBufferHandleMaterialize)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK(BM_ChannelTensor_HoloMQDeviceBufferHandleRelease)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK(BM_ChannelTensor_HoloMQDevicePointerClassification)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->UseManualTime();
BENCHMARK(BM_DescriptorSpscRing)->Arg(1)->Arg(8)->Arg(64)->UseRealTime();
BENCHMARK(BM_TensorSidecarStore_ConcurrentInsertAcquire)->Arg(1)->Arg(8)->Arg(64)->UseRealTime();
BENCHMARK(BM_ChannelTensor_EmitBlockingWithOutstandingRecords)
    ->Arg(0)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->Arg(256)
    ->Arg(1024)
    ->UseRealTime();
BENCHMARK(BM_CatalogTensorLoanAfterPublicationByCapacity)
    ->Arg(1)
    ->Arg(8)
    ->Arg(64)
    ->Arg(1024)
    ->UseRealTime();
BENCHMARK(BM_ChannelTensor_OneToOne_DeviceBlocking)->UseRealTime();
BENCHMARK(BM_ChannelTensor_OneToOne_DeviceEmitBlocking)->UseManualTime();
BENCHMARK(BM_ChannelTensor_OneToOne_DeviceReceiveBlocking)->UseManualTime();
BENCHMARK_TEMPLATE(BM_ChannelTensor_Fanout_DeviceBlocking, 2)->UseRealTime();
BENCHMARK(BM_ChannelTensor_OneToOne_HoloMQPoolBlocking)->UseRealTime();
BENCHMARK(BM_ChannelTensor_OneToOne_HoloMQBufferHandleBlocking)->UseRealTime();
BENCHMARK(BM_ChannelTensor_OneToOne_HoloMQPoolEmitBlocking)->UseManualTime();
BENCHMARK(BM_ChannelTensor_OneToOne_HoloMQPoolEventWakeEmitBlocking)->UseManualTime();
BENCHMARK(BM_ChannelTensor_OneToOne_HoloMQBufferHandleEmitBlocking)->UseManualTime();
BENCHMARK(BM_ChannelTensor_OneToOne_HoloMQPoolReceiveBlocking)->UseManualTime();
BENCHMARK(BM_ChannelTensor_OneToOne_HoloMQBufferHandleReceiveBlocking)->UseManualTime();
BENCHMARK(BM_ChannelTensor_OneToOne_HoloMQDevicePoolBlocking)->UseRealTime();
BENCHMARK(BM_ChannelTensor_OneToOne_HoloMQDeviceBufferHandleBlocking)->UseRealTime();
BENCHMARK(BM_ChannelTensor_OneToOne_HoloMQDevicePoolEmitBlocking)->UseManualTime();
BENCHMARK(BM_ChannelTensor_OneToOne_HoloMQDeviceBufferHandleEmitBlocking)->UseManualTime();
BENCHMARK(BM_ChannelTensor_OneToOne_HoloMQDevicePoolReceiveBlocking)->UseManualTime();
BENCHMARK(BM_ChannelTensor_OneToOne_HoloMQDeviceBufferHandleReceiveBlocking)->UseManualTime();
BENCHMARK_TEMPLATE(BM_ChannelTensor_Fanout_HoloMQPoolBlocking, 2)->UseRealTime();

}  // namespace
