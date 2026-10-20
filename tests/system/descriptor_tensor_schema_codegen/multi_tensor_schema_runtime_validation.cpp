/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <fmt/format.h>

#if defined(__linux__)
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <holoscan/channel/runtime.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/schema/descriptor_flatbuffer_payload.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "descriptor_frame_multi_order_traits.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_tensor_resources.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime/runtime_worker.hpp"

namespace {

bool cuda_scenario_skipped = false;

using namespace std::chrono_literals;

using Native = app3p::MultiDescriptorFrameT;
using Root = app3p::MultiDescriptorFrame;
using Fields = app3p::MultiDescriptorFrameTensorFields;
using ExternalPayload = holoscan::schema::DescriptorFlatBufferPayloadFor<Root>;
using ExternalEnvelope = holoscan::channel_internal::TensorSchemaRuntimeEnvelope<ExternalPayload>;

static_assert(ExternalPayload::max_tensors == 3U);

constexpr std::string_view kInstanceEnvironment = "HOLOSCAN_TEST_MULTI_TENSOR_INSTANCE";
constexpr std::uint32_t kDefaultQueueDepth = 2U;
constexpr holoscan::DeviceId kDevice{0};
constexpr std::array<std::uint16_t, 3U> kFieldIds{1U, 3U, 5U};
constexpr std::array<std::uint64_t, 3U> kFieldBounds{64U, 32U, 32U};
constexpr std::array<holoscan::MemoryKind, 3U> kFieldMemory{
    holoscan::MemoryKind::kHost,
    holoscan::MemoryKind::kCudaDevice,
    holoscan::MemoryKind::kPinnedHost,
};
constexpr std::array<std::int64_t, 2U> kFirstShape{2, 3};
constexpr std::array<std::int64_t, 1U> kSecondShape{4};
constexpr std::array<std::int64_t, 2U> kThirdShape{1, 2};
constexpr std::array<std::int64_t, 3U> kInvalidRankShape{1, 2, 3};
constexpr std::array<std::int64_t, 2U> kInvalidExtentShape{2, -1};
constexpr std::array<std::int64_t, 2U> kOverBoundShape{5, 5};
constexpr std::array<std::int64_t, 2U> kInvalidByteStrides{12, -4};
constexpr std::array<std::int64_t, 2U> kAxisOverBoundShape{3, 2};
constexpr std::array<std::int64_t, 2U> kInRangeFirstShape{2, 2};
constexpr std::array<std::int64_t, 2U> kPaddedFirstByteStrides{12, 4};
constexpr std::array<std::int64_t, 2U> kOverBoundThirdByteStrides{6, 2};
constexpr std::array<std::uint64_t, 2U> kThirdMaxByteStrides{4U, 2U};
constexpr std::size_t kInlineFrameCount = 3U;
constexpr std::array<float, 6U> kFirstValues{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
constexpr std::array<std::uint8_t, 4U> kSecondValues{3U, 5U, 8U, 13U};
constexpr std::array<std::int16_t, 2U> kThirdValues{21, 34};

enum class SourceMode : std::uint8_t {
  kValid,
  kMissingStagingContract,
  kMixedCarriage,
  kOverBound,
  kLoanBundle,
  kLoanBundleRestart,
  kInvalidLoanBundle,
  kFixedRateFields,
  kRateAwareFields,
  kRateAwareFieldsConfigure,
  kRateAwareFieldsRestart,
  kRateAwareFieldsRestartReject,
  kRateAwareFieldsWarmUp,
  kRateAwareFieldsArm,
};

[[nodiscard]] constexpr bool is_rate_aware_mode(const SourceMode mode) noexcept {
  return mode == SourceMode::kFixedRateFields || mode == SourceMode::kRateAwareFields ||
         mode == SourceMode::kRateAwareFieldsConfigure ||
         mode == SourceMode::kRateAwareFieldsRestart ||
         mode == SourceMode::kRateAwareFieldsRestartReject ||
         mode == SourceMode::kRateAwareFieldsWarmUp || mode == SourceMode::kRateAwareFieldsArm;
}

[[nodiscard]] constexpr bool is_rate_restart_mode(const SourceMode mode) noexcept {
  return mode == SourceMode::kRateAwareFieldsRestart ||
         mode == SourceMode::kRateAwareFieldsRestartReject;
}

[[nodiscard]] constexpr holoscan::LifecycleStage rate_binding_stage(
    const SourceMode mode) noexcept {
  if (mode == SourceMode::kRateAwareFieldsConfigure) {
    return holoscan::LifecycleStage::kConfigure;
  }
  if (mode == SourceMode::kRateAwareFieldsWarmUp) {
    return holoscan::LifecycleStage::kWarmUp;
  }
  if (mode == SourceMode::kRateAwareFieldsArm) {
    return holoscan::LifecycleStage::kArm;
  }
  return holoscan::LifecycleStage::kDiscover;
}

enum class Topology : std::uint8_t {
  kProcessLocal,
  kProcessLocalFanout,
  kCrossProcess,
  kCrossProcessFanout,
  // Drop-oldest fanout with the second sink in the producer's process group and the first in
  // another: a mixed-locality family.
  kMixedLocality,
};

struct InlineComputeRecord {
  std::uint32_t operator_id{};
  std::uint64_t frame{};
  holoscan::TagTime tag_time{};
};

struct ProbeState {
  static void reset() noexcept {
    source_starts.store(0U, std::memory_order_release);
    sink_starts.store(0U, std::memory_order_release);
    consumer_ready.store(0U, std::memory_order_release);
    source_released.store(false, std::memory_order_release);
    emitted.store(0U, std::memory_order_release);
    received.store(0U, std::memory_order_release);
    over_bound_rejections.store(0U, std::memory_order_release);
    invalid_bundle_rejections.store(0U, std::memory_order_release);
    allocate_callbacks.store(0U, std::memory_order_release);
    allocate_saw_preallocated_fields.store(false, std::memory_order_release);
    {
      const std::lock_guard lock{mutex};
      inline_compute_records.clear();
    }
    tensor_control.reset();
  }

  static void install_tensor_control(
      std::shared_ptr<holoscan::detail::RuntimeTensorOutputTestControl> control) noexcept {
    tensor_control = std::move(control);
  }

  static void record_allocate_pool_state() noexcept {
    bool preallocated = tensor_control != nullptr;
    for (std::size_t index = 0U; preallocated && index < 3U; ++index) {
      const auto snapshot = tensor_control->snapshot(index);
      preallocated = snapshot.materialization_count == 1U &&
                     snapshot.maximum_allocation_generation == 1U &&
                     snapshot.physical_slot_count != 0U && snapshot.active_entry_limit != 0U &&
                     snapshot.active_entry_limit <= snapshot.physical_slot_count;
    }
    allocate_saw_preallocated_fields.store(preallocated, std::memory_order_release);
    allocate_callbacks.fetch_add(1U, std::memory_order_acq_rel);
    notify();
  }

  static void notify() noexcept { ready.notify_all(); }

  static void record_inline_compute(const std::uint32_t operator_id, const std::uint64_t frame,
                                    const holoscan::TagTime tag_time) {
    const std::lock_guard lock{mutex};
    inline_compute_records.push_back(InlineComputeRecord{
        .operator_id = operator_id,
        .frame = frame,
        .tag_time = tag_time,
    });
  }

  [[nodiscard]] static std::vector<InlineComputeRecord> inline_compute_snapshot() {
    const std::lock_guard lock{mutex};
    return inline_compute_records;
  }

  [[nodiscard]] static bool wait_for(const std::atomic<std::size_t>& counter,
                                     const std::size_t expected) {
    std::unique_lock lock{mutex};
    return ready.wait_for(lock, 10s, [&counter, expected] {
      return counter.load(std::memory_order_acquire) >= expected;
    });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::size_t> source_starts{};
  static inline std::atomic<std::size_t> sink_starts{};
  static inline std::atomic<std::size_t> consumer_ready{};
  static inline std::atomic<bool> source_released{};
  static inline std::atomic<std::size_t> emitted{};
  static inline std::atomic<std::size_t> received{};
  static inline std::atomic<std::size_t> over_bound_rejections{};
  static inline std::atomic<std::size_t> invalid_bundle_rejections{};
  static inline std::atomic<std::size_t> allocate_callbacks{};
  static inline std::atomic<bool> allocate_saw_preallocated_fields{};
  static inline std::shared_ptr<holoscan::detail::RuntimeTensorOutputTestControl> tensor_control;
  static inline std::vector<InlineComputeRecord> inline_compute_records;
};

[[nodiscard]] bool check(const bool condition, const std::string_view message) {
  if (!condition) {
    HOLOSCAN_LOG_ERROR("{}", message);
  }
  return condition;
}

void print_diagnostics(const holoscan::RunSession& session) {
  for (const auto& diagnostic : session.diagnostics()) {
    const std::string port =
        diagnostic.source.port ? fmt::format("{}", diagnostic.source.port.value()) : "<none>";
    HOLOSCAN_LOG_ERROR("{} vertex={} port={} service={} required={} provided={}",
                       diagnostic.token,
                       diagnostic.source.vertex.value,
                       port,
                       diagnostic.service,
                       diagnostic.required,
                       diagnostic.provided);
  }
}

[[nodiscard]] holoscan::expected<holoscan::RateGenerationSnapshot, holoscan::Error>
wait_for_published_rate_generation(holoscan::RunSession& session,
                                   const holoscan::RateGenerationId after_generation = 0U) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto snapshot = session.capture_rate_generations();
    if (snapshot && snapshot->published() != nullptr &&
        snapshot->published()->rate_generation != after_generation) {
      return snapshot;
    }
    if (!snapshot && snapshot.error().code != holoscan::ErrorCode::kNotReady) {
      return holoscan::make_unexpected(std::move(snapshot.error()));
    }
    std::this_thread::sleep_for(1ms);
  }
  return holoscan::make_unexpected(
      holoscan::Error{holoscan::ErrorCode::kNotReady, "PORT_RATE_PUBLICATION_MISSING"});
}

[[nodiscard]] bool cuda_available() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > kDevice.value) {
    return true;
  }
  cuda_scenario_skipped = true;
  HOLOSCAN_LOG_INFO("SKIP: multi-Tensor runtime validation requires CUDA device 0");
  return false;
}

class FixtureTensorMemoryReference final : public holoscan::TensorMemoryReference {
 public:
  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override { return true; }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> synchronize_for_consumer(
      cudaStream_t) override {
    return {};
  }

  bool set_deallocation_stream(cudaStream_t) override { return true; }
};

[[nodiscard]] std::shared_ptr<holoscan::Tensor> make_tensor(
    void* data, const std::initializer_list<std::int64_t> shape, const DLDevice device,
    const DLDataType dtype) {
  auto context = std::make_shared<holoscan::DLManagedTensorContext>();
  if (device.device_type == kDLCUDA) {
    context->memory_reference = std::make_shared<FixtureTensorMemoryReference>();
  }
  context->dl_shape.assign(shape);
  context->tensor.dl_tensor.data = data;
  context->tensor.dl_tensor.device = device;
  context->tensor.dl_tensor.ndim = static_cast<std::int32_t>(context->dl_shape.size());
  context->tensor.dl_tensor.dtype = dtype;
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.strides = nullptr;
  context->tensor.dl_tensor.byte_offset = 0U;
  return std::make_shared<holoscan::Tensor>(std::move(context));
}

class TensorFixtureStorage {
 public:
  [[nodiscard]] bool initialize() {
    if (cudaSetDevice(kDevice.value) != cudaSuccess ||
        cudaMalloc(&device_data_, kSecondValues.size()) != cudaSuccess ||
        cudaMemcpy(
            device_data_, kSecondValues.data(), kSecondValues.size(), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        // FixtureTensorMemoryReference advertises no outstanding producer work. Complete the
        // default-stream initialization before the Tensor can reach a nonblocking consumer stream.
        cudaStreamSynchronize(nullptr) != cudaSuccess ||
        cudaMallocHost(reinterpret_cast<void**>(&pinned_data_),
                       pinned_values_.size() * sizeof(pinned_values_.front())) != cudaSuccess) {
      release();
      return false;
    }
    std::copy(pinned_values_.begin(), pinned_values_.end(), pinned_data_);
    first = make_tensor(first_values_.data(),
                        {2, 3},
                        DLDevice{.device_type = kDLCPU, .device_id = 0},
                        DLDataType{kDLFloat, 32U, 1U});
    second = make_tensor(device_data_,
                         {4},
                         DLDevice{.device_type = kDLCUDA, .device_id = kDevice.value},
                         DLDataType{kDLUInt, 8U, 1U});
    third = make_tensor(pinned_data_,
                        {1, 2},
                        DLDevice{.device_type = kDLCUDAHost, .device_id = 0},
                        DLDataType{kDLInt, 16U, 1U});
    return true;
  }

  void release() noexcept {
    first.reset();
    second.reset();
    third.reset();
    if (pinned_data_ != nullptr) {
      static_cast<void>(cudaFreeHost(pinned_data_));
      pinned_data_ = nullptr;
    }
    if (device_data_ != nullptr) {
      static_cast<void>(cudaFree(device_data_));
      device_data_ = nullptr;
    }
  }

  ~TensorFixtureStorage() { release(); }

  std::shared_ptr<holoscan::Tensor> first;
  std::shared_ptr<holoscan::Tensor> second;
  std::shared_ptr<holoscan::Tensor> third;

 private:
  std::array<float, 6U> first_values_{kFirstValues};
  std::array<std::int16_t, 2U> pinned_values_{kThirdValues};
  void* device_data_{};
  std::int16_t* pinned_data_{};
};

class HostTensorFixtureStorage {
 public:
  void initialize() {
    first = make_tensor(first_values_.data(),
                        {2, 3},
                        DLDevice{.device_type = kDLCPU, .device_id = 0},
                        DLDataType{kDLFloat, 32U, 1U});
    second = make_tensor(second_values_.data(),
                         {4},
                         DLDevice{.device_type = kDLCPU, .device_id = 0},
                         DLDataType{kDLUInt, 8U, 1U});
    third = make_tensor(third_values_.data(),
                        {1, 2},
                        DLDevice{.device_type = kDLCPU, .device_id = 0},
                        DLDataType{kDLInt, 16U, 1U});
  }

  std::shared_ptr<holoscan::Tensor> first;
  std::shared_ptr<holoscan::Tensor> second;
  std::shared_ptr<holoscan::Tensor> third;

 private:
  std::array<float, 6U> first_values_{kFirstValues};
  std::array<std::uint8_t, 4U> second_values_{kSecondValues};
  std::array<std::int16_t, 2U> third_values_{kThirdValues};
};

[[nodiscard]] Native make_native_metadata() {
  Native value{};
  value.sequence = 42U;
  value.label = "heterogeneous three-Tensor frame";
  value.exposure = 2.5F;
  value.header = std::make_unique<app3p::FrameHeaderT>();
  value.header->source = "stereo-camera";
  value.header->frame_id = 17U;
  return value;
}

[[nodiscard]] Native make_native(const TensorFixtureStorage& storage) {
  Native value = make_native_metadata();
  value.first = storage.first;
  value.second = storage.second;
  value.third = storage.third;
  return value;
}

[[nodiscard]] Native make_native(const HostTensorFixtureStorage& storage) {
  Native value = make_native_metadata();
  value.first = storage.first;
  value.second = storage.second;
  value.third = storage.third;
  return value;
}

[[nodiscard]] holoscan::DescriptorTensorTopicContract<Native> external_topic_contract(
    const std::array<holoscan::MemoryKind, 3U>& memory_kinds) {
  const auto contract = [](const holoscan::MemoryKind memory_kind,
                           const DLDataType dtype,
                           const std::uint32_t rank,
                           const holoscan::TensorShapeView shape,
                           const std::uint64_t max_byte_span) {
    return holoscan::TensorBoundaryContract{
        .representation = {.memory_kind = memory_kind,
                           .dtype = dtype,
                           .rank = rank,
                           .strides = holoscan::TensorStridePolicy::kContiguousOnly},
        .bounds = holoscan::tensor_bounds(
            shape, holoscan::TensorBoundsOptions{.max_byte_span = max_byte_span}),
        .transfer =
            holoscan::TensorTransferPolicy{.foreign_emit_policy =
                                               holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded},
    };
  };
  holoscan::DescriptorTensorTopicContract<Native> result;
  result
      .field(Fields::first,
             contract(
                 memory_kinds[0], DLDataType{kDLFloat, 32U, 1U}, 2U, kFirstShape, kFieldBounds[0]))
      .field(
          Fields::second,
          contract(memory_kinds[1], DLDataType{kDLUInt, 8U, 1U}, 1U, kSecondShape, kFieldBounds[1]))
      .field(
          Fields::third,
          contract(memory_kinds[2], DLDataType{kDLInt, 16U, 1U}, 2U, kThirdShape, kFieldBounds[2]));
  return result;
}

[[nodiscard]] holoscan::TopicOptions external_topic_options(const std::string_view endpoint_name,
                                                            const bool imported) {
  holoscan::TopicOptions options;
  options.endpoint_name = endpoint_name;
  if (imported) {
    options.bounds.publisher = holoscan::PublisherEndpointBounds{.max_loaned_samples = 1U};
  } else {
    options.bounds.subscriber =
        holoscan::SubscriberEndpointBounds{.queue_depth = 1U, .max_borrowed_samples = 1U};
  }
  return options;
}

[[nodiscard]] holoscan::expected<std::vector<std::byte>, holoscan::Error> external_topic_metadata(
    const Native& value) {
  const holoscan::detail::RuntimePortTypeOps* const ops =
      holoscan::detail::runtime_port_type_ops<Native>();
  if (ops == nullptr || ops->encode_descriptor_tensor_sample == nullptr) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kProtocolError,
                                                     "descriptor schema codec unavailable"});
  }
  const holoscan::SampleMetadata metadata{
      .sequence = 1U,
      .byte_size = static_cast<std::uint64_t>(value.first->nbytes() + value.second->nbytes() +
                                              value.third->nbytes()),
      .payload_kind = holoscan::PayloadKind::kDescriptorBacked,
  };
  alignas(std::max_align_t) std::array<std::byte, sizeof(ExternalEnvelope)> wire{};
  auto encoded = ops->encode_descriptor_tensor_sample(
      &value, metadata, holoscan::channel_internal::kCompatibilitySchemaEndpointGeneration, wire);
  if (!encoded || *encoded == 0U || *encoded > wire.size()) {
    return holoscan::make_unexpected(
        encoded ? holoscan::Error{holoscan::ErrorCode::kProtocolError,
                                  "descriptor schema metadata size is invalid"}
                : std::move(encoded).error());
  }
  ExternalEnvelope envelope{};
  std::memcpy(&envelope, wire.data(), sizeof(envelope));
  const auto metadata_bytes = envelope.payload.bytes();
  return std::vector<std::byte>{metadata_bytes.begin(), metadata_bytes.end()};
}

[[nodiscard]] auto make_loan_requests() {
  // Deliberately author a valid noncanonical order; Runtime must normalize by Tensor index.
  return std::array{
      holoscan::TensorFieldLoanRequest<Native>{
          .field = Fields::third,
          .tensor =
              holoscan::TensorLoanRequest{
                  .shape = kThirdShape,
                  .dtype = DLDataType{kDLInt, 16U, 1U},
              },
      },
      holoscan::TensorFieldLoanRequest<Native>{
          .field = Fields::first,
          .tensor =
              holoscan::TensorLoanRequest{
                  .shape = kFirstShape,
                  .dtype = DLDataType{kDLFloat, 32U, 1U},
              },
      },
      holoscan::TensorFieldLoanRequest<Native>{
          .field = Fields::second,
          .tensor =
              holoscan::TensorLoanRequest{
                  .shape = kSecondShape,
                  .dtype = DLDataType{kDLUInt, 8U, 1U},
              },
      },
  };
}

[[nodiscard]] auto make_shape_requests() {
  // Deliberately author a valid noncanonical order; Runtime must normalize by Tensor index.
  return std::array{
      holoscan::tensor_shape(Fields::third, kThirdShape),
      holoscan::tensor_shape(Fields::first, kFirstShape),
      holoscan::tensor_shape(Fields::second, kSecondShape),
  };
}

// Emits one heterogeneous generated three-Tensor value using the selected validation or loan mode.
// Representative tests: multi_tensor_schema_contracts, multi_tensor_schema_bundle-cross-process.
template <SourceMode Mode>
class MultiTensorSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr bool kRateAware = is_rate_aware_mode(Mode);
    const holoscan::TensorRepresentation first_layout{
        .memory_kind = holoscan::MemoryKind::kHost,
        .dtype = DLDataType{kDLFloat, 32U, 1U},
        .rank = 2U,
    };
    const holoscan::TensorRepresentation second_layout{
        .memory_kind = kRateAware ? holoscan::MemoryKind::kHost : holoscan::MemoryKind::kCudaDevice,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
        .rank = 1U,
    };
    const holoscan::TensorRepresentation third_layout{
        .memory_kind = kRateAware ? holoscan::MemoryKind::kHost : holoscan::MemoryKind::kPinnedHost,
        .dtype = DLDataType{kDLInt, 16U, 1U},
        .rank = 2U,
    };
    constexpr std::uint64_t first_bound = Mode == SourceMode::kOverBound ? 8U : kFieldBounds[0];
    const holoscan::TensorBounds first_bounds = [&] {
      if constexpr (Mode == SourceMode::kInvalidLoanBundle) {
        return holoscan::tensor_bounds(kFirstShape);
      }
      return holoscan::tensor_bounds(first_bound);
    }();
    const holoscan::TensorBounds third_bounds = [&] {
      if constexpr (Mode == SourceMode::kInvalidLoanBundle) {
        return holoscan::tensor_bounds(
            kThirdShape,
            holoscan::TensorBoundsOptions{.max_abs_byte_strides = kThirdMaxByteStrides});
      }
      return holoscan::tensor_bounds(kFieldBounds[2]);
    }();

    auto& output_builder = spec.output(output, "frames").max_emits_per_compute(1U);
    if constexpr (Mode == SourceMode::kFixedRateFields) {
      output_builder.nominal_rate(holoscan::Rate::hz(60U));
    } else if constexpr (kRateAware) {
      constexpr holoscan::LifecycleStage kBindingStage = rate_binding_stage(Mode);
      output_builder.lifecycle_nominal_rate(kBindingStage)
          .max_nominal_rate(holoscan::Rate::hz(120U));
      spec.lifecycle()
          .stage(kBindingStage, &MultiTensorSource::bind_rate)
          .stage(holoscan::LifecycleStage::kAllocate, &MultiTensorSource::allocate);
    }
    if constexpr (Mode == SourceMode::kMixedCarriage) {
      output_builder.tensor_field(
          Fields::first,
          holoscan::TensorOutputSpec{
              .representation = first_layout,
              .bounds = first_bounds,
              .storage = holoscan::TensorOutputStorage::kRuntimePool,
              .transfer =
                  holoscan::TensorTransferPolicy{
                      .foreign_emit_policy =
                          holoscan::InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded,
                      .max_inline_bytes = 8U,
                      .max_staging_byte_span = first_bound,
                  },
          });
    } else {
      output_builder.tensor_field(
          Fields::first,
          holoscan::TensorOutputSpec{
              .representation = first_layout,
              .bounds = first_bounds,
              .storage = holoscan::TensorOutputStorage::kRuntimePool,
              .transfer =
                  holoscan::TensorTransferPolicy{
                      .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded},
          });
    }
    if constexpr (Mode == SourceMode::kMissingStagingContract) {
      output_builder.tensor_field(Fields::second,
                                  holoscan::TensorOutputSpec{
                                      .representation = second_layout,
                                      .bounds = holoscan::tensor_bounds(kFieldBounds[1]),
                                      .storage = holoscan::TensorOutputStorage::kRuntimePool,
                                  });
    } else {
      output_builder.tensor_field(
          Fields::second,
          holoscan::TensorOutputSpec{
              .representation = second_layout,
              .bounds = holoscan::tensor_bounds(kFieldBounds[1]),
              .storage = holoscan::TensorOutputStorage::kRuntimePool,
              .transfer =
                  holoscan::TensorTransferPolicy{
                      .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded},
          });
    }
    output_builder.tensor_field(
        Fields::third,
        holoscan::TensorOutputSpec{
            .representation = third_layout,
            .bounds = third_bounds,
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
            .transfer =
                holoscan::TensorTransferPolicy{
                    .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded},
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 1ms});
    return result;
  }

  void start() override {
    if constexpr (is_rate_aware_mode(Mode)) {
      host_storage_.initialize();
      initialized_ = host_storage_.first != nullptr && host_storage_.second != nullptr &&
                     host_storage_.third != nullptr;
    } else {
      initialized_ = storage_.initialize();
    }
    ProbeState::source_starts.fetch_add(1U, std::memory_order_acq_rel);
    ProbeState::notify();
  }

  void stop() override {
    if constexpr (!is_rate_aware_mode(Mode)) {
      storage_.release();
    }
  }

  holoscan::LifecycleStatus bind_rate(holoscan::LifecycleContext& context) noexcept {
    const holoscan::Rate selected = is_rate_restart_mode(Mode) && restart_requested_
                                        ? holoscan::Rate::hz(30U)
                                        : holoscan::Rate::hz(60U);
    auto bound = context.bind_nominal_rate(output, selected);
    return bound ? holoscan::LifecycleStatus::kOk : holoscan::LifecycleStatus::kFatalFailure;
  }

  holoscan::LifecycleStatus allocate(holoscan::LifecycleContext&) noexcept {
    ProbeState::record_allocate_pool_state();
    return holoscan::LifecycleStatus::kOk;
  }

  // Keep the compile-time fixture variants and their ordered invalid-bundle probes together;
  // each variant must exercise the same finite pools before its final publication.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (!initialized_) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kNotSupported,
                                                       "CUDA fixture initialization failed"});
    }
    if (completed_) {
      return {};
    }
    if (!ProbeState::source_released.load(std::memory_order_acquire)) {
      return {};
    }
    if constexpr (Mode == SourceMode::kLoanBundleRestart) {
      if (!restart_requested_) {
        auto restart = context.request_restart(
            holoscan::LifecycleRestartOptions{.from_stage = holoscan::LifecycleStage::kStart});
        if (!restart) {
          return holoscan::make_unexpected(std::move(restart.error()));
        }
        restart_requested_ = true;
        return {};
      }
    }
    if constexpr (is_rate_restart_mode(Mode)) {
      if (!restart_requested_) {
        auto restart = context.request_restart(
            holoscan::LifecycleRestartOptions{.from_stage = holoscan::LifecycleStage::kDiscover});
        if (!restart) {
          return holoscan::make_unexpected(std::move(restart.error()));
        }
        restart_requested_ = true;
      }
      return {};
    }
    holoscan::expected<void, holoscan::Error> emitted;
    if constexpr (Mode == SourceMode::kInvalidLoanBundle) {
      auto requests = make_loan_requests();
      const auto missing = output.allocate_tensors(
          std::span<const holoscan::TensorFieldLoanRequest<Native>>{requests}.first(2U));
      requests[2].field = Fields::first;
      const auto duplicate = output.allocate_tensors(requests);
      requests = make_loan_requests();
      requests[2].field = holoscan::TensorFieldSelector<Native>{99U, 2U};
      const auto unknown = output.allocate_tensors(requests);
      requests = make_loan_requests();
      requests[1].tensor.dtype.bits = 0U;
      const auto invalid_dtype = output.allocate_tensors(requests);
      requests = make_loan_requests();
      requests[1].tensor.shape = kInvalidRankShape;
      const auto invalid_rank = output.allocate_tensors(requests);
      requests = make_loan_requests();
      requests[1].tensor.shape = kInvalidExtentShape;
      const auto invalid_shape = output.allocate_tensors(requests);
      requests = make_loan_requests();
      requests[1].tensor.byte_strides =
          holoscan::TensorByteStridesView{std::span<const std::int64_t>{kInvalidByteStrides}};
      const auto invalid_strides = output.allocate_tensors(requests);
      requests = make_loan_requests();
      requests[1].tensor.shape = kOverBoundShape;
      const auto over_bound = output.allocate_tensors(requests);
      requests = make_loan_requests();
      requests[1].tensor.shape = kAxisOverBoundShape;
      const auto axis_over_bound = output.allocate_tensors(requests);
      requests = make_loan_requests();
      requests[1].tensor.shape = kInRangeFirstShape;
      requests[1].tensor.byte_strides =
          holoscan::TensorByteStridesView{std::span<const std::int64_t>{kPaddedFirstByteStrides}};
      const auto noncontiguous = output.allocate_tensors(requests);
      requests = make_loan_requests();
      requests[0].tensor.byte_strides = holoscan::TensorByteStridesView{
          std::span<const std::int64_t>{kOverBoundThirdByteStrides}};
      const auto stride_over_bound = output.allocate_tensors(requests);
      const auto rejected_as_invalid = [](const auto& result, const std::string_view kind) {
        if (!result && result.error().code == holoscan::ErrorCode::kInvalidArgument) {
          return true;
        }
        if (!result) {
          HOLOSCAN_LOG_ERROR("bundle request was not rejected as invalid: {} code={} message={}",
                             kind,
                             holoscan::error_code_name(result.error().code),
                             result.error().message);
        } else {
          HOLOSCAN_LOG_ERROR("bundle request was not rejected as invalid: {}", kind);
        }
        return false;
      };
      // Key assertion: malformed field inventories and Tensor views fail before reserving a
      // bundle, then one valid noncanonical request set can still use the same finite pools.
      if (!rejected_as_invalid(missing, "missing") ||
          !rejected_as_invalid(duplicate, "duplicate") ||
          !rejected_as_invalid(unknown, "unknown") ||
          !rejected_as_invalid(invalid_dtype, "dtype") ||
          !rejected_as_invalid(invalid_rank, "rank") ||
          !rejected_as_invalid(invalid_shape, "shape") ||
          !rejected_as_invalid(invalid_strides, "strides") ||
          !rejected_as_invalid(over_bound, "byte span") ||
          !rejected_as_invalid(axis_over_bound, "axis extent") ||
          !rejected_as_invalid(noncontiguous, "contiguous stride family") ||
          !rejected_as_invalid(stride_over_bound, "per-axis stride")) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kFailure, "invalid bundle request set was admitted"});
      }
      auto failed_callback_bundle = output.allocate_tensors(make_shape_requests());
      if (!failed_callback_bundle) {
        return holoscan::make_unexpected(std::move(failed_callback_bundle).error());
      }
      std::array<std::size_t, 3U> failed_callback_counts{};
      auto callback_failure = failed_callback_bundle->write_all(
          holoscan::host_field(
              Fields::first,
              [&](const holoscan::TensorOutputWriteGuard&) { ++failed_callback_counts[0]; }),
          holoscan::device_field(Fields::second,
                                 context.cuda_stream(),
                                 [&](const holoscan::TensorOutputWriteGuard&)
                                     -> holoscan::expected<void, holoscan::Error> {
                                   ++failed_callback_counts[1];
                                   return holoscan::make_unexpected(holoscan::Error{
                                       holoscan::ErrorCode::kFailure, "injected callback failure"});
                                 }),
          holoscan::host_field(Fields::third, [&](const holoscan::TensorOutputWriteGuard&) {
            ++failed_callback_counts[2];
          }));
      if (callback_failure || callback_failure.error().code != holoscan::ErrorCode::kFailure ||
          failed_callback_counts != std::array<std::size_t, 3U>{1U, 1U, 0U}) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kFailure, "batch-write callback failure was not terminal"});
      }
      auto partial_emit =
          output.emit_tensors(std::move(*failed_callback_bundle), make_native(storage_));
      if (partial_emit || partial_emit.error().code != holoscan::ErrorCode::kInvalidArgument ||
          partial_emit.error().message != "TENSOR_SCHEMA_BUNDLE_FIELD_PREFLIGHT_FAILED") {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kFailure, "partially committed bundle was publishable"});
      }
      auto preflight_bundle = output.allocate_tensors(make_shape_requests());
      if (!preflight_bundle) {
        return holoscan::make_unexpected(std::move(preflight_bundle).error());
      }
      std::size_t callback_count = 0U;
      auto duplicate_write = preflight_bundle->write_all(
          holoscan::host_field(Fields::first,
                               [&](const holoscan::TensorOutputWriteGuard&) { ++callback_count; }),
          holoscan::host_field(Fields::first,
                               [&](const holoscan::TensorOutputWriteGuard&) { ++callback_count; }),
          holoscan::device_field(
              Fields::second, context.cuda_stream(), [&](const holoscan::TensorOutputWriteGuard&) {
                ++callback_count;
              }));
      if (duplicate_write ||
          duplicate_write.error().code != holoscan::ErrorCode::kInvalidArgument ||
          callback_count != 0U) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kFailure, "batch-write preflight invoked a callback"});
      }
      auto written = write_loan_bundle(context, *preflight_bundle);
      if (!written) {
        return holoscan::make_unexpected(std::move(written).error());
      }
      ProbeState::invalid_bundle_rejections.fetch_add(1U, std::memory_order_acq_rel);
      ProbeState::notify();
      emitted = output.emit_tensors(std::move(*preflight_bundle), make_native(storage_));
    } else {
      if constexpr (Mode == SourceMode::kLoanBundle || Mode == SourceMode::kLoanBundleRestart) {
        emitted = emit_loan_bundle(context);
      } else if constexpr (is_rate_aware_mode(Mode)) {
        emitted = output.emit(make_native(host_storage_),
                              holoscan::EmitOptions{.capture_time = context.activation_time()});
      } else {
        emitted = output.emit(make_native(storage_));
      }
    }
    if constexpr (Mode == SourceMode::kOverBound) {
      if (!emitted && emitted.error().code == holoscan::ErrorCode::kInvalidArgument) {
        completed_ = true;
        ProbeState::over_bound_rejections.fetch_add(1U, std::memory_order_acq_rel);
        ProbeState::notify();
        return {};
      }
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure, "over-bound field was not rejected as invalid"});
    }
    if (!emitted && (emitted.error().code == holoscan::ErrorCode::kNotReady ||
                     emitted.error().code == holoscan::ErrorCode::kBackpressured)) {
      return {};
    }
    if (!emitted) {
      HOLOSCAN_LOG_ERROR("multi-Tensor emit failed: {}: {}",
                         holoscan::error_code_name(emitted.error().code),
                         emitted.error().message);
      return holoscan::make_unexpected(std::move(emitted).error());
    }
    completed_ = true;
    ProbeState::emitted.fetch_add(1U, std::memory_order_acq_rel);
    ProbeState::notify();
    return {};
  }

  holoscan::Output<Native> output;

 private:
  [[nodiscard]] holoscan::expected<void, holoscan::Error> write_loan_bundle(
      holoscan::ExecutionContext& context, holoscan::TensorSchemaOutputLoan<Native>& bundle) {
    return bundle.write_all(
        holoscan::host_field(Fields::first,
                             [](const holoscan::TensorOutputWriteGuard& writer) {
                               std::ranges::copy(kFirstValues, writer.data_as<float>());
                             }),
        holoscan::device_field(Fields::second,
                               context.cuda_stream(),
                               [&](const holoscan::TensorOutputWriteGuard& writer)
                                   -> holoscan::expected<void, holoscan::Error> {
                                 if (cudaMemcpyAsync(writer.data(),
                                                     kSecondValues.data(),
                                                     kSecondValues.size(),
                                                     cudaMemcpyHostToDevice,
                                                     context.cuda_stream().get()) != cudaSuccess) {
                                   return holoscan::make_unexpected(
                                       holoscan::Error{holoscan::ErrorCode::kFailure});
                                 }
                                 return {};
                               }),
        holoscan::host_field(Fields::third, [](const holoscan::TensorOutputWriteGuard& writer) {
          std::ranges::copy(kThirdValues, writer.data_as<std::int16_t>());
        }));
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> emit_loan_bundle(
      holoscan::ExecutionContext& context) {
    const auto requests = make_shape_requests();
    auto bundle = output.allocate_tensors(requests);
    if (!bundle) {
      return holoscan::make_unexpected(std::move(bundle.error()));
    }
    auto written = write_loan_bundle(context, *bundle);
    if (!written) {
      return holoscan::make_unexpected(std::move(written).error());
    }
    return output.emit_tensors(std::move(*bundle), make_native(storage_));
  }

  TensorFixtureStorage storage_;
  HostTensorFixtureStorage host_storage_;
  bool initialized_{};
  bool completed_{};
  bool restart_requested_{};
};

[[nodiscard]] bool host_multi_tensor_value_matches(const Native& value);

// Receives and validates the exact metadata, placement, shape, and contents of all three fields.
// Representative tests: multi_tensor_schema_process-local, multi_tensor_schema_bundle-fanout.
template <SourceMode Mode = SourceMode::kValid>
class MultiTensorSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr bool kRateAware = is_rate_aware_mode(Mode);
    auto& input_builder = spec.input(input, "frames");
    if constexpr (!kRateAware) {
      input_builder.queue_depth(kDefaultQueueDepth);
    }
    input_builder.tensor_field(Fields::first)
        .expects_tensor(holoscan::TensorInputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = holoscan::MemoryKind::kHost,
                    .dtype = DLDataType{kDLFloat, 32U, 1U},
                    .rank = 2U,
                },
        });
    input_builder.tensor_field(Fields::second)
        .expects_tensor(holoscan::TensorInputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = kRateAware ? holoscan::MemoryKind::kHost
                                              : holoscan::MemoryKind::kCudaDevice,
                    .dtype = DLDataType{kDLUInt, 8U, 1U},
                    .rank = 1U,
                },
        });
    input_builder.tensor_field(Fields::third)
        .expects_tensor(holoscan::TensorInputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = kRateAware ? holoscan::MemoryKind::kHost
                                              : holoscan::MemoryKind::kPinnedHost,
                    .dtype = DLDataType{kDLInt, 16U, 1U},
                    .rank = 2U,
                },
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    if constexpr (is_rate_aware_mode(Mode)) {
      result.trigger(holoscan::OnClock{.period = 100ms})
          .input(input, holoscan::LatestWithin{.max_age = 1s})
          .on_missing(input, holoscan::Skip{});
    } else {
      result.trigger(holoscan::OnEach{input});
    }
    return result;
  }

  void start() override {
    ProbeState::sink_starts.fetch_add(1U, std::memory_order_acq_rel);
    ProbeState::notify();
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      HOLOSCAN_LOG_ERROR("multi-Tensor sink receive failed: {}: {}",
                         holoscan::error_code_name(sample.error().code),
                         sample.error().message);
      return holoscan::make_unexpected(std::move(sample).error());
    }
    if constexpr (is_rate_aware_mode(Mode)) {
      if (!host_multi_tensor_value_matches(sample->data)) {
        HOLOSCAN_LOG_ERROR("rate-aware host multi-Tensor validation failed");
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, "host multi-Tensor mismatch"});
      }
    } else {
      const Native& value = sample->data;
      const auto dtype_matches = [](const holoscan::Tensor& tensor, const DLDataType expected) {
        const DLDataType actual = tensor.dtype();
        return actual.code == expected.code && actual.bits == expected.bits &&
               actual.lanes == expected.lanes;
      };
      if (value.first == nullptr || value.second == nullptr || value.third == nullptr ||
          value.header == nullptr || value.sequence != 42U ||
          value.label != "heterogeneous three-Tensor frame" || value.exposure != 2.5F ||
          value.header->source != "stereo-camera" || value.header->frame_id != 17U ||
          value.first->device().device_type != kDLCPU ||
          value.second->device().device_type != kDLCUDA ||
          value.second->device().device_id != kDevice.value ||
          value.third->device().device_type != kDLCUDAHost || value.first->ndim() != 2 ||
          value.second->ndim() != 1 || value.third->ndim() != 2 ||
          !std::ranges::equal(value.first->shape_span(), kFirstShape) ||
          !std::ranges::equal(value.second->shape_span(), kSecondShape) ||
          !std::ranges::equal(value.third->shape_span(), kThirdShape) ||
          !dtype_matches(*value.first, DLDataType{kDLFloat, 32U, 1U}) ||
          !dtype_matches(*value.second, DLDataType{kDLUInt, 8U, 1U}) ||
          !dtype_matches(*value.third, DLDataType{kDLInt, 16U, 1U})) {
        HOLOSCAN_LOG_ERROR("multi-Tensor sink metadata validation failed");
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, "multi-Tensor metadata mismatch"});
      }
      auto first_ready = value.first->synchronize_for_host();
      auto third_ready = value.third->synchronize_for_host();
      const auto* first = value.first->data_as<float>();
      const auto* third = value.third->data_as<std::int16_t>();
      std::array<std::uint8_t, 4U> second{};
      auto* const stream = context.cuda_stream().get();
      cudaError_t copy_result = cudaErrorInvalidValue;
      cudaError_t synchronize_result = cudaErrorInvalidValue;
      if (first_ready && third_ready && first != nullptr && third != nullptr && stream != nullptr) {
        copy_result = cudaMemcpyAsync(
            second.data(), value.second->data(), second.size(), cudaMemcpyDeviceToHost, stream);
        if (copy_result == cudaSuccess) {
          synchronize_result = cudaStreamSynchronize(stream);
        }
      }
      if (!first_ready || !third_ready || first == nullptr || third == nullptr ||
          stream == nullptr || copy_result != cudaSuccess || synchronize_result != cudaSuccess ||
          first[0] != 1.0F || first[5] != 6.0F ||
          second != std::array<std::uint8_t, 4U>{3U, 5U, 8U, 13U} || third[0] != 21 ||
          third[1] != 34) {
        HOLOSCAN_LOG_ERROR(
            "multi-Tensor sink data/readiness validation failed: first_ready={} third_ready={} "
            "first={} third={} stream={} device_data={} readiness_contract={} copy={} "
            "synchronize={}",
            static_cast<bool>(first_ready),
            static_cast<bool>(third_ready),
            static_cast<const void*>(first),
            static_cast<const void*>(third),
            static_cast<const void*>(stream),
            value.second->data(),
            value.second->has_readiness_stream_contract(),
            cudaGetErrorName(copy_result),
            cudaGetErrorName(synchronize_result));
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, "multi-Tensor data mismatch"});
      }
    }
    ProbeState::received.fetch_add(1U, std::memory_order_acq_rel);
    ProbeState::notify();
    return {};
  }

  holoscan::Input<Native> input;
};

[[nodiscard]] bool host_multi_tensor_value_matches(const Native& value) {
  const auto dtype_matches = [](const holoscan::Tensor& tensor, const DLDataType expected) {
    const DLDataType actual = tensor.dtype();
    return actual.code == expected.code && actual.bits == expected.bits &&
           actual.lanes == expected.lanes;
  };
  if (value.first == nullptr || value.second == nullptr || value.third == nullptr ||
      value.header == nullptr || value.sequence != 42U ||
      value.label != "heterogeneous three-Tensor frame" || value.exposure != 2.5F ||
      value.header->source != "stereo-camera" || value.header->frame_id != 17U ||
      value.first->device().device_type != kDLCPU || value.second->device().device_type != kDLCPU ||
      value.third->device().device_type != kDLCPU ||
      !dtype_matches(*value.first, DLDataType{kDLFloat, 32U, 1U}) ||
      !dtype_matches(*value.second, DLDataType{kDLUInt, 8U, 1U}) ||
      !dtype_matches(*value.third, DLDataType{kDLInt, 16U, 1U}) ||
      !std::ranges::equal(value.first->shape_span(), kFirstShape) ||
      !std::ranges::equal(value.second->shape_span(), kSecondShape) ||
      !std::ranges::equal(value.third->shape_span(), kThirdShape)) {
    return false;
  }
  const auto first_ready = value.first->synchronize_for_host();
  const auto second_ready = value.second->synchronize_for_host();
  const auto third_ready = value.third->synchronize_for_host();
  const auto* first = value.first->data_as<float>();
  const auto* second = value.second->data_as<std::uint8_t>();
  const auto* third = value.third->data_as<std::int16_t>();
  return first_ready && second_ready && third_ready && first != nullptr && second != nullptr &&
         third != nullptr && std::equal(kFirstValues.begin(), kFirstValues.end(), first) &&
         std::equal(kSecondValues.begin(), kSecondValues.end(), second) &&
         std::equal(kThirdValues.begin(), kThirdValues.end(), third);
}

// Validates that an externally imported three-Tensor schema was materialized in host memory.
// Representative test: multi_tensor_schema_external_import.
class ExternalHostMultiTensorSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    auto& input_builder = spec.input(input, "frames").queue_depth(kDefaultQueueDepth);
    input_builder.tensor_field(Fields::first)
        .expects_tensor(holoscan::TensorInputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = DLDataType{kDLFloat, 32U, 1U},
                               .rank = 2U,
                               .strides = holoscan::TensorStridePolicy::kContiguousOnly},
        });
    input_builder.tensor_field(Fields::second)
        .expects_tensor(holoscan::TensorInputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 1U,
                               .strides = holoscan::TensorStridePolicy::kContiguousOnly},
        });
    input_builder.tensor_field(Fields::third)
        .expects_tensor(holoscan::TensorInputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = DLDataType{kDLInt, 16U, 1U},
                               .rank = 2U,
                               .strides = holoscan::TensorStridePolicy::kContiguousOnly},
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  void start() override {
    ProbeState::sink_starts.fetch_add(1U, std::memory_order_acq_rel);
    ProbeState::notify();
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    if (!host_multi_tensor_value_matches(sample->data)) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure, "external multi-Tensor import data mismatch"});
    }
    ProbeState::received.fetch_add(1U, std::memory_order_acq_rel);
    ProbeState::notify();
    return {};
  }

  holoscan::Input<Native> input;
};

template <holoscan::MemoryKind Kind, typename Builder>
void configure_inline_output(Builder& builder) {
  static_assert(Kind == holoscan::MemoryKind::kHost || Kind == holoscan::MemoryKind::kCudaDevice);
  const auto contract =
      [](const DLDataType dtype, const std::uint32_t rank, const std::uint64_t max_byte_span) {
        return holoscan::TensorOutputSpec{
            .representation = {.memory_kind = Kind, .dtype = dtype, .rank = rank},
            .bounds = holoscan::tensor_bounds(max_byte_span),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
            .transfer =
                holoscan::TensorTransferPolicy{
                    .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded},
        };
      };
  builder.tensor_field(Fields::first, contract(DLDataType{kDLFloat, 32U, 1U}, 2U, kFieldBounds[0]));
  builder.tensor_field(Fields::second, contract(DLDataType{kDLUInt, 8U, 1U}, 1U, kFieldBounds[1]));
  builder.tensor_field(Fields::third, contract(DLDataType{kDLInt, 16U, 1U}, 2U, kFieldBounds[2]));
}

template <holoscan::MemoryKind Kind, typename Builder>
void configure_inline_input(Builder& builder) {
  static_assert(Kind == holoscan::MemoryKind::kHost || Kind == holoscan::MemoryKind::kCudaDevice);
  const auto contract = [](const DLDataType dtype, const std::uint32_t rank) {
    return holoscan::TensorInputSpec{
        .representation = {.memory_kind = Kind, .dtype = dtype, .rank = rank}};
  };
  builder.tensor_field(Fields::first, contract(DLDataType{kDLFloat, 32U, 1U}, 2U));
  builder.tensor_field(Fields::second, contract(DLDataType{kDLUInt, 8U, 1U}, 1U));
  builder.tensor_field(Fields::third, contract(DLDataType{kDLInt, 16U, 1U}, 2U));
}

template <holoscan::MemoryKind Kind, typename Stream>
// The test fixture intentionally keeps the transactional three-field write and validation in one
// helper, and takes the owning value by value so it can move it into the emitted sample.
// NOLINTBEGIN(readability-function-cognitive-complexity, performance-unnecessary-value-param)
[[nodiscard]] holoscan::expected<void, holoscan::Error> emit_inline_bundle(
    holoscan::Output<Native>& output, Native value, Stream stream, const bool relay) {
  static_assert(Kind == holoscan::MemoryKind::kHost || Kind == holoscan::MemoryKind::kCudaDevice);
  const auto requests = make_shape_requests();
  auto bundle = output.allocate_tensors(requests);
  if (!bundle) {
    return holoscan::make_unexpected(std::move(bundle).error());
  }
  holoscan::expected<void, holoscan::Error> first_written;
  holoscan::expected<void, holoscan::Error> second_written;
  holoscan::expected<void, holoscan::Error> third_written;
  if constexpr (Kind == holoscan::MemoryKind::kHost) {
    const auto* first_source =
        relay && value.first != nullptr ? value.first->data_as<float>() : kFirstValues.data();
    const auto* second_source = relay && value.second != nullptr
                                    ? value.second->data_as<std::uint8_t>()
                                    : kSecondValues.data();
    const auto* third_source = relay && value.third != nullptr
                                   ? value.third->data_as<std::int16_t>()
                                   : kThirdValues.data();
    first_written = bundle->write_host_and_commit(
        Fields::first, [&](const holoscan::TensorOutputWriteGuard& writer) {
          std::ranges::copy(std::span{first_source, kFirstValues.size()}, writer.data_as<float>());
        });
    second_written = bundle->write_host_and_commit(
        Fields::second, [&](const holoscan::TensorOutputWriteGuard& writer) {
          std::ranges::copy(std::span{second_source, kSecondValues.size()},
                            writer.data_as<std::uint8_t>());
        });
    third_written = bundle->write_host_and_commit(
        Fields::third, [&](const holoscan::TensorOutputWriteGuard& writer) {
          std::ranges::copy(std::span{third_source, kThirdValues.size()},
                            writer.data_as<std::int16_t>());
        });
  } else {
    if (stream.get() == nullptr ||
        (relay && (value.first == nullptr || value.second == nullptr || value.third == nullptr))) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "inline CUDA stream is unavailable"});
    }
    const auto copy_kind = relay ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    first_written = bundle->write_and_commit(
        Fields::first,
        stream,
        [&](const holoscan::TensorOutputWriteGuard& writer)
            -> holoscan::expected<void, holoscan::Error> {
          const void* source = relay ? value.first->data() : kFirstValues.data();
          return cudaMemcpyAsync(writer.data(),
                                 source,
                                 kFirstValues.size() * sizeof(kFirstValues.front()),
                                 copy_kind,
                                 stream.get()) == cudaSuccess
                     ? holoscan::expected<void, holoscan::Error>{}
                     : holoscan::expected<void, holoscan::Error>{holoscan::make_unexpected(
                           holoscan::Error{holoscan::ErrorCode::kFailure})};
        });
    second_written = bundle->write_and_commit(
        Fields::second,
        stream,
        [&](const holoscan::TensorOutputWriteGuard& writer)
            -> holoscan::expected<void, holoscan::Error> {
          const void* source = relay ? value.second->data() : kSecondValues.data();
          return cudaMemcpyAsync(writer.data(),
                                 source,
                                 kSecondValues.size() * sizeof(kSecondValues.front()),
                                 copy_kind,
                                 stream.get()) == cudaSuccess
                     ? holoscan::expected<void, holoscan::Error>{}
                     : holoscan::expected<void, holoscan::Error>{holoscan::make_unexpected(
                           holoscan::Error{holoscan::ErrorCode::kFailure})};
        });
    third_written = bundle->write_and_commit(
        Fields::third,
        stream,
        [&](const holoscan::TensorOutputWriteGuard& writer)
            -> holoscan::expected<void, holoscan::Error> {
          const void* source = relay ? value.third->data() : kThirdValues.data();
          return cudaMemcpyAsync(writer.data(),
                                 source,
                                 kThirdValues.size() * sizeof(kThirdValues.front()),
                                 copy_kind,
                                 stream.get()) == cudaSuccess
                     ? holoscan::expected<void, holoscan::Error>{}
                     : holoscan::expected<void, holoscan::Error>{holoscan::make_unexpected(
                           holoscan::Error{holoscan::ErrorCode::kFailure})};
        });
  }
  if (!first_written || !second_written || !third_written) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, "inline bundle field write failed"});
  }
  return output.emit_tensors(std::move(*bundle), std::move(value));
}
// NOLINTEND(readability-function-cognitive-complexity, performance-unnecessary-value-param)

// Publishes one generated three-field descriptor schema from uniformly placed field pools.
// Representative test: multi_tensor_schema_inline_partition.
template <holoscan::MemoryKind Kind>
class InlineMultiTensorSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    auto& builder = spec.output(output, "frames").max_emits_per_compute(1U);
    configure_inline_output<Kind>(builder);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 1ns});
    return result;
  }

  void start() override {
    ProbeState::source_starts.fetch_add(1U, std::memory_order_acq_rel);
    ProbeState::notify();
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (next_frame_ == kInlineFrameCount ||
        !ProbeState::source_released.load(std::memory_order_acquire)) {
      return {};
    }
    Native value = make_native_metadata();
    value.sequence = next_frame_;
    value.header->frame_id = next_frame_;
    auto emitted = emit_inline_bundle<Kind>(output, std::move(value), context.cuda_stream(), false);
    if (!emitted) {
      return holoscan::make_unexpected(std::move(emitted).error());
    }
    ProbeState::record_inline_compute(1U, next_frame_, context.tag_time());
    ++next_frame_;
    ProbeState::emitted.fetch_add(1U, std::memory_order_acq_rel);
    ProbeState::notify();
    return {};
  }

  holoscan::Output<Native> output;

 private:
  std::uint64_t next_frame_{};
};

// Relays a generated multi-Tensor value through a second independent set of field pools.
// Representative test: multi_tensor_schema_inline_partition.
template <holoscan::MemoryKind Kind>
class InlineMultiTensorRelay final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    auto& input_builder = spec.input(input, "in").queue_depth(kDefaultQueueDepth);
    configure_inline_input<Kind>(input_builder);
    auto& output_builder = spec.output(output, "out").max_emits_per_compute(1U);
    configure_inline_output<Kind>(output_builder);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    const std::uint64_t frame = sample->data.sequence;
    auto emitted =
        emit_inline_bundle<Kind>(output, std::move(sample->data), context.cuda_stream(), true);
    if (!emitted) {
      return holoscan::make_unexpected(std::move(emitted).error());
    }
    ProbeState::record_inline_compute(2U, frame, context.tag_time());
    return {};
  }

  holoscan::Input<Native> input;
  holoscan::Output<Native> output;
};

// Validates the final uniformly placed fields after the generated schema crosses two inline edges.
// Representative test: multi_tensor_schema_inline_partition.
template <holoscan::MemoryKind Kind>
class InlineMultiTensorSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    auto& builder = spec.input(input, "frames").queue_depth(kDefaultQueueDepth);
    configure_inline_input<Kind>(builder);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  void start() override {
    ProbeState::sink_starts.fetch_add(1U, std::memory_order_acq_rel);
    ProbeState::notify();
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    const Native& value = sample->data;
    std::array<float, kFirstValues.size()> first_values{};
    std::array<std::uint8_t, kSecondValues.size()> second_values{};
    std::array<std::int16_t, kThirdValues.size()> third_values{};
    if (value.first == nullptr || value.second == nullptr || value.third == nullptr) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "inline Tensor field is missing"});
    }
    if constexpr (Kind == holoscan::MemoryKind::kHost) {
      const auto* first = value.first->data_as<float>();
      const auto* second = value.second->data_as<std::uint8_t>();
      const auto* third = value.third->data_as<std::int16_t>();
      if (first == nullptr || second == nullptr || third == nullptr) {
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, "inline host field is unavailable"});
      }
      std::ranges::copy(std::span{first, first_values.size()}, first_values.begin());
      std::ranges::copy(std::span{second, second_values.size()}, second_values.begin());
      std::ranges::copy(std::span{third, third_values.size()}, third_values.begin());
    } else {
      if (context.cuda_stream().get() == nullptr ||
          cudaMemcpyAsync(first_values.data(),
                          value.first->data(),
                          first_values.size() * sizeof(first_values.front()),
                          cudaMemcpyDeviceToHost,
                          context.cuda_stream().get()) != cudaSuccess ||
          cudaMemcpyAsync(second_values.data(),
                          value.second->data(),
                          second_values.size() * sizeof(second_values.front()),
                          cudaMemcpyDeviceToHost,
                          context.cuda_stream().get()) != cudaSuccess ||
          cudaMemcpyAsync(third_values.data(),
                          value.third->data(),
                          third_values.size() * sizeof(third_values.front()),
                          cudaMemcpyDeviceToHost,
                          context.cuda_stream().get()) != cudaSuccess ||
          cudaStreamSynchronize(context.cuda_stream().get()) != cudaSuccess) {
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, "inline CUDA readback failed"});
      }
    }
    constexpr auto expected_device_type = Kind == holoscan::MemoryKind::kHost ? kDLCPU : kDLCUDA;
    const bool device_ids_match = Kind == holoscan::MemoryKind::kHost
                                      ? value.first->device().device_id == 0 &&
                                            value.second->device().device_id == 0 &&
                                            value.third->device().device_id == 0
                                      : value.first->device().device_id == kDevice.value &&
                                            value.second->device().device_id == kDevice.value &&
                                            value.third->device().device_id == kDevice.value;
    const std::uint64_t frame = value.sequence;
    if (value.header == nullptr || frame >= kInlineFrameCount ||
        value.label != "heterogeneous three-Tensor frame" || value.exposure != 2.5F ||
        value.header->source != "stereo-camera" || value.header->frame_id != frame ||
        value.first->device().device_type != expected_device_type ||
        value.second->device().device_type != expected_device_type ||
        value.third->device().device_type != expected_device_type || !device_ids_match ||
        !std::ranges::equal(kFirstValues, first_values) ||
        !std::ranges::equal(kSecondValues, second_values) ||
        !std::ranges::equal(kThirdValues, third_values)) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "inline multi-Tensor data mismatch"});
    }
    ProbeState::record_inline_compute(3U, frame, context.tag_time());
    ProbeState::received.fetch_add(1U, std::memory_order_acq_rel);
    ProbeState::notify();
    return {};
  }

  holoscan::Input<Native> input;
};

// Reports that one isolated consumer worker has started before producer publication is released.
// Representative tests: multi_tensor_schema_cross-process, multi_tensor_schema_fanout.
class ConsumerReadyProbe final : public holoscan::Operator<> {
 public:
  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    if (!reported_) {
      reported_ = true;
      ProbeState::consumer_ready.fetch_add(1U, std::memory_order_acq_rel);
      ProbeState::notify();
    }
    return {};
  }

 private:
  bool reported_{};
};

[[nodiscard]] std::string graph_name(const Topology topology, const SourceMode mode) {
  std::string_view topology_name;
  switch (topology) {
    case Topology::kProcessLocal:
      topology_name = "local";
      break;
    case Topology::kProcessLocalFanout:
      topology_name = "local-fanout";
      break;
    case Topology::kCrossProcess:
      topology_name = "cross";
      break;
    case Topology::kCrossProcessFanout:
      topology_name = "fanout";
      break;
    case Topology::kMixedLocality:
      topology_name = "mixed";
      break;
  }
  if (const char* const instance = std::getenv(kInstanceEnvironment.data()); instance != nullptr) {
    return fmt::format(
        "multi-tensor-schema-{}-{}-{}", topology_name, static_cast<unsigned int>(mode), instance);
  }
  return fmt::format("multi-tensor-schema-{}-{}", topology_name, static_cast<unsigned int>(mode));
}

//   source(MultiTensorSource)
//      +--[frames]--> first-sink(MultiTensorSink)
//      `--[frames, fanout and mixed only]--> second-sink(MultiTensorSink)
//   first-consumer-ready(ConsumerReadyProbe)    (isolated for cross-process topologies)
//   second-consumer-ready(ConsumerReadyProbe)   (fanout and mixed only; isolated when
//                                                cross-process, with the producer when mixed)
//
// Mixed locality connects both sinks with drop-oldest edges.
//
template <SourceMode Mode>
[[nodiscard]] holoscan::ExecutionPlan compile_plan(const Topology topology,
                                                   const std::uint32_t queue_depth) {
  holoscan::Graph graph{graph_name(topology, Mode)};
  const holoscan::ClockKey clock = graph.add_clock<holoscan::RealtimeClock>("runtime-clock");
  graph.set_default_clock(clock);
  auto source = graph.op<MultiTensorSource<Mode>>("source");
  auto first = graph.op<MultiTensorSink<Mode>>("first-sink");
  auto first_ready = graph.op<ConsumerReadyProbe>("first-consumer-ready");
  const holoscan::ConnectionOptions connection{
      .queue_policy = topology == Topology::kMixedLocality ? holoscan::QueuePolicy::kDropOldest
                                                           : holoscan::QueuePolicy::kBlocking,
      .queue_depth = queue_depth};
  if constexpr (is_rate_aware_mode(Mode)) {
    graph.add_flow(source->output, first->input);
  } else {
    graph.add_flow(source->output, first->input, connection);
  }
  std::optional<holoscan::OperatorHandle<MultiTensorSink<Mode>>> second;
  std::optional<holoscan::OperatorHandle<ConsumerReadyProbe>> second_ready;
  if (topology == Topology::kProcessLocalFanout || topology == Topology::kCrossProcessFanout ||
      topology == Topology::kMixedLocality) {
    second = graph.op<MultiTensorSink<Mode>>("second-sink");
    second_ready = graph.op<ConsumerReadyProbe>("second-consumer-ready");
    if constexpr (is_rate_aware_mode(Mode)) {
      graph.add_flow(source->output, (*second)->input);
    } else {
      graph.add_flow(source->output, (*second)->input, connection);
    }
  }
  if constexpr (Mode == SourceMode::kLoanBundleRestart || is_rate_restart_mode(Mode)) {
    graph.lifecycle().allow_self_restart(source);
  }
  graph.partition("producer").add(source);
  graph.partition("first-consumer").add(first);
  graph.partition("first-consumer-ready").add(first_ready);
  if (second.has_value()) {
    graph.partition("second-consumer").add(*second);
    graph.partition("second-consumer-ready").add(*second_ready);
  }

  holoscan::CompileOptions options;
  if constexpr (Mode == SourceMode::kLoanBundleRestart || is_rate_restart_mode(Mode)) {
    options.lifecycle_transactions = holoscan::LifecycleTransactionOptions{
        .max_pending_requests = 4U,
        .max_pending_transactions = 4U,
        .max_receipts = 8U,
        .max_fenced_operators = 4U,
        .max_candidate_generations = 4U,
        .max_events = 64U,
        .max_audit_records = 8U,
        .max_quiesce_records = 8U,
        .max_rollback_records = 8U,
    };
  }
  if constexpr (!is_rate_aware_mode(Mode)) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "source",
        .output_port = "frames",
        .device = kDevice,
    });
  }
  if (topology == Topology::kMixedLocality) {
    holoscan::detail::ProcessPlacementBinding process_placement;
    const std::array producer_selector{
        holoscan::detail::GraphObjectSelector{.graph_path = "source"},
        holoscan::detail::GraphObjectSelector{.graph_path = "second-sink"},
        holoscan::detail::GraphObjectSelector{.graph_path = "second-consumer-ready"},
    };
    const std::array first_selector{
        holoscan::detail::GraphObjectSelector{.graph_path = "first-sink"},
        holoscan::detail::GraphObjectSelector{.graph_path = "first-consumer-ready"},
    };
    process_placement
        .bind_process_role("producer",
                           holoscan::detail::ProcessPlacement{
                               .members = producer_selector,
                               .launch_group = "producer-process",
                           })
        .bind_process_role("first-consumer",
                           holoscan::detail::ProcessPlacement{
                               .members = first_selector,
                               .launch_group = "first-consumer-process",
                           });
    return holoscan::detail::compile_with_process_placement(
        graph, std::move(options), std::move(process_placement));
  }
  if (topology == Topology::kCrossProcess || topology == Topology::kCrossProcessFanout) {
    holoscan::detail::ProcessPlacementBinding process_placement;
    const std::array source_selector{
        holoscan::detail::GraphObjectSelector{.graph_path = "source"},
    };
    const std::array first_selector{
        holoscan::detail::GraphObjectSelector{.graph_path = "first-sink"},
        holoscan::detail::GraphObjectSelector{.graph_path = "first-consumer-ready"},
    };
    process_placement
        .bind_process_role("producer",
                           holoscan::detail::ProcessPlacement{
                               .members = source_selector,
                               .launch_group = "producer-process",
                           })
        .bind_process_role("first-consumer",
                           holoscan::detail::ProcessPlacement{
                               .members = first_selector,
                               .launch_group = "first-consumer-process",
                           });
    if (second.has_value()) {
      const std::array second_selector{
          holoscan::detail::GraphObjectSelector{.graph_path = "second-sink"},
          holoscan::detail::GraphObjectSelector{.graph_path = "second-consumer-ready"},
      };
      process_placement.bind_process_role("second-consumer",
                                          holoscan::detail::ProcessPlacement{
                                              .members = second_selector,
                                              .launch_group = "second-consumer-process",
                                          });
    }
    return holoscan::detail::compile_with_process_placement(
        graph, std::move(options), std::move(process_placement));
  }
  return holoscan::compile(graph, std::move(options));
}

//   source(InlineMultiTensorSource)
//      |
//      v
//   relay(InlineMultiTensorRelay)
//      |
//      v
//   sink(InlineMultiTensorSink)
//
template <holoscan::MemoryKind Kind>
[[nodiscard]] holoscan::ExecutionPlan compile_inline_partition_plan() {
  holoscan::Graph graph{"multi-tensor-schema-inline-partition"};
  const holoscan::ClockKey clock = graph.add_clock<holoscan::RealtimeClock>("runtime-clock");
  graph.set_default_clock(clock);
  auto source = graph.op<InlineMultiTensorSource<Kind>>("source");
  auto relay = graph.op<InlineMultiTensorRelay<Kind>>("relay");
  auto sink = graph.op<InlineMultiTensorSink<Kind>>("sink");
  graph.add_flow(source->output, relay->input, holoscan::ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(relay->output, sink->input, holoscan::ConnectionOptions{.queue_depth = 1U});
  holoscan::CompileOptions options;
  if constexpr (Kind == holoscan::MemoryKind::kCudaDevice) {
    options.deployment
        .bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
            .operator_path = "source",
            .output_port = "frames",
            .device = kDevice,
        })
        .bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
            .operator_path = "relay",
            .output_port = "out",
            .device = kDevice,
        });
  }
  return holoscan::compile(graph, std::move(options));
}

[[nodiscard]] const holoscan::detail::ExecutionPlanState* plan_state(
    const holoscan::ExecutionPlan& plan) {
  return holoscan::detail::ExecutionPlanAccess::state(plan);
}

[[nodiscard]] bool field_contract_matches(
    const holoscan::detail::TensorFieldServiceStorageContract& field, const std::size_t index,
    const bool cross_process) {
  const std::array<DLDataType, 3U> dtypes{
      DLDataType{kDLFloat, 32U, 1U},
      DLDataType{kDLUInt, 8U, 1U},
      DLDataType{kDLInt, 16U, 1U},
  };
  constexpr std::array<std::uint32_t, 3U> ranks{2U, 1U, 2U};
  return field.field_id == kFieldIds[index] && field.tensor_index == index &&
         field.max_byte_span == kFieldBounds[index] && field.dtype.has_value() &&
         field.dtype->code == dtypes[index].code && field.dtype->bits == dtypes[index].bits &&
         field.dtype->lanes == dtypes[index].lanes && field.rank == ranks[index] &&
         field.memory_kind == kFieldMemory[index] &&
         field.device ==
             (index == 1U || (cross_process && index == 2U) ? kDevice : holoscan::DeviceId{}) &&
         field.tensor_emit_policy == holoscan::InterfaceTensorEmitPolicy::kCopyIfNeeded;
}

// Keep the hand-derived field capacity algebra beside its contributor and dependency checks
// so the complete expected inventory can be compared with the accepted physical plan.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool exact_capacity(const holoscan::ExecutionPlan& plan, const Topology topology,
                                  const std::uint32_t queue_depth) {
  if (!plan.ok()) {
    HOLOSCAN_LOG_ERROR("{}", plan.json());
    return false;
  }
  const std::uint32_t consumer_count =
      topology == Topology::kProcessLocalFanout || topology == Topology::kCrossProcessFanout ? 2U
                                                                                             : 1U;
  const bool cross_process =
      topology == Topology::kCrossProcess || topology == Topology::kCrossProcessFanout;
  const bool retained_transport = topology != Topology::kProcessLocal;
  const std::uint32_t producer = 1U;
  const std::uint32_t delivery = queue_depth * consumer_count;
  const std::uint32_t transport_retention = retained_transport ? delivery : 0U;
  // Each transported consumer can also hold one endpoint head, a fixed single slot.
  const std::uint32_t endpoint_heads = retained_transport ? consumer_count : 0U;
  const std::uint32_t retirement = 1U;
  const std::uint32_t slots_per_field =
      producer + delivery + transport_retention + endpoint_heads + retirement;
  const std::uint32_t fixed_contributors_per_field = endpoint_heads != 0U ? 3U : 2U;
  const std::uint32_t dependencies_per_field =
      fixed_contributors_per_field + consumer_count * (retained_transport ? 2U : 1U);
  bool valid = true;
  const auto allocations = plan.tensor_field_allocations();
  const auto contributors = plan.pool_capacity_contributors();
  const auto dependencies = plan.pool_capacity_dependencies();
  const auto providers = plan.provider_allocation_plans();
  const auto queue_sizing_recipes = plan.queue_sizing_recipes();
  const auto payload_edges = plan.payload_edges();
  valid &= check(allocations.size() == kFieldIds.size(),
                 "plan did not retain exactly three field pools");
  valid &= check(providers.size() == kFieldIds.size(),
                 "plan did not retain exactly three field providers");
  std::uint32_t total_slots = 0U;
  for (std::size_t index = 0U; index < allocations.size(); ++index) {
    const auto& allocation = allocations[index];
    valid &= check(
        allocation.field_id == kFieldIds[index] && allocation.tensor_index == index &&
            allocation.bounds.max_byte_span == kFieldBounds[index] &&
            allocation.memory_kind == kFieldMemory[index] &&
            allocation.device ==
                (index == 1U || (cross_process && index == 2U) ? kDevice : holoscan::DeviceId{}) &&
            allocation.slot_count == slots_per_field &&
            allocation.aggregate_byte_capacity == slots_per_field * kFieldBounds[index],
        "field allocation does not match the hand-derived contract");
    total_slots += allocation.slot_count;
    const std::array fixed_contributors{
        std::pair{std::string{".producer"}, producer},
        std::pair{std::string{".retirement"}, retirement},
        std::pair{std::string{".transport-retention"}, endpoint_heads},
    };
    for (const auto& [suffix, maximum] : fixed_contributors) {
      if (maximum == 0U) {
        continue;
      }
      const auto id = allocation.pool_id + suffix;
      const auto found =
          std::find_if(contributors.begin(), contributors.end(), [&](const auto& contributor) {
            return contributor.contributor_id == id;
          });
      valid &= check(found != contributors.end() && found->maximum_distinct_slots == maximum,
                     "fixed field contributor does not match the hand-derived maximum");
    }
    valid &= check(
        std::ranges::count(
            contributors, allocation.pool_id, &holoscan::PoolCapacityContributorRecord::pool_id) ==
            fixed_contributors_per_field,
        "field pool retained a non-static predecessor contributor");

    const auto provider = std::ranges::find(
        providers, allocation.pool_id, &holoscan::ProviderAllocationPlanRecord::pool_id);
    valid &= check(
        provider != providers.end() &&
            provider->allocation_kind == holoscan::ProviderAllocationKind::kTensorFieldOutput &&
            provider->output == allocation.output &&
            provider->tensor_field_id == allocation.field_id &&
            provider->tensor_index == allocation.tensor_index &&
            provider->sizing_dependency_count == dependencies_per_field &&
            provider->first_sizing_dependency &&
            provider->first_sizing_dependency.value + dependencies_per_field - 1U <=
                dependencies.size(),
        "field provider does not name its exact dynamic sizing range");
    if (provider == providers.end() || !provider->first_sizing_dependency ||
        provider->first_sizing_dependency.value + dependencies_per_field - 1U >
            dependencies.size()) {
      continue;
    }
    const auto provider_dependencies =
        dependencies.subspan(static_cast<std::size_t>(provider->first_sizing_dependency.value - 1U),
                             dependencies_per_field);
    std::uint32_t evaluated_slots = 0U;
    std::uint32_t producer_dependencies = 0U;
    std::uint32_t retirement_dependencies = 0U;
    std::uint32_t delivery_dependencies = 0U;
    std::uint32_t transport_dependencies = 0U;
    std::uint32_t head_dependencies = 0U;
    std::uint32_t dependency_offset = 0U;
    for (const auto& dependency : provider_dependencies) {
      valid &= check(
          dependency.id.value == provider->first_sizing_dependency.value + dependency_offset &&
              dependency.pool_id == allocation.pool_id && dependency.scenario.value == 1U &&
              dependency.alternative_group.value != 0U && dependency.mapping_multiplicity == 1U,
          "field sizing dependency lost its canonical pool composition identity");
      ++dependency_offset;
      if (dependency.source == holoscan::PoolCapacityDependencySource::kPredecessorContributor) {
        const auto predecessor =
            std::ranges::find(contributors,
                              dependency.predecessor_contributor_id,
                              &holoscan::PoolCapacityContributorRecord::contributor_id);
        valid &=
            check(predecessor != contributors.end() && predecessor->pool_id == allocation.pool_id &&
                      !dependency.queue_sizing_recipe && dependency.reasons.empty(),
                  "field sizing dependency does not join its fixed predecessor");
        if (predecessor != contributors.end()) {
          evaluated_slots += predecessor->maximum_distinct_slots;
        }
        if (dependency.origin == holoscan::PoolClaimOrigin::kProducerLoan) {
          ++producer_dependencies;
        } else if (dependency.origin == holoscan::PoolClaimOrigin::kRetirement) {
          ++retirement_dependencies;
        } else if (dependency.origin == holoscan::PoolClaimOrigin::kTransportRetention) {
          ++head_dependencies;
        } else {
          valid &= check(false, "field predecessor dependency has an unexpected origin");
        }
        continue;
      }

      valid &=
          check(dependency.source == holoscan::PoolCapacityDependencySource::kQueueSizingRecipe &&
                    dependency.queue_sizing_recipe &&
                    dependency.predecessor_contributor_id.empty() && !dependency.reasons.empty(),
                "field dynamic dependency does not reference one queue-sizing recipe");
      const auto recipe = std::ranges::find(queue_sizing_recipes,
                                            dependency.queue_sizing_recipe,
                                            &holoscan::QueueSizingRecipePlanRecord::id);
      const auto edge =
          recipe == queue_sizing_recipes.end()
              ? payload_edges.end()
              : std::ranges::find(
                    payload_edges, recipe->edge, &holoscan::PayloadEdgePlanRecord::edge);
      valid &=
          check(recipe != queue_sizing_recipes.end() && recipe->explicit_depth == queue_depth &&
                    recipe->geometry_mode == holoscan::AllocationGeometryMode::kCompileConcrete &&
                    recipe->disposition == holoscan::PlanDisposition::kAccepted &&
                    edge != payload_edges.end() && edge->source == allocation.output &&
                    edge->queue_depth == queue_depth,
                "field dynamic dependency does not preserve its edge queue depth");
      if (edge != payload_edges.end()) {
        evaluated_slots += edge->queue_depth;
      }
      if (dependency.origin == holoscan::PoolClaimOrigin::kEdgeDelivery) {
        ++delivery_dependencies;
      } else if (dependency.origin == holoscan::PoolClaimOrigin::kTransportRetention) {
        ++transport_dependencies;
      } else {
        valid &= check(false, "field queue-sizing dependency has an unexpected origin");
      }
    }
    valid &= check(producer_dependencies == 1U && retirement_dependencies == 1U &&
                       delivery_dependencies == consumer_count &&
                       transport_dependencies == (retained_transport ? consumer_count : 0U) &&
                       head_dependencies == (endpoint_heads != 0U ? 1U : 0U) &&
                       evaluated_slots == slots_per_field,
                   "field dependency algebra does not match the hand-derived pool capacity");
  }
  valid &= check(total_slots == slots_per_field * kFieldIds.size(),
                 "aggregate field buffer count does not match the hand derivation");
  valid &= check(contributors.size() == fixed_contributors_per_field * kFieldIds.size(),
                 "plan retained an unexpected fixed capacity contributor");
  valid &= check(dependencies.size() == dependencies_per_field * kFieldIds.size(),
                 "plan retained an unexpected dynamic capacity dependency");

  const auto* state = plan_state(plan);
  valid &= check(state != nullptr, "accepted plan has no private execution state");
  if (state == nullptr) {
    return false;
  }
  const auto validate_services = [&](const auto& services) {
    valid &= check(services.size() == 1U, "plan did not coalesce one expected Tensor service");
    if (services.size() != 1U) {
      return;
    }
    valid &= check(services.front().tensor_fields.size() == kFieldIds.size(),
                   "Tensor service did not retain three field contracts");
    for (std::size_t index = 0U; index < services.front().tensor_fields.size(); ++index) {
      valid &=
          check(field_contract_matches(services.front().tensor_fields[index], index, cross_process),
                "Tensor service field contract mismatch");
    }
    valid &= check(services.front().max_subscribers == consumer_count,
                   "Tensor service fanout bound mismatch");
  };
  if (cross_process) {
    validate_services(state->cross_process_channel_services);
  } else {
    validate_services(state->process_local_channel_services);
  }
  if (!valid) {
    HOLOSCAN_LOG_ERROR("{}", plan.json());
  }
  return valid;
}

[[nodiscard]] bool has_diagnostic(const holoscan::ExecutionPlan& plan,
                                  const std::string_view token) {
  return std::ranges::any_of(plan.diagnostics(),
                             [token](const auto& diagnostic) { return diagnostic.token == token; });
}

// REQ: OPERATOR-036
// REQ: CHAN-123
// REQ: CHAN-153
// REQ: CHAN-126
[[nodiscard]] bool run_contract_validation(const std::uint32_t queue_depth) {
  // Key setup: compile the same generated three-field schema as local, cross-process, and
  // coalesced cross-process fanout plans with literal per-field bounds.
  const auto local = compile_plan<SourceMode::kValid>(Topology::kProcessLocal, queue_depth);
  const auto cross = compile_plan<SourceMode::kValid>(Topology::kCrossProcess, queue_depth);
  const auto fanout = compile_plan<SourceMode::kValid>(Topology::kCrossProcessFanout, queue_depth);
  const auto missing =
      compile_plan<SourceMode::kMissingStagingContract>(Topology::kCrossProcess, queue_depth);
  const auto mixed = compile_plan<SourceMode::kMixedCarriage>(Topology::kCrossProcess, queue_depth);
  if (missing.ok() || !has_diagnostic(missing, "CHANNEL_SCHEMA_TENSOR_FIELD_UNBOUNDED")) {
    HOLOSCAN_LOG_ERROR("{}", missing.json());
  }
  if (mixed.ok() || !has_diagnostic(mixed, "PHYSICAL_TENSOR_INLINE_PROVIDER_UNSUPPORTED")) {
    HOLOSCAN_LOG_ERROR("{}", mixed.json());
  }

  // Key assertion: fixed predecessor leaves plus queue-derived delivery/transport dependencies
  // compose the exact capacity per field, while incomplete staging and mixed descriptor/inline
  // carriage fail closed.
  return exact_capacity(local, Topology::kProcessLocal, queue_depth) &&
         exact_capacity(cross, Topology::kCrossProcess, queue_depth) &&
         exact_capacity(fanout, Topology::kCrossProcessFanout, queue_depth) &&
         check(!missing.ok() && has_diagnostic(missing, "CHANNEL_SCHEMA_TENSOR_FIELD_UNBOUNDED"),
               "missing per-field staging contract was admitted") &&
         check(!mixed.ok() && has_diagnostic(mixed, "PHYSICAL_TENSOR_INLINE_PROVIDER_UNSUPPORTED"),
               "mixed descriptor/inline carriage was admitted");
}

// REQ: COMPILE-069
// REQ: LIFECYCLE-012
// Keep startup evidence, per-field pool checks, and failure diagnostics beside the common
// session cleanup for all selected lifecycle-rate variants.
template <SourceMode Mode = SourceMode::kRateAwareFields>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool run_rate_aware_fields() {
  static_assert(Mode == SourceMode::kFixedRateFields || Mode == SourceMode::kRateAwareFields ||
                Mode == SourceMode::kRateAwareFieldsConfigure);
  // Key setup: compile and run three finite-maximum field pools selected at their lifecycle stage.
  ProbeState::reset();
  const holoscan::ExecutionPlan compiled =
      compile_plan<Mode>(Topology::kProcessLocal, kDefaultQueueDepth);
  const auto* state = plan_state(compiled);
  if (!compiled.ok() || state == nullptr || state->tensor_field_allocations.size() != 3U ||
      state->provider_allocation_plans.size() != 3U ||
      !std::ranges::all_of(state->provider_allocation_plans, [](const auto& provider) {
        return provider.allocation_kind == holoscan::ProviderAllocationKind::kTensorFieldOutput &&
               provider.geometry_mode == holoscan::AllocationGeometryMode::kCompileConcrete &&
               provider.disposition == holoscan::PlanDisposition::kAccepted;
      })) {
    HOLOSCAN_LOG_ERROR("{}", compiled.json());
    return check(false, "finite-maximum field providers were not compile concrete");
  }
  const auto control = std::make_shared<holoscan::detail::RuntimeTensorOutputTestControl>();
  ProbeState::install_tensor_control(control);
  const holoscan::ExecutionPlan plan =
      holoscan::detail::RuntimeTestPlanAccess::with_tensor_output_test_control(compiled, control);
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool started = ProbeState::wait_for(ProbeState::source_starts, 1U);
  const bool sink_started = ProbeState::wait_for(ProbeState::sink_starts, 1U);
  const auto generations = wait_for_published_rate_generation(session);
  const bool published = generations && generations->published() != nullptr;
  // Wait until both endpoints and the lifecycle-selected generation are visible before releasing
  // the single sample. This turns the rate-aware scenarios into an end-to-end field-carriage
  // check without racing startup or weakening their finite-horizon rate contract.
  ProbeState::source_released.store(started && sink_started && published,
                                    std::memory_order_release);
  ProbeState::notify();
  const bool delivered = ProbeState::wait_for(ProbeState::emitted, 1U) &&
                         ProbeState::wait_for(ProbeState::received, 1U);
  session.request_stop();
  session.wait();

  bool valid = started && sink_started && published && delivered &&
               session.termination() == holoscan::RunTermination::kStopped &&
               session.diagnostics().empty();
  if constexpr (Mode != SourceMode::kFixedRateFields) {
    valid = valid && ProbeState::allocate_callbacks.load(std::memory_order_acquire) == 1U &&
            ProbeState::allocate_saw_preallocated_fields.load(std::memory_order_acquire);
  }
  if (!valid) {
    HOLOSCAN_LOG_ERROR(
        "rate-aware field publication mismatch: source_started={} sink_started={} snapshot={} "
        "published={} emitted={} received={} termination={} diagnostics={}",
        started,
        sink_started,
        static_cast<bool>(generations),
        generations && generations->published() != nullptr,
        ProbeState::emitted.load(std::memory_order_acquire),
        ProbeState::received.load(std::memory_order_acquire),
        static_cast<std::uint32_t>(session.termination()),
        session.diagnostics().size());
    if (!generations) {
      HOLOSCAN_LOG_ERROR("rate-aware field snapshot failed: code={} message={}",
                         static_cast<std::uint32_t>(generations.error().code),
                         generations.error().message);
    }
  }
  if (valid) {
    const auto providers = generations->provider_allocations(*generations->published());
    valid = providers.size() == 3U;
    if (!valid) {
      HOLOSCAN_LOG_ERROR("rate-aware field provider count mismatch: actual={} expected=3",
                         providers.size());
    }
    for (std::size_t index = 0U; valid && index < providers.size(); ++index) {
      const auto metrics = control->snapshot(index);
      const bool field_valid =
          providers[index].recipe == state->provider_allocation_plans[index].id &&
          providers[index].action == holoscan::AllocationAction::kFresh &&
          providers[index].allocation_generation.has_value() &&
          providers[index].storage_binding_generation.has_value() &&
          providers[index].provider_materialized_entries == metrics.physical_slot_count &&
          metrics.materialization_count == 1U && metrics.maximum_allocation_generation == 1U &&
          metrics.physical_slot_count >= metrics.active_entry_limit &&
          metrics.allocation_action == holoscan::detail::RuntimeTensorPoolAllocationAction::kFresh;
      if (!field_valid) {
        HOLOSCAN_LOG_ERROR(
            "rate-aware field {} mismatch: provider={} expected_provider={} action={} "
            "materialized={} physical={} active={} pool_action={}",
            index,
            providers[index].recipe.value,
            state->provider_allocation_plans[index].id.value,
            static_cast<std::uint32_t>(providers[index].action),
            providers[index].provider_materialized_entries.value_or(0U),
            metrics.physical_slot_count,
            metrics.active_entry_limit,
            static_cast<std::uint32_t>(metrics.allocation_action));
      }
      valid = field_valid;
    }
  }
  if (!valid) {
    print_diagnostics(session);
  }
  // Key assertion: every field materializes once with Fresh evidence and the selected limit.
  return check(valid,
               Mode == SourceMode::kFixedRateFields
                   ? "fixed-rate field pools did not publish startup Fresh evidence"
                   : "finite-maximum lifecycle field pools did not publish selected active limits");
}

// REQ: COMPILE-069
// REQ: LIFECYCLE-012
template <SourceMode Mode>
[[nodiscard]] bool run_rate_aware_fields_late_binding() {
  static_assert(Mode == SourceMode::kRateAwareFieldsWarmUp ||
                Mode == SourceMode::kRateAwareFieldsArm);
  // Key setup: select a smaller active rate during a late WarmUp or Arm lifecycle stage.
  ProbeState::reset();
  const holoscan::ExecutionPlan compiled =
      compile_plan<Mode>(Topology::kProcessLocal, kDefaultQueueDepth);
  const auto* state = plan_state(compiled);
  if (!compiled.ok() || state == nullptr || state->tensor_field_allocations.size() != 3U ||
      state->provider_allocation_plans.size() != 3U ||
      !std::ranges::all_of(state->provider_allocation_plans, [](const auto& provider) {
        return provider.allocation_kind == holoscan::ProviderAllocationKind::kTensorFieldOutput &&
               provider.geometry_mode == holoscan::AllocationGeometryMode::kCompileConcrete &&
               provider.disposition == holoscan::PlanDisposition::kAccepted;
      })) {
    HOLOSCAN_LOG_ERROR("{}", compiled.json());
    return check(false, "late-stage field providers were not compile concrete");
  }
  const auto control = std::make_shared<holoscan::detail::RuntimeTensorOutputTestControl>();
  ProbeState::install_tensor_control(control);
  const holoscan::ExecutionPlan plan =
      holoscan::detail::RuntimeTestPlanAccess::with_tensor_output_test_control(compiled, control);
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool started = ProbeState::wait_for(ProbeState::source_starts, 1U);
  const auto generations = wait_for_published_rate_generation(session);
  session.request_stop();
  session.wait();

  bool valid = started && generations && generations->published() != nullptr &&
               ProbeState::allocate_callbacks.load(std::memory_order_acquire) == 1U &&
               ProbeState::allocate_saw_preallocated_fields.load(std::memory_order_acquire) &&
               session.termination() == holoscan::RunTermination::kStopped &&
               session.diagnostics().empty();
  for (std::size_t index = 0U; valid && index < 3U; ++index) {
    const auto metrics = control->snapshot(index);
    valid =
        metrics.materialization_count == 1U && metrics.maximum_allocation_generation == 1U &&
        metrics.physical_slot_count != 0U && metrics.active_entry_limit != 0U &&
        metrics.active_entry_limit < metrics.physical_slot_count &&
        metrics.allocation_action == holoscan::detail::RuntimeTensorPoolAllocationAction::kFresh;
  }
  if (!valid) {
    print_diagnostics(session);
  }
  // Key assertion: late selection changes only active limits, not physical field backing.
  return check(valid,
               "late-stage lifecycle rate changed field backing instead of its active limit");
}

// REQ: COMPILE-069
// REQ: LIFECYCLE-012
[[nodiscard]] bool run_rate_aware_fields_restart() {
  // Key setup: restart three finite-maximum field pools with a smaller second generation.
  ProbeState::reset();
  const holoscan::ExecutionPlan compiled = compile_plan<SourceMode::kRateAwareFieldsRestart>(
      Topology::kProcessLocal, kDefaultQueueDepth);
  const auto* state = plan_state(compiled);
  if (!compiled.ok() || state == nullptr || state->tensor_field_allocations.size() != 3U ||
      state->provider_allocation_plans.size() != 3U) {
    HOLOSCAN_LOG_ERROR("{}", compiled.json());
    return check(false, "restart field providers were not compile concrete");
  }
  const auto control = std::make_shared<holoscan::detail::RuntimeTensorOutputTestControl>();
  const holoscan::ExecutionPlan plan =
      holoscan::detail::RuntimeTestPlanAccess::with_tensor_output_test_control(compiled, control);
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool first_started = ProbeState::wait_for(ProbeState::source_starts, 1U);
  const auto before = wait_for_published_rate_generation(session);
  std::array<holoscan::detail::RuntimeTensorOutputTestControl::PoolSnapshot, 3U> before_metrics{};
  for (std::size_t index = 0U; index < before_metrics.size(); ++index) {
    before_metrics[index] = control->snapshot(index);
  }
  const holoscan::RateGenerationId first_generation =
      before && before->published() != nullptr ? before->published()->rate_generation : 0U;

  ProbeState::source_released.store(first_started && first_generation != 0U,
                                    std::memory_order_release);
  ProbeState::notify();
  const bool second_started = ProbeState::wait_for(ProbeState::source_starts, 2U);
  const auto after = wait_for_published_rate_generation(session, first_generation);
  session.request_stop();
  session.wait();

  bool valid = first_started && second_started && before && after &&
               before->published() != nullptr && after->published() != nullptr &&
               session.termination() == holoscan::RunTermination::kStopped &&
               session.diagnostics().empty();
  if (valid) {
    const auto before_providers = before->provider_allocations(*before->published());
    const auto after_providers = after->provider_allocations(*after->published());
    valid = before_providers.size() == 3U && after_providers.size() == 3U &&
            before->published()->rate_generation != after->published()->rate_generation;
    for (const auto& prior : before_providers) {
      const auto current = std::ranges::find(
          after_providers, prior.recipe, &holoscan::ProviderAllocationGenerationRecord::recipe);
      if (!valid || current == after_providers.end() || !prior.allocation_generation ||
          !current->allocation_generation || !prior.storage_binding_generation ||
          !current->storage_binding_generation ||
          prior.action != holoscan::AllocationAction::kFresh ||
          current->action != holoscan::AllocationAction::kReusedLarger ||
          prior.allocation_generation != current->allocation_generation ||
          prior.storage_binding_generation == current->storage_binding_generation) {
        valid = false;
        break;
      }
      const std::size_t index = prior.recipe.value - 1U;
      if (index >= before_metrics.size()) {
        valid = false;
        break;
      }
      const auto current_metrics = control->snapshot(index);
      valid = before_metrics[index].materialization_count == 1U &&
              current_metrics.materialization_count == 1U &&
              current_metrics.maximum_allocation_generation ==
                  before_metrics[index].maximum_allocation_generation &&
              current_metrics.physical_slot_count == before_metrics[index].physical_slot_count &&
              current_metrics.active_entry_limit > 0U &&
              current_metrics.active_entry_limit < before_metrics[index].active_entry_limit &&
              before_metrics[index].active_entry_limit < current_metrics.physical_slot_count &&
              current_metrics.allocation_action ==
                  holoscan::detail::RuntimeTensorPoolAllocationAction::kReusedLarger;
      if (!valid) {
        break;
      }
    }
  }
  if (!valid) {
    print_diagnostics(session);
  }
  // Key assertion: restart atomically reuses allocations while publishing new storage bindings.
  return check(valid,
               "finite-maximum lifecycle field pools did not restart with atomic retained reuse");
}

// REQ: COMPILE-069
// REQ: LIFECYCLE-012
// REQ: RUNTIME-008
// REQ: TENSOR-STORAGE-011
// Keep the before/after generation and pool evidence together to prove that a failed final
// field preflight preserves every earlier field without partially publishing the restart.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool run_rate_aware_fields_restart_reject() {
  // Key setup: inject a final-field preflight failure during a three-field pool restart.
  ProbeState::reset();
  const holoscan::ExecutionPlan compiled = compile_plan<SourceMode::kRateAwareFieldsRestartReject>(
      Topology::kProcessLocal, kDefaultQueueDepth);
  const auto* state = plan_state(compiled);
  if (!compiled.ok() || state == nullptr || state->tensor_field_allocations.size() != 3U ||
      state->provider_allocation_plans.size() != 3U) {
    HOLOSCAN_LOG_ERROR("{}", compiled.json());
    return check(false, "restart-rejection field providers were not compile concrete");
  }
  const auto control = std::make_shared<holoscan::detail::RuntimeTensorOutputTestControl>();
  const holoscan::ExecutionPlan plan =
      holoscan::detail::RuntimeTestPlanAccess::with_tensor_output_test_control(compiled, control);
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool first_started = ProbeState::wait_for(ProbeState::source_starts, 1U);
  const auto before = wait_for_published_rate_generation(session);
  std::array<holoscan::detail::RuntimeTensorOutputTestControl::PoolSnapshot, 3U> before_metrics{};
  for (std::size_t index = 0U; index < before_metrics.size(); ++index) {
    before_metrics[index] = control->snapshot(index);
  }
  const holoscan::RateGenerationId first_generation =
      before && before->published() != nullptr ? before->published()->rate_generation : 0U;
  if (!first_started || first_generation == 0U) {
    session.request_stop();
    session.wait();
    print_diagnostics(session);
    return check(false, "restart-rejection setup did not publish the initial rate generation");
  }

  const std::size_t final_field_pool_index = state->tensor_allocations.size() + 2U;
  control->arm_fault_for_pool(
      holoscan::detail::RuntimeTensorOutputTestFault::kRestartPublicationPreflightFailure,
      final_field_pool_index);
  ProbeState::source_released.store(true, std::memory_order_release);
  ProbeState::notify();
  session.wait();
  const auto after = session.capture_rate_generations();

  bool valid = first_started && before && before->published() != nullptr && after &&
               after->published() != nullptr && after->last_rejected() != nullptr &&
               first_generation != 0U && after->published()->rate_generation == first_generation &&
               after->last_rejected()->rate_generation != first_generation &&
               ProbeState::source_starts.load(std::memory_order_acquire) == 2U &&
               session.termination() == holoscan::RunTermination::kFailed &&
               !session.diagnostics().empty();
  std::size_t rejected_provider_count{};
  if (valid) {
    const auto rejected_providers = after->provider_allocations(*after->last_rejected());
    rejected_provider_count = rejected_providers.size();
    valid = rejected_providers.size() == 3U &&
            std::ranges::all_of(rejected_providers, [](const auto& provider) {
              return provider.action == holoscan::AllocationAction::kRejected &&
                     provider.disposition == holoscan::PlanDisposition::kRejected &&
                     provider.previous_allocation_generation.has_value() &&
                     provider.provider_materialized_entries.has_value();
            });
  }
  for (std::size_t index = 0U; valid && index < before_metrics.size(); ++index) {
    const auto current = control->snapshot(index);
    valid = current.materialization_count == before_metrics[index].materialization_count &&
            current.maximum_allocation_generation ==
                before_metrics[index].maximum_allocation_generation &&
            current.maximum_published_pool_generation ==
                before_metrics[index].maximum_published_pool_generation &&
            current.physical_slot_count == before_metrics[index].physical_slot_count &&
            current.active_entry_limit == before_metrics[index].active_entry_limit &&
            current.allocation_action == before_metrics[index].allocation_action;
  }
  if (!valid) {
    HOLOSCAN_LOG_ERROR(
        "rate-field rejection mismatch: first_started={} before={} after={} published={} "
        "rejected={} first_generation={} published_generation={} rejected_generation={} "
        "starts={} termination={} diagnostics={} rejected_providers={}",
        first_started,
        static_cast<bool>(before),
        static_cast<bool>(after),
        after && after->published() != nullptr,
        after && after->last_rejected() != nullptr,
        first_generation,
        after && after->published() != nullptr ? after->published()->rate_generation : 0U,
        after && after->last_rejected() != nullptr ? after->last_rejected()->rate_generation : 0U,
        ProbeState::source_starts.load(std::memory_order_acquire),
        static_cast<std::uint32_t>(session.termination()),
        session.diagnostics().size(),
        rejected_provider_count);
    for (std::size_t index = 0U; index < before_metrics.size(); ++index) {
      const auto current = control->snapshot(index);
      HOLOSCAN_LOG_ERROR(
          "rate-field {} before(materialized={}, allocation={}, published={}, physical={}, "
          "active={}, action={}) after(materialized={}, allocation={}, published={}, physical={}, "
          "active={}, action={})",
          index,
          before_metrics[index].materialization_count,
          before_metrics[index].maximum_allocation_generation,
          before_metrics[index].maximum_published_pool_generation,
          before_metrics[index].physical_slot_count,
          before_metrics[index].active_entry_limit,
          static_cast<std::uint32_t>(before_metrics[index].allocation_action),
          current.materialization_count,
          current.maximum_allocation_generation,
          current.maximum_published_pool_generation,
          current.physical_slot_count,
          current.active_entry_limit,
          static_cast<std::uint32_t>(current.allocation_action));
    }
    if (after && after->last_rejected() != nullptr) {
      for (const auto& provider : after->provider_allocations(*after->last_rejected())) {
        HOLOSCAN_LOG_ERROR(
            "rejected provider {} action={} disposition={} previous_allocation={} allocation={} "
            "materialized={}",
            provider.recipe.value,
            static_cast<std::uint32_t>(provider.action),
            static_cast<std::uint32_t>(provider.disposition),
            provider.previous_allocation_generation ? provider.previous_allocation_generation->value
                                                    : 0U,
            provider.allocation_generation ? provider.allocation_generation->value : 0U,
            provider.provider_materialized_entries.value_or(0U));
      }
    }
    print_diagnostics(session);
  }
  // Key assertion: the rejected generation publishes no partial field or allocation changes.
  return check(valid, "failed finite-maximum field restart partially published one bundle member");
}

// REQ: CHAN-153
// REQ: CHAN-126
// REQ: CHAN-127
[[nodiscard]] bool run_process_local() {
  if (!cuda_available()) {
    return true;
  }
  ProbeState::reset();
  // Key setup: execute one heterogeneous generated value over direct process-local retention.
  const auto plan = compile_plan<SourceMode::kValid>(Topology::kProcessLocal, kDefaultQueueDepth);
  if (!exact_capacity(plan, Topology::kProcessLocal, kDefaultQueueDepth)) {
    return false;
  }
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool started = ProbeState::wait_for(ProbeState::source_starts, 1U);
  ProbeState::source_released.store(started, std::memory_order_release);
  ProbeState::notify();
  const bool completed = started && ProbeState::wait_for(ProbeState::received, 1U) &&
                         ProbeState::emitted.load(std::memory_order_acquire) == 1U;
  session.request_stop();
  session.wait();
  // Key assertion: all three Tensor fields and the non-Tensor header arrive exactly once.
  const bool valid = completed && session.termination() == holoscan::RunTermination::kStopped &&
                     session.diagnostics().empty();
  if (!valid) {
    HOLOSCAN_LOG_ERROR("probe source_starts={} sink_starts={} emitted={} received={}",
                       ProbeState::source_starts.load(),
                       ProbeState::sink_starts.load(),
                       ProbeState::emitted.load(),
                       ProbeState::received.load());
    print_diagnostics(session);
  }
  return check(valid, "process-local multi-Tensor execution failed");
}

// REQ: CHAN-153
// REQ: CHAN-126
// REQ: CHAN-127
[[nodiscard]] bool run_process_local_fanout() {
  if (!cuda_available()) {
    return true;
  }
  ProbeState::reset();
  // Key setup: publish one Runtime-pooled heterogeneous multi-Tensor value to two process-local
  // subscribers through the coalesced HoloMQ service.
  constexpr auto topology = Topology::kProcessLocalFanout;
  const auto plan = compile_plan<SourceMode::kLoanBundle>(topology, kDefaultQueueDepth);
  if (!exact_capacity(plan, topology, kDefaultQueueDepth)) {
    return false;
  }
  const auto* state = plan_state(plan);
  if (state == nullptr || state->process_local_channel_services.size() != 1U ||
      state->process_local_channel_services.front().delivery_kind !=
          holoscan::detail::ProcessLocalChannelDeliveryKind::kHoloMQ) {
    HOLOSCAN_LOG_ERROR("{}", plan.json());
    return check(false, "process-local multi-Tensor fanout did not select HoloMQ");
  }
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool started = ProbeState::wait_for(ProbeState::source_starts, 1U);
  ProbeState::source_released.store(started, std::memory_order_release);
  ProbeState::notify();
  const bool completed = started && ProbeState::wait_for(ProbeState::emitted, 1U) &&
                         ProbeState::wait_for(ProbeState::received, 2U);
  session.request_stop();
  session.wait();
  // Key assertion: both subscribers receive and validate all three independently pooled fields.
  const bool valid = completed && ProbeState::received.load(std::memory_order_acquire) == 2U &&
                     session.termination() == holoscan::RunTermination::kStopped &&
                     session.diagnostics().empty();
  if (!valid) {
    HOLOSCAN_LOG_ERROR("probe source_starts={} sink_starts={} emitted={} received={}",
                       ProbeState::source_starts.load(),
                       ProbeState::sink_starts.load(),
                       ProbeState::emitted.load(),
                       ProbeState::received.load());
    print_diagnostics(session);
  }
  return check(valid, "process-local multi-Tensor fanout execution failed");
}

// REQ: RUNTIME-041
template <holoscan::MemoryKind Kind>
[[nodiscard]] bool run_inline_partition() {
  if constexpr (Kind == holoscan::MemoryKind::kCudaDevice) {
    if (!cuda_available()) {
      return true;
    }
  }
  ProbeState::reset();
  // Key setup: compile a three-Operator chain using an application-owned generated schema with
  // three independently pooled Tensor fields on each internal output.
  const auto plan = compile_inline_partition_plan<Kind>();
  const auto* state = plan_state(plan);
  const bool device_resources_match =
      Kind == holoscan::MemoryKind::kHost
          ? state != nullptr && state->tensor_device_resources.empty() &&
                state->cuda_compute_streams.empty()
          : state != nullptr && state->tensor_device_resources.size() == 1U &&
                state->tensor_device_resources.front().device == kDevice &&
                state->cuda_compute_streams.size() == 1U &&
                state->cuda_compute_streams.front().fused_linear_edges.size() == 2U;
  if (!plan.ok() || state == nullptr || !state->inline_partition_execution ||
      state->inline_partition_execution->edges.size() != 2U ||
      state->tensor_field_allocations.size() != 6U || !device_resources_match) {
    HOLOSCAN_LOG_ERROR("{}", plan.json());
    return check(false, "multi-Tensor schema chain was not admitted for inline execution");
  }
  const auto counters = std::make_shared<holoscan::detail::RuntimePhaseCounters>();
  const holoscan::ExecutionPlan run_plan =
      holoscan::detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  if (!run_plan.ok()) {
    return check(false, "multi-Tensor schema phase counters could not be installed");
  }
  holoscan::RunSession session = holoscan::run_async(run_plan);
  const bool started = ProbeState::wait_for(ProbeState::source_starts, 1U) &&
                       ProbeState::wait_for(ProbeState::sink_starts, 1U);
  ProbeState::source_released.store(started, std::memory_order_release);
  ProbeState::notify();
  const bool completed = started && ProbeState::wait_for(ProbeState::received, kInlineFrameCount);
  session.request_stop();
  session.wait();
  const auto records = ProbeState::inline_compute_snapshot();
  bool causal_order = records.size() == kInlineFrameCount * 3U;
  for (std::size_t frame = 0U; causal_order && frame < kInlineFrameCount; ++frame) {
    const std::size_t source_index = frame * 3U;
    const holoscan::TagTime source_tag = records[source_index].tag_time;
    causal_order = source_tag.valid() &&
                   source_tag.microstep <= std::numeric_limits<std::uint32_t>::max() - 2U;
    for (std::size_t stage = 0U; causal_order && stage < 3U; ++stage) {
      const InlineComputeRecord& record = records[source_index + stage];
      causal_order =
          record.operator_id == stage + 1U && record.frame == frame &&
          record.tag_time.logical_ns == source_tag.logical_ns &&
          record.tag_time.microstep == source_tag.microstep + static_cast<std::uint32_t>(stage);
    }
  }
  const auto phase_snapshot = counters->snapshot();
  const auto phase_count = [&phase_snapshot](const holoscan::detail::RuntimePhaseCounter counter) {
    return phase_snapshot[holoscan::detail::RuntimePhaseCounters::index(counter)];
  };
  const std::uint64_t expected_receipts = 2U * kInlineFrameCount;
  const bool continuation_evidence =
      phase_count(holoscan::detail::RuntimePhaseCounter::kDirectContinuationReceiptsAccepted) ==
          expected_receipts &&
      phase_count(holoscan::detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts) ==
          expected_receipts &&
      phase_count(holoscan::detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted) ==
          expected_receipts &&
      phase_count(holoscan::detail::RuntimePhaseCounter::kPartitionCausalFrontierFallbacks) == 0U;
  // Key assertion: every generated multi-Tensor frame traverses the complete three-Operator chain
  // before the next clock frame starts, without a worker-lane fallback. Each logical schema
  // publication advances readiness and earns exactly one continuation receipt, regardless of its
  // three independently pooled Tensor fields.
  const bool valid = completed && causal_order && continuation_evidence &&
                     ProbeState::emitted.load(std::memory_order_acquire) == kInlineFrameCount &&
                     ProbeState::received.load(std::memory_order_acquire) == kInlineFrameCount &&
                     session.termination() == holoscan::RunTermination::kStopped &&
                     session.diagnostics().empty();
  if (!valid) {
    HOLOSCAN_LOG_ERROR(
        "inline multi-Tensor evidence emitted={} received={} records={} receipts={} attempts={} "
        "accepted={} fallbacks={}",
        ProbeState::emitted.load(),
        ProbeState::received.load(),
        records.size(),
        phase_count(holoscan::detail::RuntimePhaseCounter::kDirectContinuationReceiptsAccepted),
        phase_count(holoscan::detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts),
        phase_count(holoscan::detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted),
        phase_count(holoscan::detail::RuntimePhaseCounter::kPartitionCausalFrontierFallbacks));
    print_diagnostics(session);
  }
  return check(valid, "inline multi-Tensor schema execution failed");
}

[[nodiscard]] bool run_over_bound() {
  if (!cuda_available()) {
    return true;
  }
  ProbeState::reset();
  // Key setup: compile a valid field contract whose first 24-byte Tensor exceeds its 8-byte bound.
  const auto plan =
      compile_plan<SourceMode::kOverBound>(Topology::kProcessLocal, kDefaultQueueDepth);
  if (!plan.ok()) {
    HOLOSCAN_LOG_ERROR("{}", plan.json());
    return false;
  }
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool started = ProbeState::wait_for(ProbeState::source_starts, 1U);
  ProbeState::source_released.store(started, std::memory_order_release);
  ProbeState::notify();
  const bool rejected = started && ProbeState::wait_for(ProbeState::over_bound_rejections, 1U);
  session.request_stop();
  session.wait();
  // Key assertion: publication returns kInvalidArgument before any consumer receives the value.
  return check(rejected && ProbeState::received.load(std::memory_order_acquire) == 0U &&
                   session.termination() == holoscan::RunTermination::kStopped &&
                   session.diagnostics().empty(),
               "over-bound multi-Tensor field did not fail closed");
}

// REQ: OPERATOR-044
// REQ: CHAN-127
[[nodiscard]] bool run_invalid_bundle() {
  if (!cuda_available()) {
    return true;
  }
  ProbeState::reset();
  // Key setup: issue malformed inventory and Tensor-view transactions against active finite pools.
  const auto plan =
      compile_plan<SourceMode::kInvalidLoanBundle>(Topology::kProcessLocal, kDefaultQueueDepth);
  if (!plan.ok()) {
    HOLOSCAN_LOG_ERROR("{}", plan.json());
    return false;
  }
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool started = ProbeState::wait_for(ProbeState::source_starts, 1U);
  ProbeState::source_released.store(started, std::memory_order_release);
  ProbeState::notify();
  const bool rejected = started && ProbeState::wait_for(ProbeState::invalid_bundle_rejections, 1U);
  const bool recovered = rejected && ProbeState::wait_for(ProbeState::received, 1U);
  session.request_stop();
  session.wait();
  // Key assertion: every malformed set fails without consuming capacity, then a valid bundle
  // reaches the sink from the same pools.
  return check(recovered && ProbeState::emitted.load(std::memory_order_acquire) == 1U &&
                   ProbeState::received.load(std::memory_order_acquire) == 1U &&
                   session.termination() == holoscan::RunTermination::kStopped &&
                   session.diagnostics().empty(),
               "invalid multi-Tensor bundle requests did not recover capacity");
}

// REQ: OPERATOR-037
// REQ: CHAN-127
[[nodiscard]] bool run_bundle_restart() {
  if (!cuda_available()) {
    return true;
  }
  ProbeState::reset();
  // Key setup: restart the source from Start before its first three-field bundle publication.
  const auto plan =
      compile_plan<SourceMode::kLoanBundleRestart>(Topology::kProcessLocal, kDefaultQueueDepth);
  if (!plan.ok()) {
    HOLOSCAN_LOG_ERROR("{}", plan.json());
    return false;
  }
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool first_start = ProbeState::wait_for(ProbeState::source_starts, 1U);
  ProbeState::source_released.store(first_start, std::memory_order_release);
  ProbeState::notify();
  const bool restarted = first_start && ProbeState::wait_for(ProbeState::source_starts, 2U);
  const bool received = restarted && ProbeState::wait_for(ProbeState::received, 1U);
  session.request_stop();
  session.wait();
  // Key assertion: every field pool rebinds under the new generation and one complete bundle
  // reaches the sink without a stale child binding or partial publication.
  return check(received && ProbeState::emitted.load(std::memory_order_acquire) == 1U &&
                   ProbeState::received.load(std::memory_order_acquire) == 1U &&
                   session.termination() == holoscan::RunTermination::kStopped &&
                   session.diagnostics().empty(),
               "multi-Tensor bundle did not recover across source restart");
}

#if defined(__linux__)
[[nodiscard]] bool write_byte(const int descriptor, const char value) {
  return write(descriptor, &value, sizeof(value)) == sizeof(value);
}

template <SourceMode Mode>
[[nodiscard]] bool run_consumer_worker(const int result_descriptor, const Topology topology,
                                       const std::size_t target_index) {
  ProbeState::reset();
  // Key setup: reconstruct the full plan after exec and materialize only this consumer group.
  const auto plan = compile_plan<Mode>(topology, kDefaultQueueDepth);
  const auto* state = plan_state(plan);
  bool valid = plan.ok() && state != nullptr && state->cross_process_channel_services.size() == 1U;
  if (!valid) {
    static_cast<void>(write_byte(result_descriptor, 0));
    return false;
  }
  const auto& service = state->cross_process_channel_services.front();
  valid = target_index < service.target_process_groups.size();
  if (!valid) {
    static_cast<void>(write_byte(result_descriptor, 0));
    return false;
  }
  holoscan::RunSession consumer =
      holoscan::detail::run_async_worker(plan, service.target_process_groups[target_index]);
  const bool started = ProbeState::wait_for(ProbeState::sink_starts, 1U) &&
                       ProbeState::wait_for(ProbeState::consumer_ready, 1U) &&
                       ProbeState::source_starts.load(std::memory_order_acquire) == 0U;
  const bool ready_written = write_byte(result_descriptor, started ? 1 : 0);
  const bool received = started && ready_written && ProbeState::wait_for(ProbeState::received, 1U);
  consumer.request_stop();
  consumer.wait();
  // Key assertion: this isolated consumer imports every field and no producer Operator.
  const bool child_valid = received && ProbeState::received.load(std::memory_order_acquire) == 1U &&
                           ProbeState::source_starts.load(std::memory_order_acquire) == 0U &&
                           consumer.termination() == holoscan::RunTermination::kStopped &&
                           consumer.diagnostics().empty();
  if (!child_valid) {
    print_diagnostics(consumer);
    HOLOSCAN_LOG_ERROR("{}", plan.json());
  }
  return write_byte(result_descriptor, child_valid ? 1 : 0) && child_valid;
}

[[nodiscard]] bool read_byte(const int descriptor, char& value) {
  return read(descriptor, &value, sizeof(value)) == sizeof(value);
}

[[nodiscard]] bool reap_child(const pid_t child) {
  int status = 0;
  pid_t waited = -1;
  do {
    waited = waitpid(child, &status, 0);
  } while (waited < 0 && errno == EINTR);
  return waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

template <SourceMode Mode>
// Process orchestration is intentionally linear so each fork/exec cleanup path remains visible.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool run_cross_process(const bool fanout) {
  if (!cuda_available()) {
    return true;
  }
  const std::string instance = fmt::format("{}", static_cast<std::uint64_t>(getpid()));
  if (setenv(kInstanceEnvironment.data(), instance.c_str(), 1) != 0) {
    return check(false, "failed to set multi-Tensor service instance");
  }
  ProbeState::reset();
  const Topology topology = fanout ? Topology::kCrossProcessFanout : Topology::kCrossProcess;
  // Key setup: exec one independently compiled consumer per fanout arm before starting producer.
  const auto plan = compile_plan<Mode>(topology, kDefaultQueueDepth);
  if (!exact_capacity(plan, topology, kDefaultQueueDepth)) {
    return false;
  }
  const auto* state = plan_state(plan);
  if (state == nullptr || state->cross_process_channel_services.size() != 1U) {
    return check(false, "cross-process plan has no unique multi-Tensor service");
  }
  const auto& service = state->cross_process_channel_services.front();
  const std::size_t consumer_count = fanout ? 2U : 1U;
  if (service.target_process_groups.size() != consumer_count) {
    return check(false, "cross-process plan has incorrect consumer-group geometry");
  }

  std::array<std::array<int, 2U>, 2U> pipes{};
  std::array<pid_t, 2U> children{};
  bool valid = true;
  for (std::size_t index = 0U; index < consumer_count; ++index) {
    if (pipe(pipes[index].data()) != 0) {
      valid = false;
      break;
    }
    const pid_t child = fork();
    if (child < 0) {
      valid = false;
      break;
    }
    if (child == 0) {
      close(pipes[index][0]);
      const std::string descriptor = fmt::format("{}", pipes[index][1]);
      const std::string target = fmt::format("{}", index);
      execl("/proc/self/exe",
            "multi_tensor_schema_runtime_validation",
            "consumer",
            descriptor.c_str(),
            fanout ? "fanout" : "cross",
            target.c_str(),
            Mode == SourceMode::kLoanBundle ? "bundle" : "owning",
            static_cast<char*>(nullptr));
      std::_Exit(127);
    }
    children[index] = child;
    close(pipes[index][1]);
  }
  for (std::size_t index = 0U; valid && index < consumer_count; ++index) {
    char ready = 0;
    valid &= read_byte(pipes[index][0], ready) && ready == 1;
  }

  if (valid) {
    holoscan::RunSession producer =
        holoscan::detail::run_async_worker(plan, service.source_process_group);
    const bool started = ProbeState::wait_for(ProbeState::source_starts, 1U);
    // The source gate keeps its first publication behind producer-endpoint creation. Allow the
    // already-running consumer endpoints to complete discovery before releasing that one
    // non-retryable fanout publication.
    std::this_thread::sleep_for(300ms);
    ProbeState::source_released.store(started, std::memory_order_release);
    ProbeState::notify();
    const bool emitted = started && ProbeState::wait_for(ProbeState::emitted, 1U) &&
                         ProbeState::sink_starts.load(std::memory_order_acquire) == 0U;
    for (std::size_t index = 0U; index < consumer_count; ++index) {
      char result = 0;
      valid &= read_byte(pipes[index][0], result) && result == 1;
    }
    producer.request_stop();
    producer.wait();
    // Key assertion: one producer publication is imported exactly once by every process arm.
    valid &= emitted && producer.termination() == holoscan::RunTermination::kStopped &&
             producer.diagnostics().empty();
  }
  for (std::size_t index = 0U; index < consumer_count; ++index) {
    if (pipes[index][0] != 0) {
      close(pipes[index][0]);
    }
    if (children[index] > 0) {
      valid &= reap_child(children[index]);
    }
  }
  return check(valid, "cross-process multi-Tensor execution failed");
}

// Owning sources emit caller-owned fields, which the cross-process leg copies into staging; loan
// bundles emit field-pool loans, which it exports as they are.
// REQ: CHAN-170
template <SourceMode Mode>
// Process orchestration is intentionally linear so each fork/exec cleanup path remains visible.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool run_mixed_locality() {
  if (!cuda_available()) {
    return true;
  }
  const std::string instance = fmt::format("{}", static_cast<std::uint64_t>(getpid()));
  if (setenv(kInstanceEnvironment.data(), instance.c_str(), 1) != 0) {
    return check(false, "failed to set multi-Tensor service instance");
  }
  ProbeState::reset();
  const auto plan = compile_plan<Mode>(Topology::kMixedLocality, kDefaultQueueDepth);
  const auto* state = plan_state(plan);
  if (!plan.ok() || state == nullptr) {
    HOLOSCAN_LOG_ERROR("{}", plan.json());
    return check(false, "mixed-locality multi-Tensor plan did not compile");
  }
  // Key assertion: the local sink keeps one direct service and the remote sink one cross-process
  // service, each carrying all three fields, the pinned-host one with its synchronization device.
  const auto local =
      std::ranges::find_if(state->process_local_channel_services, [](const auto& candidate) {
        return candidate.delivery_kind ==
               holoscan::detail::ProcessLocalChannelDeliveryKind::kDirectRetained;
      });
  bool valid = local != state->process_local_channel_services.end() &&
               state->process_local_channel_services.size() == 1U &&
               state->cross_process_channel_services.size() == 1U &&
               local->tensor_fields.size() == 3U &&
               state->cross_process_channel_services.front().tensor_fields.size() == 3U;
  for (std::size_t index = 0U; valid && index < 3U; ++index) {
    valid = field_contract_matches(local->tensor_fields[index], index, true) &&
            field_contract_matches(
                state->cross_process_channel_services.front().tensor_fields[index], index, true);
  }
  if (!valid) {
    HOLOSCAN_LOG_ERROR("{}", plan.json());
    return check(false, "mixed-locality multi-Tensor plan has unexpected services");
  }
  const auto& service = state->cross_process_channel_services.front();
  if (service.target_process_groups.size() != 1U) {
    return check(false, "mixed-locality plan has incorrect consumer-group geometry");
  }

  // Key setup: exec the remote consumer before starting the producer group, which also runs the
  // local sink.
  std::array<int, 2U> pipe_descriptors{};
  if (pipe(pipe_descriptors.data()) != 0) {
    return check(false, "failed to create the mixed-locality consumer pipe");
  }
  const pid_t child = fork();
  if (child < 0) {
    close(pipe_descriptors[0]);
    close(pipe_descriptors[1]);
    return check(false, "failed to fork the mixed-locality consumer");
  }
  if (child == 0) {
    close(pipe_descriptors[0]);
    const std::string descriptor = fmt::format("{}", pipe_descriptors[1]);
    execl("/proc/self/exe",
          "multi_tensor_schema_runtime_validation",
          "consumer",
          descriptor.c_str(),
          "mixed",
          "0",
          Mode == SourceMode::kLoanBundle ? "bundle" : "owning",
          static_cast<char*>(nullptr));
    std::_Exit(127);
  }
  close(pipe_descriptors[1]);
  char ready = 0;
  valid = read_byte(pipe_descriptors[0], ready) && ready == 1;
  if (valid) {
    holoscan::RunSession producer =
        holoscan::detail::run_async_worker(plan, service.source_process_group);
    const bool started = ProbeState::wait_for(ProbeState::source_starts, 1U);
    // Let the remote consumer's endpoint finish discovery before the one publication.
    std::this_thread::sleep_for(300ms);
    ProbeState::source_released.store(started, std::memory_order_release);
    ProbeState::notify();
    const bool emitted = started && ProbeState::wait_for(ProbeState::emitted, 1U);
    const bool local_received = emitted && ProbeState::wait_for(ProbeState::received, 1U);
    char result = 0;
    valid = read_byte(pipe_descriptors[0], result) && result == 1;
    producer.request_stop();
    producer.wait();
    if (!producer.diagnostics().empty()) {
      print_diagnostics(producer);
    }
    // Key assertion: one publication reaches the local sink in this process and the remote sink
    // in the consumer process, each with every field.
    valid &= local_received && ProbeState::received.load(std::memory_order_acquire) == 1U &&
             producer.termination() == holoscan::RunTermination::kStopped &&
             producer.diagnostics().empty();
  }
  close(pipe_descriptors[0]);
  valid &= reap_child(child);
  return check(valid, "mixed-locality multi-Tensor execution failed");
}
#else
template <SourceMode Mode>
[[nodiscard]] bool run_cross_process(bool) {
  // Key setup: cross-address-space process APIs are unavailable on this platform.
  // Key assertion: report the platform-specific scenario as skipped successfully.
  return true;
}

template <SourceMode Mode>
[[nodiscard]] bool run_mixed_locality() {
  // Key setup: cross-address-space process APIs are unavailable on this platform.
  // Key assertion: report the platform-specific scenario as skipped successfully.
  return true;
}
#endif

template <holoscan::ChannelDescriptorFlatBufferPayload Payload>
[[nodiscard]] holoscan::expected<void, holoscan::Error> wait_for_schema_subscriber(
    holoscan::ChannelTensorSchemaPublisher<Payload>& publisher,
    const std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto refreshed = publisher.update_connections();
    if (!refreshed) {
      return holoscan::make_unexpected(std::move(refreshed).error());
    }
    if (publisher.has_subscribers()) {
      return {};
    }
    std::this_thread::sleep_for(1ms);
  }
  return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kTimeout});
}

template <holoscan::ChannelDescriptorFlatBufferPayload Payload>
[[nodiscard]] holoscan::expected<void, holoscan::Error> wait_for_schema_publisher(
    holoscan::ChannelTensorSchemaSubscriber<Payload>& subscriber,
    const std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto refreshed = subscriber.has_samples();
    if (!refreshed) {
      return holoscan::make_unexpected(std::move(refreshed).error());
    }
    if (subscriber.has_publishers()) {
      return {};
    }
    std::this_thread::sleep_for(1ms);
  }
  return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kTimeout});
}

template <holoscan::ChannelDescriptorFlatBufferPayload Payload>
[[nodiscard]] holoscan::expected<holoscan::ChannelResolvedTensorBundle<Payload>, holoscan::Error>
wait_for_schema_bundle(holoscan::ChannelTensorSchemaSubscriber<Payload>& subscriber,
                       const std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto ready = subscriber.has_samples();
    if (!ready) {
      return holoscan::make_unexpected(std::move(ready).error());
    }
    if (*ready) {
      auto sample = subscriber.receive_tensors();
      if (sample || sample.error().code != holoscan::ErrorCode::kNotReady) {
        return sample;
      }
    }
    std::this_thread::sleep_for(1ms);
  }
  return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kTimeout});
}

[[nodiscard]] std::string external_topic_name(const std::string_view suffix) {
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
#if defined(__linux__)
  return fmt::format("multi-tensor-external-{}-{}-{}", suffix, getpid(), nonce);
#else
  return fmt::format("multi-tensor-external-{}-{}", suffix, nonce);
#endif
}

[[nodiscard]] bool external_mixed_bundle_matches(
    const holoscan::ChannelResolvedTensorBundle<ExternalPayload>& bundle) {
  const Root* root = bundle.root();
  if (root == nullptr || root->label() == nullptr || root->header() == nullptr ||
      root->header()->source() == nullptr || root->first() == nullptr ||
      root->second() == nullptr || root->third() == nullptr || root->sequence() != 42U ||
      root->label()->string_view() != "heterogeneous three-Tensor frame" ||
      root->exposure() != 2.5F || root->header()->source()->string_view() != "stereo-camera" ||
      root->header()->frame_id() != 17U || bundle.tensors[0].device().device_type != kDLCPU ||
      bundle.tensors[1].device().device_type != kDLCUDA ||
      bundle.tensors[1].device().device_id != kDevice.value ||
      bundle.tensors[2].device().device_type != kDLCUDAHost ||
      !std::ranges::equal(bundle.tensors[0].shape_span(), kFirstShape) ||
      !std::ranges::equal(bundle.tensors[1].shape_span(), kSecondShape) ||
      !std::ranges::equal(bundle.tensors[2].shape_span(), kThirdShape) ||
      std::ranges::any_of(bundle.payloads, [](const auto& payload) {
        return payload.payload_kind != holoscan::ChannelPayloadKind::kBufferHandle;
      })) {
    return false;
  }
  const auto first_ready = bundle.tensors[0].synchronize_for_host();
  const auto second_ready = bundle.tensors[1].synchronize_for_host();
  const auto third_ready = bundle.tensors[2].synchronize_for_host();
  const auto* first = bundle.tensors[0].data_as<float>();
  const auto* third = bundle.tensors[2].data_as<std::int16_t>();
  std::array<std::uint8_t, 4U> second{};
  const cudaError_t copied =
      second_ready
          ? cudaMemcpy(
                second.data(), bundle.tensors[1].data(), second.size(), cudaMemcpyDeviceToHost)
          : cudaErrorInvalidValue;
  return first_ready && third_ready && first != nullptr && third != nullptr &&
         copied == cudaSuccess && std::equal(kFirstValues.begin(), kFirstValues.end(), first) &&
         second == kSecondValues && std::equal(kThirdValues.begin(), kThirdValues.end(), third);
}

// REQ: EXTERNAL-010
[[nodiscard]] holoscan::ExecutionPlan compile_external_topic_import_plan(
    const std::string_view endpoint_name, const bool fanout) {
  constexpr std::array kHostFields{
      holoscan::MemoryKind::kHost,
      holoscan::MemoryKind::kHost,
      holoscan::MemoryKind::kHost,
  };
  // [imported topic] -> [sink]
  //          \---- fanout --> [relay] -> [first sink]
  //                                    \-> [second sink]
  holoscan::Graph graph{fanout ? "multi-tensor-schema-external-import-fanout"
                               : "multi-tensor-schema-external-import"};
  const holoscan::ClockKey clock = graph.add_clock<holoscan::RealtimeClock>("runtime-clock");
  graph.set_default_clock(clock);
  auto imported = graph.import_topic<Native>(fmt::format("holomq://{}", endpoint_name),
                                             external_topic_options(endpoint_name, true),
                                             external_topic_contract(kHostFields));
  if (!fanout) {
    auto sink = graph.op<ExternalHostMultiTensorSink>("sink");
    graph.add_flow(imported.output(), sink->input, holoscan::ConnectionOptions{.queue_depth = 1U});
  } else {
    auto relay = graph.op<InlineMultiTensorRelay<holoscan::MemoryKind::kHost>>("relay");
    auto first = graph.op<ExternalHostMultiTensorSink>("first-sink");
    auto second = graph.op<ExternalHostMultiTensorSink>("second-sink");
    graph.add_flow(imported.output(), relay->input, holoscan::ConnectionOptions{.queue_depth = 1U});
    graph.add_flow(relay->output, first->input, holoscan::ConnectionOptions{.queue_depth = 1U});
    graph.add_flow(relay->output, second->input, holoscan::ConnectionOptions{.queue_depth = 1U});
    graph.partition("external-import-fanout").add(relay).add(first).add(second);
  }
  return holoscan::compile(graph);
}

//   [source] -> [exported topic]
//      `--[mixed locality only]--> [sink]
//
// Mixed locality connects the export and the local sink with drop-oldest edges.
[[nodiscard]] holoscan::ExecutionPlan compile_external_topic_export_plan(
    const std::string_view endpoint_name, const bool mixed_locality = false) {
  holoscan::Graph graph{"multi-tensor-schema-external-export"};
  const holoscan::ClockKey clock = graph.add_clock<holoscan::RealtimeClock>("runtime-clock");
  graph.set_default_clock(clock);
  auto exported = graph.export_topic<Native>(fmt::format("holomq://{}", endpoint_name),
                                             external_topic_options(endpoint_name, false),
                                             external_topic_contract(kFieldMemory));
  auto source = graph.op<MultiTensorSource<SourceMode::kValid>>("source");
  const holoscan::ConnectionOptions connection{
      .queue_policy =
          mixed_locality ? holoscan::QueuePolicy::kDropOldest : holoscan::QueuePolicy::kBlocking,
      .queue_depth = 1U};
  graph.add_flow(source->output, exported.input(), connection);
  if (mixed_locality) {
    auto sink = graph.op<MultiTensorSink<SourceMode::kValid>>("sink");
    graph.add_flow(source->output, sink->input, connection);
    graph.partition("source").add(source);
    graph.partition("sink").add(sink);
  }
  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "source",
      .output_port = "frames",
      .device = kDevice,
  });
  return holoscan::compile(graph, std::move(options));
}

[[nodiscard]] bool run_external_topic_import(const bool fanout) {
  ProbeState::reset();
  // Key setup: import the generated three-Tensor schema so activation must consume one
  // pre-reserved Tensor read lease per descriptor field and retained slot.
  const std::string endpoint_name = external_topic_name(fanout ? "import-fanout" : "import");
  const holoscan::ExecutionPlan plan = compile_external_topic_import_plan(endpoint_name, fanout);
  if (!plan.ok()) {
    HOLOSCAN_LOG_ERROR("{}", plan.json());
    return false;
  }

  auto external_runtime = holoscan::ChannelRuntime::create();
  if (!external_runtime) {
    return check(false, "external descriptor-schema Channel Runtime creation failed");
  }
  auto publisher = external_runtime->create_tensor_schema_publisher<ExternalPayload>(
      plan.endpoint_catalog(), endpoint_name);
  if (!publisher) {
    HOLOSCAN_LOG_ERROR("external multi-Tensor publisher failed: {}", publisher.error().message);
    return false;
  }
  holoscan::RunSession session = holoscan::run_async(plan);
  const auto connected = wait_for_schema_subscriber(*publisher);
  HostTensorFixtureStorage storage;
  storage.initialize();
  const Native value = make_native(storage);
  auto metadata = external_topic_metadata(value);
  const std::array tensors{*storage.first, *storage.second, *storage.third};
  auto emitted = metadata ? publisher->emit(*metadata, tensors)
                          : holoscan::expected<std::uint32_t, holoscan::Error>{
                                holoscan::make_unexpected(std::move(metadata).error())};
  const std::size_t expected_receives = fanout ? 2U : 1U;
  const bool received =
      emitted && *emitted == 1U && ProbeState::wait_for(ProbeState::received, expected_receives);
  session.request_stop();
  session.wait();
  if (!session.diagnostics().empty()) {
    print_diagnostics(session);
  }
  // Key assertion: first activation reaches every sink with all three fields; Runtime's bounded
  // access guard would reject the activation if construction reserved only one lease per slot.
  return check(connected && received &&
                   session.termination() == holoscan::RunTermination::kStopped &&
                   session.diagnostics().empty(),
               "external multi-Tensor schema import failed");
}

// With mixed locality a local sink beside the export keeps a direct service, and one publication
// reaches it and the external subscriber.
// REQ: CHAN-170
[[nodiscard]] bool run_external_topic_export(const bool mixed_locality = false) {
  if (!cuda_available()) {
    return true;
  }
  // Key setup: export one mixed-memory three-Tensor schema to an external local subscriber.
  ProbeState::reset();
  const std::string endpoint_name =
      external_topic_name(mixed_locality ? "export-mixed-locality" : "export");
  const holoscan::ExecutionPlan plan =
      compile_external_topic_export_plan(endpoint_name, mixed_locality);
  if (!plan.ok()) {
    HOLOSCAN_LOG_ERROR("{}", plan.json());
    return false;
  }
  if (mixed_locality) {
    // Key assertion: the local sink keeps one direct service with all three fields.
    const auto* state = plan_state(plan);
    const bool direct = state != nullptr && state->process_local_channel_services.size() == 1U &&
                        state->process_local_channel_services.front().delivery_kind ==
                            holoscan::detail::ProcessLocalChannelDeliveryKind::kDirectRetained &&
                        state->process_local_channel_services.front().tensor_fields.size() == 3U;
    if (!direct) {
      HOLOSCAN_LOG_ERROR("{}", plan.json());
      return check(false, "mixed-locality export plan has no direct local service");
    }
  }

  auto external_runtime = holoscan::ChannelRuntime::create();
  if (!external_runtime) {
    return check(false, "external descriptor-schema Channel Runtime creation failed");
  }
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool source_started = ProbeState::wait_for(ProbeState::source_starts, 1U);
  auto subscriber = external_runtime->create_tensor_schema_subscriber<ExternalPayload>(
      plan.endpoint_catalog(), endpoint_name);
  const auto connected = subscriber ? wait_for_schema_publisher(*subscriber)
                                    : holoscan::expected<void, holoscan::Error>{
                                          holoscan::make_unexpected(subscriber.error())};
  ProbeState::source_released.store(source_started && connected.has_value(),
                                    std::memory_order_release);
  ProbeState::notify();
  const bool emitted = ProbeState::wait_for(ProbeState::emitted, 1U);
  auto received = subscriber && emitted
                      ? wait_for_schema_bundle(*subscriber)
                      : holoscan::expected<holoscan::ChannelResolvedTensorBundle<ExternalPayload>,
                                           holoscan::Error>{holoscan::make_unexpected(
                            holoscan::Error{holoscan::ErrorCode::kNotReady})};
  const bool valid_bundle = received && external_mixed_bundle_matches(*received);
  const bool local_received =
      !mixed_locality || (emitted && ProbeState::wait_for(ProbeState::received, 1U));
  session.request_stop();
  session.wait();
  if (!session.diagnostics().empty()) {
    print_diagnostics(session);
  }
  // Key assertion: the subscriber resolves and validates every exported Tensor field exactly once,
  // and with mixed locality the local sink validates them too.
  return check(source_started && connected && emitted && valid_bundle && local_received &&
                   session.termination() == holoscan::RunTermination::kStopped &&
                   session.diagnostics().empty(),
               "external multi-Tensor schema export failed");
}

#if defined(__linux__)
[[nodiscard]] bool run_external_topic_export_subscriber_worker(
    const int result_descriptor, const std::string_view endpoint_name) {
  // Key setup: create the isolated external subscriber and report each readiness phase by pipe.
  const bool cuda_ready = cuda_available() && cudaSetDevice(kDevice.value) == cudaSuccess;
  const holoscan::ExecutionPlan plan = compile_external_topic_export_plan(endpoint_name);
  auto external_runtime = holoscan::ChannelRuntime::create();
  auto subscriber =
      cuda_ready && plan.ok() && external_runtime
          ? external_runtime->create_tensor_schema_subscriber<ExternalPayload>(
                plan.endpoint_catalog(), endpoint_name)
          : holoscan::expected<holoscan::ChannelTensorSchemaSubscriber<ExternalPayload>,
                               holoscan::Error>{
                holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure})};
  const bool ready_written = write_byte(result_descriptor, subscriber ? 1 : 0);
  const bool ready = subscriber.has_value() && ready_written;
  const auto connected =
      ready ? wait_for_schema_publisher(*subscriber)
            : holoscan::expected<void, holoscan::Error>{
                  holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kNotReady})};
  const bool connected_written = write_byte(result_descriptor, connected ? 1 : 0);
  auto received = connected && connected_written
                      ? wait_for_schema_bundle(*subscriber)
                      : holoscan::expected<holoscan::ChannelResolvedTensorBundle<ExternalPayload>,
                                           holoscan::Error>{holoscan::make_unexpected(
                            holoscan::Error{holoscan::ErrorCode::kNotReady})};
  const bool valid = received && external_mixed_bundle_matches(*received);
  // Key assertion: report success only after the complete external Tensor bundle is validated.
  return write_byte(result_descriptor, valid ? 1 : 0) && valid;
}

[[nodiscard]] bool run_external_topic_export_cross_process() {
  if (!cuda_available()) {
    return true;
  }
  // Key setup: fork an external subscriber before starting the mixed-memory export graph.
  ProbeState::reset();
  const std::string endpoint_name = external_topic_name("export-cross-process");
  const holoscan::ExecutionPlan plan = compile_external_topic_export_plan(endpoint_name);
  if (!plan.ok()) {
    HOLOSCAN_LOG_ERROR("{}", plan.json());
    return false;
  }

  std::array<int, 2U> result_pipe{};
  if (pipe(result_pipe.data()) != 0) {
    return check(false, "failed to create external export result pipe");
  }
  const pid_t child = fork();
  if (child < 0) {
    close(result_pipe[0]);
    close(result_pipe[1]);
    return check(false, "failed to fork external export subscriber");
  }
  if (child == 0) {
    close(result_pipe[0]);
    const std::string descriptor = fmt::format("{}", result_pipe[1]);
    execl("/proc/self/exe",
          "multi_tensor_schema_runtime_validation",
          "external-export-subscriber",
          descriptor.c_str(),
          endpoint_name.c_str(),
          static_cast<char*>(nullptr));
    std::_Exit(127);
  }
  close(result_pipe[1]);

  char ready = 0;
  bool valid = read_byte(result_pipe[0], ready) && ready == 1;
  if (!valid) {
    close(result_pipe[0]);
    static_cast<void>(reap_child(child));
    return check(false, "external export subscriber failed to initialize");
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool source_started = ProbeState::wait_for(ProbeState::source_starts, 1U);
  char connected = 0;
  const bool subscriber_connected = read_byte(result_pipe[0], connected) && connected == 1;
  ProbeState::source_released.store(source_started && subscriber_connected,
                                    std::memory_order_release);
  ProbeState::notify();
  const bool emitted = ProbeState::wait_for(ProbeState::emitted, 1U);
  char received = 0;
  const bool child_received = read_byte(result_pipe[0], received) && received == 1;
  session.request_stop();
  session.wait();
  valid = source_started && subscriber_connected && emitted && child_received &&
          session.termination() == holoscan::RunTermination::kStopped &&
          session.diagnostics().empty();
  close(result_pipe[0]);
  valid &= reap_child(child);
  if (!valid) {
    print_diagnostics(session);
  }
  // Key assertion: the child receives the full schema and both processes terminate cleanly.
  return check(valid, "cross-process external multi-Tensor schema export failed");
}
#else
[[nodiscard]] bool run_external_topic_export_cross_process() {
  // Key setup: external cross-process export APIs are unavailable on this platform.
  // Key assertion: report the platform-specific scenario as skipped successfully.
  return true;
}
#endif

[[nodiscard]] std::optional<std::uint32_t> parse_uint(const std::string_view value) {
  const std::string owned{value};
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(owned.c_str(), &end, 10);
  if (end == owned.c_str() || *end != '\0' || parsed > std::numeric_limits<std::uint32_t>::max()) {
    return std::nullopt;
  }
  return static_cast<std::uint32_t>(parsed);
}

[[nodiscard]] std::optional<std::uint32_t> parse_positive(const std::string_view value) {
  const auto parsed = parse_uint(value);
  return parsed && *parsed != 0U ? parsed : std::nullopt;
}

}  // namespace

// Scenario dispatch is intentionally centralized in this standalone validation executable.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
int main(const int argc, char** argv) {
  if (argc < 2) {
    return 2;
  }
  const std::string_view scenario{argv[1]};
  const auto scenario_result = [](const bool success) {
    return cuda_scenario_skipped ? 77 : (success ? 0 : 1);
  };
  if (scenario == "contracts" && argc == 2) {
    return scenario_result(run_contract_validation(kDefaultQueueDepth));
  }
  if (scenario == "rate-fields" && argc == 2) {
    return scenario_result(run_rate_aware_fields());
  }
  if (scenario == "rate-fields-fixed" && argc == 2) {
    return scenario_result(run_rate_aware_fields<SourceMode::kFixedRateFields>());
  }
  if (scenario == "rate-fields-configure" && argc == 2) {
    return scenario_result(run_rate_aware_fields<SourceMode::kRateAwareFieldsConfigure>());
  }
  if (scenario == "rate-fields-restart" && argc == 2) {
    return scenario_result(run_rate_aware_fields_restart());
  }
  if (scenario == "rate-fields-restart-reject" && argc == 2) {
    return scenario_result(run_rate_aware_fields_restart_reject());
  }
  if (scenario == "rate-fields-warm-up" && argc == 2) {
    return scenario_result(
        run_rate_aware_fields_late_binding<SourceMode::kRateAwareFieldsWarmUp>());
  }
  if (scenario == "rate-fields-arm" && argc == 2) {
    return scenario_result(run_rate_aware_fields_late_binding<SourceMode::kRateAwareFieldsArm>());
  }
  if (scenario == "capacity" && argc == 3) {
    const auto depth = parse_positive(argv[2]);
    return scenario_result(depth && run_contract_validation(*depth));
  }
  if (scenario == "process-local" && argc == 2) {
    return scenario_result(run_process_local());
  }
  if (scenario == "process-local-fanout" && argc == 2) {
    return scenario_result(run_process_local_fanout());
  }
  if (scenario == "inline-partition" && argc == 2) {
    return scenario_result(run_inline_partition<holoscan::MemoryKind::kHost>());
  }
  if (scenario == "inline-partition-cuda" && argc == 2) {
    return scenario_result(run_inline_partition<holoscan::MemoryKind::kCudaDevice>());
  }
  if (scenario == "over-bound" && argc == 2) {
    return scenario_result(run_over_bound());
  }
  if (scenario == "bundle-invalid" && argc == 2) {
    return scenario_result(run_invalid_bundle());
  }
  if (scenario == "bundle-restart" && argc == 2) {
    return scenario_result(run_bundle_restart());
  }
  if (scenario == "external-topic-import" && argc == 2) {
    return scenario_result(run_external_topic_import(false));
  }
  if (scenario == "external-topic-import-fanout" && argc == 2) {
    return scenario_result(run_external_topic_import(true));
  }
  if (scenario == "external-topic-export" && argc == 2) {
    return scenario_result(run_external_topic_export());
  }
  if (scenario == "external-topic-export-mixed-locality" && argc == 2) {
    return scenario_result(run_external_topic_export(true));
  }
  if (scenario == "external-topic-export-cross-process" && argc == 2) {
    return scenario_result(run_external_topic_export_cross_process());
  }
  if (scenario == "cross-process" && argc == 2) {
    return scenario_result(run_cross_process<SourceMode::kValid>(false));
  }
  if (scenario == "fanout" && argc == 2) {
    return scenario_result(run_cross_process<SourceMode::kValid>(true));
  }
  if (scenario == "bundle-cross-process" && argc == 2) {
    return scenario_result(run_cross_process<SourceMode::kLoanBundle>(false));
  }
  if (scenario == "bundle-fanout" && argc == 2) {
    return scenario_result(run_cross_process<SourceMode::kLoanBundle>(true));
  }
  if (scenario == "mixed-locality" && argc == 2) {
    return scenario_result(run_mixed_locality<SourceMode::kValid>());
  }
  if (scenario == "bundle-mixed-locality" && argc == 2) {
    return scenario_result(run_mixed_locality<SourceMode::kLoanBundle>());
  }
#if defined(__linux__)
  if (scenario == "external-export-subscriber" && argc == 4) {
    const auto descriptor = parse_uint(argv[2]);
    return descriptor && run_external_topic_export_subscriber_worker(static_cast<int>(*descriptor),
                                                                     argv[3])
               ? 0
               : 1;
  }
  if (scenario == "consumer" && argc == 6) {
    const auto descriptor = parse_positive(argv[2]);
    const auto target = parse_uint(argv[4]);
    if (!descriptor || !target) {
      return 1;
    }
    const std::string_view arm{argv[3]};
    const Topology topology = arm == "fanout"  ? Topology::kCrossProcessFanout
                              : arm == "mixed" ? Topology::kMixedLocality
                                               : Topology::kCrossProcess;
    const bool valid =
        std::string_view(argv[5]) == "bundle"
            ? run_consumer_worker<SourceMode::kLoanBundle>(
                  static_cast<int>(*descriptor), topology, static_cast<std::size_t>(*target))
            : run_consumer_worker<SourceMode::kValid>(
                  static_cast<int>(*descriptor), topology, static_cast<std::size_t>(*target));
    return valid ? 0 : 1;
  }
#endif
  return 2;
}
