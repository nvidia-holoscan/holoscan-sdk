/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_channel_binding.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <semaphore>
#include <shared_mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include <holoscan/channel/endpoint_interfaces.hpp>
#include <holoscan/channel/manual_tensor_publisher.hpp>
#include <holoscan/channel/runtime.hpp>
#include <holoscan/channel/tensor_endpoints.hpp>
#include <holoscan/channel/types.hpp>
#include <holoscan/core/domain/dlpack_utils.hpp>
#include <holoscan/logger/logger.hpp>

#include "../channel/channel_catalog_binding.hpp"
#include "../channel/channel_tensor_loan_internal.hpp"
#include "../channel/payload_descriptor_wire.hpp"
#include "../channel/pod_codec.hpp"
#include "../channel/schema_envelope_codec.hpp"
#include "../compiler/execution_plan_state.hpp"
#include "../compiler/process_local_channel_plan_validation.hpp"
#include "../core/domain/tensor_access.hpp"
#include "../core/tensor_allocation_record_validation.hpp"
#include "channel_runtime_binding_access.hpp"
#include "endpoint_clock_health_runtime.hpp"
#include "runtime_phase_counters.hpp"
#include "runtime_support_thread_control.hpp"
#include "runtime_worker.hpp"
#include "temporal_preselection.hpp"

namespace holoscan::detail {
namespace {

constexpr auto kPumpWait = std::chrono::milliseconds{10};

class CudaDeviceCursor final {
 public:
  CudaDeviceCursor() noexcept : active_(cudaGetDevice(&previous_) == cudaSuccess) {
    current_ = previous_;
  }

  CudaDeviceCursor(const CudaDeviceCursor&) = delete;
  CudaDeviceCursor& operator=(const CudaDeviceCursor&) = delete;

  ~CudaDeviceCursor() { static_cast<void>(restore()); }

  [[nodiscard]] bool active() const noexcept { return active_; }

  [[nodiscard]] bool select(const DeviceId requested) noexcept {
    if (!active_) {
      return false;
    }
    if (!requested.valid()) {
      return restore();
    }
    if (current_ != requested.value) {
      if (cudaSetDevice(requested.value) != cudaSuccess) {
        return false;
      }
      current_ = requested.value;
    }
    return true;
  }

  [[nodiscard]] bool initialize(const DeviceId requested = {}) noexcept {
    return select(requested) && cudaFree(nullptr) == cudaSuccess;
  }

  [[nodiscard]] bool restore() noexcept {
    if (!active_ || current_ == previous_) {
      return active_;
    }
    if (cudaSetDevice(previous_) != cudaSuccess) {
      return false;
    }
    current_ = previous_;
    return true;
  }

 private:
  int previous_{};
  int current_{};
  bool active_{};
};

class TensorScratchReset final {
 public:
  explicit TensorScratchReset(std::span<Tensor> tensors) noexcept : tensors_(tensors) {}
  TensorScratchReset(const TensorScratchReset&) = delete;
  TensorScratchReset& operator=(const TensorScratchReset&) = delete;
  ~TensorScratchReset() {
    for (auto& tensor : tensors_) {
      tensor = Tensor{};
    }
  }

 private:
  std::span<Tensor> tensors_;
};

void count_phase(RuntimePhaseCounters* counters, RuntimePhaseCounter counter) noexcept {
  if (counters != nullptr) {
    counters->increment(counter);
  }
}

void trace_phase(RuntimePhaseCounters* counters, RuntimePhaseTracePoint point,
                 std::uint64_t source_sequence = 0U) noexcept {
  if (counters != nullptr) {
    counters->trace(point, kNoRuntimePhaseReceiver, {}, source_sequence);
  }
}

void trace_output_emit_phase(RuntimePhaseCounters* counters, RuntimePhaseTracePoint point,
                             PortId output_port, std::uint64_t source_sequence) noexcept {
  if (counters != nullptr) {
    counters->trace(point, kNoRuntimePhaseReceiver, {}, source_sequence, output_port.value());
  }
}

[[nodiscard]] ChannelQueuePolicy channel_queue_policy(const QueuePolicy policy) noexcept {
  switch (policy) {
    case QueuePolicy::kBlocking:
      return ChannelQueuePolicy::kBlocking;
    case QueuePolicy::kDropOldest:
      return ChannelQueuePolicy::kDropOldest;
    default:
      return ChannelQueuePolicy::kBlocking;
  }
}

[[nodiscard]] bool supported_queue_policy(const QueuePolicy policy) noexcept {
  return policy == QueuePolicy::kBlocking || policy == QueuePolicy::kDropOldest;
}

void corrupt_process_local_schema_frame(
    std::span<std::byte> frame,
    const channel_internal::CatalogSchemaFrameCorruption corruption) noexcept {
  constexpr std::size_t kSchemaIdOffset = 8U;
  constexpr std::size_t kSchemaVersionOffset = 24U;
  constexpr std::size_t kEndpointGenerationOffset = 32U;
  constexpr std::size_t kPayloadOffset = kSchemaEnvelopeHeaderSize;

  switch (corruption) {
    case channel_internal::CatalogSchemaFrameCorruption::kMalformedEnvelope:
      channel_internal::schema_envelope_detail::store_little<std::uint32_t>(frame, 4U, 0U);
      break;
    case channel_internal::CatalogSchemaFrameCorruption::kSchemaId:
      frame[kSchemaIdOffset] ^= std::byte{0xffU};
      break;
    case channel_internal::CatalogSchemaFrameCorruption::kSchemaVersion:
      channel_internal::schema_envelope_detail::store_little<std::uint32_t>(
          frame, kSchemaVersionOffset, 0U);
      break;
    case channel_internal::CatalogSchemaFrameCorruption::kEndpointGeneration:
      channel_internal::schema_envelope_detail::store_little<std::uint64_t>(
          frame,
          kEndpointGenerationOffset,
          channel_internal::kInitialSchemaEndpointGeneration + 1U);
      break;
    case channel_internal::CatalogSchemaFrameCorruption::kPayloadVerifier:
      std::fill(frame.begin() + kPayloadOffset, frame.end(), std::byte{0xffU});
      break;
  }
}

[[nodiscard]] bool unique_valid_vertices(std::span<const VertexId> vertices) noexcept {
  if (vertices.empty()) {
    return false;
  }
  for (auto current = vertices.begin(); current != vertices.end(); ++current) {
    if (!*current || std::find(std::next(current), vertices.end(), *current) != vertices.end()) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] expected<void, Error> validate_import_generation_capacities(
    std::span<const ImportedGenerationCapacityPlanRecord> records) {
  if (records.empty()) {
    return {};
  }
  std::vector<std::pair<std::string_view, std::uint32_t>> services;
  services.reserve(records.size());
  for (auto current = records.begin(); current != records.end(); ++current) {
    const bool duplicate_topic = std::find_if(records.begin(), current, [&](const auto& prior) {
                                   return prior.topic == current->topic;
                                 }) != current;
    const bool restart_capacity_valid = current->restart_capable_minimum_two
                                            ? current->endpoint_max_generations >= 2U &&
                                                  current->candidate_reservation_capacity == 1U
                                            : current->endpoint_max_generations == 1U &&
                                                  current->candidate_reservation_capacity == 0U;
    if (!current->topic || current->runtime_scope_id.empty() ||
        current->endpoint_or_service_id.empty() || duplicate_topic ||
        current->endpoint_max_generations == 0U || current->service_max_generations == 0U ||
        current->runtime_max_generations == 0U ||
        current->service_max_generations < current->endpoint_max_generations ||
        current->release_record_capacity < current->endpoint_max_generations ||
        current->quarantine_record_capacity < current->endpoint_max_generations ||
        !current->effective_profile.has_value() || current->attach_max_wait_ns.value_or(1U) == 0U ||
        !restart_capacity_valid || !current->no_wrap_aba_generation ||
        !current->fallback_rejected) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const auto service = std::find_if(services.begin(), services.end(), [&](const auto& item) {
      return item.first == current->endpoint_or_service_id;
    });
    if (service == services.end()) {
      services.emplace_back(current->endpoint_or_service_id, current->service_max_generations);
    } else if (service->second != current->service_max_generations) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
  }
  std::uint64_t runtime_capacity = 0U;
  for (const auto& [service, capacity] : services) {
    static_cast<void>(service);
    runtime_capacity += capacity;
    if (runtime_capacity > std::numeric_limits<std::uint32_t>::max()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
  }
  if (std::any_of(records.begin(), records.end(), [&](const auto& record) {
        return record.runtime_max_generations != runtime_capacity;
      })) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  return {};
}

[[nodiscard]] bool valid_input_selection(const InputSelection selection) noexcept {
  switch (selection.state) {
    case InputState::kSelected:
      return selection.reason == SelectionReason::kNone;
    case InputState::kReused:
    case InputState::kSafeDefault:
    case InputState::kAbsent:
      return selection.reason != SelectionReason::kNone &&
             selection.reason != SelectionReason::kNotSelectedForActivation;
    case InputState::kNotSelected:
      return selection.reason == SelectionReason::kNotSelectedForActivation;
  }
  return false;
}

[[nodiscard]] bool tensor_snapshot_matches_plan(const TensorAccessSnapshot& snapshot,
                                                std::uint64_t max_byte_span, MemoryKind memory_kind,
                                                DeviceId device) noexcept;

[[nodiscard]] bool tensor_matches_field_contract(
    const Tensor& tensor, const TensorAccessSnapshot& snapshot,
    const TensorFieldServiceStorageContract& field) noexcept {
  if (!tensor_snapshot_matches_plan(
          snapshot, field.max_byte_span, field.memory_kind, field.device)) {
    return false;
  }
  if (field.rank && tensor.ndim() != static_cast<std::int32_t>(*field.rank)) {
    return false;
  }
  if (field.dtype) {
    const DLDataType actual = tensor.dtype();
    if (actual.code != field.dtype->code || actual.bits != field.dtype->bits ||
        actual.lanes != field.dtype->lanes) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] expected<DeviceId, Error> uniform_tensor_field_device(
    DeviceId fallback, const std::span<const TensorFieldServiceStorageContract> fields) noexcept {
  DeviceId result = fallback;
  for (const auto& field : fields) {
    if (!field.device.valid()) {
      continue;
    }
    if (result.valid() && result != field.device) {
      return make_unexpected(
          Error{ErrorCode::kProtocolError,
                "descriptor Tensor field storage spans multiple CUDA devices for one endpoint"});
    }
    result = field.device;
  }
  return result;
}

[[nodiscard]] bool tensor_matches_port_layout(const Tensor& tensor,
                                              const TensorAccessSnapshot& snapshot,
                                              const TensorPortLayout& layout) noexcept {
  if (layout.memory_kind) {
    const bool memory_matches = [&]() noexcept {
      switch (*layout.memory_kind) {
        case MemoryKind::kHost:
          return snapshot.device.device_type == kDLCPU && snapshot.device.device_id == 0;
        case MemoryKind::kPinnedHost:
          return snapshot.device.device_type == kDLCUDAHost && snapshot.device.device_id == 0;
        case MemoryKind::kCudaDevice:
          return snapshot.device.device_type == kDLCUDA && snapshot.device.device_id >= 0;
        case MemoryKind::kCudaManaged:
          return snapshot.device.device_type == kDLCUDAManaged && snapshot.device.device_id == 0;
        case MemoryKind::kUnknown:
        default:
          return false;
      }
    }();
    if (!memory_matches) {
      return false;
    }
  }
  if (layout.dtype) {
    const DLDataType actual = tensor.dtype();
    if (actual.code != layout.dtype->code || actual.bits != layout.dtype->bits ||
        actual.lanes != layout.dtype->lanes) {
      return false;
    }
  }
  if (layout.rank && static_cast<std::uint32_t>(tensor.ndim()) != *layout.rank) {
    return false;
  }
  return true;
}

[[nodiscard]] bool logical_tensor_requirements_equal(
    const LogicalTensorRequirementsRecord& left,
    const LogicalTensorRequirementsRecord& right) noexcept {
  const bool dtype_equal = left.dtype.has_value() == right.dtype.has_value() &&
                           (!left.dtype || (left.dtype->code == right.dtype->code &&
                                            left.dtype->bits == right.dtype->bits &&
                                            left.dtype->lanes == right.dtype->lanes));
  if (left.memory_kind != right.memory_kind || !dtype_equal || left.rank != right.rank ||
      left.max_rank != right.max_rank || left.max_byte_span != right.max_byte_span ||
      left.strides != right.strides || left.foreign_emit_policy != right.foreign_emit_policy ||
      left.max_inline_bytes != right.max_inline_bytes || left.axes.size() != right.axes.size()) {
    return false;
  }
  return std::equal(
      left.axes.begin(), left.axes.end(), right.axes.begin(), [](const auto& lhs, const auto& rhs) {
        return lhs.max_extent == rhs.max_extent &&
               lhs.max_abs_stride_bytes == rhs.max_abs_stride_bytes;
      });
}

[[nodiscard]] bool tensor_matches_logical_requirements(
    const Tensor& tensor, const LogicalTensorRequirementsRecord& requirements) noexcept {
  const auto rank = tensor.ndim();
  if (rank < 0) {
    return false;
  }
  const auto logical_rank = static_cast<std::uint32_t>(rank);
  if ((requirements.rank && logical_rank != *requirements.rank) ||
      (requirements.max_rank && logical_rank > *requirements.max_rank)) {
    return false;
  }
  if (requirements.dtype) {
    const DLDataType actual = tensor.dtype();
    if (actual.code != requirements.dtype->code || actual.bits != requirements.dtype->bits ||
        actual.lanes != requirements.dtype->lanes) {
      return false;
    }
  }
  const auto shape = tensor.shape_span();
  if (shape.size() != static_cast<std::size_t>(rank) || requirements.axes.size() > shape.size()) {
    return false;
  }
  for (std::size_t axis = 0U; axis < requirements.axes.size(); ++axis) {
    if (shape[axis] < 0 ||
        static_cast<std::uint64_t>(shape[axis]) > requirements.axes[axis].max_extent) {
      return false;
    }
  }
  const std::uint64_t item_size = dlpack_dtype_item_size_bytes(tensor.dtype());
  if (item_size == 0U) {
    return false;
  }
  std::uint64_t contiguous_stride = item_size;
  for (std::size_t reverse = shape.size(); reverse > 0U; --reverse) {
    const std::size_t axis = reverse - 1U;
    if (axis < requirements.axes.size() && requirements.axes[axis].max_abs_stride_bytes != 0U &&
        contiguous_stride > requirements.axes[axis].max_abs_stride_bytes) {
      return false;
    }
    const std::uint64_t extent =
        std::max<std::uint64_t>(static_cast<std::uint64_t>(shape[axis]), 1U);
    if (contiguous_stride > std::numeric_limits<std::uint64_t>::max() / extent) {
      return false;
    }
    contiguous_stride *= extent;
  }
  // EA1 external Tensor execution admits only the contiguous policy. Keeping the policy check at
  // the live import boundary prevents a producer from bypassing catalog-time loan validation with
  // an explicitly strided DLPack view that happens to fit the frozen byte-span bound.
  return requirements.strides == TensorStridePolicy::kContiguousOnly && tensor.is_contiguous();
}

class AlignedBlock {
 public:
  AlignedBlock() = default;
  AlignedBlock(const AlignedBlock&) = delete;
  AlignedBlock& operator=(const AlignedBlock&) = delete;
  AlignedBlock(AlignedBlock&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        alignment_(std::exchange(other.alignment_, alignof(std::max_align_t))) {}
  AlignedBlock& operator=(AlignedBlock&& other) noexcept {
    if (this != &other) {
      reset();
      data_ = std::exchange(other.data_, nullptr);
      alignment_ = std::exchange(other.alignment_, alignof(std::max_align_t));
    }
    return *this;
  }
  ~AlignedBlock() { reset(); }

  [[nodiscard]] static expected<AlignedBlock, Error> create(std::size_t size,
                                                            std::size_t alignment) noexcept {
    if (size == 0U || alignment == 0U || (alignment & (alignment - 1U)) != 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    void* storage = ::operator new(size, std::align_val_t{alignment}, std::nothrow);
    if (storage == nullptr) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    AlignedBlock result;
    result.data_ = storage;
    result.alignment_ = alignment;
    return result;
  }

  [[nodiscard]] void* data() noexcept { return data_; }
  [[nodiscard]] const void* data() const noexcept { return data_; }

 private:
  void reset() noexcept {
    if (data_ != nullptr) {
      ::operator delete(data_, std::align_val_t{alignment_});
      data_ = nullptr;
    }
  }

  void* data_{};
  std::size_t alignment_{alignof(std::max_align_t)};
};

class ReadClaimPool {
 public:
  struct Slot {
    ReadClaimPool* pool{};
    std::atomic<std::uint32_t> references{};
    std::atomic<bool> in_use{};
  };

  [[nodiscard]] static expected<ReadClaimPool*, Error> create(std::size_t capacity) noexcept {
    if (capacity == 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      auto* pool = new ReadClaimPool(capacity);
      return pool;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (const std::bad_alloc&) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
#endif
  }

  [[nodiscard]] expected<PublishedReadClaimBinding, Error> acquire() noexcept {
    for (std::size_t index = 0U; index < capacity_; ++index) {
      Slot& slot = slots_[index];
      bool available = false;
      if (slot.in_use.compare_exchange_strong(
              available, true, std::memory_order_acquire, std::memory_order_relaxed)) {
        slot.references.store(1U, std::memory_order_relaxed);
        live_claims_.fetch_add(1U, std::memory_order_release);
        references_.fetch_add(1U, std::memory_order_relaxed);
        return PublishedReadClaimBinding{
            .state = &slot,
            .retain = &retain,
            .release = &release,
        };
      }
    }
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }

  void release_owner() noexcept {
    if (references_.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
      delete this;
    }
  }

  [[nodiscard]] bool has_live_claims() const noexcept {
    return live_claims_.load(std::memory_order_acquire) != 0U;
  }

 private:
  // Runtime capacity is frozen before construction, while Slot contains atomics and is not
  // relocatable. A single fixed-capacity allocation therefore preserves stable slot addresses.
  explicit ReadClaimPool(std::size_t capacity)
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
      : slots_(std::make_unique<Slot[]>(capacity)), capacity_(capacity) {
    for (std::size_t index = 0U; index < capacity_; ++index) {
      slots_[index].pool = this;
    }
  }

  static void retain(void* opaque) noexcept {
    auto& slot = *static_cast<Slot*>(opaque);
    slot.references.fetch_add(1U, std::memory_order_relaxed);
  }

  static void release(void* opaque) noexcept {
    auto& slot = *static_cast<Slot*>(opaque);
    if (slot.references.fetch_sub(1U, std::memory_order_acq_rel) != 1U) {
      return;
    }
    slot.in_use.store(false, std::memory_order_release);
    ReadClaimPool* const pool = slot.pool;
    pool->live_claims_.fetch_sub(1U, std::memory_order_acq_rel);
    if (pool->references_.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
      delete pool;
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays): see the fixed-capacity invariant above.
  std::unique_ptr<Slot[]> slots_;
  std::size_t capacity_{};
  std::atomic<std::uint32_t> references_{1U};
  std::atomic<std::uint32_t> live_claims_{};
};

[[nodiscard]] SampleMetadata metadata_from_options(const EmitOptions& options,
                                                   std::size_t payload_size) noexcept {
  SampleMetadata metadata{};
  if (options.capture_time) {
    metadata.capture_timestamp_ns = options.capture_time->timestamp_ns;
    metadata.source_clock_id = options.capture_time->clock_id;
  }
  if (options.frame_id) {
    metadata.frame_id = *options.frame_id;
  }
  if (options.trace) {
    metadata.trace_id = options.trace->trace_id;
  }
  metadata.byte_size = payload_size;
  metadata.payload_kind = PayloadKind::kInlineTyped;
  metadata.flags = options.flags;
  return metadata;
}

}  // namespace

static_assert(std::is_trivially_copyable_v<RuntimeChannelDirectContinuationReceipt>);

bool runtime_channel_direct_continuation_matches(
    const RuntimeChannelDirectContinuationReceipt& receipt,
    const RuntimeChannelDirectContinuationTarget& target) noexcept {
  return receipt.plan_generation != 0U && receipt.plan_generation == target.plan_generation &&
         receipt.run_generation != 0U && receipt.run_generation == target.run_generation &&
         receipt.producer && receipt.producer == target.producer && receipt.destination &&
         receipt.destination == target.destination && receipt.source_output &&
         receipt.source_output == target.source_output && receipt.destination_input &&
         receipt.destination_input == target.destination_input && !receipt.service.empty() &&
         receipt.service == target.service && receipt.readiness_source &&
         receipt.readiness_source == target.readiness_source && receipt.reference_clock.valid() &&
         receipt.reference_clock == target.reference_clock && receipt.source_generation != 0U &&
         receipt.source_generation == target.source_generation &&
         receipt.readiness_sequence != 0U &&
         receipt.readiness_sequence == target.readiness_sequence && receipt.producer_epoch != 0U &&
         receipt.publication_sequence != 0U && receipt.source_activation_generation != 0U &&
         receipt.source_activation_generation == target.source_activation_generation &&
         receipt.destination_revision == target.destination_revision &&
         receipt.destination_revision == target.observed_destination_revision &&
         receipt.execution_lane == target.execution_lane && receipt.successor_publications == 1U &&
         receipt.publication_status == ErrorCode::kSuccess &&
         !publication_error_consumes_authority(receipt.publication_status) &&
         receipt.prior_destination_revision != std::numeric_limits<std::uint64_t>::max() &&
         receipt.destination_revision == receipt.prior_destination_revision + 1U;
}

class RuntimeChannelSession::Impl {
 public:
#if !defined(NDEBUG)
  /// Assert that one activation owner has only one mutator at a time.
  ///
  /// Different persistent partition owners may mutate their disjoint activation records in
  /// parallel. The shared barrier still rejects overlap with a whole-session control transaction.
  class OwnerMutationGuard {
   public:
    OwnerMutationGuard(std::shared_mutex& barrier, std::atomic_flag& active) noexcept
        : barrier_(barrier, std::try_to_lock), active_(&active) {
      assert(barrier_.owns_lock());
      const bool already_active = active_->test_and_set(std::memory_order_acquire);
      assert(!already_active);
    }

    OwnerMutationGuard(const OwnerMutationGuard&) = delete;
    OwnerMutationGuard& operator=(const OwnerMutationGuard&) = delete;

    ~OwnerMutationGuard() noexcept { active_->clear(std::memory_order_release); }

   private:
    std::shared_lock<std::shared_mutex> barrier_;
    std::atomic_flag* active_;
  };

  /// Assert that a reset or restart transaction does not overlap any activation-owner mutation.
  class ControlMutationGuard {
   public:
    explicit ControlMutationGuard(std::shared_mutex& barrier) noexcept
        : barrier_(barrier, std::try_to_lock) {
      assert(barrier_.owns_lock());
    }

    ControlMutationGuard(const ControlMutationGuard&) = delete;
    ControlMutationGuard& operator=(const ControlMutationGuard&) = delete;

    ~ControlMutationGuard() = default;

   private:
    std::unique_lock<std::shared_mutex> barrier_;
  };
#endif

  struct SubscriberState;
  class ReadinessBinding;

  struct SampleSlot {
    SubscriberState* subscriber{};
    AlignedBlock sample;
    TemporalSampleRecord temporal{};
    std::uint64_t readiness_sequence{};
    std::uint64_t arrival_order{};
    EndpointClockHealthDisposition clock_health_disposition{
        EndpointClockHealthDisposition::kNotApplicable};
    std::uint64_t clock_health_generation{};
    bool occupied{};
    bool event_emitted{};
    bool witness_retired{};
    bool active_selected{};
    bool reuse_retained{};
    bool temporal_retained{};
    bool retirement_selected{};
    bool typed_ingress_reserved{};
    bool sample_storage_prepared{};
    bool direct_ingress_unread{};
  };

  struct InputPortState {
    PortId port{};
    VertexId owner{};
    void* declaration{};
    const RuntimePortTypeOps* type_ops{};
    std::vector<SubscriberState*> subscribers;
    std::vector<SampleSlot*> active_slots;
    std::vector<SampleSlot*> retirement_slots;
    std::vector<SampleSlot*> tensor_access_slots;
    std::vector<TensorReadLease> tensor_read_leases;
    ReadClaimPool* claim_pool{};
    RuntimeInputBinding binding{};
    AlignedBlock predicate_view_storage;
    ClockId capture_clock{};
    ClockId reference_clock{};
    std::uint64_t conversion_id{};
    std::optional<DirectAffineClockEdge> capture_to_reference;
    RuntimePhaseCounters* phase_counters{};
    ReadinessBinding* readiness_binding{};
    // A producer publishes this equality-only dirty epoch after completing typed cache mutation.
    // Aliasing an observed value would require one complete uint64_t cycle between polls.
    std::atomic<std::uint64_t> readiness_revision{};
    bool capture_time_required{};
    SampleSlot* reuse_slot{};
    InputSelection active_selection{InputState::kAbsent, SelectionReason::kNeverSeen};
    std::uint64_t active_generation{1U};
    std::uint64_t view_generation{1U};
    bool active{};
    bool batch{};
    bool destructive{};
    bool reuse_last{};
    bool selection_prepared{};
    bool staged_now{};
    bool predicate_view_constructed{};

    ~InputPortState() {
      if (predicate_view_constructed) {
        type_ops->destroy_predicate_view(predicate_view_storage.data());
      }
      if (claim_pool != nullptr) {
        claim_pool->release_owner();
      }
    }

    [[nodiscard]] std::uint64_t publish_readiness_change() noexcept {
      const std::uint64_t published =
          readiness_revision.fetch_add(1U, std::memory_order_release) + 1U;
      count_phase(phase_counters, RuntimePhaseCounter::kChannelReadinessPublished);
      return published;
    }

    [[nodiscard]] static InputSelection selection(const void* opaque,
                                                  std::uint64_t generation) noexcept {
      const auto& state = *static_cast<const InputPortState*>(opaque);
      if (!state.active || generation != state.active_generation) {
        return {InputState::kAbsent, SelectionReason::kNeverSeen};
      }
      return state.active_selection;
    }

    [[nodiscard]] static expected<void, Error> classify(const InputPortState& state,
                                                        std::uint64_t generation,
                                                        bool batch_accessor) noexcept {
      if (!state.active || generation != state.active_generation) {
        return make_unexpected(Error{ErrorCode::kStaleEndpoint});
      }
      if (state.active_selection.state == InputState::kNotSelected) {
        return make_unexpected(Error{ErrorCode::kInputNotSelected});
      }
      if (state.batch != batch_accessor) {
        return make_unexpected(Error{ErrorCode::kInputSelectionCardinalityMismatch});
      }
      if (state.active_selection.state == InputState::kAbsent ||
          state.active_selection.state == InputState::kSafeDefault ||
          (!batch_accessor && state.active_slots.empty())) {
        return make_unexpected(Error{ErrorCode::kInputEmpty});
      }
      return {};
    }

    [[nodiscard]] static expected<RuntimeScalarReadBinding, Error> scalar(
        const void* opaque, std::uint64_t generation, bool acquire_claim) noexcept {
      auto& state = *static_cast<InputPortState*>(const_cast<void*>(opaque));
      auto classified = classify(state, generation, false);
      if (!classified) {
        return make_unexpected(std::move(classified).error());
      }
      if (state.active_slots.size() != 1U) {
        return make_unexpected(Error{ErrorCode::kInputSelectionCardinalityMismatch});
      }
      PublishedReadClaimBinding claim{};
      if (acquire_claim) {
        auto acquired = state.claim_pool->acquire();
        if (!acquired) {
          return make_unexpected(std::move(acquired).error());
        }
        claim = *acquired;
      }
      count_phase(state.phase_counters, RuntimePhaseCounter::kChannelInputReads);
      return RuntimeScalarReadBinding{
          .sample = state.active_slots.front()->sample.data(),
          .claim = claim,
      };
    }

    [[nodiscard]] static expected<RuntimeBatchReadBinding, Error> batch_read(
        const void* opaque, std::uint64_t generation) noexcept {
      const auto& state = *static_cast<const InputPortState*>(opaque);
      auto classified = classify(state, generation, true);
      if (!classified) {
        return make_unexpected(std::move(classified).error());
      }
      count_phase(state.phase_counters, RuntimePhaseCounter::kChannelInputReads);
      return RuntimeBatchReadBinding{
          .state = &state,
          .size = state.active_slots.size(),
          .active_generation = &state.active_generation,
          .expected_generation = generation,
          .sample_at = [](const void* batch_state, std::size_t index) noexcept -> const void* {
            const auto& input = *static_cast<const InputPortState*>(batch_state);
            return index < input.active_slots.size() ? input.active_slots[index]->sample.data()
                                                     : nullptr;
          },
      };
    }

    inline static const RuntimeInputBindingOps kBindingOps{
        .selection = &selection,
        .scalar = &scalar,
        .batch = &batch_read,
    };
  };

  struct SubscriberState {
    PortId source{};
    PortId target{};
    VertexId source_owner{};
    VertexId owner{};
    // The plan can be released or rebuilt after session construction. Keep the service identity
    // with the subscriber because direct-continuation targets expose it for the lifetime of this
    // session.
    std::string service;
    QueuePolicy queue_policy{QueuePolicy::kBlocking};
    std::uint32_t readiness_capacity{};
    std::uint32_t drain_capacity{};
    std::uint32_t read_claim_capacity{};
    std::uint32_t typed_slot_capacity{};
    std::uint32_t direct_ingress_capacity{};
    std::uint32_t health_dispatch_budget{};
    bool pre_witness_replacement_allowed{};
    bool direct_process_local{};
    bool cross_process_transport{};
    bool externally_driven_import{};
    bool catalog_tensor_import{};
    InputPortState* input{};
    std::unique_ptr<channel_internal::InlinePayloadSubscriber> endpoint;
    std::optional<ChannelSubscriber<Tensor>> tensor_endpoint;
    std::unique_ptr<channel_internal::TensorSchemaPayloadSubscriber> tensor_schema_endpoint;
    std::uint64_t tensor_max_byte_span{};
    MemoryKind tensor_memory_kind{MemoryKind::kUnknown};
    DeviceId tensor_device{};
    std::vector<TensorFieldServiceStorageContract> tensor_fields;
    std::vector<PayloadDescriptor> descriptor_tensor_scratch;
    std::vector<Tensor> tensor_schema_scratch;
    std::optional<TensorPortLayout> pass_through_tensor_layout;
    std::optional<LogicalTensorRequirementsRecord> tensor_logical_requirements;
    std::uint64_t wait_handle{};
    std::uint64_t terminal_wait_handle{};
    AlignedBlock scratch;
    std::size_t scratch_size{};
    std::vector<std::byte> native_scratch;
    std::optional<PodIdentityRecord> pod_identity;
    std::optional<SchemaIdentityRecord> process_local_schema_identity;
    std::vector<SampleSlot> slots;
    void* typed_ingress_observer_state{};
    void (*typed_ingress_observer)(void*) noexcept {};
    // Serialize the single transport consumer and its scratch independently from retained typed
    // state. First-use Tensor imports may map shared resources, so they must not hold the cache
    // mutex needed by readiness polling and activation retirement.
    mutable std::mutex transport_mutex;
    mutable std::mutex mutex;
    std::uint64_t next_arrival_sequence{1U};
    std::uint64_t producer_epoch{1U};
    std::uint64_t last_producer_sequence{};
    std::uint64_t last_direct_ingress_sequence{};
    std::uint64_t end_of_stream_arrival_order{};
    std::uint64_t end_of_stream_transport_sequence{};
    bool end_of_stream_received{};
    bool end_of_stream_event_emitted{};
    ClockId catalog_capture_clock{};
    ClockId plan_capture_clock{};
    PortId clock_health_endpoint{};
    EndpointClockHealthRuntimeCoordinator* clock_health_coordinator{};
    // Generation-fenced rebind state for externally driven imports. The catalog copy shares the
    // plan's immutable endpoint-catalog state so the same catalog record can be re-resolved for a
    // replacement attachment. The cold attachment consumes slot one; each successful rebind
    // monotonically consumes another plan-admitted slot. CHAN-087 deliberately never decrements or
    // reuses this no-wrap lifetime budget (unlike EA1-014's reusable Tensor physical slots).
    ChannelEndpointCatalog rebind_catalog{};
    std::string rebind_endpoint_name;
    std::uint32_t import_generation_capacity{};
    std::uint32_t import_generations_used{1U};
    bool import_restart_capable{};

    ~SubscriberState() {
      clear_samples();
      for (SampleSlot& slot : slots) {
        if (slot.sample_storage_prepared) {
          input->type_ops->destroy_sample(slot.sample.data());
          slot.sample_storage_prepared = false;
        }
      }
    }

    [[nodiscard]] std::span<std::byte> scratch_bytes() noexcept {
      return {static_cast<std::byte*>(scratch.data()), scratch_size};
    }

    [[nodiscard]] std::span<const std::byte> scratch_bytes() const noexcept {
      return {static_cast<const std::byte*>(scratch.data()), scratch_size};
    }

    [[nodiscard]] bool requires_pump_vmm_import_context() const noexcept {
      return tensor_endpoint.has_value() ||
             (tensor_schema_endpoint != nullptr && cross_process_transport);
    }

    [[nodiscard]] bool has_pending_direct_ingress() const noexcept {
      return std::any_of(slots.begin(), slots.end(), [](const auto& slot) {
        return slot.occupied && slot.direct_ingress_unread;
      });
    }

    [[nodiscard]] bool uses_device_affine_descriptor_import() const noexcept {
      return (tensor_endpoint.has_value() ||
              (tensor_schema_endpoint != nullptr && cross_process_transport)) &&
             tensor_device.valid();
    }

    [[nodiscard]] expected<std::uint64_t, Error> attach_wait_set(
        channel_internal::InlinePayloadWaitSet& wait_set) {
      const std::uint32_t endpoint_count =
          static_cast<std::uint32_t>(endpoint != nullptr) +
          static_cast<std::uint32_t>(tensor_endpoint.has_value()) +
          static_cast<std::uint32_t>(tensor_schema_endpoint != nullptr);
      if (direct_process_local || endpoint_count != 1U) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      if (endpoint != nullptr) {
        return wait_set.attach(*endpoint);
      }
      if (tensor_endpoint.has_value()) {
        return RuntimeChannelBindingAccess::attach_wait_set(wait_set, *tensor_endpoint);
      }
      return wait_set.attach(*tensor_schema_endpoint);
    }

    [[nodiscard]] std::optional<std::uint64_t> secondary_wait_set_handle() noexcept {
      if (endpoint != nullptr) {
        return channel_internal::protected_terminal_wait_handle(*endpoint);
      }
      if (tensor_endpoint.has_value()) {
        return RuntimeChannelBindingAccess::secondary_wait_set_handle(*tensor_endpoint);
      }
      if (tensor_schema_endpoint != nullptr) {
        return channel_internal::protected_terminal_wait_handle(*tensor_schema_endpoint);
      }
      return std::nullopt;
    }

    [[nodiscard]] expected<bool, Error> has_pending_transport() const noexcept {
      if (direct_process_local) {
        return has_pending_direct_ingress();
      }
      if (tensor_endpoint.has_value()) {
        return tensor_endpoint->has_samples();
      }
      if (tensor_schema_endpoint != nullptr) {
        return tensor_schema_endpoint->has_samples();
      }
      if (endpoint == nullptr) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      if (externally_driven_import || clock_health_endpoint) {
        return channel_internal::CatalogChannelBindingAccess::has_pending_operation(*endpoint);
      }
      return endpoint->has_samples();
    }

    [[nodiscard]] expected<std::uint32_t, Error> discard_process_local_transport(
        CudaDeviceCursor* const scan_device) noexcept {
      if (direct_process_local) {
        return 0U;
      }
      if (externally_driven_import || clock_health_endpoint || drain_capacity == 0U) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      const std::scoped_lock lock{mutex};
      std::uint32_t discarded_count = 0U;
      for (; discarded_count < drain_capacity; ++discarded_count) {
        auto pending = has_pending_transport();
        if (!pending) {
          return make_unexpected(std::move(pending).error());
        }
        if (!*pending) {
          return discarded_count;
        }
        auto discarded = discard_one_process_local_transport(scan_device);
        if (!discarded) {
          return make_unexpected(std::move(discarded).error());
        }
      }
      auto excess = has_pending_transport();
      if (!excess) {
        return make_unexpected(std::move(excess).error());
      }
      if (*excess) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      return discarded_count;
    }

    [[nodiscard]] expected<void, Error> discard_one_process_local_transport(
        CudaDeviceCursor* const scan_device) noexcept {
      if (tensor_endpoint.has_value()) {
        if (uses_device_affine_descriptor_import() &&
            (scan_device == nullptr || !scan_device->select(tensor_device))) {
          return make_unexpected(Error{ErrorCode::kNotSupported});
        }
        auto discarded = tensor_endpoint->receive_sample();
        if (!discarded) {
          return make_unexpected(std::move(discarded).error());
        }
        return {};
      }
      if (tensor_schema_endpoint != nullptr) {
        auto received = tensor_schema_endpoint->borrow_payload();
        if (!received) {
          return make_unexpected(std::move(received).error());
        }
        const std::span<const std::byte> wire_payload = received->payload();
        const RuntimePortTypeOps& ops = *input->type_ops;
        const std::size_t descriptor_bytes =
            static_cast<std::size_t>(ops.descriptor_tensor_static_count) *
            sizeof(PayloadDescriptor);
        if (ops.descriptor_tensor_static_count == 0U ||
            ops.descriptor_tensor_static_count > channel_internal::kMaximumDescriptorTensorFields ||
            ops.descriptor_offset > wire_payload.size() ||
            descriptor_bytes > wire_payload.size() - ops.descriptor_offset) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        for (std::uint32_t index = 0U; index < ops.descriptor_tensor_static_count; ++index) {
          auto descriptor = channel_internal::decode_payload_descriptor_wire(wire_payload.subspan(
              ops.descriptor_offset + static_cast<std::size_t>(index) * sizeof(PayloadDescriptor),
              sizeof(PayloadDescriptor)));
          if (!descriptor) {
            return make_unexpected(std::move(descriptor).error());
          }
          tensor_schema_endpoint->release_recipient_claim(*descriptor);
        }
        return {};
      }
      auto discarded = endpoint->receive(scratch_bytes());
      if (!discarded) {
        return make_unexpected(std::move(discarded).error());
      }
      return {};
    }

    void clear_samples() noexcept {
      const std::scoped_lock lock{mutex};
      for (SampleSlot& slot : slots) {
        if (slot.occupied) {
          if (slot.sample_storage_prepared) {
            input->type_ops->reset_sample_storage(slot.sample.data());
          } else {
            input->type_ops->destroy_sample(slot.sample.data());
          }
          slot.occupied = false;
        }
        slot.event_emitted = false;
        slot.witness_retired = false;
        slot.active_selected = false;
        slot.reuse_retained = false;
        slot.temporal_retained = false;
        slot.retirement_selected = false;
        slot.typed_ingress_reserved = false;
        slot.readiness_sequence = 0U;
        slot.arrival_order = 0U;
        slot.clock_health_disposition = EndpointClockHealthDisposition::kNotApplicable;
        slot.clock_health_generation = 0U;
        slot.direct_ingress_unread = false;
        slot.temporal = {};
      }
      next_arrival_sequence = 1U;
      last_producer_sequence = 0U;
      last_direct_ingress_sequence = 0U;
      end_of_stream_arrival_order = 0U;
      end_of_stream_transport_sequence = 0U;
      end_of_stream_received = false;
      end_of_stream_event_emitted = false;
    }
  };

  // Own one exact typed destination across transport receive and first-use import work performed
  // without the cache mutex. Every non-commit exit restores the prior cache visibility. release()
  // acquires the cache mutex; never let an armed reservation die while holding that mutex.
  class TypedIngressReservation {
   public:
    TypedIngressReservation() = default;

    TypedIngressReservation(SubscriberState& subscriber, SampleSlot& destination) noexcept
        : subscriber_(&subscriber), destination_(&destination) {}

    TypedIngressReservation(const TypedIngressReservation&) = delete;
    TypedIngressReservation& operator=(const TypedIngressReservation&) = delete;

    TypedIngressReservation(TypedIngressReservation&& other) noexcept
        : subscriber_(other.subscriber_), destination_(other.destination_) {
      other.subscriber_ = nullptr;
      other.destination_ = nullptr;
    }

    TypedIngressReservation& operator=(TypedIngressReservation&&) = delete;

    ~TypedIngressReservation() { release(); }

    // Transfer the protected destination to a store operation while its cache mutex is held.
    [[nodiscard]] SampleSlot* commit_locked() noexcept {
      assert(subscriber_ != nullptr);
      assert(destination_ != nullptr);
      assert(destination_->typed_ingress_reserved);
      destination_->typed_ingress_reserved = false;
      SampleSlot* const destination = destination_;
      subscriber_ = nullptr;
      destination_ = nullptr;
      return destination;
    }

    void release() noexcept {
      if (subscriber_ == nullptr) {
        return;
      }
      const std::scoped_lock cache_lock{subscriber_->mutex};
      assert(destination_ != nullptr);
      assert(destination_->typed_ingress_reserved);
      destination_->typed_ingress_reserved = false;
      subscriber_ = nullptr;
      destination_ = nullptr;
    }

   private:
    SubscriberState* subscriber_{};
    SampleSlot* destination_{};
  };

  /// Clear an occupied slot after every independently owned payload claim has retired.
  ///
  /// The caller must hold the owning subscriber mutex. Keeping this operation centralized makes
  /// it impossible for witness, activation, reuse, and temporal-cache retirement paths to reset a
  /// subset of the slot state while another path can still observe the payload.
  static void reset_sample_slot(SampleSlot& slot) noexcept {
    if (slot.occupied) {
      if (slot.sample_storage_prepared) {
        slot.subscriber->input->type_ops->reset_sample_storage(slot.sample.data());
      } else {
        slot.subscriber->input->type_ops->destroy_sample(slot.sample.data());
      }
    }
    slot.occupied = false;
    slot.event_emitted = false;
    slot.witness_retired = false;
    slot.active_selected = false;
    slot.reuse_retained = false;
    slot.temporal_retained = false;
    slot.retirement_selected = false;
    slot.typed_ingress_reserved = false;
    slot.readiness_sequence = 0U;
    slot.arrival_order = 0U;
    slot.clock_health_disposition = EndpointClockHealthDisposition::kNotApplicable;
    slot.clock_health_generation = 0U;
    slot.direct_ingress_unread = false;
    slot.temporal = {};
  }

  /// Reclaim storage only after transport attribution and every runtime payload claim have ended.
  ///
  /// The caller must hold the owning subscriber mutex. A readiness event transfers a temporal
  /// claim to the selector before the witness can retire, so witness retirement alone is never
  /// sufficient to recycle a slot retained by a cache, pending window, closed window, or reuse
  /// policy.
  static void reclaim_sample_slot_if_unclaimed(SampleSlot& slot) noexcept {
    if (slot.occupied && !slot.temporal_retained && !slot.retirement_selected &&
        !slot.typed_ingress_reserved && !slot.active_selected && !slot.reuse_retained &&
        (!slot.event_emitted || slot.witness_retired)) {
      reset_sample_slot(slot);
    }
  }

  struct OutputState;

  struct LoanControl {
    OutputState* owner{};
    std::atomic<std::uint32_t>* live_credit{};
    const RuntimePortTypeOps* type_ops{};
    AlignedBlock value;
    RuntimeLoanBinding binding{};
    std::atomic<std::uint32_t> references{1U};
    std::uint64_t activation_generation{};
    bool live{};

    void release_reference() noexcept {
      if (references.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
        type_ops->destroy_value(value.data());
        delete this;
      }
    }

    void reset_value() noexcept {
      type_ops->destroy_value(value.data());
      type_ops->construct_value(value.data());
    }

    void retire_live_authority() noexcept {
      if (!live) {
        return;
      }
      live = false;
      if (live_credit != nullptr) {
        live_credit->fetch_sub(1U, std::memory_order_acq_rel);
      }
      reset_value();
      release_reference();
    }

    static void cancel(void* opaque) noexcept {
      auto* control = static_cast<LoanControl*>(opaque);
      control->retire_live_authority();
    }
  };

  struct OutputState {
    struct EndpointState {
      std::unique_ptr<channel_internal::InlinePayloadPublisher> publisher;
      ClockId plan_capture_clock{};
      ClockId catalog_capture_clock{};
      std::uint32_t expected_recipient_count{1U};
      bool refresh_before_emit{};
      bool peer_required{true};
      bool process_local_transport{};
      bool catalog_encodes_custom_pod{};
      bool process_local_schema_envelope{};
      channel_internal::CatalogCanonicalPodPublisherBinding canonical_pod_publisher;
    };

    PortId port{};
    VertexId owner{};
    Impl* session{};
    SubscriberState* direct_subscriber{};
    void* declaration{};
    const RuntimePortTypeOps* type_ops{};
    std::vector<EndpointState> endpoints;
    std::vector<ManualTensorPublisher> tensor_endpoints;
    struct CatalogTensorEndpointState {
      ChannelPublisher<Tensor> publisher;
      ClockId plan_capture_clock{};
      ClockId catalog_capture_clock{};
      bool peer_required{true};
      bool active_for_current_fanout{};
    };
    std::vector<CatalogTensorEndpointState> catalog_tensor_endpoints;
    struct TensorSchemaEndpointState {
      std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher> publisher;
      std::uint32_t expected_recipient_count{1U};
    };
    std::vector<TensorSchemaEndpointState> tensor_schema_endpoints;
    SchemaId tensor_schema_id{};
    std::uint64_t tensor_max_byte_span{};
    MemoryKind tensor_memory_kind{MemoryKind::kUnknown};
    DeviceId tensor_device{};
    std::optional<TensorPortLayout> tensor_layout;
    std::vector<TensorFieldServiceStorageContract> tensor_fields;
    std::vector<Tensor> tensor_schema_scratch;
    std::optional<LogicalTensorRequirementsRecord> catalog_tensor_requirements;
    RuntimeOutputBinding binding{};
    AlignedBlock encode_scratch;
    AlignedBlock process_local_schema_scratch;
    std::optional<PodIdentityRecord> pod_identity;
    std::optional<SchemaIdentityRecord> process_local_schema_identity;
    std::optional<channel_internal::CatalogSchemaFrameCorruption>
        next_process_local_schema_corruption;
    RuntimePhaseCounters* phase_counters{};
    std::optional<std::size_t> throw_tensor_fanout_after_accepts_for_test;
    std::vector<LoanControl*> loans;
    std::uint64_t next_sequence{1U};
    std::uint64_t active_generation{1U};
    std::uint32_t max_emits_per_activation{};
    std::uint32_t emitted_this_activation{};
    std::uint32_t max_live_loans{};
    std::atomic<std::uint32_t> live_loan_count{};
    bool direct_continuation_capture{};
    bool active{};

    [[nodiscard]] bool tensor_payload() const noexcept {
      return type_ops == runtime_port_type_ops<Tensor>();
    }

    [[nodiscard]] bool tensor_schema_payload() const noexcept {
      return type_ops->codec_family == RuntimePayloadCodecFamily::kDescriptorTensorSchema;
    }

    ~OutputState() {
      for (LoanControl* loan : loans) {
        loan->owner = nullptr;
        loan->live_credit = nullptr;
        loan->release_reference();
      }
    }

    [[nodiscard]] expected<void, Error> classify_connection_refresh(expected<void, Error> refreshed,
                                                                    const bool submitted) noexcept {
      if (refreshed) {
        return {};
      }
      if (!submitted) {
        return make_unexpected(std::move(refreshed).error());
      }
      ++next_sequence;
      ++emitted_this_activation;
      return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
    }

    // Publication distinguishes every pre-submit, accepted, and indeterminate fanout phase in
    // branch order so partial publication can never be retried as an ordinary failure.
    // NOLINTNEXTLINE(readability-function-cognitive-complexity)
    [[nodiscard]] expected<void, Error> emit_payload(
        std::span<const std::byte> payload, SampleMetadata metadata,
        const channel_internal::InlinePayloadEncoder* const encoder = nullptr) noexcept {
      if (!active) {
        return make_unexpected(Error{ErrorCode::kStaleEndpoint});
      }
      if (emitted_this_activation == max_emits_per_activation) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      if (next_sequence == std::numeric_limits<std::uint64_t>::max()) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      const bool direct_encode = encoder != nullptr;
      if ((direct_encode &&
           (type_ops->codec_family != RuntimePayloadCodecFamily::kGeneratedSchema ||
            encoder->capacity != type_ops->wire_capacity || encoder->state == nullptr ||
            encoder->encode == nullptr || direct_subscriber != nullptr ||
            process_local_schema_identity || endpoints.size() != 1U)) ||
          (!direct_encode && payload.empty()) || payload.size() > type_ops->wire_capacity ||
          (type_ops->codec_family == RuntimePayloadCodecFamily::kInlinePod &&
           payload.size() != type_ops->wire_capacity)) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      if (type_ops->codec_family == RuntimePayloadCodecFamily::kCustomPod) {
        if (payload.size() != type_ops->wire_capacity) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
      }
      std::span<const std::byte> process_local_schema_payload;
      if (process_local_schema_identity) {
        if (type_ops->codec_family != RuntimePayloadCodecFamily::kGeneratedSchema ||
            process_local_schema_scratch.data() == nullptr) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        const std::size_t frame_size = kSchemaEnvelopeHeaderSize + payload.size();
        const std::span<std::byte> frame{
            static_cast<std::byte*>(process_local_schema_scratch.data()), frame_size};
        auto envelope = channel_internal::encode_schema_envelope(
            frame.first(kSchemaEnvelopeHeaderSize), *process_local_schema_identity, payload.size());
        if (!envelope) {
          return make_unexpected(std::move(envelope).error());
        }
        std::copy(payload.begin(), payload.end(), frame.begin() + kSchemaEnvelopeHeaderSize);
        if (next_process_local_schema_corruption) {
          corrupt_process_local_schema_frame(frame, *next_process_local_schema_corruption);
          next_process_local_schema_corruption.reset();
        }
        process_local_schema_payload = frame;
      }
      metadata.version = SampleMetadataVersion{};
      metadata.sequence = next_sequence;
      metadata.byte_size = direct_encode ? 0U : static_cast<std::uint32_t>(payload.size());
      metadata.payload_kind = type_ops->payload_kind;
      if (direct_subscriber != nullptr) {
        const bool capture_direct_continuation = direct_continuation_capture;
        const std::uint64_t prior_revision =
            capture_direct_continuation
                ? direct_subscriber->input->readiness_revision.load(std::memory_order_acquire)
                : 0U;
        const std::span<const std::byte> retained_payload =
            process_local_schema_identity ? process_local_schema_payload : payload;
        auto stored =
            session->store_direct_inline_payload(*direct_subscriber, retained_payload, metadata);
        if (!stored) {
          return make_unexpected(std::move(stored).error());
        }
        if (capture_direct_continuation) {
          const std::uint64_t destination_revision =
              direct_subscriber->input->readiness_revision.load(std::memory_order_acquire);
          session->note_direct_continuation_publication(
              *this, *direct_subscriber, metadata.sequence, prior_revision, destination_revision);
        }
        ++next_sequence;
        ++emitted_this_activation;
        return {};
      }
      if (endpoints.empty()) {
        return make_unexpected(Error{ErrorCode::kNotReady});
      }

      // Validate every catalog-bound metadata projection before any endpoint accepts the sample.
      // This keeps a deterministic plan/catalog identity error on the pre-submit side even when
      // the same Output also fans out to internal recipients earlier in endpoint order.
      for (const auto& endpoint : endpoints) {
        if (!endpoint.refresh_before_emit) {
          continue;
        }
        if (endpoint.plan_capture_clock.valid() != endpoint.catalog_capture_clock.valid()) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        if (endpoint.plan_capture_clock.valid()) {
          if (metadata.source_clock_id != endpoint.plan_capture_clock) {
            return make_unexpected(Error{ErrorCode::kInvalidArgument});
          }
        } else if (metadata.source_clock_id.valid() || metadata.capture_timestamp_ns != 0) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
      }

      trace_output_emit_phase(phase_counters,
                              RuntimePhaseTracePoint::kInlinePayloadTransportPublicationStarted,
                              port,
                              next_sequence);
      bool submitted = false;
      for (const auto& endpoint : endpoints) {
        SampleMetadata endpoint_metadata = metadata;
        if (endpoint.plan_capture_clock.valid()) {
          // Internal fanout continues to observe the plan-local metadata above. Only the copy sent
          // through this catalog endpoint enters the independently dense catalog clock namespace.
          endpoint_metadata.source_clock_id = endpoint.catalog_capture_clock;
        }
        // External peers may attach after Graph startup, which is deliberately never a peer-wait
        // barrier. Refresh immediately before the submission boundary so a late peer can
        // participate without converting an earlier successful refresh into a first-sample
        // guarantee. A refresh failure is still structurally pre-submit for this endpoint; after
        // an earlier fanout endpoint accepted the sample, the same failure is necessarily an
        // terminal partial or indeterminate publication outcome.
        if (endpoint.refresh_before_emit) {
          auto refreshed =
              classify_connection_refresh(endpoint.publisher->update_connections(), submitted);
          if (!refreshed) {
            return make_unexpected(std::move(refreshed).error());
          }
        }
        channel_internal::InlinePublicationResult published;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
        try {
#endif
          const std::span<const std::byte> endpoint_payload =
              endpoint.process_local_schema_envelope ? process_local_schema_payload : payload;
          if (endpoint_payload.empty() && !direct_encode) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
          published =
              direct_encode ? endpoint.publisher->emit_encoded_phase(*encoder, endpoint_metadata)
              : endpoint.canonical_pod_publisher
                  ? endpoint.canonical_pod_publisher.emit_phase(
                        endpoint.canonical_pod_publisher.state, endpoint_payload, endpoint_metadata)
                  : endpoint.publisher->emit_phase(endpoint_payload, endpoint_metadata);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
        } catch (...) {
          if (endpoint.process_local_transport) {
            session->invalidate_process_local_progress();
          }
          ++next_sequence;
          ++emitted_this_activation;
          return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
        }
#endif
        switch (published.phase) {
          case channel_internal::InlinePublicationPhase::kAccepted:
            if (endpoint.process_local_transport && published.recipient_count != 0U) {
              session->record_process_local_publication(published.recipient_count);
            }
            if (published.recipient_count != endpoint.expected_recipient_count &&
                (endpoint.peer_required || published.recipient_count != 0U)) {
              ++next_sequence;
              ++emitted_this_activation;
              if (published.recipient_count < endpoint.expected_recipient_count) {
                return make_unexpected(make_publication_partial_error(
                    published.recipient_count, endpoint.expected_recipient_count));
              }
              return make_unexpected(
                  Error{ErrorCode::kProtocolError, "RUNTIME_PUBLICATION_RECIPIENT_COUNT_EXCESS"});
            }
            // An optional endpoint with no attached peer is an intentional local drop. No remote
            // observer exists, so it must neither fail the activation nor make a later pre-submit
            // failure on another endpoint indeterminate.
            submitted = submitted || published.recipient_count != 0U;
            break;
          case channel_internal::InlinePublicationPhase::kPostSubmitUncertain:
            if (endpoint.process_local_transport) {
              session->invalidate_process_local_progress();
            }
            ++next_sequence;
            ++emitted_this_activation;
            if (publication_error_consumes_authority(published.error.code) ||
                published.error.code == ErrorCode::kProtocolError ||
                published.error.code == ErrorCode::kInvalidArgument) {
              return make_unexpected(std::move(published.error));
            }
            return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
          case channel_internal::InlinePublicationPhase::kPreSubmitRejected:
            if (!endpoint.peer_required && published.error.code == ErrorCode::kNotReady) {
              // Some backends report the same unobserved optional-peer drop as a pre-submit
              // readiness miss rather than an accepted zero-recipient publication.
              break;
            }
            if (!submitted) {
              return make_unexpected(published.error);
            }
            ++next_sequence;
            ++emitted_this_activation;
            return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
          default:
            // A malformed private phase is never equivalent to the proven no-peer rejection
            // above. Preserve any earlier fanout observation in the returned classification.
            if (endpoint.process_local_transport) {
              session->invalidate_process_local_progress();
            }
            if (!submitted) {
              return make_unexpected(Error{ErrorCode::kProtocolError});
            }
            ++emitted_this_activation;
            return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
        }
      }
      trace_output_emit_phase(phase_counters,
                              RuntimePhaseTracePoint::kInlinePayloadTransportPublicationReturned,
                              port,
                              next_sequence);
      ++next_sequence;
      ++emitted_this_activation;
      return {};
    }

    [[nodiscard]] static expected<void, Error> validate_emit_authority(
        const OutputState& state) noexcept {
      if (!state.active) {
        return make_unexpected(Error{ErrorCode::kStaleEndpoint});
      }
      if (state.emitted_this_activation == state.max_emits_per_activation ||
          state.next_sequence == std::numeric_limits<std::uint64_t>::max()) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      return {};
    }

    [[nodiscard]] static expected<void, Error> classify_tensor_publication(
        OutputState& state, expected<std::uint32_t, Error> published, bool& submitted) noexcept {
      if (published) {
        submitted = true;
        return {};
      }
      return classify_tensor_failure(state, std::move(published).error(), submitted);
    }

    [[nodiscard]] static expected<void, Error> classify_tensor_failure(
        OutputState& state, Error error, const bool submitted) noexcept {
      if (publication_error_consumes_authority(error.code)) {
        ++state.next_sequence;
        ++state.emitted_this_activation;
        return make_unexpected(std::move(error));
      }
      if (submitted) {
        ++state.next_sequence;
        ++state.emitted_this_activation;
        return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
      }
      // Manual Tensor publishers report bounded HoloMQ loan/callback capacity as
      // kResourceExhausted. No endpoint has accepted the sample at this boundary, so the
      // graph-bound Output contract classifies the same proven pre-submit pressure as
      // kBackpressured without consuming sequence or publication authority.
      if (error.code == ErrorCode::kResourceExhausted) {
        error = Error{ErrorCode::kBackpressured};
      }
      return make_unexpected(std::move(error));
    }

    static void inject_tensor_fanout_failure(OutputState& state, std::size_t accepted_count) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      if (state.throw_tensor_fanout_after_accepts_for_test == accepted_count) {
        state.throw_tensor_fanout_after_accepts_for_test.reset();
        throw std::runtime_error{accepted_count == 0U
                                     ? "injected pre-submit Tensor fanout failure"
                                     : "injected post-submit Tensor fanout failure"};
      }
#else
      (void)state;
      (void)accepted_count;
#endif
    }

    [[nodiscard]] static expected<void, Error> validate_catalog_tensor_metadata(
        const OutputState& state, const SampleMetadata& metadata) noexcept {
      // Plan and endpoint catalogs use independently dense clock IDs. Validate every projection
      // before any fanout arm can accept the sample so a provenance mismatch remains pre-submit.
      for (const auto& endpoint : state.catalog_tensor_endpoints) {
        if (endpoint.plan_capture_clock.valid() != endpoint.catalog_capture_clock.valid()) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        if (endpoint.plan_capture_clock.valid()) {
          if (metadata.source_clock_id != endpoint.plan_capture_clock) {
            return make_unexpected(Error{ErrorCode::kInvalidArgument});
          }
        } else if (metadata.source_clock_id.valid() || metadata.capture_timestamp_ns != 0) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
      }
      return {};
    }

    [[nodiscard]] static expected<void, Error> emit_internal_tensor_endpoints(
        OutputState& state, Sample<Tensor>& normalized, bool& submitted) {
      if (state.tensor_endpoints.empty()) {
        return {};
      }
      for (std::size_t index = 0U; index + 1U < state.tensor_endpoints.size(); ++index) {
        auto published = state.tensor_endpoints[index].emit_sample(normalized);
        if (!published && publication_error_consumes_authority(published.error().code)) {
          state.session->invalidate_process_local_progress();
        }
        const std::uint32_t delivered = published ? *published : 0U;
        auto classified = classify_tensor_publication(state, std::move(published), submitted);
        if (!classified) {
          return classified;
        }
        state.session->record_process_local_publication(delivered);
        inject_tensor_fanout_failure(state, index + 1U);
      }
      auto published = state.tensor_endpoints.back().emit_sample(std::move(normalized));
      if (!published && publication_error_consumes_authority(published.error().code)) {
        state.session->invalidate_process_local_progress();
      }
      const std::uint32_t delivered = published ? *published : 0U;
      auto classified = classify_tensor_publication(state, std::move(published), submitted);
      if (!classified) {
        return classified;
      }
      state.session->record_process_local_publication(delivered);
      inject_tensor_fanout_failure(state, state.tensor_endpoints.size());
      return {};
    }

    struct CatalogTensorFanoutPreparation {
      bool copy_required{};
      std::size_t staging_index{std::numeric_limits<std::size_t>::max()};
      std::optional<channel_internal::StagedTensorCopy> staged_copy;
    };

    [[nodiscard]] static expected<CatalogTensorFanoutPreparation, Error>
    prepare_catalog_tensor_fanout(OutputState& state, const Tensor& supplied) {
      CatalogTensorFanoutPreparation preparation;
      for (std::size_t index = 0U; index < state.catalog_tensor_endpoints.size(); ++index) {
        auto& endpoint = state.catalog_tensor_endpoints[index];
        endpoint.active_for_current_fanout = endpoint.peer_required;
        // Optional endpoints retain their pre-submit skip semantics. Required endpoints defer
        // topology admission to HoloMQ's exact recipient-count gate, avoiding a redundant shared
        // dynamic-config read on the ordinary publication path.
        if (!endpoint.peer_required && !endpoint.publisher.has_subscribers()) {
          auto refreshed =
              state.classify_connection_refresh(endpoint.publisher.update_connections(), false);
          if (!refreshed) {
            return make_unexpected(std::move(refreshed).error());
          }
          if (!endpoint.publisher.has_subscribers()) {
            continue;
          }
        }
        endpoint.active_for_current_fanout = true;

        auto disposition = channel_internal::TensorPublishLoanAccess::classify_foreign_tensor(
            endpoint.publisher, supplied);
        if (!disposition) {
          return make_unexpected(std::move(disposition).error());
        }
        if (preparation.staging_index == std::numeric_limits<std::size_t>::max()) {
          preparation.staging_index = index;
        }
        preparation.copy_required |=
            *disposition == channel_internal::ForeignTensorDisposition::kCopyRequired;
      }
      if (preparation.copy_required) {
        if (preparation.staging_index >= state.catalog_tensor_endpoints.size()) {
          return make_unexpected(Error{ErrorCode::kFailure});
        }
        auto staged = channel_internal::TensorPublishLoanAccess::stage_foreign_tensor(
            state.catalog_tensor_endpoints[preparation.staging_index].publisher,
            supplied,
            TensorAccess::readiness_stream(supplied));
        if (!staged) {
          Error error = std::move(staged).error();
          if (error.code == ErrorCode::kResourceExhausted) {
            error = Error{ErrorCode::kBackpressured};
          }
          return make_unexpected(std::move(error));
        }
        preparation.staged_copy.emplace(std::move(*staged));
      }
      return preparation;
    }

    [[nodiscard]] static expected<void, Error> emit_catalog_tensor_endpoint(
        OutputState& state, CatalogTensorEndpointState& endpoint, const Tensor& supplied,
        const SampleMetadata& normalized_metadata, CatalogTensorFanoutPreparation& preparation,
        bool& submitted) {
      if (!endpoint.active_for_current_fanout) {
        return {};
      }
      SampleMetadata endpoint_metadata = normalized_metadata;
      endpoint_metadata.publish_clock_id = {};
      endpoint_metadata.publish_timestamp_ns = 0;
      if (endpoint.plan_capture_clock.valid()) {
        endpoint_metadata.source_clock_id = endpoint.catalog_capture_clock;
      }
      expected<std::uint32_t, Error> published = make_unexpected(Error{ErrorCode::kFailure});
      if (!preparation.copy_required) {
        published = channel_internal::TensorPublishLoanAccess::emit_prevalidated_exportable_tensor(
            endpoint.publisher, supplied, endpoint_metadata);
      } else if (preparation.staged_copy) {
        published = channel_internal::TensorPublishLoanAccess::emit_prevalidated_exportable_tensor(
            endpoint.publisher, preparation.staged_copy->tensor, endpoint_metadata);
      }
      if (!endpoint.peer_required && !published && published.error().code == ErrorCode::kNotReady) {
        // The optional peer disappeared after preparation but before submit. HoloMQ proved the
        // rejection occurred before submission, so the arm remains an optional no-op.
        return {};
      }
      return classify_tensor_publication(state, std::move(published), submitted);
    }

    [[nodiscard]] static expected<void, Error> emit_tensor_fanout(
        OutputState& state, const Sample<Tensor>& supplied, Sample<Tensor> normalized) noexcept {
      auto metadata_valid = validate_catalog_tensor_metadata(state, normalized.metadata);
      if (!metadata_valid) {
        return metadata_valid;
      }
      const SampleMetadata normalized_metadata = normalized.metadata;
      bool submitted = false;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        inject_tensor_fanout_failure(state, 0U);
        auto catalog_preparation = prepare_catalog_tensor_fanout(state, supplied.data);
        if (!catalog_preparation) {
          return make_unexpected(std::move(catalog_preparation).error());
        }
        trace_output_emit_phase(state.phase_counters,
                                RuntimePhaseTracePoint::kTensorTransportPublicationStarted,
                                state.port,
                                normalized_metadata.sequence);
        auto internal = emit_internal_tensor_endpoints(state, normalized, submitted);
        if (!internal) {
          return internal;
        }
        for (std::size_t index = 0U; index < state.catalog_tensor_endpoints.size(); ++index) {
          auto published = emit_catalog_tensor_endpoint(state,
                                                        state.catalog_tensor_endpoints[index],
                                                        supplied.data,
                                                        normalized_metadata,
                                                        *catalog_preparation,
                                                        submitted);
          if (!published) {
            return published;
          }
        }
        trace_output_emit_phase(state.phase_counters,
                                RuntimePhaseTracePoint::kTensorTransportPublicationReturned,
                                state.port,
                                normalized_metadata.sequence);
        ++state.next_sequence;
        ++state.emitted_this_activation;
        return {};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (...) {
        if (!state.tensor_endpoints.empty()) {
          state.session->invalidate_process_local_progress();
        }
        if (submitted) {
          ++state.next_sequence;
          ++state.emitted_this_activation;
          return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
        }
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }

    [[nodiscard]] static expected<void, Error> emit_tensor_sample(
        OutputState& state, const Sample<Tensor>& supplied) noexcept {
      auto authority = validate_emit_authority(state);
      if (!authority) {
        return authority;
      }
      if (state.direct_subscriber == nullptr && state.tensor_endpoints.empty() &&
          state.catalog_tensor_endpoints.empty()) {
        return make_unexpected(Error{ErrorCode::kNotReady});
      }
      const std::uint64_t emitted_sequence = state.next_sequence;
      trace_output_emit_phase(state.phase_counters,
                              RuntimePhaseTracePoint::kTensorEmitStarted,
                              state.port,
                              emitted_sequence);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        const TensorAccessSnapshot snapshot = TensorAccess::snapshot(supplied.data);
        const bool has_internal_tensor_recipient =
            state.direct_subscriber != nullptr || !state.tensor_endpoints.empty();
        const bool native_contract_matches =
            !has_internal_tensor_recipient ||
            tensor_snapshot_matches_plan(snapshot,
                                         state.tensor_max_byte_span,
                                         state.tensor_memory_kind,
                                         state.tensor_device);
        const bool catalog_contract_matches =
            !state.catalog_tensor_requirements ||
            (snapshot.initialized && snapshot.valid &&
             state.catalog_tensor_requirements->max_byte_span &&
             snapshot.byte_span <= *state.catalog_tensor_requirements->max_byte_span &&
             tensor_matches_logical_requirements(supplied.data,
                                                 *state.catalog_tensor_requirements));
        if (!native_contract_matches || !catalog_contract_matches) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        Sample<Tensor> normalized{supplied};
        normalized.metadata.version = SampleMetadataVersion{};
        normalized.metadata.sequence = state.next_sequence;
        normalized.metadata.byte_size = snapshot.byte_size;
        normalized.metadata.payload_kind = PayloadKind::kDescriptorBacked;
        trace_output_emit_phase(state.phase_counters,
                                RuntimePhaseTracePoint::kTensorEmitValidated,
                                state.port,
                                emitted_sequence);
        if (state.direct_subscriber != nullptr) {
          // REQ: CHAN-098
          if (!state.tensor_endpoints.empty() || !state.catalog_tensor_endpoints.empty()) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
          auto stored = state.session->store_direct_tensor_sample(*state.direct_subscriber,
                                                                  std::move(normalized));
          if (!stored) {
            return stored;
          }
          ++state.next_sequence;
          ++state.emitted_this_activation;
          trace_output_emit_phase(state.phase_counters,
                                  RuntimePhaseTracePoint::kTensorEmitCompleted,
                                  state.port,
                                  emitted_sequence);
          return {};
        }
        auto emitted = emit_tensor_fanout(state, supplied, std::move(normalized));
        if (emitted) {
          trace_output_emit_phase(state.phase_counters,
                                  RuntimePhaseTracePoint::kTensorEmitCompleted,
                                  state.port,
                                  emitted_sequence);
        }
        return emitted;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
    }

    [[nodiscard]] static expected<void, Error> emit_tensor_schema_fanout(
        OutputState& state, const std::span<const Tensor> tensors,
        const std::span<std::byte> destination, const bool require_direct_export) noexcept {
      bool submitted = false;
      for (auto& endpoint : state.tensor_schema_endpoints) {
        auto reserved = endpoint.publisher->reserve_tensors(tensors, require_direct_export);
        if (!reserved) {
          return classify_tensor_failure(state, std::move(reserved).error(), submitted);
        }
        if (reserved->field_count != tensors.size()) {
          endpoint.publisher->release_tensor(std::move(*reserved));
          return classify_tensor_failure(state, Error{ErrorCode::kProtocolError}, submitted);
        }
        for (std::size_t index = 0U; index < tensors.size(); ++index) {
          auto* field = reserved->field_at(index);
          if (field == nullptr) {
            endpoint.publisher->release_tensor(std::move(*reserved));
            return classify_tensor_failure(state, Error{ErrorCode::kProtocolError}, submitted);
          }
          field->descriptor.schema_id = state.tensor_schema_id;
          auto descriptor_encoded = channel_internal::encode_payload_descriptor_wire(
              field->descriptor,
              destination.subspan(
                  state.type_ops->descriptor_offset + index * sizeof(PayloadDescriptor),
                  sizeof(PayloadDescriptor)));
          if (!descriptor_encoded) {
            endpoint.publisher->release_tensor(std::move(*reserved));
            return classify_tensor_failure(state, std::move(descriptor_encoded).error(), submitted);
          }
        }
        auto published = endpoint.publisher->publish_payload(destination, std::move(*reserved));
        if (!published) {
          if (publication_error_consumes_authority(published.error().code)) {
            state.session->invalidate_process_local_progress();
          }
          return classify_tensor_failure(state, std::move(published).error(), submitted);
        }
        if (*published != endpoint.expected_recipient_count) {
          ++state.next_sequence;
          ++state.emitted_this_activation;
          if (*published < endpoint.expected_recipient_count) {
            return make_unexpected(
                make_publication_partial_error(*published, endpoint.expected_recipient_count));
          }
          return make_unexpected(
              Error{ErrorCode::kProtocolError, "RUNTIME_PUBLICATION_RECIPIENT_COUNT_EXCESS"});
        }
        submitted = true;
        state.session->record_process_local_publication(*published);
      }
      ++state.next_sequence;
      ++state.emitted_this_activation;
      return {};
    }

    struct TensorSchemaEncodeContext {
      OutputState* output{};
      const void* payload{};
      std::span<const Tensor> tensors{};
      SampleMetadata metadata{};
      bool separate_tensor{};
    };

    [[nodiscard]] static expected<std::size_t, Error> encode_tensor_schema_transport(
        void* opaque, const std::span<std::byte> destination,
        const channel_internal::TensorSchemaPayloadReservation& reservation) noexcept {
      auto& context = *static_cast<TensorSchemaEncodeContext*>(opaque);
      OutputState& state = *context.output;
      expected<std::size_t, Error> encoded = make_unexpected(Error{ErrorCode::kProtocolError});
      if (context.separate_tensor) {
        if (context.tensors.size() == 1U &&
            state.type_ops->encode_descriptor_tensor_loan != nullptr) {
          encoded = state.type_ops->encode_descriptor_tensor_loan(context.payload,
                                                                  context.tensors.front(),
                                                                  context.metadata,
                                                                  state.session->plan_generation_,
                                                                  destination);
        } else if (state.type_ops->encode_descriptor_tensor_loans != nullptr) {
          encoded = state.type_ops->encode_descriptor_tensor_loans(context.payload,
                                                                   context.tensors,
                                                                   context.metadata,
                                                                   state.session->plan_generation_,
                                                                   destination);
        }
      } else if (state.type_ops->encode_descriptor_tensor_sample != nullptr) {
        encoded = state.type_ops->encode_descriptor_tensor_sample(
            context.payload, context.metadata, state.session->plan_generation_, destination);
      }
      if (!encoded || *encoded == 0U || *encoded > destination.size() ||
          reservation.field_count != context.tensors.size() ||
          state.type_ops->descriptor_offset > *encoded ||
          context.tensors.size() * sizeof(PayloadDescriptor) >
              *encoded - state.type_ops->descriptor_offset) {
        return encoded ? make_unexpected(Error{ErrorCode::kProtocolError})
                       : make_unexpected(std::move(encoded).error());
      }
      for (std::size_t index = 0U; index < context.tensors.size(); ++index) {
        const auto* field = reservation.field_at(index);
        if (field == nullptr) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        PayloadDescriptor descriptor = field->descriptor;
        descriptor.schema_id = state.tensor_schema_id;
        auto descriptor_encoded = channel_internal::encode_payload_descriptor_wire(
            descriptor,
            destination.subspan(
                state.type_ops->descriptor_offset + index * sizeof(PayloadDescriptor),
                sizeof(PayloadDescriptor)));
        if (!descriptor_encoded) {
          return make_unexpected(std::move(descriptor_encoded).error());
        }
      }
      return *encoded;
    }

    [[nodiscard]] static expected<void, Error> emit_tensor_schema_direct_loan(
        OutputState& state, const void* payload, const std::span<const Tensor> tensors,
        const SampleMetadata metadata, const bool require_direct_export) noexcept {
      if (state.tensor_schema_endpoints.size() != 1U) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      auto& endpoint = state.tensor_schema_endpoints.front();
      auto reservation = endpoint.publisher->reserve_tensors(tensors, require_direct_export);
      if (!reservation) {
        return classify_tensor_failure(state, std::move(reservation).error(), false);
      }
      TensorSchemaEncodeContext context{
          .output = &state,
          .payload = payload,
          .tensors = tensors,
          .metadata = metadata,
          .separate_tensor = require_direct_export,
      };
      const channel_internal::TensorSchemaPayloadEncoder encoder{
          .state = &context,
          .encode = &encode_tensor_schema_transport,
          .capacity = state.type_ops->wire_capacity,
      };
      auto published =
          endpoint.publisher->publish_encoded_payload(encoder, std::move(*reservation));
      if (!published) {
        if (publication_error_consumes_authority(published.error().code)) {
          state.session->invalidate_process_local_progress();
        }
        return classify_tensor_failure(state, std::move(published).error(), false);
      }
      if (*published != endpoint.expected_recipient_count) {
        ++state.next_sequence;
        ++state.emitted_this_activation;
        return make_unexpected(
            *published < endpoint.expected_recipient_count
                ? make_publication_partial_error(*published, endpoint.expected_recipient_count)
                : Error{ErrorCode::kProtocolError, "RUNTIME_PUBLICATION_RECIPIENT_COUNT_EXCESS"});
      }
      state.session->record_process_local_publication(*published);
      ++state.next_sequence;
      ++state.emitted_this_activation;
      return {};
    }

    // Schema emission is one acceptance transaction across direct and endpoint fanout paths.
    // NOLINTNEXTLINE(readability-function-cognitive-complexity)
    [[nodiscard]] static expected<void, Error> emit_tensor_schema_value(
        OutputState& state, const void* payload, const std::span<const Tensor> tensors,
        const SampleMetadata& metadata, const bool separate_tensor) noexcept {
      auto authority = validate_emit_authority(state);
      if (!authority) {
        return authority;
      }
      const std::size_t tensor_count = state.type_ops->descriptor_tensor_static_count;
      if ((state.direct_subscriber == nullptr && state.tensor_schema_endpoints.empty()) ||
          state.encode_scratch.data() == nullptr || tensor_count == 0U ||
          tensor_count != tensors.size() ||
          tensor_count > channel_internal::kMaximumDescriptorTensorFields ||
          state.type_ops->descriptor_offset > state.type_ops->wire_capacity ||
          tensor_count * sizeof(PayloadDescriptor) >
              state.type_ops->wire_capacity - state.type_ops->descriptor_offset ||
          (separate_tensor && state.type_ops->encode_descriptor_tensor_loans == nullptr &&
           (tensor_count != 1U || state.type_ops->encode_descriptor_tensor_loan == nullptr))) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      if (!state.tensor_fields.empty() && state.tensor_fields.size() != tensor_count) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      std::uint64_t aggregate_byte_size = 0U;
      for (std::size_t index = 0U; index < tensor_count; ++index) {
        const TensorAccessSnapshot snapshot = TensorAccess::snapshot(tensors[index]);
        const bool matches =
            state.tensor_fields.empty()
                ? tensor_snapshot_matches_plan(snapshot,
                                               state.tensor_max_byte_span,
                                               state.tensor_memory_kind,
                                               state.tensor_device) &&
                      (!state.tensor_layout ||
                       (tensor_count == 1U &&
                        tensor_matches_port_layout(tensors[index], snapshot, *state.tensor_layout)))
                : state.tensor_fields[index].tensor_index == index &&
                      tensor_matches_field_contract(
                          tensors[index], snapshot, state.tensor_fields[index]);
        if (!matches ||
            snapshot.byte_size > std::numeric_limits<std::uint64_t>::max() - aggregate_byte_size) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        aggregate_byte_size += snapshot.byte_size;
      }
      SampleMetadata normalized = metadata;
      normalized.version = SampleMetadataVersion{};
      normalized.sequence = state.next_sequence;
      normalized.byte_size = aggregate_byte_size;
      normalized.payload_kind = PayloadKind::kDescriptorBacked;
      if (state.direct_subscriber == nullptr && state.tensor_schema_endpoints.size() == 1U) {
        return emit_tensor_schema_direct_loan(state, payload, tensors, normalized, separate_tensor);
      }
      const std::span<std::byte> destination{static_cast<std::byte*>(state.encode_scratch.data()),
                                             state.type_ops->wire_capacity};
      expected<std::size_t, Error> encoded = make_unexpected(Error{ErrorCode::kProtocolError});
      if (separate_tensor) {
        if (tensor_count == 1U && state.type_ops->encode_descriptor_tensor_loan != nullptr) {
          encoded = state.type_ops->encode_descriptor_tensor_loan(
              payload, tensors.front(), normalized, state.session->plan_generation_, destination);
        } else {
          encoded = state.type_ops->encode_descriptor_tensor_loans(
              payload, tensors, normalized, state.session->plan_generation_, destination);
        }
      } else {
        if (state.type_ops->encode_descriptor_tensor_sample == nullptr) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        encoded = state.type_ops->encode_descriptor_tensor_sample(
            payload, normalized, state.session->plan_generation_, destination);
      }
      if (!encoded) {
        return make_unexpected(std::move(encoded).error());
      }
      if (*encoded == 0U || *encoded > state.type_ops->wire_capacity ||
          state.type_ops->descriptor_offset > *encoded ||
          tensor_count * sizeof(PayloadDescriptor) > *encoded - state.type_ops->descriptor_offset) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      if (state.direct_subscriber != nullptr) {
        // REQ: CHAN-099
        if (!state.tensor_schema_endpoints.empty()) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        auto retained = state.session->store_direct_tensor_schema_sample(
            *state.direct_subscriber, destination.first(*encoded), tensors);
        if (!retained) {
          return retained;
        }
        ++state.next_sequence;
        ++state.emitted_this_activation;
        return {};
      }
      return emit_tensor_schema_fanout(
          state, tensors, destination.first(*encoded), separate_tensor);
    }

    [[nodiscard]] static expected<void, Error> emit_tensor_schema_sample(
        OutputState& state, const void* payload, const SampleMetadata& metadata) noexcept {
      const std::uint32_t tensor_count = state.type_ops->descriptor_tensor_static_count;
      if (state.type_ops->descriptor_tensor_count == nullptr ||
          state.type_ops->descriptor_tensor_at == nullptr || tensor_count == 0U ||
          tensor_count > channel_internal::kMaximumDescriptorTensorFields ||
          state.type_ops->descriptor_tensor_count(payload) != tensor_count) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      auto& tensors = state.tensor_schema_scratch;
      if (tensors.size() != tensor_count) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      const TensorScratchReset reset{tensors};
      for (std::uint32_t index = 0U; index < tensor_count; ++index) {
        const Tensor* tensor = state.type_ops->descriptor_tensor_at(payload, index);
        if (tensor == nullptr) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        tensors[index] = *tensor;
      }
      return emit_tensor_schema_value(
          state, payload, std::span<const Tensor>{tensors}, metadata, false);
    }

    struct GeneratedSchemaEncodeContext {
      OutputState* output{};
      const void* payload{};
      std::uint64_t sequence{};
    };

    [[nodiscard]] static expected<std::size_t, Error> encode_generated_schema(
        void* opaque, const std::span<std::byte> destination) noexcept {
      auto& context = *static_cast<GeneratedSchemaEncodeContext*>(opaque);
      if (context.output == nullptr || context.payload == nullptr ||
          context.output->type_ops->encode_payload == nullptr) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      auto encoded = context.output->type_ops->encode_payload(context.payload, destination);
      if (encoded) {
        trace_output_emit_phase(context.output->phase_counters,
                                RuntimePhaseTracePoint::kInlinePayloadEncoded,
                                context.output->port,
                                context.sequence);
      }
      return encoded;
    }

    [[nodiscard]] static expected<void, Error> emit_generated_schema_payload(
        OutputState& state, const void* payload, const SampleMetadata& metadata,
        const std::uint64_t emitted_sequence) noexcept {
      if (state.type_ops->encode_payload == nullptr || state.encode_scratch.data() == nullptr) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      GeneratedSchemaEncodeContext context{
          .output = &state,
          .payload = payload,
          .sequence = emitted_sequence,
      };
      const channel_internal::InlinePayloadEncoder encoder{
          .state = &context,
          .encode = &encode_generated_schema,
          .capacity = state.type_ops->wire_capacity,
      };
      if (state.direct_subscriber == nullptr && !state.process_local_schema_identity &&
          state.endpoints.size() == 1U) {
        auto published = state.emit_payload({}, metadata, &encoder);
        if (published || published.error().code != ErrorCode::kNotSupported) {
          return published;
        }
      }
      const std::span<std::byte> destination{static_cast<std::byte*>(state.encode_scratch.data()),
                                             state.type_ops->wire_capacity};
      auto encoded = encode_generated_schema(&context, destination);
      if (!encoded) {
        return make_unexpected(std::move(encoded).error());
      }
      return state.emit_payload(destination.first(*encoded), metadata);
    }

    [[nodiscard]] static expected<void, Error> emit_non_tensor_sample(
        OutputState& state, const void* payload, const SampleMetadata& metadata) noexcept {
      const std::uint64_t emitted_sequence = state.next_sequence;
      trace_output_emit_phase(state.phase_counters,
                              RuntimePhaseTracePoint::kInlinePayloadEmitStarted,
                              state.port,
                              emitted_sequence);
      expected<void, Error> emitted = make_unexpected(Error{ErrorCode::kFailure});
      if (state.type_ops->codec_family == RuntimePayloadCodecFamily::kGeneratedSchema) {
        emitted = emit_generated_schema_payload(state, payload, metadata, emitted_sequence);
      } else if (state.type_ops->codec_family == RuntimePayloadCodecFamily::kCustomPod) {
        if (!state.pod_identity || state.encode_scratch.data() == nullptr) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        const std::span<const std::byte> native{static_cast<const std::byte*>(payload),
                                                state.type_ops->payload_size};
        const std::span<std::byte> destination{static_cast<std::byte*>(state.encode_scratch.data()),
                                               state.type_ops->wire_capacity};
        auto encoded = channel_internal::PodFieldwiseCodec::encode_prevalidated(
            *state.pod_identity, native, destination);
        if (!encoded) {
          return make_unexpected(std::move(encoded).error());
        }
        trace_output_emit_phase(state.phase_counters,
                                RuntimePhaseTracePoint::kInlinePayloadEncoded,
                                state.port,
                                emitted_sequence);
        emitted = state.emit_payload(destination, metadata);
      } else {
        trace_output_emit_phase(state.phase_counters,
                                RuntimePhaseTracePoint::kInlinePayloadEncoded,
                                state.port,
                                emitted_sequence);
        emitted = state.emit_payload(
            {static_cast<const std::byte*>(payload), state.type_ops->wire_capacity}, metadata);
      }
      if (emitted) {
        trace_output_emit_phase(state.phase_counters,
                                RuntimePhaseTracePoint::kInlinePayloadEmitCompleted,
                                state.port,
                                emitted_sequence);
      }
      return emitted;
    }

    [[nodiscard]] static expected<void, Error> emit_sample(void* opaque,
                                                           const void* sample) noexcept {
      auto& state = *static_cast<OutputState*>(opaque);
      count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitAttempts);
      auto published = [&]() -> expected<void, Error> {
        if (sample == nullptr) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        if (state.tensor_payload()) {
          return emit_tensor_sample(state, *static_cast<const Sample<Tensor>*>(sample));
        }
        const void* payload = state.type_ops->sample_payload(sample);
        const SampleMetadata* metadata = state.type_ops->sample_metadata(sample);
        if (payload == nullptr || metadata == nullptr) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        return state.tensor_schema_payload() ? emit_tensor_schema_sample(state, payload, *metadata)
                                             : emit_non_tensor_sample(state, payload, *metadata);
      }();
      if (published) {
        count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitAccepted);
      } else if (published.error().code == ErrorCode::kBackpressured) {
        count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitBackpressured);
      }
      return published;
    }

    [[nodiscard]] static expected<void, Error> emit_descriptor_tensor(
        void* opaque, const void* descriptor, const Tensor* tensor, EmitOptions options) noexcept {
      auto& state = *static_cast<OutputState*>(opaque);
      count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitAttempts);
      auto published = [&]() -> expected<void, Error> {
        if (descriptor == nullptr || tensor == nullptr || !state.tensor_schema_payload()) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        const TensorAccessSnapshot snapshot = TensorAccess::snapshot(*tensor);
        const std::array tensors{*tensor};
        return emit_tensor_schema_value(
            state, descriptor, tensors, metadata_from_options(options, snapshot.byte_size), true);
      }();
      if (published) {
        count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitAccepted);
      } else if (published.error().code == ErrorCode::kBackpressured) {
        count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitBackpressured);
      }
      return published;
    }

    [[nodiscard]] static expected<void, Error> emit_descriptor_tensors(
        void* opaque, const void* descriptor, const std::span<const Tensor> tensors,
        EmitOptions options) noexcept {
      auto& state = *static_cast<OutputState*>(opaque);
      count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitAttempts);
      auto published = [&]() -> expected<void, Error> {
        if (descriptor == nullptr || tensors.empty() || !state.tensor_schema_payload()) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        std::uint64_t byte_size = 0U;
        for (const Tensor& tensor : tensors) {
          const TensorAccessSnapshot snapshot = TensorAccess::snapshot(tensor);
          if (!snapshot.initialized || !snapshot.valid ||
              snapshot.byte_size > std::numeric_limits<std::uint64_t>::max() - byte_size) {
            return make_unexpected(Error{ErrorCode::kInvalidArgument});
          }
          byte_size += snapshot.byte_size;
        }
        return emit_tensor_schema_value(
            state, descriptor, tensors, metadata_from_options(options, byte_size), true);
      }();
      if (published) {
        count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitAccepted);
      } else if (published.error().code == ErrorCode::kBackpressured) {
        count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitBackpressured);
      }
      return published;
    }

    [[nodiscard]] static expected<RuntimeLoanAllocation, Error> allocate(void* opaque) noexcept {
      auto& state = *static_cast<OutputState*>(opaque);
      if (!state.active) {
        return make_unexpected(Error{ErrorCode::kStaleEndpoint});
      }
      const std::uint32_t prior = state.live_loan_count.fetch_add(1U, std::memory_order_acq_rel);
      if (prior >= state.max_live_loans) {
        state.live_loan_count.fetch_sub(1U, std::memory_order_acq_rel);
        return make_unexpected(
            Error{state.loans.empty() ? ErrorCode::kNotSupported : ErrorCode::kResourceExhausted});
      }
      for (LoanControl* loan : state.loans) {
        if (!loan->live) {
          loan->live = true;
          loan->activation_generation = state.active_generation;
          loan->references.fetch_add(1U, std::memory_order_relaxed);
          return RuntimeLoanAllocation{.value = loan->value.data(), .loan = &loan->binding};
        }
      }
      state.live_loan_count.fetch_sub(1U, std::memory_order_acq_rel);
      return make_unexpected(
          Error{state.loans.empty() ? ErrorCode::kNotSupported : ErrorCode::kResourceExhausted});
    }

    [[nodiscard]] static expected<void, Error> emit_loan(void* opaque, RuntimeLoanBinding* binding,
                                                         EmitOptions options) noexcept {
      auto& state = *static_cast<OutputState*>(opaque);
      count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitAttempts);
      if (binding == nullptr || binding->state == nullptr) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      auto* loan = static_cast<LoanControl*>(binding->state);
      if (loan->owner != &state || !loan->live) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      if (!state.active || loan->activation_generation != state.active_generation) {
        return make_unexpected(Error{ErrorCode::kStaleEndpoint});
      }
      const std::uint64_t emitted_sequence = state.next_sequence;
      trace_output_emit_phase(state.phase_counters,
                              RuntimePhaseTracePoint::kInlinePayloadEmitStarted,
                              state.port,
                              emitted_sequence);
      expected<void, Error> published;
      if (state.type_ops->codec_family == RuntimePayloadCodecFamily::kGeneratedSchema) {
        published = emit_generated_schema_payload(
            state, loan->value.data(), metadata_from_options(options, 0U), emitted_sequence);
      } else if (state.type_ops->codec_family == RuntimePayloadCodecFamily::kCustomPod) {
        if (!state.pod_identity || state.encode_scratch.data() == nullptr) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        const std::span<const std::byte> native{static_cast<const std::byte*>(loan->value.data()),
                                                state.type_ops->payload_size};
        const std::span<std::byte> destination{static_cast<std::byte*>(state.encode_scratch.data()),
                                               state.type_ops->wire_capacity};
        auto encoded = channel_internal::PodFieldwiseCodec::encode_prevalidated(
            *state.pod_identity, native, destination);
        if (!encoded) {
          return make_unexpected(std::move(encoded).error());
        }
        trace_output_emit_phase(state.phase_counters,
                                RuntimePhaseTracePoint::kInlinePayloadEncoded,
                                state.port,
                                emitted_sequence);
        published =
            state.emit_payload(destination, metadata_from_options(options, destination.size()));
      } else {
        trace_output_emit_phase(state.phase_counters,
                                RuntimePhaseTracePoint::kInlinePayloadEncoded,
                                state.port,
                                emitted_sequence);
        published = state.emit_payload(
            {static_cast<const std::byte*>(loan->value.data()), state.type_ops->wire_capacity},
            metadata_from_options(options, state.type_ops->wire_capacity));
      }
      if (published || publication_error_consumes_authority(published.error().code)) {
        loan->retire_live_authority();
      }
      if (published) {
        trace_output_emit_phase(state.phase_counters,
                                RuntimePhaseTracePoint::kInlinePayloadEmitCompleted,
                                state.port,
                                emitted_sequence);
        count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitAccepted);
      } else if (published.error().code == ErrorCode::kBackpressured) {
        count_phase(state.phase_counters, RuntimePhaseCounter::kChannelEmitBackpressured);
      }
      return published;
    }

    inline static const RuntimeOutputBindingOps kBindingOps{
        .emit_sample = &emit_sample,
        .emit_descriptor_tensor = &emit_descriptor_tensor,
        .emit_descriptor_tensors = &emit_descriptor_tensors,
        .allocate = &allocate,
        .emit_loan = &emit_loan,
    };
  };

  struct ActivationOwnerState {
    struct DirectContinuationPublication {
      OutputState* output{};
      SubscriberState* subscriber{};
      std::uint64_t publication_sequence{};
      std::uint64_t producer_epoch{};
      std::uint64_t prior_destination_revision{};
      std::uint64_t destination_revision{};
      std::uint64_t source_activation_generation{};
      std::uint32_t successor_publications{};
    };

    std::vector<InputPortState*> inputs;
    std::vector<SubscriberState*> subscribers;
    std::vector<OutputState*> outputs;
    // Cold, session-local topology decision populated after the destination readiness binding
    // exists. A null pointer is the permanent dormant state for this RunSession; live generation,
    // revision, closure, and capacity predicates remain checked when a continuation is armed.
    OutputState* direct_continuation_output{};
    std::optional<DirectContinuationPublication> direct_continuation;
  };

  [[nodiscard]] expected<void, Error> bind_activation_owners(std::size_t port_count) {
    if (!activation_owners_.empty() || !input_by_port_.empty() || port_count == 0U ||
        std::any_of(
            inputs_.begin(), inputs_.end(), [](const auto& input) { return !input->owner; }) ||
        std::any_of(subscribers_.begin(),
                    subscribers_.end(),
                    [](const auto& subscriber) {
                      return !subscriber->owner || subscriber->input == nullptr;
                    }) ||
        std::any_of(
            outputs_.begin(), outputs_.end(), [](const auto& output) { return !output->owner; })) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    std::uint64_t maximum_owner = 0U;
    for (const auto& input : inputs_) {
      maximum_owner = std::max(maximum_owner, input->owner.value);
    }
    for (const auto& output : outputs_) {
      maximum_owner = std::max(maximum_owner, output->owner.value);
    }
    if (maximum_owner > std::numeric_limits<std::size_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    activation_owners_.resize(static_cast<std::size_t>(maximum_owner));
    input_by_port_.resize(port_count);
    for (const auto& input : inputs_) {
      if (!input->port || input->port.value() > input_by_port_.size()) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      InputPortState*& indexed = input_by_port_[static_cast<std::size_t>(input->port.value() - 1U)];
      if (indexed != nullptr) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      indexed = input.get();
      activation_owners_[static_cast<std::size_t>(input->owner.value - 1U)].inputs.push_back(
          input.get());
    }
    for (const auto& subscriber : subscribers_) {
      activation_owners_[static_cast<std::size_t>(subscriber->owner.value - 1U)]
          .subscribers.push_back(subscriber.get());
    }
    for (const auto& output : outputs_) {
      activation_owners_[static_cast<std::size_t>(output->owner.value - 1U)].outputs.push_back(
          output.get());
    }
#if !defined(NDEBUG)
    owner_mutation_active_ =
        std::make_unique<std::atomic_flag[]>(static_cast<std::size_t>(maximum_owner));
    for (std::size_t index = 0U; index < static_cast<std::size_t>(maximum_owner); ++index) {
      owner_mutation_active_[index].clear(std::memory_order_relaxed);
    }
#endif
    return {};
  }

  [[nodiscard]] ActivationOwnerState* find_activation_owner(VertexId owner) noexcept {
    if (!owner || owner.value > activation_owners_.size()) {
      return nullptr;
    }
    auto& state = activation_owners_[static_cast<std::size_t>(owner.value - 1U)];
    return state.inputs.empty() && state.outputs.empty() ? nullptr : &state;
  }

  [[nodiscard]] const ActivationOwnerState* find_activation_owner(VertexId owner) const noexcept {
    if (!owner || owner.value > activation_owners_.size()) {
      return nullptr;
    }
    const auto& state = activation_owners_[static_cast<std::size_t>(owner.value - 1U)];
    return state.inputs.empty() && state.outputs.empty() ? nullptr : &state;
  }

#if !defined(NDEBUG)
  [[nodiscard]] std::atomic_flag& owner_mutation_flag(VertexId owner) noexcept {
    assert(owner && owner.value <= activation_owners_.size());
    assert(owner_mutation_active_ != nullptr);
    return owner_mutation_active_[static_cast<std::size_t>(owner.value - 1U)];
  }
#endif

  void cache_direct_continuation_candidate(const VertexId destination,
                                           ReadinessBinding& binding) noexcept {
    ActivationOwnerState* const destination_state = find_activation_owner(destination);
    if (destination_state == nullptr || destination_state->inputs.size() != 1U ||
        destination_state->subscribers.size() != 1U ||
        destination_state->inputs.front() == nullptr ||
        destination_state->subscribers.front() == nullptr ||
        destination_state->subscribers.front()->input != destination_state->inputs.front() ||
        destination_state->inputs.front()->readiness_binding != &binding ||
        binding.polls_external_transport_ || binding.input_revisions_.size() != 1U ||
        binding.input_revisions_.front().input != destination_state->inputs.front()) {
      return;
    }

    SubscriberState& subscriber = *destination_state->subscribers.front();
    ActivationOwnerState* const producer_state = find_activation_owner(subscriber.source_owner);
    if (producer_state == nullptr || !subscriber.direct_process_local ||
        subscriber.cross_process_transport || subscriber.externally_driven_import ||
        subscriber.service.empty() ||
        (subscriber.queue_policy != QueuePolicy::kBlocking &&
         subscriber.queue_policy != QueuePolicy::kDropOldest)) {
      return;
    }

    OutputState* output = nullptr;
    for (OutputState* const candidate : producer_state->outputs) {
      if (candidate->direct_subscriber == nullptr) {
        continue;
      }
      if (output != nullptr) {
        return;
      }
      output = candidate;
    }
    if (output == nullptr || output->direct_subscriber != &subscriber ||
        output->owner != subscriber.source_owner || output->port != subscriber.source ||
        output->tensor_payload() || output->tensor_schema_payload()) {
      return;
    }
    producer_state->direct_continuation_output = output;
  }

  [[nodiscard]] std::optional<RuntimeChannelDirectContinuationTarget> direct_continuation_target(
      const VertexId producer, const std::uint32_t execution_lane,
      const bool require_clean_revision, const bool require_empty_binding) const noexcept {
    const ActivationOwnerState* const producer_state = find_activation_owner(producer);
    const OutputState* const output =
        producer_state == nullptr ? nullptr : producer_state->direct_continuation_output;
    if (output == nullptr || output->direct_subscriber == nullptr) {
      return std::nullopt;
    }
    const SubscriberState& subscriber = *output->direct_subscriber;
    const InputPortState* const input = subscriber.input;
    const ReadinessBinding* const binding = input != nullptr ? input->readiness_binding : nullptr;
    if (subscriber.source_owner != producer || input == nullptr || binding == nullptr ||
        binding->closed() || binding->request_.execution_lane != execution_lane ||
        binding->input_revisions_.size() != 1U ||
        binding->input_revisions_.front().input != input ||
        (require_empty_binding && binding->size_ != 0U) || binding->timeout_count_ != 0U ||
        binding->closure_count_ != 0U) {
      return std::nullopt;
    }
    const std::uint64_t current_revision =
        input->readiness_revision.load(std::memory_order_acquire);
    if (require_clean_revision && binding->input_revisions_.front().observed != current_revision) {
      return std::nullopt;
    }
    return RuntimeChannelDirectContinuationTarget{
        .plan_generation = plan_generation_,
        .run_generation = binding->request_.run_generation,
        .producer = producer,
        .destination = subscriber.owner,
        .source_output = output->port,
        .destination_input = subscriber.target,
        .service = subscriber.service,
        .readiness_source = binding->request_.source,
        .reference_clock = binding->request_.reference_clock,
        .source_generation = binding->request_.source_generation,
        .readiness_sequence = binding->next_source_sequence_,
        .destination_revision = current_revision,
        .observed_destination_revision = binding->input_revisions_.front().observed,
        .source_activation_generation = output->active_generation,
        .execution_lane = execution_lane,
        .readiness_capacity = binding->storage_.size(),
    };
  }

  void note_direct_continuation_publication(OutputState& output, SubscriberState& subscriber,
                                            const std::uint64_t publication_sequence,
                                            const std::uint64_t prior_revision,
                                            const std::uint64_t destination_revision) noexcept {
    ActivationOwnerState* const owner = find_activation_owner(output.owner);
    if (owner == nullptr) {
      return;
    }
    if (!owner->direct_continuation) {
      owner->direct_continuation.emplace(ActivationOwnerState::DirectContinuationPublication{
          .output = &output,
          .subscriber = &subscriber,
          .publication_sequence = publication_sequence,
          .producer_epoch = subscriber.producer_epoch,
          .prior_destination_revision = prior_revision,
          .destination_revision = destination_revision,
          .source_activation_generation = output.active_generation,
          .successor_publications = 1U,
      });
      return;
    }
    if (owner->direct_continuation->successor_publications !=
        std::numeric_limits<std::uint32_t>::max()) {
      ++owner->direct_continuation->successor_publications;
    }
  }

  [[nodiscard]] expected<std::optional<RuntimeChannelDirectContinuationReceipt>, Error>
  poll_direct_continuation(const VertexId producer, const std::uint32_t execution_lane,
                           const TimePoint observed_at) noexcept {
    ActivationOwnerState* const owner = find_activation_owner(producer);
    const auto target = direct_continuation_target(producer, execution_lane, false, true);
    if (owner == nullptr || !target || !owner->direct_continuation) {
      if (owner != nullptr) {
        owner->direct_continuation.reset();
      }
      return std::nullopt;
    }
    const ActivationOwnerState::DirectContinuationPublication publication =
        *owner->direct_continuation;
    owner->direct_continuation.reset();
    OutputState* const output = publication.output;
    SubscriberState* const subscriber = publication.subscriber;
    InputPortState* const input = subscriber != nullptr ? subscriber->input : nullptr;
    ReadinessBinding* const binding = input != nullptr ? input->readiness_binding : nullptr;
    std::uint64_t total_publications = 0U;
    for (const OutputState* const candidate : owner->outputs) {
      total_publications += candidate->emitted_this_activation;
    }
    if (output == nullptr || subscriber == nullptr || input == nullptr || binding == nullptr ||
        output->direct_subscriber != subscriber || output->owner != producer ||
        output->active_generation != publication.source_activation_generation ||
        publication.successor_publications != 1U || total_publications != 1U ||
        subscriber->producer_epoch != publication.producer_epoch ||
        subscriber->last_producer_sequence != publication.publication_sequence ||
        publication.prior_destination_revision == std::numeric_limits<std::uint64_t>::max() ||
        publication.destination_revision != publication.prior_destination_revision + 1U ||
        input->readiness_revision.load(std::memory_order_acquire) !=
            publication.destination_revision ||
        binding->input_revisions_.front().observed != publication.prior_destination_revision ||
        !observed_at.valid() || observed_at.clock_id != target->reference_clock) {
      return std::nullopt;
    }
    const std::uint64_t prior_source_sequence = binding->next_source_sequence_;
    auto polled = binding->poll(observed_at);
    if (!polled) {
      return make_unexpected(std::move(polled).error());
    }
    if (prior_source_sequence == std::numeric_limits<std::uint64_t>::max() ||
        binding->next_source_sequence_ != prior_source_sequence + 1U || binding->size_ != 1U ||
        binding->input_revisions_.front().observed != publication.destination_revision) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    return RuntimeChannelDirectContinuationReceipt{
        .plan_generation = target->plan_generation,
        .run_generation = target->run_generation,
        .producer = target->producer,
        .destination = target->destination,
        .source_output = target->source_output,
        .destination_input = target->destination_input,
        .service = target->service,
        .readiness_source = target->readiness_source,
        .reference_clock = target->reference_clock,
        .source_generation = target->source_generation,
        .readiness_sequence = binding->next_source_sequence_,
        .producer_epoch = publication.producer_epoch,
        .publication_sequence = publication.publication_sequence,
        .prior_destination_revision = publication.prior_destination_revision,
        .destination_revision = publication.destination_revision,
        .source_activation_generation = publication.source_activation_generation,
        .execution_lane = target->execution_lane,
        .successor_publications = publication.successor_publications,
        .publication_status = ErrorCode::kSuccess,
    };
  }

  [[nodiscard]] bool validate_direct_continuation_receipt(
      const RuntimeChannelDirectContinuationReceipt& receipt) const noexcept {
    const auto target =
        direct_continuation_target(receipt.producer, receipt.execution_lane, false, false);
    if (!target || !runtime_channel_direct_continuation_matches(receipt, *target)) {
      return false;
    }
    const ActivationOwnerState* const owner = find_activation_owner(receipt.producer);
    const OutputState* const output =
        owner == nullptr ? nullptr : owner->direct_continuation_output;
    if (output == nullptr || output->port != receipt.source_output ||
        output->direct_subscriber == nullptr) {
      return false;
    }
    const SubscriberState& subscriber = *output->direct_subscriber;
    const InputPortState* const input = subscriber.input;
    const ReadinessBinding* const binding = input != nullptr ? input->readiness_binding : nullptr;
    return binding != nullptr && subscriber.producer_epoch == receipt.producer_epoch &&
           subscriber.last_producer_sequence == receipt.publication_sequence &&
           input->readiness_revision.load(std::memory_order_acquire) ==
               receipt.destination_revision &&
           binding->request_.source_generation == receipt.source_generation &&
           binding->next_source_sequence_ == receipt.readiness_sequence && binding->size_ == 1U;
  }

  [[nodiscard]] static expected<void, Error> rollback_tensor_access(InputPortState& input,
                                                                    std::size_t access_checkpoint,
                                                                    std::size_t lease_checkpoint,
                                                                    Error error) noexcept {
    input.tensor_access_slots.resize(access_checkpoint);
    input.tensor_read_leases.resize(lease_checkpoint);
    return make_unexpected(std::move(error));
  }

  [[nodiscard]] static Tensor* tensor_for_access(InputPortState& input, SampleSlot& slot,
                                                 const bool direct_tensor,
                                                 const std::uint32_t index) noexcept {
    if (direct_tensor) {
      return index == 0U ? &static_cast<Sample<Tensor>*>(slot.sample.data())->data : nullptr;
    }
    const void* payload = input.type_ops->sample_payload(slot.sample.data());
    return payload == nullptr || input.type_ops->descriptor_tensor_at == nullptr
               ? nullptr
               : const_cast<Tensor*>(input.type_ops->descriptor_tensor_at(payload, index));
  }

  [[nodiscard]] static bool tensor_device_requires_consumer_stream(
      const DLDeviceType device_type) noexcept {
    return device_type == kDLCUDA || device_type == kDLCUDAHost || device_type == kDLCUDAManaged;
  }

  [[nodiscard]] static expected<void, Error> prepare_tensor_stream_access(
      Tensor& tensor, cudaStream_t consumer_stream) noexcept {
    if (!tensor_device_requires_consumer_stream(tensor.device().device_type)) {
      return {};
    }
    if (tensor.device().device_type == kDLCUDAHost && !tensor.has_readiness_stream_contract()) {
      // A host-written pinned Tensor has no asynchronous producer work or CUDA device binding.
      return {};
    }
    if (consumer_stream == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    auto synchronized = tensor.synchronize_for_consumer_stream(consumer_stream);
    if (!synchronized) {
      return make_unexpected(std::move(synchronized).error());
    }
    if (!tensor.set_deallocation_stream(consumer_stream)) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    return {};
  }

  [[nodiscard]] expected<void, Error> initialize_pump_indices() {
    if (!wait_subscriber_by_handle_.empty() || !pump_subscribers_.empty()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    wait_subscriber_by_handle_.reserve(wait_attachment_count_);
    pump_subscribers_.reserve(subscribers_.size());
    for (const auto& subscriber : subscribers_) {
      if (subscriber->wait_handle != 0U) {
        const auto [iterator, inserted] =
            wait_subscriber_by_handle_.emplace(subscriber->wait_handle, subscriber.get());
        static_cast<void>(iterator);
        if (!inserted) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
      }
      if (subscriber->terminal_wait_handle != 0U) {
        const auto [iterator, inserted] =
            wait_subscriber_by_handle_.emplace(subscriber->terminal_wait_handle, subscriber.get());
        static_cast<void>(iterator);
        if (!inserted) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
      }
      if (!subscriber->direct_process_local && !subscriber->clock_health_endpoint &&
          !subscriber->externally_driven_import) {
        pump_subscribers_.push_back(subscriber.get());
      }
    }
    if (wait_subscriber_by_handle_.size() != wait_attachment_count_) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    return {};
  }

  [[nodiscard]] static expected<void, Error> prepare_tensor_input_access(
      InputPortState& input, cudaStream_t consumer_stream) noexcept {
    const bool direct_tensor = input.type_ops == runtime_port_type_ops<Tensor>();
    const bool tensor_schema =
        input.type_ops->codec_family == RuntimePayloadCodecFamily::kDescriptorTensorSchema;
    if (!direct_tensor && !tensor_schema) {
      return {};
    }
    const std::size_t access_checkpoint = input.tensor_access_slots.size();
    const std::size_t lease_checkpoint = input.tensor_read_leases.size();
    for (SampleSlot* const slot : input.active_slots) {
      if (std::find(input.tensor_access_slots.begin(), input.tensor_access_slots.end(), slot) !=
          input.tensor_access_slots.end()) {
        continue;
      }
      if (slot == nullptr || !slot->occupied) {
        return rollback_tensor_access(
            input, access_checkpoint, lease_checkpoint, Error{ErrorCode::kProtocolError});
      }
      const std::uint32_t tensor_count =
          direct_tensor ? 1U : input.type_ops->descriptor_tensor_static_count;
      if (tensor_count == 0U || tensor_count > channel_internal::kMaximumDescriptorTensorFields) {
        return rollback_tensor_access(
            input, access_checkpoint, lease_checkpoint, Error{ErrorCode::kProtocolError});
      }
      for (std::uint32_t index = 0U; index < tensor_count; ++index) {
        Tensor* const tensor = tensor_for_access(input, *slot, direct_tensor, index);
        if (tensor == nullptr) {
          return rollback_tensor_access(
              input, access_checkpoint, lease_checkpoint, Error{ErrorCode::kProtocolError});
        }
        auto lease = TensorAccess::acquire_read_lease(*tensor);
        if (!lease) {
          return rollback_tensor_access(
              input, access_checkpoint, lease_checkpoint, std::move(lease).error());
        }
        auto stream_access = prepare_tensor_stream_access(*tensor, consumer_stream);
        if (!stream_access) {
          return rollback_tensor_access(
              input, access_checkpoint, lease_checkpoint, std::move(stream_access).error());
        }
        input.tensor_read_leases.push_back(std::move(*lease));
      }
      input.tensor_access_slots.push_back(slot);
    }
    return {};
  }

  [[nodiscard]] bool replace_wait_handles(
      SubscriberState& subscriber, const std::uint64_t replacement,
      const std::optional<std::uint64_t> terminal_replacement) noexcept {
    if (subscriber.wait_handle == 0U || replacement == 0U ||
        (subscriber.terminal_wait_handle != 0U) != terminal_replacement.has_value()) {
      return false;
    }
    auto node = wait_subscriber_by_handle_.extract(subscriber.wait_handle);
    if (node.empty() || node.mapped() != &subscriber) {
      return false;
    }
    node.key() = replacement;
    const auto inserted = wait_subscriber_by_handle_.insert(std::move(node));
    if (!inserted.inserted) {
      return false;
    }
    if (subscriber.terminal_wait_handle == 0U) {
      return true;
    }
    auto terminal_node = wait_subscriber_by_handle_.extract(subscriber.terminal_wait_handle);
    if (terminal_node.empty() || terminal_node.mapped() != &subscriber) {
      return false;
    }
    terminal_node.key() = terminal_replacement.value_or(0U);
    const auto terminal_inserted = wait_subscriber_by_handle_.insert(std::move(terminal_node));
    return terminal_inserted.inserted;
  }

  [[nodiscard]] static InputPortState* find_activation_input(ActivationOwnerState& owner,
                                                             PortId port) noexcept {
    const auto found = std::find_if(owner.inputs.begin(),
                                    owner.inputs.end(),
                                    [&](const auto* input) { return input->port == port; });
    return found == owner.inputs.end() ? nullptr : *found;
  }

  class ReadinessBinding final : public PortBinding {
   public:
    ReadinessBinding(Impl* owner, VertexId vertex, RuntimePortBindingRequest request,
                     std::size_t candidate_capacity, std::span<InputPortState* const> inputs,
                     bool polls_external_transport)
        : owner_(owner),
          vertex_(vertex),
          request_(request),
          storage_(request.capacity),
          timeout_sequences_(request.capacity),
          closure_sequences_(request.capacity),
          polls_external_transport_(polls_external_transport) {
      candidates_.reserve(candidate_capacity);
      input_revisions_.reserve(inputs.size());
      for (InputPortState* const input : inputs) {
        input_revisions_.push_back(InputRevision{.input = input});
      }
    }

    ~ReadinessBinding() override {
      if (owner_ != nullptr) {
        owner_->unregister_readiness_binding(*this);
      }
    }

    [[nodiscard]] expected<void, Error> poll(TimePoint observed_at) noexcept override {
      if (owner_ == nullptr) {
        return make_unexpected(Error{ErrorCode::kClosed});
      }
      if (!observed_at.valid() || observed_at.clock_id != request_.reference_clock) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      for (InputRevision& revision : input_revisions_) {
        revision.pending = revision.input->readiness_revision.load(std::memory_order_acquire);
      }
      auto polled = owner_->poll_inputs(*this, observed_at);
      if (!polled) {
        return make_unexpected(std::move(polled).error());
      }
      if (*polled) {
        for (InputRevision& revision : input_revisions_) {
          revision.observed = revision.pending;
        }
      }
      return {};
    }

    [[nodiscard]] SourceId source() const noexcept override { return request_.source; }

    [[nodiscard]] std::optional<PortBindingPollTarget> poll_target() const noexcept override {
      if (owner_ == nullptr) {
        return std::nullopt;
      }
      const bool local_input_changed =
          std::any_of(input_revisions_.begin(), input_revisions_.end(), [](const auto& revision) {
            return revision.input->readiness_revision.load(std::memory_order_acquire) !=
                   revision.observed;
          });
      if (!polls_external_transport_ && !local_input_changed) {
        return std::nullopt;
      }
      // External imports merge transport and clock-health heads inside poll(), before a typed
      // cache revision exists, so they retain their compiler-selected polling lane.
      return PortBindingPollTarget{
          .execution_lane = request_.execution_lane,
          .reference_clock = request_.reference_clock,
          .busy_poll = false,
      };
    }

    [[nodiscard]] expected<void, Error> enqueue_window_timeout(
        PortId port, TimePoint terminal_time) noexcept override {
      if (!port || !terminal_time.valid() || terminal_time.clock_id != request_.reference_clock) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      if (closed()) {
        return make_unexpected(Error{ErrorCode::kClosed});
      }
      if (storage_.empty() || size_ == storage_.size() ||
          timeout_count_ == timeout_sequences_.size()) {
        return make_unexpected(Error{ErrorCode::kBackpressured});
      }
      if (next_source_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      const auto timeout_slot = std::find(timeout_sequences_.begin(), timeout_sequences_.end(), 0U);
      if (timeout_slot == timeout_sequences_.end()) {
        return make_unexpected(Error{ErrorCode::kBackpressured});
      }
      ++next_source_sequence_;
      storage_[tail_] = RuntimeEvent{
          .kind = RuntimeEventKind::kWindowTimeout,
          .source = request_.source,
          .port = port,
          .source_decision_time = terminal_time,
          .reference_time = terminal_time,
          .source_generation = request_.source_generation,
          .source_sequence = next_source_sequence_,
      };
      *timeout_slot = next_source_sequence_;
      ++timeout_count_;
      tail_ = (tail_ + 1U) % storage_.size();
      ++size_;
      return {};
    }

    [[nodiscard]] expected<void, Error> retire(std::uint64_t sequence) noexcept override {
      if (owner_ == nullptr) {
        return make_unexpected(Error{ErrorCode::kClosed});
      }
      if (sequence == 0U) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      const auto timeout =
          std::find(timeout_sequences_.begin(), timeout_sequences_.end(), sequence);
      if (timeout != timeout_sequences_.end()) {
        *timeout = 0U;
        --timeout_count_;
        return {};
      }
      const auto closure =
          std::find(closure_sequences_.begin(), closure_sequences_.end(), sequence);
      if (closure != closure_sequences_.end()) {
        *closure = 0U;
        --closure_count_;
        return {};
      }
      return owner_->retire_witness(vertex_, sequence);
    }

    void cancel() noexcept override {
      closed_ = true;
      size_ = 0U;
      head_ = 0U;
      tail_ = 0U;
      std::fill(timeout_sequences_.begin(), timeout_sequences_.end(), 0U);
      std::fill(closure_sequences_.begin(), closure_sequences_.end(), 0U);
      timeout_count_ = 0U;
      closure_count_ = 0U;
    }

    [[nodiscard]] expected<std::uint64_t, Error> advance_source_generation() noexcept override {
      if (owner_ == nullptr) {
        return make_unexpected(Error{ErrorCode::kClosed});
      }
      if (size_ != 0U || timeout_count_ != 0U || closure_count_ != 0U ||
          request_.source_generation == std::numeric_limits<std::uint64_t>::max()) {
        return make_unexpected(Error{size_ != 0U || timeout_count_ != 0U || closure_count_ != 0U
                                         ? ErrorCode::kBackpressured
                                         : ErrorCode::kResourceExhausted});
      }
      ++request_.source_generation;
      next_source_sequence_ = request_.initial_source_sequence;
      closed_ = false;
      return request_.source_generation;
    }

    [[nodiscard]] expected<std::size_t, Error> drain(
        std::span<RuntimeEvent> destination) noexcept override {
      const std::size_t count = std::min(destination.size(), size_);
      for (std::size_t index = 0U; index < count; ++index) {
        destination[index] = storage_[head_];
        head_ = (head_ + 1U) % storage_.size();
      }
      size_ -= count;
      return count;
    }

    [[nodiscard]] std::size_t capacity() const noexcept override { return storage_.size(); }
    [[nodiscard]] bool closed() const noexcept override {
      return closed_ || owner_ == nullptr || owner_->closed_.load(std::memory_order_acquire);
    }

   private:
    struct InputRevision {
      InputPortState* input{};
      std::uint64_t observed{};
      std::uint64_t pending{};
    };

    struct SlotCandidate {
      SubscriberState* subscriber{};
      SampleSlot* slot{};
      std::uint64_t arrival_order{};
    };

    [[nodiscard]] static bool later_candidate(const SlotCandidate& lhs,
                                              const SlotCandidate& rhs) noexcept {
      return lhs.arrival_order > rhs.arrival_order;
    }

    [[nodiscard]] bool owns_input(const InputPortState* input) const noexcept {
      return std::any_of(
          input_revisions_.begin(), input_revisions_.end(), [input](const InputRevision& revision) {
            return revision.input == input;
          });
    }

    void detach_from_owner() noexcept {
      owner_ = nullptr;
      input_revisions_.clear();
    }

    [[nodiscard]] expected<void, Error> push_end_of_stream(PortId port,
                                                           TimePoint observed_at) noexcept {
      if (!port || !observed_at.valid() || observed_at.clock_id != request_.reference_clock) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      if (closed()) {
        return make_unexpected(Error{ErrorCode::kClosed});
      }
      if (storage_.empty() || size_ == storage_.size() ||
          closure_count_ == closure_sequences_.size()) {
        return make_unexpected(Error{ErrorCode::kBackpressured});
      }
      if (next_source_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      const auto closure_slot = std::find(closure_sequences_.begin(), closure_sequences_.end(), 0U);
      if (closure_slot == closure_sequences_.end()) {
        return make_unexpected(Error{ErrorCode::kBackpressured});
      }
      ++next_source_sequence_;
      storage_[tail_] = RuntimeEvent{
          .kind = RuntimeEventKind::kInputEndOfStream,
          .source = request_.source,
          .port = port,
          .source_decision_time = observed_at,
          .reference_time = observed_at,
          .source_generation = request_.source_generation,
          .source_sequence = next_source_sequence_,
      };
      *closure_slot = next_source_sequence_;
      ++closure_count_;
      tail_ = (tail_ + 1U) % storage_.size();
      ++size_;
      return {};
    }

    [[nodiscard]] expected<void, Error> push(SampleSlot& slot, std::uint64_t& sequence,
                                             TimePoint observed_at) noexcept {
      if (storage_.empty() || size_ == storage_.size()) {
        return make_unexpected(Error{ErrorCode::kBackpressured});
      }
      if (next_source_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      auto coordinates = owner_->event_coordinates(slot, request_, observed_at);
      if (!coordinates) {
        return make_unexpected(std::move(coordinates).error());
      }
      // Temporal selectors operate only in the compiler-frozen reference clock. Preserve the
      // caller's original capture coordinate in the typed Sample metadata, but normalize a
      // present payload-independent capture coordinate exactly once while the readiness witness
      // still owns both domains. A capture-less process-local sample uses arrival time only for
      // readiness ordering; its TemporalSampleRecord must remain capture-less so selection applies
      // the authored missing-record policy.
      if (slot.temporal.capture_time.valid()) {
        slot.temporal.capture_time = coordinates->second;
      }
      ++next_source_sequence_;
      sequence = next_source_sequence_;
      storage_[tail_] = RuntimeEvent{
          .kind = RuntimeEventKind::kInputArrival,
          .source = request_.source,
          .port = slot.subscriber->target,
          .source_decision_time = coordinates->first,
          .reference_time = coordinates->second,
          .source_generation = request_.source_generation,
          .source_sequence = sequence,
          .clock_health_disposition = slot.clock_health_disposition,
          .clock_health_generation = slot.clock_health_generation,
      };
      tail_ = (tail_ + 1U) % storage_.size();
      ++size_;
      return {};
    }

    Impl* owner_{};
    VertexId vertex_{};
    RuntimePortBindingRequest request_{};
    std::vector<RuntimeEvent> storage_;
    std::vector<std::uint64_t> timeout_sequences_;
    std::vector<SlotCandidate> candidates_;
    std::vector<std::uint64_t> closure_sequences_;
    std::vector<InputRevision> input_revisions_;
    std::size_t head_{};
    std::size_t tail_{};
    std::size_t size_{};
    std::size_t timeout_count_{};
    std::size_t closure_count_{};
    std::uint64_t next_source_sequence_{request_.initial_source_sequence};
    bool polls_external_transport_{};
    bool closed_{};

    friend class Impl;
    friend class RuntimeChannelSessionTestAccess;
  };

  Impl(ChannelRuntime runtime, PlanGenerationId plan_generation,
       std::span<const ClockConversionRecord> clock_conversions,
       std::shared_ptr<RuntimeSupportThreadControl> support_thread_control,
       RuntimePhaseCounters* phase_counters, RuntimeChannelReadinessWake readiness_wake)
      : runtime_(std::move(runtime)),
        plan_generation_(plan_generation),
        clock_conversions_(clock_conversions.begin(), clock_conversions.end()),
        support_thread_control_(std::move(support_thread_control)),
        phase_counters_(phase_counters),
        readiness_wake_(readiness_wake) {}

  ~Impl() {
    cancel();
    static_cast<void>(join());
    observe_teardown(RuntimeChannelTeardownPhase::kPumpJoined);
    detach_readiness_bindings();
    for (const auto& input : inputs_) {
      input->type_ops->unbind_input(input->declaration, &input->binding);
    }
    for (const auto& output : outputs_) {
      output->type_ops->unbind_output(output->declaration, &output->binding);
    }
    wait_set_.reset();
    wait_subscriber_by_handle_.clear();
    pump_subscribers_.clear();
    subscribers_.clear();
    activation_owners_.clear();
    outputs_.clear();
    inputs_.clear();
    observe_teardown(RuntimeChannelTeardownPhase::kEndpointChildrenDestroyed);
    runtime_.reset();
    observe_teardown(RuntimeChannelTeardownPhase::kRuntimeDestroyed);
  }

  void unregister_readiness_binding(ReadinessBinding& binding) noexcept {
    const auto found = std::find(readiness_bindings_.begin(), readiness_bindings_.end(), &binding);
    if (found != readiness_bindings_.end()) {
      readiness_bindings_.erase(found);
    }
    for (const auto& revision : binding.input_revisions_) {
      if (revision.input != nullptr && revision.input->readiness_binding == &binding) {
        revision.input->readiness_binding = nullptr;
      }
    }
    for (ActivationOwnerState& owner : activation_owners_) {
      OutputState* const output = owner.direct_continuation_output;
      if (output == nullptr || output->direct_subscriber == nullptr ||
          !binding.owns_input(output->direct_subscriber->input)) {
        continue;
      }
      output->direct_continuation_capture = false;
      owner.direct_continuation.reset();
      owner.direct_continuation_output = nullptr;
    }
    binding.detach_from_owner();
  }

  void detach_readiness_bindings() noexcept {
    while (!readiness_bindings_.empty()) {
      unregister_readiness_binding(*readiness_bindings_.back());
    }
  }

  void observe_teardown(const RuntimeChannelTeardownPhase phase) noexcept {
    if (teardown_observer_ != nullptr) {
      teardown_observer_(teardown_observer_state_, phase);
    }
  }

  [[nodiscard]] expected<void, Error> set_teardown_observer_for_test(
      void* state, RuntimeChannelTeardownObserver observer) noexcept {
    if (armed_.load(std::memory_order_acquire) || state == nullptr || observer == nullptr ||
        teardown_observer_ != nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    teardown_observer_state_ = state;
    teardown_observer_ = observer;
    return {};
  }

  // Each branch proves one compiler-frozen clock-domain relation before exposing coordinates.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<std::pair<TimePoint, TimePoint>, Error> event_coordinates(
      const SampleSlot& slot, const RuntimePortBindingRequest& request,
      TimePoint observed_at) const noexcept {
    const InputPortState& input = *slot.subscriber->input;
    const TimePoint capture = slot.temporal.capture_time;
    if (!capture.valid()) {
      // Only process-local samples can reach retained storage without capture time. Catalog
      // imports that promise capture are rejected at the transport boundary before a slot or
      // arrival identity is committed. Preserve local absence for selector policy while using
      // observed arrival solely as deterministic readiness-order evidence.
      return std::pair{observed_at, observed_at};
    }

    if (input.reference_clock.valid()) {
      if (capture.clock_id != input.capture_clock) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      auto reference = convert_identical_or_direct(
          capture,
          input.reference_clock,
          input.capture_to_reference ? &*input.capture_to_reference : nullptr);
      if (!reference) {
        return make_unexpected(std::move(reference).error());
      }
      // Input selection always normalizes capture into its compiler-selected reference clock.
      // The readiness source, however, uses that reference coordinate as both decision and
      // reference time unless a deadline explicitly retained a distinct source clock. Exposing
      // the original capture coordinate in the ordinary no-deadline case would violate the
      // RuntimeSourceBindingRecord identity consumed by the coordinator.
      if (request.source_clock == request.reference_clock) {
        if (request.source_to_reference_conversion_id != 0U ||
            request.reference_clock != reference->clock_id) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        return std::pair{*reference, *reference};
      }
      if (request.source_to_reference_conversion_id == 0U ||
          request.source_clock != capture.clock_id ||
          request.reference_clock != reference->clock_id) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      const auto source_conversion =
          std::find_if(clock_conversions_.begin(), clock_conversions_.end(), [&](const auto& edge) {
            return edge.id == request.source_to_reference_conversion_id &&
                   edge.plan_generation == plan_generation_ && edge.from == request.source_clock &&
                   edge.to == request.reference_clock;
          });
      if (source_conversion == clock_conversions_.end()) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      const DirectAffineClockEdge source_edge{
          .source = source_conversion->from,
          .target = source_conversion->to,
          .conversion = AffineClockConversion{.scale = source_conversion->scale,
                                              .offset_ns = source_conversion->offset_ns},
      };
      auto source_reference =
          convert_identical_or_direct(capture, request.reference_clock, &source_edge);
      if (!source_reference || *source_reference != *reference) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      return std::pair{capture, *reference};
    }

    if (capture.clock_id == request.reference_clock) {
      return std::pair{capture, capture};
    }
    const auto conversion =
        std::find_if(clock_conversions_.begin(), clock_conversions_.end(), [&](const auto& edge) {
          return edge.id == request.source_to_reference_conversion_id &&
                 edge.plan_generation == plan_generation_ && edge.from == capture.clock_id &&
                 edge.to == request.reference_clock;
        });
    if (conversion == clock_conversions_.end()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const DirectAffineClockEdge edge{
        .source = conversion->from,
        .target = conversion->to,
        .conversion =
            AffineClockConversion{.scale = conversion->scale, .offset_ns = conversion->offset_ns},
    };
    auto reference = convert_identical_or_direct(capture, request.reference_clock, &edge);
    if (!reference) {
      return make_unexpected(std::move(reference).error());
    }
    return std::pair{capture, *reference};
  }

  // The visible branch sequence preserves the bounded snapshot, heap ordering, and revalidation
  // protocol across independently locked input caches.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<bool, Error> poll_inputs(ReadinessBinding& binding,
                                                  TimePoint observed_at) noexcept {
    const auto completion = [this](const bool complete) -> expected<bool, Error> {
      auto current = status();
      return current ? expected<bool, Error>{complete}
                     : expected<bool, Error>{make_unexpected(std::move(current).error())};
    };
    const auto* owner = find_activation_owner(binding.vertex_);
    if (owner == nullptr || owner->subscribers.empty()) {
      return make_unexpected(Error{ErrorCode::kNotFound});
    }
    auto monitored = drain_monitored_inputs(*owner);
    if (!monitored) {
      return make_unexpected(std::move(monitored).error());
    }
    binding.candidates_.clear();
    for (SubscriberState* const subscriber : owner->subscribers) {
      const std::unique_lock lock{subscriber->mutex, std::try_to_lock};
      if (!lock.owns_lock()) {
        // Returning without progress preserves arrival order: an earlier sample may be hidden
        // behind this contended cache lock, so selecting from later subscribers would overtake
        // it. The dispatcher may retry after the bounded owner of the lock makes progress.
        return completion(false);
      }
      auto promoted = promote_direct_ingress(*subscriber);
      if (!promoted) {
        return make_unexpected(std::move(promoted).error());
      }
      for (SampleSlot& slot : subscriber->slots) {
        // An occupied pre-witness replacement candidate remains the incumbent until its reserved
        // receive commits. Do not acknowledge the captured input revision while that incumbent is
        // temporarily hidden, because a failed receive must leave it independently pollable.
        if (slot.typed_ingress_reserved && slot.occupied && !slot.direct_ingress_unread &&
            !slot.event_emitted) {
          return completion(false);
        }
        if (!slot.occupied || slot.direct_ingress_unread || slot.event_emitted ||
            slot.typed_ingress_reserved) {
          continue;
        }
        if (binding.candidates_.size() == binding.candidates_.capacity()) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        binding.candidates_.push_back(ReadinessBinding::SlotCandidate{
            .subscriber = subscriber, .slot = &slot, .arrival_order = slot.arrival_order});
      }
      if (subscriber->end_of_stream_received && !subscriber->end_of_stream_event_emitted &&
          subscriber->end_of_stream_arrival_order != 0U &&
          !subscriber->has_pending_direct_ingress()) {
        if (binding.candidates_.size() == binding.candidates_.capacity()) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        binding.candidates_.push_back(ReadinessBinding::SlotCandidate{
            .subscriber = subscriber,
            .slot = nullptr,
            .arrival_order = subscriber->end_of_stream_arrival_order,
        });
      }
    }
    std::make_heap(
        binding.candidates_.begin(), binding.candidates_.end(), ReadinessBinding::later_candidate);
    while (!binding.candidates_.empty()) {
      std::pop_heap(binding.candidates_.begin(),
                    binding.candidates_.end(),
                    ReadinessBinding::later_candidate);
      const ReadinessBinding::SlotCandidate selected = binding.candidates_.back();
      binding.candidates_.pop_back();
      if (poll_selection_observer_ != nullptr) {
        const auto observer = std::exchange(poll_selection_observer_, nullptr);
        void* const state = std::exchange(poll_selection_observer_state_, nullptr);
        observer(state);
      }
      SubscriberState& selected_subscriber = *selected.subscriber;
      const std::unique_lock lock{selected_subscriber.mutex, std::try_to_lock};
      if (!lock.owns_lock()) {
        return completion(false);
      }
      if (selected.slot == nullptr) {
        if (!selected_subscriber.end_of_stream_received ||
            selected_subscriber.end_of_stream_event_emitted ||
            selected_subscriber.end_of_stream_arrival_order != selected.arrival_order) {
          continue;
        }
        auto pushed = binding.push_end_of_stream(selected_subscriber.target, observed_at);
        if (!pushed) {
          return make_unexpected(std::move(pushed).error());
        }
        selected_subscriber.end_of_stream_event_emitted = true;
      } else {
        SampleSlot& selected_slot = *selected.slot;
        // On the HoloMQ path, a drop-oldest producer may replace a saved slot after the owner-local
        // snapshot releases its lock. Reinsert that later identity into the heap so it cannot
        // overtake an older candidate. Direct ingress slots enter this heap only after promotion.
        if (selected_slot.typed_ingress_reserved) {
          return completion(false);
        }
        if (!selected_slot.occupied || selected_slot.event_emitted) {
          continue;
        }
        if (selected_slot.arrival_order != selected.arrival_order) {
          binding.candidates_.push_back(ReadinessBinding::SlotCandidate{
              .subscriber = &selected_subscriber,
              .slot = &selected_slot,
              .arrival_order = selected_slot.arrival_order,
          });
          std::push_heap(binding.candidates_.begin(),
                         binding.candidates_.end(),
                         ReadinessBinding::later_candidate);
          continue;
        }
        auto pushed = binding.push(selected_slot, selected_slot.readiness_sequence, observed_at);
        if (!pushed) {
          return make_unexpected(std::move(pushed).error());
        }
        // Successful readiness publication transfers one payload-lifetime claim to the temporal
        // selector. The dispatcher later returns that claim through retire_samples() only after the
        // record disappears from every cache, reuse slot, and live or closed window.
        selected_slot.temporal_retained = true;
        selected_slot.event_emitted = true;
      }
    }
    return completion(true);
  }

  [[nodiscard]] InputPortState* find_input(PortId port) noexcept {
    if (!input_by_port_.empty()) {
      return !port || port.value() > input_by_port_.size()
                 ? nullptr
                 : input_by_port_[static_cast<std::size_t>(port.value() - 1U)];
    }
    const auto found = std::find_if(
        inputs_.begin(), inputs_.end(), [&](const auto& input) { return input->port == port; });
    return found == inputs_.end() ? nullptr : found->get();
  }

  [[nodiscard]] const InputPortState* find_input(PortId port) const noexcept {
    if (!input_by_port_.empty()) {
      return !port || port.value() > input_by_port_.size()
                 ? nullptr
                 : input_by_port_[static_cast<std::size_t>(port.value() - 1U)];
    }
    const auto found = std::find_if(
        inputs_.begin(), inputs_.end(), [&](const auto& input) { return input->port == port; });
    return found == inputs_.end() ? nullptr : found->get();
  }

  [[nodiscard]] expected<void, Error> replace_output_publishers_for_test(
      PortId port,
      std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers,
      const std::span<const std::uint32_t> expected_recipient_counts,
      const bool refresh_before_emit) noexcept {
    if (armed_.load(std::memory_order_acquire) || !port || publishers.empty() ||
        (!expected_recipient_counts.empty() &&
         expected_recipient_counts.size() != publishers.size()) ||
        std::any_of(publishers.begin(),
                    publishers.end(),
                    [](const auto& publisher) { return publisher == nullptr; }) ||
        std::any_of(expected_recipient_counts.begin(),
                    expected_recipient_counts.end(),
                    [](const auto count) { return count == 0U; })) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const auto output = std::find_if(outputs_.begin(), outputs_.end(), [&](const auto& candidate) {
      return candidate->port == port;
    });
    if (output == outputs_.end() || (*output)->active) {
      return make_unexpected(
          Error{output == outputs_.end() ? ErrorCode::kNotFound : ErrorCode::kBackpressured});
    }
    (*output)->endpoints.clear();
    (*output)->endpoints.reserve(publishers.size());
    for (std::size_t index = 0U; index < publishers.size(); ++index) {
      (*output)->endpoints.push_back(OutputState::EndpointState{
          .publisher = std::move(publishers[index]),
          .expected_recipient_count =
              expected_recipient_counts.empty() ? 1U : expected_recipient_counts[index],
          .refresh_before_emit = refresh_before_emit,
          .process_local_transport = !refresh_before_emit,
      });
    }
    return {};
  }

  [[nodiscard]] expected<void, Error> throw_tensor_fanout_after_accepts_for_test(
      const PortId port, const std::size_t accepted_count) noexcept {
    if (armed_.load(std::memory_order_acquire) || !port) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const auto output = std::find_if(outputs_.begin(), outputs_.end(), [&](const auto& candidate) {
      return candidate->port == port;
    });
    if (output == outputs_.end()) {
      return make_unexpected(Error{ErrorCode::kNotFound});
    }
    auto& state = **output;
    if (state.active || !state.tensor_payload() || accepted_count > state.tensor_endpoints.size() ||
        state.throw_tensor_fanout_after_accepts_for_test) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    state.throw_tensor_fanout_after_accepts_for_test = accepted_count;
    return {};
#else
    (void)accepted_count;
    return make_unexpected(Error{ErrorCode::kNotSupported});
#endif
  }

  // Witness retirement preserves ordered cache, attribution, and readiness transitions.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<void, Error> retire_witness(VertexId owner,
                                                     std::uint64_t sequence) noexcept {
    if (!owner || sequence == 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    ActivationOwnerState* const activation_owner = find_activation_owner(owner);
    if (activation_owner == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotFound});
    }
    const bool signal_progress = owner_progress_wake_enabled();
    for (SubscriberState* const subscriber : activation_owner->subscribers) {
      bool matched = false;
      bool direct_capacity_available = false;
      {
        const std::scoped_lock lock{subscriber->mutex};
        for (SampleSlot& slot : subscriber->slots) {
          if (slot.occupied && slot.readiness_sequence == sequence) {
            if (!slot.event_emitted || slot.witness_retired) {
              return make_unexpected(Error{ErrorCode::kStaleEndpoint});
            }
            slot.witness_retired = true;
            reclaim_sample_slot_if_unclaimed(slot);
            auto promoted = promote_direct_ingress(*subscriber);
            if (!promoted) {
              return make_unexpected(std::move(promoted).error());
            }
            matched = true;
            direct_capacity_available = signal_progress && subscriber->direct_process_local &&
                                        (subscriber->direct_ingress_capacity != 0U
                                             ? writable_direct_ingress_slot(*subscriber) != nullptr
                                             : writable_slot(*subscriber) != nullptr);
            break;
          }
        }
      }
      if (!matched) {
        continue;
      }
      count_phase(phase_counters_, RuntimePhaseCounter::kChannelWitnessRetired);
      if (direct_capacity_available) {
        signal_owner_progress(subscriber->source_owner);
      }
      return {};
    }
    return make_unexpected(Error{ErrorCode::kNotFound});
  }

  [[nodiscard]] SampleSlot* writable_slot(SubscriberState& subscriber) noexcept {
    const std::size_t typed_occupied = static_cast<std::size_t>(
        std::count_if(subscriber.slots.begin(), subscriber.slots.end(), [](const auto& slot) {
          return (slot.occupied && !slot.direct_ingress_unread) || slot.typed_ingress_reserved;
        }));
    if (typed_occupied < subscriber.typed_slot_capacity) {
      const auto free =
          std::find_if(subscriber.slots.begin(), subscriber.slots.end(), [](const auto& slot) {
            return !slot.occupied && !slot.typed_ingress_reserved;
          });
      return free == subscriber.slots.end() ? nullptr : &*free;
    }
    if (typed_occupied > subscriber.typed_slot_capacity) {
      return nullptr;
    }
    if (subscriber.queue_policy == QueuePolicy::kBlocking) {
      return nullptr;
    }
    const auto eligible = [&](const SampleSlot& slot) {
      return slot.occupied && !slot.direct_ingress_unread && !slot.active_selected &&
             !slot.reuse_retained && !slot.temporal_retained && !slot.retirement_selected &&
             !slot.typed_ingress_reserved &&
             ((slot.event_emitted && slot.witness_retired) ||
              (!slot.event_emitted && subscriber.pre_witness_replacement_allowed));
    };
    const auto oldest = std::min_element(
        subscriber.slots.begin(), subscriber.slots.end(), [&](const auto& lhs, const auto& rhs) {
          const std::uint64_t lhs_order =
              eligible(lhs) ? lhs.arrival_order : std::numeric_limits<std::uint64_t>::max();
          const std::uint64_t rhs_order =
              eligible(rhs) ? rhs.arrival_order : std::numeric_limits<std::uint64_t>::max();
          return lhs_order < rhs_order;
        });
    return oldest != subscriber.slots.end() && eligible(*oldest) ? &*oldest : nullptr;
  }

  [[nodiscard]] SampleSlot* writable_direct_ingress_slot(SubscriberState& subscriber) noexcept {
    if (!subscriber.direct_process_local || subscriber.direct_ingress_capacity == 0U) {
      return nullptr;
    }
    const std::size_t ingress_occupied = static_cast<std::size_t>(
        std::count_if(subscriber.slots.begin(), subscriber.slots.end(), [](const auto& slot) {
          return slot.occupied && slot.direct_ingress_unread;
        }));
    if (ingress_occupied < subscriber.direct_ingress_capacity) {
      const auto free =
          std::find_if(subscriber.slots.begin(), subscriber.slots.end(), [](const auto& slot) {
            return !slot.occupied && !slot.typed_ingress_reserved;
          });
      return free == subscriber.slots.end() ? nullptr : &*free;
    }
    if (ingress_occupied > subscriber.direct_ingress_capacity) {
      return nullptr;
    }
    const auto oldest = std::min_element(
        subscriber.slots.begin(), subscriber.slots.end(), [](const auto& lhs, const auto& rhs) {
          const std::uint64_t lhs_sequence = lhs.occupied && lhs.direct_ingress_unread
                                                 ? lhs.temporal.source_sequence
                                                 : std::numeric_limits<std::uint64_t>::max();
          const std::uint64_t rhs_sequence = rhs.occupied && rhs.direct_ingress_unread
                                                 ? rhs.temporal.source_sequence
                                                 : std::numeric_limits<std::uint64_t>::max();
          return lhs_sequence < rhs_sequence;
        });
    return oldest != subscriber.slots.end() && oldest->occupied && oldest->direct_ingress_unread &&
                   !oldest->typed_ingress_reserved
               ? &*oldest
               : nullptr;
  }

  [[nodiscard]] std::optional<TypedIngressReservation> reserve_typed_ingress_destination_locked(
      SubscriberState& subscriber) noexcept {
    SampleSlot* const destination = writable_slot(subscriber);
    if (destination == nullptr) {
      return std::nullopt;
    }
    destination->typed_ingress_reserved = true;
    return TypedIngressReservation{subscriber, *destination};
  }

  [[nodiscard]] std::optional<TypedIngressReservation> reserve_typed_ingress_destination(
      SubscriberState& subscriber) noexcept {
    const std::scoped_lock cache_lock{subscriber.mutex};
    return reserve_typed_ingress_destination_locked(subscriber);
  }

  static void observe_typed_ingress_for_test(SubscriberState& subscriber) noexcept {
    if (subscriber.typed_ingress_observer != nullptr) {
      subscriber.typed_ingress_observer(subscriber.typed_ingress_observer_state);
    }
  }

  // Promotion selects and commits direct-ingress slots as one ordered state transition.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<std::uint32_t, Error> promote_direct_ingress(
      SubscriberState& subscriber,
      std::optional<std::uint64_t> first_arrival_order = std::nullopt) noexcept {
    if (!subscriber.direct_process_local || subscriber.direct_ingress_capacity == 0U) {
      return 0U;
    }
    std::uint32_t promoted = 0U;
    while (true) {
      const auto pending = std::min_element(
          subscriber.slots.begin(), subscriber.slots.end(), [](const auto& lhs, const auto& rhs) {
            const std::uint64_t lhs_sequence = lhs.occupied && lhs.direct_ingress_unread
                                                   ? lhs.temporal.source_sequence
                                                   : std::numeric_limits<std::uint64_t>::max();
            const std::uint64_t rhs_sequence = rhs.occupied && rhs.direct_ingress_unread
                                                   ? rhs.temporal.source_sequence
                                                   : std::numeric_limits<std::uint64_t>::max();
            return lhs_sequence < rhs_sequence;
          });
      if (pending == subscriber.slots.end() || !pending->occupied ||
          !pending->direct_ingress_unread) {
        if (subscriber.end_of_stream_received && !subscriber.end_of_stream_event_emitted &&
            subscriber.end_of_stream_arrival_order == 0U) {
          auto arrival_order = reserve_arrival_order();
          if (!arrival_order) {
            return make_unexpected(std::move(arrival_order).error());
          }
          subscriber.last_producer_sequence = subscriber.end_of_stream_transport_sequence;
          subscriber.end_of_stream_arrival_order = *arrival_order;
          trace_phase(phase_counters_,
                      RuntimePhaseTracePoint::kChannelEndOfStreamRetained,
                      subscriber.end_of_stream_transport_sequence);
          static_cast<void>(subscriber.input->publish_readiness_change());
        }
        return promoted;
      }
      SampleSlot* const destination = writable_slot(subscriber);
      if (destination == nullptr) {
        return promoted;
      }
      if (subscriber.next_arrival_sequence == std::numeric_limits<std::uint64_t>::max()) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      std::uint64_t arrival_order = 0U;
      if (first_arrival_order) {
        arrival_order = *first_arrival_order;
        first_arrival_order.reset();
      } else {
        auto reserved = reserve_arrival_order();
        if (!reserved) {
          return make_unexpected(std::move(reserved).error());
        }
        arrival_order = *reserved;
      }
      if (destination->occupied) {
        reset_sample_slot(*destination);
      }
      pending->temporal.producer_epoch = subscriber.producer_epoch;
      pending->temporal.arrival_sequence = subscriber.next_arrival_sequence;
      ++subscriber.next_arrival_sequence;
      pending->readiness_sequence = 0U;
      pending->arrival_order = arrival_order;
      pending->clock_health_disposition = EndpointClockHealthDisposition::kNotApplicable;
      pending->clock_health_generation = 0U;
      pending->event_emitted = false;
      pending->witness_retired = false;
      pending->active_selected = false;
      pending->reuse_retained = false;
      pending->temporal_retained = false;
      pending->retirement_selected = false;
      pending->direct_ingress_unread = false;
      subscriber.last_producer_sequence = pending->temporal.source_sequence;
      ++promoted;
      count_phase(phase_counters_, RuntimePhaseCounter::kChannelSamplesReceived);
      trace_phase(phase_counters_,
                  RuntimePhaseTracePoint::kChannelIngressRetained,
                  pending->temporal.source_sequence);
      static_cast<void>(subscriber.input->publish_readiness_change());
    }
  }

  [[nodiscard]] expected<std::optional<std::uint64_t>, Error>
  reserve_immediate_direct_ingress_promotion(SubscriberState& subscriber) noexcept {
    auto promoted = promote_direct_ingress(subscriber);
    if (!promoted) {
      return make_unexpected(std::move(promoted).error());
    }
    if (writable_slot(subscriber) == nullptr) {
      return std::nullopt;
    }
    if (subscriber.next_arrival_sequence == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    auto arrival_order = reserve_arrival_order();
    if (!arrival_order) {
      return make_unexpected(std::move(arrival_order).error());
    }
    return std::optional<std::uint64_t>{*arrival_order};
  }

  [[nodiscard]] expected<void, Error> validate_and_translate_catalog_capture(
      const SubscriberState& subscriber, SampleMetadata& metadata,
      std::optional<StructuralDiagnosticId>* const terminal_diagnostic) const noexcept {
    if (terminal_diagnostic != nullptr) {
      terminal_diagnostic->reset();
    }
    if (metadata.publish_clock_id.valid() || metadata.publish_timestamp_ns != 0) {
      HOLOSCAN_LOG_ERROR(
          "External Channel sample on port {} carried a noncanonical publication coordinate",
          subscriber.target.value());
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }

    const bool catalog_capture_present = subscriber.catalog_capture_clock.valid();
    const bool plan_capture_present = subscriber.plan_capture_clock.valid();
    if (catalog_capture_present != plan_capture_present) {
      HOLOSCAN_LOG_ERROR(
          "External Channel binding on port {} carried inconsistent catalog and plan capture "
          "clocks {} and {}",
          subscriber.target.value(),
          subscriber.catalog_capture_clock.value,
          subscriber.plan_capture_clock.value);
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (!catalog_capture_present) {
      if (metadata.source_clock_id.valid() || metadata.capture_timestamp_ns != 0) {
        HOLOSCAN_LOG_ERROR(
            "External Channel sample on port {} carried a capture coordinate for an uncaptured "
            "endpoint",
            subscriber.target.value());
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      return {};
    }
    if (!metadata.source_clock_id.valid()) {
      HOLOSCAN_LOG_ERROR("External Channel sample on port {} omitted promised capture clock {}",
                         subscriber.target.value(),
                         subscriber.catalog_capture_clock.value);
      if (terminal_diagnostic != nullptr) {
        *terminal_diagnostic = StructuralDiagnosticId::kTemporalCaptureTimeRequired;
      }
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (metadata.source_clock_id != subscriber.catalog_capture_clock) {
      HOLOSCAN_LOG_ERROR(
          "External Channel sample on port {} carried catalog capture clock {}; expected {} "
          "before translation to plan clock {}",
          subscriber.target.value(),
          metadata.source_clock_id.value,
          subscriber.catalog_capture_clock.value,
          subscriber.plan_capture_clock.value);
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }

    // Endpoint catalogs and execution plans own independently dense clock namespaces. Translate
    // only the metadata copy at the typed-retention boundary so every retained temporal identity
    // belongs to the enclosing plan.
    metadata.source_clock_id = subscriber.plan_capture_clock;
    return {};
  }

  // Retention validates the complete metadata, clock, sequence, capacity, and claim contract
  // before committing one sample slot; the fail-closed branches intentionally remain explicit.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<void, Error> store_received(
      SubscriberState& subscriber, SampleMetadata received,
      EndpointClockHealthDisposition clock_health_disposition,
      std::uint64_t clock_health_generation,
      std::span<const std::byte> borrowed_schema_payload = {},
      std::optional<StructuralDiagnosticId>* const terminal_diagnostic = nullptr,
      SampleSlot* const reserved_destination = nullptr) noexcept {
    HOLOSCAN_LOG_TRACE(
        "Store Channel sample on port {} with source clock {}; catalog clock {}, "
        "plan clock {}",
        subscriber.target.value(),
        received.source_clock_id.value,
        subscriber.catalog_capture_clock.value,
        subscriber.plan_capture_clock.value);
    const RuntimePortTypeOps& type_ops = *subscriber.input->type_ops;
    const bool payload_size_valid =
        type_ops.codec_family == RuntimePayloadCodecFamily::kGeneratedSchema
            ? received.byte_size != 0U && received.byte_size <= type_ops.wire_capacity
            : received.byte_size == type_ops.wire_capacity;
    if (received.version.major != 1U || received.version.minor != 0U || received.sequence == 0U ||
        (static_cast<std::uint32_t>(received.flags) & ~0x3U) != 0U || !payload_size_valid ||
        received.payload_kind != type_ops.payload_kind) {
      HOLOSCAN_LOG_ERROR("Channel sample on port {} carried malformed semantic metadata",
                         subscriber.target.value());
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (subscriber.externally_driven_import) {
      auto validated =
          validate_and_translate_catalog_capture(subscriber, received, terminal_diagnostic);
      if (!validated) {
        return make_unexpected(std::move(validated).error());
      }
    } else {
      const bool capture_coordinate_coherent =
          received.source_clock_id.valid() || received.capture_timestamp_ns == 0;
      const bool publish_coordinate_coherent =
          received.publish_clock_id.valid() || received.publish_timestamp_ns == 0;
      const bool capture_domain_admitted =
          !received.source_clock_id.valid() || !subscriber.input->capture_clock.valid() ||
          received.source_clock_id == subscriber.input->capture_clock;
      if (!capture_coordinate_coherent || !publish_coordinate_coherent ||
          !capture_domain_admitted) {
        HOLOSCAN_LOG_ERROR(
            "Process-local channel sample on port {} carried an invalid clock "
            "coordinate",
            subscriber.target.value());
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
    }
    const bool direct_ingress =
        subscriber.direct_process_local && subscriber.direct_ingress_capacity != 0U;
    if (direct_ingress && reserved_destination != nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const std::uint64_t prior_sequence = direct_ingress ? subscriber.last_direct_ingress_sequence
                                                        : subscriber.last_producer_sequence;
    if (prior_sequence != 0U && received.sequence <= prior_sequence) {
      HOLOSCAN_LOG_ERROR(
          "Channel subscriber received non-monotonic producer sequence {} after {} on port {}",
          received.sequence,
          prior_sequence,
          subscriber.target.value());
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    std::optional<std::uint64_t> immediate_arrival_order;
    if (direct_ingress) {
      auto reserved = reserve_immediate_direct_ingress_promotion(subscriber);
      if (!reserved) {
        return make_unexpected(std::move(reserved).error());
      }
      immediate_arrival_order = *reserved;
    }
    SampleSlot* const destination = reserved_destination != nullptr
                                        ? reserved_destination
                                        : (direct_ingress ? writable_direct_ingress_slot(subscriber)
                                                          : writable_slot(subscriber));
    if (destination == nullptr) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }

    if (!direct_ingress &&
        subscriber.next_arrival_sequence == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    std::uint64_t arrival_order = 0U;
    if (!direct_ingress) {
      auto reserved = reserve_arrival_order();
      if (!reserved) {
        return make_unexpected(std::move(reserved).error());
      }
      arrival_order = *reserved;
    }
    if (destination->occupied) {
      reset_sample_slot(*destination);
    }

    std::span<const std::byte> materialized_payload;
    if (!borrowed_schema_payload.empty()) {
      if (type_ops.codec_family != RuntimePayloadCodecFamily::kGeneratedSchema ||
          borrowed_schema_payload.size() != received.byte_size) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      materialized_payload = borrowed_schema_payload;
    } else if (type_ops.codec_family == RuntimePayloadCodecFamily::kGeneratedSchema &&
               !subscriber.externally_driven_import && !subscriber.cross_process_transport) {
      if (!subscriber.process_local_schema_identity) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      auto payload_size =
          channel_internal::decode_schema_envelope(subscriber.scratch_bytes(),
                                                   *subscriber.process_local_schema_identity,
                                                   type_ops.wire_capacity);
      if (!payload_size || *payload_size != received.byte_size) {
        return make_unexpected(payload_size ? Error{ErrorCode::kProtocolError}
                                            : std::move(payload_size).error());
      }
      materialized_payload =
          subscriber.scratch_bytes().subspan(kSchemaEnvelopeHeaderSize, *payload_size);
    } else {
      materialized_payload = subscriber.scratch_bytes().first(received.byte_size);
    }
    if (type_ops.codec_family == RuntimePayloadCodecFamily::kCustomPod &&
        !subscriber.externally_driven_import) {
      if (!subscriber.pod_identity || subscriber.native_scratch.size() != type_ops.payload_size) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      auto decoded = channel_internal::PodFieldwiseCodec::decode_prevalidated(
          *subscriber.pod_identity, materialized_payload, subscriber.native_scratch);
      if (!decoded) {
        Error error = std::move(decoded).error();
        error.code = ErrorCode::kProtocolError;
        return make_unexpected(std::move(error));
      }
      materialized_payload = subscriber.native_scratch;
    }
    auto constructed =
        type_ops.construct_sample(destination->sample.data(), materialized_payload, received);
    if (!constructed) {
      Error error = std::move(constructed).error();
      if (error.code == ErrorCode::kCodecError) {
        error.code = ErrorCode::kProtocolError;
      }
      return make_unexpected(std::move(error));
    }
    destination->temporal = TemporalSampleRecord{
        .capture_time = received.capture_time(),
        .source_sequence = received.sequence,
        .producer_epoch = subscriber.producer_epoch,
        .arrival_sequence = direct_ingress ? 0U : subscriber.next_arrival_sequence,
    };
    if (!direct_ingress) {
      ++subscriber.next_arrival_sequence;
    }
    destination->readiness_sequence = 0U;
    destination->arrival_order = arrival_order;
    destination->clock_health_disposition = clock_health_disposition;
    destination->clock_health_generation = clock_health_generation;
    destination->occupied = true;
    destination->event_emitted = false;
    destination->witness_retired = false;
    destination->active_selected = false;
    destination->reuse_retained = false;
    destination->temporal_retained = false;
    destination->retirement_selected = false;
    destination->direct_ingress_unread = direct_ingress;
    if (direct_ingress) {
      subscriber.last_direct_ingress_sequence = received.sequence;
      auto promoted = promote_direct_ingress(subscriber, immediate_arrival_order);
      if (!promoted) {
        return make_unexpected(std::move(promoted).error());
      }
    } else {
      subscriber.last_producer_sequence = received.sequence;
      count_phase(phase_counters_, RuntimePhaseCounter::kChannelSamplesReceived);
      trace_phase(
          phase_counters_, RuntimePhaseTracePoint::kChannelIngressRetained, received.sequence);
      static_cast<void>(subscriber.input->publish_readiness_change());
    }
    return {};
  }

  [[nodiscard]] expected<void, Error> store_direct_inline_payload(
      SubscriberState& subscriber, std::span<const std::byte> payload,
      SampleMetadata metadata) noexcept {
    // REQ: CHAN-096
    // REQ: CHAN-097
    const bool generated_schema =
        subscriber.input->type_ops->codec_family == RuntimePayloadCodecFamily::kGeneratedSchema;
    const bool payload_size_valid = generated_schema
                                        ? payload.size() >= kSchemaEnvelopeHeaderSize &&
                                              payload.size() <= subscriber.scratch_size
                                        : payload.size() == subscriber.scratch_size;
    if (!subscriber.direct_process_local || subscriber.endpoint != nullptr ||
        subscriber.tensor_endpoint.has_value() || subscriber.tensor_schema_endpoint != nullptr ||
        generated_schema != subscriber.process_local_schema_identity.has_value() ||
        !payload_size_valid) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const bool signal_progress = owner_progress_wake_enabled();
    const std::uint64_t prior_revision =
        signal_progress ? subscriber.input->readiness_revision.load(std::memory_order_acquire) : 0U;
    expected<void, Error> stored;
    {
      const std::scoped_lock cache_lock{subscriber.mutex};
      std::copy(payload.begin(), payload.end(), subscriber.scratch_bytes().begin());
      stored =
          store_received(subscriber, metadata, EndpointClockHealthDisposition::kNotApplicable, 0U);
    }
    if (stored && signal_progress &&
        subscriber.input->readiness_revision.load(std::memory_order_acquire) != prior_revision) {
      signal_owner_progress(subscriber.owner);
    }
    return stored;
  }

  // Keep receive admission, metadata fencing, and cache mutation ordered as one transaction.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<void, Error> store_received_tensor(
      SubscriberState& subscriber, Sample<Tensor> received,
      std::optional<StructuralDiagnosticId>* const terminal_diagnostic = nullptr,
      SampleSlot* const reserved_destination = nullptr) noexcept {
    const TensorAccessSnapshot snapshot = TensorAccess::snapshot(received.data);
    SampleMetadata metadata = received.metadata;
    if (subscriber.catalog_tensor_import) {
      auto validated =
          validate_and_translate_catalog_capture(subscriber, metadata, terminal_diagnostic);
      if (!validated) {
        return make_unexpected(std::move(validated).error());
      }
    }
    const bool capture_coordinate_coherent =
        metadata.source_clock_id.valid() || metadata.capture_timestamp_ns == 0;
    const bool publish_coordinate_coherent =
        metadata.publish_clock_id.valid() || metadata.publish_timestamp_ns == 0;
    const bool capture_domain_admitted =
        !metadata.source_clock_id.valid() || !subscriber.input->capture_clock.valid() ||
        metadata.source_clock_id == subscriber.input->capture_clock;
    if (metadata.version.major != 1U || metadata.version.minor != 0U || metadata.sequence == 0U ||
        (static_cast<std::uint32_t>(metadata.flags) & ~0x3U) != 0U ||
        metadata.payload_kind != PayloadKind::kDescriptorBacked ||
        metadata.byte_size != snapshot.byte_size || !capture_coordinate_coherent ||
        !publish_coordinate_coherent || !capture_domain_admitted ||
        !tensor_snapshot_matches_plan(snapshot,
                                      subscriber.tensor_max_byte_span,
                                      subscriber.tensor_memory_kind,
                                      subscriber.tensor_device) ||
        (subscriber.pass_through_tensor_layout &&
         !tensor_matches_port_layout(
             received.data, snapshot, *subscriber.pass_through_tensor_layout)) ||
        (subscriber.tensor_logical_requirements &&
         !tensor_matches_logical_requirements(received.data,
                                              *subscriber.tensor_logical_requirements))) {
      HOLOSCAN_LOG_ERROR("Tensor Channel sample on port {} violated its frozen payload contract",
                         subscriber.target.value());
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    received.metadata = metadata;
    const bool direct_ingress =
        subscriber.direct_process_local && subscriber.direct_ingress_capacity != 0U;
    if (direct_ingress && reserved_destination != nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const std::uint64_t prior_sequence = direct_ingress ? subscriber.last_direct_ingress_sequence
                                                        : subscriber.last_producer_sequence;
    if (prior_sequence != 0U && metadata.sequence <= prior_sequence) {
      HOLOSCAN_LOG_ERROR(
          "Tensor Channel subscriber received non-monotonic producer sequence {} after {} on "
          "port {}",
          metadata.sequence,
          prior_sequence,
          subscriber.target.value());
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    std::optional<std::uint64_t> immediate_arrival_order;
    if (direct_ingress) {
      auto reserved = reserve_immediate_direct_ingress_promotion(subscriber);
      if (!reserved) {
        return make_unexpected(std::move(reserved).error());
      }
      immediate_arrival_order = *reserved;
    }
    SampleSlot* const destination = reserved_destination != nullptr
                                        ? reserved_destination
                                        : (direct_ingress ? writable_direct_ingress_slot(subscriber)
                                                          : writable_slot(subscriber));
    if (destination == nullptr) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    if (!direct_ingress &&
        subscriber.next_arrival_sequence == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    std::uint64_t arrival_order = 0U;
    if (!direct_ingress) {
      auto reserved = reserve_arrival_order();
      if (!reserved) {
        return make_unexpected(std::move(reserved).error());
      }
      arrival_order = *reserved;
    }
    if (destination->occupied) {
      reset_sample_slot(*destination);
    }

    std::construct_at(static_cast<Sample<Tensor>*>(destination->sample.data()),
                      std::move(received));
    destination->temporal = TemporalSampleRecord{
        .capture_time = metadata.capture_time(),
        .source_sequence = metadata.sequence,
        .producer_epoch = subscriber.producer_epoch,
        .arrival_sequence = direct_ingress ? 0U : subscriber.next_arrival_sequence,
    };
    if (!direct_ingress) {
      ++subscriber.next_arrival_sequence;
    }
    destination->readiness_sequence = 0U;
    destination->arrival_order = arrival_order;
    destination->clock_health_disposition = EndpointClockHealthDisposition::kNotApplicable;
    destination->clock_health_generation = 0U;
    destination->occupied = true;
    destination->event_emitted = false;
    destination->witness_retired = false;
    destination->active_selected = false;
    destination->reuse_retained = false;
    destination->temporal_retained = false;
    destination->retirement_selected = false;
    destination->direct_ingress_unread = direct_ingress;
    if (direct_ingress) {
      subscriber.last_direct_ingress_sequence = metadata.sequence;
      auto promoted = promote_direct_ingress(subscriber, immediate_arrival_order);
      if (!promoted) {
        return make_unexpected(std::move(promoted).error());
      }
    } else {
      subscriber.last_producer_sequence = metadata.sequence;
      count_phase(phase_counters_, RuntimePhaseCounter::kChannelSamplesReceived);
      static_cast<void>(subscriber.input->publish_readiness_change());
    }
    return {};
  }

  [[nodiscard]] expected<void, Error> store_direct_tensor_sample(SubscriberState& subscriber,
                                                                 Sample<Tensor> received) noexcept {
    // REQ: CHAN-098
    if (!subscriber.direct_process_local || subscriber.endpoint != nullptr ||
        subscriber.tensor_endpoint.has_value() || subscriber.tensor_schema_endpoint != nullptr ||
        subscriber.input->type_ops != runtime_port_type_ops<Tensor>()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const bool signal_progress = owner_progress_wake_enabled();
    const std::uint64_t prior_revision =
        signal_progress ? subscriber.input->readiness_revision.load(std::memory_order_acquire) : 0U;
    expected<void, Error> stored;
    {
      const std::scoped_lock cache_lock{subscriber.mutex};
      stored = store_received_tensor(subscriber, std::move(received));
    }
    if (stored && signal_progress &&
        subscriber.input->readiness_revision.load(std::memory_order_acquire) != prior_revision) {
      signal_owner_progress(subscriber.owner);
    }
    return stored;
  }

  // Keep schema decoding, field validation, and cache publication in one receive transaction.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<void, Error> store_received_tensor_schema(
      SubscriberState& subscriber, const std::span<const std::byte> encoded,
      const std::span<Tensor> received, const SampleMetadata metadata,
      SampleSlot* const reserved_destination = nullptr) noexcept {
    // Catalog Tensor services admit only the raw Tensor codec. Descriptor-bearing schemas are
    // materialized exclusively by intra-plan endpoints and therefore never own catalog/plan
    // capture-clock translation state.
    const RuntimePortTypeOps& ops = *subscriber.input->type_ops;
    if (received.empty() || received.size() != ops.descriptor_tensor_static_count ||
        (!subscriber.tensor_fields.empty() && subscriber.tensor_fields.size() != received.size())) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    std::uint64_t aggregate_byte_size = 0U;
    for (std::size_t index = 0U; index < received.size(); ++index) {
      const TensorAccessSnapshot snapshot = TensorAccess::snapshot(received[index]);
      const bool matches =
          subscriber.tensor_fields.empty()
              ? tensor_snapshot_matches_plan(snapshot,
                                             subscriber.tensor_max_byte_span,
                                             subscriber.tensor_memory_kind,
                                             subscriber.tensor_device)
              : subscriber.tensor_fields[index].tensor_index == index &&
                    tensor_matches_field_contract(
                        received[index], snapshot, subscriber.tensor_fields[index]);
      if (!matches ||
          snapshot.byte_size > std::numeric_limits<std::uint64_t>::max() - aggregate_byte_size) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      aggregate_byte_size += snapshot.byte_size;
    }
    const bool capture_coordinate_coherent =
        metadata.source_clock_id.valid() || metadata.capture_timestamp_ns == 0;
    const bool publish_coordinate_coherent =
        metadata.publish_clock_id.valid() || metadata.publish_timestamp_ns == 0;
    const bool capture_domain_admitted =
        !metadata.source_clock_id.valid() || !subscriber.input->capture_clock.valid() ||
        metadata.source_clock_id == subscriber.input->capture_clock;
    const bool pass_through_layout_matches =
        !subscriber.pass_through_tensor_layout ||
        (received.size() == 1U &&
         tensor_matches_port_layout(received.front(),
                                    TensorAccess::snapshot(received.front()),
                                    *subscriber.pass_through_tensor_layout));
    if (metadata.version.major != 1U || metadata.version.minor != 0U || metadata.sequence == 0U ||
        (static_cast<std::uint32_t>(metadata.flags) & ~0x3U) != 0U ||
        metadata.payload_kind != PayloadKind::kDescriptorBacked ||
        metadata.byte_size != aggregate_byte_size || !capture_coordinate_coherent ||
        !publish_coordinate_coherent || !capture_domain_admitted || !pass_through_layout_matches ||
        ops.assign_descriptor_tensors_sample == nullptr) {
      HOLOSCAN_LOG_ERROR(
          "Tensor-bearing schema sample on port {} violated its frozen payload contract",
          subscriber.target.value());
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const bool direct_ingress =
        subscriber.direct_process_local && subscriber.direct_ingress_capacity != 0U;
    if (direct_ingress && reserved_destination != nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const std::uint64_t prior_sequence = direct_ingress ? subscriber.last_direct_ingress_sequence
                                                        : subscriber.last_producer_sequence;
    if (prior_sequence != 0U && metadata.sequence <= prior_sequence) {
      HOLOSCAN_LOG_ERROR(
          "Tensor-bearing schema subscriber received non-monotonic producer sequence {} after {} "
          "on port {}",
          metadata.sequence,
          prior_sequence,
          subscriber.target.value());
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    std::optional<std::uint64_t> immediate_arrival_order;
    if (direct_ingress) {
      auto reserved = reserve_immediate_direct_ingress_promotion(subscriber);
      if (!reserved) {
        return make_unexpected(std::move(reserved).error());
      }
      immediate_arrival_order = *reserved;
    }
    SampleSlot* const destination = reserved_destination != nullptr
                                        ? reserved_destination
                                        : (direct_ingress ? writable_direct_ingress_slot(subscriber)
                                                          : writable_slot(subscriber));
    if (destination == nullptr) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    if (!direct_ingress &&
        subscriber.next_arrival_sequence == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    std::uint64_t arrival_order = 0U;
    if (!direct_ingress) {
      auto reserved = reserve_arrival_order();
      if (!reserved) {
        return make_unexpected(std::move(reserved).error());
      }
      arrival_order = *reserved;
    }
    if (destination->occupied) {
      reset_sample_slot(*destination);
    }
    auto assigned = ops.assign_descriptor_tensors_sample(
        destination->sample.data(), encoded, received, metadata);
    if (!assigned) {
      return make_unexpected(std::move(assigned).error());
    }
    destination->temporal = TemporalSampleRecord{
        .capture_time = metadata.capture_time(),
        .source_sequence = metadata.sequence,
        .producer_epoch = subscriber.producer_epoch,
        .arrival_sequence = direct_ingress ? 0U : subscriber.next_arrival_sequence,
    };
    if (!direct_ingress) {
      ++subscriber.next_arrival_sequence;
    }
    destination->readiness_sequence = 0U;
    destination->arrival_order = arrival_order;
    destination->clock_health_disposition = EndpointClockHealthDisposition::kNotApplicable;
    destination->clock_health_generation = 0U;
    destination->occupied = true;
    destination->event_emitted = false;
    destination->witness_retired = false;
    destination->active_selected = false;
    destination->reuse_retained = false;
    destination->temporal_retained = false;
    destination->retirement_selected = false;
    destination->direct_ingress_unread = direct_ingress;
    if (direct_ingress) {
      subscriber.last_direct_ingress_sequence = metadata.sequence;
      auto promoted = promote_direct_ingress(subscriber, immediate_arrival_order);
      if (!promoted) {
        return make_unexpected(std::move(promoted).error());
      }
    } else {
      subscriber.last_producer_sequence = metadata.sequence;
      count_phase(phase_counters_, RuntimePhaseCounter::kChannelSamplesReceived);
      static_cast<void>(subscriber.input->publish_readiness_change());
    }
    return {};
  }

  [[nodiscard]] expected<void, Error> store_direct_tensor_schema_sample(
      SubscriberState& subscriber, const std::span<const std::byte> encoded,
      const std::span<const Tensor> received) noexcept {
    // REQ: CHAN-099
    const RuntimePortTypeOps& ops = *subscriber.input->type_ops;
    if (!subscriber.direct_process_local || subscriber.endpoint != nullptr ||
        subscriber.tensor_endpoint.has_value() || subscriber.tensor_schema_endpoint != nullptr ||
        ops.codec_family != RuntimePayloadCodecFamily::kDescriptorTensorSchema ||
        ops.descriptor_tensor_sample_metadata == nullptr ||
        ops.assign_descriptor_tensors_sample == nullptr ||
        received.size() != ops.descriptor_tensor_static_count ||
        received.size() > channel_internal::kMaximumDescriptorTensorFields || encoded.empty() ||
        encoded.size() > ops.wire_capacity) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const bool signal_progress = owner_progress_wake_enabled();
    const std::uint64_t prior_revision =
        signal_progress ? subscriber.input->readiness_revision.load(std::memory_order_acquire) : 0U;
    auto& retained = subscriber.tensor_schema_scratch;
    if (retained.size() != received.size()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const TensorScratchReset reset{retained};
    std::copy(received.begin(), received.end(), retained.begin());
    expected<void, Error> stored;
    {
      const std::scoped_lock cache_lock{subscriber.mutex};
      auto metadata = ops.descriptor_tensor_sample_metadata(encoded, 0U, plan_generation_);
      if (!metadata) {
        return make_unexpected(std::move(metadata).error());
      }
      stored = store_received_tensor_schema(
          subscriber, encoded, std::span<Tensor>{retained}, std::move(*metadata));
    }
    if (stored && signal_progress &&
        subscriber.input->readiness_revision.load(std::memory_order_acquire) != prior_revision) {
      signal_owner_progress(subscriber.owner);
    }
    return stored;
  }

  [[nodiscard]] bool owner_progress_wake_enabled() const noexcept {
    return readiness_wake_.signal_owner_progress != nullptr;
  }

  void signal_owner_progress(VertexId owner) noexcept {
    if (!owner_progress_wake_enabled()) {
      return;
    }
    trace_phase(phase_counters_, RuntimePhaseTracePoint::kChannelReadinessWakeSignaled);
    readiness_wake_.signal_owner_progress(readiness_wake_.state, owner);
  }

  [[nodiscard]] expected<void, Error> store_end_of_stream(
      SubscriberState& subscriber, const std::uint64_t transport_sequence) noexcept {
    if (subscriber.end_of_stream_received) {
      if (subscriber.end_of_stream_transport_sequence == transport_sequence) {
        return {};
      }
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const std::uint64_t prior_sequence =
        std::max(subscriber.last_producer_sequence, subscriber.last_direct_ingress_sequence);
    if (transport_sequence == 0U ||
        (prior_sequence != 0U && transport_sequence <= prior_sequence)) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    std::uint64_t arrival_order = 0U;
    if (!subscriber.has_pending_direct_ingress()) {
      auto reserved = reserve_arrival_order();
      if (!reserved) {
        return make_unexpected(std::move(reserved).error());
      }
      arrival_order = *reserved;
    }
    if (arrival_order != 0U) {
      subscriber.last_producer_sequence = transport_sequence;
    }
    subscriber.end_of_stream_transport_sequence = transport_sequence;
    subscriber.end_of_stream_arrival_order = arrival_order;
    subscriber.end_of_stream_received = true;
    if (arrival_order != 0U) {
      trace_phase(
          phase_counters_, RuntimePhaseTracePoint::kChannelEndOfStreamRetained, transport_sequence);
      static_cast<void>(subscriber.input->publish_readiness_change());
    }
    return {};
  }

  [[nodiscard]] expected<void, Error> receive_tensor(
      SubscriberState& subscriber, ChannelSubscriber<Tensor>& endpoint,
      CudaDeviceCursor* scan_device, TypedIngressReservation* const reservation,
      std::optional<StructuralDiagnosticId>* const terminal_diagnostic) noexcept {
    std::optional<CudaDeviceCursor> local_device;
    CudaDeviceCursor* device = scan_device;
    if (subscriber.uses_device_affine_descriptor_import()) {
      if (device == nullptr) {
        local_device.emplace();
        device = &*local_device;
      }
      if (!device->select(subscriber.tensor_device)) {
        return make_unexpected(Error{ErrorCode::kNotSupported});
      }
    }
    auto received =
        RuntimeChannelBindingAccess::receive_tensor_sample(endpoint, reservation != nullptr);
    if (!received) {
      if (received.error().code == ErrorCode::kClosed) {
        const std::uint64_t end_of_stream_sequence =
            RuntimeChannelBindingAccess::accepted_tensor_end_of_stream_sequence(endpoint);
        if (end_of_stream_sequence != 0U) {
          if (reservation != nullptr) {
            reservation->release();
          }
          const std::scoped_lock cache_lock{subscriber.mutex};
          return store_end_of_stream(subscriber, end_of_stream_sequence);
        }
      }
      return make_unexpected(std::move(received).error());
    }
    if (reservation == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    observe_typed_ingress_for_test(subscriber);
    const std::scoped_lock cache_lock{subscriber.mutex};
    SampleSlot* const destination = reservation->commit_locked();
    return store_received_tensor(
        subscriber, std::move(*received), terminal_diagnostic, destination);
  }

  // One receive transition arbitrates data, terminal, direct, and transported payload paths.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<void, Error> receive_one(
      SubscriberState& subscriber,
      std::optional<StructuralDiagnosticId>* const terminal_diagnostic = nullptr,
      CudaDeviceCursor* const scan_device = nullptr) noexcept {
    const std::scoped_lock transport_lock{subscriber.transport_mutex};
    if (subscriber.tensor_endpoint.has_value()) {
      auto reservation = reserve_typed_ingress_destination(subscriber);
      if (!reservation && !subscriber.catalog_tensor_import) {
        return make_unexpected(Error{ErrorCode::kBackpressured});
      }
      // The source-private Tensor subscriber stages one bounded transport head. Even when typed
      // catalog retention is full it can accept ordered EOS without consuming a data head; staged
      // catalog data remains pending until activation retirement makes a typed slot writable.
      // Ordinary Graph-edge Tensor queues keep their original transport-pressure boundary.
      return receive_tensor(subscriber,
                            subscriber.tensor_endpoint.value(),
                            scan_device,
                            reservation ? &*reservation : nullptr,
                            terminal_diagnostic);
    }
    if (subscriber.tensor_schema_endpoint != nullptr) {
      auto reservation = reserve_typed_ingress_destination(subscriber);
      if (!reservation) {
        return make_unexpected(Error{ErrorCode::kBackpressured});
      }
      // Tensor schemas retain sidecar claims in-process and import BufferHandles across a process
      // boundary. Their nested Tensor claims are prepared later on the exact activation stream.
      std::optional<CudaDeviceCursor> local_device;
      CudaDeviceCursor* device = scan_device;
      if (subscriber.uses_device_affine_descriptor_import()) {
        if (device == nullptr) {
          local_device.emplace();
          device = &*local_device;
        }
        if (!device->select(subscriber.tensor_device)) {
          return make_unexpected(Error{ErrorCode::kNotSupported});
        }
      }
      auto payload_loan = subscriber.tensor_schema_endpoint->borrow_payload();
      if (!payload_loan) {
        return make_unexpected(std::move(payload_loan).error());
      }
      const std::span<const std::byte> wire_payload = payload_loan->payload();
      const RuntimePortTypeOps& ops = *subscriber.input->type_ops;
      const std::uint32_t tensor_count = ops.descriptor_tensor_static_count;
      if (tensor_count == 0U || tensor_count > channel_internal::kMaximumDescriptorTensorFields ||
          ops.descriptor_offset > wire_payload.size() ||
          static_cast<std::size_t>(tensor_count) * sizeof(PayloadDescriptor) >
              wire_payload.size() - ops.descriptor_offset ||
          ops.descriptor_tensor_sample_metadata == nullptr ||
          ops.descriptor_tensor_view_metadata_at == nullptr ||
          ops.assign_descriptor_tensors_sample == nullptr) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      auto& descriptors = subscriber.descriptor_tensor_scratch;
      auto& tensors = subscriber.tensor_schema_scratch;
      if (descriptors.size() != tensor_count || tensors.size() != tensor_count) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      const TensorScratchReset reset{tensors};
      for (std::uint32_t index = 0U; index < tensor_count; ++index) {
        auto descriptor = channel_internal::decode_payload_descriptor_wire(wire_payload.subspan(
            ops.descriptor_offset + static_cast<std::size_t>(index) * sizeof(PayloadDescriptor),
            sizeof(PayloadDescriptor)));
        if (!descriptor) {
          for (std::uint32_t release_index = 0U; release_index < index; ++release_index) {
            subscriber.tensor_schema_endpoint->release_recipient_claim(descriptors[release_index]);
          }
          return make_unexpected(std::move(descriptor).error());
        }
        descriptors[index] = *descriptor;
      }
      const std::uint64_t carriage_mask = tensor_count == 64U
                                              ? std::numeric_limits<std::uint64_t>::max()
                                              : (std::uint64_t{1U} << tensor_count) - 1U;
      auto release_claims_from = [&](const std::uint32_t first) noexcept {
        for (std::uint32_t index = first; index < tensor_count; ++index) {
          subscriber.tensor_schema_endpoint->release_recipient_claim(descriptors[index]);
        }
      };
      auto metadata =
          ops.descriptor_tensor_sample_metadata(wire_payload, carriage_mask, plan_generation_);
      if (!metadata) {
        release_claims_from(0U);
        return make_unexpected(std::move(metadata).error());
      }
      for (std::uint32_t index = 0U; index < tensor_count; ++index) {
        expected<Tensor, Error> tensor = make_unexpected(Error{ErrorCode::kProtocolError});
        if (descriptors[index].payload_kind == ChannelPayloadKind::kBufferHandle) {
          auto tensor_metadata =
              ops.descriptor_tensor_view_metadata_at(wire_payload, index, plan_generation_);
          if (!tensor_metadata) {
            release_claims_from(index);
            return make_unexpected(std::move(tensor_metadata).error());
          }
          tensor = subscriber.tensor_schema_endpoint->import_buffer_handle_tensor(
              descriptors[index], *tensor_metadata, index);
        } else {
          tensor = subscriber.tensor_schema_endpoint->acquire_tensor(descriptors[index]);
        }
        if (!tensor) {
          release_claims_from(index);
          return make_unexpected(std::move(tensor).error());
        }
        tensors[index] = std::move(*tensor);
      }
      observe_typed_ingress_for_test(subscriber);
      const std::scoped_lock cache_lock{subscriber.mutex};
      SampleSlot* const destination = reservation->commit_locked();
      return store_received_tensor_schema(
          subscriber, wire_payload, std::span<Tensor>{tensors}, *metadata, destination);
    }
    if (subscriber.endpoint == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    // Reserve exact typed capacity before popping the transport head. The reservation remains
    // independently visible while transport receive runs without the cache mutex, so activation
    // cannot claim or replace the destination in that window.
    auto reservation = reserve_typed_ingress_destination(subscriber);
    if (!reservation) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    if (subscriber.input->type_ops->codec_family == RuntimePayloadCodecFamily::kGeneratedSchema) {
      auto received = subscriber.endpoint->receive_borrowed();
      if (!received) {
        return make_unexpected(std::move(received).error());
      }
      std::span<const std::byte> schema_payload = received->payload();
      if (subscriber.process_local_schema_identity) {
        auto payload_size =
            channel_internal::decode_schema_envelope(schema_payload,
                                                     *subscriber.process_local_schema_identity,
                                                     subscriber.input->type_ops->wire_capacity);
        if (!payload_size || *payload_size != received->metadata().byte_size) {
          return make_unexpected(payload_size ? Error{ErrorCode::kProtocolError}
                                              : std::move(payload_size).error());
        }
        schema_payload = schema_payload.subspan(kSchemaEnvelopeHeaderSize, *payload_size);
      }
      observe_typed_ingress_for_test(subscriber);
      const std::scoped_lock cache_lock{subscriber.mutex};
      SampleSlot* const destination = reservation->commit_locked();
      return store_received(subscriber,
                            received->metadata(),
                            EndpointClockHealthDisposition::kNotApplicable,
                            0U,
                            schema_payload,
                            terminal_diagnostic,
                            destination);
    }
    auto received = subscriber.endpoint->receive(subscriber.scratch_bytes());
    if (!received) {
      return make_unexpected(std::move(received).error());
    }
    observe_typed_ingress_for_test(subscriber);
    const std::scoped_lock cache_lock{subscriber.mutex};
    SampleSlot* const destination = reservation->commit_locked();
    return store_received(subscriber,
                          *received,
                          EndpointClockHealthDisposition::kNotApplicable,
                          0U,
                          {},
                          terminal_diagnostic,
                          destination);
  }

  // Catalog draining preserves ordered health/data heads and the finite per-dispatch budget.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<void, Error> drain_monitored_inputs(
      const ActivationOwnerState& owner) noexcept {
    for (SubscriberState* const subscriber_pointer : owner.subscribers) {
      SubscriberState& subscriber = *subscriber_pointer;
      if (!subscriber.externally_driven_import) {
        continue;
      }
      const bool monitored = static_cast<bool>(subscriber.clock_health_endpoint);
      if (monitored && (subscriber.clock_health_coordinator == nullptr ||
                        subscriber.health_dispatch_budget == 0U)) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }

      const std::scoped_lock transport_lock{subscriber.transport_mutex};
      std::uint32_t data_count = 0U;
      while (data_count < subscriber.drain_capacity) {
        std::optional<TypedIngressReservation> reservation;
        {
          const std::scoped_lock cache_lock{subscriber.mutex};
          if (subscriber.end_of_stream_received) {
            break;
          }
          auto selected = reserve_typed_ingress_destination_locked(subscriber);
          if (selected) {
            reservation.emplace(std::move(*selected));
          }
        }
        auto operation = channel_internal::CatalogChannelBindingAccess::receive_next(
            *subscriber.endpoint, subscriber.scratch_bytes(), reservation.has_value(), true);
        if (!operation) {
          return make_unexpected(std::move(operation).error());
        }
        if (operation->kind == channel_internal::CatalogReceivedOperationKind::kNone) {
          break;
        }
        if (operation->kind == channel_internal::CatalogReceivedOperationKind::kDataPending) {
          // The catalog subscriber retains this contiguous head in its bounded staging slot. Leave
          // its sequence uncommitted until activation retirement makes typed storage writable.
          count_phase(phase_counters_, RuntimePhaseCounter::kChannelReceiveBackpressured);
          break;
        }
        if (operation->kind ==
            channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged) {
          // The catalog subscriber fenced traffic from another producer incarnation: it delivered
          // no payload, kept EOS state unchanged, and mutated no receiver state. Keep the accepted
          // old-generation state fail-closed; only an explicit generation-fenced rebind (a reset
          // form) may cut over to the new incarnation.
          break;
        }

        const bool data_operation =
            operation->kind == channel_internal::CatalogReceivedOperationKind::kData;
        if (!data_operation && reservation) {
          reservation->release();
        }
        if (data_operation) {
          observe_typed_ingress_for_test(subscriber);
        }

        // The endpoint and scratch remain single-consumer under transport_mutex. Limit the typed-
        // cache mutex to applying the received operation so readiness polling and activation
        // retirement do not wait on transport receive or first-use Tensor import work.
        const std::scoped_lock cache_lock{subscriber.mutex};
        if (operation->kind == channel_internal::CatalogReceivedOperationKind::kClockHealth) {
          if (!monitored) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
          if (clock_health_serialization_mutex_ == nullptr) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
          const auto& health = operation->clock_health;
          // The wire producer field carries the producer-incarnation identity token, not an
          // ordinal. The catalog subscriber only surfaces operations of the latched incarnation,
          // so an applied report must match that token exactly; the Runtime-local producer_epoch
          // below stays a small local ordinal advanced by reset/rebind, never a wire value.
          auto latched = channel_internal::CatalogChannelBindingAccess::latched_producer_generation(
              *subscriber.endpoint);
          if (!latched) {
            return make_unexpected(std::move(latched).error());
          }
          if (*latched == 0U || health.producer_epoch != *latched) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
          // Keep report enqueue and dispatch in one coordinator transaction. Expiry and reset use
          // the same cold-bound mutex, so neither can interleave between report validation and the
          // state transition that the following data operation observes. The transport/cache
          // locks are always acquired before this innermost lock; no clock-health path acquires
          // them in the reverse order.
          const std::lock_guard clock_health_lock{*clock_health_serialization_mutex_};
          auto submitted = subscriber.clock_health_coordinator->submit_report(
              subscriber.clock_health_endpoint,
              ClockHealthReport{.assessment = health.assessment, .reason = health.reason},
              subscriber.producer_epoch,
              health.publisher_operation_sequence,
              health.report_sequence);
          if (!submitted) {
            return make_unexpected(
                Error{submitted.error().code, std::string{submitted.error().token}});
          }
          auto applied = subscriber.clock_health_coordinator->dispatch_pending(
              subscriber.clock_health_endpoint);
          if (!applied || *applied != 1U) {
            return make_unexpected(
                applied ? Error{ErrorCode::kProtocolError}
                        : Error{applied.error().code, std::string{applied.error().token}});
          }
          // The EA1 scalar plan admits one report per Dispatcher pass. Stop after applying it so
          // report bursts cannot starve data, expiry, lifecycle, or stop work; the next lane poll
          // resumes from the catalog subscriber's two retained service heads.
          break;
        }

        if (operation->kind == channel_internal::CatalogReceivedOperationKind::kEndOfStream) {
          auto stored_end_of_stream = store_end_of_stream(subscriber, operation->metadata.sequence);
          if (!stored_end_of_stream) {
            return make_unexpected(std::move(stored_end_of_stream).error());
          }
          break;
        }

        EndpointClockHealthDisposition health_disposition =
            EndpointClockHealthDisposition::kNotApplicable;
        std::uint64_t health_generation = 0U;
        if (monitored) {
          if (clock_health_serialization_mutex_ == nullptr) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
          // Freeze the disposition and generation as one health snapshot. A concurrent expiry or
          // reset may proceed only after this sample has retained the complete prior snapshot.
          const std::lock_guard clock_health_lock{*clock_health_serialization_mutex_};
          EndpointClockHealthReceiver* const receiver =
              subscriber.clock_health_coordinator->receiver(subscriber.clock_health_endpoint);
          if (receiver == nullptr) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
          health_disposition = receiver->current_disposition();
          health_generation = receiver->state().health_generation;
        }
        std::optional<StructuralDiagnosticId> terminal_diagnostic;
        if (!reservation) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        SampleSlot* const destination = reservation->commit_locked();
        auto stored = store_received(subscriber,
                                     operation->metadata,
                                     health_disposition,
                                     health_generation,
                                     operation->schema_payload,
                                     &terminal_diagnostic,
                                     destination);
        if (!stored) {
          // Every terminal transport/storage rejection closes the session. The optional
          // structural identity enriches attribution; it must not determine lifecycle state.
          fail(stored.error(), subscriber.owner, terminal_diagnostic);
          return make_unexpected(std::move(stored).error());
        }
        ++data_count;
      }
    }
    return {};
  }

  [[nodiscard]] expected<bool, Error> drain_unmonitored_inputs(
      PortId& failed_port, VertexId& failed_owner,
      std::optional<StructuralDiagnosticId>& terminal_diagnostic,
      CudaDeviceCursor* scan_device = nullptr) noexcept {
    failed_port = {};
    failed_owner = {};
    terminal_diagnostic.reset();
    bool readiness_published = false;
    std::optional<CudaDeviceCursor> local_device;
    for (SubscriberState* const subscriber_pointer : pump_subscribers_) {
      SubscriberState& subscriber = *subscriber_pointer;
      for (std::uint32_t count = 0U; count < subscriber.drain_capacity; ++count) {
        auto has_samples = subscriber.has_pending_transport();
        if (!has_samples) {
          failed_port = subscriber.target;
          failed_owner = subscriber.owner;
          return make_unexpected(std::move(has_samples).error());
        }
        if (!*has_samples) {
          break;
        }
        CudaDeviceCursor* receive_device = scan_device;
        if (subscriber.uses_device_affine_descriptor_import() && receive_device == nullptr) {
          if (!local_device.has_value()) {
            local_device.emplace();
          }
          receive_device = &*local_device;
        }
        auto received = receive_one(subscriber, &terminal_diagnostic, receive_device);
        if (!received) {
          if (received.error().code == ErrorCode::kBackpressured) {
            // Typed retention is full, but receive_one() has retained the transport head either in
            // the backend queue or in the raw-Tensor endpoint's one-record staging claim. Retry
            // after activation retirement frees a slot, while giving other process-local
            // subscribers their bounded opportunity now.
            blocked_transport_retry_required_.store(true, std::memory_order_release);
            count_phase(phase_counters_, RuntimePhaseCounter::kChannelReceiveBackpressured);
            break;
          }
          failed_port = subscriber.target;
          failed_owner = subscriber.owner;
          return make_unexpected(std::move(received).error());
        }
        record_process_local_receive();
        readiness_published = true;
        signal_owner_progress(subscriber.owner);
      }
    }
    return readiness_published;
  }

  void record_process_local_publication(const std::uint32_t deliveries) noexcept {
    record_process_local_progress(process_local_published_deliveries_, deliveries);
  }

  void record_process_local_receive() noexcept {
    record_process_local_progress(process_local_received_deliveries_, 1U);
  }

  void invalidate_process_local_progress() noexcept {
    process_local_progress_tracking_valid_.store(false, std::memory_order_release);
  }

  void record_process_local_progress(std::atomic<std::uint64_t>& counter,
                                     const std::uint32_t delta) noexcept {
    std::uint64_t current = counter.load(std::memory_order_relaxed);
    while (current <= std::numeric_limits<std::uint64_t>::max() - delta) {
      if (counter.compare_exchange_weak(
              current, current + delta, std::memory_order_release, std::memory_order_relaxed)) {
        return;
      }
    }
    process_local_progress_tracking_valid_.store(false, std::memory_order_release);
  }

  [[nodiscard]] expected<void, Error> wait_for_transport_activity() noexcept {
    if (wait_attachment_count_ == 0U) {
      std::this_thread::sleep_for(kPumpWait);
      return {};
    }
    auto waited = wait_set_->wait_for(kPumpWait);
    if (!waited) {
      HOLOSCAN_LOG_ERROR("Channel pump WaitSet wait failed with {}",
                         error_code_name(waited.error().code));
      return make_unexpected(std::move(waited).error());
    }
    if (!*waited) {
      return {};
    }
    auto ready = wait_set_->ready_handles(ready_handles_);
    if (!ready) {
      HOLOSCAN_LOG_ERROR("Channel pump WaitSet ready-handle collection failed with {}",
                         error_code_name(ready.error().code));
      return make_unexpected(std::move(ready).error());
    }
    for (std::size_t index = 0U; index < *ready; ++index) {
      if (!wait_subscriber_by_handle_.contains(ready_handles_[index])) {
        HOLOSCAN_LOG_ERROR("Channel WaitSet returned unknown readiness handle {}",
                           ready_handles_[index]);
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
    }
    trace_phase(phase_counters_, RuntimePhaseTracePoint::kChannelTransportWaitReturned);
    return {};
  }

  [[nodiscard]] bool enter_pump_support_thread() noexcept {
    if (support_thread_control_ == nullptr) {
      return true;
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      auto entered = support_thread_control_->enter(RuntimeSupportThreadRole::kChannelPump);
      if (!entered) {
        HOLOSCAN_LOG_ERROR("Channel pump thread setup failed with {}: {}",
                           error_code_name(entered.error().code),
                           entered.error().message);
        fail(entered.error().code);
        return false;
      }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      HOLOSCAN_LOG_ERROR("Channel pump thread setup raised an exception");
      fail(ErrorCode::kFailure);
      return false;
    }
#endif
    return true;
  }

  void freeze_descriptor_import_inventory() {
    descriptor_import_present_ = false;
    descriptor_import_devices_.clear();
    descriptor_import_devices_.reserve(subscribers_.size());
    for (const auto& subscriber : subscribers_) {
      if (!subscriber->requires_pump_vmm_import_context()) {
        continue;
      }
      descriptor_import_present_ = true;
      if (subscriber->uses_device_affine_descriptor_import()) {
        descriptor_import_devices_.push_back(subscriber->tensor_device);
      }
    }
    std::ranges::sort(descriptor_import_devices_, {}, &DeviceId::value);
    const auto unique = std::ranges::unique(descriptor_import_devices_);
    descriptor_import_devices_.erase(unique.begin(), unique.end());
  }

  [[nodiscard]] bool initialize_descriptor_import(CudaDeviceCursor& device) noexcept {
    bool initialized = device.active();
    for (const DeviceId import_device : descriptor_import_devices_) {
      if (!initialized) {
        break;
      }
      initialized = device.initialize(import_device);
    }
    if (initialized) {
      initialized = descriptor_import_devices_.empty()
                        ? device.initialize()
                        : device.select(descriptor_import_devices_.front());
    }
    return initialized;
  }

  [[nodiscard]] bool initialize_pump_descriptor_import(
      std::optional<CudaDeviceCursor>& pump_device) noexcept {
    if (!descriptor_import_present_) {
      return true;
    }

    pump_device.emplace();
    const bool initialized = initialize_descriptor_import(*pump_device);
    if (!initialized) {
      HOLOSCAN_LOG_ERROR("Channel pump could not initialize CUDA VMM descriptor import");
      fail(ErrorCode::kNotSupported);
      return false;
    }
    return true;
  }

  [[nodiscard]] bool initialize_controller_descriptor_import() noexcept {
    if (!descriptor_import_present_) {
      return true;
    }
    const std::thread::id current = std::this_thread::get_id();
    if (controller_descriptor_import_initialized_ &&
        controller_descriptor_import_thread_ == current) {
      return true;
    }

    // CUDA context state is thread-local. The controller may take a process-local activation cut
    // on a thread different from the pump, so initialize every compiler-frozen raw-Tensor device
    // once per sequential controller-thread handoff before any descriptor import can occur.
    CudaDeviceCursor controller_device;
    if (!initialize_descriptor_import(controller_device) || !controller_device.restore()) {
      HOLOSCAN_LOG_ERROR("Channel controller could not initialize CUDA VMM descriptor import");
      fail(ErrorCode::kNotSupported);
      return false;
    }
    controller_descriptor_import_thread_ = current;
    controller_descriptor_import_initialized_ = true;
    ++controller_descriptor_import_initialization_count_;
    return true;
  }

  void pump() noexcept {
    if (!enter_pump_support_thread()) {
      pump_startup_.release();
      return;
    }
    // HoloMQ descriptor import uses CUDA VMM even for pageable-host payload storage. CUDA Runtime
    // context state is thread-local, so cold startup attaches the pump to every compiler-frozen
    // raw-Tensor device and leaves the first one selected. Runtime owns that planned/current-device
    // selection; each HoloMQ handle carries the exporter's ordinal so the import layer can verify
    // the physical allocation location. The scan cursor changes devices only at an actual
    // plan-ordered transition and restores the thread's ambient device once.
    std::optional<CudaDeviceCursor> pump_device;
    if (!initialize_pump_descriptor_import(pump_device)) {
      pump_startup_.release();
      return;
    }
    // arm() owns the cold startup barrier. Release it only after support-thread control and every
    // frozen raw-Tensor import device are initialized so an unavailable device cannot race the
    // first activation.
    pump_startup_.release();
    while (!cancel_requested_.load(std::memory_order_acquire)) {
      auto waited = wait_for_transport_activity();
      if (!waited) {
        fail(waited.error().code);
        return;
      }
      bool readiness_published = false;
      {
        // Scan every finite process-local endpoint after each bounded wait, including a timeout.
        // The cut mutex makes the whole scan indivisible with respect to the serial controller's
        // post-compute flush, so the controller never freezes a prefix produced by one activation.
        const std::scoped_lock cut_lock{pump_scan_mutex_};
        PortId failed_port;
        VertexId failed_owner;
        std::optional<StructuralDiagnosticId> terminal_diagnostic;
        auto drained = drain_unmonitored_inputs(failed_port,
                                                failed_owner,
                                                terminal_diagnostic,
                                                pump_device.has_value() ? &*pump_device : nullptr);
        if (!drained) {
          HOLOSCAN_LOG_ERROR("Channel subscriber drain failed with {} on port {}",
                             error_code_name(drained.error().code),
                             failed_port.value());
          fail(drained.error(), failed_owner, terminal_diagnostic);
          return;
        }
        readiness_published = *drained;
      }
      // Coalesce every typed sample retained by this finite scan into one durable controller hint.
      // Exact readiness cardinality remains owned by the revision-gated input binding.
      if (readiness_published && !owner_progress_wake_enabled() &&
          readiness_wake_.signal != nullptr) {
        trace_phase(phase_counters_, RuntimePhaseTracePoint::kChannelReadinessWakeSignaled);
        readiness_wake_.signal(readiness_wake_.state);
      }
    }
  }

  void fail(const Error& error, const VertexId owner = {},
            const std::optional<StructuralDiagnosticId> terminal_diagnostic = {}) noexcept {
    bool expected = false;
    if (terminal_failure_claimed_.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
      terminal_owner_.store(owner.value, std::memory_order_relaxed);
      terminal_diagnostic_.store(
          terminal_diagnostic ? static_cast<std::uint32_t>(*terminal_diagnostic) : 0U,
          std::memory_order_relaxed);
      // Publish the code only after the winner's bounded detail is complete. Readers acquire this
      // store before observing the reason/owner, while losing failures cannot overwrite either.
      terminal_error_.store(error.code, std::memory_order_release);
    }
    closed_.store(true, std::memory_order_release);
    cancel_requested_.store(true, std::memory_order_release);
  }

  void fail(const ErrorCode code) noexcept { fail(Error{code}); }

  [[nodiscard]] expected<void, Error> status() const noexcept {
    const ErrorCode code = terminal_error_.load(std::memory_order_acquire);
    if (code == ErrorCode::kSuccess) {
      return {};
    }
    return make_unexpected(Error{code});
  }

  [[nodiscard]] expected<std::uint64_t, Error> reserve_arrival_order() noexcept {
    // The pump is the sole steady-state writer. Controller-side monitored drains execute only
    // while that pump is joined, so the atomic protects bounded test/control transitions rather
    // than authorizing parallel ingress. A future parallel drain needs prebound/sharded ordering,
    // not additional writers contending on this session-global sequence.
    std::uint64_t current = next_arrival_order_.load(std::memory_order_relaxed);
    for (;;) {
      if (current == std::numeric_limits<std::uint64_t>::max()) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      if (next_arrival_order_.compare_exchange_weak(
              current, current + 1U, std::memory_order_relaxed, std::memory_order_relaxed)) {
        return current;
      }
    }
  }

  template <typename Predicate>
  [[nodiscard]] expected<void, Error> reset_clock_health(const Predicate& affected) noexcept {
    const bool has_affected_health = std::ranges::any_of(subscribers_, [&](const auto& subscriber) {
      return affected(*subscriber) && static_cast<bool>(subscriber->clock_health_endpoint);
    });
    if (!has_affected_health) {
      return {};
    }
    if (clock_health_serialization_mutex_ == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (clock_health_lock_observer_ != nullptr) {
      const auto observer = std::exchange(clock_health_lock_observer_, nullptr);
      void* const state = std::exchange(clock_health_lock_observer_state_, nullptr);
      observer(state);
    }
    // The Runtime joins ingress before reset. Serialize the complete preflight and commit against
    // report dispatch, expiry, and data snapshotting so the receiver binding cannot change after
    // validation and before reset_batch() publishes the next generation.
    const std::lock_guard clock_health_lock{*clock_health_serialization_mutex_};
    clock_health_reset_requests_.clear();
    EndpointClockHealthRuntimeCoordinator* coordinator = nullptr;
    for (const auto& subscriber : subscribers_) {
      if (!affected(*subscriber) || !subscriber->clock_health_endpoint) {
        continue;
      }
      if (subscriber->clock_health_coordinator == nullptr) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      if (coordinator == nullptr) {
        coordinator = subscriber->clock_health_coordinator;
      } else if (coordinator != subscriber->clock_health_coordinator) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      EndpointClockHealthReceiver* const receiver =
          coordinator->receiver(subscriber->clock_health_endpoint);
      if (receiver == nullptr ||
          receiver->state().binding.producer_epoch != subscriber->producer_epoch ||
          subscriber->producer_epoch == std::numeric_limits<std::uint64_t>::max()) {
        return make_unexpected(
            Error{receiver == nullptr ||
                          receiver->state().binding.producer_epoch != subscriber->producer_epoch
                      ? ErrorCode::kProtocolError
                      : ErrorCode::kResourceExhausted});
      }
      EndpointClockHealthBindingIdentity next = receiver->state().binding;
      ++next.producer_epoch;
      clock_health_reset_requests_.push_back(EndpointClockHealthResetRequest{
          .endpoint = subscriber->clock_health_endpoint,
          .origin = EndpointClockHealthUpdateOrigin::kProducerEpochReset,
          .binding = next,
      });
    }
    if (clock_health_reset_requests_.empty()) {
      return {};
    }
    auto reset = coordinator->reset_batch(clock_health_reset_requests_);
    if (!reset) {
      const ErrorCode code = reset.error().code == ErrorCode::kNotReady ? ErrorCode::kBackpressured
                                                                        : reset.error().code;
      return make_unexpected(Error{code});
    }
    return {};
  }

  /// Gate one externally driven import for a generation-fenced rebind.
  ///
  /// Caller holds the subscriber mutex with the pump joined. The gate stays kBackpressured while
  /// pending transport of the current latched incarnation exists or a live publisher is present
  /// without proof that the only observed traffic already belongs to a new incarnation. Advisory
  /// absence alone never fabricates a replacement generation: the cutover is admitted only
  /// because the wire generation fence rejects post-cut old-incarnation frames typed. A
  /// non-restart-capable import keeps failing closed with kNotSupported, and exceeding the
  /// plan-admitted retained-generation capacity fails with kResourceExhausted before any
  /// destructive step (no eviction, wait, or growth).
  [[nodiscard]] expected<void, Error> import_rebind_gate(SubscriberState& subscriber) noexcept {
    auto quiescence =
        channel_internal::CatalogChannelBindingAccess::import_quiescence(*subscriber.endpoint);
    if (!quiescence) {
      return make_unexpected(std::move(quiescence).error());
    }
    const bool live_publisher = subscriber.endpoint->has_publishers();
    if (quiescence->pending_current || (live_publisher && !quiescence->pending_foreign)) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    // Advisory peer absence is not a generation fence. Require one accepted operation (including
    // explicit EOS) to bind the old incarnation token before a replacement subscriber can mark it
    // superseded. Otherwise a racing old publisher could become the first operation of the new
    // attachment and acquire new-generation meaning.
    if (quiescence->latched_producer_generation == 0U || !subscriber.end_of_stream_received) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    if (!subscriber.import_restart_capable || subscriber.import_generation_capacity < 2U) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    if (subscriber.import_generations_used >= subscriber.import_generation_capacity) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    return {};
  }

  /// Atomically cut one gated import over to a replacement catalog attachment.
  ///
  /// Caller holds the subscriber mutex with the pump joined and every preflight (gate, capacity,
  /// epoch exhaustion, clock-health reset) already succeeded. The endpoint services admit one
  /// subscriber, so the old attachment must be detached and dropped before the replacement is
  /// created; any failure after that first destructive step cannot be rolled back and
  /// terminalizes this session, mirroring commit_restart's discard semantics. The replacement
  /// generation slot is consumed only after the cut completes.
  [[nodiscard]] expected<void, Error> rebind_import(SubscriberState& subscriber) noexcept {
    if (!runtime_.has_value()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    ChannelRuntime& runtime = runtime_.value();
    auto endpoint_view = subscriber.rebind_catalog.endpoint(subscriber.rebind_endpoint_name);
    if (!endpoint_view) {
      return make_unexpected(std::move(endpoint_view).error());
    }
    auto superseded = channel_internal::CatalogChannelBindingAccess::latched_producer_generation(
        *subscriber.endpoint);
    if (!superseded) {
      return make_unexpected(std::move(superseded).error());
    }
    auto detached = subscriber.endpoint->detach_wait_set();
    if (!detached) {
      Error error = std::move(detached).error();
      fail(error.code);
      return make_unexpected(std::move(error));
    }
    const std::size_t retired_wait_handles = subscriber.terminal_wait_handle == 0U ? 1U : 2U;
    if (wait_attachment_count_ < retired_wait_handles) {
      fail(ErrorCode::kProtocolError);
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    wait_attachment_count_ -= retired_wait_handles;
    subscriber.endpoint.reset();
    auto replacement = channel_internal::CatalogChannelBindingAccess::create_payload_subscriber(
        runtime, *endpoint_view, subscriber.scratch_size);
    if (!replacement) {
      Error error = std::move(replacement).error();
      fail(error.code);
      return make_unexpected(std::move(error));
    }
    if (*superseded != 0U) {
      auto fenced = channel_internal::CatalogChannelBindingAccess::supersede_producer_generation(
          **replacement, *superseded);
      if (!fenced) {
        Error error = std::move(fenced).error();
        fail(error.code);
        return make_unexpected(std::move(error));
      }
    }
    subscriber.endpoint = std::move(*replacement);
    auto attached = wait_set_->attach(*subscriber.endpoint);
    if (!attached) {
      Error error = std::move(attached).error();
      fail(error.code);
      return make_unexpected(std::move(error));
    }
    const auto terminal_attached = subscriber.secondary_wait_set_handle();
    if (!replace_wait_handles(subscriber, *attached, terminal_attached)) {
      fail(ErrorCode::kProtocolError);
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    subscriber.wait_handle = *attached;
    subscriber.terminal_wait_handle = terminal_attached.value_or(0U);
    wait_attachment_count_ += terminal_attached ? 2U : 1U;
    ++subscriber.import_generations_used;
    return {};
  }

  void cancel() noexcept {
    cancel_requested_.store(true, std::memory_order_release);
    closed_.store(true, std::memory_order_release);
  }

  [[nodiscard]] expected<void, Error> join() noexcept {
    cancel();
    if (pump_thread_.joinable()) {
      if (pump_thread_.get_id() == std::this_thread::get_id()) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      pump_thread_.join();
    }
    armed_.store(false, std::memory_order_release);
    return status();
  }

  std::optional<ChannelRuntime> runtime_;
  PlanGenerationId plan_generation_{};
  std::vector<ClockConversionRecord> clock_conversions_;
  std::shared_ptr<RuntimeSupportThreadControl> support_thread_control_;
  RuntimePhaseCounters* phase_counters_{};
  RuntimeChannelReadinessWake readiness_wake_{};
  std::unique_ptr<channel_internal::InlinePayloadWaitSet> wait_set_;
  std::vector<std::unique_ptr<InputPortState>> inputs_;
  std::vector<InputPortState*> input_by_port_;
  // PortBinding instances are owned by RunSession (or a test caller), not by this session. Track
  // their borrowed back-pointers so session teardown can detach them before InputPortState dies.
  std::vector<ReadinessBinding*> readiness_bindings_;
  std::vector<std::unique_ptr<SubscriberState>> subscribers_;
  std::vector<DeviceId> descriptor_import_devices_;
  std::unordered_map<std::uint64_t, SubscriberState*> wait_subscriber_by_handle_;
  std::vector<SubscriberState*> pump_subscribers_;
  std::vector<std::unique_ptr<OutputState>> outputs_;
  std::vector<ActivationOwnerState> activation_owners_;
  // Null for the overwhelmingly common unmonitored session. Monitored report dispatch, state
  // snapshotting, expiry, and reset share this one externally owned serialization boundary.
  std::mutex* clock_health_serialization_mutex_{};
  std::vector<EndpointClockHealthResetRequest> clock_health_reset_requests_;
  std::vector<std::uint64_t> ready_handles_;
  std::size_t wait_attachment_count_{};
  std::mutex pump_scan_mutex_;
  std::binary_semaphore pump_startup_{0};
  std::thread pump_thread_;
  std::thread::id controller_descriptor_import_thread_{};
  bool controller_descriptor_import_initialized_{};
  bool descriptor_import_present_{};
  std::uint64_t controller_descriptor_import_initialization_count_{};
  std::atomic<bool> armed_{};
  std::atomic<bool> cancel_requested_{};
  std::atomic<bool> closed_{};
  std::atomic<ErrorCode> terminal_error_{ErrorCode::kSuccess};
  std::atomic_bool terminal_failure_claimed_{false};
  std::atomic<std::uint32_t> terminal_diagnostic_{};
  std::atomic<std::uint64_t> terminal_owner_{0U};
  std::atomic<std::uint64_t> next_arrival_order_{1U};
  std::atomic<std::uint64_t> process_local_published_deliveries_{};
  std::atomic<std::uint64_t> process_local_received_deliveries_{};
  std::atomic<bool> process_local_progress_tracking_valid_{true};
  std::atomic<bool> blocked_transport_retry_required_{};
#if !defined(NDEBUG)
  std::shared_mutex controller_mutation_barrier_;
  std::unique_ptr<std::atomic_flag[]> owner_mutation_active_;
#endif
  ErrorCode restart_post_discard_failure_for_test_{ErrorCode::kSuccess};
  void* teardown_observer_state_{};
  RuntimeChannelTeardownObserver teardown_observer_{};
  void* poll_selection_observer_state_{};
  RuntimeChannelPollSelectionObserver poll_selection_observer_{};
  void* clock_health_lock_observer_state_{};
  RuntimeChannelClockHealthLockObserver clock_health_lock_observer_{};
};

namespace {

[[nodiscard]] const RuntimeChannelPortReplayRecord* find_replay(
    std::span<const RuntimeChannelPortReplayRecord> replay, PortId port,
    PortDirection direction) noexcept {
  const auto found = std::find_if(replay.begin(), replay.end(), [&](const auto& candidate) {
    return candidate.port == port && candidate.direction == direction;
  });
  return found == replay.end() ? nullptr : &*found;
}

[[nodiscard]] const ProcessLocalChannelServicePlanRecord* find_service(
    const ExecutionPlanState& plan, std::string_view service_id) noexcept {
  const auto found =
      std::find_if(plan.process_local_channel_services.begin(),
                   plan.process_local_channel_services.end(),
                   [&](const auto& service) { return service.service_id == service_id; });
  return found == plan.process_local_channel_services.end() ? nullptr : &*found;
}

[[nodiscard]] const PortPlanRecord* find_port(const ExecutionPlanState& plan, PortId port,
                                              PortDirection direction) noexcept {
  const auto found = std::find_if(plan.ports.begin(), plan.ports.end(), [&](const auto& candidate) {
    return candidate.id == port && candidate.direction == direction;
  });
  return found == plan.ports.end() ? nullptr : &*found;
}

struct InlineDescriptorTensorContract {
  PortId source{};
  PortId target{};
  std::uint64_t max_byte_span{};
  MemoryKind memory_kind{MemoryKind::kUnknown};
  DeviceId device{};
  std::optional<TensorPortLayout> output_layout;
  std::optional<TensorPortLayout> input_layout;
};

[[nodiscard]] bool descriptor_tensor_identity(const PayloadIdentityRecord& identity) noexcept {
  return identity.kind == PayloadIdentityKind::kSchema &&
         identity.payload_kind == PayloadKind::kDescriptorBacked && !identity.scalar &&
         identity.schema.has_value() && !identity.pod;
}

[[nodiscard]] bool descriptor_tensor_identity_equal(const PayloadIdentityRecord& left,
                                                    const PayloadIdentityRecord& right) noexcept {
  if (!descriptor_tensor_identity(left) || !descriptor_tensor_identity(right) ||
      !left.schema.has_value() || !right.schema.has_value()) {
    return false;
  }
  const auto& left_schema = left.schema.value();
  const auto& right_schema = right.schema.value();
  return left_schema.canonical_name == right_schema.canonical_name &&
         left_schema.version == right_schema.version &&
         left_schema.schema_sha256 == right_schema.schema_sha256 &&
         left_schema.id == right_schema.id;
}

[[nodiscard]] bool tensor_layout_dtype_equal(const std::optional<DLDataType>& left,
                                             const std::optional<DLDataType>& right) noexcept {
  if (left.has_value() != right.has_value()) {
    return false;
  }
  const DLDataType left_dtype = left.value_or(DLDataType{});
  const DLDataType right_dtype = right.value_or(DLDataType{});
  return left_dtype.code == right_dtype.code && left_dtype.bits == right_dtype.bits &&
         left_dtype.lanes == right_dtype.lanes;
}

[[nodiscard]] bool tensor_port_layout_equal(const std::optional<TensorPortLayout>& left,
                                            const std::optional<TensorPortLayout>& right) noexcept {
  if (left.has_value() != right.has_value()) {
    return false;
  }
  if (!left.has_value()) {
    return true;
  }
  const auto& left_layout = left.value();
  const auto& right_layout = right.value();
  return left_layout.memory_kind == right_layout.memory_kind &&
         tensor_layout_dtype_equal(left_layout.dtype, right_layout.dtype) &&
         left_layout.rank == right_layout.rank;
}

[[nodiscard]] bool allocation_satisfies_input_layout(
    const TensorAllocationPlanRecord& allocation,
    const std::optional<TensorPortLayout>& input_layout) noexcept {
  if (!input_layout) {
    return true;
  }
  if (input_layout->memory_kind && *input_layout->memory_kind != allocation.memory_kind) {
    return false;
  }
  if (input_layout->dtype &&
      (!allocation.layout ||
       !tensor_layout_dtype_equal(input_layout->dtype, allocation.layout->dtype))) {
    return false;
  }
  return !input_layout->rank || (allocation.layout && allocation.layout->rank &&
                                 *input_layout->rank == *allocation.layout->rank);
}

// Inline admission validates the complete plan, service, allocation, and layout contract.
[[nodiscard]] expected<std::optional<InlineDescriptorTensorContract>, Error>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
find_inline_descriptor_tensor_contract(
    const ExecutionPlanState& plan, PortId source, std::optional<PortId> target,
    const ProcessLocalChannelServicePlanRecord& service) noexcept {
  if (!plan.inline_partition_execution) {
    return std::optional<InlineDescriptorTensorContract>{};
  }
  const InlinePartitionExecutionPlanRecord& inline_plan = *plan.inline_partition_execution;
  const PayloadEdgePlanRecord* matched_edge = nullptr;
  for (const auto& edge : plan.payload_edges) {
    if (edge.source != source || (target && edge.target != *target) ||
        std::find(inline_plan.edges.begin(), inline_plan.edges.end(), edge.edge) ==
            inline_plan.edges.end()) {
      continue;
    }
    if (matched_edge != nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    matched_edge = &edge;
  }
  if (matched_edge == nullptr || !descriptor_tensor_identity(matched_edge->payload_identity)) {
    return std::optional<InlineDescriptorTensorContract>{};
  }

  const auto* source_port = find_port(plan, matched_edge->source, PortDirection::kOutput);
  const auto* target_port = find_port(plan, matched_edge->target, PortDirection::kInput);
  const TensorAllocationPlanRecord* allocation = nullptr;
  for (const auto& candidate : plan.tensor_allocations) {
    if (candidate.output != matched_edge->source) {
      continue;
    }
    if (allocation != nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    allocation = &candidate;
  }
  std::vector<const TensorFieldAllocationPlanRecord*> field_allocations;
  for (const auto& candidate : plan.tensor_field_allocations) {
    if (candidate.output == matched_edge->source) {
      field_allocations.push_back(&candidate);
    }
  }
  std::sort(field_allocations.begin(),
            field_allocations.end(),
            [](const auto* left, const auto* right) { return left->field_id < right->field_id; });
  const auto publisher_count = std::count_if(plan.process_local_channel_publishers.begin(),
                                             plan.process_local_channel_publishers.end(),
                                             [&](const auto& row) {
                                               return row.source == matched_edge->source &&
                                                      row.service_id == service.service_id;
                                             });
  const auto subscriber_count = std::count_if(plan.process_local_channel_subscribers.begin(),
                                              plan.process_local_channel_subscribers.end(),
                                              [&](const auto& row) {
                                                return row.edge == matched_edge->edge &&
                                                       row.source == matched_edge->source &&
                                                       row.target == matched_edge->target &&
                                                       row.service_id == service.service_id;
                                              });
  const bool supported_direct_allocation =
      allocation != nullptr &&
      validate_tensor_allocation_record(*allocation, plan.plan_generation) &&
      allocation->allocator_process_group == inline_plan.process_group &&
      (allocation->memory_kind == MemoryKind::kHost ||
       allocation->memory_kind == MemoryKind::kCudaDevice);
  DeviceId field_device;
  bool supported_field_allocations = allocation == nullptr && !field_allocations.empty() &&
                                     service.tensor_fields.size() == field_allocations.size();
  for (std::size_t index = 0U; supported_field_allocations && index < field_allocations.size();
       ++index) {
    const auto& field = *field_allocations[index];
    const auto& storage = service.tensor_fields[index];
    const bool layout_matches = field.layout && storage.rank == field.layout->rank &&
                                tensor_layout_dtype_equal(storage.dtype, field.layout->dtype);
    supported_field_allocations =
        validate_tensor_field_allocation_record(field, plan.plan_generation) &&
        field.output == matched_edge->source &&
        field.allocator_process_group == inline_plan.process_group &&
        (field.memory_kind == MemoryKind::kHost || field.memory_kind == MemoryKind::kCudaDevice) &&
        field.tensor_index == index &&
        (index == 0U || field_allocations[index - 1U]->field_id < field.field_id) &&
        storage.field_id == field.field_id && storage.tensor_index == field.tensor_index &&
        storage.max_byte_span == field.bounds.max_byte_span &&
        storage.memory_kind == field.memory_kind && storage.device == field.device &&
        layout_matches &&
        (!field.device.valid() || !field_device.valid() || field.device == field_device);
    if (supported_field_allocations && field.device.valid()) {
      field_device = field.device;
    }
  }
  const bool supported_allocation =
      (supported_direct_allocation && field_allocations.empty() && service.tensor_fields.empty()) ||
      (allocation == nullptr && supported_field_allocations);
  const bool supported_service =
      source_port != nullptr && service.source == matched_edge->source &&
      descriptor_tensor_identity_equal(service.payload_identity, matched_edge->payload_identity) &&
      service.tensor_payload &&
      service.payload_size_bytes == source_port->serialized_payload_capacity_bytes &&
      service.tensor_pool_id.empty() && service.tensor_max_byte_span == 0U &&
      service.tensor_memory_kind == MemoryKind::kUnknown && !service.tensor_device.valid() &&
      service.delivery_kind == ProcessLocalChannelDeliveryKind::kDirectRetained &&
      !service.cross_process_transport && !service.event_wake && service.max_subscribers == 1U;
  if (source_port == nullptr || target_port == nullptr || !supported_allocation ||
      !supported_service || matched_edge->disposition != PlanDisposition::kAccepted ||
      matched_edge->locality != ChannelLocalityKind::kProcessLocal || matched_edge->fanout != 1U ||
      !descriptor_tensor_identity_equal(source_port->payload_identity,
                                        matched_edge->payload_identity) ||
      !descriptor_tensor_identity_equal(target_port->payload_identity,
                                        matched_edge->payload_identity) ||
      source_port->serialized_payload_capacity_bytes == 0U ||
      source_port->serialized_payload_capacity_bytes !=
          target_port->serialized_payload_capacity_bytes ||
      (supported_direct_allocation &&
       !allocation_satisfies_input_layout(*allocation, target_port->tensor_layout)) ||
      publisher_count != 1 || subscriber_count != 1) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  if (supported_field_allocations) {
    return InlineDescriptorTensorContract{
        .source = matched_edge->source,
        .target = matched_edge->target,
        .device = field_device,
    };
  }
  return InlineDescriptorTensorContract{
      .source = matched_edge->source,
      .target = matched_edge->target,
      .max_byte_span = allocation->bounds.max_byte_span,
      .memory_kind = allocation->memory_kind,
      .device = allocation->device,
      .output_layout = allocation->layout,
      .input_layout = target_port->tensor_layout,
  };
}

[[nodiscard]] bool runtime_codec_matches_plan(const RuntimePortTypeOps* ops,
                                              const PortPlanRecord& port,
                                              std::uint64_t service_payload_capacity,
                                              const bool process_local_schema_envelope) noexcept {
  if (ops == nullptr || ops->wire_capacity == 0U ||
      ops->wire_capacity > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  const std::uint64_t expected_service_capacity =
      ops->codec_family == RuntimePayloadCodecFamily::kGeneratedSchema &&
              process_local_schema_envelope
          ? ops->wire_capacity + kSchemaEnvelopeHeaderSize
          : ops->wire_capacity;
  if (expected_service_capacity != service_payload_capacity) {
    return false;
  }
  switch (ops->codec_family) {
    case RuntimePayloadCodecFamily::kInlinePod:
      return ops->inline_pod && ops->payload_kind == PayloadKind::kInlineTyped &&
             ops->encode_payload == nullptr && ops->wire_capacity == ops->payload_size &&
             port.payload_identity.kind == PayloadIdentityKind::kScalar &&
             port.serialized_payload_capacity_bytes == 0U;
    case RuntimePayloadCodecFamily::kCustomPod:
      return ops->inline_pod && ops->payload_kind == PayloadKind::kInlineTyped &&
             ops->encode_payload == nullptr && ops->wire_capacity == ops->payload_size &&
             port.payload_identity.kind == PayloadIdentityKind::kPod &&
             port.payload_identity.pod.has_value() &&
             channel_internal::PodFieldwiseCodec::validate(*port.payload_identity.pod)
                 .has_value() &&
             port.serialized_payload_capacity_bytes == 0U;
    case RuntimePayloadCodecFamily::kGeneratedSchema:
      return !ops->inline_pod && ops->payload_kind == PayloadKind::kSerializedBytes &&
             ops->encode_payload != nullptr &&
             port.payload_identity.kind == PayloadIdentityKind::kSchema &&
             port.serialized_payload_capacity_bytes == ops->wire_capacity;
    case RuntimePayloadCodecFamily::kDescriptorTensorSchema:
    case RuntimePayloadCodecFamily::kUnsupported:
      return false;
  }
  return false;
}

// Runtime materialization conservatively matches every payload-family and Tensor plan invariant.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool runtime_tensor_matches_plan(
    const RuntimePortTypeOps* ops, const PortPlanRecord& port,
    const ProcessLocalChannelServicePlanRecord& service) noexcept {
  if (ops == nullptr || !service.tensor_payload) {
    return false;
  }
  const bool descriptor_schema =
      port.payload_identity.kind == PayloadIdentityKind::kSchema &&
      port.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
  if (descriptor_schema) {
    const std::uint32_t tensor_count = ops->descriptor_tensor_static_count;
    const auto valid_storage = [](const TensorFieldServiceStorageContract& field) noexcept {
      return !field.pool_id.empty() && field.max_byte_span != 0U && field.dtype && field.rank &&
             ((field.memory_kind == MemoryKind::kHost && !field.device.valid()) ||
              ((field.memory_kind == MemoryKind::kPinnedHost ||
                field.memory_kind == MemoryKind::kCudaDevice) &&
               field.device.valid()));
    };
    bool storage_matches{};
    if (!service.tensor_fields.empty()) {
      storage_matches = service.tensor_fields.size() == tensor_count;
      for (std::size_t index = 0U; storage_matches && index < service.tensor_fields.size();
           ++index) {
        storage_matches =
            service.tensor_fields[index].tensor_index == index &&
            (!service.cross_process_transport || valid_storage(service.tensor_fields[index]));
      }
    } else {
      storage_matches =
          tensor_count == 1U &&
          (service.cross_process_transport
               ? !service.tensor_pool_id.empty() && service.tensor_max_byte_span != 0U &&
                     ((service.tensor_memory_kind == MemoryKind::kHost &&
                       !service.tensor_device.valid()) ||
                      ((service.tensor_memory_kind == MemoryKind::kPinnedHost ||
                        service.tensor_memory_kind == MemoryKind::kCudaDevice) &&
                       service.tensor_device.valid()))
               : service.tensor_pool_id.empty() && service.tensor_max_byte_span == 0U &&
                     service.tensor_memory_kind == MemoryKind::kUnknown &&
                     !service.tensor_device.valid());
      if (!service.cross_process_transport && tensor_count > 1U) {
        storage_matches = true;
      }
    }
    return storage_matches && service.payload_size_bytes == ops->wire_capacity &&
           ops->wire_capacity != 0U &&
           ops->wire_capacity == port.serialized_payload_capacity_bytes &&
           ops->codec_family == RuntimePayloadCodecFamily::kDescriptorTensorSchema &&
           ops->payload_kind == PayloadKind::kDescriptorBacked && ops->persistent_sample_storage &&
           tensor_count != 0U && tensor_count <= channel_internal::kMaximumDescriptorTensorFields &&
           ops->descriptor_offset <= ops->wire_capacity &&
           static_cast<std::size_t>(tensor_count) * sizeof(PayloadDescriptor) <=
               ops->wire_capacity - ops->descriptor_offset &&
           ops->descriptor_tensor_count != nullptr && ops->descriptor_tensor_at != nullptr &&
           (tensor_count != 1U || (ops->descriptor_tensor != nullptr &&
                                   ops->assign_descriptor_tensor_sample != nullptr)) &&
           ops->encode_descriptor_tensor_sample != nullptr &&
           ops->descriptor_tensor_sample_metadata != nullptr &&
           ops->descriptor_tensor_view_metadata_at != nullptr &&
           ops->assign_descriptor_tensors_sample != nullptr &&
           ops->prepare_sample_storage != nullptr && ops->reset_sample_storage != nullptr;
  }
  const bool pool_backed = !service.tensor_pool_id.empty();
  const bool supported_pool_placement =
      (service.tensor_memory_kind == MemoryKind::kHost && !service.tensor_device.valid()) ||
      service.tensor_memory_kind == MemoryKind::kPinnedHost ||
      ((service.tensor_memory_kind == MemoryKind::kCudaDevice ||
        service.tensor_memory_kind == MemoryKind::kCudaManaged) &&
       service.tensor_device.valid());
  const bool pool_contract = pool_backed
                                 ? service.tensor_max_byte_span != 0U && supported_pool_placement
                                 : service.tensor_max_byte_span == 0U &&
                                       service.tensor_memory_kind == MemoryKind::kUnknown &&
                                       !service.tensor_device.valid();
  return service.payload_size_bytes == 0U && pool_contract &&
         ops == runtime_port_type_ops<Tensor>() &&
         ops->codec_family == RuntimePayloadCodecFamily::kUnsupported &&
         ops->sample_size == sizeof(Sample<Tensor>) &&
         ops->sample_alignment == alignof(Sample<Tensor>) &&
         port.payload_identity.kind == PayloadIdentityKind::kTensor &&
         port.serialized_payload_capacity_bytes == 0U;
}

[[nodiscard]] bool runtime_external_tensor_base_matches_plan(
    const RuntimePortTypeOps* ops, const PortPlanRecord& port,
    const ChannelServiceStaticPlanRecord& service, const ChannelEndpointRecord& endpoint) noexcept {
  const bool supported_memory_kind = endpoint.tensor && endpoint.tensor->memory_kind &&
                                     (*endpoint.tensor->memory_kind == MemoryKind::kHost ||
                                      *endpoint.tensor->memory_kind == MemoryKind::kPinnedHost ||
                                      *endpoint.tensor->memory_kind == MemoryKind::kCudaDevice);
  return service.tensor_payload && service.payload_identity.kind == PayloadIdentityKind::kTensor &&
         service.payload_identity.payload_kind == PayloadKind::kDescriptorBacked &&
         supported_memory_kind && endpoint.service_static.tensor &&
         endpoint.tensor->max_byte_span && *endpoint.tensor->max_byte_span != 0U &&
         logical_tensor_requirements_equal(*endpoint.tensor, *endpoint.service_static.tensor) &&
         endpoint.service_static.tensor->max_byte_span == endpoint.tensor->max_byte_span &&
         endpoint.service_static.tensor->strides == endpoint.tensor->strides &&
         ops == runtime_port_type_ops<Tensor>() &&
         ops->codec_family == RuntimePayloadCodecFamily::kUnsupported &&
         ops->sample_size == sizeof(Sample<Tensor>) &&
         ops->sample_alignment == alignof(Sample<Tensor>) &&
         port.payload_identity.kind == PayloadIdentityKind::kTensor &&
         port.payload_identity.payload_kind == PayloadKind::kDescriptorBacked &&
         port.serialized_payload_capacity_bytes == 0U;
}

[[nodiscard]] bool runtime_external_tensor_matches_plan(
    const RuntimePortTypeOps* ops, const PortPlanRecord& port,
    const ChannelServiceStaticPlanRecord& service, const ChannelEndpointRecord& endpoint) noexcept {
  if (!runtime_external_tensor_base_matches_plan(ops, port, service, endpoint) ||
      !endpoint.tensor) {
    return false;
  }
  return endpoint.tensor->strides == TensorStridePolicy::kContiguousOnly;
}

[[nodiscard]] bool runtime_external_tensor_uses_unsupported_strides(
    const RuntimePortTypeOps* ops, const PortPlanRecord& port,
    const ChannelServiceStaticPlanRecord& service, const ChannelEndpointRecord& endpoint) noexcept {
  if (!runtime_external_tensor_base_matches_plan(ops, port, service, endpoint) ||
      !endpoint.tensor) {
    return false;
  }
  return endpoint.tensor->strides == TensorStridePolicy::kBoundedStrided;
}

[[nodiscard]] bool tensor_snapshot_matches_plan(const TensorAccessSnapshot& snapshot,
                                                const std::uint64_t max_byte_span,
                                                const MemoryKind memory_kind,
                                                const DeviceId device) noexcept {
  if (!snapshot.initialized || !snapshot.valid) {
    return false;
  }
  if (memory_kind == MemoryKind::kUnknown) {
    return max_byte_span == 0U && !device.valid();
  }
  if (max_byte_span == 0U || snapshot.byte_span > max_byte_span) {
    return false;
  }
  switch (memory_kind) {
    case MemoryKind::kHost:
      return !device.valid() && snapshot.device.device_type == kDLCPU &&
             snapshot.device.device_id == 0;
    case MemoryKind::kPinnedHost:
      // DLPack fixes pinned-host device_id at zero. A valid private plan device may still select
      // the HoloMQ generation-sync context; it is not encoded in the public Tensor metadata.
      return snapshot.device.device_type == kDLCUDAHost && snapshot.device.device_id == 0;
    case MemoryKind::kCudaDevice:
      return device.valid() && snapshot.device.device_type == kDLCUDA &&
             snapshot.device.device_id == device.value;
    case MemoryKind::kCudaManaged:
      // DLPack uses device id zero for managed allocations. Compile still requires an explicit
      // source-private DeviceId so Runtime never infers device zero from that convention.
      return device.valid() && snapshot.device.device_type == kDLCUDAManaged &&
             snapshot.device.device_id == 0;
    case MemoryKind::kUnknown:
    default:
      return false;
  }
}

[[nodiscard]] const ProcessLocalChannelPublisherPlanRecord* find_publisher(
    const ExecutionPlanState& plan, std::string_view service_id) noexcept {
  const auto found =
      std::find_if(plan.process_local_channel_publishers.begin(),
                   plan.process_local_channel_publishers.end(),
                   [&](const auto& publisher) { return publisher.service_id == service_id; });
  return found == plan.process_local_channel_publishers.end() ? nullptr : &*found;
}

[[nodiscard]] const InputSelectionPlanRecord* find_selection(const ExecutionPlanState& plan,
                                                             PortId input) noexcept {
  const auto found = std::find_if(plan.input_selections.begin(),
                                  plan.input_selections.end(),
                                  [&](const auto& selection) { return selection.input == input; });
  return found == plan.input_selections.end() ? nullptr : &*found;
}

[[nodiscard]] const ClockConversionRecord* find_conversion(const ExecutionPlanState& plan,
                                                           std::uint64_t id) noexcept {
  const auto found = std::find_if(plan.clock_conversions.begin(),
                                  plan.clock_conversions.end(),
                                  [&](const auto& conversion) { return conversion.id == id; });
  return found == plan.clock_conversions.end() ? nullptr : &*found;
}

[[nodiscard]] const ChannelServiceStaticPlanRecord* find_external_service(
    const ExecutionPlanState& plan, ExternalTopicId topic) noexcept {
  const auto found = std::find_if(plan.channel_services.begin(),
                                  plan.channel_services.end(),
                                  [&](const auto& service) { return service.topic == topic; });
  return found == plan.channel_services.end() ? nullptr : &*found;
}

[[nodiscard]] const ChannelPublisherEndpointPlanRecord* find_external_publisher(
    const ExecutionPlanState& plan, ExternalTopicId topic) noexcept {
  const auto found = std::find_if(plan.channel_publishers.begin(),
                                  plan.channel_publishers.end(),
                                  [&](const auto& publisher) { return publisher.topic == topic; });
  return found == plan.channel_publishers.end() ? nullptr : &*found;
}

[[nodiscard]] const ChannelSubscriberEndpointPlanRecord* find_external_subscriber(
    const ExecutionPlanState& plan, ExternalTopicId topic) noexcept {
  const auto found =
      std::find_if(plan.channel_subscribers.begin(),
                   plan.channel_subscribers.end(),
                   [&](const auto& subscriber) { return subscriber.topic == topic; });
  return found == plan.channel_subscribers.end() ? nullptr : &*found;
}

[[nodiscard]] const TensorInterfacePoolPlanRecord* find_tensor_interface_pool(
    const ExecutionPlanState& plan, const ExternalTopicId topic) noexcept {
  const auto found = std::find_if(plan.tensor_interface_pools.begin(),
                                  plan.tensor_interface_pools.end(),
                                  [&](const auto& pool) { return pool.topic == topic; });
  return found == plan.tensor_interface_pools.end() ? nullptr : &*found;
}

[[nodiscard]] const EndpointClockHealthPlanRecord* find_endpoint_clock_health(
    const ExecutionPlanState& plan, PortId endpoint) noexcept {
  const auto found = std::find_if(plan.endpoint_clock_health.begin(),
                                  plan.endpoint_clock_health.end(),
                                  [&](const auto& record) { return record.endpoint == endpoint; });
  return found == plan.endpoint_clock_health.end() ? nullptr : &*found;
}

[[nodiscard]] expected<ChannelEndpointCatalog::EndpointView, Error> find_catalog_endpoint(
    const ExecutionPlanState& plan, PortId boundary) noexcept {
  const auto* state = ChannelEndpointCatalogAccess::state(plan.endpoint_catalog);
  if (state == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const auto found =
      std::find_if(state->endpoints.begin(), state->endpoints.end(), [&](const auto& endpoint) {
        return endpoint.boundary_port == boundary;
      });
  if (found == state->endpoints.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  return plan.endpoint_catalog.endpoint(found->name);
}

struct CatalogCaptureClockBinding {
  ClockId catalog{};
  ClockId plan{};
};

[[nodiscard]] expected<CatalogCaptureClockBinding, Error> bind_catalog_capture_clock(
    const ExecutionPlanState& plan, ChannelEndpointCatalog::EndpointView endpoint,
    ClockId plan_capture_clock) noexcept {
  const ClockId catalog_capture_clock = ChannelEndpointCatalogAccess::capture_clock_id(endpoint);
  if (catalog_capture_clock.valid() != plan_capture_clock.valid()) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  if (!catalog_capture_clock.valid()) {
    return CatalogCaptureClockBinding{};
  }

  const auto* catalog = ChannelEndpointCatalogAccess::state(plan.endpoint_catalog);
  if (catalog == nullptr) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  const auto catalog_clock =
      std::find_if(catalog->clocks.begin(), catalog->clocks.end(), [&](const auto& candidate) {
        return candidate.id == catalog_capture_clock.value;
      });
  const auto plan_clock =
      std::find_if(plan.clocks.begin(), plan.clocks.end(), [&](const auto& candidate) {
        return candidate.id == plan_capture_clock;
      });
  if (catalog_clock == catalog->clocks.end() || plan_clock == plan.clocks.end() ||
      catalog_clock->key != plan_clock->key) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  return CatalogCaptureClockBinding{
      .catalog = catalog_capture_clock,
      .plan = plan_capture_clock,
  };
}

// Resolution follows the frozen boundary/interface chain and rejects ambiguity at every hop.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<PortId, Error> external_internal_port(const ExecutionPlanState& plan,
                                                             PortId boundary,
                                                             bool imported) noexcept {
  const PortDirection internal_direction =
      imported ? PortDirection::kInput : PortDirection::kOutput;
  const auto resolve_interface = [&](PortId endpoint) -> expected<PortId, Error> {
    for (std::size_t depth = 0U; depth <= plan.interface_port_bindings.size(); ++depth) {
      const PortPlanRecord* const port = find_port(plan, endpoint, internal_direction);
      if (port == nullptr) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      if (port->owner_kind == PortOwnerKind::kOperator) {
        return endpoint;
      }
      if (port->owner_kind != PortOwnerKind::kGraphInterface) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      const auto binding =
          std::find_if(plan.interface_port_bindings.begin(),
                       plan.interface_port_bindings.end(),
                       [&](const auto& candidate) { return candidate.public_port == endpoint; });
      if (binding == plan.interface_port_bindings.end() ||
          binding->direction != internal_direction) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      endpoint = binding->internal_port;
    }
    return make_unexpected(Error{ErrorCode::kProtocolError});
  };

  PortId result{};
  for (const auto& edge : plan.payload_edges) {
    const bool matches = imported ? edge.source == boundary : edge.target == boundary;
    if (!matches) {
      continue;
    }
    const PortId candidate = imported ? edge.target : edge.source;
    if (result && result != candidate) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    result = candidate;
  }
  if (result) {
    return resolve_interface(result);
  }
  const auto binding =
      std::find_if(plan.interface_port_bindings.begin(),
                   plan.interface_port_bindings.end(),
                   [&](const auto& candidate) { return candidate.public_port == boundary; });
  if (binding != plan.interface_port_bindings.end()) {
    return resolve_interface(binding->internal_port);
  }
  return make_unexpected(Error{ErrorCode::kNotFound});
}

[[nodiscard]] expected<ChannelOptions, Error> service_options(
    const ProcessLocalChannelServicePlanRecord& service) noexcept {
  if (!service.source || service.service_id.empty() ||
      (!service.tensor_payload && service.payload_size_bytes == 0U) ||
      service.max_subscribers == 0U || service.max_subscriber_queue_depth == 0U ||
      service.max_subscriber_borrowed_samples == 0U || service.publisher_history_depth != 0U ||
      !service.event_wake || !supported_queue_policy(service.queue_policy)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  ChannelOptions options{};
  options.queue_policy = channel_queue_policy(service.queue_policy);
  options.max_subscriber_queue_depth = service.max_subscriber_queue_depth;
  options.subscriber_queue_depth = service.max_subscriber_queue_depth;
  options.max_subscriber_borrowed_samples = service.max_subscriber_borrowed_samples;
  options.max_borrowed_samples = service.max_subscriber_borrowed_samples;
  options.publisher_history_depth = service.publisher_history_depth;
  options.expected_recipient_count = service.max_subscribers;
  options.enable_event_wake = service.event_wake;
  options.tensor_transfer_mode = service.cross_process_transport
                                     ? ChannelTensorTransferMode::kBufferHandle
                                     : ChannelTensorTransferMode::kSameProcessSidecar;
  return options;
}

[[nodiscard]] expected<ChannelOptions, Error> external_service_options(
    const ChannelServiceStaticPlanRecord& service,
    const ChannelPublisherEndpointPlanRecord& publisher,
    const ChannelSubscriberEndpointPlanRecord& subscriber) noexcept {
  if (!service.topic || service.service_id.empty() || service.max_subscriber_queue_depth == 0U ||
      service.max_subscriber_borrowed_samples == 0U || service.publisher_history_depth != 0U ||
      !service.event_wake || publisher.topic != service.topic ||
      subscriber.topic != service.topic || publisher.service_id != service.service_id ||
      subscriber.service_id != service.service_id || publisher.max_loaned_samples == 0U ||
      subscriber.requested_queue_depth == 0U || subscriber.max_borrowed_samples == 0U ||
      !subscriber.event_wake || !supported_queue_policy(service.queue_policy)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  ChannelOptions options{};
  options.queue_policy = channel_queue_policy(service.queue_policy);
  options.max_subscriber_queue_depth = service.max_subscriber_queue_depth;
  options.subscriber_queue_depth = subscriber.requested_queue_depth;
  options.max_subscriber_borrowed_samples = service.max_subscriber_borrowed_samples;
  options.max_borrowed_samples = subscriber.max_borrowed_samples;
  options.publisher_history_depth = service.publisher_history_depth;
  options.max_loaned_samples = publisher.max_loaned_samples;
  options.expected_recipient_count = 1U;
  options.enable_event_wake = service.event_wake;
  options.replacement_attribution_required = publisher.replacement_attribution_required;
  options.tensor_transfer_mode = ChannelTensorTransferMode::kBufferHandle;
  return options;
}

}  // namespace

RuntimeChannelSession::RuntimeChannelSession(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

RuntimeChannelSession::~RuntimeChannelSession() = default;

// Reconstruction validates and binds every compiler-frozen service, endpoint, replay record,
// capacity, and readiness edge as one fail-closed transaction before publishing the session.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<std::unique_ptr<RuntimeChannelSession>, Error> RuntimeChannelSession::create(
    const ExecutionPlanState& plan,
    const std::span<const RuntimeChannelPortReplayRecord> replay_ports,
    ProcessGroupId process_group, RuntimeChannelReadinessWake readiness_wake) {
  if (!plan.admitted || plan.plan_generation == 0U ||
      ((readiness_wake.state == nullptr) != (readiness_wake.signal == nullptr)) ||
      (readiness_wake.signal_owner_progress != nullptr && readiness_wake.state == nullptr) ||
      plan.process_local_channel_publishers.size() != plan.process_local_channel_services.size() ||
      plan.cross_process_channel_publishers.size() != plan.cross_process_channel_services.size()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (!valid_process_local_channel_plan(plan) || !valid_external_channel_plan(plan) ||
      !valid_cross_process_channel_plan(plan)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto generation_capacities_valid =
      validate_import_generation_capacities(plan.imported_generation_capacities);
  if (!generation_capacities_valid) {
    return make_unexpected(std::move(generation_capacities_valid).error());
  }

  auto runtime = ChannelRuntime::create();
  if (!runtime) {
    return make_unexpected(std::move(runtime).error());
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    auto impl = std::make_unique<Impl>(std::move(*runtime),
                                       plan.plan_generation,
                                       plan.clock_conversions,
                                       plan.executable_factories.support_thread_control,
                                       plan.executable_factories.phase_counters.get(),
                                       readiness_wake);
    auto wait_set = RuntimeChannelBindingAccess::create_wait_set(*impl->runtime_);
    if (!wait_set) {
      return make_unexpected(std::move(wait_set).error());
    }
    impl->wait_set_ = std::move(*wait_set);

    // Compatible fanout arms share one plan-frozen cross-process service while
    // retaining one subscriber record per edge. Reuse the process-local
    // construction path after selecting records owned by this worker.
    std::vector<ProcessLocalChannelServicePlanRecord> cross_process_services;
    std::vector<ProcessLocalChannelPublisherPlanRecord> cross_process_publishers;
    std::vector<ProcessLocalChannelSubscriberPlanRecord> cross_process_subscribers;
    cross_process_services.reserve(plan.cross_process_channel_services.size());
    cross_process_publishers.reserve(plan.cross_process_channel_publishers.size());
    cross_process_subscribers.reserve(plan.cross_process_channel_subscribers.size());
    const auto service_reaches_process_group =
        [&](const CrossProcessChannelServicePlanRecord& service) noexcept {
          return !process_group || service.source_process_group == process_group ||
                 std::find(service.target_process_groups.begin(),
                           service.target_process_groups.end(),
                           process_group) != service.target_process_groups.end();
        };
    for (const auto& service : plan.cross_process_channel_services) {
      if (!service_reaches_process_group(service)) {
        continue;
      }
      cross_process_services.push_back(
          make_process_local_service_record(service.source,
                                            service.service_id,
                                            make_tensor_service_contract(service),
                                            service.max_subscribers,
                                            true));
    }
    for (const auto& publisher : plan.cross_process_channel_publishers) {
      const auto service = std::find_if(
          plan.cross_process_channel_services.begin(),
          plan.cross_process_channel_services.end(),
          [&](const auto& candidate) { return candidate.service_id == publisher.service_id; });
      if (service == plan.cross_process_channel_services.end() ||
          !service_reaches_process_group(*service)) {
        continue;
      }
      cross_process_publishers.push_back(ProcessLocalChannelPublisherPlanRecord{
          .source = publisher.source,
          .service_id = publisher.service_id,
          .max_loaned_samples = publisher.max_loaned_samples,
          .replacement_attribution_required = publisher.replacement_attribution_required,
      });
    }
    for (const auto& subscriber : plan.cross_process_channel_subscribers) {
      const auto service = std::find_if(
          plan.cross_process_channel_services.begin(),
          plan.cross_process_channel_services.end(),
          [&](const auto& candidate) { return candidate.service_id == subscriber.service_id; });
      const auto* target = find_port(plan, subscriber.target, PortDirection::kInput);
      if (service == plan.cross_process_channel_services.end() || target == nullptr ||
          (process_group && target->owner_kind == PortOwnerKind::kOperator &&
           !worker_owns_vertex(plan, process_group, target->owner))) {
        continue;
      }
      cross_process_subscribers.push_back(ProcessLocalChannelSubscriberPlanRecord{
          .edge = subscriber.edge,
          .source = subscriber.source,
          .target = subscriber.target,
          .service_id = subscriber.service_id,
          .requested_queue_depth = subscriber.requested_queue_depth,
          .max_borrowed_samples = subscriber.max_borrowed_samples,
          .cache_capacity = subscriber.cache_capacity,
          .readiness_capacity = subscriber.readiness_capacity,
          .event_wake = subscriber.event_wake,
      });
    }
    const auto locate_service =
        [&](std::string_view service_id) noexcept -> const ProcessLocalChannelServicePlanRecord* {
      if (const auto* found = find_service(plan, service_id); found != nullptr) {
        return found;
      }
      const auto cross_found = std::find_if(
          cross_process_services.begin(), cross_process_services.end(), [&](const auto& candidate) {
            return candidate.service_id == service_id;
          });
      return cross_found == cross_process_services.end() ? nullptr : &*cross_found;
    };
    const auto locate_publisher =
        [&](std::string_view service_id) noexcept -> const ProcessLocalChannelPublisherPlanRecord* {
      if (const auto* found = find_publisher(plan, service_id); found != nullptr) {
        return found;
      }
      const auto cross_found =
          std::find_if(cross_process_publishers.begin(),
                       cross_process_publishers.end(),
                       [&](const auto& candidate) { return candidate.service_id == service_id; });
      return cross_found == cross_process_publishers.end() ? nullptr : &*cross_found;
    };

    const std::array<const std::vector<ProcessLocalChannelPublisherPlanRecord>*, 2U>
        publisher_families{&plan.process_local_channel_publishers, &cross_process_publishers};
    for (const auto* publisher_records : publisher_families) {
      for (const auto& publisher_plan : *publisher_records) {
        const auto* service = locate_service(publisher_plan.service_id);
        const auto* replay =
            find_replay(replay_ports, publisher_plan.source, PortDirection::kOutput);
        const auto* port_plan = find_port(plan, publisher_plan.source, PortDirection::kOutput);
        if (service == nullptr || port_plan == nullptr) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        if (process_group && port_plan->owner_kind == PortOwnerKind::kOperator &&
            !worker_owns_vertex(plan, process_group, port_plan->owner)) {
          continue;
        }
        if (replay == nullptr && port_plan->owner_kind != PortOwnerKind::kOperator) {
          continue;
        }
        const bool tensor_service = service->tensor_payload;
        auto inline_descriptor_contract = find_inline_descriptor_tensor_contract(
            plan, publisher_plan.source, std::nullopt, *service);
        if (!inline_descriptor_contract) {
          return make_unexpected(std::move(inline_descriptor_contract).error());
        }
        const std::uint64_t effective_tensor_max_byte_span =
            *inline_descriptor_contract ? (*inline_descriptor_contract)->max_byte_span
                                        : service->tensor_max_byte_span;
        const MemoryKind effective_tensor_memory_kind =
            *inline_descriptor_contract ? (*inline_descriptor_contract)->memory_kind
                                        : service->tensor_memory_kind;
        const DeviceId effective_tensor_device = *inline_descriptor_contract
                                                     ? (*inline_descriptor_contract)->device
                                                     : service->tensor_device;
        const std::optional<TensorPortLayout> effective_tensor_layout =
            *inline_descriptor_contract ? (*inline_descriptor_contract)->output_layout
                                        : std::optional<TensorPortLayout>{};
        const bool runtime_type_matches =
            tensor_service
                ? runtime_tensor_matches_plan(
                      replay == nullptr ? nullptr : replay->type_ops, *port_plan, *service)
                : runtime_codec_matches_plan(replay == nullptr ? nullptr : replay->type_ops,
                                             *port_plan,
                                             service->payload_size_bytes,
                                             !service->cross_process_transport);
        if (replay == nullptr || !replay->owner || replay->declaration == nullptr ||
            !runtime_type_matches || replay->type_ops->sample_payload == nullptr ||
            replay->type_ops->sample_metadata == nullptr ||
            replay->type_ops->bind_output == nullptr ||
            replay->type_ops->unbind_output == nullptr || publisher_plan.max_loaned_samples == 0U ||
            port_plan->owner != replay->owner || port_plan->max_emits_per_compute == 0U) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        const bool direct_retained =
            service->delivery_kind == ProcessLocalChannelDeliveryKind::kDirectRetained;
        ChannelOptions options;
        if (!direct_retained) {
          auto resolved_options = service_options(*service);
          if (!resolved_options) {
            return make_unexpected(std::move(resolved_options).error());
          }
          options = *resolved_options;
          options.max_loaned_samples = publisher_plan.max_loaned_samples;
          options.replacement_attribution_required =
              publisher_plan.replacement_attribution_required;
        }
        const auto existing =
            std::find_if(impl->outputs_.begin(), impl->outputs_.end(), [&](const auto& candidate) {
              return candidate->port == replay->port;
            });
        Impl::OutputState* output = nullptr;
        if (existing == impl->outputs_.end()) {
          auto created = std::make_unique<Impl::OutputState>();
          created->port = replay->port;
          created->owner = replay->owner;
          created->session = impl.get();
          created->declaration = replay->declaration;
          created->type_ops = replay->type_ops;
          created->phase_counters = impl->phase_counters_;
          created->max_emits_per_activation = port_plan->max_emits_per_compute;
          if (tensor_service) {
            created->tensor_max_byte_span = effective_tensor_max_byte_span;
            created->tensor_memory_kind = effective_tensor_memory_kind;
            created->tensor_device = effective_tensor_device;
            created->tensor_layout = effective_tensor_layout;
            created->tensor_fields = service->tensor_fields;
            if (!*inline_descriptor_contract) {
              auto field_device =
                  uniform_tensor_field_device(created->tensor_device, created->tensor_fields);
              if (!field_device) {
                return make_unexpected(std::move(field_device).error());
              }
              created->tensor_device = *field_device;
            }
            if (replay->type_ops->codec_family ==
                RuntimePayloadCodecFamily::kDescriptorTensorSchema) {
              if (!port_plan->payload_identity.schema) {
                return make_unexpected(Error{ErrorCode::kProtocolError});
              }
              created->tensor_schema_id = port_plan->payload_identity.schema->id;
              created->tensor_schema_scratch.resize(
                  replay->type_ops->descriptor_tensor_static_count);
            }
          }
          if (replay->type_ops->codec_family == RuntimePayloadCodecFamily::kCustomPod) {
            if (!port_plan->payload_identity.pod) {
              return make_unexpected(Error{ErrorCode::kProtocolError});
            }
            created->pod_identity = port_plan->payload_identity.pod;
          } else if (replay->type_ops->codec_family ==
                         RuntimePayloadCodecFamily::kGeneratedSchema &&
                     !service->cross_process_transport) {
            if (!port_plan->payload_identity.schema) {
              return make_unexpected(Error{ErrorCode::kProtocolError});
            }
            created->process_local_schema_identity = port_plan->payload_identity.schema;
          }
          output = created.get();
          impl->outputs_.push_back(std::move(created));
        } else {
          output = existing->get();
          if (output->owner != replay->owner || output->declaration != replay->declaration ||
              output->type_ops != replay->type_ops ||
              output->max_emits_per_activation != port_plan->max_emits_per_compute ||
              (tensor_service &&
               (output->tensor_max_byte_span != effective_tensor_max_byte_span ||
                output->tensor_memory_kind != effective_tensor_memory_kind ||
                output->tensor_fields != service->tensor_fields ||
                (service->tensor_fields.empty() &&
                 output->tensor_device != effective_tensor_device) ||
                !tensor_port_layout_equal(output->tensor_layout, effective_tensor_layout)))) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
        }
        if (direct_retained) {
          if (service->cross_process_transport || service->event_wake ||
              output->direct_subscriber != nullptr ||
              (tensor_service && !output->tensor_payload() && !output->tensor_schema_payload())) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
        } else if (tensor_service && replay->type_ops->codec_family ==
                                         RuntimePayloadCodecFamily::kDescriptorTensorSchema) {
          std::vector<channel_internal::TensorSchemaStagingOptions> staging;
          if (service->cross_process_transport) {
            const std::uint64_t staging_slots =
                static_cast<std::uint64_t>(options.max_loaned_samples) +
                static_cast<std::uint64_t>(options.max_subscriber_queue_depth) +
                static_cast<std::uint64_t>(options.max_subscriber_borrowed_samples);
            if (staging_slots == 0U || staging_slots > std::numeric_limits<std::uint32_t>::max()) {
              return make_unexpected(Error{ErrorCode::kResourceExhausted});
            }
            if (!service->tensor_fields.empty()) {
              staging.reserve(service->tensor_fields.size());
              for (const auto& field : service->tensor_fields) {
                staging.push_back(channel_internal::TensorSchemaStagingOptions{
                    .max_byte_span = field.max_byte_span,
                    .slot_count = static_cast<std::uint32_t>(staging_slots),
                    .dtype = field.dtype,
                    .rank = field.rank,
                    .memory_kind = field.memory_kind,
                    .device = field.device,
                    .foreign_emit_policy = field.tensor_emit_policy,
                });
              }
            } else {
              if (!port_plan->tensor_layout || !port_plan->tensor_layout->dtype ||
                  !port_plan->tensor_layout->rank || service->tensor_max_byte_span == 0U ||
                  (service->tensor_memory_kind != MemoryKind::kHost &&
                   service->tensor_memory_kind != MemoryKind::kPinnedHost &&
                   service->tensor_memory_kind != MemoryKind::kCudaDevice)) {
                return make_unexpected(Error{ErrorCode::kProtocolError});
              }
              staging.push_back(channel_internal::TensorSchemaStagingOptions{
                  .max_byte_span = service->tensor_max_byte_span,
                  .slot_count = static_cast<std::uint32_t>(staging_slots),
                  .dtype = port_plan->tensor_layout->dtype,
                  .rank = port_plan->tensor_layout->rank,
                  .memory_kind = service->tensor_memory_kind,
                  .device = service->tensor_device,
                  .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
              });
            }
          }
          std::vector<std::size_t> descriptor_offsets;
          descriptor_offsets.reserve(replay->type_ops->descriptor_tensor_static_count);
          for (std::uint32_t index = 0U; index < replay->type_ops->descriptor_tensor_static_count;
               ++index) {
            descriptor_offsets.push_back(replay->type_ops->descriptor_offset +
                                         static_cast<std::size_t>(index) *
                                             sizeof(PayloadDescriptor));
          }
          auto endpoint = RuntimeChannelBindingAccess::create_tensor_schema_publisher(
              *impl->runtime_,
              service->service_id,
              replay->type_ops->wire_capacity,
              descriptor_offsets,
              options,
              staging);
          if (!endpoint) {
            return make_unexpected(std::move(endpoint).error());
          }
          output->tensor_schema_endpoints.push_back(Impl::OutputState::TensorSchemaEndpointState{
              .publisher = std::move(*endpoint),
              .expected_recipient_count = options.expected_recipient_count,
          });
        } else if (tensor_service) {
          auto endpoint = RuntimeChannelBindingAccess::create_tensor_publisher(
              *impl->runtime_, service->service_id, options);
          if (!endpoint) {
            return make_unexpected(std::move(endpoint).error());
          }
          output->tensor_endpoints.push_back(std::move(*endpoint));
        } else {
          auto endpoint = RuntimeChannelBindingAccess::create_publisher(
              *impl->runtime_, service->service_id, service->payload_size_bytes, options);
          if (!endpoint) {
            return make_unexpected(std::move(endpoint).error());
          }
          output->endpoints.push_back(Impl::OutputState::EndpointState{
              .publisher = std::move(*endpoint),
              .expected_recipient_count = options.expected_recipient_count,
              .process_local_transport = true,
              .process_local_schema_envelope =
                  replay->type_ops->codec_family == RuntimePayloadCodecFamily::kGeneratedSchema &&
                  !service->cross_process_transport,
          });
        }
      }
    }

    struct PendingCatalogTensorPublisher {
      Impl::OutputState* output{};
      ChannelEndpointCatalog::EndpointView endpoint;
      std::uint32_t max_loaned_samples{};
      std::optional<DeviceId> planned_device;
      ClockId plan_capture_clock{};
      ClockId catalog_capture_clock{};
      bool peer_required{true};
    };
    std::vector<PendingCatalogTensorPublisher> pending_catalog_tensor_publishers;
    pending_catalog_tensor_publishers.reserve(plan.graph.external_topics().size());

    for (const auto& topic : plan.graph.external_topics()) {
      if (topic.imported) {
        continue;
      }
      auto internal_port = external_internal_port(plan, PortId{topic.graph_port.value}, false);
      if (!internal_port) {
        if (internal_port.error().code == ErrorCode::kNotFound) {
          continue;
        }
        return make_unexpected(std::move(internal_port).error());
      }
      const auto* replay = find_replay(replay_ports, *internal_port, PortDirection::kOutput);
      const auto* port_plan = find_port(plan, *internal_port, PortDirection::kOutput);
      const auto* service = find_external_service(plan, topic.id);
      const auto* publisher = find_external_publisher(plan, topic.id);
      const auto* subscriber = find_external_subscriber(plan, topic.id);
      if (process_group && port_plan != nullptr &&
          port_plan->owner_kind == PortOwnerKind::kOperator &&
          !worker_owns_vertex(plan, process_group, port_plan->owner)) {
        continue;
      }
      const bool tensor_service = service != nullptr && service->tensor_payload;
      const bool runtime_type_matches =
          tensor_service
              ? replay != nullptr && replay->type_ops == runtime_port_type_ops<Tensor>() &&
                    port_plan != nullptr &&
                    port_plan->payload_identity.kind == PayloadIdentityKind::kTensor
              : replay != nullptr && port_plan != nullptr &&
                    runtime_codec_matches_plan(
                        replay->type_ops,
                        *port_plan,
                        replay->type_ops == nullptr ? 0U : replay->type_ops->wire_capacity,
                        false);
      if (replay == nullptr || port_plan == nullptr || service == nullptr || publisher == nullptr ||
          subscriber == nullptr || replay->owner != port_plan->owner ||
          replay->declaration == nullptr || !runtime_type_matches ||
          (!tensor_service && (replay->type_ops->sample_payload == nullptr ||
                               replay->type_ops->sample_metadata == nullptr)) ||
          replay->type_ops->bind_output == nullptr || replay->type_ops->unbind_output == nullptr ||
          port_plan->max_emits_per_compute == 0U) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      auto options = external_service_options(*service, *publisher, *subscriber);
      if (!options) {
        return make_unexpected(std::move(options).error());
      }
      auto catalog_endpoint = find_catalog_endpoint(plan, PortId{topic.graph_port.value});
      const auto* const catalog_record =
          catalog_endpoint ? ChannelEndpointCatalogAccess::record(*catalog_endpoint) : nullptr;
      if (catalog_record == nullptr || catalog_record->service != service->service_id ||
          catalog_record->service_static.id != service->service_id ||
          ChannelEndpointCatalogAccess::backend_service_binding(*catalog_endpoint).empty()) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      if (tensor_service && runtime_external_tensor_uses_unsupported_strides(
                                replay->type_ops, *port_plan, *service, *catalog_record)) {
        return make_unexpected(Error{ErrorCode::kNotSupported});
      }
      if (tensor_service && !runtime_external_tensor_matches_plan(
                                replay->type_ops, *port_plan, *service, *catalog_record)) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      const PortId boundary_port{topic.graph_port.value};
      const EndpointClockHealthPlanRecord* const health_plan =
          find_endpoint_clock_health(plan, boundary_port);
      if (health_plan == nullptr || health_plan->endpoint != boundary_port ||
          health_plan->plan_generation != plan.plan_generation ||
          catalog_record->direction != TopicDirection::kExport ||
          catalog_record->boundary_port != boundary_port ||
          catalog_record->clock_health.capability != health_plan->capability ||
          catalog_record->clock_health.requirement != health_plan->requirement ||
          catalog_record->clock_health.max_report_age != health_plan->max_report_age) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      auto capture_clock_binding =
          bind_catalog_capture_clock(plan, *catalog_endpoint, health_plan->capture_clock);
      if (!capture_clock_binding) {
        return make_unexpected(std::move(capture_clock_binding).error());
      }
      std::unique_ptr<channel_internal::InlinePayloadPublisher> endpoint;
      channel_internal::CatalogCanonicalPodPublisherBinding canonical_pod_publisher;
      if (!tensor_service) {
        auto created = channel_internal::CatalogChannelBindingAccess::create_payload_publisher(
            *impl->runtime_,
            *catalog_endpoint,
            replay->type_ops->wire_capacity,
            {},
            RuntimeChannelBindingAccess::graph_blocking_pressure_options());
        if (!created) {
          return make_unexpected(std::move(created).error());
        }
        endpoint = std::move(*created);
        if (replay->type_ops->codec_family == RuntimePayloadCodecFamily::kCustomPod) {
          canonical_pod_publisher =
              channel_internal::CatalogChannelBindingAccess::bind_canonical_pod_publisher(
                  *endpoint);
          if (!canonical_pod_publisher) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
        }
      }
      const auto existing =
          std::find_if(impl->outputs_.begin(), impl->outputs_.end(), [&](const auto& candidate) {
            return candidate->port == replay->port;
          });
      Impl::OutputState* output = nullptr;
      if (existing == impl->outputs_.end()) {
        auto created = std::make_unique<Impl::OutputState>();
        created->port = replay->port;
        created->owner = replay->owner;
        created->declaration = replay->declaration;
        created->type_ops = replay->type_ops;
        created->phase_counters = impl->phase_counters_;
        created->max_emits_per_activation = port_plan->max_emits_per_compute;
        if (replay->type_ops->codec_family == RuntimePayloadCodecFamily::kCustomPod) {
          if (!port_plan->payload_identity.pod) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
          created->pod_identity = port_plan->payload_identity.pod;
        }
        output = created.get();
        impl->outputs_.push_back(std::move(created));
      } else {
        output = existing->get();
        if (output->owner != replay->owner || output->declaration != replay->declaration ||
            output->type_ops != replay->type_ops) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
      }
      if (tensor_service) {
        const auto& catalog_requirements = *catalog_record->service_static.tensor;
        const auto* const interface_pool = find_tensor_interface_pool(plan, topic.id);
        if (service->tensor_emit_policy != InterfaceTensorEmitPolicy::kRequireExportable &&
            (!catalog_requirements.memory_kind || interface_pool == nullptr ||
             interface_pool->topic != topic.id || interface_pool->graph_port != boundary_port ||
             interface_pool->memory_kind != *catalog_requirements.memory_kind ||
             ((interface_pool->memory_kind == MemoryKind::kPinnedHost ||
               interface_pool->memory_kind == MemoryKind::kCudaDevice) &&
              !interface_pool->device.valid()))) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        if (!output->catalog_tensor_requirements) {
          output->catalog_tensor_requirements = catalog_requirements;
        } else if (!logical_tensor_requirements_equal(*output->catalog_tensor_requirements,
                                                      catalog_requirements)) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        // Defer materialization until every catalog Tensor fanout arm has agreed on the one
        // Output-wide contract. This gives protocol validation deterministic precedence over
        // backend availability and avoids constructing a partially validated fanout.
        pending_catalog_tensor_publishers.push_back(PendingCatalogTensorPublisher{
            .output = output,
            .endpoint = *catalog_endpoint,
            .max_loaned_samples = publisher->max_loaned_samples,
            .planned_device = interface_pool != nullptr && interface_pool->device.valid()
                                  ? std::optional<DeviceId>{interface_pool->device}
                                  : std::nullopt,
            .plan_capture_clock = capture_clock_binding->plan,
            .catalog_capture_clock = capture_clock_binding->catalog,
            .peer_required = catalog_record->attach.peer == PeerRequirement::kRequired,
        });
      } else {
        output->endpoints.push_back(Impl::OutputState::EndpointState{
            .publisher = std::move(endpoint),
            .plan_capture_clock = capture_clock_binding->plan,
            .catalog_capture_clock = capture_clock_binding->catalog,
            .expected_recipient_count = options->expected_recipient_count,
            .refresh_before_emit = true,
            .peer_required = catalog_record->attach.peer == PeerRequirement::kRequired,
            .catalog_encodes_custom_pod =
                replay->type_ops->codec_family == RuntimePayloadCodecFamily::kCustomPod,
            .canonical_pod_publisher = canonical_pod_publisher,
        });
      }
    }

    for (auto& pending : pending_catalog_tensor_publishers) {
      auto tensor_endpoint = RuntimeChannelBindingAccess::create_catalog_tensor_publisher(
          *impl->runtime_, pending.endpoint, pending.max_loaned_samples, pending.planned_device);
      if (!tensor_endpoint) {
        HOLOSCAN_LOG_ERROR("External Tensor publisher materialization failed with {}",
                           runtime_error_message(tensor_endpoint.error()));
        return make_unexpected(std::move(tensor_endpoint).error());
      }
      pending.output->catalog_tensor_endpoints.push_back(
          Impl::OutputState::CatalogTensorEndpointState{
              .publisher = std::move(*tensor_endpoint),
              .plan_capture_clock = pending.plan_capture_clock,
              .catalog_capture_clock = pending.catalog_capture_clock,
              .peer_required = pending.peer_required,
              .active_for_current_fanout = false,
          });
    }

    for (const auto& output : impl->outputs_) {
      if (output->type_ops->codec_family == RuntimePayloadCodecFamily::kGeneratedSchema ||
          output->type_ops->codec_family == RuntimePayloadCodecFamily::kCustomPod ||
          output->type_ops->codec_family == RuntimePayloadCodecFamily::kDescriptorTensorSchema) {
        auto scratch =
            AlignedBlock::create(output->type_ops->wire_capacity, alignof(std::max_align_t));
        if (!scratch) {
          return make_unexpected(std::move(scratch).error());
        }
        output->encode_scratch = std::move(*scratch);
      }
      if (output->process_local_schema_identity) {
        auto scratch = AlignedBlock::create(
            kSchemaEnvelopeHeaderSize + output->type_ops->wire_capacity, alignof(std::max_align_t));
        if (!scratch) {
          return make_unexpected(std::move(scratch).error());
        }
        output->process_local_schema_scratch = std::move(*scratch);
      }
      output->binding =
          RuntimeOutputBinding{.state = output.get(), .ops = &Impl::OutputState::kBindingOps};
      const auto loan_plan =
          std::find_if(plan.generic_output_loans.begin(),
                       plan.generic_output_loans.end(),
                       [&](const auto& loan) { return loan.output == output->port; });
      if (loan_plan != plan.generic_output_loans.end()) {
        if (loan_plan->disposition != PlanDisposition::kAccepted ||
            loan_plan->plan_generation != plan.plan_generation || loan_plan->run_generation != 0U ||
            loan_plan->slot_count == 0U || loan_plan->max_live_loans == 0U ||
            loan_plan->max_live_loans > loan_plan->slot_count ||
            loan_plan->slot_size_bytes != output->type_ops->payload_size ||
            output->type_ops->construct_value == nullptr ||
            output->type_ops->destroy_value == nullptr) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        output->max_live_loans = loan_plan->max_live_loans;
        output->loans.reserve(loan_plan->slot_count);
        for (std::uint32_t index = 0U; index < loan_plan->slot_count; ++index) {
          auto value = AlignedBlock::create(output->type_ops->payload_size,
                                            output->type_ops->payload_alignment);
          if (!value) {
            return make_unexpected(std::move(value).error());
          }
          auto* control = new Impl::LoanControl;
          control->owner = output.get();
          control->live_credit = &output->live_loan_count;
          control->type_ops = output->type_ops;
          control->value = std::move(*value);
          output->type_ops->construct_value(control->value.data());
          control->binding =
              RuntimeLoanBinding{.state = control, .cancel = &Impl::LoanControl::cancel};
          output->loans.push_back(control);
        }
      }
      output->type_ops->bind_output(output->declaration, &output->binding);
    }

    const std::array<const std::vector<ProcessLocalChannelSubscriberPlanRecord>*, 2U>
        subscriber_families{&plan.process_local_channel_subscribers, &cross_process_subscribers};
    for (const auto* subscriber_records : subscriber_families) {
      for (const auto& subscriber_plan : *subscriber_records) {
        const auto* service = locate_service(subscriber_plan.service_id);
        const auto* replay =
            find_replay(replay_ports, subscriber_plan.target, PortDirection::kInput);
        const auto* port_plan = find_port(plan, subscriber_plan.target, PortDirection::kInput);
        const auto* publisher_plan = locate_publisher(subscriber_plan.service_id);
        if (service == nullptr || port_plan == nullptr || publisher_plan == nullptr) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        if (process_group && port_plan->owner_kind == PortOwnerKind::kOperator &&
            !worker_owns_vertex(plan, process_group, port_plan->owner)) {
          continue;
        }
        if (replay == nullptr && port_plan->owner_kind != PortOwnerKind::kOperator) {
          continue;
        }
        const bool tensor_service = service->tensor_payload;
        const bool runtime_type_matches =
            tensor_service
                ? runtime_tensor_matches_plan(
                      replay == nullptr ? nullptr : replay->type_ops, *port_plan, *service)
                : runtime_codec_matches_plan(replay == nullptr ? nullptr : replay->type_ops,
                                             *port_plan,
                                             service->payload_size_bytes,
                                             !service->cross_process_transport);
        if (replay == nullptr || !replay->owner || replay->declaration == nullptr ||
            !runtime_type_matches ||
            (!tensor_service && replay->type_ops->construct_sample == nullptr) ||
            replay->type_ops->destroy_sample == nullptr ||
            replay->type_ops->bind_input == nullptr || replay->type_ops->unbind_input == nullptr ||
            subscriber_plan.readiness_capacity == 0U ||
            subscriber_plan.requested_queue_depth == 0U ||
            subscriber_plan.max_borrowed_samples == 0U ||
            subscriber_plan.event_wake != service->event_wake ||
            port_plan->owner != replay->owner ||
            (service->queue_policy != QueuePolicy::kBlocking &&
             !publisher_plan->replacement_attribution_required)) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }

        Impl::InputPortState* input = impl->find_input(replay->port);
        if (input == nullptr) {
          auto created = std::make_unique<Impl::InputPortState>();
          created->port = replay->port;
          created->owner = replay->owner;
          created->declaration = replay->declaration;
          created->type_ops = replay->type_ops;
          created->phase_counters = impl->phase_counters_;
          const InputSelectionPlanRecord* selection = find_selection(plan, replay->port);
          if (selection != nullptr) {
            if (!selection->reference_clock.valid() || selection->cache_capacity == 0U ||
                (selection->missing == TemporalRecordMissingKind::kReuseLast &&
                 (selection->cardinality != InputSelectionCardinality::kSingleSample ||
                  selection->batch_selection_mode != BatchSelectionMode::kNotApplicable ||
                  selection->reuse_max_age_ns <= 0))) {
              return make_unexpected(Error{ErrorCode::kInvalidArgument});
            }
            created->reference_clock = selection->reference_clock;
            created->conversion_id = selection->conversion_id;
            created->capture_time_required = selection->selector != InputSelectorKind::kDrainAll;
            created->reuse_last = selection->missing == TemporalRecordMissingKind::kReuseLast;
            if (selection->conversion_id == 0U) {
              created->capture_clock = selection->reference_clock;
            } else {
              const ClockConversionRecord* conversion =
                  find_conversion(plan, selection->conversion_id);
              if (conversion == nullptr || conversion->plan_generation != plan.plan_generation ||
                  !conversion->from.valid() || conversion->to != selection->reference_clock ||
                  conversion->rounding != ClockRoundingMode::kNearestTiesToEven ||
                  conversion->overflow != ClockOverflowPolicy::kError) {
                return make_unexpected(Error{ErrorCode::kInvalidArgument});
              }
              created->capture_clock = conversion->from;
              created->capture_to_reference = DirectAffineClockEdge{
                  .source = conversion->from,
                  .target = conversion->to,
                  .conversion = AffineClockConversion{.scale = conversion->scale,
                                                      .offset_ns = conversion->offset_ns},
              };
            }
          }
          created->binding = RuntimeInputBinding{
              .state = created.get(),
              .ops = &Impl::InputPortState::kBindingOps,
              .generation = created->active_generation,
          };
          input = created.get();
          impl->inputs_.push_back(std::move(created));
        } else if (input->owner != replay->owner || input->declaration != replay->declaration ||
                   input->type_ops != replay->type_ops) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }

        const bool direct_retained =
            service->delivery_kind == ProcessLocalChannelDeliveryKind::kDirectRetained;
        ChannelOptions options;
        if (!direct_retained) {
          auto resolved_options = service_options(*service);
          if (!resolved_options) {
            return make_unexpected(std::move(resolved_options).error());
          }
          options = *resolved_options;
          options.subscriber_queue_depth = subscriber_plan.requested_queue_depth;
          options.max_borrowed_samples = subscriber_plan.max_borrowed_samples;
        }

        auto inline_descriptor_contract = find_inline_descriptor_tensor_contract(
            plan, subscriber_plan.source, subscriber_plan.target, *service);
        if (!inline_descriptor_contract) {
          return make_unexpected(std::move(inline_descriptor_contract).error());
        }
        const std::uint64_t effective_tensor_max_byte_span =
            *inline_descriptor_contract ? (*inline_descriptor_contract)->max_byte_span
                                        : service->tensor_max_byte_span;
        const MemoryKind effective_tensor_memory_kind =
            *inline_descriptor_contract ? (*inline_descriptor_contract)->memory_kind
                                        : service->tensor_memory_kind;
        const DeviceId effective_tensor_device = *inline_descriptor_contract
                                                     ? (*inline_descriptor_contract)->device
                                                     : service->tensor_device;
        auto subscriber = std::make_unique<Impl::SubscriberState>();
        subscriber->source = subscriber_plan.source;
        subscriber->target = subscriber_plan.target;
        subscriber->service = service->service_id;
        if (const auto* source_replay =
                find_replay(replay_ports, subscriber_plan.source, PortDirection::kOutput)) {
          subscriber->source_owner = source_replay->owner;
        }
        subscriber->owner = replay->owner;
        subscriber->queue_policy = service->queue_policy;
        subscriber->readiness_capacity = subscriber_plan.readiness_capacity;
        subscriber->drain_capacity = subscriber_plan.requested_queue_depth;
        subscriber->read_claim_capacity = subscriber_plan.max_borrowed_samples;
        subscriber->pre_witness_replacement_allowed =
            publisher_plan->replacement_attribution_required;
        subscriber->direct_process_local = direct_retained;
        subscriber->cross_process_transport = service->cross_process_transport;
        subscriber->input = input;
        if (tensor_service) {
          subscriber->tensor_max_byte_span = effective_tensor_max_byte_span;
          subscriber->tensor_memory_kind = effective_tensor_memory_kind;
          subscriber->tensor_device = effective_tensor_device;
          subscriber->tensor_fields = service->tensor_fields;
          if (replay->type_ops->codec_family ==
              RuntimePayloadCodecFamily::kDescriptorTensorSchema) {
            const auto tensor_count = replay->type_ops->descriptor_tensor_static_count;
            subscriber->descriptor_tensor_scratch.resize(tensor_count);
            subscriber->tensor_schema_scratch.resize(tensor_count);
          }
          if (!*inline_descriptor_contract) {
            auto field_device =
                uniform_tensor_field_device(subscriber->tensor_device, subscriber->tensor_fields);
            if (!field_device) {
              return make_unexpected(std::move(field_device).error());
            }
            subscriber->tensor_device = *field_device;
          }
        }
        if (direct_retained) {
          if (service->cross_process_transport || service->event_wake ||
              subscriber_plan.event_wake ||
              (tensor_service && replay->type_ops != runtime_port_type_ops<Tensor>() &&
               replay->type_ops->codec_family !=
                   RuntimePayloadCodecFamily::kDescriptorTensorSchema)) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
          if (tensor_service) {
            if (replay->type_ops->codec_family ==
                RuntimePayloadCodecFamily::kDescriptorTensorSchema) {
              if (replay->type_ops->wire_capacity != service->payload_size_bytes ||
                  replay->type_ops->descriptor_tensor_sample_metadata == nullptr ||
                  replay->type_ops->assign_descriptor_tensors_sample == nullptr) {
                return make_unexpected(Error{ErrorCode::kProtocolError});
              }
              if (*inline_descriptor_contract) {
                subscriber->pass_through_tensor_layout =
                    (*inline_descriptor_contract)->input_layout;
              }
            } else if (service->tensor_memory_kind == MemoryKind::kUnknown) {
              subscriber->pass_through_tensor_layout = port_plan->tensor_layout;
            }
          } else {
            auto scratch =
                AlignedBlock::create(service->payload_size_bytes, alignof(std::max_align_t));
            if (!scratch) {
              return make_unexpected(std::move(scratch).error());
            }
            subscriber->scratch = std::move(*scratch);
            subscriber->scratch_size = service->payload_size_bytes;
          }
        } else if (tensor_service && replay->type_ops->codec_family ==
                                         RuntimePayloadCodecFamily::kDescriptorTensorSchema) {
          auto endpoint = RuntimeChannelBindingAccess::create_tensor_schema_subscriber(
              *impl->runtime_,
              service->service_id,
              replay->type_ops->wire_capacity,
              replay->type_ops->descriptor_tensor_static_count,
              options);
          if (!endpoint) {
            return make_unexpected(std::move(endpoint).error());
          }
          subscriber->tensor_schema_endpoint = std::move(*endpoint);
          auto scratch =
              AlignedBlock::create(replay->type_ops->wire_capacity, alignof(std::max_align_t));
          if (!scratch) {
            return make_unexpected(std::move(scratch).error());
          }
          subscriber->scratch = std::move(*scratch);
          subscriber->scratch_size = replay->type_ops->wire_capacity;
        } else if (tensor_service) {
          auto endpoint = RuntimeChannelBindingAccess::create_tensor_subscriber(
              *impl->runtime_, service->service_id, options);
          if (!endpoint) {
            return make_unexpected(std::move(endpoint).error());
          }
          subscriber->tensor_endpoint.emplace(std::move(*endpoint));
          if (service->tensor_memory_kind == MemoryKind::kUnknown) {
            subscriber->pass_through_tensor_layout = port_plan->tensor_layout;
          }
        } else {
          auto endpoint = RuntimeChannelBindingAccess::create_subscriber(
              *impl->runtime_, service->service_id, service->payload_size_bytes, options);
          if (!endpoint) {
            return make_unexpected(std::move(endpoint).error());
          }
          subscriber->endpoint = std::move(*endpoint);
          auto scratch =
              AlignedBlock::create(service->payload_size_bytes, alignof(std::max_align_t));
          if (!scratch) {
            return make_unexpected(std::move(scratch).error());
          }
          subscriber->scratch = std::move(*scratch);
          subscriber->scratch_size = service->payload_size_bytes;
        }
        if (replay->type_ops->codec_family == RuntimePayloadCodecFamily::kCustomPod) {
          if (!port_plan->payload_identity.pod) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
          subscriber->pod_identity = port_plan->payload_identity.pod;
          subscriber->native_scratch.resize(replay->type_ops->payload_size);
        } else if (replay->type_ops->codec_family == RuntimePayloadCodecFamily::kGeneratedSchema &&
                   !service->cross_process_transport) {
          if (!port_plan->payload_identity.schema) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
          subscriber->process_local_schema_identity = port_plan->payload_identity.schema;
        }
        const std::uint32_t typed_slot_capacity =
            std::max(subscriber_plan.requested_queue_depth, subscriber_plan.cache_capacity);
        if (subscriber_plan.readiness_capacity < typed_slot_capacity) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        subscriber->typed_slot_capacity = typed_slot_capacity;
        subscriber->direct_ingress_capacity =
            direct_retained && service->queue_policy != QueuePolicy::kBlocking
                ? subscriber_plan.requested_queue_depth
                : 0U;
        if (typed_slot_capacity >
            std::numeric_limits<std::uint32_t>::max() - subscriber->direct_ingress_capacity) {
          return make_unexpected(Error{ErrorCode::kResourceExhausted});
        }
        const std::uint32_t sample_slot_capacity =
            typed_slot_capacity + subscriber->direct_ingress_capacity;
        subscriber->slots.reserve(sample_slot_capacity);
        for (std::uint32_t index = 0U; index < sample_slot_capacity; ++index) {
          auto storage = AlignedBlock::create(replay->type_ops->sample_size,
                                              replay->type_ops->sample_alignment);
          if (!storage) {
            return make_unexpected(std::move(storage).error());
          }
          Impl::SampleSlot slot;
          slot.subscriber = subscriber.get();
          slot.sample = std::move(*storage);
          subscriber->slots.push_back(std::move(slot));
          if (replay->type_ops->persistent_sample_storage) {
            if (replay->type_ops->prepare_sample_storage == nullptr ||
                replay->type_ops->reset_sample_storage == nullptr) {
              return make_unexpected(Error{ErrorCode::kProtocolError});
            }
            auto prepared =
                replay->type_ops->prepare_sample_storage(subscriber->slots.back().sample.data());
            if (!prepared) {
              return make_unexpected(std::move(prepared).error());
            }
            subscriber->slots.back().sample_storage_prepared = true;
          }
        }
        if (!subscriber->direct_process_local) {
          auto attached = subscriber->attach_wait_set(*impl->wait_set_);
          if (!attached) {
            return make_unexpected(std::move(attached).error());
          }
          subscriber->wait_handle = *attached;
          subscriber->terminal_wait_handle = subscriber->secondary_wait_set_handle().value_or(0U);
          impl->wait_attachment_count_ += subscriber->terminal_wait_handle == 0U ? 1U : 2U;
        }
        if (input->active_slots.max_size() - input->active_slots.capacity() < typed_slot_capacity) {
          return make_unexpected(Error{ErrorCode::kResourceExhausted});
        }
        input->active_slots.reserve(input->active_slots.capacity() + typed_slot_capacity);
        if (input->retirement_slots.max_size() - input->retirement_slots.capacity() <
            typed_slot_capacity) {
          return make_unexpected(Error{ErrorCode::kResourceExhausted});
        }
        input->retirement_slots.reserve(input->retirement_slots.capacity() + typed_slot_capacity);
        const std::size_t tensor_fields_per_slot =
            replay->type_ops->codec_family == RuntimePayloadCodecFamily::kDescriptorTensorSchema
                ? replay->type_ops->descriptor_tensor_static_count
                : 1U;
        if (tensor_fields_per_slot == 0U ||
            tensor_fields_per_slot >
                std::numeric_limits<std::size_t>::max() / typed_slot_capacity) {
          return make_unexpected(Error{ErrorCode::kResourceExhausted});
        }
        const std::size_t tensor_lease_capacity =
            tensor_fields_per_slot * static_cast<std::size_t>(typed_slot_capacity);
        if (input->tensor_access_slots.max_size() - input->tensor_access_slots.capacity() <
                typed_slot_capacity ||
            input->tensor_read_leases.max_size() - input->tensor_read_leases.capacity() <
                tensor_lease_capacity) {
          return make_unexpected(Error{ErrorCode::kResourceExhausted});
        }
        input->tensor_access_slots.reserve(input->tensor_access_slots.capacity() +
                                           typed_slot_capacity);
        input->tensor_read_leases.reserve(input->tensor_read_leases.capacity() +
                                          tensor_lease_capacity);
        if (direct_retained) {
          const auto source_output = std::find_if(
              impl->outputs_.begin(), impl->outputs_.end(), [&](const auto& candidate) {
                return candidate->port == subscriber_plan.source;
              });
          if (source_output == impl->outputs_.end() ||
              (*source_output)->direct_subscriber != nullptr ||
              !(*source_output)->endpoints.empty() || !(*source_output)->tensor_endpoints.empty() ||
              !(*source_output)->catalog_tensor_endpoints.empty() ||
              !(*source_output)->tensor_schema_endpoints.empty()) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
          (*source_output)->direct_subscriber = subscriber.get();
        }
        input->subscribers.push_back(subscriber.get());
        impl->subscribers_.push_back(std::move(subscriber));
      }
    }

    for (const auto& topic : plan.graph.external_topics()) {
      if (!topic.imported) {
        continue;
      }
      auto internal_port = external_internal_port(plan, PortId{topic.graph_port.value}, true);
      if (!internal_port) {
        if (internal_port.error().code == ErrorCode::kNotFound) {
          continue;
        }
        return make_unexpected(std::move(internal_port).error());
      }
      const auto* replay = find_replay(replay_ports, *internal_port, PortDirection::kInput);
      const auto* port_plan = find_port(plan, *internal_port, PortDirection::kInput);
      const auto* service = find_external_service(plan, topic.id);
      const auto* publisher = find_external_publisher(plan, topic.id);
      const auto* subscriber_plan = find_external_subscriber(plan, topic.id);
      if (process_group && port_plan != nullptr &&
          port_plan->owner_kind == PortOwnerKind::kOperator &&
          !worker_owns_vertex(plan, process_group, port_plan->owner)) {
        continue;
      }
      const bool tensor_service = service != nullptr && service->tensor_payload;
      const bool runtime_type_matches =
          tensor_service
              ? replay != nullptr && replay->type_ops == runtime_port_type_ops<Tensor>() &&
                    port_plan != nullptr &&
                    port_plan->payload_identity.kind == PayloadIdentityKind::kTensor
              : replay != nullptr && port_plan != nullptr &&
                    runtime_codec_matches_plan(
                        replay->type_ops,
                        *port_plan,
                        replay->type_ops == nullptr ? 0U : replay->type_ops->wire_capacity,
                        false);
      if (replay == nullptr || port_plan == nullptr || service == nullptr || publisher == nullptr ||
          subscriber_plan == nullptr || replay->owner != port_plan->owner ||
          replay->declaration == nullptr || !runtime_type_matches ||
          (!tensor_service && replay->type_ops->construct_sample == nullptr) ||
          replay->type_ops->destroy_sample == nullptr || replay->type_ops->bind_input == nullptr ||
          replay->type_ops->unbind_input == nullptr) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      auto options = external_service_options(*service, *publisher, *subscriber_plan);
      if (!options) {
        return make_unexpected(std::move(options).error());
      }
      auto catalog_endpoint = find_catalog_endpoint(plan, PortId{topic.graph_port.value});
      const auto* const catalog_record =
          catalog_endpoint ? ChannelEndpointCatalogAccess::record(*catalog_endpoint) : nullptr;
      if (catalog_record == nullptr || catalog_record->service != service->service_id ||
          catalog_record->service_static.id != service->service_id ||
          ChannelEndpointCatalogAccess::backend_service_binding(*catalog_endpoint).empty()) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      if (tensor_service && runtime_external_tensor_uses_unsupported_strides(
                                replay->type_ops, *port_plan, *service, *catalog_record)) {
        return make_unexpected(Error{ErrorCode::kNotSupported});
      }
      if (tensor_service && !runtime_external_tensor_matches_plan(
                                replay->type_ops, *port_plan, *service, *catalog_record)) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      const auto* const tensor_interface_pool =
          tensor_service ? find_tensor_interface_pool(plan, topic.id) : nullptr;
      if (tensor_service &&
          service->tensor_emit_policy != InterfaceTensorEmitPolicy::kRequireExportable &&
          (tensor_interface_pool == nullptr || tensor_interface_pool->topic != topic.id ||
           tensor_interface_pool->graph_port.value() != topic.graph_port.value ||
           !catalog_record->service_static.tensor->memory_kind ||
           tensor_interface_pool->memory_kind !=
               *catalog_record->service_static.tensor->memory_kind ||
           ((tensor_interface_pool->memory_kind == MemoryKind::kPinnedHost ||
             tensor_interface_pool->memory_kind == MemoryKind::kCudaDevice) &&
            !tensor_interface_pool->device.valid()))) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      const auto generation_capacity =
          std::find_if(plan.imported_generation_capacities.begin(),
                       plan.imported_generation_capacities.end(),
                       [&](const auto& record) { return record.topic == topic.id; });
      std::optional<std::uint64_t> attach_max_wait_ns;
      if (catalog_record->attach.max_wait) {
        attach_max_wait_ns = static_cast<std::uint64_t>(catalog_record->attach.max_wait->count());
      }
      if (generation_capacity == plan.imported_generation_capacities.end() ||
          generation_capacity->endpoint_or_service_id != service->service_id ||
          generation_capacity->peer_requirement != catalog_record->attach.peer ||
          generation_capacity->attach_max_wait_ns != attach_max_wait_ns) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      // Catalog Tensor imports currently own a typed descriptor attachment but no generation-
      // fenced detach/rebind implementation. Reject a restart-capable plan before materializing
      // transport instead of retaining old-generation authority across a lifecycle restart.
      if (tensor_service && generation_capacity->restart_capable_minimum_two) {
        return make_unexpected(Error{ErrorCode::kNotSupported});
      }
      const PortId boundary_port{topic.graph_port.value};
      const EndpointClockHealthPlanRecord* const health_plan =
          find_endpoint_clock_health(plan, boundary_port);
      if (health_plan == nullptr || catalog_record == nullptr ||
          health_plan->endpoint != boundary_port ||
          health_plan->plan_generation != plan.plan_generation ||
          catalog_record->direction != TopicDirection::kImport ||
          catalog_record->boundary_port != boundary_port ||
          catalog_record->clock_health.capability != health_plan->capability ||
          catalog_record->clock_health.requirement != health_plan->requirement ||
          catalog_record->clock_health.max_report_age != health_plan->max_report_age) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      const bool monitored = health_plan->capability == ClockHealthCapability::kStatusReports;
      if (tensor_service && monitored) {
        return make_unexpected(Error{ErrorCode::kNotSupported});
      }
      if (monitored &&
          (health_plan->requirement != ClockHealthRequirement::kFreshHealthyReport ||
           !health_plan->max_report_age.has_value() || health_plan->max_report_age->count() <= 0 ||
           health_plan->fixed_report_ingress_capacity != 1U ||
           health_plan->fixed_update_capacity != 4U ||
           health_plan->per_dispatch_report_budget != 1U)) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      if (!monitored && (health_plan->capability != ClockHealthCapability::kUnavailable ||
                         health_plan->requirement != ClockHealthRequirement::kNotRequired ||
                         health_plan->max_report_age.has_value() ||
                         health_plan->fixed_report_ingress_capacity != 0U ||
                         health_plan->fixed_update_capacity != 0U ||
                         health_plan->per_dispatch_report_budget != 0U)) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      auto capture_clock_binding =
          bind_catalog_capture_clock(plan, *catalog_endpoint, health_plan->capture_clock);
      if (!capture_clock_binding) {
        return make_unexpected(std::move(capture_clock_binding).error());
      }
      const auto payload_edge =
          std::find_if(plan.payload_edges.begin(), plan.payload_edges.end(), [&](const auto& edge) {
            return edge.source == PortId(topic.graph_port.value) && edge.target == *internal_port;
          });
      const std::uint32_t cache_capacity =
          payload_edge == plan.payload_edges.end() ? 0U : payload_edge->cache_capacity;
      const std::uint32_t typed_slot_capacity =
          std::max(subscriber_plan->requested_queue_depth, cache_capacity);
      if (typed_slot_capacity == 0U) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      if (typed_slot_capacity == std::numeric_limits<std::uint32_t>::max()) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      // Every catalog import can make all retained data slots ready and then accept one ordered
      // EOS control record before the controller freezes its candidate cut. Reserve that terminal
      // witness explicitly; EOS owns no typed payload slot.
      const std::uint32_t readiness_capacity = typed_slot_capacity + 1U;
      // A catalog Tensor import can retain every typed slot while one replacement is materialized
      // before the evicted backing is released. Reserve that transient receive-control claim
      // independently; it does not represent the ordered EOS readiness witness above.
      const std::uint32_t tensor_receive_control_capacity = typed_slot_capacity + 1U;
      std::unique_ptr<channel_internal::InlinePayloadSubscriber> endpoint;
      std::optional<ChannelSubscriber<Tensor>> tensor_endpoint;
      if (tensor_service) {
        auto created = RuntimeChannelBindingAccess::create_tensor_subscriber(
            *impl->runtime_,
            ChannelEndpointCatalogAccess::backend_service_binding(*catalog_endpoint),
            *options,
            catalog_record->service_static.tensor,
            tensor_receive_control_capacity);
        if (!created) {
          return make_unexpected(std::move(created).error());
        }
        tensor_endpoint.emplace(std::move(*created));
      } else {
        auto created = channel_internal::CatalogChannelBindingAccess::create_payload_subscriber(
            *impl->runtime_, *catalog_endpoint, replay->type_ops->wire_capacity);
        if (!created) {
          return make_unexpected(std::move(created).error());
        }
        endpoint = std::move(*created);
      }

      Impl::InputPortState* input = impl->find_input(replay->port);
      if (input == nullptr) {
        auto created = std::make_unique<Impl::InputPortState>();
        created->port = replay->port;
        created->owner = replay->owner;
        created->declaration = replay->declaration;
        created->type_ops = replay->type_ops;
        created->phase_counters = impl->phase_counters_;
        const InputSelectionPlanRecord* selection = find_selection(plan, replay->port);
        if (selection != nullptr) {
          if (!selection->reference_clock.valid() || selection->cache_capacity == 0U ||
              (selection->missing == TemporalRecordMissingKind::kReuseLast &&
               (selection->cardinality != InputSelectionCardinality::kSingleSample ||
                selection->batch_selection_mode != BatchSelectionMode::kNotApplicable ||
                selection->reuse_max_age_ns <= 0))) {
            return make_unexpected(Error{ErrorCode::kInvalidArgument});
          }
          created->reference_clock = selection->reference_clock;
          created->conversion_id = selection->conversion_id;
          created->capture_time_required = selection->selector != InputSelectorKind::kDrainAll;
          created->reuse_last = selection->missing == TemporalRecordMissingKind::kReuseLast;
          if (selection->conversion_id == 0U) {
            created->capture_clock = selection->reference_clock;
          } else {
            const ClockConversionRecord* conversion =
                find_conversion(plan, selection->conversion_id);
            if (conversion == nullptr || conversion->plan_generation != plan.plan_generation ||
                !conversion->from.valid() || conversion->to != selection->reference_clock ||
                conversion->rounding != ClockRoundingMode::kNearestTiesToEven ||
                conversion->overflow != ClockOverflowPolicy::kError) {
              return make_unexpected(Error{ErrorCode::kInvalidArgument});
            }
            created->capture_clock = conversion->from;
            created->capture_to_reference = DirectAffineClockEdge{
                .source = conversion->from,
                .target = conversion->to,
                .conversion = AffineClockConversion{.scale = conversion->scale,
                                                    .offset_ns = conversion->offset_ns},
            };
          }
        }
        created->binding = RuntimeInputBinding{
            .state = created.get(),
            .ops = &Impl::InputPortState::kBindingOps,
            .generation = created->active_generation,
        };
        input = created.get();
        impl->inputs_.push_back(std::move(created));
      } else if (input->owner != replay->owner || input->declaration != replay->declaration ||
                 input->type_ops != replay->type_ops) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      if ((input->capture_time_required || input->capture_clock.valid()) &&
          input->capture_clock != capture_clock_binding->plan) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }

      auto subscriber = std::make_unique<Impl::SubscriberState>();
      subscriber->source = PortId{topic.graph_port.value};
      subscriber->target = *internal_port;
      subscriber->owner = replay->owner;
      subscriber->queue_policy = service->queue_policy;
      subscriber->readiness_capacity = readiness_capacity;
      subscriber->drain_capacity = subscriber_plan->requested_queue_depth;
      subscriber->read_claim_capacity = subscriber_plan->max_borrowed_samples;
      subscriber->pre_witness_replacement_allowed = publisher->replacement_attribution_required;
      subscriber->externally_driven_import = !tensor_service;
      subscriber->catalog_tensor_import = tensor_service;
      subscriber->input = input;
      subscriber->endpoint = std::move(endpoint);
      if (tensor_endpoint) {
        subscriber->tensor_endpoint.emplace(std::move(*tensor_endpoint));
      }
      if (tensor_service) {
        subscriber->tensor_max_byte_span = *catalog_record->service_static.tensor->max_byte_span;
        subscriber->tensor_memory_kind = *catalog_record->service_static.tensor->memory_kind;
        subscriber->tensor_device =
            tensor_interface_pool != nullptr ? tensor_interface_pool->device : DeviceId{};
        subscriber->tensor_logical_requirements = *catalog_record->service_static.tensor;
      }
      subscriber->catalog_capture_clock = capture_clock_binding->catalog;
      subscriber->plan_capture_clock = capture_clock_binding->plan;
      // Admit the plan-frozen retained-generation capacity for this import. A missing or
      // non-restart-capable record keeps every rebind attempt failing closed with kNotSupported;
      // the catalog copy shares immutable state so a later rebind can re-resolve the same record.
      if (!tensor_service) {
        subscriber->rebind_catalog = plan.endpoint_catalog;
        subscriber->rebind_endpoint_name = std::string{catalog_endpoint->name()};
        subscriber->import_generation_capacity = generation_capacity->endpoint_max_generations;
        subscriber->import_restart_capable = generation_capacity->restart_capable_minimum_two;
      }
      if (monitored) {
        subscriber->clock_health_endpoint = boundary_port;
        subscriber->health_dispatch_budget = health_plan->per_dispatch_report_budget;
      }
      if (!tensor_service) {
        auto scratch =
            AlignedBlock::create(replay->type_ops->wire_capacity, alignof(std::max_align_t));
        if (!scratch) {
          return make_unexpected(std::move(scratch).error());
        }
        subscriber->scratch = std::move(*scratch);
        subscriber->scratch_size = replay->type_ops->wire_capacity;
      }
      if (replay->type_ops->codec_family == RuntimePayloadCodecFamily::kCustomPod) {
        if (!port_plan->payload_identity.pod) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        subscriber->pod_identity = port_plan->payload_identity.pod;
      }
      subscriber->typed_slot_capacity = typed_slot_capacity;
      subscriber->slots.reserve(typed_slot_capacity);
      for (std::uint32_t index = 0U; index < typed_slot_capacity; ++index) {
        auto storage =
            AlignedBlock::create(replay->type_ops->sample_size, replay->type_ops->sample_alignment);
        if (!storage) {
          return make_unexpected(std::move(storage).error());
        }
        Impl::SampleSlot slot;
        slot.subscriber = subscriber.get();
        slot.sample = std::move(*storage);
        subscriber->slots.push_back(std::move(slot));
        if (replay->type_ops->persistent_sample_storage) {
          if (replay->type_ops->prepare_sample_storage == nullptr ||
              replay->type_ops->reset_sample_storage == nullptr) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
          auto prepared =
              replay->type_ops->prepare_sample_storage(subscriber->slots.back().sample.data());
          if (!prepared) {
            return make_unexpected(std::move(prepared).error());
          }
          subscriber->slots.back().sample_storage_prepared = true;
        }
      }
      if (!subscriber->direct_process_local) {
        auto attached = subscriber->attach_wait_set(*impl->wait_set_);
        if (!attached) {
          return make_unexpected(std::move(attached).error());
        }
        subscriber->wait_handle = *attached;
        subscriber->terminal_wait_handle = subscriber->secondary_wait_set_handle().value_or(0U);
        impl->wait_attachment_count_ += subscriber->terminal_wait_handle == 0U ? 1U : 2U;
      }
      if (input->active_slots.max_size() - input->active_slots.capacity() < typed_slot_capacity) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      input->active_slots.reserve(input->active_slots.capacity() + typed_slot_capacity);
      if (input->retirement_slots.max_size() - input->retirement_slots.capacity() <
          typed_slot_capacity) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      input->retirement_slots.reserve(input->retirement_slots.capacity() + typed_slot_capacity);
      if (input->tensor_access_slots.max_size() - input->tensor_access_slots.capacity() <
              typed_slot_capacity ||
          input->tensor_read_leases.max_size() - input->tensor_read_leases.capacity() <
              typed_slot_capacity) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      input->tensor_access_slots.reserve(input->tensor_access_slots.capacity() +
                                         typed_slot_capacity);
      input->tensor_read_leases.reserve(input->tensor_read_leases.capacity() + typed_slot_capacity);
      input->subscribers.push_back(subscriber.get());
      impl->subscribers_.push_back(std::move(subscriber));
    }
    impl->clock_health_reset_requests_.reserve(impl->subscribers_.size());
    for (const auto& input : impl->inputs_) {
      if (input->type_ops->predicate_view_size == 0U ||
          input->type_ops->predicate_view_alignment == 0U ||
          input->type_ops->construct_predicate_view == nullptr ||
          input->type_ops->destroy_predicate_view == nullptr) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      auto predicate_storage = AlignedBlock::create(input->type_ops->predicate_view_size,
                                                    input->type_ops->predicate_view_alignment);
      if (!predicate_storage) {
        return make_unexpected(std::move(predicate_storage).error());
      }
      input->predicate_view_storage = std::move(*predicate_storage);
      std::size_t claim_capacity = 0U;
      for (const auto* subscriber : input->subscribers) {
        if (claim_capacity >
            std::numeric_limits<std::size_t>::max() - subscriber->read_claim_capacity) {
          return make_unexpected(Error{ErrorCode::kResourceExhausted});
        }
        claim_capacity += subscriber->read_claim_capacity;
      }
      auto claim_pool = ReadClaimPool::create(claim_capacity);
      if (!claim_pool) {
        return make_unexpected(std::move(claim_pool).error());
      }
      input->claim_pool = *claim_pool;
      input->type_ops->bind_input(input->declaration, &input->binding);
    }
    auto pump_indices = impl->initialize_pump_indices();
    if (!pump_indices) {
      return make_unexpected(std::move(pump_indices).error());
    }
    auto activation_owners = impl->bind_activation_owners(plan.ports.size());
    if (!activation_owners) {
      return make_unexpected(std::move(activation_owners).error());
    }
    impl->freeze_descriptor_import_inventory();
    impl->ready_handles_.resize(impl->wait_attachment_count_);
    return std::unique_ptr<RuntimeChannelSession>{new RuntimeChannelSession{std::move(impl)}};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }
#endif
}

bool RuntimeChannelSession::has_materialized_activation_state(const VertexId owner) const noexcept {
  if (!impl_ || !owner) {
    return false;
  }
  return impl_->find_activation_owner(owner) != nullptr;
}

bool RuntimeChannelSession::has_materialized_input_state(const VertexId owner) const noexcept {
  if (!impl_ || !owner) {
    return false;
  }
  const auto* activation_owner = impl_->find_activation_owner(owner);
  return activation_owner != nullptr && !activation_owner->inputs.empty();
}

expected<std::unique_ptr<PortBinding>, Error> RuntimeChannelSession::create_readiness_binding(
    VertexId owner, RuntimePortBindingRequest request) noexcept {
  if (!impl_ || !owner || request.kind != RuntimeEventKind::kInputArrival || !request.source ||
      !request.source_clock.valid() || !request.reference_clock.valid() || request.capacity == 0U ||
      request.source_generation == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto* activation_owner = impl_->find_activation_owner(owner);
  if (activation_owner == nullptr || activation_owner->inputs.empty() ||
      request.plan_generation != impl_->plan_generation_ || request.run_generation == 0U ||
      std::any_of(activation_owner->inputs.begin(),
                  activation_owner->inputs.end(),
                  [](const auto* input) { return input->readiness_binding != nullptr; })) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  std::size_t candidate_capacity = 0U;
  for (const auto* subscriber : activation_owner->subscribers) {
    const std::size_t subscriber_capacity = subscriber->slots.size() + 1U;
    if (subscriber_capacity == 0U ||
        candidate_capacity > std::numeric_limits<std::size_t>::max() - subscriber_capacity) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    candidate_capacity += subscriber_capacity;
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    auto binding = std::make_unique<Impl::ReadinessBinding>(
        impl_.get(),
        owner,
        request,
        candidate_capacity,
        activation_owner->inputs,
        std::any_of(activation_owner->subscribers.begin(),
                    activation_owner->subscribers.end(),
                    [](const auto* subscriber) { return subscriber->externally_driven_import; }));
    for (auto* input : activation_owner->inputs) {
      input->readiness_binding = binding.get();
    }
    impl_->readiness_bindings_.push_back(binding.get());
    impl_->cache_direct_continuation_candidate(owner, *binding);
    return std::unique_ptr<PortBinding>{std::move(binding)};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
#endif
}

expected<std::uint32_t, Error> RuntimeChannelSession::readiness_capacity(
    VertexId owner) const noexcept {
  if (!impl_ || !owner) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  std::uint32_t capacity = 0U;
  const auto* activation_owner = impl_->find_activation_owner(owner);
  if (activation_owner == nullptr || activation_owner->subscribers.empty()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  for (const auto* subscriber : activation_owner->subscribers) {
    if (capacity > std::numeric_limits<std::uint32_t>::max() - subscriber->readiness_capacity) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    capacity += subscriber->readiness_capacity;
  }
  return capacity != 0U
             ? expected<std::uint32_t, Error>{capacity}
             : expected<std::uint32_t, Error>{make_unexpected(Error{ErrorCode::kNotFound})};
}

expected<void, Error> RuntimeChannelSession::bind_clock_health(
    EndpointClockHealthRuntimeCoordinator& coordinator, std::mutex& serialization_mutex) noexcept {
  if (!impl_ || impl_->armed_.load(std::memory_order_acquire) || impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (impl_->clock_health_serialization_mutex_ != nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  // Validate the complete cold binding before publishing any coordinator pointer. A partial
  // binding could otherwise let one external import advance while another fails closed.
  for (const auto& subscriber : impl_->subscribers_) {
    if (!subscriber->clock_health_endpoint) {
      if (subscriber->clock_health_coordinator != nullptr ||
          subscriber->health_dispatch_budget != 0U) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      continue;
    }
    if (subscriber->clock_health_coordinator != nullptr ||
        subscriber->health_dispatch_budget == 0U ||
        coordinator.receiver(subscriber->clock_health_endpoint) == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
  }
  for (const auto& subscriber : impl_->subscribers_) {
    if (subscriber->clock_health_endpoint) {
      subscriber->clock_health_coordinator = &coordinator;
    }
  }
  if (coordinator.receiver_count() != 0U) {
    impl_->clock_health_serialization_mutex_ = &serialization_mutex;
  }
  return {};
}

expected<void, Error> RuntimeChannelSession::refresh_connections() noexcept {
  if (!impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  for (const auto& output : impl_->outputs_) {
    for (const auto& endpoint : output->endpoints) {
      auto refreshed = endpoint.publisher->update_connections();
      if (!refreshed) {
        return make_unexpected(std::move(refreshed).error());
      }
    }
    for (auto& endpoint : output->tensor_endpoints) {
      auto refreshed = endpoint.update_connections();
      if (!refreshed) {
        return make_unexpected(std::move(refreshed).error());
      }
    }
    for (auto& endpoint : output->catalog_tensor_endpoints) {
      auto refreshed = endpoint.publisher.update_connections();
      if (!refreshed) {
        return make_unexpected(std::move(refreshed).error());
      }
    }
  }
  return {};
}

expected<void, Error> RuntimeChannelSession::arm() noexcept {
  if (!impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  auto current = impl_->status();
  if (!current) {
    return make_unexpected(std::move(current).error());
  }
  if (impl_->armed_.exchange(true, std::memory_order_acq_rel) || impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  impl_->cancel_requested_.store(false, std::memory_order_release);
  impl_->closed_.store(false, std::memory_order_release);
  impl_->terminal_owner_.store(0U, std::memory_order_relaxed);
  impl_->terminal_diagnostic_.store(0U, std::memory_order_relaxed);
  impl_->terminal_error_.store(ErrorCode::kSuccess, std::memory_order_release);
  impl_->terminal_failure_claimed_.store(false, std::memory_order_release);
  // Keep a session with no pump-drain-eligible subscriber logically armed without starting an
  // empty worker. Direct-retained inputs need no transport drain, and externally driven imports
  // are drained by the controller together with their ordered catalog operations.
  if (impl_->pump_subscribers_.empty()) {
    return {};
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    impl_->pump_thread_ = std::thread{[impl = impl_.get()] { impl->pump(); }};
    impl_->pump_startup_.acquire();
    auto started = impl_->status();
    if (!started) {
      impl_->pump_thread_.join();
      impl_->armed_.store(false, std::memory_order_release);
      return make_unexpected(std::move(started).error());
    }
    return started;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    impl_->armed_.store(false, std::memory_order_release);
    impl_->closed_.store(true, std::memory_order_release);
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
#endif
}

expected<void, Error> RuntimeChannelSession::flush_process_local_activation_cut() noexcept {
  if (!impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (!impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto current = impl_->status();
  if (!current) {
    return make_unexpected(std::move(current).error());
  }
  if (impl_->pump_subscribers_.empty()) {
    return {};
  }
  count_phase(impl_->phase_counters_, RuntimePhaseCounter::kChannelActivationCuts);

  const auto cut_already_drained = [&]() noexcept {
    return !impl_->blocked_transport_retry_required_.load(std::memory_order_acquire) &&
           impl_->process_local_progress_tracking_valid_.load(std::memory_order_acquire) &&
           impl_->process_local_received_deliveries_.load(std::memory_order_acquire) >=
               impl_->process_local_published_deliveries_.load(std::memory_order_acquire);
  };
  if (cut_already_drained()) {
    count_phase(impl_->phase_counters_, RuntimePhaseCounter::kChannelActivationCutSkips);
    return {};
  }

  // The pump may already be inside its finite scan when compute retires. Joining that scan and
  // completing one more bounded pass establishes the activation cut: every process-local sample
  // published by the just-finished serial compute is either in typed storage or remains queued
  // behind ordinary typed-store backpressure before the controller can freeze another candidate.
  const std::scoped_lock cut_lock{impl_->pump_scan_mutex_};
  if (cut_already_drained()) {
    count_phase(impl_->phase_counters_, RuntimePhaseCounter::kChannelActivationCutSkips);
    return {};
  }
  if (!impl_->initialize_controller_descriptor_import()) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  count_phase(impl_->phase_counters_, RuntimePhaseCounter::kChannelActivationCutDrains);
  // A pump-observed blocked head remains authoritative even when unrelated inbound and outbound
  // progress totals happen to cancel in a relay process. The serialized retry below either
  // materializes that head or relatches the condition if typed retention is still full.
  impl_->blocked_transport_retry_required_.store(false, std::memory_order_release);
  PortId failed_port;
  VertexId failed_owner;
  std::optional<StructuralDiagnosticId> terminal_diagnostic;
  auto drained = impl_->drain_unmonitored_inputs(failed_port, failed_owner, terminal_diagnostic);
  if (!drained) {
    HOLOSCAN_LOG_ERROR("Post-compute Channel activation cut failed with {} on port {}",
                       error_code_name(drained.error().code),
                       failed_port.value());
    impl_->fail(drained.error(), failed_owner, terminal_diagnostic);
    return make_unexpected(std::move(drained).error());
  }
  return impl_->status();
}

std::optional<RuntimeChannelDirectContinuationTarget>
RuntimeChannelSession::arm_direct_continuation(const VertexId producer,
                                               const std::uint32_t execution_lane) noexcept {
  if (!impl_) {
    return std::nullopt;
  }
  auto target = impl_->direct_continuation_target(producer, execution_lane, true, true);
  if (!target) {
    return std::nullopt;
  }
  auto* const owner = impl_->find_activation_owner(producer);
  auto* const output = owner == nullptr ? nullptr : owner->direct_continuation_output;
  if (output == nullptr || output->port != target->source_output) {
    return std::nullopt;
  }
  output->direct_continuation_capture = true;
  return target;
}

expected<std::optional<RuntimeChannelDirectContinuationReceipt>, Error>
RuntimeChannelSession::poll_direct_continuation(const VertexId producer,
                                                const std::uint32_t execution_lane,
                                                const TimePoint observed_at) noexcept {
  if (!impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return impl_->poll_direct_continuation(producer, execution_lane, observed_at);
}

bool RuntimeChannelSession::validate_direct_continuation_receipt(
    const RuntimeChannelDirectContinuationReceipt& receipt) const noexcept {
  return impl_ != nullptr && impl_->validate_direct_continuation_receipt(receipt);
}

void RuntimeChannelSession::cancel() noexcept {
  if (impl_) {
    impl_->cancel();
  }
}

expected<void, Error> RuntimeChannelSession::join() noexcept {
  return impl_ ? impl_->join() : expected<void, Error>{make_unexpected(Error{ErrorCode::kClosed})};
}

expected<TemporalSampleRecord, Error> RuntimeChannelSession::temporal_sample(
    PortId port, std::uint64_t readiness_sequence) const noexcept {
  if (!impl_ || !port || readiness_sequence == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const Impl::InputPortState* const input = impl_->find_input(port);
  if (input == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  for (const Impl::SubscriberState* const subscriber : input->subscribers) {
    const std::scoped_lock lock{subscriber->mutex};
    for (const Impl::SampleSlot& slot : subscriber->slots) {
      if (slot.occupied && !slot.direct_ingress_unread &&
          slot.readiness_sequence == readiness_sequence) {
        return slot.temporal;
      }
    }
  }
  return make_unexpected(Error{ErrorCode::kNotFound});
}

// Activation preparation stages an all-or-nothing typed selection across owner-local ports; the
// explicit branches make rollback, duplicate detection, and claim acquisition auditable.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeChannelSession::prepare_activation(
    VertexId owner, std::uint64_t generation,
    std::span<const RuntimeChannelActivationSelection> selections) noexcept {
  if (!impl_ || !owner || generation == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  Impl::ActivationOwnerState* const activation_owner = impl_->find_activation_owner(owner);
  if (activation_owner == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
#if !defined(NDEBUG)
  Impl::OwnerMutationGuard owner_guard{impl_->controller_mutation_barrier_,
                                       impl_->owner_mutation_flag(owner)};
#endif
  bool any_active = false;
  bool all_active = true;
  for (const auto* input : activation_owner->inputs) {
    any_active = any_active || input->active;
    all_active = all_active && input->active;
    if (input->active && input->active_generation != generation) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
  }
  for (const auto* output : activation_owner->outputs) {
    any_active = any_active || output->active;
    all_active = all_active && output->active;
    if (output->active && output->active_generation != generation) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
  }
  if (any_active != all_active) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  const bool initial_prepare = !any_active;
  if (initial_prepare) {
    activation_owner->direct_continuation.reset();
    for (auto* output : activation_owner->outputs) {
      output->direct_continuation_capture = false;
    }
  }

  for (std::size_t index = 0U; index < selections.size(); ++index) {
    const auto* input = Impl::find_activation_input(*activation_owner, selections[index].port);
    if (input == nullptr || !valid_input_selection(selections[index].selection) ||
        (!initial_prepare && input->selection_prepared) ||
        std::any_of(selections.first(index).begin(),
                    selections.first(index).end(),
                    [&](const auto& prior) { return prior.port == selections[index].port; })) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
  }

  if (initial_prepare) {
    for (auto* input : activation_owner->inputs) {
      if (!input->active_slots.empty() || input->predicate_view_constructed ||
          input->selection_prepared || input->staged_now) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      input->active_generation = generation;
      input->binding.generation = generation;
      input->type_ops->bind_input(input->declaration, &input->binding);
      input->active_selection = {InputState::kNotSelected,
                                 SelectionReason::kNotSelectedForActivation};
      input->batch = false;
      input->destructive = false;
      input->active = true;
    }
    for (auto* output : activation_owner->outputs) {
      output->active_generation = generation;
      output->emitted_this_activation = 0U;
      output->active = true;
    }
  }

  const auto rollback_staging = [&]() noexcept {
    for (auto* input : activation_owner->inputs) {
      if (!input->staged_now) {
        continue;
      }
      if (input->predicate_view_constructed) {
        input->type_ops->destroy_predicate_view(input->predicate_view_storage.data());
        input->predicate_view_constructed = false;
      }
      for (Impl::SampleSlot* slot : input->active_slots) {
        const std::scoped_lock lock{slot->subscriber->mutex};
        slot->active_selected = false;
      }
      input->active_slots.clear();
      input->active_selection = {InputState::kNotSelected,
                                 SelectionReason::kNotSelectedForActivation};
      input->batch = false;
      input->destructive = false;
      input->selection_prepared = false;
      input->staged_now = false;
    }
    if (initial_prepare) {
      for (auto* input : activation_owner->inputs) {
        input->active = false;
      }
      for (auto* output : activation_owner->outputs) {
        output->active = false;
      }
    }
  };

  for (const RuntimeChannelActivationSelection& selected : selections) {
    Impl::InputPortState* input = Impl::find_activation_input(*activation_owner, selected.port);
    if (input == nullptr) {
      rollback_staging();
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    input->active_slots.clear();
    input->active_selection = selected.selection;
    input->batch = selected.batch;
    input->destructive = selected.destructive;
    input->staged_now = true;
    for (const TemporalSampleRecord& record : selected.samples) {
      Impl::SampleSlot* match = nullptr;
      for (Impl::SubscriberState* subscriber : input->subscribers) {
        bool duplicate = false;
        bool unavailable = false;
        {
          const std::scoped_lock lock{subscriber->mutex};
          const auto found = std::find_if(
              subscriber->slots.begin(), subscriber->slots.end(), [&](const auto& slot) {
                return slot.occupied && !slot.direct_ingress_unread && slot.temporal == record;
              });
          if (found != subscriber->slots.end()) {
            duplicate = match != nullptr;
            unavailable = found->active_selected || found->typed_ingress_reserved;
            if (!duplicate && !unavailable) {
              found->active_selected = true;
              match = &*found;
            }
          }
        }
        if (duplicate || unavailable) {
          rollback_staging();
          return make_unexpected(
              Error{duplicate ? ErrorCode::kProtocolError : ErrorCode::kStaleEndpoint});
        }
      }
      if (match == nullptr) {
        rollback_staging();
        return make_unexpected(Error{ErrorCode::kStaleEndpoint});
      }
      input->active_slots.push_back(match);
    }
    const bool value_state = selected.selection.state == InputState::kSelected ||
                             selected.selection.state == InputState::kReused;
    if ((!selected.batch && value_state && input->active_slots.size() != 1U) ||
        (!value_state && !input->active_slots.empty())) {
      rollback_staging();
      return make_unexpected(Error{ErrorCode::kInputSelectionCardinalityMismatch});
    }
  }

  for (auto* input : activation_owner->inputs) {
    if (!input->staged_now) {
      continue;
    }
    const bool value_state = input->active_selection.state == InputState::kSelected ||
                             input->active_selection.state == InputState::kReused;
    if (value_state && !input->batch) {
      if (input->active_slots.size() != 1U) {
        rollback_staging();
        return make_unexpected(Error{ErrorCode::kInputSelectionCardinalityMismatch});
      }
      auto constructed =
          input->type_ops->construct_predicate_view(input->predicate_view_storage.data(),
                                                    input->active_slots.front()->sample.data(),
                                                    input->active_selection,
                                                    &input->view_generation,
                                                    input->view_generation);
      if (!constructed) {
        rollback_staging();
        return make_unexpected(std::move(constructed).error());
      }
      input->predicate_view_constructed = true;
    }
  }
  for (auto* input : activation_owner->inputs) {
    if (input->staged_now) {
      input->selection_prepared = true;
      input->staged_now = false;
    }
  }
  count_phase(impl_->phase_counters_, RuntimePhaseCounter::kChannelPrepareCalls);
  return {};
}

expected<void, Error> RuntimeChannelSession::prepare_tensor_access(
    const VertexId owner, const std::uint64_t generation, cudaStream_t consumer_stream) noexcept {
  if (!impl_ || !owner || generation == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  Impl::ActivationOwnerState* const activation_owner = impl_->find_activation_owner(owner);
  if (activation_owner == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
#if !defined(NDEBUG)
  Impl::OwnerMutationGuard owner_guard{impl_->controller_mutation_barrier_,
                                       impl_->owner_mutation_flag(owner)};
#endif
  for (const auto* input : activation_owner->inputs) {
    if (!input->active || input->active_generation != generation) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
  }
  for (const auto* output : activation_owner->outputs) {
    if (!output->active || output->active_generation != generation) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
  }
  for (auto* input : activation_owner->inputs) {
    auto prepared = Impl::prepare_tensor_input_access(*input, consumer_stream);
    if (!prepared) {
      return prepared;
    }
  }
  return {};
}

// Retirement commits or rolls back every activation-scoped payload, reuse, view, and loan claim
// before publishing the next view generation.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeChannelSession::retire_activation(VertexId owner,
                                                               std::uint64_t generation,
                                                               bool activation_accepted) noexcept {
  if (!impl_ || !owner || generation == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  Impl::ActivationOwnerState* const activation_owner = impl_->find_activation_owner(owner);
  if (activation_owner == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
#if !defined(NDEBUG)
  Impl::OwnerMutationGuard owner_guard{impl_->controller_mutation_barrier_,
                                       impl_->owner_mutation_flag(owner)};
#endif
  for (const auto* input : activation_owner->inputs) {
    if (!input->active || input->active_generation != generation) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    if (input->view_generation == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
  }
  if (activation_accepted) {
    for (const auto* input : activation_owner->inputs) {
      if (!input->destructive) {
        continue;
      }
      for (const Impl::SampleSlot* slot : input->active_slots) {
        const std::scoped_lock lock{slot->subscriber->mutex};
        if (slot->event_emitted && !slot->witness_retired) {
          return make_unexpected(Error{ErrorCode::kBackpressured});
        }
      }
      const bool value_state = input->active_selection.state == InputState::kSelected ||
                               input->active_selection.state == InputState::kReused;
      if (input->reuse_last && value_state && input->active_slots.size() == 1U &&
          input->reuse_slot != nullptr && input->reuse_slot != input->active_slots.front()) {
        const std::scoped_lock lock{input->reuse_slot->subscriber->mutex};
        if (input->reuse_slot->active_selected ||
            (input->reuse_slot->event_emitted && !input->reuse_slot->witness_retired)) {
          return make_unexpected(Error{ErrorCode::kBackpressured});
        }
      }
    }
  }
  for (const auto* output : activation_owner->outputs) {
    if (!output->active || output->active_generation != generation) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    if (std::any_of(output->loans.begin(), output->loans.end(), [](const auto* loan) {
          return loan->live;
        })) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
  }
  const bool signal_direct_progress = impl_->owner_progress_wake_enabled();
  for (auto* input : activation_owner->inputs) {
    if (input->predicate_view_constructed) {
      input->type_ops->destroy_predicate_view(input->predicate_view_storage.data());
      input->predicate_view_constructed = false;
    }
    const bool value_state = input->active_selection.state == InputState::kSelected ||
                             input->active_selection.state == InputState::kReused;
    Impl::SampleSlot* retained = activation_accepted && input->destructive && input->reuse_last &&
                                         value_state && input->active_slots.size() == 1U
                                     ? input->active_slots.front()
                                     : nullptr;
    if (retained != nullptr && input->reuse_slot != nullptr && input->reuse_slot != retained) {
      Impl::SampleSlot* prior = input->reuse_slot;
      const std::scoped_lock lock{prior->subscriber->mutex};
      prior->reuse_retained = false;
      Impl::reclaim_sample_slot_if_unclaimed(*prior);
    }
    for (Impl::SampleSlot* slot : input->active_slots) {
      const std::scoped_lock lock{slot->subscriber->mutex};
      slot->active_selected = false;
      if (slot == retained) {
        slot->reuse_retained = true;
      } else {
        // Selection commit/rollback owns the temporal claim; activation retirement owns only the
        // activation and finite reuse claims. Reclaiming through the shared gate preserves scalar
        // rollback, overlapping windows, and cache-retained non-destructive selections.
        Impl::reclaim_sample_slot_if_unclaimed(*slot);
      }
    }
    for (Impl::SubscriberState* subscriber : input->subscribers) {
      bool direct_capacity_may_have_changed = false;
      {
        const std::scoped_lock lock{subscriber->mutex};
        auto promoted = impl_->promote_direct_ingress(*subscriber);
        if (!promoted) {
          return make_unexpected(std::move(promoted).error());
        }
        direct_capacity_may_have_changed =
            signal_direct_progress && subscriber->direct_process_local &&
            (subscriber->direct_ingress_capacity != 0U
                 ? impl_->writable_direct_ingress_slot(*subscriber) != nullptr
                 : impl_->writable_slot(*subscriber) != nullptr);
      }
      if (direct_capacity_may_have_changed) {
        impl_->signal_owner_progress(subscriber->source_owner);
      }
    }
    if (retained != nullptr) {
      input->reuse_slot = retained;
    }
    input->tensor_read_leases.clear();
    input->tensor_access_slots.clear();
    input->active_slots.clear();
    input->selection_prepared = false;
    input->staged_now = false;
    input->active = false;
    ++input->view_generation;
  }
  for (auto* output : activation_owner->outputs) {
    output->active = false;
    output->direct_continuation_capture = false;
  }
  activation_owner->direct_continuation.reset();
  count_phase(impl_->phase_counters_, RuntimePhaseCounter::kChannelRetireCalls);
  return {};
}

expected<const void*, Error> RuntimeChannelSession::predicate_view(
    PortId port, std::uint64_t generation) const noexcept {
  if (!impl_ || !port || generation == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const Impl::InputPortState* const input = impl_->find_input(port);
  if (input == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  if (!input->active || input->active_generation != generation) {
    return make_unexpected(Error{ErrorCode::kStaleEndpoint});
  }
  if (!input->predicate_view_constructed) {
    return make_unexpected(Error{input->active_selection.state == InputState::kNotSelected
                                     ? ErrorCode::kInputNotSelected
                                     : ErrorCode::kInputEmpty});
  }
  return input->predicate_view_storage.data();
}

// Selector-proof retirement validates the complete requested set before releasing any temporal
// claim, then revalidates under the slot locks before the infallible commit phase.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeChannelSession::retire_samples(
    PortId port, std::span<const TemporalSampleRecord> samples) noexcept {
  if (!impl_ || !port) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  Impl::InputPortState* input = impl_->find_input(port);
  if (input == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  if (samples.empty()) {
    return {};
  }
#if !defined(NDEBUG)
  Impl::OwnerMutationGuard owner_guard{impl_->controller_mutation_barrier_,
                                       impl_->owner_mutation_flag(input->owner)};
#endif
  input->retirement_slots.clear();
  const auto rollback = [&]() noexcept {
    for (Impl::SampleSlot* slot : input->retirement_slots) {
      const std::scoped_lock lock{slot->subscriber->mutex};
      slot->retirement_selected = false;
    }
    input->retirement_slots.clear();
  };
  for (const auto& record : samples) {
    Impl::SampleSlot* match = nullptr;
    bool record_seen = false;
    for (Impl::SubscriberState* subscriber : input->subscribers) {
      bool duplicate = false;
      bool unavailable = false;
      bool protected_claim = false;
      {
        const std::scoped_lock lock{subscriber->mutex};
        const auto found =
            std::find_if(subscriber->slots.begin(), subscriber->slots.end(), [&](const auto& slot) {
              return slot.occupied && !slot.direct_ingress_unread && slot.temporal == record;
            });
        if (found != subscriber->slots.end()) {
          duplicate = record_seen;
          record_seen = true;
          // retire_samples() is a trusted selector-proof boundary. It may run while the current
          // activation still borrows the selected payload; physical reclamation is then deferred
          // until retire_activation(). A retained sample outside such an activation is not a
          // valid selector release and remains fail-closed.
          protected_claim = found->reuse_retained && !input->active;
          unavailable = !found->temporal_retained || found->retirement_selected ||
                        found->typed_ingress_reserved || (found->active_selected && !input->active);
          if (!duplicate && !unavailable && !protected_claim) {
            found->retirement_selected = true;
            match = &*found;
          }
        }
      }
      if (duplicate || unavailable || protected_claim) {
        rollback();
        return make_unexpected(Error{duplicate         ? ErrorCode::kProtocolError
                                     : protected_claim ? ErrorCode::kBackpressured
                                                       : ErrorCode::kStaleEndpoint});
      }
    }
    if (match == nullptr) {
      rollback();
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    if (input->retirement_slots.size() == input->retirement_slots.capacity()) {
      rollback();
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    input->retirement_slots.push_back(match);
  }
  for (Impl::SampleSlot* slot : input->retirement_slots) {
    bool unavailable = false;
    {
      const std::scoped_lock lock{slot->subscriber->mutex};
      unavailable = !slot->occupied || !slot->temporal_retained || !slot->retirement_selected;
    }
    if (unavailable) {
      rollback();
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
  }
  for (Impl::SampleSlot* slot : input->retirement_slots) {
    const std::scoped_lock lock{slot->subscriber->mutex};
    slot->temporal_retained = false;
    slot->retirement_selected = false;
    Impl::reclaim_sample_slot_if_unclaimed(*slot);
  }
  const bool signal_direct_progress = impl_->owner_progress_wake_enabled();
  for (Impl::SubscriberState* subscriber : input->subscribers) {
    bool direct_capacity_may_have_changed = false;
    {
      const std::scoped_lock lock{subscriber->mutex};
      auto promoted = impl_->promote_direct_ingress(*subscriber);
      if (!promoted) {
        input->retirement_slots.clear();
        return make_unexpected(std::move(promoted).error());
      }
      direct_capacity_may_have_changed =
          signal_direct_progress && subscriber->direct_process_local &&
          (subscriber->direct_ingress_capacity != 0U
               ? impl_->writable_direct_ingress_slot(*subscriber) != nullptr
               : impl_->writable_slot(*subscriber) != nullptr);
    }
    if (direct_capacity_may_have_changed) {
      impl_->signal_owner_progress(subscriber->source_owner);
    }
  }
  input->retirement_slots.clear();
  return {};
}

// A whole-session reset is a two-phase transaction: preflight every endpoint and retained claim,
// then reset health and advance all producer epochs together. Keeping those phases adjacent makes
// the no-partial-reset guarantee reviewable across independently owned subscribers.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeChannelSession::reset() noexcept {
  if (!impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
#if !defined(NDEBUG)
  Impl::ControlMutationGuard control_guard{impl_->controller_mutation_barrier_};
#endif
  auto current = impl_->status();
  if (!current) {
    return make_unexpected(std::move(current).error());
  }
  if (impl_->pump_thread_.joinable() || impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  for (const auto& input : impl_->inputs_) {
    if (input->active || input->predicate_view_constructed ||
        input->claim_pool->has_live_claims()) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
  }
  for (const auto& output : impl_->outputs_) {
    if (output->active || std::any_of(output->loans.begin(),
                                      output->loans.end(),
                                      [](const auto* loan) { return loan->live; })) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
  }
  for (const auto& subscriber : impl_->subscribers_) {
    if (subscriber->catalog_tensor_import) {
      // Catalog Tensor imports have no generation-fenced detach/rebind operation yet. Refuse the
      // reset before clearing retained samples or advancing producer identity.
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    if (!subscriber->externally_driven_import) {
      continue;
    }
    // Externally driven imports pass through the generation-fenced rebind gate instead of the
    // former unconditional kNotSupported: kBackpressured while the current incarnation may still
    // publish, kNotSupported for a non-restart-capable import, kResourceExhausted beyond the
    // plan-admitted retained-generation capacity, all before any destructive step.
    const std::scoped_lock lock{subscriber->mutex};
    auto gated = impl_->import_rebind_gate(*subscriber);
    if (!gated) {
      return make_unexpected(std::move(gated).error());
    }
  }
  for (const auto& subscriber : impl_->subscribers_) {
    // Full reset has no lifecycle fence authorizing transport consumption. Observe the transport
    // head and typed cache under their owning locks so future endpoint implementations cannot
    // race this cold-path preflight. Do not clear cached samples or advance producer epochs until
    // every process-local endpoint is empty. Imports were classified by the rebind gate above; a
    // fenced foreign head must not read as pressure.
    const std::scoped_lock lock{subscriber->transport_mutex, subscriber->mutex};
    if (!subscriber->externally_driven_import) {
      auto pending = subscriber->has_pending_transport();
      if (!pending) {
        return make_unexpected(std::move(pending).error());
      }
      if (*pending) {
        return make_unexpected(Error{ErrorCode::kBackpressured});
      }
    }
    if (subscriber->producer_epoch == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
  }
  // Epoch exhaustion is preflighted for the entire session before any retained sample is
  // destroyed. A failure therefore leaves every independently owned subscriber at the same
  // observable generation and with the same typed payload claims it had on entry.
  auto reset_health = impl_->reset_clock_health([](const auto&) { return true; });
  if (!reset_health) {
    return make_unexpected(std::move(reset_health).error());
  }
  for (const auto& subscriber : impl_->subscribers_) {
    if (!subscriber->externally_driven_import) {
      continue;
    }
    // Atomic detach+rebind under the fence. A failure inside the destructive window has already
    // terminalized the session; nothing below may run against a partially cut attachment.
    const std::scoped_lock lock{subscriber->mutex};
    auto rebound = impl_->rebind_import(*subscriber);
    if (!rebound) {
      return make_unexpected(std::move(rebound).error());
    }
  }
  for (const auto& subscriber : impl_->subscribers_) {
    subscriber->clear_samples();
    ++subscriber->producer_epoch;
  }
  for (const auto& input : impl_->inputs_) {
    input->reuse_slot = nullptr;
    input->retirement_slots.clear();
    input->selection_prepared = false;
    input->staged_now = false;
  }
  for (const auto& output : impl_->outputs_) {
    output->next_sequence = 1U;
  }
  impl_->next_arrival_order_.store(1U, std::memory_order_release);
  impl_->blocked_transport_retry_required_.store(false, std::memory_order_release);
  impl_->terminal_error_.store(ErrorCode::kSuccess, std::memory_order_release);
  impl_->terminal_owner_.store(0U, std::memory_order_relaxed);
  impl_->terminal_diagnostic_.store(0U, std::memory_order_relaxed);
  impl_->terminal_failure_claimed_.store(false, std::memory_order_release);
  impl_->cancel_requested_.store(false, std::memory_order_release);
  impl_->closed_.store(false, std::memory_order_release);
  return {};
}

// Owner-scoped reset fences transport, claims, generations, and health state in contractual order.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeChannelSession::reset(VertexId owner) noexcept {
  if (!impl_ || !owner) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
#if !defined(NDEBUG)
  Impl::ControlMutationGuard control_guard{impl_->controller_mutation_barrier_};
#endif
  auto current = impl_->status();
  if (!current) {
    return make_unexpected(std::move(current).error());
  }
  if (impl_->pump_thread_.joinable() || impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  bool found = false;
  for (const auto& input : impl_->inputs_) {
    if (input->owner != owner) {
      continue;
    }
    found = true;
    if (input->active || input->predicate_view_constructed ||
        input->claim_pool->has_live_claims()) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
  }
  for (const auto& output : impl_->outputs_) {
    if (output->owner != owner) {
      continue;
    }
    found = true;
    if (output->active || std::any_of(output->loans.begin(),
                                      output->loans.end(),
                                      [](const auto* loan) { return loan->live; })) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
  }
  for (const auto& subscriber : impl_->subscribers_) {
    if (subscriber->catalog_tensor_import &&
        (subscriber->owner == owner || subscriber->source_owner == owner)) {
      // Keep the existing attachment and every observable generation intact until a typed catalog
      // Tensor detach/rebind transaction is implemented.
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    if (!subscriber->externally_driven_import ||
        (subscriber->owner != owner && subscriber->source_owner != owner)) {
      continue;
    }
    found = true;
    // Generation-fenced rebind gate; see reset() for the typed outcome contract.
    const std::scoped_lock lock{subscriber->mutex};
    auto gated = impl_->import_rebind_gate(*subscriber);
    if (!gated) {
      return make_unexpected(std::move(gated).error());
    }
  }
  for (const auto& subscriber : impl_->subscribers_) {
    if (subscriber->owner != owner && subscriber->source_owner != owner) {
      continue;
    }
    found = true;
    if (subscriber->input->active || subscriber->input->claim_pool->has_live_claims()) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    if (subscriber->externally_driven_import) {
      // Imports were classified by the rebind gate above; a fenced foreign head is not pressure.
      continue;
    }
    // Observe the endpoint and typed cache under their owning locks. This cold-path fence keeps a
    // future receive implementation from changing either half of the reset preflight between the
    // pending check and generation advance.
    const std::scoped_lock lock{subscriber->transport_mutex, subscriber->mutex};
    auto pending = subscriber->has_pending_transport();
    if (!pending) {
      return make_unexpected(std::move(pending).error());
    }
    if (*pending) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
  }
  if (!found) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }

  for (const auto& input : impl_->inputs_) {
    if (input->owner == owner &&
        (input->active_generation == std::numeric_limits<std::uint64_t>::max() ||
         input->view_generation == std::numeric_limits<std::uint64_t>::max())) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
  }
  for (const auto& output : impl_->outputs_) {
    if (output->owner == owner &&
        output->active_generation == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
  }
  for (const auto& subscriber : impl_->subscribers_) {
    if ((subscriber->owner == owner || subscriber->source_owner == owner) &&
        subscriber->producer_epoch == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
  }
  auto reset_health = impl_->reset_clock_health([&](const auto& subscriber) {
    return subscriber.owner == owner || subscriber.source_owner == owner;
  });
  if (!reset_health) {
    return make_unexpected(std::move(reset_health).error());
  }

  for (const auto& subscriber : impl_->subscribers_) {
    if (!subscriber->externally_driven_import ||
        (subscriber->owner != owner && subscriber->source_owner != owner)) {
      continue;
    }
    // Atomic detach+rebind under the fence; a destructive-window failure terminalized the session.
    const std::scoped_lock lock{subscriber->mutex};
    auto rebound = impl_->rebind_import(*subscriber);
    if (!rebound) {
      return make_unexpected(std::move(rebound).error());
    }
  }
  for (const auto& subscriber : impl_->subscribers_) {
    if (subscriber->owner == owner || subscriber->source_owner == owner) {
      subscriber->clear_samples();
      ++subscriber->producer_epoch;
    }
  }
  for (const auto& input : impl_->inputs_) {
    if (input->reuse_slot != nullptr && !input->reuse_slot->occupied) {
      input->reuse_slot = nullptr;
    }
    input->retirement_slots.clear();
  }
  for (const auto& input : impl_->inputs_) {
    if (input->owner == owner) {
      ++input->active_generation;
      ++input->view_generation;
      input->binding.generation = input->active_generation;
      input->type_ops->bind_input(input->declaration, &input->binding);
      input->active_selection = {InputState::kNotSelected,
                                 SelectionReason::kNotSelectedForActivation};
      input->selection_prepared = false;
      input->staged_now = false;
    }
  }
  for (const auto& output : impl_->outputs_) {
    if (output->owner == owner) {
      ++output->active_generation;
      output->next_sequence = 1U;
      output->emitted_this_activation = 0U;
    }
  }
  return {};
}

expected<bool, Error> RuntimeChannelSession::restart_affects(
    std::span<const VertexId> restarted_owners,
    std::span<const VertexId> affected_receivers) const noexcept {
  if (!impl_ || !unique_valid_vertices(restarted_owners) ||
      !unique_valid_vertices(affected_receivers)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto contains = [](std::span<const VertexId> vertices, VertexId owner) {
    return std::find(vertices.begin(), vertices.end(), owner) != vertices.end();
  };
  return std::any_of(
             impl_->outputs_.begin(),
             impl_->outputs_.end(),
             [&](const auto& output) { return contains(restarted_owners, output->owner); }) ||
         std::any_of(
             impl_->inputs_.begin(),
             impl_->inputs_.end(),
             [&](const auto& input) { return contains(affected_receivers, input->owner); }) ||
         std::any_of(
             impl_->subscribers_.begin(), impl_->subscribers_.end(), [&](const auto& subscriber) {
               return contains(restarted_owners, subscriber->source_owner) ||
                      contains(affected_receivers, subscriber->owner);
             });
}

// Restart commit validates the complete lifecycle closure before crossing the transport discard
// boundary, then publishes every affected generation through an infallible commit suffix.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeChannelSession::commit_restart(
    std::span<const VertexId> restarted_owners,
    std::span<const VertexId> affected_receivers) noexcept {
  if (!impl_ || !unique_valid_vertices(restarted_owners) ||
      !unique_valid_vertices(affected_receivers)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
#if !defined(NDEBUG)
  Impl::ControlMutationGuard control_guard{impl_->controller_mutation_barrier_};
#endif
  auto current = impl_->status();
  if (!current) {
    return make_unexpected(std::move(current).error());
  }
  if (impl_->pump_thread_.joinable() || impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  const auto receiver_affected = [&](VertexId owner) {
    return std::find(affected_receivers.begin(), affected_receivers.end(), owner) !=
           affected_receivers.end();
  };
  const auto owner_restarted = [&](VertexId owner) {
    return std::find(restarted_owners.begin(), restarted_owners.end(), owner) !=
           restarted_owners.end();
  };
  if (std::any_of(restarted_owners.begin(), restarted_owners.end(), [&](VertexId owner) {
        return !receiver_affected(owner);
      })) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  bool found = false;
  for (const auto& input : impl_->inputs_) {
    if (!receiver_affected(input->owner)) {
      continue;
    }
    found = true;
    if (input->active || input->predicate_view_constructed ||
        input->claim_pool->has_live_claims()) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
  }
  for (const auto& output : impl_->outputs_) {
    if (!owner_restarted(output->owner)) {
      continue;
    }
    found = true;
    if (output->active || std::any_of(output->loans.begin(),
                                      output->loans.end(),
                                      [](const auto* loan) { return loan->live; })) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
  }
  for (const auto& subscriber : impl_->subscribers_) {
    if (!owner_restarted(subscriber->source_owner) && !receiver_affected(subscriber->owner)) {
      continue;
    }
    found = true;
    if (subscriber->catalog_tensor_import) {
      // Restart cannot publish a new producer epoch while the old typed catalog attachment remains
      // live. Fail before transport discard, health reset, or retained-sample mutation.
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    if (subscriber->input->active || subscriber->input->claim_pool->has_live_claims()) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
  }
  for (const auto& subscriber : impl_->subscribers_) {
    if (!subscriber->externally_driven_import ||
        (!owner_restarted(subscriber->source_owner) && !receiver_affected(subscriber->owner))) {
      continue;
    }
    found = true;
    // Generation-fenced rebind gate; see reset() for the typed outcome contract.
    const std::scoped_lock lock{subscriber->mutex};
    auto gated = impl_->import_rebind_gate(*subscriber);
    if (!gated) {
      return make_unexpected(std::move(gated).error());
    }
  }
  if (!found) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }

  for (const auto& input : impl_->inputs_) {
    if (receiver_affected(input->owner) &&
        (input->active_generation == std::numeric_limits<std::uint64_t>::max() ||
         input->view_generation == std::numeric_limits<std::uint64_t>::max())) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
  }
  for (const auto& output : impl_->outputs_) {
    if (owner_restarted(output->owner) &&
        output->active_generation == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
  }
  for (const auto& subscriber : impl_->subscribers_) {
    if ((owner_restarted(subscriber->source_owner) || receiver_affected(subscriber->owner)) &&
        subscriber->producer_epoch == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
  }

  // The lifecycle fence joined the only process-local ingress pump and fenced the plan-owned
  // publisher before entering this method. Consume at most the compiler-frozen transport depth
  // so old-generation data cannot survive the epoch publication. A transport error or excess
  // prefix terminalizes this session: the partially drained old attachment cannot be rolled back
  // or re-armed as if its pre-restart state were intact.
  std::optional<CudaDeviceCursor> discard_device;
  const bool affected_descriptor_import =
      std::ranges::any_of(impl_->subscribers_, [&](const auto& subscriber) {
        return !subscriber->externally_driven_import &&
               subscriber->uses_device_affine_descriptor_import() &&
               (owner_restarted(subscriber->source_owner) || receiver_affected(subscriber->owner));
      });
  if (affected_descriptor_import) {
    if (!impl_->initialize_controller_descriptor_import()) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    discard_device.emplace();
  }
  bool transport_consumed = false;
  for (const auto& subscriber : impl_->subscribers_) {
    if (subscriber->externally_driven_import ||
        (!owner_restarted(subscriber->source_owner) && !receiver_affected(subscriber->owner))) {
      // An externally driven import is not discardable process-local transport; its old
      // attachment is replaced wholesale by the generation-fenced rebind below.
      continue;
    }
    auto discarded = subscriber->discard_process_local_transport(
        discard_device.has_value() ? &*discard_device : nullptr);
    if (!discarded) {
      Error error = std::move(discarded).error();
      impl_->fail(error.code);
      return make_unexpected(std::move(error));
    }
    transport_consumed = transport_consumed || *discarded != 0U;
  }
  if (discard_device.has_value() && !discard_device->restore()) {
    impl_->fail(ErrorCode::kNotSupported);
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }

  auto reset_health = impl_->reset_clock_health([&](const auto& subscriber) {
    return owner_restarted(subscriber.source_owner) || receiver_affected(subscriber.owner);
  });
  if (reset_health && impl_->restart_post_discard_failure_for_test_ != ErrorCode::kSuccess) {
    const ErrorCode injected =
        std::exchange(impl_->restart_post_discard_failure_for_test_, ErrorCode::kSuccess);
    reset_health = make_unexpected(Error{injected});
  }
  if (!reset_health) {
    Error error = std::move(reset_health).error();
    if (transport_consumed) {
      // The consumed old-generation transport prefix cannot be restored. A later health reset
      // failure therefore crosses the rollback boundary and must leave the session terminal.
      impl_->fail(error.code);
    }
    return make_unexpected(std::move(error));
  }

  for (const auto& subscriber : impl_->subscribers_) {
    if (!subscriber->externally_driven_import ||
        (!owner_restarted(subscriber->source_owner) && !receiver_affected(subscriber->owner))) {
      continue;
    }
    // Atomic detach+rebind under the fence; a destructive-window failure terminalized the session.
    const std::scoped_lock lock{subscriber->mutex};
    auto rebound = impl_->rebind_import(*subscriber);
    if (!rebound) {
      return make_unexpected(std::move(rebound).error());
    }
  }
  // With the pump joined and the bounded old transport prefix discarded above, all remaining state
  // changes are infallible and each affected subscriber is cleared exactly once before publishing
  // any new source generation.
  for (const auto& subscriber : impl_->subscribers_) {
    if (!owner_restarted(subscriber->source_owner) && !receiver_affected(subscriber->owner)) {
      continue;
    }
    subscriber->clear_samples();
    ++subscriber->producer_epoch;
  }
  for (const auto& input : impl_->inputs_) {
    if (!receiver_affected(input->owner)) {
      continue;
    }
    input->reuse_slot = nullptr;
    input->retirement_slots.clear();
    ++input->active_generation;
    ++input->view_generation;
    input->binding.generation = input->active_generation;
    input->type_ops->bind_input(input->declaration, &input->binding);
    input->active_selection = {InputState::kNotSelected,
                               SelectionReason::kNotSelectedForActivation};
    input->selection_prepared = false;
    input->staged_now = false;
  }
  for (const auto& output : impl_->outputs_) {
    if (!owner_restarted(output->owner)) {
      continue;
    }
    ++output->active_generation;
    output->next_sequence = 1U;
    output->emitted_this_activation = 0U;
  }
  return {};
}

expected<void, Error> RuntimeChannelSession::status() const noexcept {
  return impl_ ? impl_->status()
               : expected<void, Error>{make_unexpected(Error{ErrorCode::kClosed})};
}

std::optional<StructuralDiagnosticId> RuntimeChannelSession::terminal_diagnostic() const noexcept {
  if (!impl_) {
    return std::nullopt;
  }
  if (impl_->terminal_error_.load(std::memory_order_acquire) == ErrorCode::kSuccess) {
    return std::nullopt;
  }
  const std::uint32_t diagnostic = impl_->terminal_diagnostic_.load(std::memory_order_relaxed);
  return diagnostic == 0U ? std::nullopt
                          : std::optional{static_cast<StructuralDiagnosticId>(diagnostic)};
}

VertexId RuntimeChannelSession::terminal_owner() const noexcept {
  if (!impl_ || impl_->terminal_error_.load(std::memory_order_acquire) == ErrorCode::kSuccess) {
    return {};
  }
  return VertexId{impl_->terminal_owner_.load(std::memory_order_relaxed)};
}

expected<void, Error> RuntimeChannelSessionTestAccess::replace_output_publishers(
    RuntimeChannelSession& session, const PortId output,
    std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers,
    const std::span<const std::uint32_t> expected_recipient_counts,
    const bool refresh_before_emit) noexcept {
  if (!session.impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return session.impl_->replace_output_publishers_for_test(
      output, std::move(publishers), expected_recipient_counts, refresh_before_emit);
}

expected<void, Error> RuntimeChannelSessionTestAccess::throw_tensor_fanout_after_accepts(
    RuntimeChannelSession& session, const PortId output,
    const std::size_t accepted_count) noexcept {
  if (!session.impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return session.impl_->throw_tensor_fanout_after_accepts_for_test(output, accepted_count);
}

expected<void, Error> RuntimeChannelSessionTestAccess::replace_input_subscriber(
    RuntimeChannelSession& session, const PortId input,
    std::unique_ptr<channel_internal::InlinePayloadSubscriber> replacement) noexcept {
  if (!session.impl_ || !input || replacement == nullptr ||
      session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  auto& state = **subscriber;
  const std::scoped_lock lock{state.transport_mutex, state.mutex};
  if (state.endpoint == nullptr || state.tensor_endpoint ||
      state.tensor_schema_endpoint != nullptr || state.externally_driven_import ||
      state.wait_handle == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto detached = state.endpoint->detach_wait_set();
  if (!detached) {
    return make_unexpected(std::move(detached).error());
  }
  const std::size_t expected_erasures = state.terminal_wait_handle == 0U ? 1U : 2U;
  std::size_t erased = session.impl_->wait_subscriber_by_handle_.erase(state.wait_handle);
  if (state.terminal_wait_handle != 0U) {
    erased += session.impl_->wait_subscriber_by_handle_.erase(state.terminal_wait_handle);
  }
  if (erased != expected_erasures || session.impl_->wait_attachment_count_ < expected_erasures) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  session.impl_->wait_attachment_count_ -= expected_erasures;
  state.wait_handle = 0U;
  state.terminal_wait_handle = 0U;
  state.endpoint = std::move(replacement);
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::receive_one(RuntimeChannelSession& session,
                                                                   const PortId input) noexcept {
  if (!session.impl_ || !input || session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  return session.impl_->receive_one(**subscriber);
}

expected<std::size_t, Error> RuntimeChannelSessionTestAccess::typed_ingress_reservations(
    const RuntimeChannelSession& session, const PortId input) noexcept {
  if (!session.impl_ || !input) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  return static_cast<std::size_t>(
      std::count_if((*subscriber)->slots.begin(), (*subscriber)->slots.end(), [](const auto& slot) {
        return slot.typed_ingress_reserved;
      }));
}

expected<void, Error> RuntimeChannelSessionTestAccess::reclaim_reserved_input_slot(
    RuntimeChannelSession& session, const PortId input) noexcept {
  if (!session.impl_ || !input) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  RuntimeChannelSession::Impl::SampleSlot* reserved = nullptr;
  for (auto& slot : (*subscriber)->slots) {
    if (!slot.typed_ingress_reserved) {
      continue;
    }
    if (reserved != nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    reserved = &slot;
  }
  if (reserved == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  RuntimeChannelSession::Impl::reclaim_sample_slot_if_unclaimed(*reserved);
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_typed_ingress_observer(
    RuntimeChannelSession& session, const PortId input, void* const state,
    void (*const observer)(void*) noexcept) noexcept {
  if (!session.impl_ || !input || session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable() || (state == nullptr) != (observer == nullptr)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->transport_mutex, (*subscriber)->mutex};
  (*subscriber)->typed_ingress_observer_state = state;
  (*subscriber)->typed_ingress_observer = observer;
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::drain_external_inputs(
    RuntimeChannelSession& session, const VertexId owner) noexcept {
  if (!session.impl_ || !owner || session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const RuntimeChannelSession::Impl::ActivationOwnerState* const activation_owner =
      session.impl_->find_activation_owner(owner);
  if (activation_owner == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  return session.impl_->drain_monitored_inputs(*activation_owner);
}

expected<void, Error> RuntimeChannelSessionTestAccess::store_end_of_stream(
    RuntimeChannelSession& session, const PortId input,
    const std::uint64_t transport_sequence) noexcept {
  if (!session.impl_ || !input || session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  return session.impl_->store_end_of_stream(**subscriber, transport_sequence);
}

expected<void, Error>
RuntimeChannelSessionTestAccess::corrupt_next_process_local_schema_publication(
    RuntimeChannelSession& session, const PortId output,
    const channel_internal::CatalogSchemaFrameCorruption corruption) noexcept {
  if (!session.impl_ || !output || session.impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto found = std::find_if(session.impl_->outputs_.begin(),
                                  session.impl_->outputs_.end(),
                                  [&](const auto& candidate) { return candidate->port == output; });
  if (found == session.impl_->outputs_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  auto& state = **found;
  if (state.active || !state.process_local_schema_identity ||
      state.next_process_local_schema_corruption) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  state.next_process_local_schema_corruption = corruption;
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_readiness_sequence(
    PortBinding& binding, const std::uint64_t sequence) noexcept {
  auto* const readiness = dynamic_cast<RuntimeChannelSession::Impl::ReadinessBinding*>(&binding);
  if (readiness == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  readiness->next_source_sequence_ = sequence;
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_single_occupied_witness_state(
    RuntimeChannelSession& session, const PortId input, const std::uint64_t readiness_sequence,
    const bool event_emitted, const bool witness_retired) noexcept {
  if (!session.impl_ || !input || readiness_sequence == 0U || (witness_retired && !event_emitted)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  RuntimeChannelSession::Impl::SampleSlot* occupied = nullptr;
  for (auto& slot : (*subscriber)->slots) {
    if (!slot.occupied || slot.direct_ingress_unread) {
      continue;
    }
    if (occupied != nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    occupied = &slot;
  }
  if (occupied == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  occupied->readiness_sequence = readiness_sequence;
  occupied->event_emitted = event_emitted;
  occupied->witness_retired = witness_retired;
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_pre_witness_replacement_allowed(
    RuntimeChannelSession& session, const PortId input, const bool allowed) noexcept {
  if (!session.impl_ || !input || session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  (*subscriber)->pre_witness_replacement_allowed = allowed;
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_single_occupied_claim_state(
    RuntimeChannelSession& session, const PortId input, const bool active_selected,
    const bool reuse_retained, const bool temporal_retained,
    const bool retirement_selected) noexcept {
  if (!session.impl_ || !input) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  RuntimeChannelSession::Impl::SampleSlot* occupied = nullptr;
  for (auto& slot : (*subscriber)->slots) {
    if (!slot.occupied || slot.direct_ingress_unread) {
      continue;
    }
    if (occupied != nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    occupied = &slot;
  }
  if (occupied == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  occupied->active_selected = active_selected;
  occupied->reuse_retained = reuse_retained;
  occupied->temporal_retained = temporal_retained;
  occupied->retirement_selected = retirement_selected;
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::inject_direct_inline_payload(
    RuntimeChannelSession& session, const PortId input, const std::span<const std::byte> payload,
    SampleMetadata metadata) noexcept {
  if (!session.impl_ || !input || payload.empty() ||
      session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  return session.impl_->store_direct_inline_payload(**subscriber, payload, std::move(metadata));
}

expected<void, Error> RuntimeChannelSessionTestAccess::with_input_lock(
    RuntimeChannelSession& session, const PortId input, void* state,
    const RuntimeChannelInputLockObserver observer) noexcept {
  if (!session.impl_ || !input || observer == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  observer(state);
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_poll_selection_observer(
    RuntimeChannelSession& session, void* state,
    const RuntimeChannelPollSelectionObserver observer) noexcept {
  if (!session.impl_ || state == nullptr || observer == nullptr ||
      session.impl_->poll_selection_observer_ != nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  session.impl_->poll_selection_observer_state_ = state;
  session.impl_->poll_selection_observer_ = observer;
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_teardown_observer(
    RuntimeChannelSession& session, void* state,
    const RuntimeChannelTeardownObserver observer) noexcept {
  if (!session.impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return session.impl_->set_teardown_observer_for_test(state, observer);
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_clock_health_lock_observer(
    RuntimeChannelSession& session, void* state,
    const RuntimeChannelClockHealthLockObserver observer) noexcept {
  if (!session.impl_ || state == nullptr || observer == nullptr ||
      session.impl_->clock_health_serialization_mutex_ == nullptr ||
      session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable() ||
      session.impl_->clock_health_lock_observer_ != nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  session.impl_->clock_health_lock_observer_state_ = state;
  session.impl_->clock_health_lock_observer_ = observer;
  return {};
}

bool RuntimeChannelSessionTestAccess::clock_health_serialization_bound(
    const RuntimeChannelSession& session) noexcept {
  return session.impl_ != nullptr && session.impl_->clock_health_serialization_mutex_ != nullptr;
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_input_reset_counters(
    RuntimeChannelSession& session, const PortId input, const std::uint64_t active_generation,
    const std::uint64_t view_generation, const std::uint64_t producer_epoch) noexcept {
  if (!session.impl_ || !input || active_generation == 0U || view_generation == 0U ||
      producer_epoch == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto* input_state = session.impl_->find_input(input);
  if (input_state == nullptr || input_state->active || input_state->predicate_view_constructed) {
    return make_unexpected(
        Error{input_state == nullptr ? ErrorCode::kNotFound : ErrorCode::kBackpressured});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  input_state->active_generation = active_generation;
  input_state->view_generation = view_generation;
  (*subscriber)->producer_epoch = producer_epoch;
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_arrival_counters(
    RuntimeChannelSession& session, const PortId input, const std::uint64_t next_arrival_sequence,
    const std::uint64_t next_global_arrival_order) noexcept {
  if (!session.impl_ || !input || next_arrival_sequence == 0U || next_global_arrival_order == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto* input_state = session.impl_->find_input(input);
  if (input_state == nullptr || input_state->active || input_state->predicate_view_constructed ||
      session.impl_->pump_thread_.joinable() ||
      session.impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(
        Error{input_state == nullptr ? ErrorCode::kNotFound : ErrorCode::kBackpressured});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  {
    const std::scoped_lock lock{(*subscriber)->mutex};
    (*subscriber)->next_arrival_sequence = next_arrival_sequence;
  }
  session.impl_->next_arrival_order_.store(next_global_arrival_order, std::memory_order_release);
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_process_local_progress_counts(
    RuntimeChannelSession& session, const std::uint64_t published,
    const std::uint64_t received) noexcept {
  if (!session.impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  session.impl_->process_local_published_deliveries_.store(published, std::memory_order_release);
  session.impl_->process_local_received_deliveries_.store(received, std::memory_order_release);
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::inject_received_metadata(
    RuntimeChannelSession& session, const PortId input, const SampleMetadata metadata,
    std::optional<StructuralDiagnosticId>* const terminal_diagnostic) noexcept {
  if (!session.impl_ || !input || session.impl_->pump_thread_.joinable() ||
      session.impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  const auto scratch = (*subscriber)->scratch_bytes();
  std::fill(scratch.begin(), scratch.end(), std::byte{});
  return session.impl_->store_received(**subscriber,
                                       metadata,
                                       EndpointClockHealthDisposition::kNotApplicable,
                                       0U,
                                       {},
                                       terminal_diagnostic);
}

expected<void, Error> RuntimeChannelSessionTestAccess::inject_received_tensor(
    RuntimeChannelSession& session, const PortId input, Sample<Tensor> sample) noexcept {
  if (!session.impl_ || !input || session.impl_->pump_thread_.joinable() ||
      session.impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  if ((*subscriber)->input->type_ops != runtime_port_type_ops<Tensor>()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  return session.impl_->store_received_tensor(**subscriber, std::move(sample));
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_external_tensor_contract(
    RuntimeChannelSession& session, const PortId input,
    LogicalTensorRequirementsRecord requirements) noexcept {
  if (!session.impl_ || !input || !requirements.max_byte_span ||
      *requirements.max_byte_span == 0U || requirements.memory_kind != MemoryKind::kHost ||
      requirements.strides != TensorStridePolicy::kContiguousOnly ||
      session.impl_->pump_thread_.joinable() ||
      session.impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  if ((*subscriber)->input->type_ops != runtime_port_type_ops<Tensor>()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  if (std::any_of((*subscriber)->slots.begin(), (*subscriber)->slots.end(), [](const auto& slot) {
        return slot.occupied;
      })) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  (*subscriber)->externally_driven_import = false;
  (*subscriber)->catalog_tensor_import = true;
  (*subscriber)->tensor_max_byte_span = *requirements.max_byte_span;
  (*subscriber)->tensor_memory_kind = MemoryKind::kHost;
  (*subscriber)->tensor_device = {};
  (*subscriber)->tensor_logical_requirements = std::move(requirements);
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_external_metadata_contract(
    RuntimeChannelSession& session, const PortId input, const ClockId catalog_capture_clock,
    const ClockId plan_capture_clock) noexcept {
  if (!session.impl_ || !input || catalog_capture_clock.valid() != plan_capture_clock.valid() ||
      session.impl_->pump_thread_.joinable() ||
      session.impl_->armed_.load(std::memory_order_acquire)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  if (std::any_of((*subscriber)->slots.begin(), (*subscriber)->slots.end(), [](const auto& slot) {
        return slot.occupied;
      })) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  if ((*subscriber)->input->capture_time_required && !catalog_capture_clock.valid()) {
    // Production admission supplies a concrete catalog/plan clock pair whenever a selector
    // requires capture time. Keep this cold test seam from manufacturing an unreachable plan.
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  (*subscriber)->externally_driven_import = true;
  (*subscriber)->catalog_capture_clock = catalog_capture_clock;
  (*subscriber)->plan_capture_clock = plan_capture_clock;
  return {};
}

expected<RuntimeChannelInputResetState, Error> RuntimeChannelSessionTestAccess::input_reset_state(
    const RuntimeChannelSession& session, const PortId input) noexcept {
  if (!session.impl_ || !input) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto* input_state = session.impl_->find_input(input);
  if (input_state == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::scoped_lock lock{(*subscriber)->mutex};
  std::uint64_t latched_producer_generation = 0U;
  if ((*subscriber)->endpoint != nullptr) {
    auto latched = channel_internal::CatalogChannelBindingAccess::latched_producer_generation(
        *(*subscriber)->endpoint);
    if (latched) {
      latched_producer_generation = *latched;
    }
  }
  return RuntimeChannelInputResetState{
      .active_generation = input_state->active_generation,
      .view_generation = input_state->view_generation,
      .producer_epoch = (*subscriber)->producer_epoch,
      .next_arrival_sequence = (*subscriber)->next_arrival_sequence,
      .next_global_arrival_order =
          session.impl_->next_arrival_order_.load(std::memory_order_acquire),
      .occupied_samples = static_cast<std::size_t>(std::count_if(
          (*subscriber)->slots.begin(),
          (*subscriber)->slots.end(),
          [](const auto& slot) { return slot.occupied && !slot.direct_ingress_unread; })),
      .pending_direct_ingress_samples = static_cast<std::size_t>(std::count_if(
          (*subscriber)->slots.begin(),
          (*subscriber)->slots.end(),
          [](const auto& slot) { return slot.occupied && slot.direct_ingress_unread; })),
      .catalog_capture_clock = (*subscriber)->catalog_capture_clock,
      .plan_capture_clock = (*subscriber)->plan_capture_clock,
      .latched_producer_generation = latched_producer_generation,
      .end_of_stream_received = (*subscriber)->end_of_stream_received,
  };
}

expected<RuntimeChannelPumpIndexState, Error> RuntimeChannelSessionTestAccess::pump_index_state(
    const RuntimeChannelSession& session) noexcept {
  if (!session.impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  const bool wait_index_exact =
      session.impl_->wait_subscriber_by_handle_.size() == session.impl_->wait_attachment_count_ &&
      std::all_of(session.impl_->wait_subscriber_by_handle_.begin(),
                  session.impl_->wait_subscriber_by_handle_.end(),
                  [](const auto& indexed) {
                    return indexed.first != 0U && indexed.second != nullptr &&
                           (indexed.second->wait_handle == indexed.first ||
                            indexed.second->terminal_wait_handle == indexed.first);
                  });
  const auto drain_eligible = std::count_if(session.impl_->subscribers_.begin(),
                                            session.impl_->subscribers_.end(),
                                            [](const auto& subscriber) {
                                              return !subscriber->direct_process_local &&
                                                     !subscriber->clock_health_endpoint &&
                                                     !subscriber->externally_driven_import;
                                            });
  const bool drain_index_exact =
      session.impl_->pump_subscribers_.size() == static_cast<std::size_t>(drain_eligible) &&
      std::all_of(session.impl_->pump_subscribers_.begin(),
                  session.impl_->pump_subscribers_.end(),
                  [](const auto* subscriber) {
                    return subscriber != nullptr && !subscriber->direct_process_local &&
                           !subscriber->clock_health_endpoint &&
                           !subscriber->externally_driven_import;
                  });
  return RuntimeChannelPumpIndexState{
      .wait_handle_count = session.impl_->wait_subscriber_by_handle_.size(),
      .drain_subscriber_count = session.impl_->pump_subscribers_.size(),
      .exact = wait_index_exact && drain_index_exact,
  };
}

expected<void, Error> RuntimeChannelSessionTestAccess::initialize_controller_descriptor_import(
    RuntimeChannelSession& session, const PortId input, const DeviceId device) noexcept {
  auto configured = configure_descriptor_import_device(session, input, device);
  if (!configured) {
    return make_unexpected(std::move(configured).error());
  }
  if (!session.impl_->initialize_controller_descriptor_import()) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  return {};
}

expected<void, Error> RuntimeChannelSessionTestAccess::configure_descriptor_import_device(
    RuntimeChannelSession& session, const PortId input, const DeviceId device) noexcept {
  if (!session.impl_ || !input || !device.valid() ||
      session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->pump_thread_.joinable()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto subscriber =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (subscriber == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  if (!(**subscriber).tensor_endpoint.has_value() ||
      (**subscriber).tensor_schema_endpoint != nullptr || (**subscriber).externally_driven_import) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const bool device_changed = (**subscriber).tensor_device != device;
  (**subscriber).tensor_device = device;
  if (device_changed) {
    session.impl_->controller_descriptor_import_initialized_ = false;
    session.impl_->controller_descriptor_import_thread_ = {};
  }
  try {
    session.impl_->freeze_descriptor_import_inventory();
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  return {};
}

expected<std::uint64_t, Error>
RuntimeChannelSessionTestAccess::controller_descriptor_import_initialization_count(
    const RuntimeChannelSession& session) noexcept {
  if (!session.impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return session.impl_->controller_descriptor_import_initialization_count_;
}

expected<DeviceId, Error> RuntimeChannelSessionTestAccess::cuda_device_after_import_selection(
    const DeviceId selected, const DeviceId fallback) noexcept {
  if (!selected.valid()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  CudaDeviceCursor cursor;
  if (!cursor.select(selected) || !cursor.select(fallback)) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  int observed = -1;
  if (cudaGetDevice(&observed) != cudaSuccess) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  return DeviceId{observed};
}

expected<RuntimeChannelTensorTransferState, Error>
RuntimeChannelSessionTestAccess::tensor_transfer_state(const RuntimeChannelSession& session,
                                                       const PortId output,
                                                       const PortId input) noexcept {
  if (!session.impl_ || !output || !input) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto found_output = std::find_if(
      session.impl_->outputs_.begin(), session.impl_->outputs_.end(), [&](const auto& candidate) {
        return candidate->port == output;
      });
  const auto found_input =
      std::find_if(session.impl_->subscribers_.begin(),
                   session.impl_->subscribers_.end(),
                   [&](const auto& candidate) { return candidate->target == input; });
  if (found_output == session.impl_->outputs_.end() ||
      found_input == session.impl_->subscribers_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  if (!(*found_output)->tensor_payload() ||
      (*found_input)->input->type_ops != runtime_port_type_ops<Tensor>()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return RuntimeChannelTensorTransferState{
      .buffer_handle_publishers = (*found_output)->tensor_endpoints.size(),
      .inline_publishers = (*found_output)->endpoints.size(),
      .buffer_handle_subscriber = (*found_input)->tensor_endpoint.has_value(),
      .inline_subscriber = (*found_input)->endpoint != nullptr,
  };
}

expected<void, Error> RuntimeChannelSessionTestAccess::set_output_active_generation(
    RuntimeChannelSession& session, const PortId output,
    const std::uint64_t active_generation) noexcept {
  if (!session.impl_ || !output || active_generation == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto found = std::find_if(session.impl_->outputs_.begin(),
                                  session.impl_->outputs_.end(),
                                  [&](const auto& candidate) { return candidate->port == output; });
  if (found == session.impl_->outputs_.end() || (*found)->active) {
    return make_unexpected(Error{
        found == session.impl_->outputs_.end() ? ErrorCode::kNotFound : ErrorCode::kBackpressured});
  }
  (*found)->active_generation = active_generation;
  return {};
}

expected<std::uint64_t, Error> RuntimeChannelSessionTestAccess::output_active_generation(
    const RuntimeChannelSession& session, const PortId output) noexcept {
  if (!session.impl_ || !output) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto found = std::find_if(session.impl_->outputs_.begin(),
                                  session.impl_->outputs_.end(),
                                  [&](const auto& candidate) { return candidate->port == output; });
  if (found == session.impl_->outputs_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  return (*found)->active_generation;
}

expected<void, Error> RuntimeChannelSessionTestAccess::fail_next_restart_after_transport_discard(
    RuntimeChannelSession& session, const ErrorCode code) noexcept {
  if (!session.impl_ || code == ErrorCode::kSuccess) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto current = session.impl_->status();
  if (!current) {
    return make_unexpected(std::move(current).error());
  }
  if (session.impl_->pump_thread_.joinable() ||
      session.impl_->armed_.load(std::memory_order_acquire) ||
      session.impl_->restart_post_discard_failure_for_test_ != ErrorCode::kSuccess) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  // One-shot fault injection exercises the otherwise backend-dependent transition between
  // irreversible process-local transport discard and generation publication.
  session.impl_->restart_post_discard_failure_for_test_ = code;
  return {};
}

}  // namespace holoscan::detail
