/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/operator_ref.hpp>

namespace holoscan::detail {

inline constexpr std::size_t kTensorSlotClaimReasonCount =
    static_cast<std::size_t>(TensorSlotClaimReason::kRetirementService) + 1U;

/// Map plan-owned storage to its canonical DLPack device metadata.
[[nodiscard]] constexpr DLDevice runtime_tensor_dl_device(MemoryKind kind,
                                                          DeviceId device) noexcept {
  switch (kind) {
    case MemoryKind::kHost:
      return DLDevice{kDLCPU, 0};
    case MemoryKind::kPinnedHost:
      return DLDevice{kDLCUDAHost, 0};
    case MemoryKind::kCudaDevice:
      return DLDevice{kDLCUDA, device.value};
    case MemoryKind::kCudaManaged:
      return DLDevice{kDLCUDAManaged, 0};
    case MemoryKind::kUnknown:
      break;
  }
  return DLDevice{kDLCPU, 0};
}

/// Identify one checked direct-output transition or misuse reason.
enum class TensorOutputDiagnosticReason : std::uint8_t {
  kLoanOutputMismatch,
  kLoanPlanGenerationMismatch,
  kLoanActivationGenerationMismatch,
  kLoanPoolGenerationMismatch,
  kLoanSlotGenerationMismatch,
  kLoanWriterActive,
  kLoanReadinessNotCommitted,
  kAllocationCapabilityMissing,
  kGuardDroppedWithoutCommit,
  kLoanDestroyedWithActiveWriter,
  kGuardOutlivedLoan,
  kGuardDoubleCommit,
  kLoanDoubleEmit,
  kLoanDoubleCancel,
  kLoanUseAfterMove,
  kGuardUseAfterMove,
  kLoanInvalidStateTransition,
  kLoanRequestLayoutMismatch,
  kLoanStreamDeviceMismatch,
  kAllocationReadinessRecordExhausted,
  kLoanReadinessRecordFailed,
  kLoanCancelSafetyRecordFailed,
  kPublicationIndeterminate,
  kLoanRetirementStalled,
  kLoanReleaseEvidenceMissing,
  kPublicationPartial,
};

inline constexpr std::size_t kTensorOutputDiagnosticReasonCount =
    static_cast<std::size_t>(TensorOutputDiagnosticReason::kPublicationPartial) + 1U;

/// Return the exact stable checked-diagnostic token for one reason.
[[nodiscard]] inline std::string_view tensor_output_diagnostic_reason_name(
    TensorOutputDiagnosticReason reason) noexcept {
  static constexpr std::array<std::string_view, kTensorOutputDiagnosticReasonCount> kNames{
      "TENSOR_OUTPUT_LOAN_OUTPUT_MISMATCH",
      "TENSOR_OUTPUT_LOAN_PLAN_GENERATION_MISMATCH",
      "TENSOR_OUTPUT_LOAN_ACTIVATION_GENERATION_MISMATCH",
      "TENSOR_OUTPUT_LOAN_POOL_GENERATION_MISMATCH",
      "TENSOR_OUTPUT_LOAN_SLOT_GENERATION_MISMATCH",
      "TENSOR_OUTPUT_LOAN_WRITER_ACTIVE",
      "TENSOR_OUTPUT_LOAN_READINESS_NOT_COMMITTED",
      "TENSOR_OUTPUT_ALLOCATION_CAPABILITY_MISSING",
      "TENSOR_OUTPUT_GUARD_DROPPED_WITHOUT_COMMIT",
      "TENSOR_OUTPUT_LOAN_DESTROYED_WITH_ACTIVE_WRITER",
      "TENSOR_OUTPUT_GUARD_OUTLIVED_LOAN",
      "TENSOR_OUTPUT_GUARD_DOUBLE_COMMIT",
      "TENSOR_OUTPUT_LOAN_DOUBLE_EMIT",
      "TENSOR_OUTPUT_LOAN_DOUBLE_CANCEL",
      "TENSOR_OUTPUT_LOAN_USE_AFTER_MOVE",
      "TENSOR_OUTPUT_GUARD_USE_AFTER_MOVE",
      "TENSOR_OUTPUT_LOAN_INVALID_STATE_TRANSITION",
      "TENSOR_OUTPUT_LOAN_REQUEST_LAYOUT_MISMATCH",
      "TENSOR_OUTPUT_LOAN_STREAM_DEVICE_MISMATCH",
      "TENSOR_OUTPUT_ALLOCATION_READINESS_RECORD_EXHAUSTED",
      "TENSOR_OUTPUT_LOAN_READINESS_RECORD_FAILED",
      "TENSOR_OUTPUT_LOAN_CANCEL_SAFETY_RECORD_FAILED",
      "TENSOR_OUTPUT_PUBLICATION_INDETERMINATE",
      "TENSOR_OUTPUT_LOAN_RETIREMENT_STALLED",
      "TENSOR_OUTPUT_LOAN_RELEASE_EVIDENCE_MISSING",
      "TENSOR_OUTPUT_PUBLICATION_PARTIAL",
  };
  const std::size_t index = static_cast<std::size_t>(reason);
  return index < kNames.size() ? kNames[index] : std::string_view{};
}

/// Retain one pointer-free direct-output transition record.
struct TensorOutputDiagnosticRecord {
  PortId output{};
  PlanGenerationId plan_generation{};
  RunGenerationId run_generation{};
  Tag activation{};
  std::uint64_t pool_generation{};
  std::uint64_t slot_generation{};
  std::uint32_t slot{};
  std::uint8_t from_state{};
  std::uint8_t attempted_operation{};
  std::uint8_t writer_count{};
  ErrorCode error{ErrorCode::kSuccess};
  TensorOutputDiagnosticReason reason{TensorOutputDiagnosticReason::kLoanInvalidStateTransition};
};

static_assert(std::is_standard_layout_v<TensorOutputDiagnosticRecord>);
static_assert(std::is_trivially_copyable_v<TensorOutputDiagnosticRecord>);

/// Append to caller-provided fixed storage without allocation, growth, or replacement.
class TensorOutputDiagnosticBuffer final {
 public:
  void reset(std::span<TensorOutputDiagnosticRecord> storage) noexcept {
    storage_ = storage;
    count_ = 0U;
  }

  void append(TensorOutputDiagnosticRecord record) noexcept {
    if (count_ < storage_.size()) {
      storage_[count_++] = record;
    }
  }

  [[nodiscard]] std::span<const TensorOutputDiagnosticRecord> records() const noexcept {
    return storage_.first(count_);
  }

 private:
  std::span<TensorOutputDiagnosticRecord> storage_;
  std::size_t count_{};
};

/// Select one source-tree-only direct-output terminal fault.
///
/// This private seam exists only for deterministic conformance tests. Production compilation
/// never records or selects one of these values, and each armed fault is consumed at most once.
enum class RuntimeTensorOutputTestFault : std::uint8_t {
  kNone,
  kCommitReadinessFailure,
  kPostSubmitIndeterminate,
  kEmitClosedBinding,
  kRestartPublicationPreflightFailure,
  kPostSubmitPartial,
};

/// Select one source-tree-only internal-identity corruption.
enum class RuntimeTensorOutputTestCorruption : std::uint8_t {
  kNone,
  kPlanGeneration,
  kPoolGeneration,
  kSlotGeneration,
};

/// Coordinate deterministic private faults and fixed claim snapshots for source-tree tests.
class RuntimeTensorOutputTestControl final {
 public:
  /// Retain the latest bounded checked-diagnostic observation for precedence tests.
  struct DiagnosticSnapshot {
    TensorOutputDiagnosticRecord record{};
    std::size_t pool_index{};
    std::uint64_t count{};
  };

  /// Retain the fixed primitive metrics required before cumulative EA1-017 aggregation.
  struct PrimitiveMetrics {
    std::uint32_t physical_free_slot_depth{};
    std::uint32_t publisher_loan_credit{};
    std::uint32_t effective_free_slot_depth{};
    std::uint32_t minimum_physical_free_slot_depth{};
    std::uint32_t minimum_publisher_loan_credit{};
    std::uint32_t minimum_effective_free_slot_depth{};
    std::uint64_t zero_depth_total_ns{};
    std::uint64_t zero_depth_longest_ns{};
    std::uint64_t zero_depth_since_ns{std::numeric_limits<std::uint64_t>::max()};
    std::uint64_t zero_depth_interval_count{};
    std::uint64_t failed_loan_count{};
    std::uint32_t failed_loan_physical_free_slot_depth{};
    std::uint32_t failed_loan_publisher_credit{};
    std::uint32_t failed_loan_effective_free_slot_depth{};
    bool failed_loan_physical_limited{};
    bool failed_loan_credit_limited{};
    std::uint32_t failed_loan_blocking_claim_mask{};
    std::uint32_t failed_loan_oldest_retained_slot{std::numeric_limits<std::uint32_t>::max()};
    std::uint8_t failed_loan_oldest_retained_phase{};
    std::uint32_t failed_loan_oldest_retained_claim_mask{};
    std::uint64_t failed_loan_oldest_retained_age_ns{};
    std::uint32_t oldest_retained_slot{std::numeric_limits<std::uint32_t>::max()};
    std::uint8_t oldest_retained_phase{};
    std::uint32_t oldest_retained_claim_mask{};
    std::uint64_t oldest_retained_age_ns{};
    std::uint64_t retry_cancellation_total_started{};
    std::uint32_t current_retry_cancellation_demand{};
    std::uint32_t peak_retry_cancellation_demand{};
  };

  struct PoolSnapshot {
    std::uint32_t admitted_claim_mask{};
    std::uint32_t observed_claim_mask{};
    std::uint32_t active_claim_mask{};
    std::size_t active_slot_count{};
    std::size_t invariant_failure_count{};
    std::uint64_t minimum_pool_generation{std::numeric_limits<std::uint64_t>::max()};
    std::uint64_t maximum_pool_generation{};
    PrimitiveMetrics metrics;
    std::uint64_t cuda_streams_created{};
    std::uint64_t cuda_streams_destroyed{};
    std::uint64_t cuda_events_created{};
    std::uint64_t cuda_events_destroyed{};
    std::uint64_t cuda_readiness_records{};
    std::uint64_t cuda_readiness_record_elisions{};
    std::uint64_t cuda_release_stream_assignments{};
    std::uint64_t cuda_release_stream_elisions{};
    std::uint64_t cuda_release_records{};
    std::uint64_t cuda_stream_waits{};
    std::uint64_t cuda_stream_wait_elisions{};
    std::uint64_t cuda_host_waits{};
    std::uint64_t cuda_retirement_queries{};
    std::uint64_t cuda_retirement_completions{};
    std::int32_t holomq_generation_vmm_device_ordinal{-1};
    std::uint32_t minimum_retired_slot{std::numeric_limits<std::uint32_t>::max()};
    bool terminal{};
  };

  /// Prepare exact cold observation storage before any output pool is materialized.
  ///
  /// Caller-installed one-shot faults, identity corruption, and metric time are
  /// configuration for the pending materialization. They remain armed until
  /// consumed or explicitly replaced.
  void prepare(std::size_t pool_count) {
    std::lock_guard lock(mutex_);
    pools_.assign(pool_count, PoolSnapshot{});
    last_diagnostic_ = {};
  }

  void arm_fault(RuntimeTensorOutputTestFault fault) noexcept {
    fault_.store(fault, std::memory_order_release);
  }

  [[nodiscard]] bool consume_fault(RuntimeTensorOutputTestFault fault) noexcept {
    RuntimeTensorOutputTestFault expected = fault;
    return fault_.compare_exchange_strong(expected,
                                          RuntimeTensorOutputTestFault::kNone,
                                          std::memory_order_acq_rel,
                                          std::memory_order_acquire);
  }

  void arm_corruption(RuntimeTensorOutputTestCorruption corruption) noexcept {
    corruption_.store(corruption, std::memory_order_release);
  }

  [[nodiscard]] RuntimeTensorOutputTestCorruption consume_corruption() noexcept {
    return corruption_.exchange(RuntimeTensorOutputTestCorruption::kNone,
                                std::memory_order_acq_rel);
  }

  void set_metric_time_ns(std::uint64_t value) noexcept {
    metric_time_ns_.store(value, std::memory_order_release);
  }

  [[nodiscard]] std::uint64_t metric_time_ns() const noexcept {
    return metric_time_ns_.load(std::memory_order_acquire);
  }

  void record(std::size_t pool_index, std::uint32_t admitted_claim_mask,
              std::uint32_t active_claim_mask, std::size_t active_slot_count,
              std::uint64_t pool_generation, const PrimitiveMetrics& metrics,
              bool invariant_failure = false, bool terminal = false) noexcept {
    std::lock_guard lock(mutex_);
    if (pool_index >= pools_.size()) {
      return;
    }
    PoolSnapshot& pool = pools_[pool_index];
    pool.admitted_claim_mask = admitted_claim_mask;
    pool.observed_claim_mask |= active_claim_mask;
    pool.active_claim_mask = terminal ? 0U : active_claim_mask;
    pool.active_slot_count = terminal ? 0U : active_slot_count;
    if (pool_generation != 0U) {
      pool.minimum_pool_generation = std::min(pool.minimum_pool_generation, pool_generation);
      pool.maximum_pool_generation = std::max(pool.maximum_pool_generation, pool_generation);
    }
    pool.metrics = metrics;
    pool.invariant_failure_count += static_cast<std::size_t>(invariant_failure);
    pool.terminal = pool.terminal || terminal;
  }

  [[nodiscard]] PoolSnapshot snapshot(std::size_t pool_index) const noexcept {
    std::lock_guard lock(mutex_);
    return pool_index < pools_.size() ? pools_[pool_index] : PoolSnapshot{};
  }

  void record_diagnostic(std::size_t pool_index, TensorOutputDiagnosticRecord record) noexcept {
    std::lock_guard lock(mutex_);
    last_diagnostic_.record = record;
    last_diagnostic_.pool_index = pool_index;
    ++last_diagnostic_.count;
  }

  [[nodiscard]] DiagnosticSnapshot last_diagnostic() const noexcept {
    std::lock_guard lock(mutex_);
    return last_diagnostic_;
  }

  void record_cuda_events_created(std::size_t pool_index, std::uint32_t count) noexcept {
    update(pool_index, [count](PoolSnapshot& pool) { pool.cuda_events_created += count; });
  }

  void record_cuda_stream_created(std::size_t pool_index) noexcept {
    update(pool_index, [](PoolSnapshot& pool) { ++pool.cuda_streams_created; });
  }

  void record_cuda_stream_destroyed(std::size_t pool_index) noexcept {
    update(pool_index, [](PoolSnapshot& pool) { ++pool.cuda_streams_destroyed; });
  }

  void record_cuda_event_destroyed(std::size_t pool_index) noexcept {
    update(pool_index, [](PoolSnapshot& pool) { ++pool.cuda_events_destroyed; });
  }

  void record_cuda_readiness(std::size_t pool_index) noexcept {
    update(pool_index, [](PoolSnapshot& pool) { ++pool.cuda_readiness_records; });
  }

  void record_cuda_readiness_elision(std::size_t pool_index) noexcept {
    update(pool_index, [](PoolSnapshot& pool) { ++pool.cuda_readiness_record_elisions; });
  }

  void record_cuda_release_stream(std::size_t pool_index) noexcept {
    update(pool_index, [](PoolSnapshot& pool) { ++pool.cuda_release_stream_assignments; });
  }

  void record_cuda_release_stream_elision(std::size_t pool_index) noexcept {
    update(pool_index, [](PoolSnapshot& pool) { ++pool.cuda_release_stream_elisions; });
  }

  void record_cuda_release(std::size_t pool_index) noexcept {
    update(pool_index, [](PoolSnapshot& pool) { ++pool.cuda_release_records; });
  }

  void record_cuda_stream_wait(std::size_t pool_index) noexcept {
    update(pool_index, [](PoolSnapshot& pool) { ++pool.cuda_stream_waits; });
  }

  void record_cuda_stream_wait_elision(std::size_t pool_index) noexcept {
    update(pool_index, [](PoolSnapshot& pool) { ++pool.cuda_stream_wait_elisions; });
  }

  void record_cuda_host_wait(std::size_t pool_index) noexcept {
    update(pool_index, [](PoolSnapshot& pool) { ++pool.cuda_host_waits; });
  }

  void record_cuda_retirement_query(std::size_t pool_index, bool complete) noexcept {
    update(pool_index, [complete](PoolSnapshot& pool) {
      ++pool.cuda_retirement_queries;
      pool.cuda_retirement_completions += static_cast<std::uint64_t>(complete);
    });
  }

  void record_holomq_generation_vmm_device(std::size_t pool_index,
                                           std::int32_t device_ordinal) noexcept {
    update(pool_index, [device_ordinal](PoolSnapshot& pool) {
      pool.holomq_generation_vmm_device_ordinal = device_ordinal;
    });
  }

  void record_retired_slot(std::size_t pool_index, std::uint32_t slot) noexcept {
    update(pool_index, [slot](PoolSnapshot& pool) {
      pool.minimum_retired_slot = std::min(pool.minimum_retired_slot, slot);
    });
  }

 private:
  template <typename Update>
  void update(std::size_t pool_index, Update&& update_snapshot) noexcept {
    std::lock_guard lock(mutex_);
    if (pool_index < pools_.size()) {
      update_snapshot(pools_[pool_index]);
    }
  }

  mutable std::mutex mutex_;
  std::vector<PoolSnapshot> pools_;
  DiagnosticSnapshot last_diagnostic_;
  std::atomic<RuntimeTensorOutputTestFault> fault_{RuntimeTensorOutputTestFault::kNone};
  std::atomic<RuntimeTensorOutputTestCorruption> corruption_{
      RuntimeTensorOutputTestCorruption::kNone};
  std::atomic<std::uint64_t> metric_time_ns_{std::numeric_limits<std::uint64_t>::max()};
};

/// Keep the closed canonical claim vocabulary for one physical Tensor slot.
class TensorSlotClaimLedger final {
 public:
  using Admission = std::array<bool, kTensorSlotClaimReasonCount>;

  [[nodiscard]] bool add(TensorSlotClaimReason reason, const Admission& admitted) noexcept {
    const std::size_t index = static_cast<std::size_t>(reason);
    if (index >= entries_.size() || !admitted[index]) {
      return false;
    }
    entries_[index] = true;
    return true;
  }

  void remove(TensorSlotClaimReason reason) noexcept {
    const std::size_t index = static_cast<std::size_t>(reason);
    if (index < entries_.size()) {
      entries_[index] = false;
    }
  }

  void clear() noexcept { entries_.fill(false); }

  [[nodiscard]] bool contains(TensorSlotClaimReason reason) const noexcept {
    const std::size_t index = static_cast<std::size_t>(reason);
    return index < entries_.size() && entries_[index];
  }

  [[nodiscard]] bool any() const noexcept {
    for (const bool entry : entries_) {
      if (entry) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] std::size_t active_count() const noexcept {
    std::size_t count = 0U;
    for (const bool entry : entries_) {
      count += static_cast<std::size_t>(entry);
    }
    return count;
  }

  [[nodiscard]] std::uint32_t mask() const noexcept {
    std::uint32_t result = 0U;
    for (std::size_t index = 0U; index < entries_.size(); ++index) {
      result |= static_cast<std::uint32_t>(entries_[index]) << index;
    }
    return result;
  }

 private:
  std::array<bool, kTensorSlotClaimReasonCount> entries_{};
};

/// Advance one physical slot generation without allowing zero or wraparound reuse.
[[nodiscard]] inline bool advance_tensor_slot_generation(std::uint64_t& generation) noexcept {
  if (generation == 0U || generation == std::numeric_limits<std::uint64_t>::max()) {
    return false;
  }
  ++generation;
  return true;
}

struct ExecutionPlanState;
struct RuntimeOperatorInstance;

/// Own the physically materialized direct-Tensor resources for one RunSession.
///
/// Construction is cold and consumes only accepted immutable plan records. Every allocation,
/// control record, CUDA event, and process/device stream is created before lifecycle arm. The
/// callback path visits only deferred or actively retiring slots, copies metadata into pre-reserved
/// arrays, and performs nonblocking CUDA event operations.
class RuntimeTensorResources final {
 public:
  RuntimeTensorResources(const RuntimeTensorResources&) = delete;
  RuntimeTensorResources& operator=(const RuntimeTensorResources&) = delete;
  RuntimeTensorResources(RuntimeTensorResources&&) = delete;
  RuntimeTensorResources& operator=(RuntimeTensorResources&&) = delete;
  ~RuntimeTensorResources();

  /// Materialize and bind every accepted direct Tensor output for one fresh run.
  [[nodiscard]] static expected<std::unique_ptr<RuntimeTensorResources>, Error> create(
      const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
      RunGenerationId run_generation, ProcessGroupId process_group);

  /// Reject new Tensor authority and tombstone outstanding unpublished slots.
  void close() noexcept;

  /// Remove public port bindings while reconstructed Operators are still alive.
  void detach_ports() noexcept;

  /// Fence unpublished direct-output authority before lifecycle restart callbacks run.
  ///
  /// The member span is the compiler-frozen lifecycle transaction closure. A restart from
  /// `kAllocate` or an earlier stage constructs replacement pools while retaining the old pools
  /// for rollback and already-published handles. Later-stage restart retains the physical pool but
  /// still tombstones every unpublished loan/writer from the predecessor activation epoch.
  [[nodiscard]] expected<void, Error> begin_restart(std::span<const OperatorInstanceId> members,
                                                    LifecycleStage from_stage) noexcept;

  /// Verify every prepared pool before any restart participant publishes successor state.
  [[nodiscard]] expected<void, Error> preflight_restart_publication() noexcept;

  /// Publish the candidate pool generation after lifecycle restart succeeds.
  [[nodiscard]] expected<void, Error> publish_restart() noexcept;

  /// Restore retained predecessor pools after lifecycle forward rollback succeeds.
  [[nodiscard]] expected<void, Error> rollback_restart() noexcept;

  [[nodiscard]] std::size_t device_resource_count() const noexcept;
  [[nodiscard]] std::size_t output_pool_count() const noexcept;

 private:
  class Impl;
  explicit RuntimeTensorResources(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::detail
