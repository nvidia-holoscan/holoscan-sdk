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

#include "runtime_rate_state.hpp"

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
  kEventMaterializationFailure,
  kCommitReadinessFailure,
  kPostSubmitIndeterminate,
  kEmitClosedBinding,
  kRestartPublicationPreflightFailure,
  kRateReuseCompatibilityFailure,
  kPostSubmitPartial,
};

/// Select one source-tree-only internal-identity corruption.
enum class RuntimeTensorOutputTestCorruption : std::uint8_t {
  kNone,
  kPlanGeneration,
  kPoolGeneration,
  kSlotGeneration,
};

/// Classify one source-tree-private direct Tensor backing decision.
enum class RuntimeTensorPoolAllocationAction : std::uint8_t {
  kFresh = 0,
  kReusedExact = 1,
  kReusedLarger = 2,
  kReplaced = 3,
};

/// Add materialization-owned fields without discarding evidence populated by another provider.
[[nodiscard]] inline RuntimeProviderMaterializationResult with_provider_materialization(
    RuntimeProviderMaterializationResult existing, ProviderAllocationPlanRecordId provider,
    AllocationAction action, std::uint32_t materialized_entries) noexcept {
  existing.provider = provider;
  existing.action = action;
  existing.materialized_entries = materialized_entries;
  existing.reported = true;
  return existing;
}

/// Coordinate deterministic private faults and fixed claim snapshots for source-tree tests.
class RuntimeTensorOutputTestControl final {
 public:
  /// Retain the latest bounded checked-diagnostic observation for precedence tests.
  struct DiagnosticSnapshot {
    TensorOutputDiagnosticRecord record{};
    std::size_t pool_index{};
    std::uint64_t count{};
  };

  /// Retain the fixed primitive metrics required before cumulative HSDK5-017 aggregation.
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
    bool failed_loan_active_limited{};
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
    std::uint64_t maximum_published_pool_generation{};
    std::uint64_t materialization_count{};
    // These counters describe only the test-observed lifetime of OutputPool materializations.
    // Requested payload bytes exclude provider metadata, alignment, and reservation overhead.
    std::uint64_t live_materialization_count{};
    std::uint64_t peak_live_materialization_count{};
    std::uint64_t live_physical_entry_count{};
    std::uint64_t peak_live_physical_entry_count{};
    std::uint64_t live_requested_payload_backing_bytes{};
    std::uint64_t peak_live_requested_payload_backing_bytes{};
    std::uint64_t maximum_allocation_generation{};
    std::uint32_t physical_slot_count{};
    std::uint32_t active_entry_limit{};
    RuntimeTensorPoolAllocationAction allocation_action{RuntimeTensorPoolAllocationAction::kFresh};
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
    std::uint64_t cuda_stream_quiesce_ns{};
    std::int32_t holomq_generation_vmm_device_ordinal{-1};
    std::uint32_t minimum_retired_slot{std::numeric_limits<std::uint32_t>::max()};
    bool terminal{};
  };

  struct ServiceCopySnapshot {
    cudaStream_t producer_stream{};
    cudaStream_t service_stream{};
    std::uint64_t count{};
  };

  /// Prepare exact cold observation storage before any output pool is materialized.
  ///
  /// Caller-installed one-shot faults, identity corruption, and metric time are
  /// configuration for the pending materialization. They remain armed until
  /// consumed or explicitly replaced.
  void prepare(std::size_t pool_count) {
    const std::lock_guard lock(mutex_);
    pools_.assign(pool_count, PoolSnapshot{});
    last_diagnostic_ = {};
    service_copy_ = {};
  }

  void arm_fault(RuntimeTensorOutputTestFault fault) noexcept {
    fault_pool_index_.store(kAnyFaultPool, std::memory_order_relaxed);
    fault_.store(fault, std::memory_order_release);
  }

  void arm_fault_for_pool(RuntimeTensorOutputTestFault fault, std::size_t pool_index) noexcept {
    fault_pool_index_.store(pool_index, std::memory_order_relaxed);
    fault_.store(fault, std::memory_order_release);
  }

  [[nodiscard]] bool consume_fault(RuntimeTensorOutputTestFault fault) noexcept {
    // Acquire the published fault before inspecting the selector stored before the arming release.
    if (fault_.load(std::memory_order_acquire) != fault ||
        fault_pool_index_.load(std::memory_order_relaxed) != kAnyFaultPool) {
      return false;
    }
    return consume_matching_fault(fault);
  }

  [[nodiscard]] bool consume_fault(RuntimeTensorOutputTestFault fault,
                                   std::size_t pool_index) noexcept {
    // The acquire pairs with arm_fault_for_pool() before this relaxed selector read.
    if (fault_.load(std::memory_order_acquire) != fault) {
      return false;
    }
    const std::size_t selected_pool = fault_pool_index_.load(std::memory_order_relaxed);
    if (selected_pool != kAnyFaultPool && selected_pool != pool_index) {
      return false;
    }
    return consume_matching_fault(fault);
  }

 private:
  static constexpr std::size_t kAnyFaultPool = std::numeric_limits<std::size_t>::max();

  [[nodiscard]] bool consume_matching_fault(RuntimeTensorOutputTestFault fault) noexcept {
    RuntimeTensorOutputTestFault expected = fault;
    const bool consumed = fault_.compare_exchange_strong(expected,
                                                         RuntimeTensorOutputTestFault::kNone,
                                                         std::memory_order_acq_rel,
                                                         std::memory_order_acquire);
    if (consumed) {
      fault_pool_index_.store(kAnyFaultPool, std::memory_order_release);
    }
    return consumed;
  }

 public:
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
              std::uint64_t pool_generation, std::uint64_t allocation_generation,
              std::uint32_t physical_slot_count, std::uint32_t active_entry_limit,
              RuntimeTensorPoolAllocationAction allocation_action, bool published,
              const PrimitiveMetrics& metrics, bool invariant_failure = false,
              bool terminal = false) noexcept {
    const std::lock_guard lock(mutex_);
    if (pool_index >= pools_.size()) {
      return;
    }
    PoolSnapshot& pool = pools_[pool_index];
    pool.admitted_claim_mask = admitted_claim_mask;
    pool.observed_claim_mask |= active_claim_mask;
    pool.active_claim_mask = terminal ? 0U : active_claim_mask;
    pool.active_slot_count = terminal ? 0U : active_slot_count;
    pool.maximum_allocation_generation =
        std::max(pool.maximum_allocation_generation, allocation_generation);
    if (pool_generation != 0U) {
      pool.minimum_pool_generation = std::min(pool.minimum_pool_generation, pool_generation);
      pool.maximum_pool_generation = std::max(pool.maximum_pool_generation, pool_generation);
      if (published) {
        if (pool_generation >= pool.maximum_published_pool_generation) {
          pool.physical_slot_count = physical_slot_count;
          pool.active_entry_limit = active_entry_limit;
          pool.allocation_action = allocation_action;
        }
        pool.maximum_published_pool_generation =
            std::max(pool.maximum_published_pool_generation, pool_generation);
      }
    }
    pool.metrics = metrics;
    pool.invariant_failure_count += static_cast<std::size_t>(invariant_failure);
    pool.terminal = pool.terminal || terminal;
  }

  void record_materialization(std::size_t pool_index, std::uint64_t allocation_generation,
                              std::uint32_t physical_slot_count, std::uint32_t active_entry_limit,
                              std::uint64_t checked_requested_payload_backing_bytes) noexcept {
    const std::lock_guard lock(mutex_);
    if (pool_index >= pools_.size()) {
      return;
    }
    PoolSnapshot& pool = pools_[pool_index];
    ++pool.materialization_count;
    ++pool.live_materialization_count;
    pool.peak_live_materialization_count =
        std::max(pool.peak_live_materialization_count, pool.live_materialization_count);
    const std::uint64_t physical_entries = physical_slot_count;
    pool.live_physical_entry_count += physical_entries;
    pool.peak_live_physical_entry_count =
        std::max(pool.peak_live_physical_entry_count, pool.live_physical_entry_count);
    pool.live_requested_payload_backing_bytes += checked_requested_payload_backing_bytes;
    pool.peak_live_requested_payload_backing_bytes = std::max(
        pool.peak_live_requested_payload_backing_bytes, pool.live_requested_payload_backing_bytes);
    pool.maximum_allocation_generation =
        std::max(pool.maximum_allocation_generation, allocation_generation);
    pool.physical_slot_count = physical_slot_count;
    pool.active_entry_limit = active_entry_limit;
  }

  void release_materialization(std::size_t pool_index, std::uint32_t physical_slot_count,
                               std::uint64_t checked_requested_payload_backing_bytes) noexcept {
    const std::lock_guard lock(mutex_);
    if (pool_index >= pools_.size()) {
      return;
    }
    PoolSnapshot& pool = pools_[pool_index];
    const std::uint64_t physical_entries = physical_slot_count;
    pool.live_materialization_count =
        pool.live_materialization_count == 0U ? 0U : pool.live_materialization_count - 1U;
    pool.live_physical_entry_count = pool.live_physical_entry_count < physical_entries
                                         ? 0U
                                         : pool.live_physical_entry_count - physical_entries;
    pool.live_requested_payload_backing_bytes =
        pool.live_requested_payload_backing_bytes < checked_requested_payload_backing_bytes
            ? 0U
            : pool.live_requested_payload_backing_bytes - checked_requested_payload_backing_bytes;
  }

  [[nodiscard]] PoolSnapshot snapshot(std::size_t pool_index) const noexcept {
    const std::lock_guard lock(mutex_);
    return pool_index < pools_.size() ? pools_[pool_index] : PoolSnapshot{};
  }

  void record_diagnostic(std::size_t pool_index, TensorOutputDiagnosticRecord record) noexcept {
    const std::lock_guard lock(mutex_);
    last_diagnostic_.record = record;
    last_diagnostic_.pool_index = pool_index;
    ++last_diagnostic_.count;
  }

  [[nodiscard]] DiagnosticSnapshot last_diagnostic() const noexcept {
    const std::lock_guard lock(mutex_);
    return last_diagnostic_;
  }

  void record_cuda_service_copy(cudaStream_t producer_stream,
                                cudaStream_t service_stream) noexcept {
    const std::lock_guard lock(mutex_);
    service_copy_.producer_stream = producer_stream;
    service_copy_.service_stream = service_stream;
    ++service_copy_.count;
  }

  [[nodiscard]] ServiceCopySnapshot service_copy_snapshot() const noexcept {
    const std::lock_guard lock(mutex_);
    return service_copy_;
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

  void record_cuda_stream_quiesce(std::size_t pool_index, std::uint64_t duration_ns) noexcept {
    update(pool_index,
           [duration_ns](PoolSnapshot& pool) { pool.cuda_stream_quiesce_ns += duration_ns; });
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
  void update(std::size_t pool_index, Update update_snapshot) noexcept {
    const std::lock_guard lock(mutex_);
    if (pool_index < pools_.size()) {
      update_snapshot(pools_[pool_index]);
    }
  }

  mutable std::mutex mutex_;
  std::vector<PoolSnapshot> pools_;
  DiagnosticSnapshot last_diagnostic_;
  ServiceCopySnapshot service_copy_;
  std::atomic<RuntimeTensorOutputTestFault> fault_{RuntimeTensorOutputTestFault::kNone};
  std::atomic<std::size_t> fault_pool_index_{kAnyFaultPool};
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
class RuntimeCudaExecutionResources;

/// Describes the fixed address of a one-slot Tensor RuntimePool.
struct RuntimeFixedTensorAddress {
  void* data{};            ///< Address returned for every allocation from this pool.
  std::size_t capacity{};  ///< Maximum Tensor byte span supported by the address.
  MemoryKind memory_kind{MemoryKind::kUnknown};  ///< Memory location of the address.
  DeviceId device{};  ///< CUDA device for device memory; invalid for ordinary host memory.
};

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
      RunGenerationId run_generation, ProcessGroupId process_group,
      const RuntimeCudaExecutionResources& cuda_resources,
      RuntimeRateMaterializationView materialization = {});

  /// Return the single active slot used by a CUDA Graph boundary or internal edge.
  [[nodiscard]] expected<RuntimeFixedTensorAddress, Error> fixed_output_address(
      PortId output) noexcept;

  /// Return whether the fixed output slot can be acquired without consuming it.
  /// Completed Tensor retirements are reconciled before reporting availability.
  [[nodiscard]] expected<bool, Error> fixed_output_available(PortId output) noexcept;

  /// Restore the ordinary RuntimePool binding after temporary capture-time port binding.
  [[nodiscard]] bool restore_output_binding(PortId output) noexcept;

  /// Reject new Tensor authority and tombstone outstanding unpublished slots.
  void close() noexcept;

  /// Check whether one logical Operator may consume Channel terminal authority.
  ///
  /// Returns false while an unpublished Tensor output loan is still live, or an error when an
  /// owner pool is quarantined. Published backing, receiver claims, and qualified device work are
  /// deliberately excluded because they retire only after terminal delivery.
  [[nodiscard]] expected<bool, Error> preflight_owner_terminal_close(
      OperatorInstanceId owner) noexcept;

  /// Advance nonblocking retirement for every Tensor output owned by one logical Operator.
  ///
  /// Returns true only after every active and restart-retired pool for the owner has released all
  /// loans, Tensor handles, and qualified CUDA work, or an error if a pool is quarantined. The
  /// Runtime owner loop calls this after Channel terminal publication while deciding whether
  /// natural drain may commit.
  [[nodiscard]] expected<bool, Error> advance_owner_terminal_retirement(
      OperatorInstanceId owner) noexcept;

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

  /// Apply one lifecycle-selected logical limit to already-materialized startup backing.
  ///
  /// This operation is allocation-free and may run at kAllocate, kWarmUp, or kArm. A zero
  /// unresolved provider capacity is left pending for its later declared binding barrier.
  [[nodiscard]] expected<void, Error> prepare_rate_candidate_startup(
      RuntimeRateMaterializationView materialization) noexcept;

  /// Materialize lifecycle-selected replacement geometry after the selected rate barrier.
  ///
  /// The predecessor pools remain fenced and rollback-authoritative until publish_restart().
  [[nodiscard]] expected<void, Error> prepare_rate_candidate_restart(
      RuntimeRateMaterializationView materialization) noexcept;

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
