/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <holoscan/core/expected.hpp>
#include <holoscan/core/lifecycle.hpp>

namespace holoscan::detail {

/// Private origin classification used by the lifecycle transaction kernel.
enum class LifecycleTransactionRequestOrigin : std::uint8_t {
  kParameter,
  kExternalControl,
  kComputeSelf,
  kComputeCrossTarget,
};

/// Immutable compatibility identity already admitted by the compiler.
///
/// Equal values may coalesce. Different values must remain in separate
/// transactions because authorization, profile, rollback, deadline, or closure
/// constraints differ.
struct LifecycleTransactionCompatibilityKey {
  std::uint64_t authorization{};
  std::uint64_t profile{};
  std::uint64_t rollback{};
  std::uint64_t deadline{};
  std::uint64_t closure{};

  friend constexpr bool operator==(LifecycleTransactionCompatibilityKey,
                                   LifecycleTransactionCompatibilityKey) noexcept = default;
};

/// Normalized target state observed at the acceptance safe point.
enum class LifecycleTransactionTargetState : std::uint8_t {
  kRunning,
  kReady,
  kIdle,
  kFailedStopped,
  kStarting,
  kStopping,
  kStopped,
  kTerminal,
};

/// Executor acknowledgement required before the transaction may quiesce.
enum class LifecycleTransactionGate : std::uint8_t {
  kActivationRetired = 0,
  kReadyActivationCanceled = 1,
  kControlSafePoint = 2,
  kFailedStoppedNormalized = 3,
};

/// Exact finite storage supplied by the admitted execution plan.
struct LifecycleTransactionKernelCapacity {
  std::size_t max_pending_requests{};
  std::size_t max_pending_transactions{};
  std::size_t max_receipts{};
  std::size_t max_fenced_operators{};
  std::size_t max_candidate_generations{};
  std::size_t max_events{};
  std::size_t max_audit_records{};
  std::size_t max_quiesce_records{};
  std::size_t max_rollback_records{};
};

/// One compiler-normalized request. The closure must be nonempty, strictly
/// increasing, and contain the target operator.
struct LifecycleTransactionRequest {
  LifecycleTransactionRequestOrigin origin{LifecycleTransactionRequestOrigin::kParameter};
  OperatorRef target{};
  LifecycleStage requested_stage{LifecycleStage::kConfigure};
  LifecycleStage effective_stage{LifecycleStage::kConfigure};
  LifecycleTransactionCompatibilityKey compatibility{};
  std::span<const OperatorInstanceId> closure;
  ParamGeneration candidate_generation{};
  bool parameter_candidate_equivalent{};
  LifecycleTransactionTargetState target_state{LifecycleTransactionTargetState::kIdle};
  std::size_t structural_event_delta{};  ///< Compiler-derived worst-case retained event records.
};

enum class LifecycleTransactionCapacityKind : std::uint8_t {
  kNone,
  kPendingRequests,
  kPendingTransactions,
  kReceipts,
  kFencedOperators,
  kCandidateGenerations,
  kEvents,
  kAuditRecords,
  kQuiesceRecords,
  kRollbackRecords,
};

enum class LifecycleTransactionKernelErrorCode : std::uint8_t {
  kInvalidRequest,
  kInvalidRestartStage,
  kCapacityExceeded,
  kSuccessorUnavailable,
  kUnknownTransaction,
  kUnknownReceipt,
  kInvalidTransition,
  kTargetStateRejected,
  kGateUnsatisfied,
  kIngressClosed,
  kIdentityExhausted,
};

struct LifecycleTransactionKernelError {
  LifecycleTransactionKernelErrorCode code{LifecycleTransactionKernelErrorCode::kInvalidRequest};
  LifecycleTransactionCapacityKind capacity{LifecycleTransactionCapacityKind::kNone};
};

/// Immutable observation of one retained accepted receipt.
struct LifecycleTransactionReceiptView {
  LifecycleRestartReceipt receipt{};
  LifecycleTransactionState state{LifecycleTransactionState::kRejected};
  LifecycleTransactionId transaction_id{};
  LifecycleTransactionId coalesced_into_transaction{};
  LifecycleTransactionId related_transaction_id{};
  LifecycleFenceId fence_id{};
  LifecycleStage effective_stage{LifecycleStage::kConfigure};
  ParamGeneration candidate_generation{};
};

/// Pure bounded state machine used by the later RunSession executor.
///
/// Construction performs every dynamic allocation. All public transitions are
/// noexcept and reject before mutation when any finite reservation is
/// unavailable. This class owns no callbacks, threads, waits, graph objects, or
/// parameter payloads.
class LifecycleTransactionKernel {
 public:
  explicit LifecycleTransactionKernel(const LifecycleTransactionKernelCapacity& capacity);

  LifecycleTransactionKernel(const LifecycleTransactionKernel&) = delete;
  LifecycleTransactionKernel& operator=(const LifecycleTransactionKernel&) = delete;
  LifecycleTransactionKernel(LifecycleTransactionKernel&&) noexcept = default;
  LifecycleTransactionKernel& operator=(LifecycleTransactionKernel&&) noexcept = default;

  [[nodiscard]] expected<LifecycleRestartReceipt, LifecycleTransactionKernelError> submit(
      const LifecycleTransactionRequest& request) noexcept;

  [[nodiscard]] expected<void, LifecycleTransactionKernelError> begin_quiescing(
      LifecycleTransactionId transaction) noexcept;

  /// Record one executor-owned retirement/cancellation/safe-point acknowledgement.
  [[nodiscard]] expected<void, LifecycleTransactionKernelError> acknowledge_gate(
      LifecycleTransactionId transaction, OperatorInstanceId target,
      LifecycleTransactionGate gate) noexcept;

  [[nodiscard]] expected<void, LifecycleTransactionKernelError> begin_restarting(
      LifecycleTransactionId transaction) noexcept;

  [[nodiscard]] expected<void, LifecycleTransactionKernelError> publish(
      LifecycleTransactionId transaction) noexcept;

  [[nodiscard]] expected<void, LifecycleTransactionKernelError> rollback(
      LifecycleTransactionId transaction, bool restored_old_epoch) noexcept;

  /// Close ingress and cancel transaction intent while retaining safety fences.
  void request_stop(LifecycleTransactionId cause = 0U) noexcept;

  /// Release transaction storage and fences after ordinary stop cleanup completes.
  [[nodiscard]] expected<void, LifecycleTransactionKernelError> complete_stop_cleanup() noexcept;

  /// Retire one terminal observation record so its finite slot may be reused.
  [[nodiscard]] expected<void, LifecycleTransactionKernelError> retire_receipt(
      LifecycleRestartReceipt receipt) noexcept;

  [[nodiscard]] expected<LifecycleTransactionReceiptView, LifecycleTransactionKernelError> inspect(
      LifecycleRestartReceipt receipt) const noexcept;

  [[nodiscard]] expected<LifecycleTransactionState, LifecycleTransactionKernelError>
  transaction_state(LifecycleTransactionId transaction) const noexcept;

  [[nodiscard]] expected<ParamGeneration, LifecycleTransactionKernelError> candidate_generation(
      LifecycleTransactionId transaction, OperatorInstanceId owner) const noexcept;

  [[nodiscard]] expected<LifecycleStage, LifecycleTransactionKernelError> effective_stage(
      LifecycleTransactionId transaction) const noexcept;

  /// Return the frozen lifecycle-dependency union closure retained by one transaction.
  ///
  /// The returned span remains valid until the transaction reaches a terminal
  /// transition. Its contents may expand while coalescing remains open, so callers
  /// must close that window and begin quiescing before retaining it for execution.
  [[nodiscard]] expected<std::span<const OperatorInstanceId>, LifecycleTransactionKernelError>
  transaction_members(LifecycleTransactionId transaction) const noexcept;

  [[nodiscard]] bool is_fenced(OperatorInstanceId owner) const noexcept;
  [[nodiscard]] std::size_t pending_request_count() const noexcept { return pending_requests_; }
  [[nodiscard]] std::size_t pending_transaction_count() const noexcept {
    return pending_transactions_;
  }
  [[nodiscard]] std::size_t reserved_event_count() const noexcept { return reserved_events_; }
  [[nodiscard]] bool ingress_open() const noexcept { return ingress_open_; }
  [[nodiscard]] bool stop_requested() const noexcept { return stop_requested_; }

 private:
  struct CandidateRecord {
    OperatorInstanceId owner{};
    ParamGeneration generation{};
    std::uint64_t receipt_id{};
  };

  struct GateRecord {
    OperatorInstanceId target{};
    std::uint8_t required{};
    std::uint8_t satisfied{};
  };

  struct TransactionSlot {
    bool occupied{};
    LifecycleTransactionId id{};
    LifecycleTransactionState state{LifecycleTransactionState::kRejected};
    LifecycleFenceId fence_id{};
    LifecycleStage effective_stage{LifecycleStage::kConfigure};
    LifecycleTransactionCompatibilityKey compatibility{};
    bool has_explicit_intent{};
    std::vector<OperatorInstanceId> members;
    std::vector<CandidateRecord> candidates;
    std::vector<GateRecord> gates;
  };

  struct FenceRecord {
    OperatorInstanceId owner{};
    std::size_t holders{};
    bool failed_stopped{};
  };

  struct ReceiptRecord {
    LifecycleTransactionReceiptView view;
    std::size_t reserved_events{};
  };

  [[nodiscard]] TransactionSlot* find_transaction(LifecycleTransactionId transaction) noexcept;
  [[nodiscard]] const TransactionSlot* find_transaction(
      LifecycleTransactionId transaction) const noexcept;
  [[nodiscard]] ReceiptRecord* find_receipt(std::uint64_t receipt_id) noexcept;
  [[nodiscard]] const ReceiptRecord* find_receipt(std::uint64_t receipt_id) const noexcept;
  [[nodiscard]] FenceRecord* find_fence(OperatorInstanceId owner) noexcept;
  [[nodiscard]] const FenceRecord* find_fence(OperatorInstanceId owner) const noexcept;
  [[nodiscard]] TransactionSlot* find_coalescing_transaction(
      LifecycleTransactionCompatibilityKey compatibility) noexcept;
  [[nodiscard]] TransactionSlot* find_free_transaction_slot() noexcept;
  [[nodiscard]] bool request_valid(const LifecycleTransactionRequest& request) const noexcept;
  [[nodiscard]] bool is_head_transaction(LifecycleTransactionId transaction) const noexcept;
  void set_transaction_receipts_state(LifecycleTransactionId transaction,
                                      LifecycleTransactionState state) noexcept;
  void set_transaction_receipts_effective_stage(LifecycleTransactionId transaction,
                                                LifecycleStage stage) noexcept;
  void release_transaction(TransactionSlot& transaction, bool retain_failed_fence) noexcept;
  void terminalize_receipt(ReceiptRecord& receipt, LifecycleTransactionState state,
                           LifecycleTransactionId related = 0U) noexcept;

  LifecycleTransactionKernelCapacity capacity_;
  std::vector<TransactionSlot> transactions_;
  std::vector<FenceRecord> fences_;
  std::vector<ReceiptRecord> receipts_;
  std::size_t pending_requests_{};
  std::size_t pending_transactions_{};
  std::size_t candidate_generations_{};
  std::size_t reserved_events_{};
  std::size_t quiesce_records_{};
  std::size_t rollback_records_{};
  LifecycleTransactionId next_transaction_id_{1U};
  LifecycleFenceId next_fence_id_{1U};
  std::uint64_t next_receipt_id_{1U};
  std::uint64_t next_acceptance_sequence_{1U};
  bool ingress_open_{true};
  bool stop_requested_{};
};

}  // namespace holoscan::detail
