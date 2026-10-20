/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <span>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/operator_ref.hpp>

namespace holoscan {

namespace detail {
class LifecycleControlAccess;
class LifecycleControlHandleState;
}  // namespace detail

/// @brief Restrict lifecycle-control authority already admitted by a plan.
///
/// The target and stage spans are borrowed only for the duration of the mint
/// or restriction call. The returned handle copies and normalizes them. Empty
/// spans deny every target or stage respectively; a policy can never add
/// authority absent from the accepted execution plan.
struct LifecycleAccessPolicy {
  std::span<const OperatorRef> allowed_targets;         ///< Run-bound targets copied by the handle.
  std::span<const LifecycleStage> allowed_from_stages;  ///< Forward-stage subset copied by handle.
};

/// @brief Report the current state and lineage of one accepted restart receipt.
struct LifecycleRestartReceiptView {
  LifecycleRestartReceipt receipt{};  ///< Original immutable accepted identity.
  LifecycleTransactionState state{
      LifecycleTransactionState::kRejected};            ///< Current or terminal disposition.
  LifecycleTransactionId transaction_id{};              ///< Current transaction lineage root.
  LifecycleTransactionId coalesced_into_transaction{};  ///< Receiving transaction, or zero.
  LifecycleTransactionId related_transaction_id{};  ///< Predecessor/successor/stop cause, or zero.
  LifecycleFenceId fence_id{};                      ///< Acceptance-time activation-fence identity.
  LifecycleStage effective_stage{LifecycleStage::kConfigure};  ///< Compiler-resolved stage.
  ParamGeneration candidate_generation{};                      ///< Parameter candidate, or zero.
};

/// @brief Submit and observe bounded lifecycle restart transactions.
///
/// A handle is a restriction-only run capability. It retains a bounded
/// tombstone/control block, not the RunSession, ExecutionPlan, or any Operator
/// object. Copying a handle preserves exactly the same restriction authority,
/// allocates no memory, and does not broaden target or stage access. After
/// session retirement, submissions fail closed while already accepted receipts
/// remain inspectable until the final handle copy is destroyed.
///
/// Independent handle objects may be used concurrently. Concurrent mutation of
/// the same handle object by copy or move assignment requires caller
/// synchronization. The handle is a process-local opaque capability: its bytes
/// and internal address are not stable identities and must never be serialized,
/// transported, or persisted. Serialize an explicitly supported receipt or
/// diagnostic record instead.
class HOLOSCAN_CORE_EXPORT LifecycleControlHandle {
 public:
  /// @brief Create an empty handle that grants no lifecycle-control authority.
  LifecycleControlHandle() noexcept = default;

  /// @brief Retain the same lifecycle-control authority as another handle.
  /// @param other Handle to copy; an empty handle remains empty.
  /// @note This operation is nonblocking and performs no dynamic allocation.
  LifecycleControlHandle(const LifecycleControlHandle& other) noexcept;

  /// @brief Transfer one lifecycle-control capability without changing authority.
  /// @param other Handle to empty after transfer.
  /// @note This operation is nonblocking and performs no dynamic allocation.
  LifecycleControlHandle(LifecycleControlHandle&& other) noexcept;

  /// @brief Release this wrapper's retained lifecycle-control capability.
  /// @note Destruction never waits for a transaction or RunSession worker.
  ~LifecycleControlHandle() noexcept;

  /// @brief Replace this wrapper with the same authority as another handle.
  /// @param other Handle to copy; self-assignment has no effect.
  /// @return This handle.
  /// @note This operation is nonblocking and performs no dynamic allocation.
  LifecycleControlHandle& operator=(const LifecycleControlHandle& other) noexcept;

  /// @brief Replace this wrapper by transferring another handle's authority.
  /// @param other Handle to empty after transfer; self-assignment has no effect.
  /// @return This handle.
  /// @note This operation is nonblocking and performs no dynamic allocation.
  LifecycleControlHandle& operator=(LifecycleControlHandle&& other) noexcept;

  /// @brief Return a handle whose target and stage authority is no broader.
  /// @param policy Additional copied restrictions.
  /// @return Restricted handle, or a typed error without changing this handle.
  /// @retval ErrorCode::kClosed This handle is empty or lifecycle ingress has begun stopping.
  /// @retval ErrorCode::kCrossSessionOperatorRef A target is not bound to this run generation.
  /// @retval ErrorCode::kLifecycleInvalidRestartStage A requested stage is not a forward stage.
  /// @retval ErrorCode::kLifecycleRestartUnauthorized The policy exceeds this handle or the plan.
  /// @retval ErrorCode::kResourceExhausted The bounded policy copy could not be allocated.
  /// @note The returned handle owns its normalized restriction copy. The input spans
  /// are borrowed only for this call and may be released when the call returns.
  /// Restriction is a cold control-path operation serialized with submissions,
  /// receipt inspection, and final session cleanup by the run control mutex.
  [[nodiscard]] expected<LifecycleControlHandle, Error> restrict(
      const LifecycleAccessPolicy& policy) const noexcept;

  /// @brief Request restart of one generation-stamped same-session target.
  /// @param target Current-run target admitted by plan and handle policy.
  /// @param options Requested forward stage and registered bounded reason.
  /// @return Stable receipt when accepted; rejection changes no state.
  /// @retval ErrorCode::kLifecycleInvalidCallingContext Called from a Runtime callback context.
  /// @retval ErrorCode::kClosed This handle is empty or lifecycle ingress has begun stopping.
  /// @retval ErrorCode::kLifecycleTransactionsDisabled Restart transactions are unavailable.
  /// @retval ErrorCode::kLifecycleRunState The run is still starting or the target state rejects
  /// the request.
  /// @retval ErrorCode::kLifecycleInvalidRestartStage The requested stage is not a forward stage.
  /// @retval ErrorCode::kStaleOperatorRef The target is not bound to this accepted plan.
  /// @retval ErrorCode::kCrossSessionOperatorRef The target belongs to another run generation.
  /// @retval ErrorCode::kLifecycleRestartUnauthorized The target, stage, or external authority was
  /// not admitted by the plan or this handle.
  /// @retval ErrorCode::kLifecycleRestartQueueFull The bounded request or successor capacity is
  /// exhausted.
  /// @warning Do not call from compute, lifecycle callbacks, or Dispatcher code. Use
  /// \ref ExecutionContext::request_restart from compute instead.
  /// @note The call is nonblocking and does not dynamically allocate after the handle and
  /// RunSession have been constructed. Acceptance only queues the request and installs its
  /// activation fence. A target already computing finishes that callback before restart execution;
  /// an idle target restarts at the control safe point before its next compute callback.
  [[nodiscard]] expected<LifecycleRestartReceipt, Error> request_restart(
      OperatorRef target, LifecycleRestartOptions options = {}) const noexcept;

  /// @brief Inspect the current bounded state of one accepted receipt.
  /// @param receipt Receipt minted by this run session.
  /// @return Current state and lineage, or a typed error without advancing the transaction.
  /// @retval ErrorCode::kClosed This handle is empty.
  /// @retval ErrorCode::kLifecycleTransactionsDisabled Receipt storage is unavailable.
  /// @retval ErrorCode::kCrossSessionOperatorRef The receipt is unknown or belongs to another run.
  /// @note Inspection does not wait, cancel, publish, or advance the transaction.
  /// Accepted receipts remain inspectable after the originating session closes.
  /// Inspection briefly serializes on the run control mutex and is not a
  /// lock-free hot-path operation.
  [[nodiscard]] expected<LifecycleRestartReceiptView, Error> inspect(
      LifecycleRestartReceipt receipt) const noexcept;

  /// @brief Wait for a receipt to enter a terminal state.
  /// @param receipt Receipt minted by this run session.
  /// @param timeout Finite nonnegative caller-side maximum wait.
  /// @return Terminal state, or a typed error without cancelling the transaction.
  /// @retval ErrorCode::kLifecycleInvalidCallingContext Called from a Runtime callback context.
  /// @retval ErrorCode::kClosed This handle is empty.
  /// @retval ErrorCode::kInvalidArgument \p timeout is negative.
  /// @retval ErrorCode::kLifecycleTransactionsDisabled Receipt storage is unavailable.
  /// @retval ErrorCode::kCrossSessionOperatorRef The receipt is unknown or belongs to another run.
  /// @retval ErrorCode::kTimeout The receipt did not become terminal within \p timeout.
  /// @warning Do not call from compute, lifecycle callbacks, or Dispatcher code.
  /// @note This is the only potentially blocking handle operation. Timeout is a
  /// caller-side observation bound; it never cancels or mutates the transaction.
  [[nodiscard]] expected<LifecycleRestartReceiptView, Error> await(
      LifecycleRestartReceipt receipt, std::chrono::nanoseconds timeout) const;

  /// @brief Return whether this wrapper contains a control capability.
  /// @return `true` when the wrapper retains a capability, otherwise `false`.
  [[nodiscard]] explicit operator bool() const noexcept { return state_ != nullptr; }

 private:
  explicit LifecycleControlHandle(detail::LifecycleControlHandleState* state) noexcept;

  detail::LifecycleControlHandleState* state_{};

  friend class detail::LifecycleControlAccess;
};

}  // namespace holoscan
