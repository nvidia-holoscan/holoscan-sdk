/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <span>
#include <system_error>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/lifecycle_control.hpp>
#include <holoscan/core/operator_ref.hpp>
#include <holoscan/core/parameter_control.hpp>
#include <holoscan/core/rate_generation.hpp>

namespace holoscan {

namespace detail {
class RunSessionState;
class RunSessionAccess;
}  // namespace detail

/// @brief Summarize one run session's coarse execution outcome.
enum class RunTermination : std::uint8_t {
  kRunning = 0,    ///< Startup, execution, a termination request, or cleanup still owns work.
  kCompleted = 1,  ///< Natural finite execution and mandatory cleanup completed successfully.
  /// An explicit immediate whole-run success request and mandatory cleanup completed successfully.
  kRequestedCompleted = 2,
  kStopped = 3,  ///< Cooperative stop completed successfully, or this wrapper is moved-from.
  kFailed = 4,   ///< Startup, execution, or mandatory cleanup failed.
};

/// @brief Report terminal progress for one session-local planned Operator completion rule.
///
/// Records are immutable, bounded by the selected process group's canonical
/// plan subsequence, and available only after the RunSession outcome has
/// frozen. They are inspection state, not a mutable counter or a run-control
/// surface.
struct OperatorCompletionProgressRecord {
  OperatorRef owner{};  ///< Run-bound logical Operator identity.
  OperatorCompletionRuleKind kind{
      OperatorCompletionRuleKind::kComputeCalls};  ///< Closed completion-rule discriminator.
  std::uint64_t configured_limit{};                ///< Exact configured unsigned limit.
  std::uint64_t successful_compute_calls{};        ///< Successful calls committed by this session.
  bool limit_reached{};  ///< True exactly when progress reached the limit.

  friend constexpr bool operator==(const OperatorCompletionProgressRecord&,
                                   const OperatorCompletionProgressRecord&) noexcept = default;
};

/// @brief Own one accepted-plan attempt and every runtime activity it starts.
///
/// The source Graph and caller's ExecutionPlan wrapper may be destroyed after
/// construction. A live wrapper is strict RAII: destruction requests
/// cooperative stop and joins all session-owned work. Moving transfers the
/// complete owner and leaves the source in a defined inert state.
/// Plan admission does not imply that every required runtime binding is
/// available; startup may fail closed before invoking user callbacks.
/// Query, stop, wait, diagnostic, identity-binding, Parameter-capability minting, and Parameter-
/// ingress closure calls may be concurrent. Parameter-capability operations synchronize through
/// their own run-scoped state and remain valid until the session publishes its terminal fence.
/// Moving or destroying the wrapper requires external exclusion from every
/// other operation on that wrapper.
class HOLOSCAN_CORE_EXPORT RunSession {
 public:
  /// @brief Prohibit construction without an accepted run attempt.
  RunSession() = delete;

  /// @brief Transfer sole ownership of the complete runtime generation.
  /// @param other Live or inert session wrapper to consume.
  /// @note The source becomes inert and reports `RunTermination::kStopped`.
  RunSession(RunSession&& other) noexcept;

  /// @brief Prohibit shared wrapper ownership.
  RunSession(const RunSession&) = delete;

  /// @brief Prohibit copied ownership.
  RunSession& operator=(const RunSession&) = delete;

  /// @brief Prohibit replacement of a destination that may own a live run.
  RunSession& operator=(RunSession&&) = delete;

  /// @brief Request stop, join every owned activity, and destroy the session.
  /// @note Destruction never detaches, throws, or force-kills user code.
  /// @warning Destruction from a Runtime compute, lifecycle, or Dispatcher
  /// callback context is a fatal contract violation and terminates the process.
  /// @warning Destruction may block indefinitely when user code ignores the
  /// cooperative cancellation view and does not return.
  ~RunSession() noexcept;

  /// @brief Return whether any run-owned work remains outstanding.
  /// @return True during startup, execution, stop-requested, or cleanup state.
  [[nodiscard]] bool is_running() const noexcept;

  /// @brief Return the current coarse session outcome.
  /// @return Running while work remains, then one stable terminal outcome.
  [[nodiscard]] RunTermination termination() const noexcept;

  /// @brief Request cooperative stop without waiting for completion.
  /// @note This operation is thread-safe, idempotent, nonblocking, and a no-op
  /// for a moved-from wrapper. It closes lifecycle-control ingress, publishes
  /// the cancellation latch, and wakes the worker; receipt finalization,
  /// diagnostic publication, callback cleanup, and lane cleanup remain
  /// worker-owned.
  void request_stop() noexcept;

  /// @brief Join every run-owned activity.
  /// @note Concurrent external callers share one physical join operation;
  /// repeated calls after completion return immediately. Calling from the
  /// run-owned worker or any Runtime compute, lifecycle, or Dispatcher callback
  /// context is a fatal contract violation and terminates the process.
  /// @throws std::system_error if the platform thread join operation fails.
  /// @warning This call may block indefinitely when user code ignores the
  /// cooperative cancellation view and does not return.
  /// @warning Completion includes quiescing every runtime-owned CUDA stream. A
  /// CUDA kernel that does not terminate can therefore block this call indefinitely.
  void wait();

  /// @brief Return bounded structural diagnostics retained by this session.
  /// @return Borrowed prefix of currently published records.
  /// @note The returned storage and observed prefix remain valid until this wrapper is moved or
  /// destroyed. Later calls may expose a longer append-only prefix. Before terminal completion,
  /// callers must not treat the observed prefix as complete.
  [[nodiscard]] std::span<const Diagnostic> diagnostics() const noexcept;

  /// @brief Return bounded terminal progress for planned Operator completion rules.
  /// @return Empty before terminal outcome freeze, after move, for an
  /// unselected multi-process attempt, or when cold startup fails before
  /// session-local progress storage exists; otherwise one immutable record per rule
  /// owned by this session's selected process group, in canonical plan order.
  /// @note The borrowed span remains valid until this wrapper is moved or destroyed.
  /// A remote rule is absent rather than represented by a zero-valued record.
  [[nodiscard]] std::span<const OperatorCompletionProgressRecord> operator_completion_progress()
      const noexcept;

  /// @brief Bind a plan-only Operator identity to this run generation.
  /// @param plan_ref Valid zero-run-generation reference from this session's plan.
  /// @return Current-run identity, or a typed stale/foreign reference error.
  /// @retval ErrorCode::kStaleOperatorRef The reference is invalid, foreign,
  /// already run-bound, unresolved, terminal, or the wrapper is moved-from.
  /// @note Identity binding grants no lifecycle or parameter authority.
  [[nodiscard]] expected<OperatorRef, Error> operator_ref(OperatorRef plan_ref) const noexcept;

  /// @brief Mint a restriction-only lifecycle control capability for this run.
  /// @param policy Run-bound target and stage subset to copy into the handle.
  /// @return Bounded handle, or a typed error without changing run state.
  /// @retval ErrorCode::kClosed This wrapper is moved-from, startup failed, or lifecycle ingress
  /// has begun stopping.
  /// @retval ErrorCode::kLifecycleTransactionsDisabled The accepted plan disabled transactions.
  /// @retval ErrorCode::kCrossSessionOperatorRef A policy target is not bound to this run.
  /// @retval ErrorCode::kLifecycleInvalidRestartStage A policy stage is not a forward stage.
  /// @retval ErrorCode::kLifecycleRestartUnauthorized The policy exceeds plan-authorized external
  /// targets or stages.
  /// @retval ErrorCode::kResourceExhausted The bounded policy or handle state could not be
  /// allocated.
  /// @note The plan must explicitly enable lifecycle transactions and external
  /// authority for every target. The policy can remove but never add authority.
  /// This cold control-path operation may allocate the bounded normalized policy, briefly
  /// serializes with lifecycle-control ingress and cleanup, and never waits for a transaction to
  /// execute.
  [[nodiscard]] expected<LifecycleControlHandle, Error> lifecycle_control(
      const LifecycleAccessPolicy& policy);

  /// @brief Mint a restricted capability for this run's admitted Parameter control surface.
  /// @param policy Requested origin and permissions; the session may only remove authority.
  /// @return Shared run-scoped capability, or a typed closed, unsupported, or policy error.
  /// @retval ErrorCode::kInvalidArgument The policy has an invalid enum, scope shape, or foreign
  /// run/plan/field identity, or the call occurs from a Runtime callback context.
  /// @retval ErrorCode::kResourceExhausted The capability state or immutable catalog projection
  /// could not be allocated.
  /// @retval ErrorCode::kClosed Startup retired the run before attachment completed.
  /// @note This cold operation may block without a timeout until asynchronous startup attaches the
  /// Parameter coordinator or retires the run, and may allocate the capability and immutable
  /// catalog projection. The returned handle does not keep execution alive and remains safe to
  /// destroy or query after session teardown. Its prepared catalog remains available; live
  /// operations return \ref ErrorCode::kClosed. The call is forbidden from Runtime compute,
  /// lifecycle, and Dispatcher callbacks.
  [[nodiscard]] expected<ParamControlHandle, Error> parameter_control(
      const ParamAccessPolicy& policy);

  /// @brief Irreversibly close every Parameter-update ingress for this run.
  /// @return Stable terminal accepted sequence and whether an earlier call already closed ingress.
  /// @retval ErrorCode::kInvalidArgument The call occurs from a Runtime callback context.
  /// @retval ErrorCode::kClosed This wrapper is moved-from or startup retired before attaching the
  /// Parameter coordinator.
  /// @note The operation is idempotent. It does not stop execution; accepted updates through the
  /// returned sequence remain awaitable to a terminal state. During asynchronous startup this call
  /// may block without a timeout until coordinator attachment or startup retirement.
  [[nodiscard]] expected<ParamStopFenceResult, Error> close_parameter_update_ingress();

  /// @brief Capture owning Parameter state using privileged local session authority.
  /// @param options Published stopped capture by default, or an explicit live-per-Operator mode.
  /// @return Owning value-safe snapshot that survives this session wrapper.
  /// @retval ErrorCode::kInvalidArgument The call occurs from a Runtime callback context.
  /// @retval ErrorCode::kNotReady Stopped capture was requested while callbacks may still run or an
  /// explicit stopped frontier has not completed.
  /// @retval ErrorCode::kNotSupported A reserved consistency/view or unsupported strict value was
  /// requested.
  /// @retval ErrorCode::kResourceExhausted Owning snapshot storage could not be allocated.
  /// @retval ErrorCode::kClosed This wrapper is moved-from or startup retired before attaching the
  /// Parameter coordinator.
  /// @note Ordinary snapshot projection still excludes transient, sensitive, internal, and
  /// unsupported values; privileged session authority does not bypass redaction policy. During
  /// asynchronous startup this call may block without a timeout until coordinator attachment or
  /// startup retirement. The call is forbidden from Runtime callbacks.
  [[nodiscard]] expected<ParamSnapshot, Error> capture_params(ParamCaptureOptions options = {});

  /// @brief Capture owning evidence for the active and latest rejected rate generations.
  /// @return A coherent value snapshot that remains valid after this session wrapper.
  /// @retval ErrorCode::kInvalidArgument The call occurs from a Runtime callback context.
  /// @retval ErrorCode::kClosed This wrapper is moved-from or startup retired before attaching
  /// rate-generation state.
  /// @retval ErrorCode::kNotReady No rate generation has reached a terminal published or rejected
  /// state.
  /// @retval ErrorCode::kResourceExhausted Owning snapshot storage could not be allocated.
  /// @note During asynchronous startup this cold operation may wait until rate state attaches or
  /// the run retires. It never changes the active rate, queue, or provider generation.
  [[nodiscard]] expected<RateGenerationSnapshot, Error> capture_rate_generations();

 private:
  explicit RunSession(detail::RunSessionState* state) noexcept;

  // One opaque pointer keeps the installed ABI independent of the standard
  // library's shared_ptr layout. Strict RAII guarantees the worker is joined
  // before this state is deleted.
  detail::RunSessionState* state_{};

  friend class detail::RunSessionAccess;
};

static_assert(sizeof(RunSession) == sizeof(void*));

}  // namespace holoscan
