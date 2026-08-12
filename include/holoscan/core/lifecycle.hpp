/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <concepts>
#include <cstdint>
#include <string_view>
#include <type_traits>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator_ref.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/readiness_source.hpp>
#include <holoscan/time/clock_registry.hpp>

namespace holoscan {

class OperatorPath;

namespace detail {
struct LifecycleContextState;
class LifecycleContextTestAccess;
class LifecycleContextRuntimeAccess;

/// @brief Terminate after an always-on exact parameter-type contract violation.
/// @warning This function never returns and is not an application extension point.
[[noreturn]] HOLOSCAN_CORE_EXPORT void parameter_type_contract_violation() noexcept;
}  // namespace detail

/// @brief Identify one global operator lifecycle barrier.
///
/// The first six values are forward stages. \c kStop and \c kRelease are
/// teardown stages and are never valid restart \c from_stage values.
enum class LifecycleStage : std::uint8_t {
  kConfigure = 0,  ///< Validate immutable configuration and derive cold intent.
  kDiscover = 1,   ///< Discover external capabilities without publishing readiness.
  kAllocate = 2,   ///< Reserve admitted finite runtime resources.
  kWarmUp = 3,     ///< Prepare resources without making the operator activatable.
  kArm = 4,        ///< Bind current-generation external and readiness capabilities.
  kStart = 5,      ///< Publish the operator as ready for steady-state activation.
  kStop = 6,       ///< Close activation acceptance and quiesce steady-state work.
  kRelease = 7,    ///< Release resources owned by the active lifecycle epoch.
};

/// @brief Report the bounded disposition of one lifecycle callback.
enum class LifecycleStatus : std::uint8_t {
  kOk = 0,                ///< Stage completed successfully.
  kSkipped = 1,           ///< Optional stage had no work for this operator.
  kRetryableFailure = 2,  ///< Stage failed with an explicitly retryable cold-path cause.
  kFatalFailure = 3,      ///< Stage failed and cannot continue the run.
};

/// @brief Identify which transition phase owns a lifecycle callback.
enum class LifecycleTransitionPhase : std::uint8_t {
  kNone = 0,                ///< Event or operation outside a lifecycle callback phase.
  kStartupForward = 1,      ///< Initial forward stages using generation zero.
  kPreRestartTeardown = 2,  ///< Teardown of the old active epoch before restart.
  kRestartForward = 3,      ///< Forward suffix using candidate or retained current state.
  kForwardRollback = 4,     ///< Unwind successful forward work after a primary failure.
  kRestoreOld = 5,          ///< Restore or re-enter the old published lifecycle state.
  kShutdownTeardown = 6,    ///< Final teardown of the active published epoch.
};

/// @brief Return whether a lifecycle stage is a valid forward restart entry.
/// @param stage Candidate lifecycle stage.
/// @return \c true only for \c kConfigure through \c kStart.
[[nodiscard]] constexpr bool is_lifecycle_forward_stage(LifecycleStage stage) noexcept {
  return static_cast<std::uint8_t>(stage) <= static_cast<std::uint8_t>(LifecycleStage::kStart);
}

/// @brief Observe cooperative stop or fatal shutdown during one callback.
///
/// The view is borrowed, non-owning, read-only, nonblocking, and allocation-free. It is valid only
/// for the callback lifetime. Once a read returns true, the callback's monotonic latch never
/// returns false. Reading grants no stop, wait, rollback, or publication authority.
class CancellationView {
 public:
  /// @brief Construct an inert view that reports no stop request.
  CancellationView() noexcept = default;

  /// @brief Return whether cooperative stop or fatal shutdown was requested.
  /// @return The monotonic callback-scoped latch using acquire ordering, or \c false for an inert
  /// view.
  [[nodiscard]] bool stop_requested() const noexcept {
    return requested_ != nullptr && requested_->load(std::memory_order_acquire);
  }

 private:
  explicit CancellationView(const std::atomic_bool* requested) noexcept : requested_(requested) {}

  const std::atomic_bool* requested_{};

  friend class ExecutionContext;
  friend class LifecycleContext;
};

/// @brief Name one completed ordering fact in a lifecycle-stage DAG.
///
/// This value carries only the stable-name component of an ordering fact.
/// The surrounding lifecycle-builder operation supplies the graph-instance
/// scope and lifecycle stage. The key grants no typed object access,
/// ownership, borrowing, or lifetime authority on its own.
struct LifecycleDependencyKey {
  std::string_view stable_name{};  ///< Qualified name copied before the builder call returns.
};

/// @brief Represent a registered fixed-size audit reason; zero is unspecified.
using ReasonId = std::uint64_t;

/// @brief Supply a registered bounded reason for one restart request.
struct LifecycleRestartReason {
  ReasonId stable_id{};  ///< Registered reason identity; zero means unspecified.
};

/// @brief Select the requested first forward lifecycle stage of a restart.
struct LifecycleRestartOptions {
  /// Requested earliest invalidated and first forward stage. Only
  /// \c kConfigure through \c kStart are valid request values.
  LifecycleStage from_stage{LifecycleStage::kConfigure};
  LifecycleRestartReason reason{};  ///< Registered audit reason with no hot-path text.
};

/// @brief Identify one accepted restart submission in a run session.
///
/// A receipt is an immutable run-scoped identity and does not itself wait,
/// cancel, retain a session, or grant additional target authority.
struct LifecycleRestartReceipt {
  std::uint64_t receipt_id{};           ///< Nonzero identity within the run session.
  std::uint64_t acceptance_sequence{};  ///< FIFO order assigned at accepted submission.
  OperatorRef target{};                 ///< Generation-stamped same-session target.
  LifecycleStage requested_stage{LifecycleStage::kConfigure};  ///< Original caller request.
};

/// @brief Identify one lifecycle transaction within a run session.
using LifecycleTransactionId = std::uint64_t;

/// @brief Identify one callback declaration slot within an Operator lifecycle contract.
///
/// EA1 uses the unique stage slot as semantic identity. It is pointer-free and portable, but it is
/// not a compiler-specific member-pointer encoding or a standalone binary fingerprint.
using LifecycleMethodId = std::uint8_t;

/// @brief Identify one published lifecycle epoch for an Operator.
using LifecycleEpoch = std::uint64_t;

/// @brief Identify one atomically installed activation fence.
using LifecycleFenceId = std::uint64_t;

/// @brief Report one bounded lifecycle transaction disposition.
enum class LifecycleTransactionState : std::uint8_t {
  kRejected = 0,          ///< Validation rejected before acceptance; no fence or receipt exists.
  kAcceptedNoChange = 1,  ///< Equivalent parameter-only candidate terminated without execution.
  kQueued = 2,            ///< Accepted, fenced, and still open to compatible coalescing.
  kQuiescing = 3,         ///< Coalescing cutoff passed while affected work retires.
  kRestarting = 4,        ///< Teardown or forward lifecycle callbacks are executing.
  kPublished = 5,         ///< Successful \c kStart published the new lifecycle epoch.
  kSuperseded = 6,        ///< A later complete candidate replaced this submission before publish.
  kCanceled = 7,          ///< Stop or activation failure won before publication.
  kRolledBack = 8,        ///< Forward failure restored the prior published epoch.
  kFailedStopped = 9,     ///< Recovery could not restore a coherent active epoch.
};

/// @brief Record the bounded stage decision for one accepted lifecycle transaction.
struct LifecycleStageDecision {
  LifecycleTransactionId transaction{};  ///< Nonzero run-local transaction identity.
  LifecycleStage requested{LifecycleStage::kConfigure};  ///< Earliest stage requested by ingress.
  LifecycleStage effective{LifecycleStage::kConfigure};  ///< Earliest compiler/executor stage used.
  ReasonId widening_reason{};  ///< Stable reason when \ref effective is earlier than requested.
};

/// @brief Record one completed callback retained for deterministic rollback.
struct LifecycleRollbackRecord {
  OperatorInstanceId owner{};                        ///< Plan-local callback owner.
  LifecycleStage stage{LifecycleStage::kConfigure};  ///< Completed stage being unwound.
  LifecycleTransitionPhase phase{LifecycleTransitionPhase::kForwardRollback};
  ParamGeneration params_generation{};  ///< Exact generation that created the completed work.
};

/// @brief Provide cold, callback-lifetime lifecycle services.
///
/// This facade contains one hidden-state pointer. It exposes immutable phase-bound state and
/// identity binding only; it cannot submit or await a restart, mutate graph topology, or invoke
/// another lifecycle callback. References returned from it expire no later than the callback or
/// lifecycle generation stated by the returned value's own contract.
class HOLOSCAN_CORE_EXPORT LifecycleContext {
 public:
  /// @brief Return the callback-lifetime cooperative cancellation view.
  /// @return Borrowed monotonic stop/fatal-shutdown latch. Restart acceptance alone does not set
  /// it.
  [[nodiscard]] CancellationView cancellation() const noexcept;

  /// @brief Return the complete immutable parameter aggregate selected for this callback phase.
  /// @tparam ParamsT Exact unqualified aggregate type declared by the callback's Operator.
  /// @return Callback-lifetime reference. Repeated calls return the same address.
  /// @warning A wrong exact type terminates through the bounded Core contract handler.
  template <typename ParamsT>
    requires std::same_as<ParamsT, std::remove_cvref_t<ParamsT>> &&
             std::default_initializable<ParamsT> && std::copy_constructible<ParamsT> &&
             std::is_aggregate_v<ParamsT>
  [[nodiscard]] const ParamsT& params() const noexcept {
    const void* const address =
        parameter_address(param_type_descriptor<ParamsT>().local_type_token);
    if (address == nullptr) {
      detail::parameter_type_contract_violation();
    }
    return *static_cast<const ParamsT*>(address);
  }

  /// @brief Return the phase-selected complete parameter generation.
  /// @return Generation zero for startup/NoParams or the exact old, candidate, or current value.
  [[nodiscard]] ParamGeneration params_generation() const noexcept;

  /// @brief Return the transition phase that selected this callback's state view.
  /// @return One non-\c kNone phase inside a callback; \c kNone for an inert context.
  [[nodiscard]] LifecycleTransitionPhase transition_phase() const noexcept;

  /// @brief Resolve a declared clock for this lifecycle generation.
  /// @param key Owned cold clock identity; the omitted key requests the admitted default.
  /// @return Generation-scoped reference or a typed binding/capability error.
  /// @note Resolution is cold-only and installs no compute-time lookup.
  [[nodiscard]] expected<ClockRef, Error> resolve_clock(ClockKey key) const noexcept;

  /// @brief Return the current generation's bounded notification capability.
  /// @param source Declared notification source owned by the callback's Operator.
  /// @return Generation-bound sender or a typed ownership/stage/binding error.
  [[nodiscard]] expected<NotificationSender, Error> notification_sender(
      NotificationSource& source) const noexcept;

  /// @brief Bind an already-authorized plan target to the current run generation.
  /// @param plan_ref Plan-bound target whose run generation is zero.
  /// @return Fixed-size run-bound target or a stale/foreign/unauthorized error.
  /// @note This operation resolves identity only; it neither submits nor waits for lifecycle work.
  [[nodiscard]] expected<OperatorRef, Error> restart_target(OperatorRef plan_ref) const noexcept;

  /// @brief Resolve an already-authorized owning Operator path into current-run identity.
  /// @param path Canonical owning cold Operator path.
  /// @return Fixed-size run-bound target or a typed resolution/authorization error.
  /// @note Path resolution is cold-only. Store the returned value for later compute use.
  [[nodiscard]] expected<OperatorRef, Error> restart_target(
      const OperatorPath& path) const noexcept;

 private:
  explicit LifecycleContext(detail::LifecycleContextState* state) noexcept : state_(state) {}
  [[nodiscard]] const void* parameter_address(const void* exact_type) const noexcept;

  detail::LifecycleContextState* state_{};

  friend class detail::LifecycleContextTestAccess;
  friend class detail::LifecycleContextRuntimeAccess;
};

static_assert(std::is_standard_layout_v<LifecycleRestartOptions>);
static_assert(std::is_trivially_copyable_v<LifecycleRestartOptions>);
static_assert(std::is_standard_layout_v<LifecycleRestartReceipt>);
static_assert(std::is_trivially_copyable_v<LifecycleRestartReceipt>);
static_assert(std::is_standard_layout_v<LifecycleStageDecision>);
static_assert(std::is_trivially_copyable_v<LifecycleStageDecision>);
static_assert(std::is_standard_layout_v<LifecycleRollbackRecord>);
static_assert(std::is_trivially_copyable_v<LifecycleRollbackRecord>);
static_assert(sizeof(LifecycleContext) == sizeof(void*));

}  // namespace holoscan
