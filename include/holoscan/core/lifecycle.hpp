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
#include <holoscan/core/lifecycle_types.hpp>
#include <holoscan/core/operator_ref.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/rate.hpp>
#include <holoscan/core/readiness_source.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/time/clock_registry.hpp>

namespace holoscan {

class ForeignOperatorCore;
class OperatorPath;
template <PublishedPayload T>
class Output;

namespace detail {
struct LifecycleContextState;
class LifecycleContextTestAccess;
class LifecycleContextRuntimeAccess;

/// @brief Terminate after an always-on exact parameter-type contract violation.
/// @warning This function never returns and is not an application extension point.
[[noreturn]] HOLOSCAN_CORE_EXPORT void parameter_type_contract_violation() noexcept;
}  // namespace detail

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

  /// @brief Supply bounded explanatory detail for a failed advanced lifecycle callback.
  /// @param code Original error category.
  /// @param message Borrowed text copied before returning; do not include secrets.
  /// @note The first report wins. The callback's returned LifecycleStatus remains authoritative;
  /// successful callbacks discard this detail. Reporting never allocates or throws.
  void report_error(ErrorCode code, std::string_view message) noexcept;

  /// @brief Supply an existing error as callback failure detail.
  /// @param error Error whose category and bounded message are copied before returning.
  void report_error(const Error& error) noexcept { report_error(error.code, error.message); }

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

  /// @brief Bind this output's lifecycle-selected nominal rate for the current candidate.
  /// @tparam T Exact published payload type.
  /// @param output Output owned by the callback's Operator and declared lifecycle-selectable.
  /// @param value Exact positive rational nominal average selected for this candidate.
  /// @return Success, including an identical normalized repeat, or a typed ownership, stage,
  /// deadline, conflict, maximum, or capacity error.
  /// @retval ErrorCode::kRateBindingInvalid The supplied rate is zero or has a zero denominator.
  /// @retval ErrorCode::kRateBindingAboveMaximum The supplied rate exceeds the declared maximum.
  /// @retval ErrorCode::kRateBindingConflict This candidate already has a different binding.
  /// @retval ErrorCode::kLifecycleInvalidCallingContext The context is inert or the call is not
  /// made during a startup/restart forward binding phase. This case does not reject a candidate.
  /// @retval ErrorCode::kPermissionDenied The declaration, owner, direction, or binding stage is
  /// unauthorized. During a forward binding phase this rejects the candidate.
  /// @warning Any declaration or value validation failure during a forward binding phase
  /// permanently rejects the private candidate. A later forward-phase call in that candidate
  /// returns the original failure; retrying with a fallback value cannot repair it. Return a
  /// lifecycle failure status instead.
  template <PublishedPayload T>
  [[nodiscard]] expected<void, Error> bind_nominal_rate(Output<T>& output, Rate value) noexcept {
    return bind_nominal_rate_port(static_cast<const void*>(&output), value);
  }

  /// @brief Return the phase-visible rate view for one owned input.
  /// @tparam T Exact published payload type.
  /// @param input Exact \c Input member subobject registered by this callback's Operator during
  /// \c setup(). A copied declaration, foreign Operator declaration, or output declaration does
  /// not carry query authority.
  /// @return Candidate rate view during startup/restart forward or forward rollback; otherwise the
  /// current published view, or a typed error without state change.
  /// @retval ErrorCode::kLifecycleInvalidCallingContext The context has no active lifecycle rate
  /// resolver.
  /// @retval ErrorCode::kInvalidArgument The declaration is not the exact registered member
  /// subobject, the port is not owned by this Operator, or its direction is not input.
  /// @retval ErrorCode::kNotReady The phase-selected candidate/current generation is unavailable.
  /// @note Call only during the active lifecycle callback and do not retain the context after
  /// callback return. The returned \c PortRateView is a value and may be retained. The query is
  /// O(1) and allocation-free.
  template <PublishedPayload T>
  [[nodiscard]] expected<PortRateView, Error> rate(const Input<T>& input) const noexcept {
    return rate_port(static_cast<const void*>(&input), false);
  }

  /// @brief Return the phase-visible rate view for one owned output.
  /// @tparam T Exact published payload type.
  /// @param output Exact \c Output member subobject registered by this callback's Operator during
  /// \c setup(). A copied declaration, foreign Operator declaration, or input declaration does
  /// not carry query authority.
  /// @return Candidate rate view during startup/restart forward or forward rollback; otherwise the
  /// current published view, or a typed error without state change.
  /// @retval ErrorCode::kLifecycleInvalidCallingContext The context has no active lifecycle rate
  /// resolver.
  /// @retval ErrorCode::kInvalidArgument The declaration is not the exact registered member
  /// subobject, the port is not owned by this Operator, or its direction is not output.
  /// @retval ErrorCode::kNotReady The phase-selected candidate/current generation is unavailable.
  /// @note Call only during the active lifecycle callback and do not retain the context after
  /// callback return. The returned \c PortRateView is a value and may be retained. The query is
  /// O(1) and allocation-free.
  template <PublishedPayload T>
  [[nodiscard]] expected<PortRateView, Error> rate(const Output<T>& output) const noexcept {
    return rate_port(static_cast<const void*>(&output), true);
  }

 private:
  explicit LifecycleContext(detail::LifecycleContextState* state) noexcept : state_(state) {}
  [[nodiscard]] const void* parameter_address(const void* exact_type) const noexcept;
  [[nodiscard]] expected<void, Error> bind_nominal_rate_port(const void* output,
                                                             Rate value) noexcept;
  [[nodiscard]] expected<PortRateView, Error> rate_port(const void* declaration,
                                                        bool output) const noexcept;
  void record_callback_failure(Error error) noexcept;

  detail::LifecycleContextState* state_{};

  friend class detail::LifecycleContextTestAccess;
  friend class detail::LifecycleContextRuntimeAccess;
  friend class ForeignOperatorCore;
};

static_assert(std::is_standard_layout_v<LifecycleRestartReceipt>);
static_assert(std::is_trivially_copyable_v<LifecycleRestartReceipt>);
static_assert(std::is_standard_layout_v<LifecycleRollbackRecord>);
static_assert(std::is_trivially_copyable_v<LifecycleRollbackRecord>);
static_assert(sizeof(LifecycleContext) == sizeof(void*));

}  // namespace holoscan
