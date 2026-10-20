/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/stream>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string_view>
#include <type_traits>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/activation.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/operator_ref.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_compute_control.hpp>
#include <holoscan/core/rate.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/tag.hpp>
#include <holoscan/time/clock_registry.hpp>
#include <holoscan/time/time_point.hpp>

namespace holoscan {

namespace detail {
struct ExecutionContextState;
class SyntheticOperatorBinding;
class OperatorRuntimeAccess;
}  // namespace detail

/// @brief Borrow the runtime services and immutable evidence for one Operator callback.
///
/// The facade contains one hidden-state pointer and is valid only during \c compute(). All queries
/// are O(1), nonblocking, and allocation-free. It exposes no graph mutation, scheduler wait, clock
/// registry lookup, or \c reference_time() operation. Restart methods enqueue only through an
/// active lifecycle-transaction binding and otherwise return a typed disabled error.
/// During startup CUDA Graph capture, an eligible member receives only the services needed to
/// record its GPU work. Activation time, rate queries, parameter updates, lifecycle restart, and
/// termination requests are unavailable; the Operator author must not depend on them during
/// capture.
/// Retaining or using the facade, or any view borrowed from it, after \c compute() returns is
/// invalid; no post-callback safety or no-op behavior is promised.
class HOLOSCAN_CORE_EXPORT ExecutionContext {
 public:
  /// @brief Return the activation identity.
  /// @return Tag frozen for this callback.
  [[nodiscard]] Tag tag() const noexcept;

  /// @brief Return the Runtime-owned CUDA stream assigned to this activation.
  /// @return Borrowed stream reference kept alive by the RunSession through framework-managed
  /// readiness, publication, and deallocation ordering.
  /// @note Enqueue callback CUDA work on this stream to use automatic Tensor input waits and
  /// release ordering. A deliberately selected different stream remains caller-owned and requires
  /// explicit Tensor synchronization and deallocation registration.
  /// @warning The callback must not destroy the returned stream or retain its borrowed reference
  /// beyond the callback. A saved raw handle has the same restriction: do not submit through it
  /// from start(), stop(), a destructor, or another thread.
  [[nodiscard]] ::cuda::stream_ref cuda_stream() noexcept;

  /// @brief Return the admitted default activation clock.
  /// @return Borrowed generation-scoped clock reference.
  [[nodiscard]] ClockRef clock() const noexcept;

  /// @brief Return the deterministic logical activation coordinate.
  /// @return Tag time frozen for this callback.
  [[nodiscard]] TagTime tag_time() const noexcept;

  /// @brief Return the dispatch activation time.
  /// @return Time in the admitted activation clock, distinct from source capture time.
  /// @note Returns an invalid time during CUDA Graph member capture because capture is not a
  /// runtime activation.
  [[nodiscard]] TimePoint activation_time() const noexcept;

  /// @brief Return the immutable trace context for this callback.
  /// @return Callback-lifetime trace reference.
  [[nodiscard]] const TraceContext& trace() const noexcept;

  /// @brief Return the trace identity for this callback.
  /// @return Equivalent to `trace().trace_id`.
  [[nodiscard]] TraceId trace_id() const noexcept;

  /// @brief Return the operator-level activation and violation outcome.
  /// @return Frozen status; reading it emits no witness and changes no state.
  [[nodiscard]] ActivationStatus activation() const noexcept;

  /// @brief Return the callback-lifetime cooperative cancellation view.
  /// @return Borrowed monotonic stop view.
  [[nodiscard]] CancellationView cancellation() const noexcept;

  /// @brief Return the current run-bound Operator identity.
  /// @return Run- and plan-generation-stamped identity for this callback, or an invalid reference
  /// when the context is inert.
  /// @note This identity lets fixed compute-originated control records name their own Operator
  /// without retaining an authoring handle or a runtime object address.
  [[nodiscard]] OperatorRef self() const noexcept;

  /// @brief Return the canonical path of the Operator executing this callback.
  /// @return Nonempty canonical percent-encoded path with no leading slash for a Runtime-bound
  /// callback, or an empty view when the context is inert or otherwise unbound.
  /// @warning The returned view is valid only for the active \c compute() callback and must not be
  /// retained after callback return.
  [[nodiscard]] std::string_view operator_path() const noexcept;

  /// @brief Return the immutable parameter aggregate pinned for this callback.
  /// @tparam ParamsT Exact unqualified aggregate type declared by the Operator.
  /// @return Callback-lifetime reference from the coherent pinned parameter header.
  /// @warning A wrong exact type is a Core contract violation and terminates through a bounded
  /// noexcept handler; cv/ref, convertible, base/derived, and layout-compatible substitutes reject
  /// at compile time or terminate rather than receiving a recoverable error.
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

  /// @brief Return the generation of the parameter aggregate pinned here.
  /// @return Stable callback generation; `NoParams` uses zero.
  [[nodiscard]] ParamGeneration params_generation() const noexcept;

  /// @brief Borrow the compute-originated fixed parameter-update seam.
  /// @return Callback-lifetime facade for nonblocking preallocated submissions.
  /// @note The returned facade does not expose general reads, owning submission, waits, session
  /// ingress closure, or lifecycle restart authority. It becomes invalid when this callback
  /// returns, and every accepted update targets a future immutable generation rather than the
  /// generation pinned by this context.
  /// @note During CUDA Graph member capture, the returned facade rejects submissions with
  /// `ParamUpdateStatus::kUpdatesDisabled`.
  [[nodiscard]] ParamComputeControl param_control() const noexcept;

  /// @brief Request a bounded lifecycle restart of the current Operator.
  /// @param options Requested first forward stage and registered bounded reason; defaults to
  /// `kConfigure`.
  /// @return Run-bound receipt after enqueue admission, or a typed error without state change.
  /// @retval ErrorCode::kStaleOperatorRef The callback's self reference is not bound to an active
  /// run.
  /// @retval ErrorCode::kLifecycleTransactionsDisabled No active restart ingress is bound.
  /// @retval ErrorCode::kLifecycleInvalidRestartStage The requested stage is not a forward stage.
  /// @retval ErrorCode::kLifecycleRestartUnauthorized Self-restart was not admitted by the plan.
  /// @retval ErrorCode::kLifecycleRestartQueueFull The bounded request or successor capacity is
  /// exhausted.
  /// @retval ErrorCode::kLifecycleRunState The run cannot accept the request in its current phase.
  /// @retval ErrorCode::kClosed Lifecycle restart ingress has begun stopping or has retired.
  /// @note This call never invokes lifecycle code, traverses the graph, dynamically allocates, or
  /// waits for completion. Acceptance only queues the request and installs its activation fence;
  /// restart execution begins after this compute callback retires and before this Operator can
  /// begin another compute callback.
  /// @note During CUDA Graph member capture, no restart binding exists; an otherwise-valid request
  /// returns `ErrorCode::kLifecycleTransactionsDisabled`.
  [[nodiscard]] expected<LifecycleRestartReceipt, Error> request_restart(
      LifecycleRestartOptions options = {}) noexcept;

  /// @brief Request a bounded lifecycle restart of an authorized Operator.
  /// @param target Same-session generation-stamped target.
  /// @param options Requested first forward stage and registered bounded reason.
  /// @return Run-bound receipt after enqueue admission, or a typed error before state change.
  /// @retval ErrorCode::kStaleOperatorRef The target or callback self reference is unbound, or the
  /// target belongs to a different plan generation.
  /// @retval ErrorCode::kCrossSessionOperatorRef The target belongs to another run generation of
  /// the same plan.
  /// @retval ErrorCode::kLifecycleTransactionsDisabled No active restart ingress is bound.
  /// @retval ErrorCode::kLifecycleInvalidRestartStage The requested stage is not a forward stage.
  /// @retval ErrorCode::kLifecycleRestartUnauthorized Cross-target authority, the target, or the
  /// requested stage was not admitted by the plan.
  /// @retval ErrorCode::kLifecycleRestartQueueFull The bounded request or successor capacity is
  /// exhausted.
  /// @retval ErrorCode::kLifecycleRunState The run cannot accept the request in its current phase.
  /// @retval ErrorCode::kClosed Lifecycle restart ingress has begun stopping or has retired.
  /// @warning The call cannot grant authority, choose a closure, or wait for its own receipt.
  /// @note This call is nonblocking and does not dynamically allocate. Acceptance only queues the
  /// request and installs its activation fence. A target already computing finishes that callback
  /// before restart execution; an idle target restarts at the control safe point before its next
  /// compute callback.
  /// @note During CUDA Graph member capture, no restart binding exists; an otherwise-valid request
  /// returns `ErrorCode::kLifecycleTransactionsDisabled`.
  [[nodiscard]] expected<LifecycleRestartReceipt, Error> request_restart(
      OperatorRef target, LifecycleRestartOptions options = {}) noexcept;

  /// @brief Return the atomically published rate view for one owned input.
  /// @tparam T Exact published payload type.
  /// @param input Exact \c Input member subobject registered by this callback's Operator during
  /// \c setup(). A copied declaration, foreign Operator declaration, or output declaration does
  /// not carry query authority.
  /// @return Current published rate view for \p input, or a typed error without state change.
  /// @retval ErrorCode::kInvalidArgument The context is not bound, the declaration is not the
  /// exact registered member subobject, the port is not owned by this Operator, or its direction
  /// is not input.
  /// @retval ErrorCode::kNotReady No rate generation has been published for this run.
  /// @note Call only during the active \c compute() callback and do not retain the context after
  /// callback return. The returned \c PortRateView is a value and may be retained. The query is
  /// O(1), allocation-free, nonblocking, and observes one atomically pinned generation.
  /// @note During CUDA Graph member capture, this returns `ErrorCode::kInvalidArgument` because no
  /// runtime rate generation is bound.
  template <PublishedPayload T>
  [[nodiscard]] expected<PortRateView, Error> rate(const Input<T>& input) const noexcept {
    return rate_port(static_cast<const void*>(&input), false);
  }

  /// @brief Return the atomically published rate view for one owned output.
  /// @tparam T Exact published payload type.
  /// @param output Exact \c Output member subobject registered by this callback's Operator during
  /// \c setup(). A copied declaration, foreign Operator declaration, or input declaration does
  /// not carry query authority.
  /// @return Current published rate view for \p output, or a typed error without state change.
  /// @retval ErrorCode::kInvalidArgument The context is not bound, the declaration is not the
  /// exact registered member subobject, the port is not owned by this Operator, or its direction
  /// is not output.
  /// @retval ErrorCode::kNotReady No rate generation has been published for this run.
  /// @note Call only during the active \c compute() callback and do not retain the context after
  /// callback return. The returned \c PortRateView is a value and may be retained. The query is
  /// O(1), allocation-free, nonblocking, and observes one atomically pinned generation.
  /// @note During CUDA Graph member capture, this returns `ErrorCode::kInvalidArgument` because no
  /// runtime rate generation is bound.
  template <PublishedPayload T>
  [[nodiscard]] expected<PortRateView, Error> rate(const Output<T>& output) const noexcept {
    return rate_port(static_cast<const void*>(&output), true);
  }

  /// @brief Request explicit immediate whole-run success.
  /// @note This call is bounded, nonblocking, and allocation-free. It closes the session-wide
  /// callback-admission gate at one monotonic publication point. Callbacks admitted before that
  /// point may retire; no callback is admitted afterward. After mandatory cleanup, Runtime reports
  /// \c RunTermination::kRequestedCompleted unless a stop or failure takes precedence.
  /// @warning Requested completion is imperative cancellation, not EOS or natural completion. It
  /// does not guarantee source closure, downstream drain, or delivery of queued samples.
  /// @warning This method may be called only through the \c ExecutionContext of the active
  /// \c compute() callback. Using a retained context after callback return is invalid.
  /// @note This has no effect during CUDA Graph member capture. Use the fused source's temporal
  /// contract for completion policy.
  void request_completion() noexcept;

  /// @brief Request cooperative stop of the current run.
  /// @note This call is bounded, nonblocking, and allocation-free. It closes the same session-wide
  /// callback-admission gate and reports \c RunTermination::kStopped after mandatory cleanup unless
  /// a failure occurs. A stop request promotes an earlier completion request; completion never
  /// weakens stop.
  /// @note Once the stop is accepted, a blocking publication to a full recipient returns its
  /// publication-pressure error at once instead of waiting out the graph blocking budget. Its
  /// message begins with \c RUNTIME_PUBLICATION_STOPPED. A callback that returns that error
  /// unchanged still ends the run stopped, with a warning-level
  /// \c RUNTIME_OPERATOR_PUBLICATION_STOPPED diagnostic.
  /// @warning This method may be called only through the \c ExecutionContext of the active
  /// \c compute() callback. Using a retained context after callback return is invalid.
  /// @note This has no effect during CUDA Graph member capture. Request stop outside the captured
  /// region.
  void request_stop() noexcept;

 private:
  explicit ExecutionContext(detail::ExecutionContextState* state) noexcept : state_(state) {}
  [[nodiscard]] const void* parameter_address(const void* exact_type) const noexcept;
  [[nodiscard]] expected<PortRateView, Error> rate_port(const void* declaration,
                                                        bool output) const noexcept;

  detail::ExecutionContextState* state_{};

  friend class detail::SyntheticOperatorBinding;
  friend class detail::OperatorRuntimeAccess;
};

static_assert(sizeof(ExecutionContext) == sizeof(void*));

}  // namespace holoscan
