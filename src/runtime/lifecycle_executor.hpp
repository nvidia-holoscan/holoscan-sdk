/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include <holoscan/core/expected.hpp>

#include "../compiler/execution_plan_state.hpp"
#include "../core/diagnostic_context.hpp"
#include "../core/lifecycle_kernel.hpp"
#include "../core/lifecycle_parameter_view.hpp"
#include "runtime_materialization.hpp"
#include "runtime_rate_state.hpp"

namespace holoscan::detail {

struct RuntimeOperatorInstance;
class GraphExec;
class RuntimeNotificationSourceRegistry;
class ExecutionLaneSet;
class RuntimeLifecycleExecutorTestAccess;

/// One private Parameter candidate retained by the shared lifecycle transaction.
struct LifecycleRestartParameterCandidate {
  OperatorRef owner{};
  ParamGeneration generation{};
};

/// Complete executor input for one accepted shared lifecycle transaction.
struct LifecycleRestartExecutionRequest {
  LifecycleTransactionId transaction{};
  OperatorRef target{};
  LifecycleStage stage{LifecycleStage::kConfigure};
  std::span<const OperatorInstanceId> members;
  std::span<const LifecycleRestartParameterCandidate> parameter_candidates;
};

/// Distinguish a published candidate from a recovered rollback and a stopped failure.
enum class LifecycleRestartExecutionDisposition : std::uint8_t {
  kPublished,
  kRolledBack,
  kFailedStopped,
};

/// Preserve the bounded lifecycle-kernel result through runtime publication.
///
/// The four failure slots retain independent evidence for the candidate transition, candidate
/// rollback, old-generation restoration, and restoration rollback. Copying this result never
/// re-enters lifecycle execution; the Dispatcher consumes only `disposition`, while the lifecycle
/// receipt diagnostic path retains the structured failure fields for post-run inspection.
struct LifecycleRestartExecutionOutcome {
  LifecycleRestartExecutionDisposition disposition{
      LifecycleRestartExecutionDisposition::kFailedStopped};
  LifecycleExecutionFailure primary_failure;
  LifecycleExecutionFailure rollback_failure;
  LifecycleExecutionFailure restoration_failure;
  LifecycleExecutionFailure restoration_rollback_failure;
};

static_assert(std::is_standard_layout_v<LifecycleRestartExecutionOutcome>);
static_assert(std::is_trivially_copyable_v<LifecycleRestartExecutionOutcome>);

struct RuntimeLifecycleFailure {
  VertexId vertex{};
  LifecycleExecutionFailure failure;
  DiagnosticCapture capture;
  std::string_view component_path;
  LifecycleTransactionId transaction_id{};
  std::string_view token;
};

/// Execute the immutable lifecycle stage graph over freshly replayed Operators.
class RuntimeLifecycleExecutor {
 public:
  RuntimeLifecycleExecutor(const RuntimeLifecycleExecutor&) = delete;
  RuntimeLifecycleExecutor& operator=(const RuntimeLifecycleExecutor&) = delete;
  RuntimeLifecycleExecutor(RuntimeLifecycleExecutor&&) = delete;
  RuntimeLifecycleExecutor& operator=(RuntimeLifecycleExecutor&&) = delete;
  ~RuntimeLifecycleExecutor();

  [[nodiscard]] static expected<std::unique_ptr<RuntimeLifecycleExecutor>, Error> create(
      const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
      RunGenerationId run_generation, const std::atomic_bool* cancellation, GraphExec* exec,
      ExecutionLaneSet* lanes, const RuntimePackageCapabilityMatchReport& package,
      RuntimeNotificationSourceRegistry* notifications = nullptr, RuntimeRateState* rates = nullptr,
      RuntimeRateMaterializationBinding rate_materialization = {});

  void clear_failure_evidence() noexcept {
    failure_count_ = 0U;
    transaction_id_ = 0U;
    callback_capture_.clear();
    result_ = {};
    failure_diagnostic_.store(0U, std::memory_order_release);
  }

  [[nodiscard]] bool startup() noexcept;
  [[nodiscard]] LifecycleRestartExecutionOutcome restart(
      const LifecycleRestartExecutionRequest& request) noexcept;
  [[nodiscard]] bool shutdown() noexcept;
  [[nodiscard]] std::string_view failure_token() const noexcept;
  [[nodiscard]] const LifecycleExecutionResult& result() const noexcept { return result_; }
  [[nodiscard]] std::span<const RuntimeLifecycleFailure> failures() const noexcept {
    return {failures_.data(), failure_count_};
  }

  /// Install the registry materialized by first ordered kAllocate work before kArm callbacks.
  [[nodiscard]] expected<void, Error> set_notifications(
      RuntimeNotificationSourceRegistry* notifications) noexcept;

 private:
  friend class RuntimeLifecycleExecutorTestAccess;

  struct CallbackReceiver;
  struct RestartRuntime;
  struct RestartTargetBinding;

  RuntimeLifecycleExecutor() = default;

  void record_callback_failure(const CallbackReceiver& bound, LifecycleStage stage,
                               LifecycleTransitionPhase phase, LifecycleStatus status) noexcept;
  [[nodiscard]] LifecycleStatus finish_callback(const CallbackReceiver& bound, LifecycleStage stage,
                                                LifecycleTransitionPhase phase,
                                                LifecycleStatus status) noexcept;
  [[nodiscard]] static LifecycleStatus invoke_callback(void* receiver, LifecycleStage stage,
                                                       LifecycleTransitionPhase phase) noexcept;
  [[nodiscard]] static expected<OperatorRef, Error> resolve_restart_target(
      void* receiver, OperatorRef plan_ref) noexcept;
  [[nodiscard]] static expected<OperatorRef, Error> resolve_restart_path(
      void* receiver, const OperatorPath& path) noexcept;
  [[nodiscard]] static expected<NotificationSender, Error> resolve_notification_sender(
      void* receiver, NotificationSource& source) noexcept;
  [[nodiscard]] static expected<void, Error> bind_nominal_rate(void* receiver,
                                                               const void* declaration,
                                                               Rate value) noexcept;
  [[nodiscard]] static expected<PortRateView, Error> resolve_rate(void* receiver,
                                                                  const void* declaration,
                                                                  bool output) noexcept;
  [[nodiscard]] static LifecycleStatus invoke_callback_direct(
      CallbackReceiver& bound, LifecycleStage stage, LifecycleTransitionPhase phase) noexcept;
  [[nodiscard]] static expected<LifecycleStageGraph, Error> build_graph(
      const ExecutionPlanState& plan, std::span<const OperatorInstanceId> members,
      std::span<const CallbackReceiver> receivers,
      const RuntimeMaterializationState* materialization, const RuntimeRateState* rates);
  [[nodiscard]] static std::vector<LifecycleCallbackBinding> bindings_for(
      std::span<CallbackReceiver> receivers, std::span<const OperatorInstanceId> members,
      bool include_materialization, bool include_rate_barriers);
  [[nodiscard]] static expected<void, Error> initialize_restart_runtime(
      const ExecutionPlanState& plan, std::span<CallbackReceiver> receivers,
      std::vector<OperatorInstanceId> members, bool union_graph, RestartRuntime& runtime);
  [[nodiscard]] std::size_t startup_position(OperatorInstanceId id) const noexcept;
  void record_failure_diagnostic(const Error& error) noexcept;

  const ExecutionPlanState* plan_{};
  GraphExec* exec_{};
  ExecutionLaneSet* lanes_{};
  RuntimeNotificationSourceRegistry* notifications_{};
  RuntimeRateState* rates_{};
  RuntimeRateMaterializationBinding rate_materialization_{};
  RunGenerationId run_generation_{};
  const std::atomic_bool* cancellation_{};
  std::span<RuntimeOperatorInstance> operators_;
  std::vector<CallbackReceiver> receivers_;
  std::optional<RuntimeMaterializationState> materialization_;
  LifecycleStageGraph graph_;
  std::vector<LifecycleCallbackBinding> bindings_;
  LifecycleKernelState state_;
  std::vector<RestartRuntime> restarts_;
  std::vector<RestartTargetBinding> restart_target_bindings_;
  std::span<const OperatorInstanceId> active_restart_members_;
  std::span<const LifecycleRestartParameterCandidate> active_restart_candidates_;
  DiagnosticCapture callback_capture_;
  std::uint32_t callback_diagnostic_{};
  std::array<RuntimeLifecycleFailure, 4U> failures_;
  LifecycleExecutionResult result_;
  std::size_t failure_count_{};
  LifecycleTransactionId transaction_id_{};
  std::atomic<std::uint32_t> failure_diagnostic_{};
  bool startup_in_progress_{};
  std::size_t startup_legacy_stop_cutoff_{};
};

}  // namespace holoscan::detail
