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
#include "../core/lifecycle_kernel.hpp"
#include "../core/lifecycle_parameter_view.hpp"
#include "runtime_materialization.hpp"

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
      RuntimeNotificationSourceRegistry* notifications = nullptr);

  [[nodiscard]] bool startup() noexcept;
  [[nodiscard]] LifecycleRestartExecutionOutcome restart(
      const LifecycleRestartExecutionRequest& request) noexcept;
  [[nodiscard]] bool shutdown() noexcept;
  [[nodiscard]] std::string_view failure_token() const noexcept;

 private:
  friend class RuntimeLifecycleExecutorTestAccess;

  struct CallbackReceiver;
  struct RestartRuntime;
  struct RestartTargetBinding;

  RuntimeLifecycleExecutor() = default;

  [[nodiscard]] static LifecycleStatus invoke_callback(void* receiver, LifecycleStage stage,
                                                       LifecycleTransitionPhase phase) noexcept;
  [[nodiscard]] static expected<OperatorRef, Error> resolve_restart_target(
      void* receiver, OperatorRef plan_ref) noexcept;
  [[nodiscard]] static expected<OperatorRef, Error> resolve_restart_path(
      void* receiver, const OperatorPath& path) noexcept;
  [[nodiscard]] static expected<NotificationSender, Error> resolve_notification_sender(
      void* receiver, NotificationSource& source) noexcept;
  [[nodiscard]] static LifecycleStatus invoke_callback_direct(
      CallbackReceiver& bound, LifecycleStage stage, LifecycleTransitionPhase phase) noexcept;
  [[nodiscard]] static expected<LifecycleStageGraph, Error> build_graph(
      const ExecutionPlanState& plan, std::span<const OperatorInstanceId> members,
      std::span<const CallbackReceiver> receivers,
      const RuntimeMaterializationState* materialization);
  [[nodiscard]] static std::vector<LifecycleCallbackBinding> bindings_for(
      std::span<CallbackReceiver> receivers, std::span<const OperatorInstanceId> members,
      bool include_materialization);
  [[nodiscard]] std::size_t startup_position(OperatorInstanceId id) const noexcept;

  const ExecutionPlanState* plan_{};
  GraphExec* exec_{};
  ExecutionLaneSet* lanes_{};
  RuntimeNotificationSourceRegistry* notifications_{};
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
  bool startup_in_progress_{};
  std::size_t startup_legacy_stop_cutoff_{};
};

}  // namespace holoscan::detail
