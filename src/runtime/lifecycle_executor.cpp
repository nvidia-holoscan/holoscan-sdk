/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lifecycle_executor.hpp"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/parameter_catalog.hpp>

#include "../compiler/execution_plan_lookup.hpp"
#include "../core/diagnostic_registry.hpp"
#include "../core/lifecycle_context_internal.hpp"
#include "../core/operator_runtime_access.hpp"
#include "../core/operator_spec_runtime_access.hpp"
#include "execution_lane.hpp"
#include "graph_exec.hpp"
#include "parameter_control_state.hpp"
#include "runtime_callback_context.hpp"
#include "runtime_notification_source.hpp"
#include "runtime_rate_state.hpp"

namespace holoscan::detail {
namespace {

constexpr std::size_t kForwardStageCount = 6U;

[[nodiscard]] LifecycleNodeId first_materialization_node_id(
    std::span<const OperatorInstanceId> members) noexcept {
  return members.empty() ? 1U : static_cast<LifecycleNodeId>(members.back() + 1U);
}

[[nodiscard]] LifecycleNodeId materialization_node_id(std::span<const OperatorInstanceId> members,
                                                      std::size_t index) noexcept {
  return static_cast<LifecycleNodeId>(first_materialization_node_id(members) + index);
}

[[nodiscard]] std::optional<LifecycleNodeId> rate_barrier_node_id(
    const ExecutionPlanState& plan) noexcept {
  OperatorInstanceId maximum_operator{};
  for (const auto& factory : plan.executable_factories.operators) {
    maximum_operator = std::max(maximum_operator, factory.operator_id);
  }
  const std::size_t materialization_count =
      plan.profile_manifest.materialization_obligations.size();
  if (maximum_operator == std::numeric_limits<LifecycleNodeId>::max() ||
      materialization_count > std::numeric_limits<LifecycleNodeId>::max() - maximum_operator - 1U) {
    return std::nullopt;
  }
  return static_cast<LifecycleNodeId>(maximum_operator + materialization_count + 1U);
}

[[nodiscard]] bool has_rate_barriers(const RuntimeRateState* rates) noexcept {
  if (rates == nullptr) {
    return false;
  }
  for (std::size_t index = 0U; index < kForwardStageCount; ++index) {
    if (rates->requires_barrier(static_cast<LifecycleStage>(index))) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool has_rate_work(const RuntimeRateState* rates) noexcept {
  return rates != nullptr &&
         (has_rate_barriers(rates) || rates->requires_resource_materialization());
}

[[nodiscard]] bool contains(std::span<const OperatorInstanceId> members,
                            OperatorInstanceId id) noexcept {
  return std::binary_search(members.begin(), members.end(), id);
}

[[nodiscard]] RuntimeOperatorInstance* find_operator(std::span<RuntimeOperatorInstance> operators,
                                                     OperatorInstanceId id) noexcept {
  const auto found = std::find_if(operators.begin(), operators.end(), [id](const auto& instance) {
    return instance.operator_id == id;
  });
  return found == operators.end() ? nullptr : &*found;
}

[[nodiscard]] Error lifecycle_error(std::string token) {
  return Error{ErrorCode::kFailure, std::move(token)};
}

[[nodiscard]] bool checkpoint_levels_match(const LifecycleStageGraph& graph,
                                           const ExecutionPlanState& plan,
                                           std::span<const OperatorInstanceId> members) noexcept {
  for (const auto& checkpoint : plan.lifecycle_checkpoints) {
    if (!contains(members, checkpoint.operator_instance.id)) {
      continue;
    }
    if (static_cast<std::size_t>(checkpoint.stage) >= graph.stages.size()) {
      return false;
    }
    const auto& stage = graph.stages[static_cast<std::size_t>(checkpoint.stage)];
    const auto level = std::find_if(stage.levels.begin(), stage.levels.end(), [&](const auto& row) {
      return std::find(row.node_ids.begin(), row.node_ids.end(), checkpoint.operator_instance.id) !=
             row.node_ids.end();
    });
    if (level == stage.levels.end() ||
        checkpoint.stable_topological_level !=
            static_cast<std::uint32_t>(std::distance(stage.levels.begin(), level))) {
      return false;
    }
  }
  return true;
}

void apply_runtime_stop_order(LifecycleStageGraph& graph) {
  std::vector<LifecycleNodeId> start_order;
  start_order.reserve(graph.nodes.size());
  const auto& start = graph.stages[static_cast<std::size_t>(LifecycleStage::kStart)];
  for (const auto& level : start.levels) {
    start_order.insert(start_order.end(), level.node_ids.begin(), level.node_ids.end());
  }

  auto& stop = graph.stages[static_cast<std::size_t>(LifecycleStage::kStop)];
  for (auto& level : stop.levels) {
    // Exact-stage stop dependencies already separate predecessors and successors into
    // topological levels. Reverse only independent nodes within a level so ordinary legacy
    // stop hooks unwind successful startup order without overriding authored teardown edges.
    std::sort(level.node_ids.begin(), level.node_ids.end(), [&](const auto lhs, const auto rhs) {
      const auto lhs_start = std::find(start_order.begin(), start_order.end(), lhs);
      const auto rhs_start = std::find(start_order.begin(), start_order.end(), rhs);
      return lhs_start > rhs_start;
    });
  }
}

}  // namespace

struct RuntimeLifecycleExecutor::CallbackReceiver {
  RuntimeLifecycleExecutor* executor{};
  RuntimeOperatorInstance* instance{};
  ExecutionLaneSet* lanes{};
  LifecycleNodeId node_id{};
  std::optional<std::size_t> materialization_index;
  bool rate_barrier{};
  std::size_t callback_index{};
  LifecycleStage stage{LifecycleStage::kConfigure};
  bool advanced{};
  std::unique_ptr<LifecycleContextState> context;
  std::optional<LifecycleParameterViewSelection> creating_parameter_view;
};

struct RuntimeLifecycleExecutor::RestartRuntime {
  std::vector<OperatorInstanceId> members;
  LifecycleStageGraph graph;
  std::vector<LifecycleCallbackBinding> bindings;
  LifecycleKernelState state;
  bool union_graph{};
};

struct RuntimeLifecycleExecutor::RestartTargetBinding {
  OperatorRef target{};
  OperatorPath path;
  LifecycleRestartTargetDisposition disposition{LifecycleRestartTargetDisposition::kDisabled};
  bool self_allowed{};
};

RuntimeLifecycleExecutor::~RuntimeLifecycleExecutor() = default;

expected<OperatorRef, Error> RuntimeLifecycleExecutor::resolve_restart_target(
    void* receiver, OperatorRef plan_ref) noexcept {
  auto* const bound = static_cast<CallbackReceiver*>(receiver);
  if (bound == nullptr || bound->executor == nullptr || bound->instance == nullptr) {
    return make_unexpected(Error{ErrorCode::kLifecycleInvalidCallingContext});
  }
  RuntimeLifecycleExecutor& executor = *bound->executor;
  if (plan_ref.run_generation != 0U) {
    return make_unexpected(Error{plan_ref.run_generation == executor.run_generation_
                                     ? ErrorCode::kStaleOperatorRef
                                     : ErrorCode::kCrossSessionOperatorRef});
  }
  if (!plan_ref.plan_bound() || executor.plan_ == nullptr ||
      plan_ref.plan_generation != executor.plan_->plan_generation) {
    return make_unexpected(Error{ErrorCode::kStaleOperatorRef});
  }

  const auto target = std::find_if(
      executor.restart_target_bindings_.begin(),
      executor.restart_target_bindings_.end(),
      [&](const RestartTargetBinding& candidate) { return candidate.target == plan_ref; });
  if (target == executor.restart_target_bindings_.end()) {
    return make_unexpected(Error{ErrorCode::kStaleOperatorRef});
  }
  if (target->disposition != LifecycleRestartTargetDisposition::kAdmitted) {
    return make_unexpected(Error{ErrorCode::kLifecycleRestartUnauthorized});
  }

  const OperatorInstanceId requester = bound->instance->operator_id;
  const OperatorRef requester_ref{
      .id = requester,
      .plan_generation = executor.plan_->plan_generation,
      .run_generation = 0U,
  };
  const bool self_authorized = requester == target->target.id && target->self_allowed;
  const bool cross_authorized = requester != target->target.id &&
                                std::any_of(executor.plan_->lifecycle_restart_permissions.begin(),
                                            executor.plan_->lifecycle_restart_permissions.end(),
                                            [&](const auto& permission) {
                                              return permission.requester == requester_ref &&
                                                     permission.target == target->target;
                                            });
  if (!self_authorized && !cross_authorized) {
    return make_unexpected(Error{ErrorCode::kLifecycleRestartUnauthorized});
  }

  OperatorRef result = target->target;
  result.run_generation = executor.run_generation_;
  return result;
}

expected<OperatorRef, Error> RuntimeLifecycleExecutor::resolve_restart_path(
    void* receiver, const OperatorPath& path) noexcept {
  auto* const bound = static_cast<CallbackReceiver*>(receiver);
  if (bound == nullptr || bound->executor == nullptr || bound->instance == nullptr) {
    return make_unexpected(Error{ErrorCode::kLifecycleInvalidCallingContext});
  }
  RuntimeLifecycleExecutor& executor = *bound->executor;
  const auto target =
      std::find_if(executor.restart_target_bindings_.begin(),
                   executor.restart_target_bindings_.end(),
                   [&](const RestartTargetBinding& candidate) { return candidate.path == path; });
  if (target == executor.restart_target_bindings_.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  return resolve_restart_target(receiver, target->target);
}

expected<NotificationSender, Error> RuntimeLifecycleExecutor::resolve_notification_sender(
    void* receiver, NotificationSource& source) noexcept {
  auto* const bound = static_cast<CallbackReceiver*>(receiver);
  if (bound == nullptr || bound->executor == nullptr || bound->instance == nullptr ||
      bound->context == nullptr || bound->executor->notifications_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kLifecycleInvalidCallingContext});
  }
  return bound->executor->notifications_->acquire_sender(
      bound->instance->vertex, source, bound->stage, bound->context->phase);
}

expected<void, Error> RuntimeLifecycleExecutor::bind_nominal_rate(void* receiver,
                                                                  const void* declaration,
                                                                  Rate value) noexcept {
  auto* const bound = static_cast<CallbackReceiver*>(receiver);
  if (bound == nullptr || bound->executor == nullptr || bound->executor->rates_ == nullptr ||
      bound->instance == nullptr || bound->context == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kLifecycleInvalidCallingContext, "PORT_RATE_BIND_NOT_AUTHORIZED"});
  }
  auto result = bound->executor->rates_->bind_nominal_rate(
      bound->instance->operator_id, bound->stage, bound->context->phase, declaration, value);
  const bool forward = bound->context->phase == LifecycleTransitionPhase::kStartupForward ||
                       bound->context->phase == LifecycleTransitionPhase::kRestartForward;
  if (!result && forward) {
    bound->executor->record_failure_diagnostic(result.error().message);
  }
  return result;
}

expected<PortRateView, Error> RuntimeLifecycleExecutor::resolve_rate(void* receiver,
                                                                     const void* declaration,
                                                                     bool output) noexcept {
  auto* const bound = static_cast<CallbackReceiver*>(receiver);
  if (bound == nullptr || bound->executor == nullptr || bound->executor->rates_ == nullptr ||
      bound->instance == nullptr || bound->context == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "PORT_RATE_VIEW_INVALID"});
  }
  return bound->executor->rates_->resolve_lifecycle(
      bound->instance->operator_id, bound->stage, bound->context->phase, declaration, output);
}

LifecycleStatus RuntimeLifecycleExecutor::invoke_callback(void* receiver, LifecycleStage stage,
                                                          LifecycleTransitionPhase phase) noexcept {
  auto& bound = *static_cast<CallbackReceiver*>(receiver);
  const auto finish = [&](LifecycleStatus status) noexcept {
    if (bound.executor != nullptr && bound.executor->startup_in_progress_ &&
        phase == LifecycleTransitionPhase::kStartupForward && status != LifecycleStatus::kOk &&
        status != LifecycleStatus::kSkipped) {
      // Apply startup rollback bookkeeping at the single kernel callback
      // boundary so failures during materialization, lane dispatch, Parameter
      // pinning, or user invocation all exclude legacy stop() callbacks whose
      // matching start() was never entered.
      bound.executor->startup_legacy_stop_cutoff_ =
          !bound.materialization_index.has_value() && !bound.rate_barrier &&
                  bound.instance != nullptr && stage == LifecycleStage::kStart
              ? bound.executor->startup_position(bound.instance->operator_id)
              : 0U;
    }
    return status;
  };
  if (bound.materialization_index.has_value() || bound.rate_barrier) {
    return finish(invoke_callback_direct(bound, stage, phase));
  }
  if (bound.instance == nullptr || bound.lanes == nullptr) {
    return finish(LifecycleStatus::kFatalFailure);
  }
  if (!bound.executor->active_restart_members_.empty() &&
      !contains(bound.executor->active_restart_members_, bound.instance->operator_id)) {
    return finish(LifecycleStatus::kSkipped);
  }
  struct LaneInvocation {
    CallbackReceiver* receiver{};
    LifecycleStage stage{LifecycleStage::kConfigure};
    LifecycleTransitionPhase phase{LifecycleTransitionPhase::kStartupForward};
    LifecycleStatus result{LifecycleStatus::kFatalFailure};
  } invocation{
      .receiver = &bound,
      .stage = stage,
      .phase = phase,
  };
  const auto dispatched =
      bound.lanes->dispatch(bound.instance->execution_lane,
                            ExecutionLaneTask{
                                .context = &invocation,
                                .invoke = [](void* state) noexcept -> expected<void, Error> {
                                  auto& task = *static_cast<LaneInvocation*>(state);
                                  task.result = RuntimeLifecycleExecutor::invoke_callback_direct(
                                      *task.receiver, task.stage, task.phase);
                                  return {};
                                },
                                .callback_context = RuntimeCallbackContext::kLifecycle,
                            });
  return finish(dispatched ? invocation.result : LifecycleStatus::kFatalFailure);
}

// This callback is the single transaction boundary that validates the bound receiver, pins the
// phase-correct Parameter generation, records rollback identity, invokes either lifecycle API,
// and releases the pin. Keeping those ordered failure exits together makes the generation claim
// auditable and prevents a helper split from accidentally widening a pin lifetime.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
LifecycleStatus RuntimeLifecycleExecutor::invoke_callback_direct(
    CallbackReceiver& bound, LifecycleStage stage, LifecycleTransitionPhase phase) noexcept {
  const RuntimeCallbackContextScope callback_identity_scope{RuntimeCallbackIdentity{
      .context = RuntimeCallbackContext::kLifecycle,
      .operator_id = bound.instance == nullptr ? 0U : bound.instance->operator_id,
      .lifecycle_stage = stage,
      .lifecycle_phase = phase,
  }};
  if (bound.rate_barrier) {
    if (bound.executor == nullptr || bound.executor->rates_ == nullptr || bound.stage != stage) {
      return LifecycleStatus::kFatalFailure;
    }
    if (bound.executor->rates_->requires_barrier(stage)) {
      auto closed = bound.executor->rates_->close_stage_barrier(stage, phase);
      if (!closed) {
        bound.executor->record_failure_diagnostic(closed.error().message);
        return LifecycleStatus::kFatalFailure;
      }
    }
    const bool materialize = phase == LifecycleTransitionPhase::kStartupForward ||
                             phase == LifecycleTransitionPhase::kRestartForward;
    const bool apply_candidate =
        materialize && (stage == LifecycleStage::kAllocate || stage == LifecycleStage::kWarmUp ||
                        stage == LifecycleStage::kArm);
    if (!apply_candidate) {
      return LifecycleStatus::kOk;
    }
    auto& binding = bound.executor->rate_materialization_;
    if (binding.state == nullptr || binding.ensure == nullptr) {
      return LifecycleStatus::kFatalFailure;
    }
    struct Invocation {
      RuntimeRateMaterializationBinding* binding{};
      LifecycleStage stage{LifecycleStage::kAllocate};
      LifecycleTransitionPhase phase{LifecycleTransitionPhase::kStartupForward};
    } invocation{.binding = &binding, .stage = stage, .phase = phase};
    const bool require_all_resolved = stage == LifecycleStage::kArm;
    auto ensured = bound.executor->rates_->visit_candidate_materialization(
        &invocation,
        [](void* state,
           const RuntimeRateMaterializationView& view) noexcept -> expected<void, Error> {
          auto& call = *static_cast<Invocation*>(state);
          return call.binding->ensure(call.binding->state, view, call.stage, call.phase);
        },
        require_all_resolved);
    if (!ensured) {
      bound.executor->record_failure_diagnostic(ensured.error().message);
    }
    return ensured ? LifecycleStatus::kOk : LifecycleStatus::kFatalFailure;
  }
  if (bound.materialization_index.has_value()) {
    if (bound.executor == nullptr || !bound.executor->materialization_.has_value() ||
        bound.stage != stage) {
      return LifecycleStatus::kFatalFailure;
    }
    return bound.executor->materialization_->transition(*bound.materialization_index, stage, phase);
  }
  if (bound.instance == nullptr || bound.instance->op == nullptr || bound.stage != stage) {
    return LifecycleStatus::kFatalFailure;
  }
  if (bound.context == nullptr) {
    return LifecycleStatus::kFatalFailure;
  }
  if (bound.executor == nullptr || bound.executor->exec_ == nullptr) {
    return LifecycleStatus::kFatalFailure;
  }
  if (!bound.advanced && stage == LifecycleStage::kStop && bound.executor->startup_in_progress_ &&
      phase == LifecycleTransitionPhase::kForwardRollback &&
      bound.executor->startup_position(bound.instance->operator_id) >=
          bound.executor->startup_legacy_stop_cutoff_) {
    // A startup failure can occur while preparing a later legacy start()
    // callback, before user code is entered. Filter the matching rollback
    // stop() here, before Parameter pinning or any other callback preparation,
    // so an intentionally unstarted Operator cannot turn rollback itself into
    // a second failure or prevent earlier started Operators from stopping.
    return LifecycleStatus::kSkipped;
  }

  ParameterUpdateCoordinator& parameters = bound.executor->exec_->parameter_updates();
  const OperatorRef owner{
      .id = bound.instance->operator_id,
      .plan_generation = bound.executor->plan_->plan_generation,
      .run_generation = bound.executor->run_generation_,
  };
  auto published = parameters.pin_published(owner);
  if (!published || published->header() == nullptr) {
    return LifecycleStatus::kFatalFailure;
  }
  const auto private_candidate =
      std::find_if(bound.executor->active_restart_candidates_.begin(),
                   bound.executor->active_restart_candidates_.end(),
                   [&](const LifecycleRestartParameterCandidate& candidate) {
                     return candidate.owner == owner;
                   });
  const ParamGeneration published_generation = published->header()->generation;
  const LifecycleParameterViewInputs view_inputs{
      .old_published = published_generation,
      .current_published = published_generation,
      .private_candidate = private_candidate == bound.executor->active_restart_candidates_.end()
                               ? std::optional<ParamGeneration>{}
                               : std::optional<ParamGeneration>{private_candidate->generation},
      .creating_view = bound.creating_parameter_view,
  };
  auto selected = select_lifecycle_parameter_view(phase, view_inputs);
  if (!selected) {
    parameters.release(*published);
    return LifecycleStatus::kFatalFailure;
  }
  if (phase == LifecycleTransitionPhase::kStartupForward ||
      phase == LifecycleTransitionPhase::kRestartForward) {
    // Rollback invokes the node's Stop/Release receiver, not necessarily the receiver that
    // created the forward work. Retain the exact startup-zero or restart-candidate view across
    // every callback receiver for this Operator so rollback cannot accidentally consult another
    // generation.
    for (auto& receiver : bound.executor->receivers_) {
      if (receiver.instance == bound.instance) {
        receiver.creating_parameter_view = *selected;
      }
    }
  }

  ParameterGenerationPin selected_pin;
  if (selected->generation == published_generation) {
    selected_pin = std::move(*published);
  } else {
    parameters.release(*published);
    auto candidate = parameters.pin_lifecycle_generation(owner, selected->generation);
    if (!candidate || candidate->header() == nullptr) {
      return LifecycleStatus::kFatalFailure;
    }
    selected_pin = std::move(*candidate);
  }
  bound.context->phase = phase;
  bound.context->params = selected_pin.header()->value;
  bound.context->params_type = selected_pin.header()->local_type_token;
  bound.context->params_generation = selected_pin.header()->generation;
  if (!bound.advanced) {
    // Legacy start()/stop() callbacks have no LifecycleContext argument. Keep
    // their typed accessor coherent with the exact phase view selected above;
    // a later rollback or restore callback replaces it again with that phase's
    // old-published view before invoking user code. Advanced callbacks consume
    // the pinned view directly through LifecycleContext instead.
    OperatorRuntimeAccess::replace_runtime_parameter_value(*bound.instance->op,
                                                           selected_pin.header()->value);
  }
  const auto invoke_user = [&]() noexcept {
    const ParamGeneralControlContextScope general_control_scope;
    if (bound.advanced) {
      if (bound.instance->replay_spec == nullptr) {
        return LifecycleStatus::kFatalFailure;
      }
      return OperatorSpecRuntimeAccess::invoke(*bound.instance->replay_spec,
                                               bound.callback_index,
                                               bound.instance->op.get(),
                                               *bound.context);
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      if (stage == LifecycleStage::kStart) {
        bound.instance->op->start();
        return LifecycleStatus::kOk;
      }
      if (stage == LifecycleStage::kStop) {
        bound.instance->op->stop();
        return LifecycleStatus::kOk;
      }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      return LifecycleStatus::kFatalFailure;
    }
#endif
    return LifecycleStatus::kFatalFailure;
  };
  const LifecycleStatus status = invoke_user();
  parameters.release(selected_pin);
  return status;
}

// Build the complete lifecycle/materialization graph as one fail-closed transaction so callback
// and cleanup edges cannot be accepted from different partial views.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<LifecycleStageGraph, Error> RuntimeLifecycleExecutor::build_graph(
    const ExecutionPlanState& plan, std::span<const OperatorInstanceId> members,
    std::span<const CallbackReceiver> receivers, const RuntimeMaterializationState* materialization,
    const RuntimeRateState* rates) {
  LifecycleStageGraphInput input;
  input.scopes.push_back({.scope_id = 1U, .parent_scope_id = std::nullopt});
  const std::size_t materialization_count =
      materialization == nullptr ? 0U : materialization->size();
  const bool include_rate_barrier = has_rate_work(rates);
  const auto rate_node = include_rate_barrier ? rate_barrier_node_id(plan) : std::nullopt;
  if (include_rate_barrier && !rate_node) {
    return make_unexpected(lifecycle_error("RUNTIME_LIFECYCLE_PLAN_INVALID"));
  }
  if (materialization_count != 0U &&
      ((!members.empty() && members.back() == std::numeric_limits<OperatorInstanceId>::max()) ||
       materialization_count - 1U >
           std::numeric_limits<LifecycleNodeId>::max() - first_materialization_node_id(members))) {
    return make_unexpected(lifecycle_error("PROFILE_MATERIALIZATION_FAILED"));
  }
  input.nodes.reserve(members.size() + materialization_count + (include_rate_barrier ? 1U : 0U));
  for (const OperatorInstanceId id : members) {
    const auto factory =
        std::find_if(plan.executable_factories.operators.begin(),
                     plan.executable_factories.operators.end(),
                     [id](const auto& record) { return record.operator_id == id; });
    if (factory == plan.executable_factories.operators.end()) {
      return make_unexpected(lifecycle_error("RUNTIME_OPERATOR_INVENTORY_MISMATCH"));
    }
    const auto vertex = std::find_if(
        plan.graph.vertices().begin(), plan.graph.vertices().end(), [&](const auto& record) {
          return record.id == factory->vertex;
        });
    if (vertex == plan.graph.vertices().end()) {
      return make_unexpected(lifecycle_error("RUNTIME_OPERATOR_INVENTORY_MISMATCH"));
    }
    const std::size_t parameter_index = static_cast<std::size_t>(id - 1U);
    if (parameter_index >= plan.operator_parameters.size()) {
      return make_unexpected(lifecycle_error("RUNTIME_OPERATOR_INVENTORY_MISMATCH"));
    }
    const auto& parameters = plan.operator_parameters[parameter_index];
    if (parameters.owner.id != id || parameters.owner.plan_generation != plan.plan_generation ||
        parameters.owner.run_generation != 0U || parameters.operator_path.empty()) {
      return make_unexpected(lifecycle_error("RUNTIME_OPERATOR_INVENTORY_MISMATCH"));
    }
    input.nodes.push_back({
        .node_id = id,
        .scope_id = 1U,
        .stable_name = parameters.operator_path,
        .stable_order = static_cast<std::uint32_t>(id - 1U),
    });
  }

  for (std::size_t index = 0U; index < materialization_count; ++index) {
    const auto& record = materialization->record(index);
    const LifecycleNodeId node_id = materialization_node_id(members, index);
    input.nodes.push_back({
        .node_id = node_id,
        .scope_id = 1U,
        .stable_name = "materialization:" + record.object_id + ":" + record.strategy_id,
        .stable_order = static_cast<std::uint32_t>(members.size() + index),
    });
    for (const LifecycleStage stage :
         {record.complete_before_stage, LifecycleStage::kStop, LifecycleStage::kRelease}) {
      input.callbacks.push_back({
          .node_id = node_id,
          .stage = stage,
          .required = true,
          .method = static_cast<LifecycleMethodId>(static_cast<std::uint8_t>(stage) + 1U),
          .origin = LifecycleLogicalCallbackOrigin::kAdvancedMember,
      });
    }
    for (const OperatorInstanceId member : members) {
      input.direct_edges.push_back({
          .declaration_scope_id = 1U,
          .before_node_id = node_id,
          .before_stage = record.complete_before_stage,
          .after_node_id = member,
          .after_stage = record.complete_before_stage,
      });
      for (const LifecycleStage stage : {LifecycleStage::kStop, LifecycleStage::kRelease}) {
        input.direct_edges.push_back({
            .declaration_scope_id = 1U,
            .before_node_id = member,
            .before_stage = stage,
            .after_node_id = node_id,
            .after_stage = stage,
        });
      }
    }
    if (index != 0U) {
      const auto& previous = materialization->record(index - 1U);
      const LifecycleNodeId previous_id = materialization_node_id(members, index - 1U);
      if (previous.complete_before_stage == record.complete_before_stage) {
        input.direct_edges.push_back({
            .declaration_scope_id = 1U,
            .before_node_id = previous_id,
            .before_stage = record.complete_before_stage,
            .after_node_id = node_id,
            .after_stage = record.complete_before_stage,
        });
      }
      for (const LifecycleStage stage : {LifecycleStage::kStop, LifecycleStage::kRelease}) {
        input.direct_edges.push_back({
            .declaration_scope_id = 1U,
            .before_node_id = node_id,
            .before_stage = stage,
            .after_node_id = previous_id,
            .after_stage = stage,
        });
      }
    }
  }

  if (include_rate_barrier) {
    input.nodes.push_back({
        .node_id = *rate_node,
        .scope_id = 1U,
        .stable_name = "runtime:port-rate-barrier",
        .stable_order = static_cast<std::uint32_t>(members.size() + materialization_count),
    });
    for (std::size_t index = 0U; index < kForwardStageCount; ++index) {
      const LifecycleStage stage = static_cast<LifecycleStage>(index);
      const bool rate_resolution_barrier = rates->requires_barrier(stage);
      const bool allocation_materialization =
          rates->requires_resource_materialization() && stage == LifecycleStage::kAllocate;
      if (!rate_resolution_barrier && !allocation_materialization) {
        continue;
      }
      input.callbacks.push_back({
          .node_id = *rate_node,
          .stage = stage,
          .required = true,
          .method = static_cast<LifecycleMethodId>(static_cast<std::uint8_t>(stage) + 1U),
          .origin = LifecycleLogicalCallbackOrigin::kAdvancedMember,
      });
      for (const OperatorInstanceId member : members) {
        input.direct_edges.push_back({
            .declaration_scope_id = 1U,
            .before_node_id = allocation_materialization ? *rate_node : member,
            .before_stage = stage,
            .after_node_id = allocation_materialization ? member : *rate_node,
            .after_stage = stage,
        });
      }
    }
  }

  for (const auto& checkpoint : plan.lifecycle_checkpoints) {
    if (!contains(members, checkpoint.operator_instance.id) || !checkpoint.callback_present) {
      continue;
    }
    const bool advanced =
        std::any_of(receivers.begin(), receivers.end(), [&](const auto& receiver) {
          return receiver.instance != nullptr &&
                 receiver.instance->operator_id == checkpoint.operator_instance.id &&
                 receiver.stage == checkpoint.stage && receiver.advanced;
        });
    input.callbacks.push_back({
        .node_id = checkpoint.operator_instance.id,
        .stage = checkpoint.stage,
        .required = checkpoint.required,
        .method = static_cast<LifecycleMethodId>(static_cast<std::uint8_t>(checkpoint.stage) + 1U),
        .origin = advanced ? LifecycleLogicalCallbackOrigin::kAdvancedMember
                  : checkpoint.stage == LifecycleStage::kStart
                      ? LifecycleLogicalCallbackOrigin::kLegacyStart
                  : checkpoint.stage == LifecycleStage::kStop
                      ? LifecycleLogicalCallbackOrigin::kLegacyStop
                      : LifecycleLogicalCallbackOrigin::kAdvancedMember,
    });
  }
  for (const auto& dependency : plan.lifecycle_dependencies) {
    if (!contains(members, dependency.before.id) || !contains(members, dependency.after.id)) {
      continue;
    }
    input.direct_edges.push_back({
        .declaration_scope_id = 1U,
        .before_node_id = dependency.before.id,
        .before_stage = dependency.stage,
        .after_node_id = dependency.after.id,
        .after_stage = dependency.stage,
    });
  }

  const LifecycleKernelCapacity capacity{
      .max_scopes = 1U,
      .max_nodes = input.nodes.size(),
      .max_callbacks = input.callbacks.size(),
      .max_dependency_records = 0U,
      .max_direct_edges = input.direct_edges.size(),
      .max_completions = input.nodes.size() * kForwardStageCount,
  };
  auto graph = build_lifecycle_stage_graph(input, capacity);
  if (!graph) {
    return make_unexpected(lifecycle_error("RUNTIME_LIFECYCLE_PLAN_INVALID"));
  }
  return std::move(graph).value();
}

std::vector<LifecycleCallbackBinding> RuntimeLifecycleExecutor::bindings_for(
    std::span<CallbackReceiver> receivers, std::span<const OperatorInstanceId> members,
    bool include_materialization, bool include_rate_barriers) {
  std::vector<LifecycleCallbackBinding> result;
  result.reserve(receivers.size());
  for (auto& receiver : receivers) {
    const bool selected_operator =
        receiver.instance != nullptr && contains(members, receiver.instance->operator_id);
    const bool selected_materialization =
        include_materialization && receiver.materialization_index.has_value();
    const bool selected_rate_barrier = include_rate_barriers && receiver.rate_barrier;
    if (selected_operator || selected_materialization || selected_rate_barrier) {
      result.push_back({
          .node_id = receiver.node_id,
          .stage = receiver.stage,
          .self = &receiver,
          .callback = &RuntimeLifecycleExecutor::invoke_callback,
      });
    }
  }
  return result;
}

expected<void, Error> RuntimeLifecycleExecutor::initialize_restart_runtime(
    const ExecutionPlanState& plan, std::span<CallbackReceiver> receivers,
    std::vector<OperatorInstanceId> members, bool union_graph, RestartRuntime& runtime) {
  const RuntimeRateState* const rates = receivers.empty() || receivers.front().executor == nullptr
                                            ? nullptr
                                            : receivers.front().executor->rates_;
  auto graph = build_graph(plan, members, receivers, nullptr, rates);
  if (!graph) {
    return make_unexpected(std::move(graph.error()));
  }
  runtime.members = std::move(members);
  runtime.graph = std::move(graph).value();
  runtime.bindings = bindings_for(receivers, runtime.members, false, has_rate_work(rates));
  runtime.state = make_lifecycle_kernel_state(runtime.graph);
  for (std::size_t stage_index = 0U; stage_index < kForwardStageCount; ++stage_index) {
    for (const auto& level : runtime.graph.stages[stage_index].levels) {
      for (const LifecycleNodeId node : level.node_ids) {
        runtime.state.completed.push_back(
            {.node_id = node, .stage = static_cast<LifecycleStage>(stage_index)});
      }
    }
  }
  runtime.state.steady_state = true;
  runtime.union_graph = union_graph;
  return {};
}

std::size_t RuntimeLifecycleExecutor::startup_position(OperatorInstanceId id) const noexcept {
  std::size_t position = 0U;
  const auto& start = graph_.stages[static_cast<std::size_t>(LifecycleStage::kStart)];
  for (const auto& level : start.levels) {
    for (const LifecycleNodeId node_id : level.node_ids) {
      if (node_id == id) {
        return position;
      }
      ++position;
    }
  }
  return position;
}

expected<void, Error> RuntimeLifecycleExecutor::set_notifications(
    RuntimeNotificationSourceRegistry* notifications) noexcept {
  if (notifications == nullptr || (notifications_ != nullptr && notifications_ != notifications)) {
    return make_unexpected(lifecycle_error("RUNTIME_LIFECYCLE_PLAN_INVALID"));
  }
  notifications_ = notifications;
  return {};
}

// Cold reconstruction validates every callback, materialization barrier, and restart closure
// before publishing an executor; keeping the inventory adjacent makes omissions fail closed.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<std::unique_ptr<RuntimeLifecycleExecutor>, Error> RuntimeLifecycleExecutor::create(
    const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
    RunGenerationId run_generation, const std::atomic_bool* cancellation, GraphExec* exec,
    ExecutionLaneSet* lanes, const RuntimePackageCapabilityMatchReport& package,
    RuntimeNotificationSourceRegistry* notifications, RuntimeRateState* rates,
    RuntimeRateMaterializationBinding rate_materialization) {
  if (lanes == nullptr || !lanes->ready()) {
    return make_unexpected(lifecycle_error("RUNTIME_LANES_NOT_READY"));
  }
  if (rates != nullptr && rates->requires_resource_materialization() &&
      (rate_materialization.state == nullptr || rate_materialization.ensure == nullptr)) {
    return make_unexpected(lifecycle_error("RUNTIME_LIFECYCLE_PLAN_INVALID"));
  }
  auto result = std::unique_ptr<RuntimeLifecycleExecutor>{new RuntimeLifecycleExecutor};
  result->plan_ = &plan;
  result->exec_ = exec;
  result->lanes_ = lanes;
  result->notifications_ = notifications;
  result->rates_ = rates;
  result->rate_materialization_ = rate_materialization;
  result->run_generation_ = run_generation;
  result->cancellation_ = cancellation;
  result->operators_ = operators;

  result->restart_target_bindings_.reserve(plan.lifecycle_restart_targets.size());
  for (const auto& target : plan.lifecycle_restart_targets) {
    if (find_operator(operators, target.target.id) == nullptr) {
      continue;
    }
    const auto* parameters = find_operator_parameters_by_owner(plan, target.target);
    if (parameters == nullptr) {
      return make_unexpected(lifecycle_error("RUNTIME_LIFECYCLE_PLAN_INVALID"));
    }
    auto path = OperatorPath::parse(parameters->operator_path);
    if (!path) {
      return make_unexpected(lifecycle_error("RUNTIME_LIFECYCLE_PLAN_INVALID"));
    }
    result->restart_target_bindings_.push_back({
        .target = target.target,
        .path = std::move(path).value(),
        .disposition = target.disposition,
        .self_allowed = target.self_allowed,
    });
  }

  std::vector<OperatorInstanceId> all_members;
  all_members.reserve(operators.size());
  for (auto& instance : operators) {
    all_members.push_back(instance.operator_id);
    if (instance.replay_spec == nullptr) {
      return make_unexpected(lifecycle_error("RUNTIME_OPERATOR_REPLAY_MISMATCH"));
    }
    std::vector<bool> advanced_stage(8U, false);
    for (std::size_t index = 0U;
         index < OperatorSpecRuntimeAccess::lifecycle_callback_count(*instance.replay_spec);
         ++index) {
      const LifecycleStage stage =
          OperatorSpecRuntimeAccess::lifecycle_stage(*instance.replay_spec, index);
      advanced_stage[static_cast<std::uint8_t>(stage)] = true;
      result->receivers_.push_back({
          .executor = result.get(),
          .instance = &instance,
          .lanes = lanes,
          .node_id = instance.operator_id,
          .callback_index = index,
          .stage = stage,
          .advanced = true,
          .context = std::make_unique<LifecycleContextState>(),
      });
    }
    for (const LifecycleStage stage : {LifecycleStage::kStart, LifecycleStage::kStop}) {
      const auto checkpoint =
          std::find_if(plan.lifecycle_checkpoints.begin(),
                       plan.lifecycle_checkpoints.end(),
                       [&](const auto& record) {
                         return record.operator_instance.id == instance.operator_id &&
                                record.stage == stage && record.callback_present;
                       });
      if (checkpoint != plan.lifecycle_checkpoints.end() &&
          !advanced_stage[static_cast<std::uint8_t>(stage)]) {
        result->receivers_.push_back({
            .executor = result.get(),
            .instance = &instance,
            .lanes = lanes,
            .node_id = instance.operator_id,
            .callback_index = 0U,
            .stage = stage,
            .advanced = false,
            .context = std::make_unique<LifecycleContextState>(),
        });
      }
    }
  }
  std::sort(all_members.begin(), all_members.end());
  if (has_rate_work(rates)) {
    const auto node_id = rate_barrier_node_id(plan);
    if (!node_id) {
      return make_unexpected(lifecycle_error("RUNTIME_LIFECYCLE_PLAN_INVALID"));
    }
    for (std::size_t index = 0U; index < kForwardStageCount; ++index) {
      const LifecycleStage stage = static_cast<LifecycleStage>(index);
      const bool rate_resolution_barrier = rates->requires_barrier(stage);
      const bool allocation_materialization =
          rates->requires_resource_materialization() && stage == LifecycleStage::kAllocate;
      if (!rate_resolution_barrier && !allocation_materialization) {
        continue;
      }
      result->receivers_.push_back({
          .executor = result.get(),
          .instance = nullptr,
          .lanes = nullptr,
          .node_id = *node_id,
          .materialization_index = std::nullopt,
          .rate_barrier = true,
          .callback_index = 0U,
          .stage = stage,
          .advanced = false,
          .context = nullptr,
      });
    }
  }
  // Compiler checkpoints describe the authored logical callback graph. Runtime-only
  // rate barriers and resource materializers intentionally add ordering levels, but
  // must not make those immutable authored checkpoint levels appear stale.
  auto logical_graph = build_graph(plan, all_members, result->receivers_, nullptr, nullptr);
  if (!logical_graph || !checkpoint_levels_match(logical_graph.value(), plan, all_members)) {
    return make_unexpected(lifecycle_error("RUNTIME_LIFECYCLE_PLAN_INVALID"));
  }

  auto materialization = RuntimeMaterializationState::create(plan, package, lanes);
  if (!materialization) {
    return make_unexpected(std::move(materialization.error()));
  }
  result->materialization_ = std::move(materialization).value();
  for (std::size_t index = 0U; index < result->materialization_->size(); ++index) {
    const auto& record = result->materialization_->record(index);
    const LifecycleNodeId node_id = materialization_node_id(all_members, index);
    for (const LifecycleStage stage :
         {record.complete_before_stage, LifecycleStage::kStop, LifecycleStage::kRelease}) {
      result->receivers_.push_back({
          .executor = result.get(),
          .instance = nullptr,
          .lanes = nullptr,
          .node_id = node_id,
          .materialization_index = index,
          .callback_index = 0U,
          .stage = stage,
          .advanced = false,
          .context = nullptr,
      });
    }
  }

  for (auto& receiver : result->receivers_) {
    if (receiver.context == nullptr) {
      continue;
    }
    // Lifecycle callbacks borrow the run-owned monotonic latch directly so a
    // stop requested during a restart becomes visible inside that callback.
    // The context-local atomic remains the fallback used by isolated tests.
    receiver.context->cancellation_source = cancellation;
    receiver.context->service_state = &receiver;
    receiver.context->notification_sender = &RuntimeLifecycleExecutor::resolve_notification_sender;
    receiver.context->restart_target = &RuntimeLifecycleExecutor::resolve_restart_target;
    receiver.context->restart_path = &RuntimeLifecycleExecutor::resolve_restart_path;
    receiver.context->rate_state = &receiver;
    receiver.context->bind_nominal_rate = &RuntimeLifecycleExecutor::bind_nominal_rate;
    receiver.context->resolve_rate = &RuntimeLifecycleExecutor::resolve_rate;
  }

  auto graph =
      build_graph(plan, all_members, result->receivers_, &*result->materialization_, rates);
  if (!graph) {
    return make_unexpected(std::move(graph.error()));
  }
  result->graph_ = std::move(graph).value();
  apply_runtime_stop_order(result->graph_);
  result->bindings_ = bindings_for(result->receivers_, all_members, true, has_rate_work(rates));
  result->state_ = make_lifecycle_kernel_state(result->graph_);

  result->restarts_.reserve(plan.lifecycle_restart_closures.size() + 1U);
  std::vector<OperatorInstanceId> union_members;
  union_members.reserve(all_members.size());
  for (const auto& closure : plan.lifecycle_restart_closures) {
    if (find_operator(operators, closure.target.id) == nullptr) {
      continue;
    }
    const auto target =
        std::find_if(plan.lifecycle_restart_targets.begin(),
                     plan.lifecycle_restart_targets.end(),
                     [&](const auto& record) { return record.target.id == closure.target.id; });
    if (target == plan.lifecycle_restart_targets.end() ||
        target->disposition != LifecycleRestartTargetDisposition::kAdmitted) {
      continue;
    }
    std::vector<OperatorInstanceId> members;
    members.reserve(closure.closure.size());
    for (const OperatorRef ref : closure.closure) {
      if (find_operator(operators, ref.id) == nullptr) {
        return make_unexpected(lifecycle_error("RUNTIME_LIFECYCLE_PLAN_INVALID"));
      }
      members.push_back(ref.id);
      const auto position = std::lower_bound(union_members.begin(), union_members.end(), ref.id);
      if (position == union_members.end() || *position != ref.id) {
        union_members.insert(position, ref.id);
      }
    }
    RestartRuntime runtime;
    auto initialized =
        initialize_restart_runtime(plan, result->receivers_, std::move(members), false, runtime);
    if (!initialized) {
      return make_unexpected(std::move(initialized.error()));
    }
    result->restarts_.push_back(std::move(runtime));
  }
  const auto existing_union =
      std::find_if(result->restarts_.begin(), result->restarts_.end(), [&](const auto& runtime) {
        return runtime.members == union_members;
      });
  if (existing_union != result->restarts_.end()) {
    existing_union->union_graph = true;
  } else if (!union_members.empty()) {
    RestartRuntime runtime;
    auto initialized = initialize_restart_runtime(
        plan, result->receivers_, std::move(union_members), true, runtime);
    if (!initialized) {
      return make_unexpected(std::move(initialized.error()));
    }
    result->restarts_.push_back(std::move(runtime));
  }
  return result;
}

bool RuntimeLifecycleExecutor::startup() noexcept {
  failure_diagnostic_.store(0U, std::memory_order_release);
  for (auto& receiver : receivers_) {
    if (receiver.context != nullptr) {
      receiver.context->cancellation_requested.store(
          cancellation_ != nullptr && cancellation_->load(std::memory_order_acquire),
          std::memory_order_release);
    }
  }
  startup_in_progress_ = true;
  startup_legacy_stop_cutoff_ = graph_.nodes.size();
  const auto result = execute_lifecycle_startup(graph_, bindings_, state_);
  startup_in_progress_ = false;
  if (result.disposition != LifecycleExecutionDisposition::kCompleted && materialization_) {
    materialization_->cleanup_remaining();
  }
  return result.disposition == LifecycleExecutionDisposition::kCompleted &&
         (!materialization_ || !materialization_->failed());
}

LifecycleRestartExecutionOutcome RuntimeLifecycleExecutor::restart(
    const LifecycleRestartExecutionRequest& request) noexcept {
  failure_diagnostic_.store(0U, std::memory_order_release);
  auto found = std::find_if(restarts_.begin(), restarts_.end(), [&](const auto& runtime) {
    return runtime.members.size() == request.members.size() &&
           std::equal(runtime.members.begin(), runtime.members.end(), request.members.begin());
  });
  if (found == restarts_.end()) {
    found = std::find_if(restarts_.begin(), restarts_.end(), [](const auto& runtime) {
      return runtime.union_graph;
    });
  }
  if (found == restarts_.end()) {
    return {};
  }
  for (auto& receiver : receivers_) {
    if (receiver.context != nullptr) {
      receiver.context->cancellation_requested.store(
          cancellation_ != nullptr && cancellation_->load(std::memory_order_acquire),
          std::memory_order_release);
    }
  }
  active_restart_members_ = request.members;
  active_restart_candidates_ = request.parameter_candidates;
  for (auto& receiver : receivers_) {
    receiver.creating_parameter_view.reset();
  }
  const LifecycleExecutionResult result =
      execute_lifecycle_restart(found->graph, found->bindings, request.stage, found->state);
  active_restart_members_ = {};
  active_restart_candidates_ = {};
  for (auto& receiver : receivers_) {
    receiver.creating_parameter_view.reset();
  }
  const LifecycleRestartExecutionDisposition disposition = [&] {
    switch (result.disposition) {
      case LifecycleExecutionDisposition::kCompleted:
        return LifecycleRestartExecutionDisposition::kPublished;
      case LifecycleExecutionDisposition::kRolledBack:
        return LifecycleRestartExecutionDisposition::kRolledBack;
      case LifecycleExecutionDisposition::kInvalidState:
      case LifecycleExecutionDisposition::kInvalidRestartStage:
      case LifecycleExecutionDisposition::kInvalidBinding:
      case LifecycleExecutionDisposition::kStageFailed:
      case LifecycleExecutionDisposition::kRollbackFailed:
      case LifecycleExecutionDisposition::kCapacityExceeded:
        return LifecycleRestartExecutionDisposition::kFailedStopped;
    }
    return LifecycleRestartExecutionDisposition::kFailedStopped;
  }();
  return {
      .disposition = disposition,
      .primary_failure = result.primary_failure,
      .rollback_failure = result.rollback_failure,
      .restoration_failure = result.restoration_failure,
      .restoration_rollback_failure = result.restoration_rollback_failure,
  };
}

bool RuntimeLifecycleExecutor::shutdown() noexcept {
  failure_diagnostic_.store(0U, std::memory_order_release);
  for (auto& receiver : receivers_) {
    if (receiver.context != nullptr) {
      receiver.context->cancellation_requested.store(true, std::memory_order_release);
    }
  }
  const auto result = execute_lifecycle_shutdown(graph_, bindings_, state_);
  if (materialization_) {
    materialization_->cleanup_remaining();
  }
  return result.disposition == LifecycleExecutionDisposition::kCompleted &&
         (!materialization_ || !materialization_->failed());
}

std::string_view RuntimeLifecycleExecutor::failure_token() const noexcept {
  const auto raw = failure_diagnostic_.load(std::memory_order_acquire);
  if (raw != 0U) {
    const auto* descriptor = find_structural_diagnostic(static_cast<StructuralDiagnosticId>(raw));
    if (descriptor != nullptr) {
      return descriptor->token;
    }
  }
  if (materialization_ && materialization_->failed()) {
    return "PROFILE_MATERIALIZATION_FAILED";
  }
  return "RUNTIME_LIFECYCLE_CALLBACK_FAILED";
}

void RuntimeLifecycleExecutor::record_failure_diagnostic(const std::string_view token) noexcept {
  const std::uint32_t id = diagnostic_id(token).value;
  if (id == 0U) {
    return;
  }
  std::uint32_t expected{};
  (void)failure_diagnostic_.compare_exchange_strong(
      expected, id, std::memory_order_release, std::memory_order_relaxed);
}

}  // namespace holoscan::detail
