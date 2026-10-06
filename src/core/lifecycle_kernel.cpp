/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lifecycle_kernel.hpp"

#include "lifecycle_parameter_view.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>
#include "core/diagnostic_reporting.hpp"

namespace holoscan::detail {
namespace {

constexpr std::size_t kLifecycleStageCount = 8U;
constexpr std::size_t kForwardStageCount = 6U;

void default_lifecycle_log_sink(void*, const LifecycleLogRecord& record) {
  constexpr std::array operations{"startup", "restart", "shutdown"};
  constexpr std::array stages{
      "configure", "discover", "allocate", "warm_up", "arm", "start", "stop", "release"};
  constexpr std::array phases{"none",
                              "startup_forward",
                              "pre_restart_teardown",
                              "restart_forward",
                              "forward_rollback",
                              "restore_old",
                              "shutdown_teardown"};
  constexpr std::array statuses{"ok", "skipped", "retryable_failure", "fatal_failure"};
  constexpr std::array dispositions{"completed",
                                    "rolled_back",
                                    "invalid_state",
                                    "invalid_restart_stage",
                                    "invalid_binding",
                                    "stage_failed",
                                    "rollback_failed",
                                    "capacity_exceeded"};
  const auto name = [](auto value, const auto& names) noexcept {
    const auto index = static_cast<std::size_t>(value);
    return index < names.size() ? names[index] : "unknown";
  };
  switch (lifecycle_log_severity(record)) {
    case LifecycleLogSeverity::kDebug:
      if (record.kind == LifecycleLogEventKind::kTransitionBegin) {
        HOLOSCAN_LOG_DEBUG("Lifecycle transition begin: operation={}",
                           name(record.operation, operations));
      } else {
        HOLOSCAN_LOG_DEBUG("Lifecycle transition completed: operation={}",
                           name(record.operation, operations));
      }
      break;
    case LifecycleLogSeverity::kTrace:
      HOLOSCAN_LOG_TRACE("Lifecycle callback: event={} node={} stage={} phase={} status={}",
                         static_cast<unsigned>(record.kind),
                         record.node_id,
                         name(record.stage, stages),
                         name(record.phase, phases),
                         name(record.status, statuses));
      break;
    case LifecycleLogSeverity::kError:
      HOLOSCAN_LOG_AUTOMATIC_ERROR(
          "Lifecycle transition failed: operation={} disposition={} node={} "
          "stage={} phase={} status={}",
          name(record.operation, operations),
          name(record.disposition, dispositions),
          record.node_id,
          name(record.stage, stages),
          name(record.phase, phases),
          name(record.status, statuses));
      break;
  }
}

void emit_kernel_log(const LifecycleKernelState& state, const LifecycleLogRecord& record) noexcept {
  if (state.defer_failure_report &&
      lifecycle_log_severity(record) == LifecycleLogSeverity::kError) {
    return;
  }
  emit_lifecycle_log_noexcept(
      state.log_sink != nullptr ? state.log_sink : default_lifecycle_log_sink,
      state.log_sink != nullptr ? state.log_state : nullptr,
      record);
}

void log_transition_begin(LifecycleLogOperation operation,
                          const LifecycleKernelState& state) noexcept {
  emit_kernel_log(state, {.kind = LifecycleLogEventKind::kTransitionBegin, .operation = operation});
}

[[nodiscard]] LifecycleExecutionResult log_transition_end(
    LifecycleLogOperation operation, LifecycleExecutionResult result,
    const LifecycleKernelState& state) noexcept {
  const LifecycleExecutionFailure* failure = nullptr;
  if (result.primary_failure.present) {
    failure = &result.primary_failure;
  } else if (result.rollback_failure.present) {
    failure = &result.rollback_failure;
  } else if (result.restoration_failure.present) {
    failure = &result.restoration_failure;
  } else if (result.restoration_rollback_failure.present) {
    failure = &result.restoration_rollback_failure;
  }
  LifecycleLogRecord record{
      .kind = LifecycleLogEventKind::kTransitionEnd,
      .operation = operation,
      .disposition = result.disposition,
  };
  if (failure != nullptr) {
    record.node_id = failure->node_id;
    record.stage = failure->stage;
    record.phase = failure->phase;
    record.status = failure->status;
  }
  emit_kernel_log(state, record);
  return result;
}

[[nodiscard]] constexpr std::size_t stage_index(LifecycleStage stage) noexcept {
  switch (stage) {
    case LifecycleStage::kConfigure:
      return 0U;
    case LifecycleStage::kDiscover:
      return 1U;
    case LifecycleStage::kAllocate:
      return 2U;
    case LifecycleStage::kWarmUp:
      return 3U;
    case LifecycleStage::kArm:
      return 4U;
    case LifecycleStage::kStart:
      return 5U;
    case LifecycleStage::kStop:
      return 6U;
    case LifecycleStage::kRelease:
      return 7U;
  }
  return kLifecycleStageCount;
}

[[nodiscard]] constexpr LifecycleStage stage_from_index(std::size_t index) noexcept {
  return static_cast<LifecycleStage>(index);
}

[[nodiscard]] LifecycleKernelError plan_error(LifecycleKernelErrorCode code,
                                              LifecycleNodeId node_id = {},
                                              LifecycleScopeId scope_id = {},
                                              LifecycleStage stage = LifecycleStage::kConfigure,
                                              std::string stable_name = {}) {
  return LifecycleKernelError{
      .code = code,
      .scope_id = scope_id,
      .node_id = node_id,
      .stage = stage,
      .stable_name = std::move(stable_name),
  };
}

[[nodiscard]] const LifecycleLogicalNodeRecord* find_node(
    const std::vector<LifecycleLogicalNodeRecord>& nodes, LifecycleNodeId node_id) noexcept {
  const auto match = std::find_if(
      nodes.begin(), nodes.end(), [node_id](const auto& node) { return node.node_id == node_id; });
  return match == nodes.end() ? nullptr : &*match;
}

[[nodiscard]] std::size_t find_node_index(const LifecycleStageGraph& graph,
                                          LifecycleNodeId node_id) noexcept {
  const auto match = std::find_if(graph.nodes.begin(),
                                  graph.nodes.end(),
                                  [node_id](const auto& node) { return node.node_id == node_id; });
  return match == graph.nodes.end()
             ? graph.nodes.size()
             : static_cast<std::size_t>(std::distance(graph.nodes.begin(), match));
}

[[nodiscard]] bool node_before(const LifecycleLogicalNodeRecord& lhs,
                               const LifecycleLogicalNodeRecord& rhs) noexcept {
  if (lhs.stable_order != rhs.stable_order) {
    return lhs.stable_order < rhs.stable_order;
  }
  if (lhs.scope_id != rhs.scope_id) {
    return lhs.scope_id < rhs.scope_id;
  }
  if (lhs.stable_name != rhs.stable_name) {
    return lhs.stable_name < rhs.stable_name;
  }
  return lhs.node_id < rhs.node_id;
}

[[nodiscard]] bool key_equal(const LifecycleLogicalDependencyKey& lhs,
                             const LifecycleLogicalDependencyKey& rhs) noexcept {
  return lhs.scope_id == rhs.scope_id && lhs.stage == rhs.stage &&
         lhs.stable_name == rhs.stable_name;
}

[[nodiscard]] bool record_counts_fit(const LifecycleStageGraphInput& input,
                                     const LifecycleKernelCapacity& capacity) noexcept {
  if (input.scopes.empty() || input.scopes.size() > capacity.max_scopes || input.nodes.empty() ||
      input.nodes.size() > capacity.max_nodes || input.callbacks.size() > capacity.max_callbacks ||
      input.direct_edges.size() > capacity.max_direct_edges) {
    return false;
  }
  if (input.producers.size() > capacity.max_dependency_records ||
      input.requirements.size() > capacity.max_dependency_records - input.producers.size()) {
    return false;
  }
  if (input.nodes.size() > std::numeric_limits<std::size_t>::max() / kForwardStageCount) {
    return false;
  }
  return input.nodes.size() * kForwardStageCount <= capacity.max_completions;
}

struct NodeEdge {
  LifecycleNodeId before{};
  LifecycleNodeId after{};
};

[[nodiscard]] bool same_edge(const NodeEdge& lhs, const NodeEdge& rhs) noexcept {
  return lhs.before == rhs.before && lhs.after == rhs.after;
}

void append_unique_edge(std::vector<NodeEdge>& edges, NodeEdge edge) {
  if (std::find_if(edges.begin(), edges.end(), [&edge](const auto& existing) {
        return same_edge(existing, edge);
      }) == edges.end()) {
    edges.push_back(edge);
  }
}

[[nodiscard]] expected<LifecycleStagePlan, LifecycleKernelError> make_stage_plan(
    LifecycleStage stage, const std::vector<LifecycleLogicalNodeRecord>& nodes,
    const std::vector<NodeEdge>& edges) {
  const std::size_t node_count = nodes.size();
  std::vector<std::vector<std::size_t>> outgoing(node_count);
  std::vector<std::size_t> indegree(node_count, 0U);

  for (const auto& edge : edges) {
    const auto* before = find_node(nodes, edge.before);
    const auto* after = find_node(nodes, edge.after);
    if (before == nullptr || after == nullptr) {
      return make_unexpected(plan_error(LifecycleKernelErrorCode::kInvalidNode));
    }
    const std::size_t before_index = static_cast<std::size_t>(std::distance(nodes.data(), before));
    const std::size_t after_index = static_cast<std::size_t>(std::distance(nodes.data(), after));
    outgoing[before_index].push_back(after_index);
    ++indegree[after_index];
  }

  std::vector<std::size_t> ready;
  ready.reserve(node_count);
  for (std::size_t index = 0U; index < node_count; ++index) {
    if (indegree[index] == 0U) {
      ready.push_back(index);
    }
  }

  const auto stable_index_before = [&nodes](std::size_t lhs, std::size_t rhs) {
    return node_before(nodes[lhs], nodes[rhs]);
  };
  std::sort(ready.begin(), ready.end(), stable_index_before);

  LifecycleStagePlan plan{.stage = stage, .levels = {}};
  std::size_t visited = 0U;
  while (!ready.empty()) {
    LifecycleStageLevel level;
    level.node_ids.reserve(ready.size());
    for (const std::size_t index : ready) {
      level.node_ids.push_back(nodes[index].node_id);
    }
    visited += ready.size();
    plan.levels.push_back(std::move(level));

    std::vector<std::size_t> next;
    next.reserve(node_count - visited);
    for (const std::size_t index : ready) {
      for (const std::size_t successor : outgoing[index]) {
        --indegree[successor];
        if (indegree[successor] == 0U) {
          next.push_back(successor);
        }
      }
    }
    std::sort(next.begin(), next.end(), stable_index_before);
    ready = std::move(next);
  }

  if (visited != node_count) {
    return make_unexpected(plan_error(LifecycleKernelErrorCode::kCycle, {}, {}, stage));
  }
  return plan;
}

[[nodiscard]] const LifecycleLogicalCallbackRecord* find_callback_record(
    const LifecycleStageGraph& graph, LifecycleNodeId node_id, LifecycleStage stage) noexcept {
  const auto match = std::find_if(
      graph.callbacks.begin(), graph.callbacks.end(), [node_id, stage](const auto& callback) {
        return callback.node_id == node_id && callback.stage == stage;
      });
  return match == graph.callbacks.end() ? nullptr : &*match;
}

[[nodiscard]] bool bindings_valid(const LifecycleStageGraph& graph,
                                  const std::vector<LifecycleCallbackBinding>& bindings) noexcept {
  for (std::size_t index = 0U; index < bindings.size(); ++index) {
    const auto& binding = bindings[index];
    if (binding.callback == nullptr || binding.self == nullptr ||
        stage_index(binding.stage) == kLifecycleStageCount ||
        find_callback_record(graph, binding.node_id, binding.stage) == nullptr) {
      return false;
    }
    for (std::size_t other = index + 1U; other < bindings.size(); ++other) {
      if (bindings[other].node_id == binding.node_id && bindings[other].stage == binding.stage) {
        return false;
      }
    }
  }
  for (const auto& callback : graph.callbacks) {
    if (callback.required &&
        std::find_if(bindings.begin(), bindings.end(), [&callback](const auto& binding) {
          return binding.node_id == callback.node_id && binding.stage == callback.stage;
        }) == bindings.end()) {
      return false;
    }
  }
  return true;
}

struct CallbackResult {
  bool binding_valid{true};
  LifecycleStatus status{LifecycleStatus::kSkipped};
};

[[nodiscard]] CallbackResult invoke_callback(const LifecycleStageGraph& graph,
                                             const std::vector<LifecycleCallbackBinding>& bindings,
                                             LifecycleNodeId node_id, LifecycleStage stage,
                                             LifecycleTransitionPhase phase,
                                             LifecycleLogOperation operation,
                                             const LifecycleKernelState& state) noexcept {
  const auto* callback_record = find_callback_record(graph, node_id, stage);
  if (callback_record == nullptr) {
    emit_kernel_log(state,
                    {.kind = LifecycleLogEventKind::kCallbackBegin,
                     .operation = operation,
                     .node_id = node_id,
                     .stage = stage,
                     .phase = phase});
    emit_kernel_log(state,
                    {.kind = LifecycleLogEventKind::kCallbackEnd,
                     .operation = operation,
                     .node_id = node_id,
                     .stage = stage,
                     .phase = phase,
                     .status = LifecycleStatus::kSkipped});
    return {};
  }
  const auto binding =
      std::find_if(bindings.begin(), bindings.end(), [node_id, stage](const auto& candidate) {
        return candidate.node_id == node_id && candidate.stage == stage;
      });
  if (binding == bindings.end()) {
    if (callback_record->required) {
      return {.binding_valid = false, .status = LifecycleStatus::kFatalFailure};
    }
    emit_kernel_log(state,
                    {.kind = LifecycleLogEventKind::kCallbackBegin,
                     .operation = operation,
                     .node_id = node_id,
                     .stage = stage,
                     .phase = phase});
    emit_kernel_log(state,
                    {.kind = LifecycleLogEventKind::kCallbackEnd,
                     .operation = operation,
                     .node_id = node_id,
                     .stage = stage,
                     .phase = phase,
                     .status = LifecycleStatus::kSkipped});
    return {};
  }
  if (binding->callback == nullptr || binding->self == nullptr) {
    return {.binding_valid = false, .status = LifecycleStatus::kFatalFailure};
  }
  emit_kernel_log(state,
                  {.kind = LifecycleLogEventKind::kCallbackBegin,
                   .operation = operation,
                   .node_id = node_id,
                   .stage = stage,
                   .phase = phase});
  const LifecycleStatus status = binding->callback(binding->self, stage, phase);
  emit_kernel_log(state,
                  {.kind = LifecycleLogEventKind::kCallbackEnd,
                   .operation = operation,
                   .node_id = node_id,
                   .stage = stage,
                   .phase = phase,
                   .status = status});
  return {.binding_valid = true, .status = status};
}

[[nodiscard]] constexpr bool callback_succeeded(LifecycleStatus status) noexcept {
  return status == LifecycleStatus::kOk || status == LifecycleStatus::kSkipped;
}

[[nodiscard]] LifecycleExecutionFailure execution_failure(LifecycleNodeId node_id,
                                                          LifecycleStage stage,
                                                          LifecycleTransitionPhase phase,
                                                          LifecycleStatus status) noexcept {
  return LifecycleExecutionFailure{
      .present = true,
      .node_id = node_id,
      .stage = stage,
      .phase = phase,
      .status = status,
  };
}

[[nodiscard]] bool has_completed(const LifecycleKernelState& state, LifecycleNodeId node_id,
                                 LifecycleStage stage) noexcept {
  return std::find_if(state.completed.begin(),
                      state.completed.end(),
                      [node_id, stage](const auto& completion) {
                        return completion.node_id == node_id && completion.stage == stage;
                      }) != state.completed.end();
}

void erase_from_stage(LifecycleKernelState& state, LifecycleStage first) {
  const std::size_t first_index = stage_index(first);
  state.completed.erase(std::remove_if(state.completed.begin(),
                                       state.completed.end(),
                                       [first_index](const auto& completion) {
                                         const std::size_t index = stage_index(completion.stage);
                                         return index >= first_index && index < kForwardStageCount;
                                       }),
                        state.completed.end());
}

struct TeardownResult {
  LifecycleExecutionFailure first_failure;
  bool invalid_binding{};
};

[[nodiscard]] TeardownResult run_planned_teardown(
    const LifecycleStageGraph& graph, const std::vector<LifecycleCallbackBinding>& bindings,
    LifecycleStage stage, LifecycleTransitionPhase phase, LifecycleLogOperation operation,
    LifecycleKernelState& state) noexcept {
  TeardownResult result;
  const auto& plan = graph.stages[stage_index(stage)];
  // Teardown plans already contain the dependency direction authored for that exact stage.
  // Execute them in normal topological order; a compiler that derives ownership teardown edges
  // is responsible for recording those edges in reverse-ownership order before normalization.
  for (const auto& level : plan.levels) {
    for (const LifecycleNodeId node_id : level.node_ids) {
      const bool eligible = stage == LifecycleStage::kStop
                                ? (has_completed(state, node_id, LifecycleStage::kArm) ||
                                   has_completed(state, node_id, LifecycleStage::kStart))
                                : has_completed(state, node_id, LifecycleStage::kAllocate);
      if (!eligible) {
        continue;
      }
      const CallbackResult callback =
          invoke_callback(graph, bindings, node_id, stage, phase, operation, state);
      if (!callback.binding_valid) {
        result.invalid_binding = true;
      }
      if ((!callback.binding_valid || !callback_succeeded(callback.status)) &&
          !result.first_failure.present) {
        result.first_failure = execution_failure(node_id, stage, phase, callback.status);
      }
    }
  }

  return result;
}

[[nodiscard]] LifecycleExecutionFailure run_reverse_rollback_stage(
    const LifecycleStageGraph& graph, const std::vector<LifecycleCallbackBinding>& bindings,
    LifecycleStage rollback_stage, std::size_t completion_begin, LifecycleLogOperation operation,
    LifecycleKernelState& state) noexcept {
  LifecycleExecutionFailure first_failure;
  std::fill(state.rollback_visited.begin(), state.rollback_visited.end(), false);
  for (std::size_t offset = state.completed.size(); offset > completion_begin; --offset) {
    const auto& completion = state.completed[offset - 1U];
    const bool selected = rollback_stage == LifecycleStage::kStop
                              ? (completion.stage == LifecycleStage::kArm ||
                                 completion.stage == LifecycleStage::kStart)
                              : completion.stage == LifecycleStage::kAllocate;
    if (!selected) {
      continue;
    }
    const std::size_t node_index = find_node_index(graph, completion.node_id);
    if (node_index == graph.nodes.size() || state.rollback_visited[node_index]) {
      continue;
    }
    state.rollback_visited[node_index] = true;
    const CallbackResult callback = invoke_callback(graph,
                                                    bindings,
                                                    completion.node_id,
                                                    rollback_stage,
                                                    LifecycleTransitionPhase::kForwardRollback,
                                                    operation,
                                                    state);
    if ((!callback.binding_valid || !callback_succeeded(callback.status)) &&
        !first_failure.present) {
      first_failure = execution_failure(completion.node_id,
                                        rollback_stage,
                                        LifecycleTransitionPhase::kForwardRollback,
                                        callback.status);
    }
  }
  return first_failure;
}

[[nodiscard]] LifecycleExecutionFailure rollback_completed_work(
    const LifecycleStageGraph& graph, const std::vector<LifecycleCallbackBinding>& bindings,
    std::size_t completion_begin, bool release_new_allocation, LifecycleLogOperation operation,
    LifecycleKernelState& state) noexcept {
  LifecycleExecutionFailure first_failure = run_reverse_rollback_stage(
      graph, bindings, LifecycleStage::kStop, completion_begin, operation, state);
  if (release_new_allocation) {
    const auto release_failure = run_reverse_rollback_stage(
        graph, bindings, LifecycleStage::kRelease, completion_begin, operation, state);
    if (!first_failure.present) {
      first_failure = release_failure;
    }
  }
  state.completed.resize(completion_begin);
  state.steady_state = false;
  return first_failure;
}

[[nodiscard]] LifecycleExecutionResult completed_result(
    const LifecycleKernelState& state) noexcept {
  return LifecycleExecutionResult{
      .disposition = LifecycleExecutionDisposition::kCompleted,
      .primary_failure = {},
      .rollback_failure = {},
      .restoration_failure = {},
      .restoration_rollback_failure = {},
      .steady_state = state.steady_state,
  };
}

struct ForwardStageResult {
  bool completed{true};
  bool invalid_binding{};
  bool capacity_exceeded{};
  LifecycleExecutionFailure failure;
};

[[nodiscard]] ForwardStageResult run_forward_stage(
    const LifecycleStageGraph& graph, const std::vector<LifecycleCallbackBinding>& bindings,
    LifecycleStage stage, LifecycleTransitionPhase phase, LifecycleLogOperation operation,
    LifecycleKernelState& state) noexcept {
  const auto& plan = graph.stages[stage_index(stage)];
  for (const auto& level : plan.levels) {
    for (const LifecycleNodeId node_id : level.node_ids) {
      // Reserve the completion ledger before user work. A callback that succeeds must always be
      // representable in rollback bookkeeping; otherwise the bounded kernel rejects without
      // entering user code.
      if (state.completed.size() == state.completion_capacity) {
        return {
            .completed = false,
            .invalid_binding = false,
            .capacity_exceeded = true,
            .failure = execution_failure(node_id, stage, phase, LifecycleStatus::kFatalFailure),
        };
      }
      const CallbackResult callback =
          invoke_callback(graph, bindings, node_id, stage, phase, operation, state);
      if (!callback.binding_valid || !callback_succeeded(callback.status)) {
        return {
            .completed = false,
            .invalid_binding = !callback.binding_valid,
            .capacity_exceeded = false,
            .failure = execution_failure(node_id, stage, phase, callback.status),
        };
      }
      state.completed.push_back({.node_id = node_id, .stage = stage});
    }
  }
  return {};
}

[[nodiscard]] LifecycleExecutionResult make_forward_failure_result(
    const ForwardStageResult& forward, const LifecycleExecutionFailure& rollback) noexcept {
  LifecycleExecutionDisposition disposition = LifecycleExecutionDisposition::kStageFailed;
  if (forward.invalid_binding) {
    disposition = LifecycleExecutionDisposition::kInvalidBinding;
  } else if (rollback.present) {
    disposition = LifecycleExecutionDisposition::kRollbackFailed;
  } else if (forward.capacity_exceeded) {
    disposition = LifecycleExecutionDisposition::kCapacityExceeded;
  }
  return {
      .disposition = disposition,
      .primary_failure = forward.failure,
      .rollback_failure = rollback,
      .restoration_failure = {},
      .restoration_rollback_failure = {},
      .steady_state = false,
  };
}

[[nodiscard]] LifecycleExecutionResult run_forward(
    const LifecycleStageGraph& graph, const std::vector<LifecycleCallbackBinding>& bindings,
    LifecycleStage first_stage, LifecycleTransitionPhase phase, LifecycleLogOperation operation,
    LifecycleKernelState& state) noexcept {
  const std::size_t first_index = stage_index(first_stage);
  const std::size_t completion_begin = state.completed.size();
  for (std::size_t index = first_index; index < kForwardStageCount; ++index) {
    const LifecycleStage stage = stage_from_index(index);
    const auto forward = run_forward_stage(graph, bindings, stage, phase, operation, state);
    if (!forward.completed) {
      const auto rollback =
          rollback_completed_work(graph,
                                  bindings,
                                  completion_begin,
                                  first_index <= stage_index(LifecycleStage::kAllocate),
                                  operation,
                                  state);
      return make_forward_failure_result(forward, rollback);
    }
  }
  state.steady_state = true;
  return completed_result(state);
}

[[nodiscard]] expected<void, LifecycleKernelError> validate_nodes(
    const std::vector<LifecycleScopeRecord>& scopes,
    std::vector<LifecycleLogicalNodeRecord>& nodes) {
  for (std::size_t index = 0U; index < nodes.size(); ++index) {
    const auto& node = nodes[index];
    const bool known_scope = std::find_if(scopes.begin(), scopes.end(), [&node](const auto& scope) {
                               return scope.scope_id == node.scope_id;
                             }) != scopes.end();
    if (node.node_id == 0U || node.stable_name.empty() || !known_scope) {
      return make_unexpected(
          plan_error(LifecycleKernelErrorCode::kInvalidNode, node.node_id, node.scope_id));
    }
    for (std::size_t other = index + 1U; other < nodes.size(); ++other) {
      if (nodes[other].node_id == node.node_id) {
        return make_unexpected(plan_error(LifecycleKernelErrorCode::kDuplicateNode,
                                          node.node_id,
                                          node.scope_id,
                                          LifecycleStage::kConfigure,
                                          node.stable_name));
      }
      if (nodes[other].scope_id == node.scope_id && nodes[other].stable_name == node.stable_name) {
        return make_unexpected(plan_error(LifecycleKernelErrorCode::kAmbiguousNodeIdentity,
                                          node.node_id,
                                          node.scope_id,
                                          LifecycleStage::kConfigure,
                                          node.stable_name));
      }
    }
  }
  std::sort(nodes.begin(), nodes.end(), node_before);
  return {};
}

[[nodiscard]] const LifecycleScopeRecord* find_scope(
    const std::vector<LifecycleScopeRecord>& scopes, LifecycleScopeId scope_id) noexcept {
  const auto found = std::find_if(scopes.begin(), scopes.end(), [scope_id](const auto& scope) {
    return scope.scope_id == scope_id;
  });
  return found == scopes.end() ? nullptr : &*found;
}

[[nodiscard]] expected<void, LifecycleKernelError> validate_scopes(
    const std::vector<LifecycleScopeRecord>& scopes) {
  for (std::size_t index = 0U; index < scopes.size(); ++index) {
    const auto& scope = scopes[index];
    if (scope.scope_id == 0U ||
        (scope.parent_scope_id && *scope.parent_scope_id == scope.scope_id)) {
      return make_unexpected(
          plan_error(LifecycleKernelErrorCode::kInvalidScope, {}, scope.scope_id));
    }
    for (std::size_t other = index + 1U; other < scopes.size(); ++other) {
      if (scopes[other].scope_id == scope.scope_id) {
        return make_unexpected(
            plan_error(LifecycleKernelErrorCode::kInvalidScope, {}, scope.scope_id));
      }
    }
    if (scope.parent_scope_id && find_scope(scopes, *scope.parent_scope_id) == nullptr) {
      return make_unexpected(
          plan_error(LifecycleKernelErrorCode::kInvalidScope, {}, scope.scope_id));
    }
    const LifecycleScopeRecord* current = &scope;
    for (std::size_t depth = 0U; current->parent_scope_id; ++depth) {
      if (depth >= scopes.size()) {
        return make_unexpected(
            plan_error(LifecycleKernelErrorCode::kInvalidScope, {}, scope.scope_id));
      }
      current = find_scope(scopes, *current->parent_scope_id);
      if (current == nullptr) {
        return make_unexpected(
            plan_error(LifecycleKernelErrorCode::kInvalidScope, {}, scope.scope_id));
      }
    }
  }
  return {};
}

[[nodiscard]] bool is_descendant_or_same(const std::vector<LifecycleScopeRecord>& scopes,
                                         LifecycleScopeId candidate,
                                         LifecycleScopeId ancestor) noexcept {
  const LifecycleScopeRecord* current = find_scope(scopes, candidate);
  for (std::size_t depth = 0U; current != nullptr && depth <= scopes.size(); ++depth) {
    if (current->scope_id == ancestor) {
      return true;
    }
    current = current->parent_scope_id ? find_scope(scopes, *current->parent_scope_id) : nullptr;
  }
  return false;
}

[[nodiscard]] bool callback_descriptor_valid(
    const LifecycleLogicalCallbackRecord& callback) noexcept {
  const std::size_t index = stage_index(callback.stage);
  if (index == kLifecycleStageCount ||
      callback.method != static_cast<LifecycleMethodId>(index + 1U)) {
    return false;
  }
  switch (callback.origin) {
    case LifecycleLogicalCallbackOrigin::kAdvancedMember:
      return true;
    case LifecycleLogicalCallbackOrigin::kLegacyStart:
      return callback.required && callback.stage == LifecycleStage::kStart;
    case LifecycleLogicalCallbackOrigin::kLegacyStop:
      return callback.required && callback.stage == LifecycleStage::kStop;
  }
  return false;
}

[[nodiscard]] expected<void, LifecycleKernelError> validate_callbacks(LifecycleStageGraph& graph) {
  for (std::size_t index = 0U; index < graph.callbacks.size(); ++index) {
    const auto& callback = graph.callbacks[index];
    const bool invalid_stage = stage_index(callback.stage) == kLifecycleStageCount;
    if (find_node(graph.nodes, callback.node_id) == nullptr || invalid_stage) {
      return make_unexpected(plan_error(invalid_stage ? LifecycleKernelErrorCode::kInvalidStage
                                                      : LifecycleKernelErrorCode::kInvalidNode,
                                        callback.node_id,
                                        {},
                                        callback.stage));
    }
    if (!callback_descriptor_valid(callback)) {
      return make_unexpected(plan_error(
          LifecycleKernelErrorCode::kInvalidCallback, callback.node_id, {}, callback.stage));
    }
    for (std::size_t other = index + 1U; other < graph.callbacks.size(); ++other) {
      if (graph.callbacks[other].node_id == callback.node_id &&
          graph.callbacks[other].stage == callback.stage) {
        return make_unexpected(plan_error(
            LifecycleKernelErrorCode::kDuplicateCallback, callback.node_id, {}, callback.stage));
      }
    }
  }
  std::sort(
      graph.callbacks.begin(), graph.callbacks.end(), [&graph](const auto& lhs, const auto& rhs) {
        if (lhs.stage != rhs.stage) {
          return stage_index(lhs.stage) < stage_index(rhs.stage);
        }
        const auto* lhs_node = find_node(graph.nodes, lhs.node_id);
        const auto* rhs_node = find_node(graph.nodes, rhs.node_id);
        return node_before(*lhs_node, *rhs_node);
      });
  return {};
}

[[nodiscard]] expected<void, LifecycleKernelError> validate_dependency_record(
    const std::vector<LifecycleLogicalNodeRecord>& nodes, LifecycleNodeId node_id,
    LifecycleStage declared_stage, const LifecycleLogicalDependencyKey& key) {
  const auto* node = find_node(nodes, node_id);
  if (node == nullptr) {
    return make_unexpected(plan_error(LifecycleKernelErrorCode::kInvalidNode,
                                      node_id,
                                      key.scope_id,
                                      declared_stage,
                                      key.stable_name));
  }
  if (stage_index(declared_stage) == kLifecycleStageCount ||
      stage_index(key.stage) == kLifecycleStageCount) {
    return make_unexpected(plan_error(LifecycleKernelErrorCode::kInvalidStage,
                                      node_id,
                                      key.scope_id,
                                      declared_stage,
                                      key.stable_name));
  }
  if (declared_stage != key.stage) {
    return make_unexpected(plan_error(LifecycleKernelErrorCode::kCrossStageDependency,
                                      node_id,
                                      key.scope_id,
                                      declared_stage,
                                      key.stable_name));
  }
  if (node->scope_id != key.scope_id) {
    return make_unexpected(plan_error(LifecycleKernelErrorCode::kCrossScopeDependency,
                                      node_id,
                                      key.scope_id,
                                      declared_stage,
                                      key.stable_name));
  }
  if (key.stable_name.empty()) {
    return make_unexpected(plan_error(
        LifecycleKernelErrorCode::kInvalidDependencyKey, node_id, key.scope_id, declared_stage));
  }
  return {};
}

[[nodiscard]] expected<void, LifecycleKernelError> validate_producers(
    const LifecycleStageGraphInput& input, const LifecycleStageGraph& graph) {
  for (std::size_t index = 0U; index < input.producers.size(); ++index) {
    const auto& producer = input.producers[index];
    auto validation = validate_dependency_record(
        graph.nodes, producer.node_id, producer.declared_stage, producer.key);
    if (!validation) {
      return make_unexpected(std::move(validation.error()));
    }
    const auto duplicate = std::find_if(
        input.producers.begin() + static_cast<std::ptrdiff_t>(index + 1U),
        input.producers.end(),
        [&producer](const auto& candidate) { return key_equal(candidate.key, producer.key); });
    if (duplicate != input.producers.end()) {
      return make_unexpected(plan_error(LifecycleKernelErrorCode::kDuplicateProducer,
                                        producer.node_id,
                                        producer.key.scope_id,
                                        producer.declared_stage,
                                        producer.key.stable_name));
    }
  }
  return {};
}

[[nodiscard]] expected<void, LifecycleKernelError> normalize_requirements(
    const LifecycleStageGraphInput& input, const LifecycleStageGraph& graph,
    std::array<std::vector<NodeEdge>, kLifecycleStageCount>& edges) {
  for (const auto& requirement : input.requirements) {
    auto validation = validate_dependency_record(
        graph.nodes, requirement.node_id, requirement.declared_stage, requirement.key);
    if (!validation) {
      return make_unexpected(std::move(validation.error()));
    }
    const auto producer = std::find_if(
        input.producers.begin(), input.producers.end(), [&requirement](const auto& candidate) {
          return key_equal(candidate.key, requirement.key);
        });
    if (producer == input.producers.end()) {
      return make_unexpected(plan_error(LifecycleKernelErrorCode::kMissingProducer,
                                        requirement.node_id,
                                        requirement.key.scope_id,
                                        requirement.declared_stage,
                                        requirement.key.stable_name));
    }
    append_unique_edge(edges[stage_index(requirement.declared_stage)],
                       {.before = producer->node_id, .after = requirement.node_id});
  }
  return {};
}

[[nodiscard]] expected<void, LifecycleKernelError> normalize_direct_edges(
    const LifecycleStageGraphInput& input, const LifecycleStageGraph& graph,
    std::array<std::vector<NodeEdge>, kLifecycleStageCount>& edges) {
  for (const auto& edge : input.direct_edges) {
    const auto* before = find_node(graph.nodes, edge.before_node_id);
    const auto* after = find_node(graph.nodes, edge.after_node_id);
    if (before == nullptr || after == nullptr) {
      return make_unexpected(
          plan_error(LifecycleKernelErrorCode::kInvalidNode,
                     before == nullptr ? edge.before_node_id : edge.after_node_id,
                     edge.declaration_scope_id,
                     edge.before_stage));
    }
    if (stage_index(edge.before_stage) == kLifecycleStageCount ||
        stage_index(edge.after_stage) == kLifecycleStageCount) {
      return make_unexpected(plan_error(LifecycleKernelErrorCode::kInvalidStage,
                                        edge.before_node_id,
                                        edge.declaration_scope_id,
                                        edge.before_stage));
    }
    if (edge.before_stage != edge.after_stage) {
      return make_unexpected(plan_error(LifecycleKernelErrorCode::kCrossStageDependency,
                                        edge.before_node_id,
                                        edge.declaration_scope_id,
                                        edge.before_stage));
    }
    if (find_scope(input.scopes, edge.declaration_scope_id) == nullptr ||
        !is_descendant_or_same(input.scopes, before->scope_id, edge.declaration_scope_id) ||
        !is_descendant_or_same(input.scopes, after->scope_id, edge.declaration_scope_id)) {
      return make_unexpected(plan_error(LifecycleKernelErrorCode::kCrossScopeDependency,
                                        edge.before_node_id,
                                        edge.declaration_scope_id,
                                        edge.before_stage));
    }
    append_unique_edge(edges[stage_index(edge.before_stage)],
                       {.before = edge.before_node_id, .after = edge.after_node_id});
  }
  return {};
}

[[nodiscard]] expected<void, LifecycleKernelError> build_stage_plans(
    LifecycleStageGraph& graph,
    const std::array<std::vector<NodeEdge>, kLifecycleStageCount>& edges) {
  for (std::size_t index = 0U; index < kLifecycleStageCount; ++index) {
    auto plan = make_stage_plan(stage_from_index(index), graph.nodes, edges[index]);
    if (!plan) {
      return make_unexpected(std::move(plan.error()));
    }
    graph.stages[index] = std::move(*plan);
  }
  return {};
}

}  // namespace

void emit_lifecycle_log_noexcept(LifecycleLogSink sink, void* state,
                                 const LifecycleLogRecord& record) noexcept {
  if (sink == nullptr) {
    return;
  }
  try {
    sink(state, record);
  } catch (...) {
    // Logging is diagnostic-only. A backend or test sink failure must never change lifecycle
    // ordering, rollback, publication, or the result returned to the caller.
    return;
  }
}

LifecycleLogSeverity lifecycle_log_severity(const LifecycleLogRecord& record) noexcept {
  switch (record.kind) {
    case LifecycleLogEventKind::kCallbackBegin:
    case LifecycleLogEventKind::kCallbackEnd:
      return LifecycleLogSeverity::kTrace;
    case LifecycleLogEventKind::kTransitionBegin:
      return LifecycleLogSeverity::kDebug;
    case LifecycleLogEventKind::kTransitionEnd:
      return record.disposition == LifecycleExecutionDisposition::kCompleted
                 ? LifecycleLogSeverity::kDebug
                 : LifecycleLogSeverity::kError;
  }
  return LifecycleLogSeverity::kError;
}

expected<LifecycleStageGraph, LifecycleKernelError> build_lifecycle_stage_graph(
    const LifecycleStageGraphInput& input, const LifecycleKernelCapacity& capacity) {
  if (!record_counts_fit(input, capacity)) {
    return make_unexpected(plan_error(LifecycleKernelErrorCode::kCapacityExceeded));
  }

  LifecycleStageGraph graph{
      .capacity = capacity,
      .nodes = input.nodes,
      .callbacks = input.callbacks,
      .stages = {},
  };

  auto validation = validate_scopes(input.scopes);
  if (!validation) {
    return make_unexpected(std::move(validation.error()));
  }
  validation = validate_nodes(input.scopes, graph.nodes);
  if (!validation) {
    return make_unexpected(std::move(validation.error()));
  }
  validation = validate_callbacks(graph);
  if (!validation) {
    return make_unexpected(std::move(validation.error()));
  }
  validation = validate_producers(input, graph);
  if (!validation) {
    return make_unexpected(std::move(validation.error()));
  }

  std::array<std::vector<NodeEdge>, kLifecycleStageCount> edges;
  validation = normalize_requirements(input, graph, edges);
  if (!validation) {
    return make_unexpected(std::move(validation.error()));
  }
  validation = normalize_direct_edges(input, graph, edges);
  if (!validation) {
    return make_unexpected(std::move(validation.error()));
  }
  validation = build_stage_plans(graph, edges);
  if (!validation) {
    return make_unexpected(std::move(validation.error()));
  }
  return graph;
}

LifecycleKernelState make_lifecycle_kernel_state(const LifecycleStageGraph& graph) {
  LifecycleKernelState state{
      .steady_state = false,
      .completion_capacity = graph.capacity.max_completions,
      .completed = {},
      .rollback_visited = std::vector<bool>(graph.nodes.size(), false),
  };
  state.completed.reserve(state.completion_capacity);
  return state;
}

expected<LifecycleParameterViewSelection, LifecycleParameterViewSelectionError>
select_lifecycle_parameter_view(LifecycleTransitionPhase phase,
                                const LifecycleParameterViewInputs& inputs) noexcept {
  switch (phase) {
    case LifecycleTransitionPhase::kStartupForward:
      return LifecycleParameterViewSelection{
          .kind = LifecycleParameterViewKind::kStartupGenerationZero,
          .generation = {},
      };
    case LifecycleTransitionPhase::kPreRestartTeardown:
      return LifecycleParameterViewSelection{
          .kind = LifecycleParameterViewKind::kOldPublished,
          .generation = inputs.old_published,
      };
    case LifecycleTransitionPhase::kRestartForward:
      if (inputs.private_candidate) {
        return LifecycleParameterViewSelection{
            .kind = LifecycleParameterViewKind::kPrivateCandidate,
            .generation = *inputs.private_candidate,
        };
      }
      return LifecycleParameterViewSelection{
          .kind = LifecycleParameterViewKind::kRetainedCurrent,
          .generation = inputs.current_published,
      };
    case LifecycleTransitionPhase::kForwardRollback:
      if (!inputs.creating_view) {
        return make_unexpected(LifecycleParameterViewSelectionError::kMissingCreatingView);
      }
      return *inputs.creating_view;
    case LifecycleTransitionPhase::kRestoreOld:
      return LifecycleParameterViewSelection{
          .kind = LifecycleParameterViewKind::kOldPublished,
          .generation = inputs.old_published,
      };
    case LifecycleTransitionPhase::kShutdownTeardown:
      return LifecycleParameterViewSelection{
          .kind = LifecycleParameterViewKind::kCurrentPublished,
          .generation = inputs.current_published,
      };
    case LifecycleTransitionPhase::kNone:
      return make_unexpected(LifecycleParameterViewSelectionError::kOutsideCallback);
  }
  return make_unexpected(LifecycleParameterViewSelectionError::kInvalidTransitionPhase);
}

LifecycleExecutionResult execute_lifecycle_startup(
    const LifecycleStageGraph& graph, const std::vector<LifecycleCallbackBinding>& bindings,
    LifecycleKernelState& state) noexcept {
  log_transition_begin(LifecycleLogOperation::kStartup, state);
  if (state.steady_state || !state.completed.empty()) {
    return log_transition_end(LifecycleLogOperation::kStartup,
                              {.disposition = LifecycleExecutionDisposition::kInvalidState,
                               .primary_failure = {},
                               .rollback_failure = {},
                               .restoration_failure = {},
                               .restoration_rollback_failure = {},
                               .steady_state = state.steady_state},
                              state);
  }
  if (!bindings_valid(graph, bindings)) {
    return log_transition_end(LifecycleLogOperation::kStartup,
                              {.disposition = LifecycleExecutionDisposition::kInvalidBinding,
                               .primary_failure = {},
                               .rollback_failure = {},
                               .restoration_failure = {},
                               .restoration_rollback_failure = {},
                               .steady_state = false},
                              state);
  }
  return log_transition_end(LifecycleLogOperation::kStartup,
                            run_forward(graph,
                                        bindings,
                                        LifecycleStage::kConfigure,
                                        LifecycleTransitionPhase::kStartupForward,
                                        LifecycleLogOperation::kStartup,
                                        state),
                            state);
}

LifecycleExecutionResult execute_lifecycle_restart(
    const LifecycleStageGraph& graph, const std::vector<LifecycleCallbackBinding>& bindings,
    LifecycleStage requested_stage, LifecycleKernelState& state) noexcept {
  log_transition_begin(LifecycleLogOperation::kRestart, state);
  const std::size_t requested_index = stage_index(requested_stage);
  if (requested_index >= kForwardStageCount) {
    return log_transition_end(LifecycleLogOperation::kRestart,
                              {.disposition = LifecycleExecutionDisposition::kInvalidRestartStage,
                               .primary_failure = {},
                               .rollback_failure = {},
                               .restoration_failure = {},
                               .restoration_rollback_failure = {},
                               .steady_state = state.steady_state},
                              state);
  }
  if (!state.steady_state) {
    return log_transition_end(LifecycleLogOperation::kRestart,
                              {.disposition = LifecycleExecutionDisposition::kInvalidState,
                               .primary_failure = {},
                               .rollback_failure = {},
                               .restoration_failure = {},
                               .restoration_rollback_failure = {},
                               .steady_state = false},
                              state);
  }
  if (!bindings_valid(graph, bindings)) {
    return log_transition_end(LifecycleLogOperation::kRestart,
                              {.disposition = LifecycleExecutionDisposition::kInvalidBinding,
                               .primary_failure = {},
                               .rollback_failure = {},
                               .restoration_failure = {},
                               .restoration_rollback_failure = {},
                               .steady_state = true},
                              state);
  }

  state.steady_state = false;
  const auto stop = run_planned_teardown(graph,
                                         bindings,
                                         LifecycleStage::kStop,
                                         LifecycleTransitionPhase::kPreRestartTeardown,
                                         LifecycleLogOperation::kRestart,
                                         state);
  TeardownResult release;
  if (requested_index <= stage_index(LifecycleStage::kAllocate)) {
    release = run_planned_teardown(graph,
                                   bindings,
                                   LifecycleStage::kRelease,
                                   LifecycleTransitionPhase::kPreRestartTeardown,
                                   LifecycleLogOperation::kRestart,
                                   state);
  }
  const auto primary = stop.first_failure.present ? stop.first_failure : release.first_failure;
  if (primary.present) {
    return log_transition_end(
        LifecycleLogOperation::kRestart,
        {
            .disposition = stop.invalid_binding || release.invalid_binding
                               ? LifecycleExecutionDisposition::kInvalidBinding
                               : LifecycleExecutionDisposition::kStageFailed,
            .primary_failure = primary,
            .rollback_failure = {},
            .restoration_failure = {},
            .restoration_rollback_failure = {},
            .steady_state = false,
        },
        state);
  }

  erase_from_stage(state, requested_stage);
  auto candidate = run_forward(graph,
                               bindings,
                               requested_stage,
                               LifecycleTransitionPhase::kRestartForward,
                               LifecycleLogOperation::kRestart,
                               state);
  if (candidate.disposition == LifecycleExecutionDisposition::kCompleted ||
      candidate.rollback_failure.present ||
      candidate.disposition == LifecycleExecutionDisposition::kInvalidBinding) {
    return log_transition_end(LifecycleLogOperation::kRestart, candidate, state);
  }

  // Candidate rollback returned the ledger to the retained predecessor prefix. Re-run the exact
  // invalidated suffix against the old published state. The original candidate failure remains
  // primary evidence even when restoration itself fails.
  const auto restoration = run_forward(graph,
                                       bindings,
                                       requested_stage,
                                       LifecycleTransitionPhase::kRestoreOld,
                                       LifecycleLogOperation::kRestart,
                                       state);
  if (restoration.disposition == LifecycleExecutionDisposition::kCompleted) {
    candidate.disposition = LifecycleExecutionDisposition::kRolledBack;
    candidate.steady_state = true;
    return log_transition_end(LifecycleLogOperation::kRestart, candidate, state);
  }
  candidate.disposition = LifecycleExecutionDisposition::kRollbackFailed;
  candidate.restoration_failure = restoration.primary_failure;
  candidate.restoration_rollback_failure = restoration.rollback_failure;
  candidate.steady_state = false;
  return log_transition_end(LifecycleLogOperation::kRestart, candidate, state);
}

LifecycleExecutionResult execute_lifecycle_shutdown(
    const LifecycleStageGraph& graph, const std::vector<LifecycleCallbackBinding>& bindings,
    LifecycleKernelState& state) noexcept {
  log_transition_begin(LifecycleLogOperation::kShutdown, state);
  if (state.completed.empty()) {
    state.steady_state = false;
    return log_transition_end(LifecycleLogOperation::kShutdown, completed_result(state), state);
  }
  if (!bindings_valid(graph, bindings)) {
    return log_transition_end(LifecycleLogOperation::kShutdown,
                              {.disposition = LifecycleExecutionDisposition::kInvalidBinding,
                               .primary_failure = {},
                               .rollback_failure = {},
                               .restoration_failure = {},
                               .restoration_rollback_failure = {},
                               .steady_state = state.steady_state},
                              state);
  }

  state.steady_state = false;
  const auto stop = run_planned_teardown(graph,
                                         bindings,
                                         LifecycleStage::kStop,
                                         LifecycleTransitionPhase::kShutdownTeardown,
                                         LifecycleLogOperation::kShutdown,
                                         state);
  const auto release = run_planned_teardown(graph,
                                            bindings,
                                            LifecycleStage::kRelease,
                                            LifecycleTransitionPhase::kShutdownTeardown,
                                            LifecycleLogOperation::kShutdown,
                                            state);
  state.completed.clear();
  const auto primary = stop.first_failure.present ? stop.first_failure : release.first_failure;
  if (primary.present) {
    return log_transition_end(
        LifecycleLogOperation::kShutdown,
        {
            .disposition = stop.invalid_binding || release.invalid_binding
                               ? LifecycleExecutionDisposition::kInvalidBinding
                               : LifecycleExecutionDisposition::kStageFailed,
            .primary_failure = primary,
            .rollback_failure = {},
            .restoration_failure = {},
            .restoration_rollback_failure = {},
            .steady_state = false,
        },
        state);
  }
  return log_transition_end(LifecycleLogOperation::kShutdown, completed_result(state), state);
}

}  // namespace holoscan::detail
