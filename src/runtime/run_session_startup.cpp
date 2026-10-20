/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "run_session_internal.hpp"

#include "../compiler/execution_plan_lookup.hpp"
#include "../core/operator_spec_compiler_access.hpp"
#include "../core/source_id_internal.hpp"
#include "../core/temporal_contract_internal.hpp"
#include "core/diagnostic_reporting.hpp"
#include "runtime_cuda_graph_region_resources.hpp"
#include "runtime_support_thread_placement.hpp"
#include "runtime_test_controls.hpp"
#include "runtime_worker.hpp"

namespace holoscan::detail {

// Cold RunSession materialization and unpublished-resource rollback.

namespace {

[[nodiscard]] VertexId readiness_owner_for_source(const ExecutionPlanState& plan,
                                                  SourceId source) noexcept {
  const auto found = std::ranges::find(plan.readiness, source, &ReadinessPlanRecord::source_id);
  return found != plan.readiness.end() ? found->owner : VertexId{};
}

[[nodiscard]] RuntimeEventKind runtime_event_kind(ReadinessSourceKind kind) noexcept {
  switch (kind) {
    case ReadinessSourceKind::kReadable:
      return RuntimeEventKind::kReadable;
    case ReadinessSourceKind::kNotification:
      return RuntimeEventKind::kNotification;
    case ReadinessSourceKind::kDoorbell:
      return RuntimeEventKind::kDoorbell;
    case ReadinessSourceKind::kSpin:
      return RuntimeEventKind::kSpin;
  }
  return RuntimeEventKind::kSourceError;
}

[[nodiscard]] expected<std::optional<std::uint32_t>, Error> bind_runtime_derived_term(
    const RuntimeOperatorInstance& instance, ParameterDerivedTermKind kind, PortId input,
    std::int64_t initial_value_ns) noexcept {
  std::optional<std::uint32_t> matched;
  bool found{};
  for (std::size_t index = 0U; index < instance.parameter_derived_terms.size(); ++index) {
    const ParameterDerivedTermPlanRecord& term = instance.parameter_derived_terms[index];
    if (term.kind != kind || term.input != input) {
      continue;
    }
    if (found || index > std::numeric_limits<std::uint32_t>::max() ||
        term.owner.id != instance.operator_id || term.owner.run_generation != 0U ||
        term.initial_value_ns != initial_value_ns) {
      return make_unexpected(Error{ErrorCode::kIncompatible});
    }
    found = true;
    switch (term.update_disposition) {
      case ParameterDerivedTermUpdateDisposition::kRuntimeAtomic:
        if (!instance.runtime_parameter_updates_enabled ||
            term.mutability != ParamMutability::kRuntime) {
          return make_unexpected(Error{ErrorCode::kIncompatible});
        }
        matched = static_cast<std::uint32_t>(index);
        break;
      case ParameterDerivedTermUpdateDisposition::kFrozenCompileTime:
      case ParameterDerivedTermUpdateDisposition::kRecompileRequired:
        break;
    }
  }
  return matched;
}

[[nodiscard]] bool input_trigger_kind(TemporalRecordTriggerKind kind) noexcept {
  return kind == TemporalRecordTriggerKind::kAnyInput || kind == TemporalRecordTriggerKind::kEach ||
         kind == TemporalRecordTriggerKind::kAligned || kind == TemporalRecordTriggerKind::kWindow;
}

[[nodiscard]] bool clock_source_kind(const TemporalTriggerPlanRecord& trigger,
                                     const PredicateTriggerPlanRecord* const predicate) noexcept {
  return trigger.kind == TemporalRecordTriggerKind::kClock ||
         (predicate != nullptr && predicate->source == PredicateEvaluationSourceKind::kClock);
}

[[nodiscard]] bool input_source_kind(const TemporalTriggerPlanRecord& trigger,
                                     const PredicateTriggerPlanRecord* const predicate) noexcept {
  return input_trigger_kind(trigger.kind) ||
         (predicate != nullptr && predicate->source == PredicateEvaluationSourceKind::kInputUpdate);
}

void advance_auxiliary_source_id(std::uint32_t& next_source, const SourceId source) noexcept {
  if (next_source == 0U) {
    return;
  }
  const std::uint32_t value = source_id_value(source);
  next_source =
      value == std::numeric_limits<std::uint32_t>::max() ? 0U : std::max(next_source, value + 1U);
}

}  // namespace

[[nodiscard]] bool RunSessionState::ReadyRunState::uses_inline_invoke() const noexcept {
  return inline_partition_execution || partition_owner_execution;
}

RunSessionState::StartupRuntimeTeardown::StartupRuntimeTeardown(
    RunSessionState& session, ReadySharedResources& shared) noexcept
    : session_(&session), shared_(&shared) {}

RunSessionState::StartupRuntimeTeardown::~StartupRuntimeTeardown() noexcept {
  if (active_) {
    session_->teardown_startup_runtime(*shared_);
  }
}

void RunSessionState::StartupRuntimeTeardown::dismiss() noexcept {
  active_ = false;
}

void RunSessionState::bind_shared_resources(ReadyRunState& state,
                                            ReadySharedResources& shared) noexcept {
  // Refresh the complete borrowed cache together so a vector reallocation cannot leave one
  // partition state with a dangling span while the others observe the replacement storage.
  state.clock_health = shared.clock_health.get();
  state.lanes = shared.lanes.get();
  state.channels = shared.channels.get();
  state.notifications = shared.notifications.get();
  state.channel_activation_owners = shared.channel_activation_owners;
  state.channel_input_owners = shared.channel_input_owners;
}

[[nodiscard]] bool RunSessionState::owns_receiver(const ReadyRunState& state,
                                                  std::size_t receiver) noexcept {
  return state.owned_receivers.empty() ||
         (receiver < state.owned_receivers.size() && state.owned_receivers[receiver] != 0U);
}

[[nodiscard]] bool RunSessionState::owns_vertex(const ReadyRunState& state, const GraphExec& exec,
                                                VertexId vertex) noexcept {
  const auto receiver = exec.receiver_index(vertex);
  return receiver && owns_receiver(state, *receiver);
}

[[nodiscard]] bool RunSessionState::has_owned_unwired_inputs(const GraphExec& exec,
                                                             const ReadyRunState& state) const {
  return std::any_of(
      plan_->input_selections.begin(),
      plan_->input_selections.end(),
      [&](const InputSelectionPlanRecord& selection) {
        const auto port = std::find_if(plan_->graph.ports().begin(),
                                       plan_->graph.ports().end(),
                                       [&](const PortRecord& record) {
                                         return record.id.value == selection.input.value() &&
                                                record.direction == PortDirection::kInput;
                                       });
        if (port == plan_->graph.ports().end() || !owns_vertex(state, exec, port->owner) ||
            !worker_owns_vertex(*plan_, exec.process_group(), port->owner)) {
          return false;
        }
        return selection.unwired_safe_default;
      });
}

[[nodiscard]] bool RunSessionState::bind_channel_publication_context(
    const GraphExec& exec, RuntimeChannelSession& channels) {
  // Publication coordinates are stamped from the Graph default clock, so the session must own a
  // readable reference to it before any output can publish.
  if (fixture_observation_ != nullptr) {
    auto observed = channels.observe_fixture_publications(fixture_observation_);
    if (!observed) {
      fixture_observation_->invalidate();
      append_diagnostic("RUNTIME_FIXTURE_OBSERVATION_UNSUPPORTED",
                        observed.error().owner,
                        {},
                        nullptr,
                        "cold canonical inline-POD outputs",
                        "unsupported fixture publication binding",
                        ::holoscan::DiagnosticSeverity::kError,
                        {},
                        observed.error().port);
      return false;
    }
  }
  if (auto bound = channels.bind_publish_clock(exec.default_clock()); !bound) {
    append_diagnostic("RUNTIME_CLOCK_BINDING_UNAVAILABLE");
    return false;
  }
  return true;
}

[[nodiscard]] bool RunSessionState::initialize_channel_session(
    GraphExec& exec, ReadyRunState& state, ReadySharedResources& shared,
    const RuntimeCudaExecutionResources& cuda_resources,
    RuntimeChannelSizingOverride sizing_override) {
  const bool channel_plan_present =
      !plan_->process_local_channel_services.empty() || !plan_->channel_services.empty() ||
      !plan_->cross_process_channel_services.empty() || has_owned_unwired_inputs(exec, state);
  if (!channel_plan_present) {
    return true;
  }

  shared.channel_replay_ports.reserve(plan_->ports.size());
  for (RuntimeOperatorInstance& instance : exec.operators()) {
    if (instance.replay_spec == nullptr) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
      return false;
    }
    for (std::size_t index = 0U;
         index < OperatorSpecCompilerAccess::port_count(*instance.replay_spec);
         ++index) {
      const auto replay = OperatorSpecCompilerAccess::port(*instance.replay_spec, index);
      const auto planned = std::find_if(
          plan_->ports.begin(), plan_->ports.end(), [&](const PortPlanRecord& candidate) {
            return candidate.owner == instance.vertex &&
                   candidate.owner_kind == PortOwnerKind::kOperator &&
                   candidate.direction == replay.direction && candidate.local_name == replay.name;
          });
      if (planned == plan_->ports.end() || replay.declaration == nullptr ||
          replay.runtime_type_ops == nullptr) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
        return false;
      }
      shared.channel_replay_ports.push_back(RuntimeChannelPortReplayRecord{
          .port = planned->id,
          .owner = instance.vertex,
          .direction = replay.direction,
          .declaration = const_cast<void*>(replay.declaration),
          .type_ops = replay.runtime_type_ops,
          .tensor_failure = &instance.tensor_failure,
      });
    }
  }
  auto channels = RuntimeChannelSession::create(
        *plan_,
        shared.channel_replay_ports,
        exec.process_group(),
        RuntimeChannelReadinessWake{
            .state = this,
            .signal =
                [](void* opaque) noexcept {
                  auto* const session = static_cast<RunSessionState*>(opaque);
                  session->signal_controller_wake(RuntimePhaseCounter::kWakeSignals);
                },
            .signal_owner_progress =
                local_partition_runner_count_ > 1U
                    ? +[](void* opaque, VertexId owner) noexcept {
                        static_cast<RunSessionState*>(opaque)->signal_channel_owner_progress(owner);
                      }
                    : nullptr,
            .publication_stop = &publication_stop_latch_,
        },
        &cuda_resources,
        sizing_override,
        run_generation_);
  if (!channels) {
    append_diagnostic(
        registered_token_or(channels.error().message, "RUNTIME_PORT_BINDING_UNAVAILABLE"));
    return false;
  }
  if (!bind_channel_publication_context(exec, **channels)) {
    return false;
  }
  if (const auto& control = plan_->executable_factories.test_channel_teardown_control;
      control != nullptr) {
    if (auto observed = (*channels)->set_teardown_observer_for_test(
            control.get(), &RuntimeChannelTeardownTestControl::observe);
        !observed) {
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE");
      return false;
    }
  }
  if (plan_->executable_factories.test_fail_next_channel_activation_retirement) {
    if (auto injected = (*channels)->fail_next_activation_retirement_for_test(
            plan_->executable_factories.test_fail_only_failed_callback_retirement);
        !injected) {
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE");
      return false;
    }
  }
  if (const auto& control = plan_->executable_factories.test_channel_session_bind_control;
      control != nullptr) {
    if (auto configured = control->bind(**channels); !configured) {
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE");
      return false;
    }
  }
  shared.channel_activation_owners.assign(exec.operators().size(), 0U);
  shared.channel_input_owners.assign(exec.operators().size(), 0U);
  for (std::size_t receiver = 0U; receiver < exec.operators().size(); ++receiver) {
    shared.channel_activation_owners[receiver] = static_cast<std::uint8_t>(
        (*channels)->has_materialized_activation_state(exec.operators()[receiver].vertex));
    shared.channel_input_owners[receiver] = static_cast<std::uint8_t>(
        (*channels)->has_materialized_input_state(exec.operators()[receiver].vertex));
  }
  shared.channels = std::move(*channels);
  bind_shared_resources(state, shared);
  return true;
}

// Materialize each selector together with its parameter-derived term and retained-claim bounds;
// partial initialization cannot be exposed to dispatch.
[[nodiscard]] expected<RuntimeQueueCapacity, Error> RunSessionState::selector_rate_capacity(
    PortId input, std::uint32_t planned_capacity,
    const RuntimeRateMaterializationView* materialization) const noexcept {
  if (!input || planned_capacity == 0U) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
  }
  const PayloadEdgePlanRecord* edge{};
  for (const auto& candidate : plan_->payload_edges) {
    if (candidate.target != input) {
      continue;
    }
    if (edge != nullptr) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
    }
    edge = &candidate;
  }
  if (edge == nullptr) {
    return RuntimeQueueCapacity{.backing = planned_capacity, .active = planned_capacity};
  }
  const auto recipe = std::ranges::find(
      plan_->queue_sizing_recipes, edge->edge, &QueueSizingRecipePlanRecord::edge);
  if (recipe == plan_->queue_sizing_recipes.end() ||
      std::ranges::count(
          plan_->queue_sizing_recipes, edge->edge, &QueueSizingRecipePlanRecord::edge) != 1 ||
      edge->queue_depth == 0U) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
  }
  RuntimeQueueCapacity result{
      .backing = std::max(planned_capacity, edge->queue_depth),
      .active = planned_capacity,
  };
  if (materialization != nullptr) {
    const auto depth = std::ranges::find(
        materialization->queue_depths, edge->edge, &RuntimeQueueDepthGeneration::edge);
    if (depth == materialization->queue_depths.end() ||
        std::ranges::count(
            materialization->queue_depths, edge->edge, &RuntimeQueueDepthGeneration::edge) != 1 ||
        depth->active_logical_depth == 0U) {
      return make_unexpected(Error{ErrorCode::kNotReady, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
    }
    result.active = depth->active_logical_depth;
    if (recipe->geometry_mode == AllocationGeometryMode::kLifecycleCandidate) {
      result.backing = result.active;
    }
  }
  if (result.active == 0U || result.active > result.backing) {
    return make_unexpected(
        Error{ErrorCode::kResourceExhausted, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
  }
  return result;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool RunSessionState::initialize_input_selectors(
    GraphExec& exec, ReadyRunState& state, const RuntimeRateMaterializationView* materialization) {
  state.input_selectors.reserve(plan_->input_selections.size());
  std::size_t max_selected = 0U;
  std::size_t local_selection_count = 0U;
  for (const InputSelectionPlanRecord& selection : plan_->input_selections) {
    const auto port = std::find_if(
        plan_->graph.ports().begin(), plan_->graph.ports().end(), [&](const PortRecord& record) {
          return record.id.value == selection.input.value() &&
                 record.direction == PortDirection::kInput;
        });
    if (port == plan_->graph.ports().end()) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
      return false;
    }
    if (!owns_vertex(state, exec, port->owner)) {
      continue;
    }
    if (!worker_owns_vertex(*plan_, exec.process_group(), port->owner)) {
      continue;
    }
    ++local_selection_count;
    const bool connected = !selection.unwired_safe_default;
    const auto bracketed = std::find_if(
        plan_->bracketed_windows.begin(),
        plan_->bracketed_windows.end(),
        [&](const BracketedWindowPlanRecord& record) { return record.input == selection.input; });
    const BracketedWindowPlanRecord* bracketed_plan =
        bracketed == plan_->bracketed_windows.end() ? nullptr : &*bracketed;
    std::uint32_t max_pending_windows = 0U;
    if (bracketed_plan != nullptr) {
      const auto window = std::find_if(plan_->window_triggers.begin(),
                                       plan_->window_triggers.end(),
                                       [&](const WindowTriggerPlanRecord& record) {
                                         return record.owner == bracketed_plan->owner;
                                       });
      if (window == plan_->window_triggers.end()) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", port->owner);
        return false;
      }
      max_pending_windows = window->max_pending_windows;
    }
    auto rate_capacity =
        selector_rate_capacity(selection.input, selection.cache_capacity, materialization);
    if (!rate_capacity) {
      append_diagnostic(
          registered_token_or(rate_capacity.error().message, "RUNTIME_OPERATOR_REPLAY_MISMATCH"),
          port->owner);
      return false;
    }
    auto created = RuntimeInputSelectionState::create(
        selection, bracketed_plan, max_pending_windows, *rate_capacity);
    if (!created) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", port->owner);
      return false;
    }
    const std::size_t maximum_selected_capacity = created->maximum_selected_capacity();
    if (maximum_selected_capacity == 0U ||
        max_selected > std::numeric_limits<std::size_t>::max() - maximum_selected_capacity) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
      return false;
    }
    max_selected += maximum_selected_capacity;
    const auto predicate_input = std::find_if(
        plan_->predicate_inputs.begin(),
        plan_->predicate_inputs.end(),
        [&](const PredicateInputPlanRecord& input) { return input.input == selection.input; });
    const std::size_t payload_claim_capacity = created->payload_claim_capacity();
    if (payload_claim_capacity == 0U) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", port->owner);
      return false;
    }
    ReadyRunState::InputSelectorSlot slot{
        .owner = port->owner,
        .input = selection.input,
        .connected = connected,
        .selector = std::move(created).value(),
        .predicate_id = predicate_input == plan_->predicate_inputs.end()
                            ? std::string_view{}
                            : std::string_view{predicate_input->predicate_id},
        .predicate_ordinal = predicate_input == plan_->predicate_inputs.end()
                                 ? 0U
                                 : predicate_input->argument_ordinal,
        .predicate_input = predicate_input != plan_->predicate_inputs.end(),
        .aligned_member = false,
    };
    if (slot.connected) {
      slot.channel_payload_claims.reserve(payload_claim_capacity);
      slot.channel_payload_claim_scratch.resize(payload_claim_capacity);
      slot.channel_payload_retire_scratch.resize(payload_claim_capacity);
    }
    state.input_selectors.push_back(std::move(slot));
  }
  std::sort(state.input_selectors.begin(),
            state.input_selectors.end(),
            [](const auto& left, const auto& right) {
              if (left.owner != right.owner) {
                return left.owner.value < right.owner.value;
              }
              if (left.predicate_input != right.predicate_input) {
                return left.predicate_input;
              }
              if (left.predicate_input && left.predicate_id != right.predicate_id) {
                return left.predicate_id < right.predicate_id;
              }
              if (left.predicate_input && left.predicate_ordinal != right.predicate_ordinal) {
                return left.predicate_ordinal < right.predicate_ordinal;
              }
              return left.input.value() < right.input.value();
            });
  const std::size_t trigger_pin_capacity = static_cast<std::size_t>(std::count_if(
      plan_->temporal_triggers.begin(), plan_->temporal_triggers.end(), [&](const auto& trigger) {
        const bool local = exec.operator_by_id(trigger.owner.id) != nullptr;
        return local && (trigger.kind == TemporalRecordTriggerKind::kAnyInput ||
                         trigger.kind == TemporalRecordTriggerKind::kEach);
      }));
  if (local_selection_count > std::numeric_limits<std::size_t>::max() - trigger_pin_capacity ||
      max_selected > std::numeric_limits<std::size_t>::max() - trigger_pin_capacity) {
    append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
    return false;
  }
  const std::size_t selection_capacity = local_selection_count + trigger_pin_capacity;
  state.selection.selector_transactions.reserve(local_selection_count);
  state.selection.input_selections.reserve(selection_capacity);
  state.selection.selected_samples.reserve(max_selected + trigger_pin_capacity);
  state.selection.maximum_selected_samples = max_selected + trigger_pin_capacity;
  state.selection.channel_selections.reserve(selection_capacity);
  state.selection.predicate_inputs.reserve(local_selection_count);
  state.freshness_input_fences.reserve(local_selection_count);
  return true;
}

// Validate and materialize each aligned-member family before publishing any selector state.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool RunSessionState::initialize_aligned_selectors(
    GraphExec& exec, ReadyRunState& state, const RuntimeRateMaterializationView* materialization) {
  state.aligned_selectors.reserve(plan_->aligned_triggers.size());
  state.selection.aligned_transactions.reserve(plan_->aligned_triggers.size());
  for (const AlignedTriggerPlanRecord& aligned : plan_->aligned_triggers) {
    RuntimeOperatorInstance* const instance = exec.operator_by_id(aligned.owner.id);
    const auto receiver = exec.receiver_index(aligned.owner.id);
    if (instance == nullptr || !receiver || !owns_receiver(state, *receiver)) {
      continue;
    }
    std::vector<RuntimeQueueCapacity> rate_capacities;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      rate_capacities.reserve(aligned.members.size());
      for (const auto& member : aligned.members) {
        const auto selection = std::ranges::find(
            plan_->input_selections, member.input, &InputSelectionPlanRecord::input);
        if (selection == plan_->input_selections.end() ||
            std::ranges::count(
                plan_->input_selections, member.input, &InputSelectionPlanRecord::input) != 1) {
          append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance->vertex);
          return false;
        }
        auto capacity =
            selector_rate_capacity(member.input, member.cache_capacity, materialization);
        if (!capacity) {
          append_diagnostic(
              registered_token_or(capacity.error().message, "RUNTIME_OPERATOR_REPLAY_MISMATCH"),
              instance->vertex);
          return false;
        }
        rate_capacities.push_back(*capacity);
      }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (const std::bad_alloc&) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance->vertex);
      return false;
    }
#endif
    auto created =
        RuntimeAlignedSelectionState::create(aligned, plan_->input_selections, rate_capacities);
    if (!created) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance->vertex);
      return false;
    }
    state.aligned_selectors.push_back(ReadyRunState::AlignedSelectorSlot{
        .owner = instance->vertex,
        .selector = std::move(*created),
    });
    for (const AlignedMemberPlanRecord& member : aligned.members) {
      const auto input =
          std::find_if(state.input_selectors.begin(),
                       state.input_selectors.end(),
                       [&](const ReadyRunState::InputSelectorSlot& slot) {
                         return slot.owner == instance->vertex && slot.input == member.input;
                       });
      if (input == state.input_selectors.end() || input->predicate_input || input->aligned_member) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance->vertex);
        return false;
      }
      input->aligned_member = true;
      const std::size_t payload_claim_capacity =
          state.aligned_selectors.back().selector.payload_claim_capacity(member.input);
      if (payload_claim_capacity == 0U) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance->vertex);
        return false;
      }
      input->channel_payload_claims.reserve(payload_claim_capacity);
      input->channel_payload_claim_scratch.resize(payload_claim_capacity);
      input->channel_payload_retire_scratch.resize(payload_claim_capacity);
    }
  }
  return true;
}

// Runner materialization validates and commits the complete per-partition runtime state.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool RunSessionState::materialize_partition_runner_states(
    const GraphExec& exec, std::vector<ReadyRunState>& states) {
  std::vector<const PartitionRunnerPlanRecord*> local_runners;
  local_runners.reserve(plan_->partition_runners.size());
  for (const PartitionRunnerPlanRecord& runner : plan_->partition_runners) {
    if (runner.process_group != exec.process_group()) {
      continue;
    }
    local_runners.push_back(&runner);
  }
  if (local_runners.empty()) {
    append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
    return false;
  }

  std::sort(local_runners.begin(),
            local_runners.end(),
            [](const PartitionRunnerPlanRecord* lhs, const PartitionRunnerPlanRecord* rhs) {
              return lhs->ordinal < rhs->ordinal;
            });
  if (local_runners.size() != local_partition_runner_count_ ||
      (local_runners.size() == 1U && !partition_wakes_.empty()) ||
      (local_runners.size() > 1U && partition_wakes_.size() != local_runners.size())) {
    append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
    return false;
  }
  partition_wake_by_vertex_.clear();
  if (local_runners.size() > 1U) {
    partition_wake_by_vertex_.reset(plan_->graph.vertices().size() + 1U);
  }
  std::vector<std::uint8_t> covered(exec.operators().size(), 0U);
  std::vector<std::uint8_t> lanes(plan_->execution_lanes.size(), 0U);
  states.clear();
  states.reserve(local_runners.size());

  for (std::size_t runner_index = 0U; runner_index < local_runners.size(); ++runner_index) {
    const PartitionRunnerPlanRecord* const runner = local_runners[runner_index];
    if (runner == nullptr || runner->execution_lane >= lanes.size() ||
        lanes[runner->execution_lane] != 0U || runner->operators.empty()) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
      return false;
    }
    lanes[runner->execution_lane] = 1U;
    states.emplace_back();
    ReadyRunState& state = states.back();
    state.runner = runner;
    state.owned_receivers.assign(exec.operators().size(), 0U);
    if (local_runners.size() == 1U) {
      state.wake = &readiness_wake_;
      state.inline_execution_lane = runner->execution_lane;
      state.inline_partition_execution = true;
    } else {
      state.wake = partition_wakes_[runner_index].get();
      state.accounting_partition = static_cast<std::uint32_t>(runner_index);
      state.partition_owner_execution = true;
    }
    // Both owner shapes, a sole local owner included, wait as their lane's placement admits.
    const auto lane = std::ranges::find(
        plan_->execution_lanes, runner->execution_lane, &ExecutionLanePlanRecord::execution_lane);
    if (lane == plan_->execution_lanes.end()) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
      return false;
    }
    state.owner_busy_poll = lane->accepted_placement.wake_mode == WakeMode::kBusyPoll;
    for (const PartitionRunnerOperatorPlanRecord& member : runner->operators) {
      const RuntimeOperatorInstance* const instance = exec.operator_by_vertex(member.vertex);
      const auto receiver = exec.receiver_index(member.vertex);
      if (instance == nullptr || !receiver || instance->operator_id != member.operator_id ||
          instance->partition != runner->partition ||
          instance->execution_lane != runner->execution_lane) {
        append_diagnostic(
            "RUNTIME_OPERATOR_REPLAY_MISMATCH", member.vertex, "partition_runner_member");
        return false;
      }
      if (covered[*receiver] != 0U) {
        append_diagnostic(
            "RUNTIME_OPERATOR_REPLAY_MISMATCH", member.vertex, "partition_runner_duplicate");
        return false;
      }
      covered[*receiver] = 1U;
      state.owned_receivers[*receiver] = 1U;
      if (!partition_wake_by_vertex_.empty()) {
        if (!partition_wake_by_vertex_.bind(member.vertex, state.wake)) {
          append_diagnostic(
              "RUNTIME_OPERATOR_REPLAY_MISMATCH", member.vertex, "fused_region_identity");
          return false;
        }
      }
      // Only the fused source remains schedulable. The other reconstructed region members are
      // startup-capture callbacks and are covered by the source's runner.
      if (member.fused_cuda_graph_region) {
        const auto* capture = find_unique_cuda_graph_capture_plan_by_region(
            *plan_, member.fused_cuda_graph_region->graph, member.fused_cuda_graph_region->region);
        if (capture == nullptr || capture->members.empty() ||
            capture->members.front().vertex != member.vertex) {
          append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", member.vertex);
          return false;
        }
        for (const auto& captured : capture->members) {
          if (captured.vertex == member.vertex) {
            continue;
          }
          const RuntimeOperatorInstance* const captured_instance =
              exec.operator_by_vertex(captured.vertex);
          const auto captured_receiver = exec.receiver_index(captured.vertex);
          if (captured_instance == nullptr || !captured_receiver ||
              captured_instance->operator_id != captured.operator_ref.id ||
              captured_instance->partition != runner->partition ||
              captured_instance->execution_lane != runner->execution_lane ||
              covered[*captured_receiver] != 0U) {
            append_diagnostic(
                "RUNTIME_OPERATOR_REPLAY_MISMATCH", captured.vertex, "capture_member_identity");
            return false;
          }
          covered[*captured_receiver] = 1U;
          if (!partition_wake_by_vertex_.empty() &&
              !partition_wake_by_vertex_.bind(captured.vertex, state.wake)) {
            append_diagnostic(
                "RUNTIME_OPERATOR_REPLAY_MISMATCH", captured.vertex, "capture_member_wake");
            return false;
          }
        }
      }
    }
  }
  if (std::ranges::find(covered, std::uint8_t{0U}) != covered.end()) {
    append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
    return false;
  }

  if (plan_->inline_partition_execution) {
    if (states.size() != 1U) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
      return false;
    }
    ReadyRunState& state = states.front();
    const PartitionRunnerPlanRecord* const sole_runner = state.runner;
    const InlinePartitionExecutionPlanRecord& record = *plan_->inline_partition_execution;
    const bool owns_record = !exec.process_group() || exec.process_group() == record.process_group;
    if (owns_record) {
      const bool exact_operator_cover =
          exec.operators().size() == record.operators.size() &&
          std::all_of(exec.operators().begin(),
                      exec.operators().end(),
                      [&](const RuntimeOperatorInstance& instance) {
                        return instance.partition == record.partition &&
                               instance.execution_lane == record.execution_lane &&
                               std::count(record.operators.begin(),
                                          record.operators.end(),
                                          instance.vertex) == 1;
                      });
      if (!exact_operator_cover) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
        return false;
      }
      if (sole_runner != nullptr && (record.partition != sole_runner->partition ||
                                     record.execution_lane != sole_runner->execution_lane ||
                                     record.process_group != sole_runner->process_group)) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
        return false;
      }
      state.inline_execution_lane = record.execution_lane;
      state.inline_partition_execution = true;
    }
  }
  return true;
}

[[nodiscard]] bool RunSessionState::initialize_lanes(const GraphExec& exec, ReadyRunState& state,
                                                     ReadySharedResources& shared) {
  std::vector<ExecutionLanePlanRecord> worker_lanes;
  worker_lanes.reserve(exec.operators().size());
  for (const RuntimeOperatorInstance& instance : exec.operators()) {
    if (instance.execution_lane >= plan_->execution_lanes.size() ||
        plan_->execution_lanes[instance.execution_lane].execution_lane != instance.execution_lane) {
      append_diagnostic("RUNTIME_EXECUTION_LANE_PLAN_INVALID", instance.vertex);
      return false;
    }
    worker_lanes.push_back(plan_->execution_lanes[instance.execution_lane]);
  }
  std::sort(worker_lanes.begin(),
            worker_lanes.end(),
            [](const ExecutionLanePlanRecord& lhs, const ExecutionLanePlanRecord& rhs) {
              return lhs.execution_lane < rhs.execution_lane;
            });
  worker_lanes.erase(
      std::unique(worker_lanes.begin(),
                  worker_lanes.end(),
                  [](const ExecutionLanePlanRecord& lhs, const ExecutionLanePlanRecord& rhs) {
                    return lhs.execution_lane == rhs.execution_lane;
                  }),
      worker_lanes.end());
  if (worker_lanes.empty()) {
    append_diagnostic("RUNTIME_EXECUTION_LANE_PLAN_INVALID");
    return false;
  }

  SchedulerPlanRecord local_scheduler = plan_->scheduler_plan;
  // Preserve global lane identities as the topology upper bound while projecting the selected
  // process group's materialized callback owners onto their exact local scheduler geometry.
  local_scheduler.worker_count = static_cast<std::uint32_t>(worker_lanes.size());
  local_scheduler.max_concurrent_activations = local_scheduler.worker_count;
  local_scheduler.activation_overlap =
      local_scheduler.worker_count > 1U && !state.inline_execution_lane;
  local_scheduler.intra_path_parallelism = 1U;
  auto lanes = ExecutionLaneSet::create(local_scheduler,
                                        worker_lanes,
                                        plan_->executable_factories.lane_placement_adapter,
                                        plan_->executable_factories.phase_counters.get(),
                                        state.inline_execution_lane,
                                        pre_controller_affinity_);
  if (!lanes) {
    append_diagnostic(
        registered_token_or(lanes.error().message, "RUNTIME_EXECUTION_LANE_PLAN_INVALID"));
    return false;
  }
  shared.lanes = std::move(lanes).value();
  bind_shared_resources(state, shared);
  auto started = shared.lanes->start();
  lane_startup_evidence_ = shared.lanes->take_startup_evidence();
  if (!started) {
    append_diagnostic(registered_token_or(execution_lane_startup_diagnostic_token(started.error()),
                                          "RUNTIME_LANE_STARTUP_FAILED"));
    return false;
  }
  return true;
}

[[nodiscard]] bool RunSessionState::stop_lanes(ExecutionLaneSet* lanes) noexcept {
  if (lanes == nullptr) {
    return true;
  }
  lanes->request_stop();
  const auto joined = lanes->join();
  if (!joined) {
    RuntimeDiagnosticDetail detail{.present = true,
                                   .operation = DiagnosticOperation::kRuntime,
                                   .role = DiagnosticFailureRole::kTeardown};
    detail.capture.emplace();
    detail.capture->append(joined.error().code, joined.error().message);
    append_diagnostic(
        registered_token_or(joined.error().message, "RUNTIME_LANE_JOIN_FAILED"), {}, {}, &detail);
    return false;
  }
  return true;
}

void RunSessionState::teardown_startup_runtime(ReadySharedResources& shared) noexcept {
  if (shared.notifications != nullptr) {
    shared.notifications->close();
  }
  static_cast<void>(stop_channels(shared.channels.get()));
  static_cast<void>(stop_lanes(shared.lanes.get()));
}

[[nodiscard]] expected<std::optional<RuntimePredicateEvaluatorEngine>, Error>
RunSessionState::initialize_predicate(const RuntimeOperatorInstance& instance,
                                      const TemporalTriggerPlanRecord& trigger,
                                      PredicateReceiver& receiver) {
  if (trigger.kind != TemporalRecordTriggerKind::kPredicate) {
    receiver = {};
    return std::optional<RuntimePredicateEvaluatorEngine>{};
  }
  const PredicateTriggerPlanRecord* const predicate = instance.predicate_trigger;
  if (predicate == nullptr) {
    return make_unexpected(Error{ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"});
  }
  std::vector<PredicateInputPlanRecord> inputs;
  inputs.reserve(plan_->predicate_inputs.size());
  for (const auto& input : plan_->predicate_inputs) {
    if (input.predicate_id == predicate->id) {
      inputs.push_back(input);
    }
  }
  std::sort(inputs.begin(), inputs.end(), [](const auto& left, const auto& right) {
    return left.argument_ordinal < right.argument_ordinal;
  });
  const PredicateCallablePlanRecord* const callable =
      find_predicate_callable_by_id(*plan_, predicate->id);
  if (callable == nullptr) {
    return make_unexpected(Error{ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"});
  }

  const auto compute_input_count = static_cast<std::size_t>(std::count_if(
      plan_->input_selections.begin(), plan_->input_selections.end(), [&](const auto& selection) {
        const auto* port = find_port_by_id(*plan_, selection.input);
        if (port == nullptr || port->owner != instance.vertex) {
          return false;
        }
        return std::none_of(inputs.begin(), inputs.end(), [&](const auto& input) {
          return input.input == selection.input;
        });
      }));

  expected<TemporalContractSnapshot, Error> normalized =
      make_unexpected(Error{ErrorCode::kFailure});
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    normalized = ContractTestAccess::normalize(instance.op->contract(), 0U);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"});
  }
#endif
  if (!normalized || normalized->trigger.kind != TemporalTriggerKind::kPredicate ||
      normalized->trigger.predicate_clock_source !=
          (predicate->source == PredicateEvaluationSourceKind::kClock)) {
    return make_unexpected(Error{ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"});
  }
  auto storage = ContractTestAccess::clone_predicate_storage(normalized->trigger.predicate);
  if (!storage) {
    return make_unexpected(std::move(storage.error()));
  }
  auto engine = RuntimePredicateEvaluatorEngine::create(
      *predicate, inputs, compute_input_count, *callable, std::move(*storage));
  if (!engine) {
    return make_unexpected(std::move(engine.error()));
  }
  engine->reset(PredicateResetReason::kColdRunStart, instance.tags->generation());
  receiver =
      PredicateReceiver{.address = instance.op.get(),
                        .local_type_token = normalized->trigger.predicate_receiver_type_token};
  return std::optional<RuntimePredicateEvaluatorEngine>{std::move(*engine)};
}

// Freeze one coherent readiness inventory before any source can publish into the Dispatcher.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool RunSessionState::initialize_ready_state(
    GraphExec& exec, ReadyRunState& state, ReadySharedResources& shared,
    const RuntimeRateMaterializationView* materialization) {
  if (shared.clock_health == nullptr) {
    auto health = EndpointClockHealthRuntimeCoordinator::create(plan_->endpoint_clock_health,
                                                                plan_->clocks,
                                                                plan_->plan_generation,
                                                                run_generation_,
                                                                exec.clock_registry());
    if (!health) {
      append_diagnostic(health.error().token.empty()
                            ? std::string_view{"ENDPOINT_CLOCK_HEALTH_RUNTIME_PLAN_INVALID"}
                            : health.error().token);
      return false;
    }
    shared.clock_health = std::move(health).value();
    if (shared.channels != nullptr) {
      auto bound = shared.channels->bind_clock_health(*shared.clock_health, clock_health_mutex_);
      if (!bound) {
        append_diagnostic("ENDPOINT_CLOCK_HEALTH_RUNTIME_PLAN_INVALID");
        return false;
      }
    }
  }
  bind_shared_resources(state, shared);
  if (!initialize_input_selectors(exec, state, materialization)) {
    return false;
  }
  if (!initialize_aligned_selectors(exec, state, materialization)) {
    return false;
  }

  std::size_t max_outstanding = 0U;
  const std::size_t operator_count = exec.operators().size();
  if (operator_count > std::numeric_limits<std::uint32_t>::max() ||
      operator_count > std::numeric_limits<std::size_t>::max() / 2U) {
    append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
    return false;
  }
  std::uint32_t next_auxiliary_source = 1U;
  const auto advance_auxiliary_source = [&](const SourceId source) {
    if (next_auxiliary_source == 0U) {
      return;
    }
    const std::uint32_t value = source_id_value(source);
    if (value == std::numeric_limits<std::uint32_t>::max()) {
      next_auxiliary_source = 0U;
      return;
    }
    next_auxiliary_source = std::max(next_auxiliary_source, value + 1U);
  };
  for (const ReadinessPlanRecord& readiness : plan_->readiness) {
    advance_auxiliary_source(readiness.source_id);
  }
  for (const TemporalTriggerPlanRecord& trigger : plan_->temporal_triggers) {
    advance_auxiliary_source(trigger.readiness_source);
  }

  std::vector<RuntimeSourceBindingRecord> sources;
  std::vector<TagAllocator*> tag_allocators;
  sources.reserve(operator_count * 2U);
  tag_allocators.reserve(operator_count);
  state.binding_requests.reserve(operator_count);
  state.channel_binding_requests.reserve(operator_count);
  state.clock_binding_requests.reserve(operator_count);
  state.bindings.reserve(operator_count * 2U);
  state.clock_bindings.reserve(operator_count);
  state.activation_selections.reserve(operator_count);
  state.predicates.reserve(operator_count);
  state.predicate_receivers.reserve(operator_count);
  state.max_start_delay_bindings.reserve(operator_count);
  state.parameter_policies.reserve(operator_count);
  state.restart.receivers.reserve(operator_count);
  state.restart.member_vertices.reserve(operator_count);
  state.restart.vertices.reserve(operator_count);
  state.restart.aggregate_member_vertices.reserve(operator_count);
  state.restart.aggregate_vertices.reserve(operator_count);
  state.restart.phases.resize(operator_count);
  for (std::size_t index = 0U; index < operator_count; ++index) {
    const RuntimeOperatorInstance& instance = exec.operators()[index];
    const bool receiver_owned = owns_receiver(state, index);
    const auto* cuda_graph_region =
        find_unique_cuda_graph_execution_region_by_member(*plan_, instance.vertex);
    const bool capture_only =
        cuda_graph_region != nullptr && cuda_graph_region->fused_operator != instance.vertex;
    if (capture_only) {
      // Capture-only members retain reconstruction and lifecycle state, but the fused source owns
      // all steady-state readiness and Dispatcher state.
      tag_allocators.push_back(instance.tags.get());
      state.predicates.emplace_back();
      state.predicate_receivers.emplace_back();
      state.max_start_delay_bindings.emplace_back();
      state.parameter_policies.emplace_back();
      auto activation = ActivationSelectionState::create(0U, 0U);
      if (!activation) {
        append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", instance.vertex);
        return false;
      }
      state.activation_selections.push_back(std::move(*activation));
      continue;
    }
    const TemporalTriggerPlanRecord* const trigger = instance.temporal_trigger;
    if (trigger == nullptr || !trigger->readiness_source ||
        trigger->owner.plan_generation != plan_->plan_generation) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
      return false;
    }
    const PredicateTriggerPlanRecord* const predicate = instance.predicate_trigger;
    const bool clock_source =
        trigger->kind == TemporalRecordTriggerKind::kClock ||
        (predicate != nullptr && predicate->source == PredicateEvaluationSourceKind::kClock);
    const bool input_source =
        input_trigger_kind(trigger->kind) ||
        (predicate != nullptr && predicate->source == PredicateEvaluationSourceKind::kInputUpdate);
    if (receiver_owned && input_source) {
      state.input_terminal_receivers.push_back(ReadyRunState::InputTerminalDrainState{
          .receiver = index,
          .phase = InputTerminalDrainPhase::kOpen,
      });
    }
    // Every non-input trigger owns activation authority independently from connected Channel
    // inputs. Materialize one auxiliary arrival source for those inputs so their bounded
    // temporal state advances without allowing an arrival to activate the Operator.
    const bool has_channel_side_inputs = !input_source &&
                                         index < state.channel_input_owners.size() &&
                                         state.channel_input_owners[index] != 0U;
    SourceId selector_readiness_source = input_source ? trigger->readiness_source : SourceId{};
    const bool direct_source = !clock_source && !input_source;
    const ReadinessPlanRecord* const readiness =
        direct_source ? find_readiness_by_owner(*plan_, instance.vertex) : nullptr;
    const ClockTriggerPlanRecord* const clock =
        clock_source ? find_clock_trigger_by_owner(*plan_, trigger->owner) : nullptr;
    if ((!clock_source && !input_source && readiness == nullptr) ||
        (clock_source && clock == nullptr) ||
        ((input_source || has_channel_side_inputs) && state.channels == nullptr)) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
      return false;
    }
    expected<std::uint32_t, Error> channel_capacity = 0U;
    if (input_source || has_channel_side_inputs) {
      channel_capacity = state.channels->readiness_capacity(instance.vertex);
      if (!channel_capacity) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
        return false;
      }
    }
    const std::uint32_t source_capacity = clock_source   ? 1U
                                          : input_source ? *channel_capacity
                                                         : readiness->accepted_capacity;
    const ClockId reference_clock =
        direct_source ? readiness->resolved_clock : trigger->reference_clock;
    const std::uint64_t source_generation =
        direct_source ? readiness->initial_source_generation : 1U;
    const std::uint64_t source_sequence = direct_source ? readiness->initial_source_sequence : 0U;
    const RuntimeEventKind event_kind = clock_source   ? RuntimeEventKind::kClockTick
                                        : input_source ? RuntimeEventKind::kInputArrival
                                                       : runtime_event_kind(readiness->kind);
    // A production Channel readiness source is owner-multiplexed: polling drains every input
    // arrival in deterministic order so non-trigger samples can populate temporal selectors
    // before the trigger witness is considered. Preserve the concrete event port on each
    // RuntimeEvent, but keep the frozen source port open so the coordinator accepts the complete
    // owner-local prefix. Explicitly port-scoped lower-level bindings still retain exact-port
    // validation in RuntimeCandidateCoordinator.
    const PortId source_port{};
    if ((direct_source &&
         (readiness == nullptr || readiness->source_id != trigger->readiness_source ||
          !readiness->referenced || !readiness->materialized)) ||
        (clock_source &&
         (clock == nullptr || clock->clock != reference_clock || clock->period_ns <= 0)) ||
        !reference_clock.valid() || source_capacity == 0U ||
        (receiver_owned &&
         max_outstanding > std::numeric_limits<std::size_t>::max() - source_capacity)) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
      return false;
    }
    auto max_start_delay = bind_runtime_max_start_delay(plan_->max_start_delays,
                                                        *trigger,
                                                        plan_->clock_conversions,
                                                        plan_->plan_generation,
                                                        predicate);
    if (!max_start_delay) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
      return false;
    }
    const std::optional<RuntimeMaxStartDelayBindingRecord> max_start_delay_binding =
        std::move(max_start_delay).value();
    const RuntimeMaxStartDelayBindingRecord max_start_delay_values =
        max_start_delay_binding.value_or(RuntimeMaxStartDelayBindingRecord{});
    const ClockId source_clock =
        max_start_delay_binding ? max_start_delay_values.source_clock : reference_clock;
    const std::uint64_t conversion_id =
        max_start_delay_binding ? max_start_delay_values.conversion_id : 0U;
    if (receiver_owned) {
      max_outstanding += source_capacity;
      sources.push_back(RuntimeSourceBindingRecord{
          .candidate_kind = event_kind,
          .source = trigger->readiness_source,
          .port = source_port,
          .source_clock = source_clock,
          .reference_clock = reference_clock,
          .source_to_reference_conversion_id = conversion_id,
          .source_order = static_cast<std::uint32_t>(index),
          .capacity = source_capacity,
          .source_generation = source_generation,
          .tag_generation = instance.tags->generation(),
          .receiver_index = index,
          .partition = instance.partition,
          .execution_lane = instance.execution_lane,
          .vertex = instance.vertex,
          .plan_vertex_order = static_cast<std::uint32_t>(index),
          .stable_candidate_tie =
              direct_source ? readiness->declaration_ordinal : static_cast<std::uint64_t>(index),
          .max_start_delay = max_start_delay_binding,
      });
    }
    const RuntimePortBindingRequest binding_request{
        .plan_generation = plan_->plan_generation,
        .run_generation = run_generation_,
        .kind = event_kind,
        .source = trigger->readiness_source,
        .port = source_port,
        .source_clock = source_clock,
        .reference_clock = reference_clock,
        .source_to_reference_conversion_id = conversion_id,
        .execution_lane = instance.execution_lane,
        .capacity = source_capacity,
        .source_generation = source_generation,
        .initial_source_sequence = source_sequence,
    };
    if (receiver_owned) {
      if (clock_source) {
        state.clock_binding_requests.push_back(ReadyRunState::ClockBindingRequest{
            .source = binding_request, .schedule = *clock, .receiver_index = index});
      } else if (input_source) {
        state.channel_binding_requests.emplace_back(instance.vertex, binding_request);
      } else {
        state.binding_requests.push_back(binding_request);
      }
    }
    if (has_channel_side_inputs) {
      const std::size_t auxiliary_order = operator_count + index;
      if (next_auxiliary_source == 0U ||
          auxiliary_order > std::numeric_limits<std::uint32_t>::max() ||
          (receiver_owned &&
           max_outstanding > std::numeric_limits<std::size_t>::max() - *channel_capacity)) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
        return false;
      }
      const SourceId auxiliary_source = make_source_id(next_auxiliary_source);
      selector_readiness_source = auxiliary_source;
      next_auxiliary_source = next_auxiliary_source == std::numeric_limits<std::uint32_t>::max()
                                  ? 0U
                                  : next_auxiliary_source + 1U;
      if (receiver_owned) {
        max_outstanding += *channel_capacity;
        sources.push_back(RuntimeSourceBindingRecord{
            .candidate_kind = RuntimeEventKind::kInputArrival,
            .source = auxiliary_source,
            .port = {},
            .source_clock = reference_clock,
            .reference_clock = reference_clock,
            .source_to_reference_conversion_id = 0U,
            .source_order = static_cast<std::uint32_t>(auxiliary_order),
            .capacity = *channel_capacity,
            .source_generation = 1U,
            .tag_generation = instance.tags->generation(),
            .receiver_index = index,
            .partition = instance.partition,
            .execution_lane = instance.execution_lane,
            .vertex = instance.vertex,
            .plan_vertex_order = static_cast<std::uint32_t>(index),
            .stable_candidate_tie = static_cast<std::uint64_t>(auxiliary_order),
        });
        // Independent clock/readiness activation authority remains on the primary source. This
        // private, plan-derived Channel source only admits connected samples into bounded
        // selectors; its arrival candidates are retired before Dispatcher eligibility.
        state.channel_binding_requests.emplace_back(instance.vertex,
                                                    RuntimePortBindingRequest{
                                                        .plan_generation = plan_->plan_generation,
                                                        .run_generation = run_generation_,
                                                        .kind = RuntimeEventKind::kInputArrival,
                                                        .source = auxiliary_source,
                                                        .port = {},
                                                        .source_clock = reference_clock,
                                                        .reference_clock = reference_clock,
                                                        .source_to_reference_conversion_id = 0U,
                                                        .execution_lane = instance.execution_lane,
                                                        .capacity = *channel_capacity,
                                                        .source_generation = 1U,
                                                        .initial_source_sequence = 0U,
                                                    });
      }
    }
    tag_allocators.push_back(instance.tags.get());

    PredicateReceiver predicate_receiver;
    auto predicate_engine = initialize_predicate(instance, *trigger, predicate_receiver);
    if (!predicate_engine) {
      append_diagnostic(
          registered_token_or(predicate_engine.error().message, "RUNTIME_OPERATOR_REPLAY_MISMATCH"),
          instance.vertex);
      return false;
    }
    state.predicates.push_back(std::move(*predicate_engine));
    state.predicate_receivers.push_back(predicate_receiver);
    state.max_start_delay_bindings.push_back(max_start_delay_binding);

    std::size_t max_inputs = 0U;
    std::size_t max_selected = 0U;
    ReadyRunState::OperatorParameterPolicy parameter_policy;
    // Side-only data cannot activate its owner. Its readiness revision remains visible to the
    // next trigger scan, so admitting the sample does not need to wake the owner. Data on a
    // predicate argument or an aligned member can activate it, and an unconnected input receives
    // nothing.
    const auto mark_side_only = [&](const ReadyRunState::InputSelectorSlot& slot,
                                    const bool requires_publication_order) {
      if (plan_->executable_factories.test_keep_side_only_data_hints || !slot.connected ||
          slot.predicate_input || slot.aligned_member) {
        return true;
      }
      if (auto marked =
              state.channels->mark_side_only_input(slot.input, requires_publication_order);
          !marked) {
        append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
        return false;
      }
      return true;
    };
    for (std::size_t selector_index = 0U; selector_index < state.input_selectors.size();
         ++selector_index) {
      auto& selector = state.input_selectors[selector_index];
      if (selector.owner != instance.vertex) {
        continue;
      }
      if (selector.connected && !selector_readiness_source) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
        return false;
      }
      selector.receiver_index = index;
      selector.readiness_source = selector.connected ? selector_readiness_source : SourceId{};
      selector.witness_port = trigger->trigger_input;
      if (trigger->kind == TemporalRecordTriggerKind::kAnyInput) {
        const std::size_t trigger_input_count = static_cast<std::size_t>(
            std::count_if(plan_->ports.begin(), plan_->ports.end(), [&](const auto& port) {
              return port.owner == instance.vertex && port.direction == PortDirection::kInput;
            }));
        if (trigger_input_count == 1U) {
          selector.witness_port = selector.input;
        }
      }
      selector.witness_source = trigger->readiness_source;
      selector.nontrigger_prerequisite =
          freshness_input_is_nontrigger_prerequisite(selector.selector.plan(),
                                                     selector.witness_port,
                                                     selector.readiness_source,
                                                     selector.witness_source);
      if (selector.selector.plan().selector == InputSelectorKind::kLatestWithin ||
          selector.selector.plan().missing == TemporalRecordMissingKind::kReuseLast) {
        auto supported = state.channels->supports_freshness_refill(selector.input);
        if (!supported) {
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
          return false;
        }
        selector.freshness_reconcile = *supported;
        if (*supported && clock_source) {
          auto drop_oldest_supported =
              state.channels->supports_drop_oldest_freshness_refill(selector.input);
          if (!drop_oldest_supported) {
            append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
            return false;
          }
          selector.upcoming_clock_freshness = qualifies_upcoming_clock_freshness(
              selector.selector.plan(), *drop_oldest_supported, true);
        }
      } else if (selector.selector.plan().selector == InputSelectorKind::kDrainAll &&
                 selector.input != selector.witness_port &&
                 selector.readiness_source != selector.witness_source) {
        // A side-only input of an independently triggered owner: whether the trigger at T
        // observes a capture-less arrival must not depend on how soon the owner polls it.
        // Input-triggered owners poll their trigger and side arrivals together instead.
        const auto ordered =
            state.channels->order_capture_absent_arrivals_by_publication(selector.input);
        if (!ordered) {
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
          return false;
        }
        // The owner witnesses a drop-oldest side input on arrival, which pins its typed slots
        // until the next trigger. Only the reconcile can then keep the newest unread survivors
        // in the batch; nothing bounds an input-triggered owner's next trigger for it.
        auto drop_oldest_supported =
            state.channels->supports_drop_oldest_freshness_refill(selector.input);
        if (!drop_oldest_supported) {
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
          return false;
        }
        selector.drop_oldest_drain = *drop_oldest_supported;
        selector.freshness_reconcile = *drop_oldest_supported;
        // A clock-triggered owner reconciles this input at its tick scan. Every successor has a
        // fixed coordinate, and one extra staging survivor outlives a publication that races
        // the scan, so that scan alone keeps the newest survivor at or before the tick.
        if (*drop_oldest_supported && *ordered &&
            trigger->kind == TemporalRecordTriggerKind::kClock) {
          auto enabled = state.channels->enable_tick_reconciled_staging(selector.input);
          if (!enabled) {
            append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
            return false;
          }
          selector.tick_fenced = *enabled;
          const auto* port = find_port_by_id(*plan_, selector.input);
          if (port == nullptr) {
            append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
            return false;
          }
          const auto& payload = port->payload_identity;
          const bool tensor_backed = payload.kind == PayloadIdentityKind::kTensor ||
                                     (payload.kind == PayloadIdentityKind::kSchema &&
                                      payload.payload_kind == PayloadKind::kDescriptorBacked);
          selector.tick_reconcilable = *enabled && !tensor_backed;
        }
        if (!mark_side_only(selector, true)) {
          return false;
        }
      } else if (selector.selector.plan().selector == InputSelectorKind::kDrainAll &&
                 trigger->kind == TemporalRecordTriggerKind::kEach &&
                 selector.input != selector.witness_port &&
                 selector.readiness_source == selector.witness_source) {
        // An OnEach owner polls its trigger and side arrivals through one binding in arrival
        // order, so the trigger's own wake ingests every earlier side sample.
        if (!mark_side_only(selector, false)) {
          return false;
        }
      }
      if (selector_index > std::numeric_limits<std::uint32_t>::max()) {
        append_diagnostic("RUNTIME_PARAMETER_BINDING_UNAVAILABLE", instance.vertex);
        return false;
      }
      if (selector.selector.plan().selector == InputSelectorKind::kLatestWithin) {
        auto term = bind_runtime_derived_term(instance,
                                              ParameterDerivedTermKind::kLatestWithinMaxAge,
                                              selector.input,
                                              selector.selector.plan().max_age_ns);
        if (!term) {
          append_diagnostic("RUNTIME_PARAMETER_BINDING_UNAVAILABLE", instance.vertex);
          return false;
        }
        // Preserve the exact cold binding for HSDK5-012's production selector transaction.
        selector.derived_term_index = *term;
      }
      parameter_policy.selector_indices.push_back(static_cast<std::uint32_t>(selector_index));
      ++max_inputs;
      const std::size_t maximum_selected_capacity = selector.selector.maximum_selected_capacity();
      if (maximum_selected_capacity == 0U ||
          max_selected > std::numeric_limits<std::size_t>::max() - maximum_selected_capacity) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
        return false;
      }
      max_selected += maximum_selected_capacity;
    }
    if (receiver_owned) {
      for (std::size_t term_index = 0U; term_index < instance.parameter_derived_terms.size();
           ++term_index) {
        const auto& term = instance.parameter_derived_terms[term_index];
        if (term.kind != ParameterDerivedTermKind::kLatestWithinMaxAge ||
            term.update_disposition != ParameterDerivedTermUpdateDisposition::kRuntimeAtomic) {
          continue;
        }
        const bool has_consumer = term_index <= std::numeric_limits<std::uint32_t>::max() &&
                                  std::any_of(state.input_selectors.begin(),
                                              state.input_selectors.end(),
                                              [&](const auto& selector) {
                                                return selector.owner == instance.vertex &&
                                                       selector.derived_term_index ==
                                                           static_cast<std::uint32_t>(term_index);
                                              });
        if (!has_consumer) {
          append_diagnostic("RUNTIME_PARAMETER_BINDING_UNAVAILABLE", instance.vertex);
          return false;
        }
      }
    }
    if (max_start_delay_binding.has_value()) {
      auto term = bind_runtime_derived_term(instance,
                                            ParameterDerivedTermKind::kContractMaxStartDelay,
                                            {},
                                            max_start_delay_values.max_start_delay->budget_ns);
      if (!term) {
        append_diagnostic("RUNTIME_PARAMETER_BINDING_UNAVAILABLE", instance.vertex);
        return false;
      }
      parameter_policy.max_start_delay_term_index = *term;
    }
    if (trigger != nullptr && (trigger->kind == TemporalRecordTriggerKind::kAnyInput ||
                               trigger->kind == TemporalRecordTriggerKind::kEach)) {
      if (max_inputs == std::numeric_limits<std::size_t>::max() ||
          max_selected == std::numeric_limits<std::size_t>::max()) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
        return false;
      }
      ++max_inputs;
      ++max_selected;
    }
    auto activation = ActivationSelectionState::create(max_inputs, max_selected);
    if (!activation) {
      HOLOSCAN_LOG_AUTOMATIC_ERROR(
          "Runtime activation-selection storage failed with {} for vertex {} ({} inputs, {} "
          "selected samples)",
          error_code_name(activation.error().code),
          instance.vertex.value,
          max_inputs,
          max_selected);
      const RuntimeDiagnosticDetail detail{.present = true, .error = activation.error()};
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", instance.vertex, {}, &detail);
      return false;
    }
    state.activation_selections.push_back(std::move(activation).value());
    const auto completion =
        std::lower_bound(operator_completion_progress_.begin(),
                         operator_completion_progress_.end(),
                         instance.operator_id,
                         [](const OperatorCompletionProgressRecord& record,
                            OperatorInstanceId owner) { return record.owner.id < owner; });
    if (receiver_owned && completion != operator_completion_progress_.end() &&
        completion->owner.id == instance.operator_id) {
      state.completion_receivers.push_back(ReadyRunState::CompletionReceiverState{
          .receiver = index,
          .completion_index = static_cast<std::size_t>(
              std::distance(operator_completion_progress_.begin(), completion)),
      });
    }
    state.parameter_policies.push_back(std::move(parameter_policy));
  }
  state.compute_lane_invoke = state.completion_receivers.empty()
                                  ? compute_lane_invoke_
                                  : compute_lane_with_completion_invoke_;
  state.active_completion_receivers.reserve(state.completion_receivers.size());
  state.active_input_terminal_receivers.reserve(state.input_terminal_receivers.size());
  state.closed_input_terminal_receivers.reserve(state.input_terminal_receivers.size());
  state.dispatcher_scheduler = plan_->scheduler_plan;
  if (state.runner != nullptr) {
    state.dispatcher_scheduler.max_concurrent_activations = 1U;
    state.dispatcher_scheduler.activation_overlap = false;
    state.dispatcher_scheduler.worker_count = 1U;
    state.dispatcher_scheduler.intra_path_parallelism = 1U;
  }
  auto dispatcher = Dispatcher::create(state.dispatcher_scheduler, max_outstanding);
  if (!dispatcher) {
    append_diagnostic(registered_token_or(dispatcher.error().message, "READY_POLICY_PLAN_INVALID"));
    return false;
  }
  state.dispatcher = std::make_unique<Dispatcher>(std::move(dispatcher).value());
  auto coordinator = RuntimeCandidateCoordinator::create_with_tags(
      sources, tag_allocators, max_outstanding, phase_counters_);
  if (!coordinator) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR("Runtime candidate coordinator construction failed with {}",
                                 error_code_name(coordinator.error().code));
    const RuntimeDiagnosticDetail detail{.present = true, .error = coordinator.error()};
    append_diagnostic(
        registered_token_or(coordinator.error().message, "RUNTIME_READINESS_BINDING_UNAVAILABLE"),
        {},
        {},
        &detail);
    return false;
  }
  state.coordinator = std::make_unique<RuntimeCandidateCoordinator>(std::move(coordinator).value());
  state.expected_binding_count = sources.size();
  state.selection.candidates.reserve(max_outstanding);
  state.selection.frozen_candidates.reserve(max_outstanding);
  state.restart.events.resize(max_outstanding);
  return true;
}

[[nodiscard]] std::optional<RuntimeNotificationSourceDescriptor>
RunSessionState::resolve_notification_descriptor(
    GraphExec& exec, const RuntimePortBindingRequest& request) noexcept {
  const auto readiness = std::find_if(
      plan_->readiness.begin(), plan_->readiness.end(), [&](const ReadinessPlanRecord& row) {
        return row.source_id == request.source && row.owner && row.referenced && row.materialized;
      });
  const auto lane = std::find_if(plan_->execution_lanes.begin(),
                                 plan_->execution_lanes.end(),
                                 [&](const ExecutionLanePlanRecord& row) {
                                   return row.execution_lane == request.execution_lane;
                                 });
  auto clock = exec.resolve_clock(request.reference_clock);
  if (readiness == plan_->readiness.end() ||
      readiness->kind != ReadinessSourceKind::kNotification ||
      readiness->accepted_capacity != request.capacity ||
      readiness->accepted_sender_reference_capacity == 0U || lane == plan_->execution_lanes.end() ||
      !clock) {
    append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
    return std::nullopt;
  }
  return RuntimeNotificationSourceDescriptor{
      .owner = readiness->owner,
      .request = request,
      .wake_requirement = readiness->wake_requirement,
      .wake_mode = lane->accepted_placement.wake_mode,
      .sender_reference_capacity = readiness->accepted_sender_reference_capacity,
      .clock = *clock,
      .wake_state = this,
      .wake =
          [](void* opaque) noexcept {
            auto* const session = static_cast<RunSessionState*>(opaque);
            session->signal_controller_wake(RuntimePhaseCounter::kWakeSignals);
          },
  };
}

// Bind production notifications before Arm so lifecycle callbacks can acquire sender authority.
// One registry owns every participating partition; posts remain not-ready until the complete
// Start barrier succeeds. Single-state insertion order and shared-state canonical source order
// remain the same as the pre-consolidation paths.
[[nodiscard]] bool RunSessionState::bind_shared_production_notifications(
    GraphExec& exec, std::span<ReadyRunState> states, ReadySharedResources& shared) {
  if (plan_->executable_factories.test_direct_activation_count != 0U ||
      plan_->executable_factories.port_binding_factory != nullptr) {
    return true;
  }
  struct PendingDescriptor {
    std::size_t state_index{};
    RuntimeNotificationSourceDescriptor descriptor;
  };
  std::vector<PendingDescriptor> pending;
  for (std::size_t state_index = 0U; state_index < states.size(); ++state_index) {
    const ReadyRunState& state = states[state_index];
    for (const auto& request : state.binding_requests) {
      if (request.kind != RuntimeEventKind::kNotification) {
        continue;
      }
      auto descriptor = resolve_notification_descriptor(exec, request);
      if (!descriptor) {
        return false;
      }
      pending.push_back(PendingDescriptor{
          .state_index = state_index,
          .descriptor = std::move(*descriptor),
      });
    }
  }
  if (pending.empty()) {
    return true;
  }
  if (states.size() > 1U) {
    std::sort(pending.begin(),
              pending.end(),
              [](const PendingDescriptor& lhs, const PendingDescriptor& rhs) {
                return source_id_value(lhs.descriptor.request.source) <
                       source_id_value(rhs.descriptor.request.source);
              });
  }
  std::vector<RuntimeNotificationSourceDescriptor> descriptors;
  descriptors.reserve(pending.size());
  std::vector<std::size_t> binding_counts(states.size());
  for (const PendingDescriptor& row : pending) {
    descriptors.push_back(row.descriptor);
    ++binding_counts[row.state_index];
  }
  for (std::size_t state_index = 0U; state_index < states.size(); ++state_index) {
    states[state_index].notification_bindings.reserve(binding_counts[state_index]);
  }
  auto registry = RuntimeNotificationSourceRegistry::create(descriptors);
  if (!registry) {
    append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
    return false;
  }
  for (std::size_t index = 0U; index < pending.size(); ++index) {
    const RuntimeNotificationSourceDescriptor& descriptor = descriptors[index];
    auto start_coordinate = descriptor.clock.now();
    if (!start_coordinate) {
      append_diagnostic("RUNTIME_CLOCK_READ_FAILED");
      return false;
    }
    auto binding = (*registry)->bind(descriptor.request, *start_coordinate);
    if (!binding || *binding == nullptr || (*binding)->capacity() < descriptor.request.capacity) {
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
      return false;
    }
    states[pending[index].state_index].notification_bindings.push_back(std::move(*binding));
  }
  shared.notifications = std::move(*registry);
  for (ReadyRunState& state : states) {
    bind_shared_resources(state, shared);
  }
  return true;
}

// Bind every remaining physical, Channel, and clock source before opening lifecycle ingress;
// partial binding is a startup failure and must not publish a runnable session.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool RunSessionState::bind_ready_sources(GraphExec& exec, ReadyRunState& state,
                                                       bool manage_channel_session) {
  const std::uint32_t direct_witness_count =
      plan_->executable_factories.test_direct_activation_count;
  const std::shared_ptr<RuntimePortBindingFactory>& configured_factory =
      plan_->executable_factories.port_binding_factory;
  if (direct_witness_count != 0U && configured_factory != nullptr) {
    append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
    return false;
  }
  std::optional<TestDirectPortBindingFactory> direct_factory;
  RuntimePortBindingFactory* factory = configured_factory.get();
  if (direct_witness_count != 0U) {
    direct_factory.emplace(direct_witness_count);
    factory = &*direct_factory;
  }
  const bool has_unbound_direct_request = std::any_of(
      state.binding_requests.begin(),
      state.binding_requests.end(),
      [&](const RuntimePortBindingRequest& request) {
        return request.kind != RuntimeEventKind::kNotification || state.notifications == nullptr;
      });
  if ((has_unbound_direct_request && factory == nullptr) ||
      state.binding_requests.size() + state.channel_binding_requests.size() +
              state.clock_binding_requests.size() !=
          state.expected_binding_count) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR(
        "Runtime readiness binding inventory is inconsistent: {} direct, {} Channel, {} clock "
        "requests for {} frozen sources across {} operators",
        state.binding_requests.size(),
        state.channel_binding_requests.size(),
        state.clock_binding_requests.size(),
        state.expected_binding_count,
        exec.operators().size());
    append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
    return false;
  }
  state.bindings = std::move(state.notification_bindings);
  state.clock_bindings.assign(exec.operators().size(), nullptr);
  for (const auto& request : state.binding_requests) {
    if (request.kind == RuntimeEventKind::kNotification && state.notifications != nullptr) {
      continue;
    }
    auto clock = exec.resolve_clock(request.reference_clock);
    auto start_coordinate =
        clock ? clock->now() : expected<TimePoint, Error>{make_unexpected(clock.error())};
    if (!start_coordinate) {
      append_diagnostic("RUNTIME_CLOCK_READ_FAILED");
      return false;
    }
    if (request.reference_clock != start_coordinate->clock_id) {
      append_diagnostic("RUNTIME_CLOCK_BINDING_UNAVAILABLE");
      return false;
    }
    auto binding = factory->bind(request, *start_coordinate);
    if (!binding || binding.value() == nullptr || binding.value()->capacity() < request.capacity) {
      const std::string_view candidate =
          !binding ? std::string_view{binding.error().message} : std::string_view{};
      const RuntimeDiagnosticDetail detail{
          .present = !binding,
          .error = !binding ? binding.error() : Error{ErrorCode::kSuccess},
          .operation = DiagnosticOperation::kRuntime};
      append_diagnostic(registered_token_or(candidate, "RUNTIME_READINESS_BINDING_UNAVAILABLE"),
                        readiness_owner_for_source(*plan_, request.source),
                        {},
                        !binding ? &detail : nullptr);
      return false;
    }
    state.bindings.push_back(std::move(binding).value());
  }
  const std::size_t first_channel_binding = state.bindings.size();
  for (const auto& binding_request : state.channel_binding_requests) {
    // Use ordinary references so clang-tidy can track initialization in the logging lambda.
    const auto& owner = binding_request.first;
    const auto& request = binding_request.second;
    auto binding = state.channels->create_readiness_binding(owner, request);
    if (!binding || binding.value() == nullptr || binding.value()->capacity() < request.capacity) {
      const Error error = !binding
                              ? binding.error()
                              : Error{binding.value() == nullptr ? ErrorCode::kNotFound
                                                                 : ErrorCode::kResourceExhausted};
      HOLOSCAN_LOG_AUTOMATIC_ERROR(
          "Channel readiness binding failed with {} for vertex {} (required capacity {})",
          error_code_name(error.code),
          owner.value,
          request.capacity);
      const RuntimeDiagnosticDetail detail{.present = true, .error = error};
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", owner, {}, &detail);
      return false;
    }
    state.bindings.push_back(std::move(*binding));
  }
  const std::size_t channel_binding_end = state.bindings.size();
  for (const auto& request : state.clock_binding_requests) {
    auto clock = exec.resolve_clock(request.schedule.clock);
    if (!clock) {
      append_diagnostic("RUNTIME_CLOCK_BINDING_UNAVAILABLE");
      return false;
    }
    TimePoint phase;
    if (request.schedule.phase_origin == ClockPhaseOrigin::kCapturedAtSuccessfulStart) {
      auto captured = clock->now();
      if (!captured) {
        append_diagnostic("RUNTIME_CLOCK_READ_FAILED");
        return false;
      }
      phase = *captured;
    } else if (request.schedule.phase_origin == ClockPhaseOrigin::kExplicitAbsolute &&
               request.schedule.explicit_phase) {
      auto converted =
          exec.clock_registry().convert(*request.schedule.explicit_phase, request.schedule.clock);
      if (!converted) {
        append_diagnostic("RUNTIME_CLOCK_BINDING_UNAVAILABLE");
        return false;
      }
      phase = *converted;
    } else {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
      return false;
    }
    auto binding = ClockTriggerPortBinding::create(
        request.source,
        *clock,
        phase,
        std::chrono::nanoseconds{request.schedule.period_ns},
        request.schedule.phase_origin == ClockPhaseOrigin::kCapturedAtSuccessfulStart);
    if (!binding) {
      append_diagnostic(
          registered_token_or(binding.error().message, "RUNTIME_CLOCK_BINDING_UNAVAILABLE"));
      return false;
    }
    state.clock_bindings[request.receiver_index] = binding->get();
    state.bindings.push_back(std::move(*binding));
  }
  state.contention_retry_eligible_by_binding.assign(state.bindings.size(), 0U);
  for (std::size_t binding_index = first_channel_binding; binding_index < channel_binding_end;
       ++binding_index) {
    state.contention_retry_eligible_by_binding[binding_index] = 1U;
  }
  if (cuda_graph_region_resources_ != nullptr) {
    auto gated = cuda_graph_region_resources_->bind_output_readiness(state.bindings);
    if (!gated) {
      append_diagnostic(
          registered_token_or(gated.error().message, "RUNTIME_READINESS_BINDING_UNAVAILABLE"));
      return false;
    }
  }
  state.freshness_bindings.clear();
  state.freshness_bindings.reserve(state.input_selectors.size());
  state.freshness_binding_index_by_binding.clear();
  for (std::size_t binding_index = 0U; binding_index < state.bindings.size(); ++binding_index) {
    auto& binding = state.bindings[binding_index];
    const bool owns_freshness_selector = std::any_of(
        state.input_selectors.begin(), state.input_selectors.end(), [&](const auto& selector) {
          return selector.freshness_reconcile && selector.readiness_source == binding->source();
        });
    if (owns_freshness_selector) {
      if (state.freshness_binding_index_by_binding.empty()) {
        state.freshness_binding_index_by_binding.assign(state.bindings.size(),
                                                        std::numeric_limits<std::size_t>::max());
      }
      state.freshness_binding_index_by_binding[binding_index] = state.freshness_bindings.size();
      const bool tick_reconciled = std::all_of(
          state.input_selectors.begin(), state.input_selectors.end(), [&](const auto& selector) {
            return !selector.freshness_reconcile ||
                   selector.readiness_source != binding->source() || selector.tick_fenced;
          });
      state.freshness_bindings.push_back(ReadyRunState::FreshnessBindingSlot{
          .binding = binding.get(), .tick_reconciled = tick_reconciled});
    }
  }
  for (const auto& selector : state.input_selectors) {
    if (!selector.freshness_reconcile) {
      continue;
    }
    const bool source_bound = std::any_of(
        state.freshness_bindings.begin(), state.freshness_bindings.end(), [&](const auto& slot) {
          return slot.binding != nullptr && slot.binding->source() == selector.readiness_source;
        });
    if (!selector.readiness_source || !source_bound) {
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", selector.owner);
      return false;
    }
  }
  // Window timeouts and clock-health expiry are evaluated by polling, and a clock whose rate
  // differs from the steady clock's cannot be waited out by a steady-clock deadline. A freshness
  // reconcile keeps the newest survivors only if the owner witnesses side arrivals between
  // triggers, which side-only inputs no longer announce, unless its tick scan alone reconciles it.
  const bool owns_window_trigger = std::any_of(
      plan_->window_triggers.begin(), plan_->window_triggers.end(), [&](const auto& window) {
        const auto receiver = exec.receiver_index(window.owner.id);
        return window.owner.plan_generation == plan_->plan_generation && receiver &&
               owns_receiver(state, *receiver);
      });
  state.late_drain_elidable = state.freshness_bindings.empty() && !owns_window_trigger &&
                              cuda_graph_region_resources_ == nullptr;
  state.publishes_tick_fences = std::any_of(
      state.input_selectors.begin(), state.input_selectors.end(), [](const auto& selector) {
        return selector.tick_fenced;
      });
  state.long_park_eligible =
      !state.bindings.empty() && !owns_window_trigger &&
      std::all_of(state.input_selectors.begin(),
                  state.input_selectors.end(),
                  [](const auto& selector) {
                    return !selector.freshness_reconcile || selector.tick_reconcilable;
                  }) &&
      state.clock_health->receiver_count() == 0U &&
      std::all_of(state.bindings.begin(),
                  state.bindings.end(),
                  [](const auto& binding) { return binding->progress_announced(); }) &&
      std::all_of(state.clock_bindings.begin(), state.clock_bindings.end(), [&](const auto* clock) {
        return clock == nullptr || exec.clock_registry().advances_at_steady_rate(clock->clock_id());
      });
  if (manage_channel_session && !refresh_and_arm_channels(state.channels)) {
    return false;
  }
  return initialize_direct_composites(exec, state);
}

[[nodiscard]] bool RunSessionState::refresh_and_arm_channels(RuntimeChannelSession* channels) {
  if (channels == nullptr) {
    return true;
  }
  auto refreshed = channels->refresh_connections();
  if (!refreshed) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR("Channel connection refresh failed with {}",
                                 error_code_name(refreshed.error().code));
    const RuntimeDiagnosticDetail detail{.present = true, .error = refreshed.error()};
    append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", {}, {}, &detail);
    return false;
  }
  return arm_channels_with_diagnostics(channels);
}

[[nodiscard]] bool RunSessionState::arm_channels_with_diagnostics(
    RuntimeChannelSession* channels) noexcept {
  if (channels == nullptr) {
    return true;
  }
  auto armed = channels->arm();
  if (armed) {
    return true;
  }
  const std::string_view token = channel_terminal_token(channels->terminal_diagnostic());
  HOLOSCAN_LOG_AUTOMATIC_ERROR("Channel readiness arm failed with {}",
                               error_code_name(armed.error().code));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    RuntimeDiagnosticDetail detail{.present = true, .error = armed.error()};
    std::string required;
    std::string provided;
    if (const auto* placement = channels->support_thread_placement_failure()) {
      detail.error = placement->error;
      detail.operation = DiagnosticOperation::kRuntime;
      detail.component_path = "support-thread/channel-pump";
      required = runtime_support_thread_placement_required(*placement);
      provided = runtime_support_thread_placement_provided(*placement);
    }
    append_diagnostic(token, {}, {}, &detail, required, provided);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    // Diagnostic enrichment is best effort on this noexcept restart/startup boundary.
    append_diagnostic(token);
  }
#endif
  return false;
}

[[nodiscard]] PortBinding* RunSessionState::physical_binding_for(ReadyRunState& state,
                                                                 SourceId source) noexcept {
  const auto found =
      std::find_if(state.bindings.begin(), state.bindings.end(), [&](const auto& binding) {
        return binding->source() && binding->source() == source;
      });
  return found == state.bindings.end() ? nullptr : found->get();
}

// Direct-chain admission derives one all-or-nothing topology and binding contract.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool RunSessionState::initialize_direct_chain(GraphExec& exec,
                                                            ReadyRunState& state) noexcept {
  constexpr std::size_t kNoReceiver = std::numeric_limits<std::size_t>::max();
  state.direct_chain.successor_by_receiver.assign(exec.operators().size(), kNoReceiver);
  state.direct_chain.binding_by_producer.assign(exec.operators().size(), nullptr);
  state.direct_chain.root_source = {};
  state.direct_chain.root_clock = {};
  state.direct_chain.root_binding = nullptr;
  state.direct_chain.root_receiver = kNoReceiver;
  state.direct_chain.next_root_wake.reset();
  state.direct_chain.boundary_bindings.clear();
  const bool has_boundary_edges =
      state.runner != nullptr && (!state.runner->incoming_boundary_edges.empty() ||
                                  !state.runner->outgoing_boundary_edges.empty());
  if (has_boundary_edges && plan_->executable_factories.test_disable_boundary_causal_chain) {
    return true;
  }
  if (state.runner == nullptr || state.runner->operators.size() < 2U ||
      state.runner->internal_edges.size() + 1U != state.runner->operators.size() ||
      std::ranges::count_if(state.runner->operators,
                            [](const auto& member) { return member.topology_root; }) != 1 ||
      !state.runner->operators.front().topology_root ||
      state.runner->operators.front().trigger_kind != TemporalRecordTriggerKind::kClock ||
      !state.runner->operators.front().readiness_source) {
    return true;
  }
  // Preserve the already-proven two-Operator closed-island transaction. Besides being the
  // tighter execution shape, that transaction owns exact no-emission, failure, stop, and
  // generation-boundary semantics. The causal chain specializes longer compiler-proven chains;
  // installing both would create competing successor TagTime authorities.
  if (state.runner->operators.size() == 2U &&
      std::ranges::any_of(plan_->closed_island_activations,
                          [&](const ClosedIslandActivationPlanRecord& record) {
                            return record.process_group == exec.process_group() &&
                                   record.source == state.runner->operators[0].vertex &&
                                   record.successor == state.runner->operators[1].vertex &&
                                   record.source_partition == state.runner->partition &&
                                   record.successor_partition == state.runner->partition;
                          })) {
    return true;
  }
  const TemporalTriggerPlanRecord* const root_trigger =
      find_temporal_trigger_by_owner(*plan_, state.runner->operators.front().operator_id);
  if (root_trigger == nullptr ||
      root_trigger->readiness_source != state.runner->operators.front().readiness_source ||
      !root_trigger->trigger_clock.valid() ||
      root_trigger->trigger_clock != root_trigger->reference_clock) {
    return true;
  }
  RuntimeOperatorInstance* const root_instance =
      exec.operator_by_vertex(state.runner->operators.front().vertex);
  const auto root_receiver = exec.receiver_index(state.runner->operators.front().vertex);
  if (root_instance == nullptr || !root_receiver ||
      root_instance->operator_id != state.runner->operators.front().operator_id) {
    return false;
  }
  if (*root_receiver >= state.clock_bindings.size() ||
      state.clock_bindings[*root_receiver] == nullptr ||
      state.clock_bindings[*root_receiver]->source() != root_trigger->readiness_source) {
    return false;
  }
  for (std::size_t index = 1U; index < state.runner->operators.size(); ++index) {
    if (state.runner->operators[index].trigger_kind != TemporalRecordTriggerKind::kEach) {
      return true;
    }
  }
  // A boundary input may feed a member only as a side input: the clock root's, or an OnEach
  // member's other than its trigger. An arrival there cannot activate anything, so the chain's
  // successor order is still the whole causal frame. Outgoing boundary edges need nothing: a
  // continuation polls only the successor's binding, so another partition observes its recipient's
  // store exactly as without one.
  std::vector<PortBinding*> boundary_bindings;
  for (const EdgeId edge_id : state.runner->incoming_boundary_edges) {
    const auto edge = std::find_if(
        plan_->payload_edges.begin(),
        plan_->payload_edges.end(),
        [&](const PayloadEdgePlanRecord& candidate) { return candidate.edge == edge_id; });
    const auto* target =
        edge != plan_->payload_edges.end() ? find_port_by_id(*plan_, edge->target) : nullptr;
    if (target == nullptr) {
      return true;
    }
    const auto member = std::ranges::find_if(state.runner->operators, [&](const auto& candidate) {
      return candidate.vertex == target->owner;
    });
    if (member == state.runner->operators.end()) {
      return true;
    }
    if (member != state.runner->operators.begin()) {
      const TemporalTriggerPlanRecord* const trigger =
          find_temporal_trigger_by_owner(*plan_, member->operator_id);
      if (trigger == nullptr || trigger->trigger_input == target->id) {
        return true;
      }
    }
    const auto selector = std::ranges::find_if(
        state.input_selectors, [&](const auto& slot) { return slot.input == target->id; });
    PortBinding* const binding = selector != state.input_selectors.end()
                                     ? physical_binding_for(state, selector->readiness_source)
                                     : nullptr;
    if (binding == nullptr) {
      return true;
    }
    if (std::ranges::find(boundary_bindings, binding) == boundary_bindings.end()) {
      boundary_bindings.push_back(binding);
    }
  }

  std::vector<std::size_t> successor_by_receiver(exec.operators().size(), kNoReceiver);
  std::vector<PortBinding*> binding_by_producer(exec.operators().size(), nullptr);
  std::vector<PortId> chain_inputs;
  chain_inputs.reserve(state.runner->operators.size() - 1U);
  for (std::size_t index = 0U; index + 1U < state.runner->operators.size(); ++index) {
    const auto& producer = state.runner->operators[index];
    const auto& successor = state.runner->operators[index + 1U];
    RuntimeOperatorInstance* const producer_instance = exec.operator_by_vertex(producer.vertex);
    RuntimeOperatorInstance* const successor_instance = exec.operator_by_vertex(successor.vertex);
    const auto producer_receiver = exec.receiver_index(producer.vertex);
    const auto successor_receiver = exec.receiver_index(successor.vertex);
    if (producer_instance == nullptr || successor_instance == nullptr || !producer_receiver ||
        !successor_receiver || producer_instance->operator_id != producer.operator_id ||
        successor_instance->operator_id != successor.operator_id) {
      return false;
    }
    std::size_t edge_count = 0U;
    PortId chain_input{};
    for (const EdgeId edge_id : state.runner->internal_edges) {
      const auto edge = std::find_if(
          plan_->payload_edges.begin(),
          plan_->payload_edges.end(),
          [&](const PayloadEdgePlanRecord& candidate) { return candidate.edge == edge_id; });
      if (edge == plan_->payload_edges.end()) {
        continue;
      }
      const auto* source = find_port_by_id(*plan_, edge->source);
      const auto* target = find_port_by_id(*plan_, edge->target);
      if (source != nullptr && target != nullptr && source->owner == producer.vertex &&
          target->owner == successor.vertex) {
        ++edge_count;
        chain_input = target->id;
      }
    }
    if (edge_count != 1U) {
      return true;
    }
    chain_inputs.push_back(chain_input);
    if (!owns_receiver(state, *producer_receiver) || !owns_receiver(state, *successor_receiver) ||
        successor_by_receiver[*producer_receiver] != kNoReceiver) {
      return false;
    }
    PortBinding* const successor_binding = physical_binding_for(state, successor.readiness_source);
    if (successor_binding == nullptr || !successor.readiness_source ||
        successor_binding->source() != successor.readiness_source) {
      return false;
    }
    successor_by_receiver[*producer_receiver] = *successor_receiver;
    binding_by_producer[*producer_receiver] = successor_binding;
  }
  // A freshness reconcile runs in the ordinary scan, before the owner of its selector is frozen. A
  // selector on the clock root or on a member's side input keeps that: the root is never continued
  // into, and continuation already keeps the hop into such a member on the ordinary scan. Only a
  // chain input would be selected by a continuation without its reconcile.
  if (has_boundary_edges && std::ranges::any_of(state.input_selectors, [&](const auto& slot) {
        return slot.freshness_reconcile &&
               std::ranges::find(chain_inputs, slot.input) != chain_inputs.end();
      })) {
    return true;
  }
  state.direct_chain.successor_by_receiver = std::move(successor_by_receiver);
  state.direct_chain.binding_by_producer = std::move(binding_by_producer);
  state.direct_chain.root_source = state.runner->operators.front().readiness_source;
  state.direct_chain.root_clock = root_trigger->reference_clock;
  state.direct_chain.root_binding = state.clock_bindings[*root_receiver];
  state.direct_chain.root_receiver = *root_receiver;
  state.direct_chain.boundary_bindings = std::move(boundary_bindings);

  // Continue into a member whose other inputs are plain DrainAll side inputs, and through the
  // producer's direct fanout output. A side input's selection depends only on the arrivals already
  // witnessed, and the continued poll requires that it hold none unobserved, so the member selects
  // what the ordinary scan would at that instant. Gating selectors (LatestWithin, ReuseLast,
  // predicate, and aligned inputs) keep their hop on the ordinary scan.
  if (state.channels != nullptr &&
      !plan_->executable_factories.test_disable_side_input_continuation) {
    for (std::size_t index = 0U; index < chain_inputs.size(); ++index) {
      const auto& producer = state.runner->operators[index];
      const auto& successor = state.runner->operators[index + 1U];
      const TemporalTriggerPlanRecord* const trigger =
          find_temporal_trigger_by_owner(*plan_, successor.operator_id);
      const bool side_inputs_only_drain =
          std::ranges::all_of(state.input_selectors, [&](const auto& slot) {
            if (slot.owner != successor.vertex || slot.input == chain_inputs[index]) {
              return true;
            }
            const InputSelectionPlanRecord& plan = slot.selector.plan();
            return plan.selector == InputSelectorKind::kDrainAll &&
                   plan.missing != TemporalRecordMissingKind::kReuseLast &&
                   !slot.nontrigger_prerequisite && !slot.freshness_reconcile &&
                   !slot.predicate_input && !slot.aligned_member;
          });
      if (trigger != nullptr && trigger->trigger_input == chain_inputs[index] &&
          side_inputs_only_drain) {
        static_cast<void>(
            state.channels->enable_chain_continuation(producer.vertex, chain_inputs[index]));
      }
    }
  }
  return true;
}

// Composite initialization preserves conservative fallback across every candidate island.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool RunSessionState::initialize_direct_composites(GraphExec& exec,
                                                                 ReadyRunState& state) noexcept {
  state.direct_chain.composites_by_receiver.assign(exec.operators().size(), {});
  state.direct_chain.composite_owner_by_receiver.assign(exec.operators().size(),
                                                        std::numeric_limits<std::size_t>::max());
  if (!initialize_direct_chain(exec, state)) {
    return false;
  }
  // A strict chain owns every one of its causal edges. Do not also install a closed-island
  // composite for that same runner; its independent successor freeze would create a second
  // TagTime authority. Matching two-Operator closed islands were deliberately retained above
  // and therefore leave direct_chain.root_source empty.
  if (state.direct_chain.root_source) {
    return true;
  }
  if (!plan_->inline_partition_execution && !state.inline_partition_execution) {
    return true;
  }
  for (const ClosedIslandActivationPlanRecord& record : plan_->closed_island_activations) {
    if (record.process_group != exec.process_group()) {
      continue;
    }
    RuntimeOperatorInstance* const producer = exec.operator_by_vertex(record.source);
    RuntimeOperatorInstance* const successor = exec.operator_by_vertex(record.successor);
    const auto producer_receiver = exec.receiver_index(record.source);
    const auto successor_receiver = exec.receiver_index(record.successor);
    if (producer == nullptr || successor == nullptr || !producer_receiver || !successor_receiver ||
        producer->partition != record.source_partition ||
        successor->partition != record.successor_partition ||
        producer->execution_lane != record.execution_lane ||
        successor->execution_lane != record.execution_lane) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", record.source);
      return false;
    }
    if (!owns_receiver(state, *producer_receiver)) {
      continue;
    }
    if (!owns_receiver(state, *successor_receiver)) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", record.source);
      return false;
    }
    const auto edge = std::find_if(plan_->payload_edges.begin(),
                                   plan_->payload_edges.end(),
                                   [&](const PayloadEdgePlanRecord& candidate) {
                                     return candidate.edge == record.edge &&
                                            candidate.source == record.source_output &&
                                            candidate.target == record.successor_input;
                                   });
    const TemporalTriggerPlanRecord* const source_trigger = producer->temporal_trigger;
    bool readiness_replayed = false;
    if (source_trigger != nullptr && source_trigger->readiness_source == record.readiness_source) {
      if (record.readiness_root == ClosedIslandReadinessRootKind::kClock) {
        const auto request =
            std::find_if(state.clock_binding_requests.begin(),
                         state.clock_binding_requests.end(),
                         [&](const ReadyRunState::ClockBindingRequest& candidate) {
                           return candidate.receiver_index == *producer_receiver &&
                                  candidate.source.kind == RuntimeEventKind::kClockTick &&
                                  candidate.source.source == record.readiness_source &&
                                  candidate.source.source_clock == record.readiness_clock &&
                                  candidate.source.reference_clock == record.readiness_clock &&
                                  candidate.schedule.clock == record.readiness_clock;
                         });
        readiness_replayed = request != state.clock_binding_requests.end();
      } else {
        const ReadinessPlanRecord* const readiness = find_readiness_by_owner(*plan_, record.source);
        const bool notification =
            readiness != nullptr && readiness->kind == ReadinessSourceKind::kNotification;
        const auto request =
            std::find_if(state.binding_requests.begin(),
                         state.binding_requests.end(),
                         [&](const RuntimePortBindingRequest& candidate) {
                           return readiness != nullptr &&
                                  candidate.kind == runtime_event_kind(readiness->kind) &&
                                  candidate.source == record.readiness_source &&
                                  candidate.source_clock == record.readiness_clock &&
                                  candidate.reference_clock == record.readiness_clock &&
                                  candidate.execution_lane == record.execution_lane;
                         });
        readiness_replayed =
            readiness != nullptr && readiness->source_id == record.readiness_source &&
            readiness->resolved_clock == record.readiness_clock &&
            notification ==
                (record.readiness_root == ClosedIslandReadinessRootKind::kNotification) &&
            request != state.binding_requests.end();
      }
    }
    if (edge == plan_->payload_edges.end() || !readiness_replayed) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", record.source);
      return false;
    }
    if (*producer_receiver >= state.direct_chain.composites_by_receiver.size() ||
        *successor_receiver >= state.direct_chain.composite_owner_by_receiver.size() ||
        state.direct_chain.composites_by_receiver[*producer_receiver].valid() ||
        state.direct_chain.composite_owner_by_receiver[*producer_receiver] !=
            std::numeric_limits<std::size_t>::max() ||
        state.direct_chain.composite_owner_by_receiver[*successor_receiver] !=
            std::numeric_limits<std::size_t>::max()) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", record.source);
      return false;
    }
    const auto request =
        std::find_if(state.channel_binding_requests.begin(),
                     state.channel_binding_requests.end(),
                     [&](const auto& candidate) {
                       return candidate.first == record.successor &&
                              candidate.second.kind == RuntimeEventKind::kInputArrival &&
                              candidate.second.execution_lane == record.execution_lane &&
                              !candidate.second.port &&
                              candidate.second.source == record.successor_readiness_source;
                     });
    if (request == state.channel_binding_requests.end()) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", record.successor);
      return false;
    }
    PortBinding* const successor_binding = physical_binding_for(state, request->second.source);
    if (successor_binding == nullptr || successor_binding->poll_target().has_value()) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", record.successor);
      return false;
    }
    state.direct_chain.composites_by_receiver[*producer_receiver] = RuntimeDirectCompositeBinding{
        .producer_receiver = *producer_receiver,
        .successor_receiver = *successor_receiver,
        .producer = record.source,
        .successor = record.successor,
        .source_output = record.source_output,
        .successor_input = record.successor_input,
        .readiness_source = record.successor_readiness_source,
        .successor_binding = successor_binding,
        .execution_lane = record.execution_lane,
    };
    state.direct_chain.composite_owner_by_receiver[*producer_receiver] = *producer_receiver;
    state.direct_chain.composite_owner_by_receiver[*successor_receiver] = *producer_receiver;
  }
  return true;
}

[[nodiscard]] bool RunSessionState::cancel_ready_sources(ReadyRunState& state) noexcept {
  bool retired_all = true;
  if (!state.partition_owner_execution && state.notifications != nullptr) {
    state.notifications->close();
  }
  // A lifecycle-candidate plan can fail in Configure/Discover before the first Core
  // kAllocate materializer constructs readiness. Treat that expected fail-closed state as an
  // empty source inventory instead of dereferencing an absent coordinator during shutdown.
  if (state.coordinator == nullptr) {
    for (auto& binding : state.bindings) {
      binding->cancel();
    }
    return state.selection.frozen_candidates.empty();
  }
  // Every pending witness is frozen before dispatch begins. Retire that
  // complete frozen prefix exactly once, then let coordinator cancellation
  // release any witnesses left pending by a freeze failure.
  for (auto& candidate : state.selection.frozen_candidates) {
    if (!candidate.source_retired) {
      if (!retire_frozen_source(state, candidate)) {
        retired_all = false;
      }
    }
  }
  state.coordinator->cancel_all();
  for (auto& binding : state.bindings) {
    binding->cancel();
  }
  return retired_all;
}

[[nodiscard]] bool RunSessionState::stop_channels(RuntimeChannelSession* channels) noexcept {
  if (channels == nullptr) {
    return true;
  }
  channels->cancel();
  const auto joined = channels->join();
  if (!joined) {
    RuntimeDiagnosticDetail detail{.present = true,
                                   .operation = DiagnosticOperation::kRuntime,
                                   .role = DiagnosticFailureRole::kTeardown};
    detail.capture.emplace();
    detail.capture->append(joined.error().code, joined.error().message);
    append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", {}, {}, &detail);
    return false;
  }
  return true;
}

void RunSessionState::initialize_local_operator_completion_progress() {
  const auto factory_for = [this](OperatorInstanceId owner) {
    const auto& operators = plan_->executable_factories.operators;
    const auto factory = std::find_if(operators.begin(), operators.end(), [&](const auto& row) {
      return row.operator_id == owner;
    });
    const std::size_t factory_count = static_cast<std::size_t>(
        std::count_if(operators.begin(), operators.end(), [&](const auto& row) {
          return row.operator_id == owner;
        }));
    if (factory == operators.end() || factory_count != 1U ||
        !process_group_for(*plan_, factory->vertex)) {
      throw RuntimeError{Error{ErrorCode::kIncompatible, "RUNTIME_OPERATOR_REPLAY_MISMATCH"}};
    }
    return &*factory;
  };

  std::size_t local_rule_count = 0U;
  OperatorInstanceId previous_owner{};
  for (const OperatorCompletionPlanRecord& rule : plan_->operator_completion_rules) {
    if (!rule.owner.plan_bound() || rule.owner.plan_generation != plan_->plan_generation ||
        rule.kind != OperatorCompletionRuleKind::kComputeCalls || rule.owner.id <= previous_owner) {
      throw RuntimeError{Error{ErrorCode::kIncompatible, "RUNTIME_OPERATOR_REPLAY_MISMATCH"}};
    }
    const ExecutableOperatorFactoryRecord* const factory = factory_for(rule.owner.id);
    if (worker_owns_vertex(*plan_, process_group_, factory->vertex)) {
      ++local_rule_count;
    }
    previous_owner = rule.owner.id;
  }

  operator_completion_progress_.reserve(local_rule_count);
  operator_completion_phases_.reserve(local_rule_count);
  for (const OperatorCompletionPlanRecord& rule : plan_->operator_completion_rules) {
    const ExecutableOperatorFactoryRecord* const factory = factory_for(rule.owner.id);
    if (!worker_owns_vertex(*plan_, process_group_, factory->vertex)) {
      continue;
    }
    operator_completion_progress_.push_back(OperatorCompletionProgressRecord{
        .owner =
            OperatorRef{
                .id = rule.owner.id,
                .plan_generation = rule.owner.plan_generation,
                .run_generation = run_generation_,
            },
        .kind = rule.kind,
        .configured_limit = rule.limit,
    });
    operator_completion_phases_.push_back(OperatorCompletionPhase::kOpen);
  }

  const auto& seed_control = plan_->executable_factories.test_operator_completion_progress_seed;
  if (seed_control == nullptr) {
    return;
  }
  std::uint64_t seeded_successful_compute_calls{};
  if (operator_completion_progress_.size() != 1U || operator_completion_phases_.size() != 1U ||
      operator_completion_progress_.front().kind != OperatorCompletionRuleKind::kComputeCalls ||
      !seed_control->consume(seeded_successful_compute_calls) ||
      seeded_successful_compute_calls >= operator_completion_progress_.front().configured_limit) {
    throw RuntimeError{Error{ErrorCode::kIncompatible, "RUNTIME_OPERATOR_REPLAY_MISMATCH"}};
  }
  operator_completion_progress_.front().successful_compute_calls = seeded_successful_compute_calls;
}

void RunSessionState::initialize_after_progress_materialization() {
  const auto& startup_failure_control =
      plan_->executable_factories.test_startup_progress_failure_control;
  if (startup_failure_control != nullptr && startup_failure_control->consume()) {
    throw RuntimeError{Error{ErrorCode::kResourceExhausted}};
  }
  lifecycle_control_ =
      std::make_shared<LifecycleControlState>(plan_, run_generation_, process_group_);
  local_partition_runner_count_ =
      static_cast<std::size_t>(std::count_if(plan_->partition_runners.begin(),
                                             plan_->partition_runners.end(),
                                             [&](const PartitionRunnerPlanRecord& runner) {
                                               return runner.process_group == process_group_;
                                             }));
  if (local_partition_runner_count_ > 1U) {
    partition_wakes_.reserve(local_partition_runner_count_);
    for (std::size_t index = 0U; index < local_partition_runner_count_; ++index) {
      partition_wakes_.push_back(std::make_unique<RuntimeWakeState>());
      partition_wakes_.back()->accounting_partition = static_cast<std::uint32_t>(index);
    }
  }
  lifecycle_control_->bind_wake(this, [](void* state) noexcept {
    auto* const session = static_cast<RunSessionState*>(state);
    session->signal_controller_wake(RuntimePhaseCounter::kWakeSignals);
  });
  std::vector<ParamAddress> parameter_fields;
  parameter_fields.reserve(plan_->parameter_fields.size());
  for (const ParameterFieldPlanRecord& field : plan_->parameter_fields) {
    parameter_fields.push_back(ParamAddress{
        .operator_instance =
            OperatorRef{
                .id = field.owner.id,
                .plan_generation = plan_->plan_generation,
                .run_generation = run_generation_,
            },
        .field = field.field,
    });
  }
  auto link = ParamControlAccess::create_unattached_link(
      plan_->plan_generation,
      run_generation_,
      static_cast<OperatorInstanceId>(plan_->operator_parameters.size()),
      parameter_fields);
  if (!link) {
    throw RuntimeError{std::move(link.error())};
  }
  parameter_control_link_ = std::move(link).value();
}

void RunSessionState::fail_startup() noexcept {
  append_diagnostic("RUNTIME_STARTUP_FAILED");
  finish(RunTermination::kFailed);
}

}  // namespace holoscan::detail
