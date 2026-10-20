/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "optional_input_plan_validation.hpp"

#include <algorithm>

#include "core/graph_compiler_access.hpp"
#include "execution_plan_lookup.hpp"
#include "execution_plan_state.hpp"
#include "snapshot_lookup.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] bool eligible_optional_selection(const ExecutionPlanState& state,
                                               const PortPlanRecord& port,
                                               const InputSelectionPlanRecord& selection) {
  const ExecutableOperatorFactoryRecord* factory = nullptr;
  for (const auto& candidate : state.executable_factories.operators) {
    if (candidate.vertex == port.owner) {
      if (factory != nullptr) {
        return false;
      }
      factory = &candidate;
    }
  }
  if (factory == nullptr) {
    return false;
  }
  const auto* trigger = find_unique_temporal_trigger_by_owner(state, factory->operator_id);
  if (trigger == nullptr || trigger->owner.plan_generation != state.plan_generation ||
      trigger->owner.run_generation != 0U || trigger->trigger_input == port.id ||
      selection.selector != InputSelectorKind::kExact ||
      selection.cardinality != InputSelectionCardinality::kSingleSample ||
      selection.max_selected_samples != 1U ||
      selection.max_selected_samples_origin != SelectedSampleBoundOrigin::kSingleSample ||
      selection.batch_selection_mode != BatchSelectionMode::kNotApplicable ||
      selection.missing != TemporalRecordMissingKind::kSafeDefault || selection.max_age_ns != 0 ||
      selection.reuse_max_age_ns != 0 || selection.lookbehind != 0U || selection.lookahead != 0U ||
      selection.reference_clock != trigger->reference_clock || !selection.reference_clock.valid() ||
      std::find(trigger->declared_inputs.begin(), trigger->declared_inputs.end(), port.id) ==
          trigger->declared_inputs.end() ||
      std::find(trigger->presence_guaranteed_inputs.begin(),
                trigger->presence_guaranteed_inputs.end(),
                port.id) != trigger->presence_guaranteed_inputs.end() ||
      std::any_of(state.predicate_inputs.begin(),
                  state.predicate_inputs.end(),
                  [&](const auto& input) { return input.input == port.id; }) ||
      std::any_of(state.aligned_triggers.begin(),
                  state.aligned_triggers.end(),
                  [&](const auto& aligned) {
                    return std::any_of(aligned.members.begin(),
                                       aligned.members.end(),
                                       [&](const auto& member) { return member.input == port.id; });
                  }) ||
      std::any_of(
          state.bracketed_windows.begin(), state.bracketed_windows.end(), [&](const auto& window) {
            return window.input == port.id || window.anchor == port.id;
          })) {
    return false;
  }
  return !selection.unwired_safe_default ||
         (selection.cache_capacity == 1U && selection.conversion_id == 0U);
}

[[nodiscard]] bool valid_optional_selection(const ExecutionPlanState& state,
                                            const GraphSnapshotData& snapshot,
                                            const InputSelectionPlanRecord& selection) {
  const auto* port = find_port_by_id(state, selection.input);
  if (port == nullptr) {
    return !selection.unwired_safe_default;
  }
  if (!port->optional && !selection.unwired_safe_default) {
    return true;
  }
  const auto* authored = find_port(snapshot, GraphPortId{selection.input.value()});
  if (authored == nullptr || !authored->optional || !port->optional ||
      !eligible_optional_selection(state, *port, selection)) {
    return false;
  }
  const auto aliases = interface_alias_component(snapshot, authored->id);
  const bool connected =
      std::any_of(snapshot.edges.begin(), snapshot.edges.end(), [&](const auto& edge) {
        return std::find(aliases.begin(), aliases.end(), edge.target) != aliases.end();
      });
  if (selection.unwired_safe_default == connected) {
    return false;
  }
  return !selection.unwired_safe_default ||
         std::none_of(state.payload_edges.begin(),
                      state.payload_edges.end(),
                      [&](const auto& edge) { return edge.target == selection.input; });
}

}  // namespace

bool valid_optional_input_plan(const ExecutionPlanState& state) {
  // Rejected plans retain partial records solely to explain compilation errors.
  if (!state.admitted) {
    return true;
  }
  const auto snapshot = GraphCompilerAccess::retain_snapshot(state.graph);
  if (!snapshot) {
    return std::none_of(state.ports.begin(),
                        state.ports.end(),
                        [](const auto& port) { return port.optional; }) &&
           std::none_of(state.input_selections.begin(),
                        state.input_selections.end(),
                        [](const auto& input) { return input.unwired_safe_default; });
  }
  for (const auto& port : state.ports) {
    const auto* authored = find_port(*snapshot, GraphPortId{port.id.value()});
    if (port.optional != (authored != nullptr && authored->optional)) {
      return false;
    }
  }
  for (const auto& authored : snapshot->ports) {
    if (!authored.optional) {
      continue;
    }
    const auto* port = find_port_by_id(state, PortId{authored.id.value});
    if (authored.owner_kind != PortOwnerKind::kOperator ||
        authored.direction != PortDirection::kInput || port == nullptr || !port->optional ||
        port->owner != authored.owner || port->owner_kind != authored.owner_kind ||
        port->direction != authored.direction ||
        std::count_if(state.input_selections.begin(),
                      state.input_selections.end(),
                      [&](const auto& input) { return input.input == port->id; }) != 1) {
      return false;
    }
  }
  return std::all_of(
      state.input_selections.begin(), state.input_selections.end(), [&](const auto& selection) {
        return valid_optional_selection(state, *snapshot, selection);
      });
}

}  // namespace holoscan::detail
