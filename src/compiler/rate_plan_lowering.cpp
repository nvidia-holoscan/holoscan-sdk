/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/rate_plan_lowering.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../core/diagnostic_registry.hpp"
#include "../core/rate_math_internal.hpp"
#include "compiler/snapshot_lookup.hpp"

namespace holoscan::detail {
namespace {

enum class ResolutionState : std::uint8_t {
  kUnvisited = 0,
  kResolving = 1,
  kResolved = 2,
  kAbsent = 3,
  kPending = 4,
  kFailed = 5,
};

struct PortResolution {
  ResolutionState state{ResolutionState::kUnvisited};
  std::optional<Rate> rate;
  std::optional<Rate> maximum;
  std::optional<PortId> immediate;
  std::optional<PortId> root;
  ResolvedRateOrigin origin{ResolvedRateOrigin::kUnresolved};
  RateComparison expectation_comparison{RateComparison::kNotCompared};
};

[[nodiscard]] DiagnosticSource source_for(const GraphSnapshotData& snapshot,
                                          const PortRecord& port) {
  return DiagnosticSource{
      .graph_generation = snapshot.generation, .vertex = port.owner, .port = PortId{port.id.value}};
}

void add_diagnostic(std::vector<PlanLoweringDiagnostic>& diagnostics, std::string_view token,
                    const GraphSnapshotData& snapshot, const PortRecord& port) {
  static_cast<void>(append_unique_plan_lowering_diagnostic(
      diagnostics,
      PlanLoweringDiagnostic{.token = std::string(token), .source = source_for(snapshot, port)}));
}

[[nodiscard]] const EdgeRecord* incoming_edge(const GraphSnapshotData& snapshot,
                                              GraphPortId input) noexcept {
  const EdgeRecord* match{};
  for (const auto& edge : snapshot.edges) {
    if (resolve_interface_port(snapshot, edge.target) != input) {
      continue;
    }
    if (match != nullptr) {
      return nullptr;
    }
    match = &edge;
  }
  return match;
}

[[nodiscard]] const TemporalTriggerPlanRecord* trigger_for_owner(
    std::span<const TemporalTriggerPlanRecord> triggers, VertexId owner) noexcept {
  const auto found = std::find_if(triggers.begin(), triggers.end(), [&](const auto& trigger) {
    return trigger.owner.id == owner.value;
  });
  return found == triggers.end() ? nullptr : &*found;
}

[[nodiscard]] const WindowTriggerPlanRecord* window_trigger_for_owner(
    std::span<const WindowTriggerPlanRecord> windows, VertexId owner) noexcept {
  const WindowTriggerPlanRecord* match{};
  for (const auto& window : windows) {
    if (window.owner.id != owner.value) {
      continue;
    }
    if (match != nullptr) {
      return nullptr;
    }
    match = &window;
  }
  return match;
}

[[nodiscard]] const PredicateTriggerPlanRecord* predicate_trigger_for_owner(
    std::span<const PredicateTriggerPlanRecord> predicates, VertexId owner) noexcept {
  const PredicateTriggerPlanRecord* match{};
  for (const auto& predicate : predicates) {
    if (predicate.owner.id != owner.value) {
      continue;
    }
    if (match != nullptr) {
      return nullptr;
    }
    match = &predicate;
  }
  return match;
}

[[nodiscard]] bool has_compile_finite_horizon(
    std::span<const TemporalTriggerPlanRecord> triggers,
    std::span<const WindowTriggerPlanRecord> windows,
    std::span<const PredicateTriggerPlanRecord> predicates, VertexId owner) noexcept {
  const auto* trigger = trigger_for_owner(triggers, owner);
  if (trigger == nullptr) {
    return false;
  }
  if (trigger->kind == TemporalRecordTriggerKind::kClock) {
    return trigger->trigger_clock.value != 0U && trigger->period_ns > 0;
  }
  if (trigger->kind == TemporalRecordTriggerKind::kWindow) {
    const auto* window = window_trigger_for_owner(windows, owner);
    return window != nullptr && window->owner == trigger->owner &&
           trigger->trigger_input.value() != 0U && window->anchor.value() != 0U &&
           window->anchor == trigger->trigger_input && trigger->trigger_clock.value != 0U &&
           window->wait_clock == trigger->trigger_clock && window->max_wait_ns > 0;
  }
  if (trigger->kind == TemporalRecordTriggerKind::kPredicate) {
    const auto* predicate = predicate_trigger_for_owner(predicates, owner);
    return predicate != nullptr && predicate->owner == trigger->owner &&
           predicate->source == PredicateEvaluationSourceKind::kClock && predicate->period_ns > 0 &&
           predicate->period_ns == trigger->period_ns && trigger->trigger_clock.value != 0U &&
           predicate->clock == trigger->trigger_clock;
  }
  return false;
}

[[nodiscard]] std::pair<QueueDepthOrigin, std::optional<std::uint32_t>> explicit_depth(
    const EdgeRecord& edge, const PortRecord& target) noexcept {
  if (edge.graph_override.queue_depth) {
    return {QueueDepthOrigin::kGraphEdgeOverride, edge.graph_override.queue_depth};
  }
  if (edge.add_flow_options.queue_depth) {
    return {QueueDepthOrigin::kAddFlowOption, edge.add_flow_options.queue_depth};
  }
  if (target.graph_override.queue_depth) {
    return {QueueDepthOrigin::kGraphInputOverride, target.graph_override.queue_depth};
  }
  if (target.setup.queue_depth) {
    return {QueueDepthOrigin::kOperatorSetup, target.setup.queue_depth};
  }
  return {QueueDepthOrigin::kFrameworkFallback, std::nullopt};
}

[[nodiscard]] std::uint32_t resolved_burst(const PortRecord& output) noexcept {
  return output.graph_override.max_emits_per_compute.value_or(
      output.setup.max_emits_per_compute.value_or(1U));
}

[[nodiscard]] std::optional<std::uint32_t> independent_selector_retention_floor(
    PortId input, std::span<const InputSelectionPlanRecord> selections) noexcept {
  std::optional<std::uint32_t> result;
  for (const auto& selection : selections) {
    if (selection.input != input) {
      continue;
    }
    switch (selection.max_selected_samples_origin) {
      case SelectedSampleBoundOrigin::kSingleSample:
      case SelectedSampleBoundOrigin::kSelectorMaxSamples:
        result = std::max(result.value_or(1U), selection.max_selected_samples);
        break;
      case SelectedSampleBoundOrigin::kResolvedRetainedCapacity:
        // This is the queue/cache capacity viewed through DrainAll, not independent storage.
        break;
    }
  }
  return result;
}

[[nodiscard]] std::optional<std::uint32_t> advancement_capacity(
    PortId input, std::span<const InputSelectionPlanRecord> selections,
    std::uint32_t resolved_retained_capacity) noexcept {
  const auto found = std::ranges::find(selections, input, &InputSelectionPlanRecord::input);
  if (found == selections.end()) {
    return std::nullopt;
  }
  switch (found->selector) {
    case InputSelectorKind::kExact:
    case InputSelectorKind::kLatestWithin:
      return found->max_selected_samples == 0U ? std::nullopt : std::optional<std::uint32_t>{1U};
    case InputSelectorKind::kDrainAll:
      if (found->batch_selection_mode != BatchSelectionMode::kConsumeSelected) {
        return std::nullopt;
      }
      if (found->max_selected_samples_origin ==
          SelectedSampleBoundOrigin::kResolvedRetainedCapacity) {
        return resolved_retained_capacity == 0U
                   ? std::nullopt
                   : std::optional<std::uint32_t>{resolved_retained_capacity};
      }
      return found->max_selected_samples == 0U
                 ? std::nullopt
                 : std::optional<std::uint32_t>{found->max_selected_samples};
    case InputSelectorKind::kBracketedWindow:
      return std::nullopt;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::uint32_t> window_floor(
    PortId input, std::span<const BracketedWindowPlanRecord> windows,
    std::span<const TemporalTriggerPlanRecord> triggers) noexcept {
  std::uint32_t result = 1U;
  for (const auto& window : windows) {
    if (window.input != input) {
      continue;
    }
    const auto trigger = std::find_if(triggers.begin(), triggers.end(), [&](const auto& item) {
      return item.owner.id == window.owner.id && item.kind == TemporalRecordTriggerKind::kWindow;
    });
    if (trigger == triggers.end() || trigger->bounded_state_capacity == 0U ||
        window.max_samples == 0U ||
        window.max_samples >
            std::numeric_limits<std::uint32_t>::max() / trigger->bounded_state_capacity) {
      return std::nullopt;
    }
    result = std::max(result, window.max_samples * trigger->bounded_state_capacity);
  }
  return result;
}

struct HorizonRecipe {
  std::optional<ExactDuration> horizon;
  std::optional<PortId> inverse_rate_port;
  AccumulationHorizonOrigin origin{AccumulationHorizonOrigin::kExternalCadence};
  std::uint32_t source_record{};
};

[[nodiscard]] std::optional<HorizonRecipe> horizon_recipe_for_edge(
    const PortRecord& target, std::span<const TemporalTriggerPlanRecord> triggers,
    std::span<const WindowTriggerPlanRecord> windows,
    std::span<const PredicateTriggerPlanRecord> predicates) noexcept {
  const auto* trigger = trigger_for_owner(triggers, target.owner);
  if (trigger == nullptr) {
    return std::nullopt;
  }
  if (trigger->kind == TemporalRecordTriggerKind::kClock && trigger->trigger_clock.value != 0U &&
      trigger->period_ns > 0) {
    return HorizonRecipe{
        .horizon = ExactDuration{.numerator_ns = static_cast<std::uint64_t>(trigger->period_ns),
                                 .denominator = 1U},
        .origin = AccumulationHorizonOrigin::kClockPeriod,
        .source_record = static_cast<std::uint32_t>(trigger->owner.id),
    };
  }
  if (trigger->kind == TemporalRecordTriggerKind::kWindow) {
    const auto* window = window_trigger_for_owner(windows, target.owner);
    if (window == nullptr || window->owner != trigger->owner ||
        trigger->trigger_input.value() == 0U || window->anchor.value() == 0U ||
        window->anchor != trigger->trigger_input || trigger->trigger_clock.value == 0U ||
        window->wait_clock != trigger->trigger_clock || window->max_wait_ns <= 0) {
      return std::nullopt;
    }
    return HorizonRecipe{
        .horizon = ExactDuration{.numerator_ns = static_cast<std::uint64_t>(window->max_wait_ns),
                                 .denominator = 1U},
        .origin = AccumulationHorizonOrigin::kTemporalWindowBound,
        .source_record = static_cast<std::uint32_t>(window->owner.id),
    };
  }
  if (trigger->kind == TemporalRecordTriggerKind::kPredicate) {
    const auto* predicate = predicate_trigger_for_owner(predicates, target.owner);
    if (predicate == nullptr || predicate->owner != trigger->owner ||
        predicate->source != PredicateEvaluationSourceKind::kClock || predicate->period_ns <= 0 ||
        predicate->period_ns != trigger->period_ns || trigger->trigger_clock.value == 0U ||
        predicate->clock != trigger->trigger_clock) {
      return std::nullopt;
    }
    return HorizonRecipe{
        .horizon = ExactDuration{.numerator_ns = static_cast<std::uint64_t>(predicate->period_ns),
                                 .denominator = 1U},
        .origin = AccumulationHorizonOrigin::kClockPeriod,
        .source_record = static_cast<std::uint32_t>(predicate->owner.id),
    };
  }
  if (trigger->kind != TemporalRecordTriggerKind::kEach || !trigger->trigger_input) {
    return std::nullopt;
  }
  return HorizonRecipe{
      .inverse_rate_port = trigger->trigger_input,
      .origin = trigger->trigger_input == PortId(target.id.value)
                    ? AccumulationHorizonOrigin::kProducerInterarrival
                    : AccumulationHorizonOrigin::kDriverInterarrival,
      .source_record = static_cast<std::uint32_t>(trigger->owner.id),
  };
}

[[nodiscard]] std::optional<ExactDuration> evaluate_horizon_recipe(
    const HorizonRecipe& recipe, std::span<const ResolvedPortRate> resolved_rates) noexcept {
  if (recipe.horizon) {
    return recipe.horizon;
  }
  if (!recipe.inverse_rate_port || !*recipe.inverse_rate_port ||
      recipe.inverse_rate_port->value() > resolved_rates.size()) {
    return std::nullopt;
  }
  const auto& driver =
      resolved_rates[static_cast<std::size_t>(recipe.inverse_rate_port->value() - 1U)];
  return driver.resolved_rate ? rate_math::inverse_horizon(*driver.resolved_rate) : std::nullopt;
}

}  // namespace

const ResolvedQueueSizing* find_resolved_queue_sizing(std::span<const ResolvedQueueSizing> sizings,
                                                      EdgeId edge) noexcept {
  const auto found = std::ranges::find(sizings, edge, &ResolvedQueueSizing::edge);
  return found == sizings.end() ? nullptr : &*found;
}

std::optional<std::uint32_t> compose_hard_floor(
    std::span<const HardFloorContributorPlanRecord> contributors, EdgeId edge) noexcept {
  struct GroupFloor {
    HardFloorScenarioId scenario{};
    HardFloorAlternativeGroupId group{};
    std::uint32_t slots{};
  };
  std::vector<GroupFloor> groups;
  groups.reserve(contributors.size());
  for (const auto& contributor : contributors) {
    if (contributor.edge != edge || contributor.required_slots == 0U) {
      continue;
    }
    const auto found = std::find_if(groups.begin(), groups.end(), [&](const auto& candidate) {
      return candidate.scenario == contributor.scenario &&
             candidate.group == contributor.alternative_group;
    });
    if (found == groups.end()) {
      groups.push_back(GroupFloor{.scenario = contributor.scenario,
                                  .group = contributor.alternative_group,
                                  .slots = contributor.required_slots});
    } else {
      found->slots = std::max(found->slots, contributor.required_slots);
    }
  }
  std::ranges::sort(groups, [](const auto& left, const auto& right) {
    if (left.scenario.value != right.scenario.value) {
      return left.scenario.value < right.scenario.value;
    }
    return left.group.value < right.group.value;
  });

  std::uint32_t hard_floor = 1U;
  std::uint32_t scenario_floor{};
  std::optional<HardFloorScenarioId> scenario;
  for (const auto& group : groups) {
    if (scenario.value_or(group.scenario) != group.scenario) {
      hard_floor = std::max(hard_floor, scenario_floor);
      scenario_floor = 0U;
    }
    scenario = group.scenario;
    if (group.slots > std::numeric_limits<std::uint32_t>::max() - scenario_floor) {
      return std::nullopt;
    }
    scenario_floor += group.slots;
  }
  return std::max(hard_floor, scenario_floor);
}

// This cold finalizer deliberately joins every temporal retention family to one queue authority.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void finalize_temporal_queue_capacities(
    const GraphSnapshotData& snapshot, std::span<const ResolvedQueueSizing> sizings,
    std::span<InputSelectionPlanRecord> input_selections,
    std::span<BracketedWindowPlanRecord> bracketed_windows,
    std::span<AlignedTriggerPlanRecord> aligned_triggers,
    std::span<PredicateInputPlanRecord> predicate_inputs) noexcept {
  const auto depth_for_input = [&](PortId input) -> std::optional<std::uint32_t> {
    const GraphPortId graph_input{input.value()};
    const EdgeRecord* match{};
    for (const auto& edge : snapshot.edges) {
      if (resolve_interface_port(snapshot, edge.target) != graph_input) {
        continue;
      }
      if (match != nullptr) {
        return std::nullopt;
      }
      match = &edge;
    }
    if (match == nullptr) {
      return std::nullopt;
    }
    const auto* sizing = find_resolved_queue_sizing(sizings, match->id);
    return sizing == nullptr ? std::nullopt
                             : std::optional<std::uint32_t>{sizing->effective_logical_depth};
  };

  for (auto& selection : input_selections) {
    if (const auto depth = depth_for_input(selection.input)) {
      selection.cache_capacity = *depth;
      if (selection.selector == InputSelectorKind::kDrainAll &&
          selection.max_selected_samples_origin ==
              SelectedSampleBoundOrigin::kResolvedRetainedCapacity) {
        selection.max_selected_samples = *depth;
      }
    }
  }
  for (auto& window : bracketed_windows) {
    if (const auto depth = depth_for_input(window.input)) {
      window.queue_capacity = *depth;
      window.cache_capacity = *depth;
      window.retained_overlap_capacity = *depth;
    }
  }
  for (auto& trigger : aligned_triggers) {
    for (auto& member : trigger.members) {
      if (const auto depth = depth_for_input(member.input)) {
        member.queue_capacity = *depth;
        member.cache_capacity = *depth;
      }
    }
  }
  for (auto& predicate_input : predicate_inputs) {
    if (const auto depth = depth_for_input(predicate_input.input)) {
      predicate_input.cache_capacity = *depth;
    }
  }
}

// Rate resolution is compile-cold and intentionally centralized here. Runtime consumes dense
// generation views and never repeats this graph traversal or rational arithmetic.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
RatePlanLoweringResult lower_rate_plan(
    const GraphSnapshotData& snapshot, RateRatio headroom_ratio,
    std::span<const TemporalTriggerPlanRecord> temporal_triggers,
    std::span<const InputSelectionPlanRecord> input_selections,
    std::span<const BracketedWindowPlanRecord> bracketed_windows,
    [[maybe_unused]] std::span<const std::optional<Profile>> effective_profiles,
    std::span<const RateBoundedEvidenceRequirement> rate_evidence_requirements,
    [[maybe_unused]] PlanGenerationId plan_generation,
    std::span<const std::uint32_t> process_group_by_vertex,
    std::span<const WindowTriggerPlanRecord> window_triggers,
    std::span<const PredicateTriggerPlanRecord> predicate_triggers) {
  RatePlanLoweringResult result;
  const auto normalized_headroom = rate_math::normalize(headroom_ratio);
  const bool headroom_invalid =
      !normalized_headroom || normalized_headroom->numerator < normalized_headroom->denominator;
  if (headroom_invalid) {
    result.errors.push_back(PlanLoweringDiagnostic{.token = "RATE_HEADROOM_INVALID"});
    // Rejected artifacts still require a canonical total recipe inventory. This ratio is only a
    // deterministic serialization placeholder; RATE_HEADROOM_INVALID keeps every recipe rejected.
    headroom_ratio = RateRatio{1U, 1U};
  } else {
    headroom_ratio = *normalized_headroom;
  }

  std::vector<PortResolution> resolutions(snapshot.ports.size());
  result.port_rates.reserve(snapshot.ports.size());
  for (const auto& port : snapshot.ports) {
    const auto nominal = port.setup.declared_nominal_rate
                             ? rate_math::normalize(*port.setup.declared_nominal_rate)
                             : std::nullopt;
    const auto expected = port.setup.declared_expected_rate
                              ? rate_math::normalize(*port.setup.declared_expected_rate)
                              : std::nullopt;
    const auto maximum = port.setup.declared_max_nominal_rate
                             ? rate_math::normalize(*port.setup.declared_max_nominal_rate)
                             : std::nullopt;
    const auto ratio =
        port.setup.follows_ratio ? rate_math::normalize(*port.setup.follows_ratio) : std::nullopt;
    if ((port.setup.declared_nominal_rate && !nominal) ||
        (port.setup.declared_expected_rate && !expected) ||
        (port.setup.declared_max_nominal_rate && !maximum) ||
        (port.setup.follows_ratio && !ratio)) {
      add_diagnostic(result.errors, "RATE_DECLARATION_INVALID", snapshot, port);
    }
    result.port_rates.push_back(PortRatePlanRecord{
        .port = PortId{port.id.value},
        .kind = port.setup.rate_kind,
        .declared_nominal_rate = nominal,
        .declared_expected_rate = expected,
        .follows_input = port.setup.follows_input
                             ? std::optional<PortId>{PortId{port.setup.follows_input->value}}
                             : std::nullopt,
        .follows_ratio = ratio,
        .lifecycle_binding_stage = port.setup.lifecycle_binding_stage,
        .declared_max_nominal_rate = maximum,
        .disposition = ((port.setup.declared_nominal_rate && !nominal) ||
                        (port.setup.declared_expected_rate && !expected) ||
                        (port.setup.declared_max_nominal_rate && !maximum) ||
                        (port.setup.follows_ratio && !ratio))
                           ? PlanDisposition::kRejected
                           : PlanDisposition::kAccepted,
    });
  }

  std::function<void(GraphPortId)> resolve_port;
  // The compile-cold DFS keeps cycle detection, provenance, fallback, and checked arithmetic in
  // one state transition so a failed derivation cannot be mistaken for an absent declaration.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  resolve_port = [&](GraphPortId authored_id) {
    const GraphPortId port_id = resolve_interface_port(snapshot, authored_id);
    const PortRecord* port = find_port(snapshot, port_id);
    if (port == nullptr || port_id.value == 0U || port_id.value > resolutions.size()) {
      return;
    }
    auto& resolution = resolutions[static_cast<std::size_t>(port_id.value - 1U)];
    if (resolution.state == ResolutionState::kResolved ||
        resolution.state == ResolutionState::kAbsent ||
        resolution.state == ResolutionState::kPending ||
        resolution.state == ResolutionState::kFailed) {
      return;
    }
    if (resolution.state == ResolutionState::kResolving) {
      resolution.state = ResolutionState::kFailed;
      add_diagnostic(result.errors, "RATE_DEPENDENCY_CYCLE", snapshot, *port);
      return;
    }
    resolution.state = ResolutionState::kResolving;

    const auto declaration = result.port_rates[static_cast<std::size_t>(port_id.value - 1U)];
    if (port->direction == PortDirection::kOutput) {
      switch (declaration.kind) {
        case PortRateKind::kFixedNominal:
          if (!declaration.declared_nominal_rate) {
            resolution.state = ResolutionState::kFailed;
            add_diagnostic(result.errors, "RATE_DECLARATION_INVALID", snapshot, *port);
            return;
          }
          resolution.rate = declaration.declared_nominal_rate;
          resolution.maximum = declaration.declared_nominal_rate;
          resolution.root = PortId{port_id.value};
          resolution.origin = ResolvedRateOrigin::kOutputNominal;
          resolution.state = ResolutionState::kResolved;
          return;
        case PortRateKind::kLifecycleNominal:
          resolution.maximum = declaration.declared_max_nominal_rate;
          resolution.root = PortId{port_id.value};
          resolution.state = ResolutionState::kPending;
          return;
        case PortRateKind::kFollowsInput: {
          if (!port->setup.follows_input || !declaration.follows_ratio) {
            resolution.state = ResolutionState::kFailed;
            add_diagnostic(result.errors, "RATE_FOLLOWS_INVALID", snapshot, *port);
            return;
          }
          resolve_port(*port->setup.follows_input);
          const GraphPortId dependency_id =
              resolve_interface_port(snapshot, *port->setup.follows_input);
          if (!dependency_id || dependency_id.value > resolutions.size()) {
            resolution.state = ResolutionState::kFailed;
            add_diagnostic(result.errors, "RATE_FOLLOWS_INVALID", snapshot, *port);
            return;
          }
          const auto& dependency = resolutions[static_cast<std::size_t>(dependency_id.value - 1U)];
          if (dependency.state == ResolutionState::kFailed) {
            resolution.state = ResolutionState::kFailed;
            return;
          }
          if (dependency.rate) {
            resolution.rate = rate_math::multiply(*dependency.rate, *declaration.follows_ratio);
            if (!resolution.rate) {
              resolution.state = ResolutionState::kFailed;
              add_diagnostic(result.errors, "RATE_ARITHMETIC_OVERFLOW", snapshot, *port);
              return;
            }
          }
          if (dependency.maximum) {
            resolution.maximum =
                rate_math::multiply(*dependency.maximum, *declaration.follows_ratio);
            if (!resolution.maximum) {
              resolution.state = ResolutionState::kFailed;
              add_diagnostic(result.errors, "RATE_ARITHMETIC_OVERFLOW", snapshot, *port);
              return;
            }
          }
          resolution.immediate = PortId{dependency_id.value};
          resolution.root = dependency.root;
          resolution.origin = resolution.rate ? ResolvedRateOrigin::kFollowsExpression
                                              : ResolvedRateOrigin::kUnresolved;
          resolution.state = resolution.rate ? ResolutionState::kResolved
                                             : (dependency.state == ResolutionState::kPending
                                                    ? ResolutionState::kPending
                                                    : ResolutionState::kAbsent);
          return;
        }
        case PortRateKind::kUnspecified:
        case PortRateKind::kExpectedInput:
          resolution.state = ResolutionState::kAbsent;
          return;
      }
    }

    const EdgeRecord* edge = incoming_edge(snapshot, port_id);
    const PortRecord* source{};
    if (edge != nullptr) {
      source = find_port(snapshot, resolve_interface_port(snapshot, edge->source));
    }
    if (source != nullptr) {
      resolve_port(source->id);
      const auto& upstream = resolutions[static_cast<std::size_t>(source->id.value - 1U)];
      if (upstream.state == ResolutionState::kFailed) {
        resolution.state = ResolutionState::kFailed;
        return;
      }
      resolution.maximum = upstream.maximum;
      if (upstream.rate) {
        resolution.rate = upstream.rate;
        resolution.immediate = PortId{source->id.value};
        resolution.root = upstream.root;
        resolution.origin = ResolvedRateOrigin::kConnectedOutput;
        resolution.state = ResolutionState::kResolved;
        if (declaration.declared_expected_rate) {
          resolution.expectation_comparison = *declaration.declared_expected_rate == *upstream.rate
                                                  ? RateComparison::kEqual
                                                  : RateComparison::kMismatch;
          if (resolution.expectation_comparison == RateComparison::kMismatch) {
            add_diagnostic(result.warnings, "RATE_EXPECTATION_MISMATCH", snapshot, *port);
          }
        }
        return;
      }
      if (upstream.state == ResolutionState::kPending) {
        resolution.immediate = PortId{source->id.value};
        resolution.root = upstream.root;
        resolution.state = ResolutionState::kPending;
        return;
      }
    }

    bool has_finite_horizon = has_compile_finite_horizon(
        temporal_triggers, window_triggers, predicate_triggers, port->owner);
    if (!has_finite_horizon) {
      const auto* trigger = trigger_for_owner(temporal_triggers, port->owner);
      if (trigger != nullptr && trigger->kind == TemporalRecordTriggerKind::kEach &&
          trigger->trigger_input && trigger->trigger_input.value() != port_id.value &&
          trigger->trigger_input.value() <= resolutions.size()) {
        resolve_port(GraphPortId{trigger->trigger_input.value()});
        has_finite_horizon =
            resolutions[static_cast<std::size_t>(trigger->trigger_input.value() - 1U)]
                .rate.has_value();
      }
    }
    if (declaration.declared_expected_rate && has_finite_horizon) {
      resolution.rate = declaration.declared_expected_rate;
      resolution.root = PortId{port_id.value};
      resolution.origin = ResolvedRateOrigin::kInputExpectationFallback;
      resolution.state = ResolutionState::kResolved;
      return;
    }
    resolution.state = ResolutionState::kAbsent;
  };

  for (const auto& port : snapshot.ports) {
    resolve_port(port.id);
  }

  result.resolved_rates.reserve(snapshot.ports.size());
  for (std::size_t index = 0U; index < snapshot.ports.size(); ++index) {
    const auto& resolution = resolutions[index];
    result.resolved_rates.push_back(ResolvedPortRate{
        .port = PortId{index + 1U},
        .resolved_rate = resolution.rate,
        .max_nominal_rate = resolution.maximum,
        .immediate_provenance_port = resolution.immediate,
        .root_provenance_port = resolution.root,
        .origin = resolution.origin,
        .expectation_comparison = resolution.expectation_comparison,
        .genuinely_absent = resolution.state == ResolutionState::kAbsent,
    });
  }

  for (const auto& resolved : result.resolved_rates) {
    if (!resolved.root_provenance_port || !resolved.port ||
        resolved.root_provenance_port->value() > snapshot.ports.size() ||
        resolved.port.value() > snapshot.ports.size()) {
      continue;
    }
    const auto& root = snapshot.ports[resolved.root_provenance_port->value() - 1U];
    const auto& dependent = snapshot.ports[resolved.port.value() - 1U];
    if (root.direction != PortDirection::kOutput ||
        root.setup.rate_kind != PortRateKind::kLifecycleNominal || root.owner == dependent.owner) {
      continue;
    }
    result.restart_dependencies.push_back(
        {.source_owner = root.owner, .dependent_owner = dependent.owner});
  }
  std::uint32_t next_floor_id = 1U;
  std::uint32_t next_floor_group = 1U;
  for (const auto& edge : snapshot.edges) {
    const std::size_t edge_error_begin = result.errors.size();
    const GraphPortId source_id = resolve_interface_port(snapshot, edge.source);
    const GraphPortId target_id = resolve_interface_port(snapshot, edge.target);
    const PortRecord* source = find_port(snapshot, source_id);
    const PortRecord* target = find_port(snapshot, target_id);
    if (source == nullptr || target == nullptr || source->direction != PortDirection::kOutput ||
        target->direction != PortDirection::kInput) {
      continue;
    }

    const auto [authored_origin, authored_depth] = explicit_depth(edge, *target);
    const auto owner_requires_bounded_evidence = [&](VertexId owner) {
      return owner.value != 0U && owner.value <= rate_evidence_requirements.size() &&
             rate_evidence_requirements[static_cast<std::size_t>(owner.value - 1U)] ==
                 RateBoundedEvidenceRequirement::kRequired;
    };
    const bool rate_aware_authority = !authored_depth ||
                                      source->setup.rate_kind != PortRateKind::kUnspecified ||
                                      target->setup.rate_kind != PortRateKind::kUnspecified;
    const bool source_bounded_evidence_missing =
        rate_aware_authority && owner_requires_bounded_evidence(source->owner);
    const bool target_bounded_evidence_missing =
        rate_aware_authority && owner_requires_bounded_evidence(target->owner);
    const bool bounded_evidence_missing =
        source_bounded_evidence_missing || target_bounded_evidence_missing;
    if (source_bounded_evidence_missing) {
      add_diagnostic(result.errors, "RATE_BOUNDED_EVIDENCE_REQUIRED", snapshot, *source);
    }
    if (target_bounded_evidence_missing) {
      add_diagnostic(result.errors, "RATE_BOUNDED_EVIDENCE_REQUIRED", snapshot, *target);
    }
    const HardFloorAlternativeGroupId arrival_group{next_floor_group++};
    const HardFloorAlternativeGroupId retention_group{next_floor_group++};
    const auto append_floor =
        [&](HardFloorOrigin origin, std::uint32_t value, HardFloorAlternativeGroupId group) {
          if (value == 0U) {
            return;
          }
          result.hard_floor_contributors.push_back(HardFloorContributorPlanRecord{
              .id = HardFloorContributorId{next_floor_id++},
              .edge = edge.id,
              .origin = origin,
              .required_slots = value,
              .scenario = HardFloorScenarioId{static_cast<std::uint32_t>(edge.id.value)},
              .alternative_group = group,
          });
        };
    append_floor(HardFloorOrigin::kMinimumQueue, 1U, arrival_group);
    append_floor(HardFloorOrigin::kProducerBurst, resolved_burst(*source), arrival_group);
    const PortId target_port{target_id.value};
    const bool has_input_selection = std::ranges::any_of(
        input_selections, [&](const auto& selection) { return selection.input == target_port; });
    if (has_input_selection) {
      if (const auto selector_floor =
              independent_selector_retention_floor(target_port, input_selections)) {
        append_floor(HardFloorOrigin::kSelectorRetention, *selector_floor, retention_group);
      }
    }
    const bool has_bracketed_window = std::ranges::any_of(
        bracketed_windows, [&](const auto& window) { return window.input == target_port; });
    if (has_bracketed_window) {
      const auto complete_window_floor =
          window_floor(target_port, bracketed_windows, temporal_triggers);
      if (!complete_window_floor) {
        add_diagnostic(result.errors, "RATE_CAPACITY_OVERFLOW", snapshot, *target);
      } else {
        append_floor(HardFloorOrigin::kWindowRetention, *complete_window_floor, retention_group);
      }
    }
    const auto composed_hard_floor = compose_hard_floor(result.hard_floor_contributors, edge.id);
    const std::uint32_t hard_floor = composed_hard_floor.value_or(1U);
    if (!composed_hard_floor) {
      add_diagnostic(result.errors, "RATE_CAPACITY_OVERFLOW", snapshot, *target);
    }

    const QueueSizingRecipeId recipe_id{static_cast<std::uint32_t>(edge.id.value)};
    const auto& producer_rate =
        result.resolved_rates[static_cast<std::size_t>(source_id.value - 1U)];
    const auto& consumer_rate =
        result.resolved_rates[static_cast<std::size_t>(target_id.value - 1U)];
    const auto& sizing_rate =
        !producer_rate.resolved_rate &&
                consumer_rate.origin == ResolvedRateOrigin::kInputExpectationFallback
            ? consumer_rate
            : producer_rate;
    const auto horizon_recipe =
        horizon_recipe_for_edge(*target, temporal_triggers, window_triggers, predicate_triggers);
    const auto horizon = horizon_recipe
                             ? evaluate_horizon_recipe(*horizon_recipe, result.resolved_rates)
                             : std::nullopt;
    if (horizon_recipe && horizon_recipe->inverse_rate_port && *horizon_recipe->inverse_rate_port) {
      const std::size_t driver_index =
          static_cast<std::size_t>(horizon_recipe->inverse_rate_port->value() - 1U);
      if (driver_index >= result.resolved_rates.size()) {
        add_diagnostic(result.errors, "RATE_DECLARATION_INVALID", snapshot, *target);
      } else {
        const auto& driver = result.resolved_rates[driver_index];
        if (driver.resolved_rate && !horizon) {
          add_diagnostic(result.errors, "RATE_ARITHMETIC_OVERFLOW", snapshot, *target);
        }
        if (driver.root_provenance_port &&
            driver.root_provenance_port->value() <= snapshot.ports.size()) {
          const auto& root = snapshot.ports[driver.root_provenance_port->value() - 1U];
          if (root.direction == PortDirection::kOutput &&
              root.setup.rate_kind == PortRateKind::kLifecycleNominal) {
            for (const VertexId dependent_owner : {source->owner, target->owner}) {
              if (root.owner != dependent_owner) {
                result.restart_dependencies.push_back(
                    {.source_owner = root.owner, .dependent_owner = dependent_owner});
              }
            }
          }
        }
      }
    }
    std::optional<std::uint32_t> arrivals;
    std::optional<std::uint32_t> recommendation;
    std::optional<std::uint32_t> ceiling;
    if (sizing_rate.resolved_rate && horizon) {
      arrivals = rate_math::ceil_arrivals(*sizing_rate.resolved_rate, *horizon);
      recommendation =
          arrivals ? rate_math::apply_headroom(*arrivals, headroom_ratio) : std::nullopt;
      if (!arrivals || !recommendation) {
        add_diagnostic(result.errors, "RATE_CAPACITY_OVERFLOW", snapshot, *target);
      }
    }
    const bool symbolic_producer_interarrival =
        horizon_recipe &&
        horizon_recipe->origin == AccumulationHorizonOrigin::kProducerInterarrival &&
        horizon_recipe->inverse_rate_port == std::optional<PortId>{target_port} &&
        consumer_rate.immediate_provenance_port == std::optional<PortId>{producer_rate.port} &&
        consumer_rate.root_provenance_port == producer_rate.root_provenance_port;
    if (producer_rate.max_nominal_rate && (horizon || symbolic_producer_interarrival)) {
      const auto maximum_arrivals =
          symbolic_producer_interarrival
              ? std::optional<std::uint32_t>{1U}
              : rate_math::ceil_arrivals(*producer_rate.max_nominal_rate, *horizon);
      ceiling = maximum_arrivals ? rate_math::apply_headroom(*maximum_arrivals, headroom_ratio)
                                 : std::nullopt;
      if (!maximum_arrivals || !ceiling) {
        add_diagnostic(result.errors, "RATE_CAPACITY_OVERFLOW", snapshot, *target);
      } else {
        ceiling = std::max(*ceiling, hard_floor);
      }
    }

    if (horizon_recipe) {
      result.accumulation_horizons.push_back(AccumulationHorizonPlanRecord{
          .id =
              HorizonRecordId{static_cast<std::uint32_t>(result.accumulation_horizons.size() + 1U)},
          .edge = edge.id,
          .horizon = horizon_recipe->horizon,
          .inverse_rate_port = horizon_recipe->inverse_rate_port,
          .origin = horizon_recipe->origin,
          .source_record = horizon_recipe->source_record,
      });
    }

    QueueDepthOrigin origin = authored_origin;
    std::uint32_t effective = hard_floor;
    if (authored_depth) {
      effective = *authored_depth;
      if (effective < hard_floor && rate_aware_authority) {
        add_diagnostic(result.errors, "RATE_QUEUE_DEPTH_BELOW_HARD_FLOOR", snapshot, *target);
      } else if (recommendation && effective < *recommendation) {
        add_diagnostic(result.warnings, "RATE_QUEUE_DEPTH_BELOW_RECOMMENDATION", snapshot, *target);
      }
    } else if (recommendation) {
      origin = QueueDepthOrigin::kRateAndContractDerived;
      effective = std::max(hard_floor, *recommendation);
    } else if (hard_floor > 1U) {
      origin = QueueDepthOrigin::kContractDerived;
    } else {
      origin = QueueDepthOrigin::kFrameworkFallback;
    }

    StabilityClass stability = StabilityClass::kUnresolvedBoundedEvidence;
    if (!bounded_evidence_missing && sizing_rate.resolved_rate && horizon) {
      const QueuePolicy policy =
          edge.graph_override.queue_policy.value_or(edge.add_flow_options.queue_policy);
      if (const auto capacity = advancement_capacity(target_port, input_selections, effective);
          arrivals && capacity) {
        if (*arrivals <= *capacity) {
          stability = StabilityClass::kStableNominalEstimate;
        } else {
          stability = policy == QueuePolicy::kDropOldest
                          ? StabilityClass::kPredictedReplacementOrDrop
                          : StabilityClass::kPredictedSustainedBlocking;
        }
      }
    }
    if (stability == StabilityClass::kPredictedSustainedBlocking) {
      add_diagnostic(result.warnings, "RATE_SUSTAINED_BACKPRESSURE", snapshot, *target);
    } else if (stability == StabilityClass::kPredictedReplacementOrDrop) {
      add_diagnostic(result.warnings, "RATE_PREDICTED_REPLACEMENT", snapshot, *target);
    }

    std::optional<LifecycleStage> latest_binding_stage;
    const auto merge_binding_stage = [&](const ResolvedPortRate& rate) {
      if (!rate.root_provenance_port || !*rate.root_provenance_port) {
        return;
      }
      const auto root_index = static_cast<std::size_t>(rate.root_provenance_port->value() - 1U);
      if (root_index < result.port_rates.size() &&
          result.port_rates[root_index].kind == PortRateKind::kLifecycleNominal) {
        const auto stage = result.port_rates[root_index].lifecycle_binding_stage;
        if (stage &&
            (!latest_binding_stage || static_cast<std::uint8_t>(*stage) >
                                          static_cast<std::uint8_t>(*latest_binding_stage))) {
          latest_binding_stage = stage;
        }
      }
    };
    merge_binding_stage(producer_rate);
    if (horizon_recipe && horizon_recipe->inverse_rate_port && *horizon_recipe->inverse_rate_port &&
        horizon_recipe->inverse_rate_port->value() <= result.resolved_rates.size()) {
      merge_binding_stage(result.resolved_rates[static_cast<std::size_t>(
          horizon_recipe->inverse_rate_port->value() - 1U)]);
    }
    const bool lifecycle_candidate = latest_binding_stage && !authored_depth && !ceiling &&
                                     static_cast<std::uint8_t>(*latest_binding_stage) <=
                                         static_cast<std::uint8_t>(LifecycleStage::kDiscover);
    const bool late_binding_without_preallocation =
        latest_binding_stage && !authored_depth && !ceiling &&
        static_cast<std::uint8_t>(*latest_binding_stage) >
            static_cast<std::uint8_t>(LifecycleStage::kDiscover);
    if (late_binding_without_preallocation) {
      add_diagnostic(result.errors, "QUEUE_PREALLOCATION_BOUND_REQUIRED", snapshot, *target);
    }
    const bool edge_has_new_error = result.errors.size() != edge_error_begin;
    const bool recipe_accepted =
        !headroom_invalid && !bounded_evidence_missing && !edge_has_new_error;
    DiagnosticId recipe_reason{};
    if (headroom_invalid) {
      recipe_reason = diagnostic_id("RATE_HEADROOM_INVALID");
    } else if (bounded_evidence_missing && !edge_has_new_error) {
      recipe_reason = diagnostic_id("RATE_BOUNDED_EVIDENCE_REQUIRED");
    } else if (edge_has_new_error) {
      recipe_reason = diagnostic_id(result.errors[edge_error_begin].token);
    }
    result.queue_sizing_recipes.push_back(QueueSizingRecipePlanRecord{
        .id = recipe_id,
        .edge = edge.id,
        .origin = origin,
        .explicit_depth = authored_depth,
        .headroom_ratio = headroom_ratio,
        .latest_binding_stage = latest_binding_stage,
        .geometry_mode = lifecycle_candidate ? AllocationGeometryMode::kLifecycleCandidate
                                             : AllocationGeometryMode::kCompileConcrete,
        .disposition = recipe_accepted ? PlanDisposition::kAccepted : PlanDisposition::kRejected,
        .reason = recipe_reason,
    });
    result.queue_sizings.push_back(ResolvedQueueSizing{
        .edge = edge.id,
        .recipe = recipe_id,
        .origin = origin,
        .explicit_depth = authored_depth,
        .hard_floor = hard_floor,
        .nominal_arrivals = arrivals,
        .rate_recommendation = recommendation,
        .admitted_depth_ceiling = ceiling,
        .stability = stability,
        .effective_logical_depth = effective,
    });
  }

  std::ranges::sort(result.restart_dependencies, [](const auto& left, const auto& right) {
    return std::pair{left.source_owner.value, left.dependent_owner.value} <
           std::pair{right.source_owner.value, right.dependent_owner.value};
  });
  result.restart_dependencies.erase(
      std::unique(result.restart_dependencies.begin(), result.restart_dependencies.end()),
      result.restart_dependencies.end());

  // Lifecycle-selected rates are candidate transactions, not merely numeric storage hints. The
  // frozen dependency relation already contains every connected, follows-derived, and
  // inverse-driver owner that must participate in barrier/restart publication. There is no
  // distributed candidate exchange, so reject a relation that crosses the frozen process cover.
  if (!process_group_by_vertex.empty()) {
    for (const auto& dependency : result.restart_dependencies) {
      if (!dependency.source_owner || !dependency.dependent_owner ||
          dependency.source_owner.value > process_group_by_vertex.size() ||
          dependency.dependent_owner.value > process_group_by_vertex.size()) {
        continue;  // Process-plan lowering owns malformed/incomplete placement diagnostics.
      }
      const std::uint32_t source_group =
          process_group_by_vertex[static_cast<std::size_t>(dependency.source_owner.value - 1U)];
      const std::uint32_t dependent_group =
          process_group_by_vertex[static_cast<std::size_t>(dependency.dependent_owner.value - 1U)];
      if (source_group == 0U || dependent_group == 0U || source_group == dependent_group) {
        continue;
      }
      static_cast<void>(append_unique_plan_lowering_diagnostic(
          result.errors,
          PlanLoweringDiagnostic{
              .token = "RATE_LIFECYCLE_CROSS_PROCESS_UNSUPPORTED",
              .source = DiagnosticSource{.graph_generation = snapshot.generation,
                                         .vertex = dependency.source_owner},
              .required = "lifecycle-derived rate dependency within one process group",
              .provided = "source_group=" + std::to_string(source_group) +
                          ",dependent_group=" + std::to_string(dependent_group),
          }));
    }
  }

  canonicalize_plan_lowering_diagnostics(result.errors);
  canonicalize_plan_lowering_diagnostics(result.warnings);
  return result;
}

}  // namespace holoscan::detail
