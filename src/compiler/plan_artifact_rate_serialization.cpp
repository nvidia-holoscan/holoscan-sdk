/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/plan_artifact_rate_serialization.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "compiler/execution_plan_contracts.hpp"
#include "compiler/plan_artifact_serialization_internal.hpp"
#include "core/canonical_text_encoding.hpp"
#include "core/tensor_allocation_record_validation.hpp"

namespace holoscan::detail::plan_artifact_rate_serialization_internal {
namespace {

using plan_artifact_serialization_internal::append_array;
using plan_artifact_serialization_internal::append_bool;
using plan_artifact_serialization_internal::append_number;
using plan_artifact_serialization_internal::append_optional_profile;
using plan_artifact_serialization_internal::append_optional_tensor_layout;
using plan_artifact_serialization_internal::append_tensor_allocation_bounds;
using plan_artifact_serialization_internal::append_tensor_axes;

void append_optional_rate(std::string& output, const std::optional<Rate>& rate) {
  if (!rate) {
    output += "null";
    return;
  }
  output += "{\"denominator_hz\":";
  append_number(output, rate->denominator_hz);
  output += ",\"numerator_hz\":";
  append_number(output, rate->numerator_hz);
  output.push_back('}');
}

void append_optional_rate_ratio(std::string& output, const std::optional<RateRatio>& ratio) {
  if (!ratio) {
    output += "null";
    return;
  }
  output += "{\"denominator\":";
  append_number(output, ratio->denominator);
  output += ",\"numerator\":";
  append_number(output, ratio->numerator);
  output.push_back('}');
}

void append_rate_ratio(std::string& output, RateRatio ratio) {
  append_optional_rate_ratio(output, std::optional<RateRatio>{ratio});
}

template <typename T>
void append_optional_number(std::string& output, const std::optional<T>& value) {
  if (value) {
    append_number(output, *value);
  } else {
    output += "null";
  }
}

[[nodiscard]] bool canonical_rate(Rate rate) noexcept {
  return rate.numerator_hz != 0U && rate.denominator_hz != 0U &&
         std::gcd(rate.numerator_hz, rate.denominator_hz) == 1U;
}

[[nodiscard]] bool canonical_rate_ratio(RateRatio ratio) noexcept {
  return ratio.numerator != 0U && ratio.denominator != 0U &&
         std::gcd(ratio.numerator, ratio.denominator) == 1U;
}

[[nodiscard]] bool valid_tensor_provider_template(
    const TensorProviderTemplatePlanRecord& tensor_template) noexcept {
  const bool memory_and_device_valid = [&]() noexcept {
    switch (tensor_template.memory_kind) {
      case MemoryKind::kHost:
        return !tensor_template.device.valid();
      case MemoryKind::kPinnedHost:
        return true;
      case MemoryKind::kCudaDevice:
      case MemoryKind::kCudaManaged:
        return tensor_template.device.valid();
      case MemoryKind::kUnknown:
        return false;
    }
    return false;
  }();

  return !tensor_template.allocator_root_id.empty() &&
         static_cast<bool>(tensor_template.allocator_process_group) &&
         tensor_template.bounds.max_rank != 0U && tensor_template.bounds.max_byte_span != 0U &&
         tensor_template.layout && tensor_template.layout->memory_kind &&
         *tensor_template.layout->memory_kind == tensor_template.memory_kind &&
         (!tensor_template.layout->dtype || valid_tensor_dtype(*tensor_template.layout->dtype)) &&
         (!tensor_template.layout->rank ||
          *tensor_template.layout->rank <= tensor_template.bounds.max_rank) &&
         validate_tensor_allocation_axes(
             tensor_template.bounds, tensor_template.layout, tensor_template.axes) &&
         memory_and_device_valid && tensor_template.max_emits_per_compute != 0U &&
         tensor_template.max_outstanding_loans == tensor_template.max_emits_per_compute &&
         (!tensor_template.effective_profile ||
          static_cast<std::uint8_t>(*tensor_template.effective_profile) <=
              static_cast<std::uint8_t>(Profile::kResearch)) &&
         tensor_template.fallback_rejected;
}

// The private artifact validator intentionally checks the complete cross-record rate identity.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool valid_rate_plan_records(const ExecutionPlanState& state) noexcept {
  // Rate recipes are graph-semantic records and remain valid evidence when a later partition or
  // physical phase rejects before publishing accepted payload edges. Compiler-produced states
  // therefore validate them against the canonical Graph. Focused artifact fixtures that predate a
  // GraphView retain the accepted physical-edge fallback.
  const auto graph_ports = state.graph.ports();
  const auto graph_edges = state.graph.edges();
  const bool has_graph_port_authority = !graph_ports.empty();
  const bool has_graph_edge_authority = !graph_edges.empty();
  const std::size_t canonical_port_count =
      has_graph_port_authority ? graph_ports.size() : state.ports.size();
  const std::size_t canonical_edge_count =
      has_graph_edge_authority ? graph_edges.size() : state.payload_edges.size();
  const auto canonical_port_id = [&](std::size_t index) noexcept {
    return has_graph_port_authority ? graph_ports[index].id.value : state.ports[index].id.value();
  };
  const auto canonical_port_owner = [&](std::size_t index) noexcept {
    return has_graph_port_authority ? graph_ports[index].owner : state.ports[index].owner;
  };
  const auto canonical_port_direction = [&](std::size_t index) noexcept {
    return has_graph_port_authority ? graph_ports[index].direction : state.ports[index].direction;
  };
  const auto canonical_port_index = [&](PortId port) noexcept -> std::optional<std::size_t> {
    if (!port || port.value() > canonical_port_count) {
      return std::nullopt;
    }
    const std::size_t index = static_cast<std::size_t>(port.value() - 1U);
    return canonical_port_id(index) == port.value() ? std::optional<std::size_t>{index}
                                                    : std::nullopt;
  };
  const auto resolve_graph_interface_port = [&](GraphPortId port) noexcept {
    const auto bindings = state.graph.interface_bindings();
    for (std::size_t depth = 0U; depth <= bindings.size(); ++depth) {
      const auto found = std::lower_bound(
          bindings.begin(), bindings.end(), port.value, [](const auto& binding, auto value) {
            return binding.public_port.value < value;
          });
      if (found == bindings.end() || found->public_port != port) {
        return port;
      }
      port = found->internal_port;
    }
    return GraphPortId{};
  };
  const auto canonical_edge_occurrences = [&](EdgeId edge) noexcept {
    return has_graph_edge_authority
               ? std::ranges::count(graph_edges, edge, &EdgeRecord::id)
               : std::ranges::count(state.payload_edges, edge, &PayloadEdgePlanRecord::edge);
  };
  const auto canonical_edge_target = [&](EdgeId edge) noexcept -> std::optional<PortId> {
    if (has_graph_edge_authority) {
      const auto found = std::ranges::find(graph_edges, edge, &EdgeRecord::id);
      if (found == graph_edges.end()) {
        return std::nullopt;
      }
      const GraphPortId target = resolve_graph_interface_port(found->target);
      return target ? std::optional<PortId>{PortId{target.value}} : std::nullopt;
    }
    const auto found = std::ranges::find(state.payload_edges, edge, &PayloadEdgePlanRecord::edge);
    return found != state.payload_edges.end() && found->target
               ? std::optional<PortId>{found->target}
               : std::nullopt;
  };
  if (state.port_rates.size() != canonical_port_count ||
      state.queue_sizing_recipes.size() != canonical_edge_count) {
    return false;
  }
  for (std::size_t index = 0U; index < state.port_rates.size(); ++index) {
    const auto& record = state.port_rates[index];
    const PortDirection port_direction = canonical_port_direction(index);
    const VertexId port_owner = canonical_port_owner(index);
    if (!record.port || record.port.value() != canonical_port_id(index) ||
        record.disposition != PlanDisposition::kAccepted || record.reason ||
        static_cast<std::uint8_t>(record.kind) >
            static_cast<std::uint8_t>(PortRateKind::kLifecycleNominal) ||
        (record.declared_nominal_rate && !canonical_rate(*record.declared_nominal_rate)) ||
        (record.declared_expected_rate && !canonical_rate(*record.declared_expected_rate)) ||
        (record.declared_max_nominal_rate && !canonical_rate(*record.declared_max_nominal_rate)) ||
        (record.follows_ratio && !canonical_rate_ratio(*record.follows_ratio)) ||
        (record.follows_input && !*record.follows_input) ||
        (record.lifecycle_binding_stage &&
         !is_lifecycle_forward_stage(*record.lifecycle_binding_stage))) {
      return false;
    }
    const bool no_fixed = !record.declared_nominal_rate;
    const bool no_expected = !record.declared_expected_rate;
    const bool no_follows = !record.follows_input && !record.follows_ratio;
    const bool no_lifecycle = !record.lifecycle_binding_stage && !record.declared_max_nominal_rate;
    switch (record.kind) {
      case PortRateKind::kUnspecified:
        if (!no_fixed || !no_expected || !no_follows || !no_lifecycle) {
          return false;
        }
        break;
      case PortRateKind::kFixedNominal:
        if (port_direction != PortDirection::kOutput || no_fixed || !no_expected || !no_follows ||
            !no_lifecycle) {
          return false;
        }
        break;
      case PortRateKind::kExpectedInput:
        if (port_direction != PortDirection::kInput || !no_fixed || no_expected || !no_follows ||
            !no_lifecycle) {
          return false;
        }
        break;
      case PortRateKind::kFollowsInput: {
        const auto followed_index =
            record.follows_input ? canonical_port_index(*record.follows_input) : std::nullopt;
        if (port_direction != PortDirection::kOutput || !no_fixed || !no_expected ||
            !record.follows_input || !record.follows_ratio || !no_lifecycle || !followed_index) {
          return false;
        }
        if (canonical_port_direction(*followed_index) != PortDirection::kInput ||
            canonical_port_owner(*followed_index) != port_owner) {
          return false;
        }
        break;
      }
      case PortRateKind::kLifecycleNominal:
        if (port_direction != PortDirection::kOutput || !no_fixed || !no_expected || !no_follows ||
            !record.lifecycle_binding_stage ||
            (*record.lifecycle_binding_stage != LifecycleStage::kConfigure &&
             *record.lifecycle_binding_stage != LifecycleStage::kDiscover &&
             *record.lifecycle_binding_stage != LifecycleStage::kWarmUp &&
             *record.lifecycle_binding_stage != LifecycleStage::kArm)) {
          return false;
        }
        break;
    }
  }

  for (std::size_t index = 0U; index < state.accumulation_horizons.size(); ++index) {
    const auto& record = state.accumulation_horizons[index];
    const bool fixed_horizon = record.horizon.has_value();
    const bool inverse_rate_horizon = record.inverse_rate_port.has_value();
    if (record.id.value != index + 1U || !record.edge ||
        canonical_edge_occurrences(record.edge) != 1 ||
        std::ranges::count(
            state.accumulation_horizons, record.edge, &AccumulationHorizonPlanRecord::edge) != 1 ||
        static_cast<std::uint8_t>(record.origin) >
            static_cast<std::uint8_t>(AccumulationHorizonOrigin::kExternalCadence) ||
        fixed_horizon == inverse_rate_horizon || record.source_record == 0U ||
        (fixed_horizon &&
         (record.horizon->numerator_ns == 0U || record.horizon->denominator == 0U ||
          std::gcd(record.horizon->numerator_ns, record.horizon->denominator) != 1U))) {
      return false;
    }
    const auto edge_target = canonical_edge_target(record.edge);
    const auto target_index = edge_target ? canonical_port_index(*edge_target) : std::nullopt;
    if (!target_index) {
      return false;
    }
    const VertexId target_owner = canonical_port_owner(*target_index);
    const TemporalTriggerPlanRecord* target_trigger = nullptr;
    for (const auto& trigger : state.temporal_triggers) {
      if (trigger.owner.id != target_owner.value) {
        continue;
      }
      if (target_trigger != nullptr) {
        return false;
      }
      target_trigger = &trigger;
    }
    if (target_trigger == nullptr ||
        target_trigger->owner.plan_generation != state.plan_generation ||
        target_trigger->owner.run_generation != 0U ||
        record.source_record != static_cast<std::uint32_t>(target_trigger->owner.id)) {
      return false;
    }
    if (inverse_rate_horizon) {
      const auto driver_index = canonical_port_index(*record.inverse_rate_port);
      if (!driver_index) {
        return false;
      }
      const bool producer_interarrival =
          record.inverse_rate_port.value_or(PortId{}) == edge_target.value_or(PortId{});
      if (canonical_port_direction(*driver_index) != PortDirection::kInput ||
          canonical_port_owner(*driver_index) != target_owner ||
          target_trigger->kind != TemporalRecordTriggerKind::kEach ||
          target_trigger->trigger_input != record.inverse_rate_port ||
          (producer_interarrival &&
           record.origin != AccumulationHorizonOrigin::kProducerInterarrival) ||
          (!producer_interarrival &&
           record.origin != AccumulationHorizonOrigin::kDriverInterarrival)) {
        return false;
      }
    } else {
      const bool clock_horizon = target_trigger->kind == TemporalRecordTriggerKind::kClock &&
                                 record.origin == AccumulationHorizonOrigin::kClockPeriod;
      const bool window_horizon = target_trigger->kind == TemporalRecordTriggerKind::kWindow &&
                                  record.origin == AccumulationHorizonOrigin::kTemporalWindowBound;
      const bool predicate_clock_horizon =
          target_trigger->kind == TemporalRecordTriggerKind::kPredicate &&
          record.origin == AccumulationHorizonOrigin::kClockPeriod;
      if (!clock_horizon && !window_horizon && !predicate_clock_horizon) {
        return false;
      }
      if (clock_horizon &&
          (target_trigger->trigger_clock.value == 0U || target_trigger->period_ns <= 0 ||
           record.horizon->denominator != 1U ||
           record.horizon->numerator_ns != static_cast<std::uint64_t>(target_trigger->period_ns))) {
        return false;
      }
      if (window_horizon) {
        const WindowTriggerPlanRecord* target_window = nullptr;
        for (const auto& window : state.window_triggers) {
          if (window.owner.id != target_owner.value) {
            continue;
          }
          if (target_window != nullptr) {
            return false;
          }
          target_window = &window;
        }
        if (target_window == nullptr || target_window->owner != target_trigger->owner ||
            target_trigger->trigger_input.value() == 0U || target_window->anchor.value() == 0U ||
            target_window->anchor != target_trigger->trigger_input ||
            target_trigger->trigger_clock.value == 0U ||
            target_window->wait_clock != target_trigger->trigger_clock ||
            target_window->max_wait_ns <= 0 || record.horizon->denominator != 1U ||
            record.horizon->numerator_ns !=
                static_cast<std::uint64_t>(target_window->max_wait_ns)) {
          return false;
        }
      }
      if (predicate_clock_horizon) {
        const PredicateTriggerPlanRecord* target_predicate = nullptr;
        for (const auto& predicate : state.predicate_triggers) {
          if (predicate.owner.id != target_owner.value) {
            continue;
          }
          if (target_predicate != nullptr) {
            return false;
          }
          target_predicate = &predicate;
        }
        if (target_predicate == nullptr || target_predicate->owner != target_trigger->owner ||
            target_predicate->source != PredicateEvaluationSourceKind::kClock ||
            target_trigger->trigger_clock.value == 0U ||
            target_predicate->clock != target_trigger->trigger_clock ||
            target_predicate->period_ns <= 0 ||
            target_predicate->period_ns != target_trigger->period_ns ||
            record.horizon->denominator != 1U ||
            record.horizon->numerator_ns !=
                static_cast<std::uint64_t>(target_predicate->period_ns)) {
          return false;
        }
      }
    }
  }

  for (std::size_t index = 0U; index < state.hard_floor_contributors.size(); ++index) {
    const auto& record = state.hard_floor_contributors[index];
    if (record.id.value != index + 1U || !record.edge ||
        canonical_edge_occurrences(record.edge) != 1 ||
        static_cast<std::uint8_t>(record.origin) >
            static_cast<std::uint8_t>(HardFloorOrigin::kProviderMinimum) ||
        record.required_slots == 0U || record.scenario.value == 0U ||
        record.alternative_group.value == 0U) {
      return false;
    }
  }

  for (std::size_t index = 0U; index < state.queue_sizing_recipes.size(); ++index) {
    const auto& record = state.queue_sizing_recipes[index];
    const bool accepted = record.disposition == PlanDisposition::kAccepted && !record.reason;
    const bool rejected =
        record.disposition == PlanDisposition::kRejected && static_cast<bool>(record.reason);
    if (record.id.value != index + 1U || !record.edge ||
        canonical_edge_occurrences(record.edge) != 1 ||
        std::ranges::count(
            state.queue_sizing_recipes, record.edge, &QueueSizingRecipePlanRecord::edge) != 1 ||
        std::ranges::none_of(
            state.hard_floor_contributors,
            [&](const auto& contributor) { return contributor.edge == record.edge; }) ||
        static_cast<std::uint8_t>(record.origin) >
            static_cast<std::uint8_t>(QueueDepthOrigin::kFrameworkFallback) ||
        !canonical_rate_ratio(record.headroom_ratio) ||
        record.headroom_ratio.numerator < record.headroom_ratio.denominator ||
        (record.explicit_depth && *record.explicit_depth == 0U) ||
        (record.latest_binding_stage &&
         !is_lifecycle_forward_stage(*record.latest_binding_stage)) ||
        static_cast<std::uint8_t>(record.geometry_mode) >
            static_cast<std::uint8_t>(AllocationGeometryMode::kLifecycleCandidate) ||
        (record.geometry_mode == AllocationGeometryMode::kLifecycleCandidate &&
         (record.explicit_depth || !record.latest_binding_stage ||
          static_cast<std::uint8_t>(*record.latest_binding_stage) >
              static_cast<std::uint8_t>(LifecycleStage::kDiscover))) ||
        static_cast<std::uint8_t>(record.reuse_policy) >
            static_cast<std::uint8_t>(ProviderReusePolicy::kFreshOnly) ||
        (!accepted && !rejected)) {
      return false;
    }
    const auto edge_target = canonical_edge_target(record.edge);
    if (!edge_target) {
      return false;
    }
    for (const auto& selection : state.input_selections) {
      if (selection.input != *edge_target || selection.selector != InputSelectorKind::kDrainAll ||
          selection.max_selected_samples_origin !=
              SelectedSampleBoundOrigin::kResolvedRetainedCapacity) {
        continue;
      }
      if (std::ranges::any_of(state.hard_floor_contributors, [&](const auto& contributor) {
            return contributor.edge == record.edge &&
                   contributor.origin == HardFloorOrigin::kSelectorRetention;
          })) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool canonical_pool_claim_reasons(
    std::span<const TensorSlotClaimReason> reasons) noexcept {
  if (reasons.empty()) {
    return false;
  }
  for (std::size_t index = 0U; index < reasons.size(); ++index) {
    if (static_cast<std::uint8_t>(reasons[index]) >
            static_cast<std::uint8_t>(TensorSlotClaimReason::kRetirementService) ||
        (index != 0U && static_cast<std::uint8_t>(reasons[index - 1U]) >=
                            static_cast<std::uint8_t>(reasons[index]))) {
      return false;
    }
  }
  return true;
}

// The private artifact validator deliberately keeps provider/dependency range joins atomic.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool valid_pool_capacity_plan_records(const ExecutionPlanState& state) noexcept {
  if ((!state.tensor_allocations.empty() || !state.tensor_field_allocations.empty()) &&
      (state.pool_capacity_dependencies.empty() || state.provider_allocation_plans.empty())) {
    return false;
  }
  for (std::size_t index = 0U; index < state.pool_capacity_contributors.size(); ++index) {
    const auto& contributor = state.pool_capacity_contributors[index];
    if (contributor.pool_id.empty() || contributor.contributor_id.empty() ||
        contributor.maximum_distinct_slots == 0U ||
        !canonical_pool_claim_reasons(contributor.reasons) ||
        std::ranges::count_if(state.pool_capacity_contributors, [&](const auto& candidate) {
          return candidate.pool_id == contributor.pool_id &&
                 candidate.contributor_id == contributor.contributor_id;
        }) != 1) {
      return false;
    }
  }

  std::vector<bool> dependency_covered(state.pool_capacity_dependencies.size(), false);
  for (std::size_t index = 0U; index < state.pool_capacity_dependencies.size(); ++index) {
    const auto& dependency = state.pool_capacity_dependencies[index];
    if (dependency.id.value != index + 1U || dependency.pool_id.empty() ||
        dependency.scenario.value == 0U || dependency.alternative_group.value == 0U ||
        dependency.mapping_multiplicity == 0U ||
        static_cast<std::uint8_t>(dependency.origin) >
            static_cast<std::uint8_t>(PoolClaimOrigin::kProviderSpecific) ||
        static_cast<std::uint8_t>(dependency.source) >
            static_cast<std::uint8_t>(PoolCapacityDependencySource::kPredecessorContributor)) {
      return false;
    }
    if (dependency.source == PoolCapacityDependencySource::kQueueSizingRecipe) {
      if (!dependency.queue_sizing_recipe || !dependency.predecessor_contributor_id.empty() ||
          !canonical_pool_claim_reasons(dependency.reasons) ||
          std::ranges::count(state.queue_sizing_recipes,
                             dependency.queue_sizing_recipe,
                             &QueueSizingRecipePlanRecord::id) != 1) {
        return false;
      }
    } else if (dependency.queue_sizing_recipe || dependency.predecessor_contributor_id.empty() ||
               !dependency.reasons.empty()) {
      return false;
    } else {
      const auto predecessor = std::ranges::find(state.pool_capacity_contributors,
                                                 dependency.predecessor_contributor_id,
                                                 &PoolCapacityContributorRecord::contributor_id);
      if (predecessor == state.pool_capacity_contributors.end() ||
          predecessor->pool_id != dependency.pool_id ||
          std::ranges::count(state.pool_capacity_contributors,
                             dependency.predecessor_contributor_id,
                             &PoolCapacityContributorRecord::contributor_id) != 1) {
        return false;
      }
    }
  }

  for (std::size_t index = 0U; index < state.provider_allocation_plans.size(); ++index) {
    const auto& provider = state.provider_allocation_plans[index];
    if (provider.id.value != index + 1U || provider.pool_id.empty() ||
        !provider.first_sizing_dependency || provider.sizing_dependency_count == 0U ||
        static_cast<std::uint8_t>(provider.allocation_kind) >
            static_cast<std::uint8_t>(ProviderAllocationKind::kTensorInterface) ||
        static_cast<std::uint8_t>(provider.geometry_mode) >
            static_cast<std::uint8_t>(AllocationGeometryMode::kLifecycleCandidate) ||
        provider.complete_before != LifecycleStage::kAllocate ||
        provider.disposition != PlanDisposition::kAccepted || provider.reason) {
      return false;
    }
    const auto dependencies =
        provider_sizing_dependencies(state.pool_capacity_dependencies, provider);
    if (!dependencies) {
      return false;
    }
    bool lifecycle_candidate = false;
    const std::size_t first = static_cast<std::size_t>(provider.first_sizing_dependency.value - 1U);
    for (std::size_t offset = 0U; offset < dependencies->size(); ++offset) {
      const std::size_t dependency_index = first + offset;
      const auto& dependency = (*dependencies)[offset];
      if (dependency_covered[dependency_index]) {
        return false;
      }
      dependency_covered[dependency_index] = true;
      if (dependency.source == PoolCapacityDependencySource::kQueueSizingRecipe) {
        const auto recipe = std::ranges::find(state.queue_sizing_recipes,
                                              dependency.queue_sizing_recipe,
                                              &QueueSizingRecipePlanRecord::id);
        lifecycle_candidate =
            lifecycle_candidate ||
            (recipe != state.queue_sizing_recipes.end() &&
             recipe->geometry_mode == AllocationGeometryMode::kLifecycleCandidate);
      }
    }
    const bool direct_tensor_identity =
        provider.allocation_kind == ProviderAllocationKind::kTensorOutput && provider.output &&
        !provider.tensor_field_id && !provider.tensor_index && !provider.interface_topic;
    const bool tensor_field_identity =
        provider.allocation_kind == ProviderAllocationKind::kTensorFieldOutput && provider.output &&
        provider.tensor_field_id && provider.tensor_index && !provider.interface_topic;
    if (!direct_tensor_identity && !tensor_field_identity) {
      return false;
    }
    const std::size_t allocation_count =
        direct_tensor_identity
            ? std::ranges::count_if(state.tensor_allocations,
                                    [&](const auto& allocation) {
                                      return allocation.output == *provider.output &&
                                             allocation.pool_id == provider.pool_id;
                                    })
            : std::ranges::count_if(state.tensor_field_allocations, [&](const auto& allocation) {
                return allocation.output == *provider.output &&
                       allocation.field_id == *provider.tensor_field_id &&
                       allocation.tensor_index == *provider.tensor_index &&
                       allocation.pool_id == provider.pool_id;
              });
    if (provider.geometry_mode == AllocationGeometryMode::kCompileConcrete) {
      if (lifecycle_candidate || allocation_count != 1U || provider.tensor_template) {
        return false;
      }
    } else {
      if (!lifecycle_candidate || !direct_tensor_identity || allocation_count != 0U ||
          !provider.tensor_template || !valid_tensor_provider_template(*provider.tensor_template) ||
          provider.tensor_template->memory_kind != MemoryKind::kHost ||
          provider.tensor_template->device.valid() || !provider.tensor_template->layout ||
          !provider.tensor_template->layout->memory_kind ||
          *provider.tensor_template->layout->memory_kind != MemoryKind::kHost) {
        return false;
      }
      bool found_direct_retained_service{};
      for (const auto& service : state.process_local_channel_services) {
        if (service.source != *provider.output) {
          continue;
        }
        if (service.delivery_kind != ProcessLocalChannelDeliveryKind::kDirectRetained ||
            service.cross_process_transport || service.queue_policy != QueuePolicy::kBlocking) {
          return false;
        }
        found_direct_retained_service = true;
      }
      if (!found_direct_retained_service ||
          std::ranges::any_of(state.cross_process_channel_services, [&](const auto& service) {
            return service.source == *provider.output;
          })) {
        return false;
      }
    }
    if (std::ranges::count_if(state.provider_allocation_plans, [&](const auto& candidate) {
          return candidate.pool_id == provider.pool_id && candidate.output == provider.output &&
                 candidate.tensor_field_id == provider.tensor_field_id &&
                 candidate.tensor_index == provider.tensor_index &&
                 candidate.interface_topic == provider.interface_topic;
        }) != 1) {
      return false;
    }
    for (const auto& dependency : *dependencies) {
      if (dependency.source != PoolCapacityDependencySource::kQueueSizingRecipe) {
        continue;
      }
      const auto recipe = std::ranges::find(state.queue_sizing_recipes,
                                            dependency.queue_sizing_recipe,
                                            &QueueSizingRecipePlanRecord::id);
      if (recipe == state.queue_sizing_recipes.end() || !provider.output ||
          std::ranges::none_of(state.payload_edges, [&](const auto& edge) {
            return edge.edge == recipe->edge && edge.source == *provider.output;
          })) {
        return false;
      }
    }
  }
  return std::ranges::all_of(dependency_covered, [](bool covered) { return covered; });
}

}  // namespace

bool rate_plan_records_valid(const ExecutionPlanState& state) noexcept {
  return valid_rate_plan_records(state);
}

bool pool_capacity_plan_records_valid(const ExecutionPlanState& state) noexcept {
  return valid_pool_capacity_plan_records(state);
}

void append_rate_plan_sections(std::string& output, const ExecutionPlanState& state) {
  output += ",\"port_rates\":";
  append_array(output, state.port_rates, [](std::string& json, const auto& record) {
    json += "{\"declared_expected_rate\":";
    append_optional_rate(json, record.declared_expected_rate);
    json += ",\"declared_max_nominal_rate\":";
    append_optional_rate(json, record.declared_max_nominal_rate);
    json += ",\"declared_nominal_rate\":";
    append_optional_rate(json, record.declared_nominal_rate);
    json += ",\"disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.disposition));
    json += ",\"follows_input\":";
    if (record.follows_input) {
      append_number(json, record.follows_input->value());
    } else {
      json += "null";
    }
    json += ",\"follows_ratio\":";
    append_optional_rate_ratio(json, record.follows_ratio);
    json += ",\"kind\":";
    append_number(json, static_cast<std::uint32_t>(record.kind));
    json += ",\"lifecycle_binding_stage\":";
    if (record.lifecycle_binding_stage) {
      append_number(json, static_cast<std::uint32_t>(*record.lifecycle_binding_stage));
    } else {
      json += "null";
    }
    json += ",\"port\":";
    append_number(json, record.port.value());
    json += ",\"reason\":";
    append_number(json, record.reason.value);
    json.push_back('}');
  });
  output += ",\"accumulation_horizons\":";
  append_array(output, state.accumulation_horizons, [](std::string& json, const auto& record) {
    json += "{\"edge\":";
    append_number(json, record.edge.value);
    json += ",\"horizon\":";
    if (record.horizon) {
      json += "{\"denominator\":";
      append_number(json, record.horizon->denominator);
      json += ",\"numerator_ns\":";
      append_number(json, record.horizon->numerator_ns);
      json.push_back('}');
    } else {
      json += "null";
    }
    json += ",\"id\":";
    append_number(json, record.id.value);
    json += ",\"inverse_rate_port\":";
    if (record.inverse_rate_port) {
      append_number(json, record.inverse_rate_port->value());
    } else {
      json += "null";
    }
    json += ",\"origin\":";
    append_number(json, static_cast<std::uint32_t>(record.origin));
    json += ",\"source_record\":";
    append_number(json, record.source_record);
    json.push_back('}');
  });
  output += ",\"hard_floor_contributors\":";
  append_array(output, state.hard_floor_contributors, [](std::string& json, const auto& record) {
    json += "{\"alternative_group\":";
    append_number(json, record.alternative_group.value);
    json += ",\"edge\":";
    append_number(json, record.edge.value);
    json += ",\"id\":";
    append_number(json, record.id.value);
    json += ",\"origin\":";
    append_number(json, static_cast<std::uint32_t>(record.origin));
    json += ",\"required_slots\":";
    append_number(json, record.required_slots);
    json += ",\"scenario\":";
    append_number(json, record.scenario.value);
    json.push_back('}');
  });
  output += ",\"queue_sizing_recipes\":";
  append_array(output, state.queue_sizing_recipes, [](std::string& json, const auto& record) {
    json += "{\"disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.disposition));
    json += ",\"edge\":";
    append_number(json, record.edge.value);
    json += ",\"explicit_depth\":";
    append_optional_number(json, record.explicit_depth);
    json += ",\"geometry_mode\":";
    append_number(json, static_cast<std::uint32_t>(record.geometry_mode));
    json += ",\"headroom_ratio\":";
    append_rate_ratio(json, record.headroom_ratio);
    json += ",\"id\":";
    append_number(json, record.id.value);
    json += ",\"latest_binding_stage\":";
    if (record.latest_binding_stage) {
      append_number(json, static_cast<std::uint32_t>(*record.latest_binding_stage));
    } else {
      json += "null";
    }
    json += ",\"origin\":";
    append_number(json, static_cast<std::uint32_t>(record.origin));
    json += ",\"reason\":";
    append_number(json, record.reason.value);
    json += ",\"reuse_policy\":";
    append_number(json, static_cast<std::uint32_t>(record.reuse_policy));
    json.push_back('}');
  });
}

void append_pool_capacity_plan_sections(std::string& output, const ExecutionPlanState& state) {
  output += ",\"pool_capacity_contributors\":";
  append_array(output, state.pool_capacity_contributors, [](std::string& json, const auto& record) {
    json += "{\"contributor_id\":";
    append_json_string(json, record.contributor_id);
    json += ",\"maximum_distinct_slots\":";
    append_number(json, record.maximum_distinct_slots);
    json += ",\"pool_id\":";
    append_json_string(json, record.pool_id);
    json += ",\"reasons\":";
    append_array(json, record.reasons, [](std::string& values, TensorSlotClaimReason reason) {
      append_json_string(values, tensor_slot_claim_reason_name(reason));
    });
    json.push_back('}');
  });
  output += ",\"pool_capacity_dependencies\":";
  append_array(output, state.pool_capacity_dependencies, [](std::string& json, const auto& record) {
    json += "{\"alternative_group\":";
    append_number(json, record.alternative_group.value);
    json += ",\"id\":";
    append_number(json, record.id.value);
    json += ",\"mapping_multiplicity\":";
    append_number(json, record.mapping_multiplicity);
    json += ",\"origin\":";
    append_number(json, static_cast<std::uint32_t>(record.origin));
    json += ",\"pool_id\":";
    append_json_string(json, record.pool_id);
    json += ",\"predecessor_contributor_id\":";
    append_json_string(json, record.predecessor_contributor_id);
    json += ",\"queue_sizing_recipe\":";
    append_number(json, record.queue_sizing_recipe.value);
    json += ",\"reasons\":";
    append_array(json, record.reasons, [](std::string& values, TensorSlotClaimReason reason) {
      append_json_string(values, tensor_slot_claim_reason_name(reason));
    });
    json += ",\"scenario\":";
    append_number(json, record.scenario.value);
    json += ",\"source\":";
    append_number(json, static_cast<std::uint32_t>(record.source));
    json.push_back('}');
  });
  output += ",\"provider_allocation_plans\":";
  append_array(output, state.provider_allocation_plans, [](std::string& json, const auto& record) {
    json += "{\"allocation_kind\":";
    append_number(json, static_cast<std::uint32_t>(record.allocation_kind));
    json += ",\"complete_before\":";
    append_number(json, static_cast<std::uint32_t>(record.complete_before));
    json += ",\"disposition\":";
    append_number(json, static_cast<std::uint32_t>(record.disposition));
    json += ",\"first_sizing_dependency\":";
    append_number(json, record.first_sizing_dependency.value);
    json += ",\"geometry_mode\":";
    append_number(json, static_cast<std::uint32_t>(record.geometry_mode));
    json += ",\"id\":";
    append_number(json, record.id.value);
    json += ",\"interface_topic\":";
    if (record.interface_topic) {
      append_number(json, record.interface_topic->value);
    } else {
      json += "null";
    }
    json += ",\"output\":";
    if (record.output) {
      append_number(json, record.output->value());
    } else {
      json += "null";
    }
    json += ",\"pool_id\":";
    append_json_string(json, record.pool_id);
    json += ",\"reason\":";
    append_number(json, record.reason.value);
    json += ",\"sizing_dependency_count\":";
    append_number(json, record.sizing_dependency_count);
    json += ",\"tensor_field_id\":";
    append_optional_number(json, record.tensor_field_id);
    json += ",\"tensor_index\":";
    append_optional_number(json, record.tensor_index);
    json += ",\"tensor_template\":";
    if (record.tensor_template) {
      const auto& tensor_template = *record.tensor_template;
      json += "{\"allocator_process_group\":";
      append_number(json, tensor_template.allocator_process_group.value);
      json += ",\"allocator_root_id\":";
      append_json_string(json, tensor_template.allocator_root_id);
      json += ",\"axes\":";
      append_tensor_axes(json, tensor_template.axes);
      json += ",\"bounds\":";
      append_tensor_allocation_bounds(json, tensor_template.bounds);
      json += ",\"device\":";
      append_number(json, tensor_template.device.value);
      json += ",\"effective_profile\":";
      append_optional_profile(json, tensor_template.effective_profile);
      json += ",\"fallback_rejected\":";
      append_bool(json, tensor_template.fallback_rejected);
      json += ",\"layout\":";
      append_optional_tensor_layout(json, tensor_template.layout);
      json += ",\"max_emits_per_compute\":";
      append_number(json, tensor_template.max_emits_per_compute);
      json += ",\"max_outstanding_loans\":";
      append_number(json, tensor_template.max_outstanding_loans);
      json += ",\"memory_kind\":";
      append_number(json, static_cast<std::uint32_t>(tensor_template.memory_kind));
      json.push_back('}');
    } else {
      json += "null";
    }
    json.push_back('}');
  });
}

}  // namespace holoscan::detail::plan_artifact_rate_serialization_internal
