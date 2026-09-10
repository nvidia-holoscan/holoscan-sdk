/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_rate_state.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../compiler/execution_plan_contracts.hpp"
#include "../compiler/execution_plan_state.hpp"
#include "../core/diagnostic_registry.hpp"
#include "../core/operator_spec_compiler_access.hpp"
#include "../core/rate_math_internal.hpp"
#include "graph_exec.hpp"
#include "runtime_worker.hpp"

namespace holoscan::detail {
namespace {

enum class RuntimeResolutionState : std::uint8_t {
  kUnvisited = 0,
  kResolving = 1,
  kComplete = 2,
  kFailed = 3,
};

[[nodiscard]] Error rate_error(ErrorCode code, const char* token) {
  return Error{code, token};
}

[[nodiscard]] DiagnosticId rate_diagnostic_id(std::string_view token) noexcept {
  const auto separator = token.find(':');
  const DiagnosticId id = diagnostic_id(token.substr(0U, separator));
  return id ? id : diagnostic_id("RATE_GENERATION_REJECTED");
}

[[nodiscard]] const StructuralDiagnosticDescriptor* rate_diagnostic_descriptor(
    DiagnosticId id) noexcept {
  for (const auto& descriptor : kStructuralDiagnosticRegistry) {
    if (static_cast<std::uint32_t>(descriptor.id) == id.value) {
      return &descriptor;
    }
  }
  return nullptr;
}

[[nodiscard]] std::optional<std::uint64_t> checked_capacity_bytes(
    std::optional<std::uint64_t> bytes_per_slot, std::optional<std::uint32_t> entries) noexcept {
  if (!bytes_per_slot || !entries || *entries == 0U ||
      *bytes_per_slot > std::numeric_limits<std::uint64_t>::max() / *entries) {
    return std::nullopt;
  }
  return *bytes_per_slot * *entries;
}

[[nodiscard]] std::optional<std::uint32_t> compose_hard_floor(const ExecutionPlanState& plan,
                                                              EdgeId edge) noexcept {
  struct Group {
    HardFloorScenarioId scenario{};
    HardFloorAlternativeGroupId alternative{};
    std::uint32_t slots{};
  };
  std::vector<Group> groups;
  groups.reserve(plan.hard_floor_contributors.size());
  for (const auto& contributor : plan.hard_floor_contributors) {
    if (contributor.edge != edge) {
      continue;
    }
    if (contributor.required_slots == 0U || contributor.scenario.value == 0U ||
        contributor.alternative_group.value == 0U) {
      return std::nullopt;
    }
    const auto found = std::ranges::find_if(groups, [&](const auto& group) {
      return group.scenario == contributor.scenario &&
             group.alternative == contributor.alternative_group;
    });
    if (found == groups.end()) {
      groups.push_back(Group{.scenario = contributor.scenario,
                             .alternative = contributor.alternative_group,
                             .slots = contributor.required_slots});
    } else {
      found->slots = std::max(found->slots, contributor.required_slots);
    }
  }
  std::ranges::sort(groups, [](const auto& left, const auto& right) {
    return left.scenario.value != right.scenario.value
               ? left.scenario.value < right.scenario.value
               : left.alternative.value < right.alternative.value;
  });
  std::uint32_t result = 1U;
  std::uint32_t scenario_total{};
  std::optional<HardFloorScenarioId> scenario;
  for (const auto& group : groups) {
    if (scenario.value_or(group.scenario) != group.scenario) {
      result = std::max(result, scenario_total);
      scenario_total = 0U;
    }
    scenario = group.scenario;
    if (group.slots > std::numeric_limits<std::uint32_t>::max() - scenario_total) {
      return std::nullopt;
    }
    scenario_total += group.slots;
  }
  return std::max(result, scenario_total);
}

[[nodiscard]] std::optional<std::uint32_t> advancement_capacity(
    PortId input, std::uint32_t candidate_active_depth,
    std::span<const InputSelectionPlanRecord> selections) noexcept {
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
        return candidate_active_depth == 0U ? std::nullopt
                                            : std::optional<std::uint32_t>{candidate_active_depth};
      }
      return found->max_selected_samples == 0U
                 ? std::nullopt
                 : std::optional<std::uint32_t>{found->max_selected_samples};
    case InputSelectorKind::kBracketedWindow:
      return std::nullopt;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<LifecycleStage> provider_latest_binding_stage(
    const ExecutionPlanState& plan, const ProviderAllocationPlanRecord& provider) noexcept {
  const auto dependencies = provider_sizing_dependencies(plan.pool_capacity_dependencies, provider);
  if (!dependencies) {
    return std::nullopt;
  }
  std::optional<LifecycleStage> latest;
  for (const auto& dependency : *dependencies) {
    if (dependency.source != PoolCapacityDependencySource::kQueueSizingRecipe) {
      continue;
    }
    const auto recipe = std::ranges::find(plan.queue_sizing_recipes,
                                          dependency.queue_sizing_recipe,
                                          &QueueSizingRecipePlanRecord::id);
    if (recipe == plan.queue_sizing_recipes.end() || !recipe->latest_binding_stage) {
      continue;
    }
    const LifecycleStage recipe_stage = recipe->latest_binding_stage.value_or(LifecycleStage{});
    if (!latest || static_cast<std::uint8_t>(latest.value_or(recipe_stage)) <
                       static_cast<std::uint8_t>(recipe_stage)) {
      latest = recipe_stage;
    }
  }
  return latest;
}

[[nodiscard]] bool forward_rate_phase(LifecycleTransitionPhase phase) noexcept {
  return phase == LifecycleTransitionPhase::kStartupForward ||
         phase == LifecycleTransitionPhase::kRestartForward;
}

[[nodiscard]] std::size_t pointer_hash(const void* pointer) noexcept {
  std::uintptr_t value = reinterpret_cast<std::uintptr_t>(pointer);
  value ^= value >> 17U;
  value *= static_cast<std::uintptr_t>(0xed5ad4bbU);
  value ^= value >> 11U;
  return static_cast<std::size_t>(value);
}

}  // namespace

RuntimeRateGenerationPin::RuntimeRateGenerationPin(RuntimeRateState* state, std::size_t slot,
                                                   OperatorInstanceId owner) noexcept
    : state_(state), slot_(slot), owner_(owner) {}

RuntimeRateGenerationPin::RuntimeRateGenerationPin(RuntimeRateGenerationPin&& other) noexcept
    : state_(std::exchange(other.state_, nullptr)),
      slot_(std::exchange(other.slot_, RuntimeRateState::kInvalidSlot)),
      owner_(std::exchange(other.owner_, OperatorInstanceId{})) {}

RuntimeRateGenerationPin& RuntimeRateGenerationPin::operator=(
    RuntimeRateGenerationPin&& other) noexcept {
  if (this != &other) {
    release();
    state_ = std::exchange(other.state_, nullptr);
    slot_ = std::exchange(other.slot_, RuntimeRateState::kInvalidSlot);
    owner_ = std::exchange(other.owner_, OperatorInstanceId{});
  }
  return *this;
}

RuntimeRateGenerationPin::~RuntimeRateGenerationPin() {
  release();
}

void RuntimeRateGenerationPin::release() noexcept {
  if (state_ != nullptr) {
    state_->release_pin(slot_);
  }
  state_ = nullptr;
  slot_ = RuntimeRateState::kInvalidSlot;
  owner_ = {};
}

expected<PortRateView, Error> RuntimeRateGenerationPin::resolve(void* state,
                                                                const void* declaration,
                                                                bool output) noexcept {
  auto* const pin = static_cast<RuntimeRateGenerationPin*>(state);
  if (pin == nullptr || pin->state_ == nullptr || pin->slot_ == RuntimeRateState::kInvalidSlot) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return pin->state_->resolve_pinned(pin->slot_, pin->owner_, declaration, output);
}

RuntimeRateCallbackBinding::RuntimeRateCallbackBinding(RuntimeRateState* state,
                                                       OperatorInstanceId owner) noexcept
    : state_(state), owner_(owner) {}

expected<PortRateView, Error> RuntimeRateCallbackBinding::resolve(void* state,
                                                                  const void* declaration,
                                                                  bool output) noexcept {
  auto* const binding = static_cast<RuntimeRateCallbackBinding*>(state);
  if (binding == nullptr || binding->state_ == nullptr || binding->owner_ == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (binding->pin_.state_ == nullptr) {
    auto pin = binding->state_->pin_published(binding->owner_);
    if (!pin) {
      return make_unexpected(std::move(pin.error()));
    }
    binding->pin_ = std::move(*pin);
  }
  return RuntimeRateGenerationPin::resolve(&binding->pin_, declaration, output);
}

RuntimeRateState::RuntimeRateState(const ExecutionPlanState& plan, RunGenerationId run_generation,
                                   ProcessGroupId process_group) noexcept
    : plan_(&plan), run_generation_(run_generation), process_group_(process_group) {}

RuntimeRateState::~RuntimeRateState() = default;

// Runtime construction deliberately validates and joins the complete immutable rate-plan family.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<std::unique_ptr<RuntimeRateState>, Error> RuntimeRateState::create(
    const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
    RunGenerationId run_generation, std::size_t execution_lane_count,
    ProcessGroupId process_group) {
  if (run_generation == 0U || plan.port_rates.size() != plan.ports.size() ||
      (process_group &&
       std::ranges::none_of(plan.vertex_process_placements, [&](const auto& placement) {
         return placement.process_group == process_group;
       }))) {
    return make_unexpected(rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_PLAN_INVALID"));
  }
  auto result =
      std::unique_ptr<RuntimeRateState>{new RuntimeRateState(plan, run_generation, process_group)};
  result->port_owners_.assign(plan.ports.size(), OperatorInstanceId{});
  result->port_directions_.assign(plan.ports.size(), PortDirection::kInput);
  result->input_sources_.assign(plan.ports.size(), PortId{});
  result->local_ports_.assign(plan.ports.size(), process_group ? 0U : 1U);
  result->local_edges_.assign(plan.payload_edges.size(), process_group ? 0U : 1U);
  result->local_providers_.assign(plan.provider_allocation_plans.size(), process_group ? 0U : 1U);
  result->local_dependencies_.assign(plan.pool_capacity_dependencies.size(),
                                     process_group ? 0U : 1U);
  result->evidence_ports_.assign(plan.ports.size(), process_group ? 0U : 1U);
  result->evidence_queues_.assign(plan.payload_edges.size(), process_group ? 0U : 1U);
  for (std::size_t index = 0U; index < plan.ports.size(); ++index) {
    const auto& port = plan.ports[index];
    if (port.id.value() != index + 1U || plan.port_rates[index].port != port.id || !port.owner) {
      return make_unexpected(rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_PLAN_INVALID"));
    }
    result->port_directions_[index] = port.direction;
    if (process_group && worker_owns_vertex(plan, process_group, port.owner)) {
      result->local_ports_[index] = 1U;
    }
  }

  std::size_t declaration_count = 0U;
  for (const RuntimeOperatorInstance& instance : operators) {
    if (instance.replay_spec == nullptr) {
      return make_unexpected(
          rate_error(ErrorCode::kInvalidArgument, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
    }
    declaration_count += OperatorSpecCompilerAccess::port_count(*instance.replay_spec);
  }
  std::size_t table_size = 1U;
  while (table_size < std::max<std::size_t>(2U, declaration_count * 2U)) {
    if (table_size > std::numeric_limits<std::size_t>::max() / 2U) {
      return make_unexpected(
          rate_error(ErrorCode::kResourceExhausted, "PORT_RATE_INDEX_EXHAUSTED"));
    }
    table_size *= 2U;
  }
  result->declaration_index_.resize(table_size);

  for (RuntimeOperatorInstance& instance : operators) {
    if (process_group && !worker_owns_vertex(plan, process_group, instance.vertex)) {
      return make_unexpected(
          rate_error(ErrorCode::kInvalidArgument, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
    }
    for (std::size_t index = 0U;
         index < OperatorSpecCompilerAccess::port_count(*instance.replay_spec);
         ++index) {
      const auto replay = OperatorSpecCompilerAccess::port(*instance.replay_spec, index);
      const auto planned =
          std::find_if(plan.ports.begin(), plan.ports.end(), [&](const PortPlanRecord& candidate) {
            return candidate.owner == instance.vertex &&
                   candidate.owner_kind == PortOwnerKind::kOperator &&
                   candidate.direction == replay.direction && candidate.local_name == replay.name;
          });
      if (planned == plan.ports.end() || replay.declaration == nullptr || !planned->id ||
          planned->id.value() > plan.ports.size()) {
        return make_unexpected(
            rate_error(ErrorCode::kInvalidArgument, "RUNTIME_OPERATOR_REPLAY_MISMATCH"));
      }
      const std::size_t port_index = static_cast<std::size_t>(planned->id.value() - 1U);
      if (instance.operator_id == 0U || plan.port_rates[port_index].port != planned->id ||
          plan.ports[port_index].id != planned->id || result->port_owners_[port_index] != 0U ||
          result->port_directions_[port_index] != replay.direction) {
        return make_unexpected(rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_PLAN_INVALID"));
      }
      // PortPlanRecord::owner is a canonical VertexId. Reusable-Graph interface vertices are
      // plan-only, so that identity is not interchangeable with the reconstructed dense
      // OperatorInstanceId. Bind the Runtime owner from the exact vertex/name/direction replay
      // match instead of casting between the two domains.
      result->port_owners_[port_index] = instance.operator_id;

      const std::size_t mask = result->declaration_index_.size() - 1U;
      std::size_t bucket = pointer_hash(replay.declaration) & mask;
      bool inserted = false;
      for (std::size_t probe = 0U; probe < result->declaration_index_.size(); ++probe) {
        auto& entry = result->declaration_index_[bucket];
        if (entry.declaration == nullptr) {
          entry = DeclarationEntry{
              .declaration = replay.declaration,
              .port = planned->id,
              .owner = instance.operator_id,
              .direction = replay.direction,
          };
          inserted = true;
          break;
        }
        if (entry.declaration == replay.declaration) {
          return make_unexpected(
              rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_DECLARATION_AMBIGUOUS"));
        }
        bucket = (bucket + 1U) & mask;
      }
      if (!inserted) {
        return make_unexpected(
            rate_error(ErrorCode::kResourceExhausted, "PORT_RATE_INDEX_EXHAUSTED"));
      }
    }
  }

  for (std::size_t index = 0U; index < plan.ports.size(); ++index) {
    if (plan.ports[index].id.value() != index + 1U ||
        plan.port_rates[index].port.value() != index + 1U) {
      return make_unexpected(rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_PLAN_INVALID"));
    }
  }
  for (std::size_t edge_index = 0U; edge_index < plan.payload_edges.size(); ++edge_index) {
    const auto& edge = plan.payload_edges[edge_index];
    if (!edge.source || !edge.target || edge.source.value() > plan.ports.size() ||
        edge.target.value() > plan.ports.size()) {
      return make_unexpected(rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_PLAN_INVALID"));
    }
    auto& source = result->input_sources_[static_cast<std::size_t>(edge.target.value() - 1U)];
    if (source && source != edge.source) {
      return make_unexpected(rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_PLAN_INVALID"));
    }
    source = edge.source;
    result->local_edges_[edge_index] =
        result->local_ports_[static_cast<std::size_t>(edge.target.value() - 1U)];
  }
  for (std::size_t index = 0U; index < plan.port_rates.size(); ++index) {
    const PortRatePlanRecord& record = plan.port_rates[index];
    if (record.kind != PortRateKind::kLifecycleNominal) {
      continue;
    }
    if (!record.lifecycle_binding_stage ||
        static_cast<std::size_t>(*record.lifecycle_binding_stage) >= kForwardStageCount ||
        *record.lifecycle_binding_stage == LifecycleStage::kAllocate ||
        *record.lifecycle_binding_stage == LifecycleStage::kStart) {
      return make_unexpected(rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_PLAN_INVALID"));
    }
    if (result->local_ports_[index] != 0U) {
      result->barrier_stages_[static_cast<std::size_t>(*record.lifecycle_binding_stage)] = true;
    }
  }

  if (plan.ports.size() > std::numeric_limits<std::uint32_t>::max() ||
      plan.payload_edges.size() > std::numeric_limits<std::uint32_t>::max() ||
      plan.pool_capacity_dependencies.size() > std::numeric_limits<std::uint32_t>::max() ||
      plan.provider_allocation_plans.size() > std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(
        rate_error(ErrorCode::kResourceExhausted, "PORT_RATE_GENERATION_EXHAUSTED"));
  }
  result->published_provider_identities_.resize(plan.provider_allocation_plans.size());
  result->provider_latest_binding_stages_.reserve(plan.provider_allocation_plans.size());
  for (std::size_t provider_index = 0U; provider_index < plan.provider_allocation_plans.size();
       ++provider_index) {
    const auto& provider = plan.provider_allocation_plans[provider_index];
    const auto dependencies =
        provider_sizing_dependencies(plan.pool_capacity_dependencies, provider);
    if (!dependencies) {
      return make_unexpected(rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_PLAN_INVALID"));
    }
    for (const auto& dependency : *dependencies) {
      if (dependency.source == PoolCapacityDependencySource::kQueueSizingRecipe &&
          std::ranges::count(plan.queue_sizing_recipes,
                             dependency.queue_sizing_recipe,
                             &QueueSizingRecipePlanRecord::id) != 1) {
        return make_unexpected(rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_PLAN_INVALID"));
      }
    }
    if (process_group) {
      PortId owner_port{};
      if (provider.output) {
        owner_port = *provider.output;
      } else if (provider.interface_topic) {
        const auto interface_pool =
            std::ranges::find_if(plan.tensor_interface_pools, [&](const auto& candidate) {
              return candidate.topic == *provider.interface_topic &&
                     candidate.pool_id == provider.pool_id;
            });
        if (interface_pool == plan.tensor_interface_pools.end() ||
            std::ranges::count_if(plan.tensor_interface_pools, [&](const auto& candidate) {
              return candidate.topic == *provider.interface_topic &&
                     candidate.pool_id == provider.pool_id;
            }) != 1) {
          return make_unexpected(rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_PLAN_INVALID"));
        }
        owner_port = interface_pool->graph_port;
      }
      if (!owner_port || owner_port.value() > result->local_ports_.size()) {
        return make_unexpected(rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_PLAN_INVALID"));
      }
      result->local_providers_[provider_index] =
          result->local_ports_[static_cast<std::size_t>(owner_port.value() - 1U)];
    }
    result->provider_latest_binding_stages_.push_back(
        provider_latest_binding_stage(plan, provider));
  }

  if (process_group) {
    result->evidence_ports_ = result->local_ports_;
    result->evidence_queues_ = result->local_edges_;
    for (std::size_t provider_index = 0U; provider_index < plan.provider_allocation_plans.size();
         ++provider_index) {
      if (result->local_providers_[provider_index] == 0U) {
        continue;
      }
      const auto& provider = plan.provider_allocation_plans[provider_index];
      const std::size_t first = provider.first_sizing_dependency.value - 1U;
      for (std::size_t offset = 0U; offset < provider.sizing_dependency_count; ++offset) {
        const std::size_t dependency_index = first + offset;
        result->local_dependencies_[dependency_index] = 1U;
        const auto& dependency = plan.pool_capacity_dependencies[dependency_index];
        if (dependency.source != PoolCapacityDependencySource::kQueueSizingRecipe) {
          continue;
        }
        const auto recipe = std::ranges::find(plan.queue_sizing_recipes,
                                              dependency.queue_sizing_recipe,
                                              &QueueSizingRecipePlanRecord::id);
        if (recipe == plan.queue_sizing_recipes.end()) {
          return make_unexpected(rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_PLAN_INVALID"));
        }
        const auto edge =
            std::ranges::find(plan.payload_edges, recipe->edge, &PayloadEdgePlanRecord::edge);
        if (edge == plan.payload_edges.end()) {
          return make_unexpected(rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_PLAN_INVALID"));
        }
        result->evidence_queues_[static_cast<std::size_t>(
            std::distance(plan.payload_edges.begin(), edge))] = 1U;
      }
    }
    for (std::size_t edge_index = 0U; edge_index < plan.payload_edges.size(); ++edge_index) {
      if (result->evidence_queues_[edge_index] == 0U) {
        continue;
      }
      const auto& edge = plan.payload_edges[edge_index];
      result->evidence_ports_[static_cast<std::size_t>(edge.source.value() - 1U)] = 1U;
      result->evidence_ports_[static_cast<std::size_t>(edge.target.value() - 1U)] = 1U;
    }
    bool changed = true;
    while (changed) {
      changed = false;
      for (std::size_t index = 0U; index < plan.port_rates.size(); ++index) {
        if (result->evidence_ports_[index] == 0U) {
          continue;
        }
        const auto& declaration = plan.port_rates[index];
        const PortId dependency = declaration.follows_input.value_or(result->input_sources_[index]);
        if (dependency && dependency.value() <= result->evidence_ports_.size() &&
            result->evidence_ports_[static_cast<std::size_t>(dependency.value() - 1U)] == 0U) {
          result->evidence_ports_[static_cast<std::size_t>(dependency.value() - 1U)] = 1U;
          changed = true;
        }
      }
    }
  }

  const auto recipe_is_local = [&](const QueueSizingRecipePlanRecord& recipe) {
    const auto edge =
        std::ranges::find(plan.payload_edges, recipe.edge, &PayloadEdgePlanRecord::edge);
    return edge != plan.payload_edges.end() &&
           result->local_edges_[static_cast<std::size_t>(
               std::distance(plan.payload_edges.begin(), edge))] != 0U;
  };
  result->resource_materialization_required_ =
      std::ranges::any_of(plan.queue_sizing_recipes, [&](const auto& recipe) {
        return recipe_is_local(recipe) && recipe.latest_binding_stage.has_value();
      });
  result->deferred_resource_materialization_required_ =
      std::ranges::any_of(plan.queue_sizing_recipes, [&](const auto& recipe) {
        return recipe_is_local(recipe) &&
               recipe.geometry_mode == AllocationGeometryMode::kLifecycleCandidate;
      });
  for (std::size_t index = 0U; index < plan.provider_allocation_plans.size(); ++index) {
    if (result->local_providers_[index] == 0U) {
      continue;
    }
    if (result->provider_latest_binding_stages_[index]) {
      result->resource_materialization_required_ = true;
    }
    if (plan.provider_allocation_plans[index].geometry_mode ==
        AllocationGeometryMode::kLifecycleCandidate) {
      result->deferred_resource_materialization_required_ = true;
    }
  }

  const std::size_t slot_count = std::max<std::size_t>(4U, execution_lane_count + 3U);
  result->slots_.reserve(slot_count);
  for (std::size_t index = 0U; index < slot_count; ++index) {
    auto slot = std::make_unique<GenerationSlot>();
    slot->views.resize(plan.ports.size());
    slot->bindings.resize(plan.ports.size());
    slot->resolution_state.resize(plan.ports.size());
    slot->immediate_provenance.resize(plan.ports.size());
    slot->root_provenance.resize(plan.ports.size());
    slot->expectation_comparisons.resize(plan.ports.size());
    slot->queue_depths.resize(plan.payload_edges.size());
    slot->pool_contributions.resize(plan.pool_capacity_dependencies.size());
    slot->pool_capacities.resize(plan.provider_allocation_plans.size());
    slot->pool_capacity_terms.resize(plan.pool_capacity_dependencies.size());
    slot->mapped_pool_capacity_terms.resize(plan.pool_capacity_dependencies.size());
    slot->provider_results.resize(plan.provider_allocation_plans.size());
    slot->provider_records.resize(plan.provider_allocation_plans.size());
    slot->provider_decision_required.resize(plan.provider_allocation_plans.size());
    for (std::size_t provider_index = 0U; provider_index < plan.provider_allocation_plans.size();
         ++provider_index) {
      const auto& provider = plan.provider_allocation_plans[provider_index];
      if (result->local_providers_[provider_index] == 0U) {
        continue;
      }
      if (provider.geometry_mode != AllocationGeometryMode::kLifecycleCandidate ||
          provider.allocation_kind != ProviderAllocationKind::kTensorOutput || !provider.output ||
          !provider.tensor_template) {
        continue;
      }
      const auto& tensor_template = *provider.tensor_template;
      slot->tensor_allocations.push_back(RuntimeTensorAllocationGeneration{
          .provider = provider.id,
          .run_generation = run_generation,
          .rate_generation = 0U,
          .allocation =
              TensorAllocationPlanRecord{
                  .output = *provider.output,
                  .plan_generation = plan.plan_generation,
                  .run_generation = 0U,
                  .pool_id = provider.pool_id,
                  .allocator_root_id = tensor_template.allocator_root_id,
                  .allocator_process_group = tensor_template.allocator_process_group,
                  .bounds = tensor_template.bounds,
                  .axes = tensor_template.axes,
                  .layout = tensor_template.layout,
                  .memory_kind = tensor_template.memory_kind,
                  .device = tensor_template.device,
                  .max_outstanding_loans = tensor_template.max_outstanding_loans,
                  .max_emits_per_compute = tensor_template.max_emits_per_compute,
                  .materialize_before = LifecycleStage::kArm,
                  .effective_profile = tensor_template.effective_profile,
                  .disposition = PlanDisposition::kAccepted,
                  .fallback_rejected = tensor_template.fallback_rejected,
              },
      });
    }
    result->slots_.push_back(std::move(slot));
  }
  return result;
}

const RuntimeRateState::DeclarationEntry* RuntimeRateState::find_declaration(
    const void* declaration) const noexcept {
  if (declaration == nullptr || declaration_index_.empty()) {
    return nullptr;
  }
  const std::size_t mask = declaration_index_.size() - 1U;
  std::size_t bucket = pointer_hash(declaration) & mask;
  for (std::size_t probe = 0U; probe < declaration_index_.size(); ++probe) {
    const auto& entry = declaration_index_[bucket];
    if (entry.declaration == declaration) {
      return &entry;
    }
    if (entry.declaration == nullptr) {
      return nullptr;
    }
    bucket = (bucket + 1U) & mask;
  }
  return nullptr;
}

expected<std::size_t, Error> RuntimeRateState::acquire_candidate_slot_locked() noexcept {
  if (candidate_slot_) {
    return make_unexpected(rate_error(ErrorCode::kLifecycleRunState, "PORT_RATE_CANDIDATE_ACTIVE"));
  }
  const std::size_t published = published_slot_.load(std::memory_order_acquire);
  for (std::size_t index = 0U; index < slots_.size(); ++index) {
    if (index != published && (!rejected_slot_ || index != *rejected_slot_) &&
        slots_[index]->readers.load(std::memory_order_acquire) == 0U) {
      candidate_slot_ = index;
      return index;
    }
  }
  return make_unexpected(
      rate_error(ErrorCode::kResourceExhausted, "PORT_RATE_GENERATION_SLOTS_EXHAUSTED"));
}

expected<RateGenerationId, Error> RuntimeRateState::reserve_generation_locked() noexcept {
  if (next_generation_ == 0U || next_generation_ > kMaximumPackedRateGeneration) {
    return make_unexpected(
        rate_error(ErrorCode::kResourceExhausted, "PORT_RATE_GENERATION_EXHAUSTED"));
  }
  return next_generation_++;
}

void RuntimeRateState::initialize_candidate_locked(GenerationSlot& slot,
                                                   RateGenerationId generation) noexcept {
  slot.predecessor_slot = kInvalidSlot;
  slot.generation = generation;
  slot.failure = Error{ErrorCode::kSuccess};
  slot.failure_source = {};
  slot.preflighted = false;
  slot.provider_records_sealed = false;
  slot.closed_stages.fill(false);
  std::fill(slot.bindings.begin(), slot.bindings.end(), std::nullopt);
  std::fill(slot.resolution_state.begin(), slot.resolution_state.end(), 0U);
  std::fill(slot.immediate_provenance.begin(), slot.immediate_provenance.end(), PortId{});
  std::fill(slot.root_provenance.begin(), slot.root_provenance.end(), PortId{});
  std::fill(slot.expectation_comparisons.begin(),
            slot.expectation_comparisons.end(),
            RateComparison::kNotCompared);
  for (std::size_t index = 0U; index < slot.views.size(); ++index) {
    slot.views[index] = PortRateView{
        .resolved_nominal_rate = std::nullopt,
        .sizing_rate = std::nullopt,
        .max_nominal_rate = plan_->port_rates[index].declared_max_nominal_rate,
        .origin = ResolvedRateOrigin::kUnresolved,
        .state = PortRateViewState::kUnresolved,
        .run_generation = run_generation_,
        .rate_generation = generation,
    };
  }
  for (std::size_t index = 0U; index < slot.queue_depths.size(); ++index) {
    slot.queue_depths[index] = RuntimeQueueDepthGeneration{
        .edge = plan_->payload_edges[index].edge,
        .active_logical_depth = 0U,
    };
  }
  for (std::size_t index = 0U; index < slot.pool_contributions.size(); ++index) {
    slot.pool_contributions[index] = RuntimePoolCapacityContributionGeneration{
        .dependency = plan_->pool_capacity_dependencies[index].id,
    };
  }
  for (std::size_t index = 0U; index < slot.pool_capacities.size(); ++index) {
    slot.pool_capacities[index] = RuntimePoolCapacityGeneration{
        .provider = plan_->provider_allocation_plans[index].id,
        .run_generation = run_generation_,
        .rate_generation = generation,
    };
  }
  for (std::size_t index = 0U; index < slot.provider_results.size(); ++index) {
    slot.provider_results[index] = RuntimeProviderMaterializationResult{
        .provider = plan_->provider_allocation_plans[index].id,
    };
    slot.provider_records[index] = ProviderAllocationGenerationRecord{};
    slot.provider_decision_required[index] = 0U;
  }
  for (auto& generated : slot.tensor_allocations) {
    generated.run_generation = run_generation_;
    generated.rate_generation = generation;
    generated.allocation.slot_count = 0U;
    generated.allocation.aggregate_byte_capacity = 0U;
    generated.allocation.control_capacity = 0U;
    generated.allocation.readiness_capacity = 0U;
    generated.allocation.release_capacity = 0U;
    generated.allocation.retirement_capacity = 0U;
    generated.allocation.quarantine_capacity = 0U;
  }
}

void RuntimeRateState::fail_candidate_locked(GenerationSlot& slot, Error error) noexcept {
  if (slot.failure.code == ErrorCode::kSuccess) {
    slot.failure = std::move(error);
  }
  slot.preflighted = false;
}

// Resolution is one recursive, lock-held transaction over mutually dependent port declarations.
// Splitting it would separate cycle state from the candidate views it protects.
expected<void, Error>
RuntimeRateState::resolve_candidate_locked(  // NOLINT(readability-function-cognitive-complexity)
    GenerationSlot& slot) noexcept {
  if (slot.failure.code != ErrorCode::kSuccess) {
    return make_unexpected(slot.failure);
  }
  std::fill(slot.resolution_state.begin(), slot.resolution_state.end(), 0U);
  for (std::size_t index = 0U; index < slot.views.size(); ++index) {
    slot.views[index] = PortRateView{
        .resolved_nominal_rate = std::nullopt,
        .sizing_rate = std::nullopt,
        .max_nominal_rate = plan_->port_rates[index].declared_max_nominal_rate,
        .origin = ResolvedRateOrigin::kUnresolved,
        .state = PortRateViewState::kUnresolved,
        .run_generation = run_generation_,
        .rate_generation = slot.generation,
    };
  }

  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  auto resolve = [&](auto&& self, std::size_t index) noexcept -> bool {
    auto& state = slot.resolution_state[index];
    if (state == static_cast<std::uint8_t>(RuntimeResolutionState::kComplete)) {
      return true;
    }
    if (state == static_cast<std::uint8_t>(RuntimeResolutionState::kResolving) ||
        state == static_cast<std::uint8_t>(RuntimeResolutionState::kFailed)) {
      fail_candidate_locked(slot,
                            rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_RESOLUTION_CYCLE"));
      state = static_cast<std::uint8_t>(RuntimeResolutionState::kFailed);
      return false;
    }
    state = static_cast<std::uint8_t>(RuntimeResolutionState::kResolving);
    auto& view = slot.views[index];
    const auto& declaration = plan_->port_rates[index];
    const bool output = port_directions_[index] == PortDirection::kOutput;

    if (output) {
      switch (declaration.kind) {
        case PortRateKind::kFixedNominal:
          view.resolved_nominal_rate = declaration.declared_nominal_rate;
          view.sizing_rate = declaration.declared_nominal_rate;
          view.max_nominal_rate = declaration.declared_nominal_rate;
          view.origin = declaration.declared_nominal_rate ? ResolvedRateOrigin::kOutputNominal
                                                          : ResolvedRateOrigin::kUnresolved;
          view.state = declaration.declared_nominal_rate ? PortRateViewState::kResolved
                                                         : PortRateViewState::kUnresolved;
          if (declaration.declared_nominal_rate) {
            slot.root_provenance[index] = declaration.port;
          }
          break;
        case PortRateKind::kLifecycleNominal: {
          if (!declaration.lifecycle_binding_stage) {
            fail_candidate_locked(
                slot, rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_PLAN_INVALID"));
            state = static_cast<std::uint8_t>(RuntimeResolutionState::kFailed);
            return false;
          }
          const std::size_t stage = static_cast<std::size_t>(*declaration.lifecycle_binding_stage);
          if (local_ports_[index] == 0U) {
            view.state = PortRateViewState::kPendingBarrier;
            break;
          }
          if (!slot.closed_stages[stage]) {
            view.state = PortRateViewState::kPendingBarrier;
            break;
          }
          if (!slot.bindings[index]) {
            fail_candidate_locked(slot,
                                  rate_error(ErrorCode::kNotReady, "PORT_RATE_BINDING_MISSING"));
            state = static_cast<std::uint8_t>(RuntimeResolutionState::kFailed);
            return false;
          }
          view.resolved_nominal_rate = slot.bindings[index];
          view.sizing_rate = slot.bindings[index];
          view.origin = ResolvedRateOrigin::kLifecycleBinding;
          view.state = PortRateViewState::kResolved;
          slot.root_provenance[index] = declaration.port;
          break;
        }
        case PortRateKind::kFollowsInput: {
          if (!declaration.follows_input || !declaration.follows_ratio ||
              !*declaration.follows_input ||
              declaration.follows_input->value() > slot.views.size()) {
            fail_candidate_locked(
                slot, rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_PLAN_INVALID"));
            state = static_cast<std::uint8_t>(RuntimeResolutionState::kFailed);
            return false;
          }
          const std::size_t dependency =
              static_cast<std::size_t>(declaration.follows_input->value() - 1U);
          slot.immediate_provenance[index] = *declaration.follows_input;
          if (!self(self, dependency)) {
            state = static_cast<std::uint8_t>(RuntimeResolutionState::kFailed);
            return false;
          }
          const auto& source = slot.views[dependency];
          if (source.resolved_nominal_rate) {
            view.resolved_nominal_rate =
                rate_math::multiply(*source.resolved_nominal_rate, *declaration.follows_ratio);
            if (!view.resolved_nominal_rate) {
              fail_candidate_locked(
                  slot, rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_ARITHMETIC_OVERFLOW"));
              state = static_cast<std::uint8_t>(RuntimeResolutionState::kFailed);
              return false;
            }
            view.sizing_rate = view.resolved_nominal_rate;
            view.origin = ResolvedRateOrigin::kFollowsExpression;
            view.state = PortRateViewState::kResolved;
            slot.root_provenance[index] = slot.root_provenance[dependency]
                                              ? slot.root_provenance[dependency]
                                              : *declaration.follows_input;
          } else {
            view.state = source.state == PortRateViewState::kPendingBarrier
                             ? PortRateViewState::kPendingBarrier
                             : PortRateViewState::kUnresolved;
          }
          if (source.max_nominal_rate) {
            view.max_nominal_rate =
                rate_math::multiply(*source.max_nominal_rate, *declaration.follows_ratio);
            if (!view.max_nominal_rate) {
              fail_candidate_locked(
                  slot, rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_ARITHMETIC_OVERFLOW"));
              state = static_cast<std::uint8_t>(RuntimeResolutionState::kFailed);
              return false;
            }
          }
          break;
        }
        case PortRateKind::kUnspecified:
        case PortRateKind::kExpectedInput:
          view.state = PortRateViewState::kUnresolved;
          break;
      }
    } else {
      bool horizon_resolution_failed = false;
      const auto expected_fallback_has_finite_horizon = [&]() noexcept {
        const VertexId owner = plan_->ports[index].owner;
        const auto trigger = std::ranges::find_if(plan_->temporal_triggers, [&](const auto& item) {
          return item.owner.id == owner.value;
        });
        if (trigger == plan_->temporal_triggers.end()) {
          return false;
        }
        if ((trigger->kind == TemporalRecordTriggerKind::kClock ||
             trigger->kind == TemporalRecordTriggerKind::kWindow) &&
            trigger->period_ns > 0) {
          return true;
        }
        if (trigger->kind != TemporalRecordTriggerKind::kEach || !trigger->trigger_input ||
            trigger->trigger_input.value() == index + 1U ||
            trigger->trigger_input.value() > slot.views.size()) {
          return false;
        }
        const std::size_t driver_index =
            static_cast<std::size_t>(trigger->trigger_input.value() - 1U);
        if (!self(self, driver_index)) {
          horizon_resolution_failed = true;
          return false;
        }
        return slot.views[driver_index].resolved_nominal_rate.has_value();
      };
      const PortId source_port = input_sources_[index];
      if (source_port) {
        const std::size_t source_index = static_cast<std::size_t>(source_port.value() - 1U);
        slot.immediate_provenance[index] = source_port;
        if (!self(self, source_index)) {
          state = static_cast<std::uint8_t>(RuntimeResolutionState::kFailed);
          return false;
        }
        const auto& source = slot.views[source_index];
        view.max_nominal_rate = source.max_nominal_rate;
        if (source.resolved_nominal_rate) {
          view.resolved_nominal_rate = source.resolved_nominal_rate;
          view.sizing_rate = source.sizing_rate;
          view.origin = ResolvedRateOrigin::kConnectedOutput;
          view.state = PortRateViewState::kResolved;
          slot.root_provenance[index] =
              slot.root_provenance[source_index] ? slot.root_provenance[source_index] : source_port;
          if (declaration.declared_expected_rate) {
            slot.expectation_comparisons[index] =
                *declaration.declared_expected_rate == *source.resolved_nominal_rate
                    ? RateComparison::kEqual
                    : RateComparison::kMismatch;
          }
        } else if (source.state == PortRateViewState::kPendingBarrier) {
          view.state = PortRateViewState::kPendingBarrier;
        } else if (declaration.declared_expected_rate && expected_fallback_has_finite_horizon()) {
          view.resolved_nominal_rate = declaration.declared_expected_rate;
          view.sizing_rate = declaration.declared_expected_rate;
          view.origin = ResolvedRateOrigin::kInputExpectationFallback;
          view.state = PortRateViewState::kResolved;
          slot.root_provenance[index] = declaration.port;
        }
      } else if (declaration.declared_expected_rate && expected_fallback_has_finite_horizon()) {
        view.resolved_nominal_rate = declaration.declared_expected_rate;
        view.sizing_rate = declaration.declared_expected_rate;
        view.origin = ResolvedRateOrigin::kInputExpectationFallback;
        view.state = PortRateViewState::kResolved;
        slot.root_provenance[index] = declaration.port;
      }
      if (horizon_resolution_failed) {
        state = static_cast<std::uint8_t>(RuntimeResolutionState::kFailed);
        return false;
      }
    }
    state = static_cast<std::uint8_t>(RuntimeResolutionState::kComplete);
    return true;
  };

  for (std::size_t index = 0U; index < slot.views.size(); ++index) {
    if (!resolve(resolve, index)) {
      return make_unexpected(slot.failure);
    }
  }
  auto queues = evaluate_queue_depths_locked(slot);
  if (!queues) {
    return queues;
  }
  return evaluate_pool_capacities_locked(slot);
}

// Queue evaluation applies every recipe and contributor to one provisional generation while the
// lifecycle lock is held; keeping the proof contiguous makes fail-closed publication auditable.
expected<void, Error> RuntimeRateState::
    evaluate_queue_depths_locked(  // NOLINT(readability-function-cognitive-complexity)
        GenerationSlot& slot) noexcept {
  for (std::size_t index = 0U; index < plan_->payload_edges.size(); ++index) {
    const auto& edge = plan_->payload_edges[index];
    auto& generated = slot.queue_depths[index];
    generated = RuntimeQueueDepthGeneration{.edge = edge.edge};
    const auto recipe = std::ranges::find(
        plan_->queue_sizing_recipes, edge.edge, &QueueSizingRecipePlanRecord::edge);
    if (recipe == plan_->queue_sizing_recipes.end()) {
      fail_candidate_locked(slot,
                            rate_error(ErrorCode::kInvalidArgument, "QUEUE_SIZING_RECIPE_MISSING"));
      return make_unexpected(slot.failure);
    }
    generated.recipe = recipe->id;
    const auto hard_floor = compose_hard_floor(*plan_, edge.edge);
    if (!hard_floor) {
      fail_candidate_locked(slot,
                            rate_error(ErrorCode::kResourceExhausted, "QUEUE_CAPACITY_OVERFLOW"));
      return make_unexpected(slot.failure);
    }
    const std::uint32_t hard_floor_slots = hard_floor.value_or(1U);
    generated.hard_floor = hard_floor_slots;
    for (const auto& contributor : plan_->hard_floor_contributors) {
      if (contributor.edge != edge.edge) {
        continue;
      }
      if (generated.hard_floor_contributor_count == 0U) {
        generated.first_hard_floor_contributor = contributor.id;
      }
      ++generated.hard_floor_contributor_count;
    }
    if (!edge.source || edge.source.value() > slot.views.size() || !edge.target ||
        edge.target.value() > slot.views.size()) {
      fail_candidate_locked(slot,
                            rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_PLAN_INVALID"));
      return make_unexpected(slot.failure);
    }
    const auto& source = slot.views[static_cast<std::size_t>(edge.source.value() - 1U)];
    const auto& target = slot.views[static_cast<std::size_t>(edge.target.value() - 1U)];
    const bool input_expectation_is_sizing_authority =
        !source.resolved_nominal_rate &&
        target.origin == ResolvedRateOrigin::kInputExpectationFallback;
    const PortId sizing_port = input_expectation_is_sizing_authority ? edge.target : edge.source;
    const auto& sizing = input_expectation_is_sizing_authority ? target : source;
    generated.producer = sizing_port;
    if (sizing.state == PortRateViewState::kPendingBarrier) {
      continue;
    }
    const auto horizon = std::ranges::find(
        plan_->accumulation_horizons, edge.edge, &AccumulationHorizonPlanRecord::edge);
    std::optional<ExactDuration> evaluated_horizon;
    if (horizon != plan_->accumulation_horizons.end()) {
      generated.accumulation_horizon = horizon->id;
      if (horizon->horizon) {
        evaluated_horizon = horizon->horizon;
      } else if (horizon->inverse_rate_port) {
        const PortId inverse_rate_port = horizon->inverse_rate_port.value_or(PortId{});
        if (!inverse_rate_port || inverse_rate_port.value() > slot.views.size()) {
          fail_candidate_locked(slot,
                                rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_PLAN_INVALID"));
          return make_unexpected(slot.failure);
        }
        const auto& driver = slot.views[static_cast<std::size_t>(inverse_rate_port.value() - 1U)];
        if (driver.state == PortRateViewState::kPendingBarrier) {
          continue;
        }
        if (driver.resolved_nominal_rate) {
          evaluated_horizon = rate_math::inverse_horizon(*driver.resolved_nominal_rate);
          if (!evaluated_horizon) {
            fail_candidate_locked(
                slot, rate_error(ErrorCode::kResourceExhausted, "QUEUE_CAPACITY_OVERFLOW"));
            return make_unexpected(slot.failure);
          }
        }
      }
    }
    if (sizing.sizing_rate && evaluated_horizon) {
      generated.nominal_arrivals =
          rate_math::ceil_arrivals(*sizing.sizing_rate, *evaluated_horizon);
      generated.rate_recommendation =
          generated.nominal_arrivals
              ? rate_math::apply_headroom(*generated.nominal_arrivals, recipe->headroom_ratio)
              : std::nullopt;
      if (!generated.rate_recommendation) {
        fail_candidate_locked(slot,
                              rate_error(ErrorCode::kResourceExhausted, "QUEUE_CAPACITY_OVERFLOW"));
        return make_unexpected(slot.failure);
      }
    }

    if (recipe->explicit_depth) {
      const auto& source_rate =
          plan_->port_rates[static_cast<std::size_t>(edge.source.value() - 1U)];
      const auto& target_rate =
          plan_->port_rates[static_cast<std::size_t>(edge.target.value() - 1U)];
      const bool rate_aware_authority = source_rate.kind != PortRateKind::kUnspecified ||
                                        target_rate.kind != PortRateKind::kUnspecified;
      if (recipe->explicit_depth.value_or(hard_floor_slots) < hard_floor_slots &&
          rate_aware_authority) {
        fail_candidate_locked(
            slot, rate_error(ErrorCode::kInvalidArgument, "QUEUE_DEPTH_BELOW_HARD_MINIMUM"));
        return make_unexpected(slot.failure);
      }
      generated.active_logical_depth = recipe->explicit_depth.value_or(hard_floor_slots);
    } else if (generated.rate_recommendation) {
      generated.active_logical_depth =
          std::max(hard_floor_slots, generated.rate_recommendation.value_or(hard_floor_slots));
    } else {
      generated.active_logical_depth = hard_floor_slots;
    }
    if (recipe->geometry_mode == AllocationGeometryMode::kCompileConcrete &&
        edge.queue_depth != 0U) {
      generated.admitted_depth_ceiling = edge.queue_depth;
    }
    if (recipe->geometry_mode == AllocationGeometryMode::kCompileConcrete &&
        generated.active_logical_depth > edge.queue_depth && recipe->latest_binding_stage &&
        edge.queue_depth != 0U) {
      fail_candidate_locked(
          slot, rate_error(ErrorCode::kResourceExhausted, "QUEUE_PREALLOCATION_BOUND_REQUIRED"));
      return make_unexpected(slot.failure);
    }

    if (const auto capacity = advancement_capacity(
            edge.target, generated.active_logical_depth, plan_->input_selections);
        generated.nominal_arrivals && capacity) {
      if (*generated.nominal_arrivals <= *capacity) {
        generated.stability = StabilityClass::kStableNominalEstimate;
      } else if (edge.queue_policy == QueuePolicy::kDropOldest) {
        generated.stability = StabilityClass::kPredictedReplacementOrDrop;
        generated.stability_reason = diagnostic_id("RATE_PREDICTED_REPLACEMENT");
      } else {
        generated.stability = StabilityClass::kPredictedSustainedBlocking;
        generated.stability_reason = diagnostic_id("RATE_SUSTAINED_BACKPRESSURE");
      }
    }
    if (local_edges_[index] != 0U && recipe->latest_binding_stage && !recipe->explicit_depth) {
      if (slot.predecessor_slot == kInvalidSlot) {
        generated.active_depth_update_required = true;
      } else {
        if (slot.predecessor_slot >= slots_.size() ||
            slots_[slot.predecessor_slot]->queue_depths.size() != slot.queue_depths.size()) {
          fail_candidate_locked(slot,
                                rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_PLAN_INVALID"));
          return make_unexpected(slot.failure);
        }
        const auto& predecessor = slots_[slot.predecessor_slot]->queue_depths[index];
        if (predecessor.edge != generated.edge || predecessor.active_logical_depth == 0U) {
          fail_candidate_locked(slot,
                                rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_PLAN_INVALID"));
          return make_unexpected(slot.failure);
        }
        generated.active_depth_update_required =
            predecessor.active_logical_depth != generated.active_logical_depth;
      }
    }
  }
  return {};
}

// Pool evaluation reduces all dependency terms and stages the corresponding allocation geometry
// in one lock-held transaction. Separating it would obscure the shared rollback boundary.
expected<void, Error> RuntimeRateState::
    evaluate_pool_capacities_locked(  // NOLINT(readability-function-cognitive-complexity)
        GenerationSlot& slot) noexcept {
  const auto fail = [&](const char* token) -> expected<void, Error> {
    fail_candidate_locked(slot, rate_error(ErrorCode::kResourceExhausted, token));
    return make_unexpected(slot.failure);
  };
  const auto queue_depth_for = [&](QueueSizingRecipeId id) -> const RuntimeQueueDepthGeneration* {
    const auto recipe =
        std::ranges::find(plan_->queue_sizing_recipes, id, &QueueSizingRecipePlanRecord::id);
    if (recipe == plan_->queue_sizing_recipes.end()) {
      return nullptr;
    }
    const auto depth =
        std::ranges::find(slot.queue_depths, recipe->edge, &RuntimeQueueDepthGeneration::edge);
    return depth == slot.queue_depths.end() ? nullptr : &*depth;
  };
  const auto dependency_capacity = [&](const PoolCapacityDependencyPlanRecord& dependency)
      -> std::optional<std::pair<std::uint32_t, std::optional<std::uint32_t>>> {
    if (!dependency.id || dependency.id.value > slot.pool_contributions.size()) {
      return std::nullopt;
    }
    std::uint32_t active{};
    std::optional<std::uint32_t> ceiling;
    if (dependency.source == PoolCapacityDependencySource::kQueueSizingRecipe) {
      const auto recipe = std::ranges::find(plan_->queue_sizing_recipes,
                                            dependency.queue_sizing_recipe,
                                            &QueueSizingRecipePlanRecord::id);
      const RuntimeQueueDepthGeneration* const depth =
          queue_depth_for(dependency.queue_sizing_recipe);
      if (recipe == plan_->queue_sizing_recipes.end() || depth == nullptr ||
          depth->active_logical_depth == 0U) {
        return std::nullopt;
      }
      active = depth->active_logical_depth;
      if (recipe->geometry_mode == AllocationGeometryMode::kCompileConcrete) {
        const auto edge =
            std::ranges::find(plan_->payload_edges, recipe->edge, &PayloadEdgePlanRecord::edge);
        if (edge == plan_->payload_edges.end() || edge->queue_depth == 0U) {
          return std::nullopt;
        }
        ceiling = edge->queue_depth;
      }
    } else if (dependency.source == PoolCapacityDependencySource::kPredecessorContributor) {
      const auto predecessor = std::ranges::find(plan_->pool_capacity_contributors,
                                                 dependency.predecessor_contributor_id,
                                                 &PoolCapacityContributorRecord::contributor_id);
      if (predecessor == plan_->pool_capacity_contributors.end() ||
          predecessor->pool_id != dependency.pool_id || predecessor->maximum_distinct_slots == 0U) {
        return std::nullopt;
      }
      active = predecessor->maximum_distinct_slots;
      ceiling = active;
    } else {
      return std::nullopt;
    }
    return std::pair{active, ceiling};
  };

  for (std::size_t provider_index = 0U; provider_index < plan_->provider_allocation_plans.size();
       ++provider_index) {
    const auto& provider = plan_->provider_allocation_plans[provider_index];
    auto& generated = slot.pool_capacities[provider_index];
    generated = RuntimePoolCapacityGeneration{
        .provider = provider.id,
        .run_generation = run_generation_,
        .rate_generation = slot.generation,
        .first_contribution = provider.first_sizing_dependency,
        .contribution_count = provider.sizing_dependency_count,
    };
    if (!provider.first_sizing_dependency || provider.sizing_dependency_count == 0U) {
      return fail("QUEUE_POOL_DEPENDENCY_INVALID");
    }
    const std::size_t first = provider.first_sizing_dependency.value - 1U;
    if (first > plan_->pool_capacity_dependencies.size() ||
        provider.sizing_dependency_count > plan_->pool_capacity_dependencies.size() - first) {
      return fail("QUEUE_POOL_DEPENDENCY_INVALID");
    }
    const auto dependencies = std::span<const PoolCapacityDependencyPlanRecord>{
        plan_->pool_capacity_dependencies.data() + first, provider.sizing_dependency_count};
    bool unresolved{};
    for (std::size_t offset = 0U; offset < dependencies.size(); ++offset) {
      const auto& dependency = dependencies[offset];
      const auto capacity = dependency_capacity(dependency);
      if (dependency.pool_id != provider.pool_id || !capacity) {
        unresolved = true;
        break;
      }
      slot.pool_capacity_terms[offset] =
          PoolCapacityTerm{.scenario = dependency.scenario,
                           .alternative_group = dependency.alternative_group,
                           .active_entries = capacity->first,
                           .admitted_entry_ceiling = capacity->second,
                           .mapping_multiplicity = dependency.mapping_multiplicity};
    }
    if (unresolved) {
      continue;
    }
    const auto terms =
        std::span<const PoolCapacityTerm>{slot.pool_capacity_terms}.first(dependencies.size());
    auto mapped = std::span<MappedPoolCapacityTerm>{slot.mapped_pool_capacity_terms}.first(
        dependencies.size());
    const auto capacity = reduce_pool_capacity(terms, mapped);
    if (!capacity) {
      return fail(capacity.error() == PoolCapacityReductionError::kOverflow
                      ? "QUEUE_POOL_CAPACITY_UNBOUNDED"
                      : "QUEUE_POOL_DEPENDENCY_INVALID");
    }
    for (std::size_t offset = 0U; offset < dependencies.size(); ++offset) {
      const auto& dependency = dependencies[offset];
      slot.pool_contributions[dependency.id.value - 1U] = RuntimePoolCapacityContributionGeneration{
          .dependency = dependency.id,
          .queue_sizing_recipe =
              dependency.source == PoolCapacityDependencySource::kQueueSizingRecipe
                  ? dependency.queue_sizing_recipe
                  : QueueSizingRecipeId{},
          .mapped_entries = mapped[offset].active_entries,
      };
    }
    generated.required_entries = capacity->required_entries;
    generated.admitted_entry_ceiling = capacity->admitted_entry_ceiling;

    if (provider.geometry_mode != AllocationGeometryMode::kLifecycleCandidate ||
        local_providers_[provider_index] == 0U) {
      continue;
    }
    auto allocation = std::ranges::find(
        slot.tensor_allocations, provider.id, &RuntimeTensorAllocationGeneration::provider);
    if (allocation == slot.tensor_allocations.end() || generated.required_entries == 0U ||
        allocation->allocation.bounds.max_byte_span >
            std::numeric_limits<std::uint64_t>::max() / generated.required_entries) {
      return fail("QUEUE_POOL_CAPACITY_UNBOUNDED");
    }
    auto& record = allocation->allocation;
    record.slot_count = generated.required_entries;
    record.aggregate_byte_capacity =
        static_cast<std::uint64_t>(record.slot_count) * record.bounds.max_byte_span;
    record.control_capacity = record.slot_count;
    record.readiness_capacity = record.slot_count;
    record.release_capacity = record.slot_count;
    record.retirement_capacity = record.slot_count;
    record.quarantine_capacity = record.slot_count;
  }
  return {};
}

expected<void, Error> RuntimeRateState::begin_startup_candidate() noexcept {
  const std::lock_guard lock{lifecycle_mutex_};
  auto slot_index = acquire_candidate_slot_locked();
  if (!slot_index) {
    return make_unexpected(std::move(slot_index.error()));
  }
  auto generation = reserve_generation_locked();
  if (!generation) {
    candidate_slot_.reset();
    return make_unexpected(std::move(generation.error()));
  }
  GenerationSlot& slot = *slots_[*slot_index];
  initialize_candidate_locked(slot, *generation);
  for (std::size_t index = 0U; index < slot.provider_decision_required.size(); ++index) {
    slot.provider_decision_required[index] =
        local_providers_[index] != 0U && provider_latest_binding_stages_[index].has_value() ? 1U
                                                                                            : 0U;
  }
  auto resolved = resolve_candidate_locked(slot);
  if (!resolved) {
    reject_candidate_locked(resolved.error(), {});
    return resolved;
  }
  return {};
}

expected<void, Error> RuntimeRateState::begin_restart_candidate(
    std::span<const OperatorInstanceId> members, LifecycleStage from_stage) noexcept {
  const std::lock_guard lock{lifecycle_mutex_};
  const std::size_t published = published_slot_.load(std::memory_order_acquire);
  if (published == kInvalidSlot || published >= slots_.size()) {
    return make_unexpected(rate_error(ErrorCode::kNotReady, "PORT_RATE_PUBLICATION_MISSING"));
  }
  auto slot_index = acquire_candidate_slot_locked();
  if (!slot_index) {
    return make_unexpected(std::move(slot_index.error()));
  }
  auto generation = reserve_generation_locked();
  if (!generation) {
    candidate_slot_.reset();
    return make_unexpected(std::move(generation.error()));
  }
  GenerationSlot& slot = *slots_[*slot_index];
  const GenerationSlot& old = *slots_[published];
  initialize_candidate_locked(slot, *generation);
  slot.predecessor_slot = published;
  for (std::size_t index = 0U; index < slot.provider_decision_required.size(); ++index) {
    if (local_providers_[index] == 0U) {
      continue;
    }
    const auto latest = provider_latest_binding_stages_[index];
    if (!latest || static_cast<std::uint8_t>(from_stage) > static_cast<std::uint8_t>(*latest)) {
      continue;
    }
    const auto& provider = plan_->provider_allocation_plans[index];
    bool owner_is_member = !provider.output;
    if (provider.output && provider.output->value() <= port_owners_.size()) {
      const OperatorInstanceId owner =
          port_owners_[static_cast<std::size_t>(provider.output->value() - 1U)];
      owner_is_member = std::ranges::find(members, owner) != members.end();
    }
    slot.provider_decision_required[index] = owner_is_member ? 1U : 0U;
  }
  slot.bindings = old.bindings;
  slot.closed_stages = old.closed_stages;
  const std::size_t first_stage = static_cast<std::size_t>(from_stage);
  if (first_stage < kForwardStageCount) {
    for (std::size_t stage = first_stage; stage < kForwardStageCount; ++stage) {
      slot.closed_stages[stage] = false;
    }
  }
  for (std::size_t index = 0U; index < plan_->port_rates.size(); ++index) {
    const auto& declaration = plan_->port_rates[index];
    if (declaration.kind != PortRateKind::kLifecycleNominal ||
        !declaration.lifecycle_binding_stage ||
        static_cast<std::size_t>(*declaration.lifecycle_binding_stage) < first_stage) {
      continue;
    }
    if (std::find(members.begin(), members.end(), port_owners_[index]) != members.end()) {
      slot.bindings[index].reset();
    }
  }
  auto resolved = resolve_candidate_locked(slot);
  if (!resolved) {
    reject_candidate_locked(resolved.error(), {});
    return resolved;
  }
  return {};
}

// Provider sealing is one compile-cold transaction over every allocation identity and outcome.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeRateState::seal_provider_records_locked(
    GenerationSlot& slot) noexcept {
  if (slot.provider_records_sealed) {
    return {};
  }
  if (slot.provider_records.size() != plan_->provider_allocation_plans.size() ||
      slot.provider_results.size() != slot.provider_records.size() ||
      slot.pool_capacities.size() != slot.provider_records.size() ||
      published_provider_identities_.size() != slot.provider_records.size()) {
    return make_unexpected(rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_PLAN_INVALID"));
  }
  const std::size_t provider_count = slot.provider_records.size();
  if (provider_count > std::numeric_limits<std::uint64_t>::max() - next_allocation_generation_) {
    return make_unexpected(
        rate_error(ErrorCode::kResourceExhausted, "QUEUE_ALLOCATION_GENERATION_INVALID"));
  }

  for (std::size_t index = 0U; index < provider_count; ++index) {
    const auto& recipe = plan_->provider_allocation_plans[index];
    const auto& capacity = slot.pool_capacities[index];
    auto& result = slot.provider_results[index];
    auto& record = slot.provider_records[index];
    const auto& old_identity = published_provider_identities_[index];

    if (local_providers_[index] == 0U) {
      if (result.reported) {
        return make_unexpected(
            rate_error(ErrorCode::kInvalidArgument, "QUEUE_ALLOCATION_GENERATION_INVALID"));
      }
      record = ProviderAllocationGenerationRecord{};
      continue;
    }
    if (!result.reported) {
      if (slot.provider_decision_required[index] != 0U) {
        return make_unexpected(rate_error(ErrorCode::kNotReady, "RATE_PROVIDER_EVIDENCE_MISSING"));
      }
      record = ProviderAllocationGenerationRecord{};
      continue;
    }
    if (result.provider != recipe.id || !result.materialized_entries ||
        *result.materialized_entries < capacity.required_entries ||
        result.action == AllocationAction::kRejected) {
      return make_unexpected(
          rate_error(ErrorCode::kInvalidArgument, "QUEUE_ALLOCATION_GENERATION_INVALID"));
    }

    record = ProviderAllocationGenerationRecord{
        .id =
            ProviderAllocationGenerationRecordId{rate_generation_record_id(slot.generation, index)},
        .recipe = recipe.id,
        .pool_capacity_generation =
            PoolCapacityGenerationId{rate_generation_record_id(slot.generation, index)},
        .run_generation = run_generation_,
        .rate_generation = slot.generation,
        .previous_decision =
            old_identity.decision
                ? std::optional<ProviderAllocationGenerationRecordId>{old_identity.decision}
                : std::nullopt,
        .previous_allocation_generation =
            old_identity.allocation ? std::optional<AllocationGenerationId>{old_identity.allocation}
                                    : std::nullopt,
        .action = result.action,
        .compatibility_reason = result.compatibility_reason,
        .provider_materialized_entries = result.materialized_entries,
        .provider_stride_bytes_per_entry = result.stride_bytes_per_entry,
        .provider_reserved_bytes_total = result.reserved_bytes_total,
        .disposition = PlanDisposition::kAccepted,
    };
    switch (result.action) {
      case AllocationAction::kFresh:
      case AllocationAction::kReplaced:
        record.allocation_generation = AllocationGenerationId{next_allocation_generation_++};
        break;
      case AllocationAction::kReusedExact:
      case AllocationAction::kReusedLarger:
        if (!old_identity.allocation) {
          return make_unexpected(
              rate_error(ErrorCode::kInvalidArgument, "QUEUE_ALLOCATION_GENERATION_INVALID"));
        }
        record.allocation_generation = old_identity.allocation;
        break;
      case AllocationAction::kRejected:
        return make_unexpected(
            rate_error(ErrorCode::kInvalidArgument, "QUEUE_ALLOCATION_GENERATION_INVALID"));
    }
  }
  slot.provider_records_sealed = true;
  return {};
}

expected<void, Error> RuntimeRateState::bind_nominal_rate(OperatorInstanceId owner,
                                                          LifecycleStage stage,
                                                          LifecycleTransitionPhase phase,
                                                          const void* declaration,
                                                          Rate value) noexcept {
  const std::lock_guard lock{lifecycle_mutex_};
  if (!candidate_slot_ || *candidate_slot_ >= slots_.size()) {
    return make_unexpected(
        rate_error(ErrorCode::kLifecycleRunState, "PORT_RATE_CANDIDATE_MISSING"));
  }
  GenerationSlot& slot = *slots_[*candidate_slot_];
  if (!forward_rate_phase(phase)) {
    return make_unexpected(
        rate_error(ErrorCode::kLifecycleInvalidCallingContext, "PORT_RATE_BIND_NOT_AUTHORIZED"));
  }
  if (slot.failure.code != ErrorCode::kSuccess) {
    return make_unexpected(slot.failure);
  }
  const auto reject = [&](Error error) -> expected<void, Error> {
    fail_candidate_locked(slot, std::move(error));
    return make_unexpected(slot.failure);
  };
  const DeclarationEntry* const entry = find_declaration(declaration);
  if (entry == nullptr || entry->owner != owner || entry->direction != PortDirection::kOutput ||
      static_cast<std::size_t>(stage) >= kForwardStageCount) {
    return reject(rate_error(ErrorCode::kPermissionDenied, "PORT_RATE_BIND_NOT_AUTHORIZED"));
  }
  const std::size_t index = static_cast<std::size_t>(entry->port.value() - 1U);
  const auto& plan = plan_->port_rates[index];
  if (plan.kind != PortRateKind::kLifecycleNominal || !plan.lifecycle_binding_stage ||
      *plan.lifecycle_binding_stage != stage ||
      slot.closed_stages[static_cast<std::size_t>(stage)]) {
    return reject(rate_error(ErrorCode::kPermissionDenied, "PORT_RATE_BIND_NOT_AUTHORIZED"));
  }
  auto normalized = rate_math::normalize(value);
  if (!normalized) {
    return reject(rate_error(ErrorCode::kRateBindingInvalid, "PORT_RATE_BINDING_INVALID"));
  }
  if (plan.declared_max_nominal_rate &&
      !rate_math::less_equal(*normalized, *plan.declared_max_nominal_rate)) {
    return reject(
        rate_error(ErrorCode::kRateBindingAboveMaximum, "PORT_RATE_BINDING_ABOVE_MAXIMUM"));
  }
  if (slot.bindings[index]) {
    if (slot.bindings[index].value_or(normalized.value()) == normalized.value()) {
      return {};
    }
    return reject(rate_error(ErrorCode::kRateBindingConflict, "PORT_RATE_BINDING_CONFLICT"));
  }
  slot.bindings[index] = normalized.value();
  return {};
}

expected<PortRateView, Error> RuntimeRateState::resolve_lifecycle(OperatorInstanceId owner,
                                                                  LifecycleStage,
                                                                  LifecycleTransitionPhase phase,
                                                                  const void* declaration,
                                                                  bool output) noexcept {
  const std::lock_guard lock{lifecycle_mutex_};
  const DeclarationEntry* const entry = find_declaration(declaration);
  if (entry == nullptr || entry->owner != owner ||
      entry->direction != (output ? PortDirection::kOutput : PortDirection::kInput)) {
    return make_unexpected(rate_error(ErrorCode::kInvalidArgument, "PORT_RATE_VIEW_INVALID"));
  }
  std::size_t slot_index = kInvalidSlot;
  if (candidate_slot_ &&
      (forward_rate_phase(phase) || phase == LifecycleTransitionPhase::kForwardRollback)) {
    slot_index = *candidate_slot_;
  } else {
    slot_index = published_slot_.load(std::memory_order_acquire);
  }
  if (slot_index == kInvalidSlot || slot_index >= slots_.size()) {
    return make_unexpected(rate_error(ErrorCode::kNotReady, "PORT_RATE_PUBLICATION_MISSING"));
  }
  return slots_[slot_index]->views[static_cast<std::size_t>(entry->port.value() - 1U)];
}

expected<void, Error> RuntimeRateState::close_stage_barrier(
    LifecycleStage stage, LifecycleTransitionPhase phase) noexcept {
  if (!forward_rate_phase(phase)) {
    return {};
  }
  const std::lock_guard lock{lifecycle_mutex_};
  if (!candidate_slot_ || static_cast<std::size_t>(stage) >= kForwardStageCount) {
    return make_unexpected(
        rate_error(ErrorCode::kLifecycleRunState, "PORT_RATE_CANDIDATE_MISSING"));
  }
  GenerationSlot& slot = *slots_[*candidate_slot_];
  const std::size_t stage_index = static_cast<std::size_t>(stage);
  slot.closed_stages[stage_index] = true;
  for (std::size_t index = 0U; index < plan_->port_rates.size(); ++index) {
    const auto& declaration = plan_->port_rates[index];
    if (declaration.kind == PortRateKind::kLifecycleNominal && local_ports_[index] != 0U &&
        declaration.lifecycle_binding_stage == stage && !slot.bindings[index]) {
      fail_candidate_locked(slot, rate_error(ErrorCode::kNotReady, "PORT_RATE_BINDING_MISSING"));
      return make_unexpected(slot.failure);
    }
  }
  return resolve_candidate_locked(slot);
}

expected<void, Error> RuntimeRateState::preflight_candidate_publication() noexcept {
  const std::lock_guard lock{lifecycle_mutex_};
  if (!candidate_slot_) {
    return make_unexpected(
        rate_error(ErrorCode::kLifecycleRunState, "PORT_RATE_CANDIDATE_MISSING"));
  }
  GenerationSlot& slot = *slots_[*candidate_slot_];
  if (slot.failure.code != ErrorCode::kSuccess) {
    return make_unexpected(slot.failure);
  }
  for (std::size_t index = 0U; index < plan_->port_rates.size(); ++index) {
    if (local_ports_[index] != 0U &&
        plan_->port_rates[index].kind == PortRateKind::kLifecycleNominal &&
        slot.views[index].state != PortRateViewState::kResolved) {
      fail_candidate_locked(slot, rate_error(ErrorCode::kNotReady, "PORT_RATE_BINDING_MISSING"));
      return make_unexpected(slot.failure);
    }
  }
  for (std::size_t index = 0U; index < slot.queue_depths.size(); ++index) {
    const auto& depth = slot.queue_depths[index];
    if (local_edges_[index] != 0U && (!depth.edge || depth.active_logical_depth == 0U)) {
      fail_candidate_locked(slot,
                            rate_error(ErrorCode::kNotReady, "QUEUE_RATE_UNRESOLVED_FALLBACK"));
      return make_unexpected(slot.failure);
    }
  }
  for (std::size_t index = 0U; index < slot.pool_capacities.size(); ++index) {
    const auto& capacity = slot.pool_capacities[index];
    if (local_providers_[index] != 0U && (!capacity.provider || capacity.required_entries == 0U)) {
      fail_candidate_locked(slot,
                            rate_error(ErrorCode::kNotReady, "QUEUE_POOL_CAPACITY_UNRESOLVED"));
      return make_unexpected(slot.failure);
    }
  }
  if (std::ranges::any_of(slot.tensor_allocations, [](const auto& allocation) {
        return !allocation.provider || allocation.allocation.slot_count == 0U;
      })) {
    fail_candidate_locked(slot, rate_error(ErrorCode::kNotReady, "QUEUE_POOL_CAPACITY_UNRESOLVED"));
    return make_unexpected(slot.failure);
  }
  if (auto sealed = seal_provider_records_locked(slot); !sealed) {
    fail_candidate_locked(slot, sealed.error());
    return make_unexpected(slot.failure);
  }
  const std::uint64_t storage_binding_count = static_cast<std::uint64_t>(std::ranges::count_if(
      slot.provider_records, [](const auto& record) { return static_cast<bool>(record.id); }));
  if (storage_binding_count != 0U &&
      (next_storage_binding_generation_ == 0U ||
       storage_binding_count - 1U >
           std::numeric_limits<std::uint64_t>::max() - next_storage_binding_generation_)) {
    fail_candidate_locked(
        slot, rate_error(ErrorCode::kResourceExhausted, "QUEUE_ALLOCATION_GENERATION_INVALID"));
    return make_unexpected(slot.failure);
  }
  slot.preflighted = true;
  return {};
}

expected<void, Error> RuntimeRateState::publish_candidate() noexcept {
  const std::lock_guard lock{lifecycle_mutex_};
  if (!candidate_slot_) {
    return make_unexpected(
        rate_error(ErrorCode::kLifecycleRunState, "PORT_RATE_CANDIDATE_MISSING"));
  }
  GenerationSlot& slot = *slots_[*candidate_slot_];
  if (!slot.preflighted || slot.failure.code != ErrorCode::kSuccess) {
    return make_unexpected(slot.failure.code == ErrorCode::kSuccess
                               ? rate_error(ErrorCode::kNotReady, "PORT_RATE_PUBLICATION_MISSING")
                               : slot.failure);
  }
  // Preflight sealed every record and reserved the complete counter range. This apply pass is
  // intentionally infallible so restart cannot discover a deterministic publication failure after
  // Channel, selector, Tensor, and notification participants have committed.
  for (std::size_t index = 0U; index < slot.provider_records.size(); ++index) {
    auto& record = slot.provider_records[index];
    if (!record.id) {
      continue;
    }
    record.storage_binding_generation =
        StorageBindingGenerationId{next_storage_binding_generation_++};
    // Preflight requires an allocation generation for every provider record carrying an identity.
    published_provider_identities_[index] = ProviderPublishedIdentity{
        .decision = record.id,
        .allocation = *record.allocation_generation,  // NOLINT(bugprone-unchecked-optional-access)
    };
  }
  published_slot_.store(*candidate_slot_, std::memory_order_release);
  candidate_slot_.reset();
  return {};
}

void RuntimeRateStateTestAccess::set_next_storage_binding_generation(RuntimeRateState& state,
                                                                     std::uint64_t next) noexcept {
  const std::lock_guard lock{state.lifecycle_mutex_};
  state.next_storage_binding_generation_ = next;
}

void RuntimeRateState::discard_candidate() noexcept {
  const std::lock_guard lock{lifecycle_mutex_};
  candidate_slot_.reset();
}

void RuntimeRateState::reject_candidate_locked(Error error, DiagnosticSource source) noexcept {
  if (!candidate_slot_ || *candidate_slot_ >= slots_.size()) {
    return;
  }
  GenerationSlot& slot = *slots_[*candidate_slot_];
  if (slot.failure.code == ErrorCode::kSuccess) {
    slot.failure = std::move(error);
  }
  if (slot.failure_source.graph_generation == 0U && slot.failure_source.graph.value == 0U &&
      slot.failure_source.vertex.value == 0U && !slot.failure_source.port) {
    slot.failure_source = source;
  }
  slot.preflighted = false;
  const DiagnosticId rejection = rate_diagnostic_id(slot.failure.message);
  for (std::size_t index = 0U; index < slot.provider_records.size(); ++index) {
    auto& record = slot.provider_records[index];
    const auto& recipe = plan_->provider_allocation_plans[index];
    const auto& old = published_provider_identities_[index];
    const auto& result = slot.provider_results[index];
    if (!result.reported && slot.provider_decision_required[index] == 0U) {
      record = ProviderAllocationGenerationRecord{};
      continue;
    }
    if (!record.id) {
      record = ProviderAllocationGenerationRecord{
          .id = ProviderAllocationGenerationRecordId{rate_generation_record_id(slot.generation,
                                                                               index)},
          .recipe = recipe.id,
          .pool_capacity_generation =
              PoolCapacityGenerationId{rate_generation_record_id(slot.generation, index)},
          .run_generation = run_generation_,
          .rate_generation = slot.generation,
          .previous_decision =
              old.decision ? std::optional<ProviderAllocationGenerationRecordId>{old.decision}
                           : std::nullopt,
          .previous_allocation_generation =
              old.allocation ? std::optional<AllocationGenerationId>{old.allocation} : std::nullopt,
      };
    }
    if (result.reported) {
      record.compatibility_reason = result.compatibility_reason;
      record.provider_materialized_entries = result.materialized_entries;
      record.provider_stride_bytes_per_entry = result.stride_bytes_per_entry;
      record.provider_reserved_bytes_total = result.reserved_bytes_total;
    }
    record.allocation_generation.reset();
    record.storage_binding_generation.reset();
    record.action = AllocationAction::kRejected;
    record.disposition = PlanDisposition::kRejected;
    record.reason = rejection;
  }
  rejected_slot_ = candidate_slot_;
  candidate_slot_.reset();
}

void RuntimeRateState::reject_candidate(Error error, DiagnosticSource source) noexcept {
  const std::lock_guard lock{lifecycle_mutex_};
  reject_candidate_locked(std::move(error), source);
}

// Snapshot assembly intentionally joins all per-generation record families into one owning view.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeRateState::append_snapshot_generation_locked(
    RateGenerationSnapshot& snapshot, const GenerationSlot& slot,
    RateGenerationDisposition disposition) const {
  const bool rejected = disposition == RateGenerationDisposition::kRejected;
  const DiagnosticId generation_reason =
      rejected ? rate_diagnostic_id(slot.failure.message) : DiagnosticId{};
  const auto* const rejection_descriptor =
      rejected ? rate_diagnostic_descriptor(generation_reason) : nullptr;
  const std::string_view rejection_token = rejection_descriptor == nullptr
                                               ? std::string_view{"RATE_GENERATION_REJECTED"}
                                               : rejection_descriptor->token;
  RateGenerationEvidenceRecord generation{
      .run_generation = run_generation_,
      .rate_generation = slot.generation,
      .disposition = disposition,
      .error_code = rejected ? slot.failure.code : ErrorCode::kSuccess,
      .diagnostic = rejected
                        ? Diagnostic{
                              .id = generation_reason,
                              .severity = holoscan::DiagnosticSeverity::kError,
                              .token = std::string{rejection_token},
                              .source = slot.failure_source,
                              .provided = slot.failure.message.empty()
                                              ? std::string{rejection_token}
                                              : slot.failure.message,
                          }
                        : Diagnostic{},
  };

  const std::size_t port_count =
      static_cast<std::size_t>(std::ranges::count(evidence_ports_, std::uint8_t{1U}));
  const auto first_port = std::ranges::find(evidence_ports_, std::uint8_t{1U});
  if (first_port != evidence_ports_.end()) {
    generation.first_port_rate = PortRateGenerationRecordId{rate_generation_record_id(
        slot.generation,
        static_cast<std::size_t>(std::distance(evidence_ports_.begin(), first_port)))};
    generation.port_rate_count = static_cast<std::uint32_t>(port_count);
  }
  snapshot.port_rates_.reserve(snapshot.port_rates_.size() + port_count);
  for (std::size_t index = 0U; index < slot.views.size(); ++index) {
    if (evidence_ports_[index] == 0U) {
      continue;
    }
    const auto& view = slot.views[index];
    const auto& declaration = plan_->port_rates[index];
    const bool propagated = view.origin == ResolvedRateOrigin::kConnectedOutput ||
                            view.origin == ResolvedRateOrigin::kFollowsExpression;
    snapshot.port_rates_.push_back(PortRateGenerationRecord{
        .id = PortRateGenerationRecordId{rate_generation_record_id(slot.generation, index)},
        .port = declaration.port,
        .run_generation = run_generation_,
        .rate_generation = slot.generation,
        .selected_binding_rate = declaration.kind == PortRateKind::kLifecycleNominal
                                     ? slot.bindings[index]
                                     : std::nullopt,
        .propagated_rate = propagated ? view.resolved_nominal_rate : std::nullopt,
        .resolved_rate = view.resolved_nominal_rate,
        .sizing_rate = view.sizing_rate,
        .max_nominal_rate = view.max_nominal_rate,
        .immediate_provenance_port = slot.immediate_provenance[index]
                                         ? std::optional<PortId>{slot.immediate_provenance[index]}
                                         : std::nullopt,
        .root_provenance_port = slot.root_provenance[index]
                                    ? std::optional<PortId>{slot.root_provenance[index]}
                                    : std::nullopt,
        .resolved_origin = view.origin,
        .state = view.state,
        .expectation_comparison = slot.expectation_comparisons[index],
        .disposition = rejected ? PlanDisposition::kRejected : PlanDisposition::kAccepted,
        .reason = generation_reason,
    });
  }

  const std::size_t queue_count =
      static_cast<std::size_t>(std::ranges::count(evidence_queues_, std::uint8_t{1U}));
  const auto first_queue = std::ranges::find(evidence_queues_, std::uint8_t{1U});
  if (first_queue != evidence_queues_.end()) {
    generation.first_queue_sizing = QueueSizingGenerationId{rate_generation_record_id(
        slot.generation,
        static_cast<std::size_t>(std::distance(evidence_queues_.begin(), first_queue)))};
    generation.queue_sizing_count = static_cast<std::uint32_t>(queue_count);
  }
  snapshot.queue_sizings_.reserve(snapshot.queue_sizings_.size() + queue_count);
  for (std::size_t index = 0U; index < slot.queue_depths.size(); ++index) {
    if (evidence_queues_[index] == 0U) {
      continue;
    }
    const auto& queue = slot.queue_depths[index];
    snapshot.queue_sizings_.push_back(QueueSizingGenerationRecord{
        .id = QueueSizingGenerationId{rate_generation_record_id(slot.generation, index)},
        .recipe = queue.recipe,
        .run_generation = run_generation_,
        .rate_generation = slot.generation,
        .producer_rate =
            queue.producer
                ? std::optional<PortRateGenerationRecordId>{PortRateGenerationRecordId{
                      rate_generation_record_id(slot.generation, queue.producer.value() - 1U)}}
                : std::nullopt,
        .accumulation_horizon = queue.accumulation_horizon
                                    ? std::optional<HorizonRecordId>{queue.accumulation_horizon}
                                    : std::nullopt,
        .first_hard_floor_contributor = queue.first_hard_floor_contributor,
        .hard_floor_contributor_count = queue.hard_floor_contributor_count,
        .hard_floor = queue.hard_floor,
        .nominal_arrivals = queue.nominal_arrivals,
        .rate_recommendation = queue.rate_recommendation,
        .admitted_depth_ceiling = queue.admitted_depth_ceiling,
        .stability = queue.stability,
        .stability_reason = queue.stability_reason,
        .effective_logical_depth = queue.active_logical_depth,
        .disposition = rejected ? PlanDisposition::kRejected : PlanDisposition::kAccepted,
        .reason = generation_reason,
    });
  }

  const std::size_t contribution_count =
      static_cast<std::size_t>(std::ranges::count(local_dependencies_, std::uint8_t{1U}));
  const auto first_contribution = std::ranges::find(local_dependencies_, std::uint8_t{1U});
  if (first_contribution != local_dependencies_.end()) {
    generation.first_pool_contribution = PoolCapacityContributionGenerationId{
        rate_generation_record_id(slot.generation,
                                  static_cast<std::size_t>(std::distance(
                                      local_dependencies_.begin(), first_contribution)))};
    generation.pool_contribution_count = static_cast<std::uint32_t>(contribution_count);
  }
  snapshot.pool_contributions_.reserve(snapshot.pool_contributions_.size() + contribution_count);
  for (std::size_t index = 0U; index < slot.pool_contributions.size(); ++index) {
    if (local_dependencies_[index] == 0U) {
      continue;
    }
    const auto& contribution = slot.pool_contributions[index];
    std::optional<QueueSizingGenerationId> queue_generation;
    if (contribution.queue_sizing_recipe) {
      const auto recipe = std::ranges::find(plan_->queue_sizing_recipes,
                                            contribution.queue_sizing_recipe,
                                            &QueueSizingRecipePlanRecord::id);
      if (recipe != plan_->queue_sizing_recipes.end()) {
        const auto queue =
            std::ranges::find(slot.queue_depths, recipe->edge, &RuntimeQueueDepthGeneration::edge);
        if (queue != slot.queue_depths.end()) {
          queue_generation = QueueSizingGenerationId{rate_generation_record_id(
              slot.generation,
              static_cast<std::size_t>(std::distance(slot.queue_depths.begin(), queue)))};
        }
      }
    }
    snapshot.pool_contributions_.push_back(PoolCapacityContributionGenerationRecord{
        .id =
            PoolCapacityContributionGenerationId{rate_generation_record_id(slot.generation, index)},
        .dependency = contribution.dependency,
        .edge_sizing_generation = queue_generation,
        .mapped_entries = contribution.mapped_entries,
        .disposition = rejected ? PlanDisposition::kRejected : PlanDisposition::kAccepted,
        .reason = generation_reason,
    });
  }

  const std::size_t pool_count =
      static_cast<std::size_t>(std::ranges::count(local_providers_, std::uint8_t{1U}));
  const auto first_pool = std::ranges::find(local_providers_, std::uint8_t{1U});
  if (first_pool != local_providers_.end()) {
    generation.first_pool_capacity = PoolCapacityGenerationId{rate_generation_record_id(
        slot.generation,
        static_cast<std::size_t>(std::distance(local_providers_.begin(), first_pool)))};
    generation.pool_capacity_count = static_cast<std::uint32_t>(pool_count);
  }
  snapshot.pool_capacities_.reserve(snapshot.pool_capacities_.size() + pool_count);
  for (std::size_t index = 0U; index < slot.pool_capacities.size(); ++index) {
    if (local_providers_[index] == 0U) {
      continue;
    }
    const auto& pool = slot.pool_capacities[index];
    const auto& provider = plan_->provider_allocation_plans[index];
    std::optional<std::uint64_t> bytes_per_slot;
    if (provider.tensor_template) {
      bytes_per_slot = provider.tensor_template->bounds.max_byte_span;
    } else if (provider.output) {
      const auto allocation =
          std::ranges::find_if(plan_->tensor_allocations, [&](const auto& item) {
            return item.output == *provider.output && item.pool_id == provider.pool_id;
          });
      if (allocation != plan_->tensor_allocations.end()) {
        bytes_per_slot = allocation->bounds.max_byte_span;
      } else {
        const auto field =
            std::ranges::find_if(plan_->tensor_field_allocations, [&](const auto& item) {
              return item.output == *provider.output && item.pool_id == provider.pool_id;
            });
        if (field != plan_->tensor_field_allocations.end()) {
          bytes_per_slot = field->bounds.max_byte_span;
        }
      }
    }
    snapshot.pool_capacities_.push_back(PoolCapacityGenerationRecord{
        .id = PoolCapacityGenerationId{rate_generation_record_id(slot.generation, index)},
        .recipe = provider.id,
        .pool_id = provider.pool_id,
        .run_generation = run_generation_,
        .rate_generation = slot.generation,
        .first_contribution = pool.first_contribution
                                  ? PoolCapacityContributionGenerationId{rate_generation_record_id(
                                        slot.generation, pool.first_contribution.value - 1U)}
                                  : PoolCapacityContributionGenerationId{},
        .contribution_count = pool.contribution_count,
        .required_pool_entries = pool.required_entries,
        .admitted_pool_entry_ceiling = pool.admitted_entry_ceiling,
        .max_payload_byte_span_per_slot = bytes_per_slot,
        .logical_payload_capacity_bytes = checked_capacity_bytes(
            bytes_per_slot,
            pool.required_entries == 0U ? std::nullopt
                                        : std::optional<std::uint32_t>{pool.required_entries}),
        .admitted_logical_payload_capacity_ceiling_bytes =
            checked_capacity_bytes(bytes_per_slot, pool.admitted_entry_ceiling),
        .disposition = rejected ? PlanDisposition::kRejected : PlanDisposition::kAccepted,
        .reason = generation_reason,
    });
  }

  const std::size_t provider_count = static_cast<std::size_t>(std::ranges::count_if(
      slot.provider_records, [](const auto& record) { return static_cast<bool>(record.id); }));
  snapshot.provider_allocations_.reserve(snapshot.provider_allocations_.size() + provider_count);
  for (const auto& record : slot.provider_records) {
    if (!record.id) {
      continue;
    }
    if (!generation.first_provider_allocation) {
      generation.first_provider_allocation = record.id;
    }
    snapshot.provider_allocations_.push_back(record);
  }
  generation.provider_allocation_count = static_cast<std::uint32_t>(provider_count);
  snapshot.generations_.push_back(std::move(generation));
  return {};
}

expected<RateGenerationSnapshot, Error> RuntimeRateState::capture_snapshot() const {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    const std::lock_guard lock{lifecycle_mutex_};
    const std::size_t published = published_slot_.load(std::memory_order_acquire);
    const bool has_published = published != kInvalidSlot && published < slots_.size();
    const bool has_rejected = rejected_slot_ && *rejected_slot_ < slots_.size();
    if (!has_published && !has_rejected) {
      return make_unexpected(rate_error(ErrorCode::kNotReady, "PORT_RATE_PUBLICATION_MISSING"));
    }
    RateGenerationSnapshot snapshot;
    snapshot.plan_digest_ = plan_->digest;
    snapshot.plan_generation_ = plan_->plan_generation;
    snapshot.run_generation_ = run_generation_;
    snapshot.generations_.reserve(static_cast<std::size_t>(has_published) +
                                  static_cast<std::size_t>(has_rejected));
    const GenerationSlot* first = has_published ? slots_[published].get() : nullptr;
    RateGenerationDisposition first_disposition = RateGenerationDisposition::kPublished;
    const GenerationSlot* second = has_rejected ? slots_[*rejected_slot_].get() : nullptr;
    RateGenerationDisposition second_disposition = RateGenerationDisposition::kRejected;
    if (first != nullptr && second != nullptr && first->generation > second->generation) {
      std::swap(first, second);
      std::swap(first_disposition, second_disposition);
    }
    if (first != nullptr) {
      if (auto appended = append_snapshot_generation_locked(snapshot, *first, first_disposition);
          !appended) {
        return make_unexpected(std::move(appended.error()));
      }
    }
    if (second != nullptr) {
      if (auto appended = append_snapshot_generation_locked(snapshot, *second, second_disposition);
          !appended) {
        return make_unexpected(std::move(appended.error()));
      }
    }
    return snapshot;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    return make_unexpected(
        rate_error(ErrorCode::kResourceExhausted, "RATE_GENERATION_SNAPSHOT_ALLOCATION_FAILED"));
  } catch (...) {
    return make_unexpected(rate_error(ErrorCode::kFailure, "RATE_GENERATION_SNAPSHOT_FAILED"));
  }
#endif
}

expected<RuntimeRateGenerationPin, Error> RuntimeRateState::pin_published(
    OperatorInstanceId owner) noexcept {
  if (owner == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  for (;;) {
    const std::size_t slot_index = published_slot_.load(std::memory_order_acquire);
    if (slot_index == kInvalidSlot || slot_index >= slots_.size()) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    GenerationSlot& slot = *slots_[slot_index];
    slot.readers.fetch_add(1U, std::memory_order_acq_rel);
    if (published_slot_.load(std::memory_order_acquire) == slot_index) {
      return RuntimeRateGenerationPin{this, slot_index, owner};
    }
    slot.readers.fetch_sub(1U, std::memory_order_release);
  }
}

expected<PortRateView, Error> RuntimeRateState::resolve_pinned(std::size_t slot,
                                                               OperatorInstanceId owner,
                                                               const void* declaration,
                                                               bool output) const noexcept {
  if (slot >= slots_.size()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const DeclarationEntry* const entry = find_declaration(declaration);
  if (entry == nullptr || entry->owner != owner ||
      entry->direction != (output ? PortDirection::kOutput : PortDirection::kInput)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return slots_[slot]->views[static_cast<std::size_t>(entry->port.value() - 1U)];
}

void RuntimeRateState::release_pin(std::size_t slot) noexcept {
  if (slot < slots_.size()) {
    slots_[slot]->readers.fetch_sub(1U, std::memory_order_release);
  }
}

bool RuntimeRateState::requires_barrier(LifecycleStage stage) const noexcept {
  const std::size_t index = static_cast<std::size_t>(stage);
  return index < barrier_stages_.size() && barrier_stages_[index];
}

bool RuntimeRateState::requires_resource_materialization() const noexcept {
  return resource_materialization_required_;
}

bool RuntimeRateState::requires_deferred_resource_materialization() const noexcept {
  return deferred_resource_materialization_required_;
}

expected<void, Error> RuntimeRateState::visit_candidate_queue_depths(
    void* state,
    expected<void, Error> (*visitor)(void*, std::span<const RuntimeQueueDepthGeneration>) noexcept,
    bool require_all_resolved) noexcept {
  if (state == nullptr || visitor == nullptr) {
    return make_unexpected(
        rate_error(ErrorCode::kInvalidArgument, "QUEUE_RATE_MATERIALIZATION_INVALID"));
  }
  struct Invocation {
    void* state{};
    expected<void, Error> (*visitor)(void*,
                                     std::span<const RuntimeQueueDepthGeneration>) noexcept {};
  } invocation{.state = state, .visitor = visitor};
  return visit_candidate_materialization(
      &invocation,
      [](void* opaque, const RuntimeRateMaterializationView& view) noexcept {
        auto& call = *static_cast<Invocation*>(opaque);
        return call.visitor(call.state, view.queue_depths);
      },
      require_all_resolved);
}

expected<void, Error> RuntimeRateState::visit_candidate_materialization(
    void* state,
    expected<void, Error> (*visitor)(void*, const RuntimeRateMaterializationView&) noexcept,
    bool require_all_resolved) noexcept {
  if (state == nullptr || visitor == nullptr) {
    return make_unexpected(
        rate_error(ErrorCode::kInvalidArgument, "QUEUE_RATE_MATERIALIZATION_INVALID"));
  }
  const std::lock_guard lock{lifecycle_mutex_};
  if (!candidate_slot_ || *candidate_slot_ >= slots_.size()) {
    return make_unexpected(
        rate_error(ErrorCode::kLifecycleRunState, "PORT_RATE_CANDIDATE_MISSING"));
  }
  GenerationSlot& slot = *slots_[*candidate_slot_];
  bool unresolved = slot.failure.code != ErrorCode::kSuccess;
  for (std::size_t index = 0U; !unresolved && index < slot.queue_depths.size(); ++index) {
    const auto& depth = slot.queue_depths[index];
    unresolved = local_edges_[index] != 0U &&
                 (!depth.edge || (require_all_resolved && depth.active_logical_depth == 0U));
  }
  for (std::size_t index = 0U; !unresolved && index < slot.pool_capacities.size(); ++index) {
    const auto& capacity = slot.pool_capacities[index];
    unresolved = local_providers_[index] != 0U &&
                 (!capacity.provider || (require_all_resolved && capacity.required_entries == 0U));
  }
  unresolved =
      unresolved || std::ranges::any_of(slot.tensor_allocations, [&](const auto& allocation) {
        return !allocation.provider ||
               (require_all_resolved && allocation.allocation.slot_count == 0U);
      });
  if (unresolved) {
    const Error error = slot.failure.code == ErrorCode::kSuccess
                            ? rate_error(ErrorCode::kNotReady, "QUEUE_RATE_UNRESOLVED_FALLBACK")
                            : slot.failure;
    fail_candidate_locked(slot, error);
    return make_unexpected(slot.failure);
  }
  const RuntimeRateMaterializationView view{
      .queue_depths = slot.queue_depths,
      .pool_capacities = slot.pool_capacities,
      .tensor_allocations = slot.tensor_allocations,
      .provider_results = slot.provider_results,
  };
  auto visited = visitor(state, view);
  if (!visited) {
    fail_candidate_locked(slot, visited.error());
    return visited;
  }
  return {};
}

}  // namespace holoscan::detail
