/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <algorithm>
#include <holoscan/testing/operator_test_adapters.hpp>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include "runtime/runtime_tensor_provenance.hpp"

namespace holoscan::testing::detail {
namespace {
[[noreturn]] void reject(std::string_view reason) {
  throw std::invalid_argument{"operator test Retain requires a proved finite source: " +
                              std::string{reason}};
}
void require_fixed_capacity_recipe(const ExecutionPlan& plan, QueueSizingRecipeId id) {
  const auto recipes = plan.queue_sizing_recipes();
  const auto recipe =
      std::find_if(recipes.begin(), recipes.end(), [&](const auto& row) { return row.id == id; });
  // Compile-concrete can still be a preallocation ceiling for lifecycle-bound active
  // capacity. Its replacement/reuse protocol is outside this fixed-capacity proof.
  if (recipe == recipes.end() || recipe->latest_binding_stage)
    reject("lifecycle-bound pool capacity is not qualified");
}
void prove_pool(const ExecutionPlan& plan, std::string_view pool_id, std::uint32_t slots,
                std::uint64_t records) {
  const auto providers = plan.provider_allocation_plans();
  const auto provider = std::find_if(
      providers.begin(), providers.end(), [&](const auto& row) { return row.pool_id == pool_id; });
  if (provider == providers.end() ||
      provider->geometry_mode != AllocationGeometryMode::kCompileConcrete)
    reject("pool geometry is not fixed at compile time");
  const auto first = provider->first_sizing_dependency.value;
  const auto end = std::uint64_t{first} + provider->sizing_dependency_count;
  std::uint64_t reserve{};
  bool producer_seen{}, retirement_seen{};
  for (const auto& dependency : plan.pool_capacity_dependencies()) {
    if (dependency.id.value < first || dependency.id.value >= end)
      continue;
    if (dependency.pool_id != pool_id)
      reject("inconsistent pool dependency");
    if (dependency.source == PoolCapacityDependencySource::kQueueSizingRecipe)
      require_fixed_capacity_recipe(plan, dependency.queue_sizing_recipe);
    if (dependency.origin != PoolClaimOrigin::kProducerLoan &&
        dependency.origin != PoolClaimOrigin::kRetirement)
      continue;
    if (dependency.source != PoolCapacityDependencySource::kPredecessorContributor ||
        dependency.mapping_multiplicity != 1U)
      reject("unqualified reserve geometry");
    const auto contributors = plan.pool_capacity_contributors();
    const auto contributor =
        std::find_if(contributors.begin(), contributors.end(), [&](const auto& row) {
          return row.pool_id == pool_id &&
                 row.contributor_id == dependency.predecessor_contributor_id;
        });
    if (contributor == contributors.end() ||
        contributor->maximum_distinct_slots > std::numeric_limits<std::uint64_t>::max() - reserve)
      reject("missing or overflowing pool reserve");
    reserve += contributor->maximum_distinct_slots;
    producer_seen |= dependency.origin == PoolClaimOrigin::kProducerLoan;
    retirement_seen |= dependency.origin == PoolClaimOrigin::kRetirement;
  }
  if (!producer_seen || !retirement_seen || reserve > slots || records > slots - reserve)
    reject("whole-run retained records plus native reserves exceed pool capacity");
}
void prove_output_pools(const ExecutionPlan& plan, const RetainRequest& request,
                        RetainPlan& result) {
  if (request.field_ids.size() <= 1U) {
    const auto allocations = plan.tensor_allocations();
    if (allocations.size() != 1U || allocations.front().output != result.output ||
        !plan.tensor_field_allocations().empty())
      reject("raw Tensor pool identity mismatch");
    const auto& allocation = allocations.front();
    prove_pool(plan, allocation.pool_id, allocation.slot_count, result.records);
    result.pool_ids.push_back(allocation.pool_id);
  } else {
    const auto allocations = plan.tensor_field_allocations();
    if (allocations.size() != request.field_ids.size() || !plan.tensor_allocations().empty())
      reject("every schema Tensor field needs its own proved pool");
    for (std::size_t index = 0; index < request.field_ids.size(); ++index) {
      const auto allocation =
          std::find_if(allocations.begin(), allocations.end(), [&](const auto& row) {
            return row.output == result.output && row.field_id == request.field_ids[index] &&
                   row.tensor_index == index;
          });
      if (allocation == allocations.end())
        reject("schema field pool identity mismatch");
      prove_pool(plan, allocation->pool_id, allocation->slot_count, result.records);
      result.pool_ids.push_back(allocation->pool_id);
    }
  }
}
}  // namespace

std::shared_ptr<const RetainPlan> prepare_retention(const ExecutionPlan& plan,
                                                    std::string_view operator_name,
                                                    std::string_view output_name,
                                                    const RetainRequest& request) {
  const auto graph = plan.graph();
  if (graph.vertices().size() != 2U || graph.graphs().size() != 1U ||
      plan.execution_partitions().size() != 1U)
    reject("one source, one collector and one partition are required");
  const auto vertices = graph.vertices();
  const auto vertex = std::find_if(
      vertices.begin(), vertices.end(), [&](const auto& row) { return row.name == operator_name; });
  const auto parameters = plan.operator_parameters();
  const auto owner = std::find_if(parameters.begin(), parameters.end(), [&](const auto& row) {
    const auto path = OperatorPath::parse(row.operator_path);
    return path && path->segments().size() == 1U && path->segments().front() == operator_name;
  });
  if (vertex == vertices.end() || owner == parameters.end())
    reject("operator identity missing");
  const PortPlanRecord* output{};
  for (const auto& port : plan.ports()) {
    if (port.owner != vertex->id)
      continue;
    if (port.direction != PortDirection::kOutput || output != nullptr ||
        port.local_name != output_name)
      reject("exactly one output and no inputs are currently qualified");
    output = &port;
  }
  const auto rules = plan.operator_completion_rules();
  if (output == nullptr || rules.size() != 1U || rules.front().owner != owner->owner ||
      rules.front().kind != OperatorCompletionRuleKind::kComputeCalls ||
      output->max_emits_per_compute == 0U ||
      rules.front().limit > request.capacity / output->max_emits_per_compute)
    reject("native ComputeCalls and output emission bound must fit max_records");
  const auto edges = plan.payload_edges();
  if (edges.size() != 1U || edges.front().source != output->id ||
      edges.front().queue_policy != QueuePolicy::kBlocking || edges.front().fanout != 1U ||
      edges.front().locality != ChannelLocalityKind::kProcessLocal)
    reject("one non-dropping local collector edge is required");
  auto result = std::make_shared<RetainPlan>();
  result->owner = owner->owner;
  result->output = output->id;
  result->compute_calls = rules.front().limit;
  result->records = rules.front().limit * output->max_emits_per_compute;
  prove_output_pools(plan, request, *result);
  return result;
}

expected<void, Error> validate_retained_tensor(const Tensor& tensor, const RetainPlan& plan,
                                               std::size_t field, OperatorRef collector) {
  if (field >= plan.pool_ids.size() || !collector.run_bound() ||
      collector.plan_generation != plan.owner.plan_generation ||
      !holoscan::detail::runtime_tensor_backing_matches(tensor,
                                                        plan.output,
                                                        plan.owner.plan_generation,
                                                        collector.run_generation,
                                                        plan.pool_ids[field])) {
    return make_unexpected(
        Error{ErrorCode::kIncompatible, "operator test Retain backing provenance mismatch"});
  }
  // Complete producer readiness while the Runtime-owned stream is still alive.
  return tensor.synchronize_for_host();
}

bool retained_boundary_completed(const RetainEvidence& evidence,
                                 const RunSession& session) noexcept {
  if (!evidence.plan || evidence.records != evidence.plan->records)
    return false;
  const bool empty = evidence.plan->records == 0U;
  if (empty) {
    if (session.termination() != RunTermination::kCompleted)
      return false;
  } else if (!evidence.boundary_observed || evidence.run == 0U ||
             session.termination() != RunTermination::kStopped) {
    return false;
  }
  for (const auto& progress : session.operator_completion_progress()) {
    if (progress.owner.id == evidence.plan->owner.id &&
        progress.owner.plan_generation == evidence.plan->owner.plan_generation &&
        progress.owner.run_bound() && (empty || progress.owner.run_generation == evidence.run) &&
        progress.kind == OperatorCompletionRuleKind::kComputeCalls &&
        progress.configured_limit == evidence.plan->compute_calls &&
        progress.successful_compute_calls == evidence.plan->compute_calls && progress.limit_reached)
      return true;
  }
  return false;
}
}  // namespace holoscan::testing::detail
