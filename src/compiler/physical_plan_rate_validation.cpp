/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/physical_plan_rate_validation.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "compiler/execution_plan_contracts.hpp"
#include "compiler/physical_plan_lowering_internal.hpp"
#include "compiler/pool_capacity_plan_lowering.hpp"
#include "compiler/snapshot_lookup.hpp"
#include "core/tensor_allocation_record_validation.hpp"

namespace holoscan::detail {
namespace {

constexpr std::string_view kTensorFieldAllocationOrderInvalid{
    "PHYSICAL_TENSOR_FIELD_ALLOCATION_ORDER_INVALID"};
constexpr std::string_view kQueueActiveDepthNotEnforceable{"QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"};
constexpr std::string_view kReadinessInvalid{"PHYSICAL_READINESS_CAPACITY_INSUFFICIENT"};

}  // namespace

void validate_lifecycle_queue_geometry(
    PhysicalPlanLoweringResult& result, const GraphSnapshotData& snapshot,
    std::span<const QueueSizingRecipePlanRecord> queue_sizing_recipes) {
  // Every automatic lifecycle-selected queue needs an independent active logical limit. A finite
  // maximum can preallocate backing but does not make a smaller selected depth enforceable.
  // HoloMQ, cross-process transport, event-wake endpoints, and replacement queues freeze or
  // consume capacity through provider-specific state that the current contract cannot safely
  // return.
  for (const auto& recipe : queue_sizing_recipes) {
    if (recipe.disposition != PlanDisposition::kAccepted || !recipe.latest_binding_stage ||
        recipe.explicit_depth) {
      continue;
    }
    const auto edge =
        std::ranges::find(result.payload_edges, recipe.edge, &PayloadEdgePlanRecord::edge);
    const auto subscriber = std::ranges::find(result.process_local_channel_subscribers,
                                              recipe.edge,
                                              &ProcessLocalChannelSubscriberPlanRecord::edge);
    const ProcessLocalChannelServicePlanRecord* service = nullptr;
    if (subscriber != result.process_local_channel_subscribers.end()) {
      const auto found = std::ranges::find(result.process_local_channel_services,
                                           subscriber->service_id,
                                           &ProcessLocalChannelServicePlanRecord::service_id);
      if (found != result.process_local_channel_services.end()) {
        service = &*found;
      }
    }
    const bool enforceable =
        edge != result.payload_edges.end() &&
        edge->locality == ChannelLocalityKind::kProcessLocal &&
        edge->queue_policy == QueuePolicy::kBlocking &&
        subscriber != result.process_local_channel_subscribers.end() && service != nullptr &&
        service->delivery_kind == ProcessLocalChannelDeliveryKind::kDirectRetained &&
        !service->cross_process_transport && !service->event_wake && !subscriber->event_wake;
    if (enforceable) {
      continue;
    }
    const auto* target = edge == result.payload_edges.end()
                             ? nullptr
                             : find_port(snapshot, GraphPortId{edge->target.value()});
    reject_physical_plan(
        result,
        kQueueActiveDepthNotEnforceable,
        source_for(snapshot,
                   target == nullptr ? VertexId{} : target->owner,
                   edge == result.payload_edges.end() ? PortId{} : edge->target),
        "/queue_sizing_recipes/" + std::to_string(recipe.id.value) + "/geometry_mode",
        "process-local direct-retained blocking queue without event wake",
        service == nullptr
            ? "provider=unresolved"
            : "delivery_kind=" +
                  std::to_string(static_cast<std::uint32_t>(service->delivery_kind)) +
                  ",cross_process=" +
                  std::to_string(static_cast<std::uint32_t>(service->cross_process_transport)) +
                  ",event_wake=" + std::to_string(static_cast<std::uint32_t>(service->event_wake)));
  }
}

void validate_tensor_provider_geometry(
    PhysicalPlanLoweringResult& result, const GraphSnapshotData& snapshot,
    PlanGenerationId plan_generation, std::span<const ResolvedQueueSizing> resolved_queue_sizings) {
  if (!validate_tensor_field_allocation_order(result.tensor_field_allocations)) {
    reject_physical_plan(result,
                         kTensorFieldAllocationOrderInvalid,
                         {},
                         "/tensor_field_allocations",
                         "ascending (output, field_id)",
                         "non-canonical order");
  }
  const auto validate_pool_provider = [&](std::string_view pool_id,
                                          PortId output,
                                          ProviderAllocationKind allocation_kind,
                                          std::optional<std::uint16_t>
                                              field_id,
                                          std::optional<std::uint16_t>
                                              tensor_index,
                                          std::uint32_t slot_count) {
    const auto provider = std::ranges::find(
        result.provider_allocation_plans, pool_id, &ProviderAllocationPlanRecord::pool_id);
    if (provider == result.provider_allocation_plans.end() ||
        std::ranges::count(result.provider_allocation_plans,
                           pool_id,
                           &ProviderAllocationPlanRecord::pool_id) != 1 ||
        provider->allocation_kind != allocation_kind || provider->output != output ||
        provider->tensor_field_id != field_id || provider->tensor_index != tensor_index ||
        provider->interface_topic || !provider->first_sizing_dependency ||
        provider->sizing_dependency_count == 0U ||
        provider->complete_before != LifecycleStage::kAllocate ||
        provider->disposition != PlanDisposition::kAccepted || provider->reason) {
      return false;
    }
    const auto dependencies =
        provider_sizing_dependencies(result.pool_capacity_dependencies, *provider);
    if (!dependencies || std::ranges::count(result.pool_capacity_dependencies,
                                            pool_id,
                                            &PoolCapacityDependencyPlanRecord::pool_id) !=
                             provider->sizing_dependency_count) {
      return false;
    }
    const auto capacity = evaluate_pool_capacity_plan(pool_id,
                                                      result.pool_capacity_dependencies,
                                                      result.pool_capacity_contributors,
                                                      resolved_queue_sizings);
    return capacity && capacity->admitted_entry_ceiling == slot_count;
  };
  for (const auto& tensor : result.tensor_allocations) {
    if (!validate_tensor_allocation_record(tensor, plan_generation) ||
        !validate_pool_provider(tensor.pool_id,
                                tensor.output,
                                ProviderAllocationKind::kTensorOutput,
                                std::nullopt,
                                std::nullopt,
                                tensor.slot_count)) {
      reject_physical_plan(result, kReadinessInvalid, source_for(snapshot, {}, tensor.output));
    }
  }
  for (const auto& tensor : result.tensor_field_allocations) {
    if (!validate_tensor_field_allocation_record(tensor, plan_generation) ||
        !validate_pool_provider(tensor.pool_id,
                                tensor.output,
                                ProviderAllocationKind::kTensorFieldOutput,
                                tensor.field_id,
                                tensor.tensor_index,
                                tensor.slot_count)) {
      reject_physical_plan(
          result,
          kReadinessInvalid,
          source_for(snapshot, {}, tensor.output),
          port_pointer(tensor.output, "/tensor_fields/" + std::to_string(tensor.field_id)));
    }
  }
  for (const auto& tensor : result.tensor_interface_pools) {
    if (tensor.readiness_capacity < tensor.publisher_loan_capacity ||
        tensor.control_capacity < tensor.slot_count ||
        tensor.release_capacity < tensor.slot_count ||
        tensor.retirement_capacity < tensor.slot_count ||
        tensor.quarantine_capacity < tensor.slot_count) {
      reject_physical_plan(result, kReadinessInvalid, source_for(snapshot, {}, tensor.graph_port));
    }
  }
}

}  // namespace holoscan::detail
