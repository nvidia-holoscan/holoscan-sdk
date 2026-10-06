/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/cuda_execution_plan_validation.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <string_view>
#include <tuple>

#include "compiler/cuda_service_stream_plan.hpp"
#include "compiler/execution_plan_contracts.hpp"
#include "compiler/execution_plan_lookup.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/operator_cuda_device_obligations.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] expected<void, Error> invalid_cuda_plan() {
  return make_unexpected(
      Error{.code = ErrorCode::kInvalidArgument, .message = "CUDA_EXECUTION_PLAN_INVALID"});
}

[[nodiscard]] constexpr bool valid_source(CudaGreenContextAssignmentSource value) noexcept {
  switch (value) {
    case CudaGreenContextAssignmentSource::kDeveloperBinding:
    case CudaGreenContextAssignmentSource::kAutomaticPolicy:
    case CudaGreenContextAssignmentSource::kResidualPolicy:
      return true;
  }
  return false;
}

[[nodiscard]] constexpr bool valid_role(CudaGreenContextResourceRole value) noexcept {
  switch (value) {
    case CudaGreenContextResourceRole::kDedicatedPartition:
    case CudaGreenContextResourceRole::kSharedServiceResidual:
    case CudaGreenContextResourceRole::kReservedResidual:
      return true;
  }
  return false;
}

[[nodiscard]] constexpr bool valid_stream_policy(CudaComputeStreamPolicy value) noexcept {
  switch (value) {
    case CudaComputeStreamPolicy::kPerExecutionPartition:
    case CudaComputeStreamPolicy::kSharedProcessDevice:
      return true;
  }
  return false;
}

[[nodiscard]] constexpr bool valid_pool_policy(
    const CudaGreenContextPoolPlanRecord& pool) noexcept {
  return pool.co_scheduling == CudaGreenContextCoScheduling::kPreserveDeviceAlignment &&
         pool.workqueue_scope == CudaGreenContextWorkqueueScope::kDeviceContext &&
         pool.workqueue_concurrency_limit == 0U &&
         pool.sm_set_guarantee == CudaGreenContextSmSetGuarantee::kDriverDocumented;
}

[[nodiscard]] const ExecutionPartitionRecord* find_partition(const ExecutionPlanState& state,
                                                             ExecutionPartitionId id) noexcept {
  const auto found = std::find_if(state.execution_partitions.begin(),
                                  state.execution_partitions.end(),
                                  [&](const auto& row) { return row.id == id; });
  return found == state.execution_partitions.end() ? nullptr : &*found;
}

}  // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> validate_cuda_execution_plan_records(const ExecutionPlanState& state) {
  // Preserve the exact invalid request for inspection only when compilation already rejected it
  // with the policy diagnostic. Such a plan cannot run; letting the value reach the consistency
  // checks below avoids rewriting authored evidence and grants it no execution authority.
  const bool rejected_unknown_stream_policy =
      !valid_stream_policy(state.cuda_compute_stream_policy) && !state.admitted &&
      std::ranges::any_of(state.diagnostics, [](const auto& diagnostic) {
        return diagnostic.token == "CUDA_STREAM_POLICY_VIOLATION";
      });
  if (!std::is_sorted(state.cuda_green_context_pools.begin(),
                      state.cuda_green_context_pools.end(),
                      canonical_cuda_green_context_pool_order) ||
      !std::is_sorted(state.cuda_green_context_resources.begin(),
                      state.cuda_green_context_resources.end(),
                      canonical_cuda_green_context_resource_order) ||
      !std::is_sorted(state.cuda_green_context_requests.begin(),
                      state.cuda_green_context_requests.end(),
                      canonical_cuda_green_context_request_order) ||
      !std::is_sorted(state.cuda_green_context_assignments.begin(),
                      state.cuda_green_context_assignments.end(),
                      canonical_cuda_green_context_assignment_order) ||
      !std::is_sorted(state.cuda_compute_streams.begin(),
                      state.cuda_compute_streams.end(),
                      canonical_cuda_compute_stream_order) ||
      !std::is_sorted(state.operator_cuda_stream_bindings.begin(),
                      state.operator_cuda_stream_bindings.end(),
                      canonical_operator_cuda_stream_binding_order) ||
      !std::is_sorted(state.tensor_device_resources.begin(),
                      state.tensor_device_resources.end(),
                      canonical_tensor_device_resource_order) ||
      (!valid_stream_policy(state.cuda_compute_stream_policy) && !rejected_unknown_stream_policy)) {
    return invalid_cuda_plan();
  }

  std::set<std::string, std::less<>> pool_ids;
  std::set<std::tuple<std::uint32_t, std::int32_t>> pool_keys;
  std::set<std::string, std::less<>> context_ids;
  std::set<std::string, std::less<>> stream_ids;
  std::set<std::tuple<std::uint32_t, std::int32_t>> tensor_keys;
  std::set<std::tuple<std::uint32_t, std::int32_t, std::uint32_t>> assignment_keys;
  std::set<std::tuple<std::uint32_t, std::int32_t, std::uint32_t>> partition_stream_keys;
  std::set<std::tuple<std::uint32_t, std::int32_t, std::string>> owner_stream_keys;
  std::set<std::uint32_t> fused_edges;
  std::set<std::uint64_t> bound_operators;
  const auto obligations =
      collect_operator_cuda_device_obligations(OperatorCudaDeviceObligationView{
          .ports = state.ports,
          .interface_bindings = state.interface_port_bindings,
          .tensor_allocations = state.tensor_allocations,
          .tensor_field_allocations = state.tensor_field_allocations,
          .process_local_services = state.process_local_channel_services,
          .process_local_subscribers = state.process_local_channel_subscribers,
          .cross_process_services = state.cross_process_channel_services,
          .cross_process_subscribers = state.cross_process_channel_subscribers,
          .tensor_interface_pools = state.tensor_interface_pools,
      });

  const auto release_stream_capacity_matches = [&](PortId output,
                                                   std::optional<std::uint16_t>
                                                       tensor_index,
                                                   ProcessGroupId allocator_process_group,
                                                   DeviceId device,
                                                   std::uint32_t capacity) {
    const auto output_port = std::ranges::find(state.ports, output, &PortPlanRecord::id);
    const VertexId allocating_producer =
        output_port == state.ports.end() ? VertexId{} : output_port->owner;
    const auto expected =
        tensor_output_release_stream_capacity(state.process_local_channel_services,
                                              state.cross_process_channel_services,
                                              state.operator_cuda_stream_bindings,
                                              allocating_producer,
                                              output,
                                              tensor_index,
                                              allocator_process_group,
                                              device);
    return expected && capacity == *expected;
  };
  if (std::ranges::any_of(state.tensor_allocations,
                          [&](const auto& allocation) {
                            return !release_stream_capacity_matches(
                                allocation.output,
                                std::nullopt,
                                allocation.allocator_process_group,
                                allocation.device,
                                allocation.release_stream_capacity);
                          }) ||
      std::ranges::any_of(state.tensor_field_allocations, [&](const auto& allocation) {
        return !release_stream_capacity_matches(allocation.output,
                                                allocation.tensor_index,
                                                allocation.allocator_process_group,
                                                allocation.device,
                                                allocation.release_stream_capacity);
      })) {
    return invalid_cuda_plan();
  }

  for (const auto& tensor : state.tensor_device_resources) {
    if (tensor.plan_generation != state.plan_generation || tensor.run_generation != 0U ||
        !tensor.process_group || !tensor.device.valid() || tensor.allocator_root_id.empty() ||
        tensor.materialize_before != LifecycleStage::kArm || tensor.gpu_resident_backend_selected ||
        !tensor.fallback_rejected ||
        !tensor_keys.emplace(tensor.process_group.value, tensor.device.value).second) {
      return invalid_cuda_plan();
    }
  }

  for (const auto& pool : state.cuda_green_context_pools) {
    if (pool.plan_generation != state.plan_generation || pool.run_generation != 0U ||
        !pool.process_group || !pool.device.valid() || pool.deployment_device_id.empty() ||
        pool.pool_id.empty() || pool.device_sm_count == 0U || pool.min_sm_partition_size == 0U ||
        pool.sm_coscheduled_alignment == 0U || pool.structured_sm_quantum == 0U ||
        pool.max_structured_context_count == 0U ||
        pool.structured_sm_quantum !=
            std::lcm(pool.min_sm_partition_size, pool.sm_coscheduled_alignment) ||
        pool.materialize_before != LifecycleStage::kArm || !pool.fallback_rejected ||
        !valid_pool_policy(pool) || !pool_ids.insert(pool.pool_id).second ||
        !pool_keys.emplace(pool.process_group.value, pool.device.value).second) {
      return invalid_cuda_plan();
    }
  }

  for (const auto& resource : state.cuda_green_context_resources) {
    const auto pool = std::find_if(state.cuda_green_context_pools.begin(),
                                   state.cuda_green_context_pools.end(),
                                   [&](const auto& row) {
                                     return row.process_group == resource.process_group &&
                                            row.device == resource.device &&
                                            row.pool_id == resource.pool_id;
                                   });
    if (resource.plan_generation != state.plan_generation || resource.run_generation != 0U ||
        !resource.process_group || !resource.device.valid() || resource.pool_id.empty() ||
        resource.green_context_id.empty() || resource.accepted_sm_count == 0U ||
        !valid_role(resource.role) || pool == state.cuda_green_context_pools.end() ||
        !context_ids.insert(resource.green_context_id).second) {
      return invalid_cuda_plan();
    }
  }

  for (const auto& request : state.cuda_green_context_requests) {
    const auto* partition = find_partition(state, request.partition);
    if (request.plan_generation != state.plan_generation || request.run_generation != 0U ||
        !valid_source(request.source) ||
        (partition != nullptr && partition->execution_lane != request.execution_lane)) {
      return invalid_cuda_plan();
    }
    if (request.disposition == PlanDisposition::kAccepted) {
      if (request.authored_name.empty() || partition == nullptr || !request.accepted ||
          *request.accepted != request.requested || request.reason) {
        return invalid_cuda_plan();
      }
    } else if (request.disposition == PlanDisposition::kRejected) {
      if (request.accepted || !request.reason) {
        return invalid_cuda_plan();
      }
    } else {
      return invalid_cuda_plan();
    }
  }

  for (const auto& assignment : state.cuda_green_context_assignments) {
    const auto* partition = find_partition(state, assignment.partition);
    const auto pool = std::find_if(state.cuda_green_context_pools.begin(),
                                   state.cuda_green_context_pools.end(),
                                   [&](const auto& row) {
                                     return row.process_group == assignment.process_group &&
                                            row.device == assignment.device &&
                                            row.pool_id == assignment.pool_id;
                                   });
    const auto context = std::find_if(
        state.cuda_green_context_resources.begin(),
        state.cuda_green_context_resources.end(),
        [&](const auto& row) {
          return row.process_group == assignment.process_group && row.device == assignment.device &&
                 row.pool_id == assignment.pool_id &&
                 row.green_context_id == assignment.green_context_id &&
                 row.role == CudaGreenContextResourceRole::kDedicatedPartition;
        });
    const auto request = std::find_if(state.cuda_green_context_requests.begin(),
                                      state.cuda_green_context_requests.end(),
                                      [&](const auto& row) {
                                        return row.partition == assignment.partition &&
                                               row.disposition == PlanDisposition::kAccepted &&
                                               row.accepted &&
                                               row.accepted->device == assignment.device;
                                      });
    if (assignment.plan_generation != state.plan_generation || assignment.run_generation != 0U ||
        !assignment.process_group || !assignment.device.valid() || assignment.pool_id.empty() ||
        assignment.green_context_id.empty() || !valid_source(assignment.source) ||
        partition == nullptr || partition->execution_lane != assignment.execution_lane ||
        pool == state.cuda_green_context_pools.end() ||
        context == state.cuda_green_context_resources.end() ||
        request == state.cuda_green_context_requests.end() ||
        !assignment_keys
             .emplace(assignment.process_group.value,
                      assignment.device.value,
                      assignment.partition.value)
             .second) {
      return invalid_cuda_plan();
    }
  }

  for (const auto& request : state.cuda_green_context_requests) {
    const auto count = std::count_if(state.cuda_green_context_assignments.begin(),
                                     state.cuda_green_context_assignments.end(),
                                     [&](const auto& row) {
                                       return row.partition == request.partition &&
                                              row.device == request.requested.device;
                                     });
    if ((request.disposition == PlanDisposition::kAccepted && count != 1) ||
        (request.disposition == PlanDisposition::kRejected && count != 0)) {
      return invalid_cuda_plan();
    }
  }

  for (const auto& transfer : state.tensor_transfer_plans) {
    const bool owns_copy_service = transfer.disposition == PlanDisposition::kAccepted &&
                                   transfer.requires_exportable_copy_pool;
    if ((!owns_copy_service && !transfer.service_owner_id.empty()) ||
        (owns_copy_service && !state.cuda_green_context_pools.empty() &&
         transfer.service_owner_id.empty())) {
      return invalid_cuda_plan();
    }
  }

  for (const auto& stream : state.cuda_compute_streams) {
    const auto* partition = stream.partition ? find_partition(state, *stream.partition) : nullptr;
    const auto* tensor_resource =
        find_tensor_device_resource(state, stream.process_group, stream.device);
    const auto context =
        stream.green_context_id
            ? std::find_if(state.cuda_green_context_resources.begin(),
                           state.cuda_green_context_resources.end(),
                           [&](const auto& row) {
                             return row.process_group == stream.process_group &&
                                    row.device == stream.device &&
                                    row.green_context_id == *stream.green_context_id;
                           })
            : state.cuda_green_context_resources.end();
    if (stream.plan_generation != state.plan_generation || stream.run_generation != 0U ||
        !stream.process_group || !stream.device.valid() || stream.owner_id.empty() ||
        stream.stream_id.empty() || stream.materialize_before != LifecycleStage::kArm ||
        !stream_ids.insert(stream.stream_id).second ||
        !owner_stream_keys.emplace(stream.process_group.value, stream.device.value, stream.owner_id)
             .second ||
        (!stream.green_context_id &&
         pool_keys.contains(std::tuple{stream.process_group.value, stream.device.value})) ||
        (stream.partition &&
         (partition == nullptr ||
          !partition_stream_keys
               .emplace(stream.process_group.value, stream.device.value, stream.partition->value)
               .second)) ||
        (stream.green_context_id && context == state.cuda_green_context_resources.end())) {
      return invalid_cuda_plan();
    }
    if (stream.partition) {
      if (stream.owner_id != "partition." + std::to_string(stream.partition->value)) {
        return invalid_cuda_plan();
      }
    } else {
      if (tensor_resource == nullptr) {
        return invalid_cuda_plan();
      }
      const bool ordinary_owner =
          !stream.green_context_id && stream.owner_id == tensor_resource->allocator_root_id;
      const bool green_service_owner =
          stream.green_context_id &&
          context->role == CudaGreenContextResourceRole::kSharedServiceResidual &&
          valid_cuda_service_stream_requirement(state, stream);
      if (!ordinary_owner && !green_service_owner) {
        return invalid_cuda_plan();
      }
    }
    if (stream.partition && stream.green_context_id &&
        context->role == CudaGreenContextResourceRole::kDedicatedPartition) {
      const auto assignment =
          std::find_if(state.cuda_green_context_assignments.begin(),
                       state.cuda_green_context_assignments.end(),
                       [&](const auto& row) {
                         return row.process_group == stream.process_group &&
                                row.device == stream.device && row.partition == *stream.partition &&
                                row.green_context_id == *stream.green_context_id;
                       });
      if (assignment == state.cuda_green_context_assignments.end()) {
        return invalid_cuda_plan();
      }
    }
    if (!std::is_sorted(stream.fused_linear_edges.begin(),
                        stream.fused_linear_edges.end(),
                        [](EdgeId left, EdgeId right) { return left.value < right.value; })) {
      return invalid_cuda_plan();
    }
    for (const EdgeId edge : stream.fused_linear_edges) {
      if (!edge || !fused_edges.insert(edge.value).second ||
          std::none_of(state.payload_edges.begin(),
                       state.payload_edges.end(),
                       [&](const auto& row) { return row.edge == edge; })) {
        return invalid_cuda_plan();
      }
    }
  }

  std::map<std::uint64_t, DeviceId> expected_operator_devices;
  std::set<std::uint64_t> conflicting_operator_devices;
  const auto permits_rejected_operator_device_conflict = [&](VertexId vertex) {
    return !state.admitted && std::ranges::any_of(state.diagnostics, [&](const auto& diagnostic) {
      return diagnostic.token == "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID" &&
             diagnostic.source.vertex == vertex &&
             diagnostic.provided.find("reason=operator_device_conflict") != std::string::npos;
    });
  };
  const auto require_operator_device = [&](VertexId vertex, DeviceId device) {
    if (!vertex || !device.valid()) {
      return true;
    }
    if (conflicting_operator_devices.contains(vertex.value)) {
      return false;
    }
    const auto [found, inserted] = expected_operator_devices.emplace(vertex.value, device);
    if (!inserted && found->second != device) {
      expected_operator_devices.erase(found);
      conflicting_operator_devices.insert(vertex.value);
      return false;
    }
    return true;
  };
  for (const auto& obligation : obligations) {
    if (!require_operator_device(obligation.operator_vertex, obligation.device) &&
        !permits_rejected_operator_device_conflict(obligation.operator_vertex)) {
      return invalid_cuda_plan();
    }
  }
  for (const auto& assignment : state.cuda_green_context_assignments) {
    const auto* partition = find_partition(state, assignment.partition);
    if (partition == nullptr) {
      continue;
    }
    for (const VertexId member : partition->members) {
      const auto graph_vertex =
          std::ranges::find(state.graph.vertices(), member, &VertexRecord::id);
      const bool operator_member =
          graph_vertex != state.graph.vertices().end()
              ? graph_vertex->kind == VertexKind::kOperator
              : std::ranges::any_of(state.ports, [&](const auto& port) {
                  return port.owner == member && port.owner_kind == PortOwnerKind::kOperator;
                });
      if (operator_member) {
        if (!require_operator_device(member, assignment.device) &&
            !permits_rejected_operator_device_conflict(member)) {
          return invalid_cuda_plan();
        }
      }
    }
  }
  // CUDA Graph fusion removes the internal channel services and subscribers before artifact
  // validation. Recover the corresponding member obligations from the accepted execution region
  // so those Operators retain their exact stream bindings in the post-fusion plan.
  for (const auto& region : state.cuda_graph_execution_regions) {
    const auto capture = std::ranges::find_if(state.cuda_graph_capture_plans, [&](const auto& row) {
      return row.graph == region.graph && row.region == region.region;
    });
    if (capture == state.cuda_graph_capture_plans.end()) {
      return invalid_cuda_plan();
    }
    for (const auto& member : capture->members) {
      if (!require_operator_device(member.vertex, region.device) &&
          !permits_rejected_operator_device_conflict(member.vertex)) {
        return invalid_cuda_plan();
      }
    }
  }
  if (expected_operator_devices.size() != state.operator_cuda_stream_bindings.size()) {
    return invalid_cuda_plan();
  }

  for (const auto& binding : state.operator_cuda_stream_bindings) {
    const auto stream = std::find_if(
        state.cuda_compute_streams.begin(), state.cuda_compute_streams.end(), [&](const auto& row) {
          return row.stream_id == binding.stream_id;
        });
    const auto* partition = find_partition(state, binding.partition);
    const auto placement =
        std::find_if(state.vertex_process_placements.begin(),
                     state.vertex_process_placements.end(),
                     [&](const auto& row) { return row.vertex == binding.operator_vertex; });
    const auto expected_device = expected_operator_devices.find(binding.operator_vertex.value);
    if (binding.plan_generation != state.plan_generation || binding.run_generation != 0U ||
        !binding.operator_vertex || !binding.process_group || !binding.partition ||
        !binding.device.valid() || binding.stream_id.empty() ||
        !bound_operators.insert(binding.operator_vertex.value).second ||
        expected_device == expected_operator_devices.end() ||
        expected_device->second != binding.device || stream == state.cuda_compute_streams.end() ||
        stream->process_group != binding.process_group || stream->device != binding.device ||
        (!stream->partition && stream->green_context_id) ||
        (stream->partition && stream->partition != binding.partition) || partition == nullptr ||
        std::find(partition->members.begin(), partition->members.end(), binding.operator_vertex) ==
            partition->members.end() ||
        placement == state.vertex_process_placements.end() ||
        placement->process_group != binding.process_group) {
      return invalid_cuda_plan();
    }
    if (!stream->green_context_id &&
        ((state.cuda_compute_stream_policy == CudaComputeStreamPolicy::kSharedProcessDevice &&
          stream->partition) ||
         (state.cuda_compute_stream_policy == CudaComputeStreamPolicy::kPerExecutionPartition &&
          stream->partition != binding.partition))) {
      return invalid_cuda_plan();
    }
  }

  for (const auto& stream : state.cuda_compute_streams) {
    if (stream.partition && !stream.green_context_id &&
        std::ranges::none_of(state.operator_cuda_stream_bindings, [&](const auto& binding) {
          return binding.stream_id == stream.stream_id;
        })) {
      return invalid_cuda_plan();
    }
  }

  for (const auto& stream : state.cuda_compute_streams) {
    for (const EdgeId edge_id : stream.fused_linear_edges) {
      const auto edge = std::find_if(state.payload_edges.begin(),
                                     state.payload_edges.end(),
                                     [&](const auto& row) { return row.edge == edge_id; });
      if (edge == state.payload_edges.end() || !edge->source || !edge->target ||
          edge->source.value() > state.ports.size() || edge->target.value() > state.ports.size()) {
        return invalid_cuda_plan();
      }
      const auto& source_port = state.ports[static_cast<std::size_t>(edge->source.value() - 1U)];
      const auto& target_port = state.ports[static_cast<std::size_t>(edge->target.value() - 1U)];
      const auto source_binding =
          std::find_if(state.operator_cuda_stream_bindings.begin(),
                       state.operator_cuda_stream_bindings.end(),
                       [&](const auto& row) { return row.operator_vertex == source_port.owner; });
      const auto target_binding =
          std::find_if(state.operator_cuda_stream_bindings.begin(),
                       state.operator_cuda_stream_bindings.end(),
                       [&](const auto& row) { return row.operator_vertex == target_port.owner; });
      if (source_binding == state.operator_cuda_stream_bindings.end() ||
          target_binding == state.operator_cuda_stream_bindings.end() ||
          source_binding->stream_id != stream.stream_id ||
          target_binding->stream_id != stream.stream_id) {
        return invalid_cuda_plan();
      }
    }
  }

  for (const auto& pool : state.cuda_green_context_pools) {
    std::uint64_t accounted = pool.unprovisioned_sm_count;
    std::uint64_t provisioned = 0U;
    std::uint32_t resource_count = 0U;
    for (const auto& resource : state.cuda_green_context_resources) {
      if (resource.process_group == pool.process_group && resource.device == pool.device &&
          resource.pool_id == pool.pool_id) {
        accounted += resource.accepted_sm_count;
        provisioned += resource.accepted_sm_count;
        ++resource_count;
      }
    }
    const std::uint64_t structured_sm_capacity =
        static_cast<std::uint64_t>(pool.max_structured_context_count) * pool.structured_sm_quantum;
    const bool whole_device_context = resource_count == 1U && provisioned == pool.device_sm_count &&
                                      pool.unprovisioned_sm_count == 0U;
    if (resource_count == 0U || resource_count > pool.max_structured_context_count ||
        (!whole_device_context && provisioned > structured_sm_capacity) ||
        structured_sm_capacity > pool.device_sm_count || accounted != pool.device_sm_count) {
      return invalid_cuda_plan();
    }
  }
  return {};
}

}  // namespace holoscan::detail
