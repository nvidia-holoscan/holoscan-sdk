/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compiler/cuda_execution_plan_validation.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <set>
#include <string>
#include <string_view>
#include <tuple>

#include "compiler/cuda_service_stream_plan.hpp"
#include "compiler/execution_plan_lookup.hpp"
#include "compiler/execution_plan_state.hpp"

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
      !std::is_sorted(state.tensor_device_resources.begin(),
                      state.tensor_device_resources.end(),
                      canonical_tensor_device_resource_order)) {
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
         (!partition ||
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
