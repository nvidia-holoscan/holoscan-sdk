/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda_green_context_lowering.hpp"

#include <holoscan/core/compile.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "../core/diagnostic_registry.hpp"
#include "compile_internal.hpp"
#include "compiler_diagnostic.hpp"
#include "cuda_execution_plan_validation.hpp"
#include "cuda_service_stream_plan.hpp"
#include "execution_plan_identity.hpp"
#include "execution_plan_state.hpp"
#include "operator_cuda_device_obligations.hpp"
#include "plan_lowering_diagnostic.hpp"
#include "snapshot_lookup.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] std::optional<ProcessGroupId> partition_process_group(
    const ExecutionPartitionRecord& partition, const detail::ExecutionPlanState& state) {
  std::optional<ProcessGroupId> result;
  for (const VertexId member : partition.members) {
    const auto placement =
        std::find_if(state.vertex_process_placements.begin(),
                     state.vertex_process_placements.end(),
                     [&](const auto& candidate) { return candidate.vertex == member; });
    if (placement == state.vertex_process_placements.end() || !placement->process_group) {
      return std::nullopt;
    }
    if (result && *result != placement->process_group) {
      return std::nullopt;
    }
    result = placement->process_group;
  }
  return result;
}

[[nodiscard]] std::optional<detail::CudaGreenContextCapabilityEvidence> green_context_capability(
    const detail::RuntimeCompileEvidenceView& evidence, DeviceId device) {
  const auto first =
      std::find_if(evidence.cuda_green_context_capabilities.begin(),
                   evidence.cuda_green_context_capabilities.end(),
                   [&](const auto& candidate) { return candidate.device == device; });
  if (first != evidence.cuda_green_context_capabilities.end()) {
    // Injected compile evidence is authoritative only when exactly one row names the device.
    // Duplicate rows fail closed; a live driver query is allowed only when no row was supplied.
    if (std::find_if(std::next(first),
                     evidence.cuda_green_context_capabilities.end(),
                     [&](const auto& candidate) { return candidate.device == device; }) !=
        evidence.cuda_green_context_capabilities.end()) {
      return std::nullopt;
    }
    return *first;
  }
  auto queried = query_cuda_green_context_capability(device);
  if (!queried) {
    return std::nullopt;
  }
  return detail::CudaGreenContextCapabilityEvidence{
      .device = queried->device,
      .deployment_device_id = queried->deployment_device_id,
      .total_sm_count = queried->total_sm_count,
      .min_sm_partition_size = queried->min_sm_partition_size,
      .sm_coscheduled_alignment = queried->sm_coscheduled_alignment,
      .max_structured_context_count = queried->max_structured_context_count,
      .evidence_id = "cuda.driver.device_resource/v1",
      .probe_status = PlacementProbeStatus::kPassed,
  };
}

[[nodiscard]] constexpr bool green_context_profile_permitted(Profile profile) noexcept {
  return profile == Profile::kStandard || profile == Profile::kResearch ||
         profile == Profile::kRealTime;
}

[[nodiscard]] bool green_context_partition_profile_permitted(
    const ExecutionPartitionRecord& partition,
    std::span<const std::optional<Profile>> effective_profiles) noexcept {
  return std::all_of(partition.members.begin(), partition.members.end(), [&](VertexId member) {
    const auto profile = detail::effective_profile_for_vertex(effective_profiles, member);
    return profile && green_context_profile_permitted(*profile);
  });
}

}  // namespace

// Green Context admission is cold, read-only, and deterministic over canonical plan/evidence rows.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] CudaGreenContextLoweringResult lower_cuda_green_context_plan(
    const detail::GraphSnapshotData& snapshot, const DeploymentBinding& deployment,
    const detail::RuntimeCompileEvidenceView& evidence,
    std::span<const std::optional<Profile>> effective_profiles,
    std::span<const detail::CudaServiceStreamRequirement> service_stream_requirements,
    const detail::ExecutionPlanState& state) {
  struct Candidate {
    std::size_t request_index{};
    const ExecutionPartitionRecord* partition{};
    ProcessGroupId process_group{};
    detail::CudaGreenContextCapabilityEvidence capability;
  };
  struct NormalizedRequest {
    std::size_t request_index{};
    const PartitionCudaGreenContextBinding* binding{};
    const ExecutionPartitionRecord* partition{};
    bool duplicate{};
  };
  std::vector<PartitionCudaGreenContextBinding> authored(deployment.cuda_green_contexts().begin(),
                                                         deployment.cuda_green_contexts().end());
  std::sort(authored.begin(), authored.end(), [](const auto& left, const auto& right) {
    return std::tie(left.partition_name, left.placement.device.value, left.placement.sm_count) <
           std::tie(right.partition_name, right.placement.device.value, right.placement.sm_count);
  });
  CudaGreenContextLoweringResult result;
  result.requests.reserve(authored.size());
  result.compute_streams = state.cuda_compute_streams;
  result.operator_stream_bindings = state.operator_cuda_stream_bindings;
  const auto operator_device_obligations =
      detail::collect_operator_cuda_device_obligations(detail::OperatorCudaDeviceObligationView{
          .ports = state.ports,
          .interface_bindings = state.interface_port_bindings,
          .tensor_allocations = state.tensor_allocations,
          .tensor_field_allocations = state.tensor_field_allocations,
          .process_local_services = state.process_local_channel_services,
          .process_local_subscribers = state.process_local_channel_subscribers,
          .cross_process_services = state.cross_process_channel_services,
          .cross_process_subscribers = state.cross_process_channel_subscribers,
          .tensor_interface_pools = state.tensor_interface_pools,
          .payload_edges = state.payload_edges,
      });
  // Map each device a partition's Tensor obligations require to the first port requiring it.
  const auto partition_obligation_devices = [&](const ExecutionPartitionRecord& partition) {
    std::map<std::int32_t, PortId> devices;
    for (const auto& obligation : operator_device_obligations) {
      if (obligation.device.valid() &&
          std::ranges::find(partition.members, obligation.operator_vertex) !=
              partition.members.end()) {
        devices.emplace(obligation.device.value, obligation.source_port);
      }
    }
    return devices;
  };
  const auto describe_device_port = [&](const auto& device_port) {
    return std::to_string(device_port.first) + ":" +
           diagnostic_port_label(snapshot, GraphPortId{device_port.second.value()});
  };
  const auto partition_binding_devices = [&](ExecutionPartitionId partition) {
    std::set<std::int32_t> devices;
    for (const auto& binding : result.operator_stream_bindings) {
      if (binding.partition == partition && binding.device.valid()) {
        devices.insert(binding.device.value);
      }
    }
    return devices;
  };
  std::vector<EdgeId> fusion_candidates;
  for (auto& stream : result.compute_streams) {
    fusion_candidates.insert(fusion_candidates.end(),
                             stream.fused_linear_edges.begin(),
                             stream.fused_linear_edges.end());
    stream.fused_linear_edges.clear();
  }
  std::sort(fusion_candidates.begin(), fusion_candidates.end(), [](EdgeId left, EdgeId right) {
    return left.value < right.value;
  });
  fusion_candidates.erase(std::unique(fusion_candidates.begin(), fusion_candidates.end()),
                          fusion_candidates.end());
  std::vector<NormalizedRequest> normalized;
  normalized.reserve(authored.size());
  std::vector<Candidate> candidates;
  candidates.reserve(authored.size());

  const auto reject = [&](std::size_t request_index,
                          std::string_view token,
                          const ExecutionPartitionRecord* partition,
                          std::string required,
                          std::string provided) {
    auto& request = result.requests[request_index];
    request.accepted.reset();
    request.disposition = PlanDisposition::kRejected;
    request.reason = detail::diagnostic_id(token);
    result.errors.push_back(detail::PlanLoweringDiagnostic{
        .token = std::string(token),
        .source = DiagnosticSource{.vertex = partition == nullptr || partition->members.empty()
                                                 ? VertexId{}
                                                 : partition->members.front()},
        .json_pointer = "/cuda_green_context_requests/" + std::to_string(request_index),
        .required = std::move(required),
        .provided = std::move(provided),
    });
  };

  // Phase 1: normalize all authored bindings into canonical request rows without admitting any.
  for (std::size_t index = 0U; index < authored.size(); ++index) {
    const auto& binding = authored[index];
    const auto partition_row = std::find_if(
        state.execution_partitions.begin(), state.execution_partitions.end(), [&](const auto& row) {
          return row.authored_name && *row.authored_name == binding.partition_name;
        });
    const ExecutionPartitionRecord* partition =
        partition_row == state.execution_partitions.end() ? nullptr : &*partition_row;
    result.requests.push_back(CudaGreenContextRequestPlanRecord{
        .plan_generation = state.plan_generation,
        .run_generation = 0U,
        .partition = partition == nullptr ? ExecutionPartitionId{} : partition->id,
        .authored_name = binding.partition_name,
        .execution_lane = partition == nullptr ? 0U : partition->execution_lane,
        .requested = binding.placement,
        .accepted = std::nullopt,
        .source = CudaGreenContextAssignmentSource::kDeveloperBinding,
        .disposition = PlanDisposition::kRejected,
    });
    const std::size_t request_index = result.requests.size() - 1U;
    const bool duplicate =
        (index > 0U && authored[index - 1U].partition_name == binding.partition_name) ||
        (index + 1U < authored.size() &&
         authored[index + 1U].partition_name == binding.partition_name);
    normalized.push_back(NormalizedRequest{
        .request_index = request_index,
        .binding = &binding,
        .partition = partition,
        .duplicate = duplicate,
    });
  }

  // Phase 2: admit normalized requests against immutable profile, placement, and capability
  // evidence. Stable normalized indices are also the final diagnostic JSON-pointer indices.
  for (const auto& normalized_request : normalized) {
    const std::size_t request_index = normalized_request.request_index;
    const auto& binding = *normalized_request.binding;
    const ExecutionPartitionRecord* partition = normalized_request.partition;
    if (binding.partition_name.empty() || binding.partition_name.size() > 255U ||
        normalized_request.duplicate || partition == nullptr) {
      reject(request_index,
             "CUDA_GREEN_CONTEXT_UNKNOWN_OR_DUPLICATE",
             partition,
             "exactly one named manual execution partition",
             binding.partition_name.empty() ? "<empty>" : binding.partition_name);
      continue;
    }
    if (!binding.placement.device.valid() || binding.placement.sm_count == 0U) {
      reject(request_index,
             "CUDA_GREEN_CONTEXT_INVALID_PLACEMENT",
             partition,
             "nonnegative explicit device and positive exact SM count",
             "device=" + std::to_string(binding.placement.device.value) +
                 ",sm_count=" + std::to_string(binding.placement.sm_count));
      continue;
    }
    const auto process_group = partition_process_group(*partition, state);
    if (!process_group) {
      reject(request_index,
             "CUDA_GREEN_CONTEXT_DEVICE_OWNERSHIP_UNPROVEN",
             partition,
             "one owning process group",
             "partition process ownership is unresolved or split");
      continue;
    }
    const auto tensor_devices = partition_obligation_devices(*partition);
    const std::string requested = "partition=" + binding.partition_name +
                                  ",requested=" + std::to_string(binding.placement.device.value);
    if (tensor_devices.size() > 1U) {
      std::string devices;
      for (const auto& device_port : tensor_devices) {
        devices += (devices.empty() ? "" : "|") + describe_device_port(device_port);
      }
      reject(request_index,
             "CUDA_GREEN_CONTEXT_PARTITION_MULTI_DEVICE",
             partition,
             "at most one partition compute device",
             requested + ",tensor_devices=" + devices);
      continue;
    }
    if (!tensor_devices.empty() &&
        tensor_devices.begin()->first != binding.placement.device.value) {
      reject(request_index,
             "CUDA_GREEN_CONTEXT_DEVICE_CONFLICT",
             partition,
             "requested device must equal existing Tensor compute authority",
             requested + ",tensor=" + describe_device_port(*tensor_devices.begin()));
      continue;
    }
    const bool profile_permitted =
        green_context_partition_profile_permitted(*partition, effective_profiles);
    if (!profile_permitted) {
      reject(request_index,
             "CUDA_GREEN_CONTEXT_CAPABILITY_UNSUPPORTED",
             partition,
             "phase-1 profile policy admits only Standard, Research, or RealTime partitions",
             "partition=" + binding.partition_name +
                 ",device=" + std::to_string(binding.placement.device.value) +
                 ",sm_count=" + std::to_string(binding.placement.sm_count));
      continue;
    }
    auto capability = green_context_capability(evidence, binding.placement.device);
    if (!capability || capability->deployment_device_id.empty() ||
        capability->evidence_id.empty() || capability->total_sm_count == 0U ||
        capability->min_sm_partition_size == 0U || capability->sm_coscheduled_alignment == 0U) {
      reject(request_index,
             "CUDA_GREEN_CONTEXT_CAPABILITY_UNSUPPORTED",
             partition,
             "qualified CUDA Green Context device-resource evidence",
             "device=" + std::to_string(binding.placement.device.value));
      result.placement_probes.push_back(PlacementProbePlanRecord{
          .partition = partition->id,
          .probe_id = "cuda.green_context.split",
          .status = PlacementProbeStatus::kRejected,
          .reason = detail::diagnostic_id("CUDA_GREEN_CONTEXT_CAPABILITY_UNSUPPORTED"),
      });
      continue;
    }
    if (capability->probe_status != PlacementProbeStatus::kPassed &&
        capability->probe_status != PlacementProbeStatus::kRuntimeRequired) {
      reject(request_index,
             "CUDA_GREEN_CONTEXT_CAPABILITY_UNSUPPORTED",
             partition,
             "passed local evidence or a versioned deferred target probe",
             "partition=" + binding.partition_name + ",stable_device=" +
                 capability->deployment_device_id + ",evidence=" + capability->evidence_id);
      continue;
    }
    const auto quantum =
        std::lcm(capability->min_sm_partition_size, capability->sm_coscheduled_alignment);
    const std::uint64_t structured_sm_capacity =
        static_cast<std::uint64_t>(capability->max_structured_context_count) * quantum;
    if (quantum == 0U || capability->max_structured_context_count == 0U ||
        structured_sm_capacity > capability->total_sm_count) {
      reject(request_index,
             "CUDA_GREEN_CONTEXT_CAPABILITY_UNSUPPORTED",
             partition,
             "coherent driver-qualified structured CUDA capacity",
             "device=" + std::to_string(binding.placement.device.value) +
                 ",device_total=" + std::to_string(capability->total_sm_count) +
                 ",quantum=" + std::to_string(quantum) +
                 ",max_contexts=" + std::to_string(capability->max_structured_context_count));
      continue;
    }
    if (binding.placement.sm_count < capability->min_sm_partition_size) {
      const auto lower = (binding.placement.sm_count / quantum) * quantum;
      const auto upper = lower + quantum;
      reject(request_index,
             "CUDA_GREEN_CONTEXT_SPLIT_GRANULARITY",
             partition,
             "quantum=" + std::to_string(quantum) +
                 ",device_total=" + std::to_string(capability->total_sm_count) +
                 ",evidence=" + capability->evidence_id,
             "request=" + std::to_string(binding.placement.sm_count) + ",nearest_lower=" +
                 std::to_string(lower) + ",nearest_upper=" + std::to_string(upper));
      continue;
    }
    // Phase 1 has no deployment catalog capable of proving that another process's local ordinal
    // names a different physical GPU. Fail closed on every foreign-process CUDA obligation.
    const bool competing_process =
        std::any_of(state.tensor_device_resources.begin(),
                    state.tensor_device_resources.end(),
                    [&](const auto& resource) { return resource.process_group != *process_group; });
    if (competing_process) {
      reject(request_index,
             "CUDA_GREEN_CONTEXT_DEVICE_OWNERSHIP_UNPROVEN",
             partition,
             "one proven HSDK CUDA-active process group for stable physical device " +
                 capability->deployment_device_id,
             "partition=" + binding.partition_name +
                 ",process_group=" + std::to_string(process_group->value) +
                 ",local_device=" + std::to_string(binding.placement.device.value) +
                 ",another process group has an unmapped CUDA obligation");
      continue;
    }
    const bool gpu_backend_conflict =
        std::any_of(state.tensor_device_resources.begin(),
                    state.tensor_device_resources.end(),
                    [&](const auto& resource) {
                      return resource.process_group == *process_group &&
                             resource.device == binding.placement.device &&
                             resource.gpu_resident_backend_selected;
                    });
    if (gpu_backend_conflict) {
      reject(request_index,
             "CUDA_GREEN_CONTEXT_GPU_BACKEND_CONFLICT",
             partition,
             "host-managed partition execution with an injected Green Context stream",
             "partition=" + binding.partition_name +
                 ",process_group=" + std::to_string(process_group->value) +
                 ",local_device=" + std::to_string(binding.placement.device.value) +
                 ",stable_device=" + capability->deployment_device_id);
      continue;
    }
    result.placement_probes.push_back(PlacementProbePlanRecord{
        .partition = partition->id,
        .probe_id = "cuda.green_context.split",
        .status = capability->probe_status,
        .evidence_id = capability->evidence_id,
    });
    candidates.push_back(Candidate{
        .request_index = request_index,
        .partition = partition,
        .process_group = *process_group,
        .capability = std::move(*capability),
    });
  }

  // Phase 3: admit complete process/device groups and calculate each residual obligation.
  const auto residual_work_required = [&](ProcessGroupId process_group, DeviceId device) {
    if (std::any_of(service_stream_requirements.begin(),
                    service_stream_requirements.end(),
                    [&](const auto& service) {
                      return service.process_group == process_group && service.device == device;
                    })) {
      return true;
    }
    return std::any_of(result.operator_stream_bindings.begin(),
                       result.operator_stream_bindings.end(),
                       [&](const auto& binding) {
                         if (binding.process_group != process_group || binding.device != device) {
                           return false;
                         }
                         return std::none_of(
                             candidates.begin(), candidates.end(), [&](const auto& candidate) {
                               const auto& request = result.requests[candidate.request_index];
                               return candidate.process_group == process_group &&
                                      candidate.partition->id == binding.partition &&
                                      request.requested.device == device;
                             });
                       });
  };

  for (auto& candidate : candidates) {
    auto& request = result.requests[candidate.request_index];
    const bool ambiguous_physical_mapping =
        std::any_of(candidates.begin(), candidates.end(), [&](const auto& peer) {
          const auto& peer_request = result.requests[peer.request_index];
          return &peer != &candidate &&
                 peer.capability.deployment_device_id ==
                     candidate.capability.deployment_device_id &&
                 (peer.process_group != candidate.process_group ||
                  peer_request.requested.device != request.requested.device);
        });
    if (ambiguous_physical_mapping) {
      reject(candidate.request_index,
             "CUDA_GREEN_CONTEXT_DEVICE_OWNERSHIP_UNPROVEN",
             candidate.partition,
             "one process/local-ordinal mapping for stable physical device " +
                 candidate.capability.deployment_device_id,
             "partition=" + request.authored_name +
                 ",process_group=" + std::to_string(candidate.process_group.value) +
                 ",local_device=" + std::to_string(request.requested.device.value));
      continue;
    }
    const auto invalid_residual_participant =
        std::find_if(state.execution_partitions.begin(),
                     state.execution_partitions.end(),
                     [&](const auto& partition) {
                       const auto group = partition_process_group(partition, state);
                       const auto devices = partition_binding_devices(partition.id);
                       const bool dedicated =
                           std::any_of(candidates.begin(), candidates.end(), [&](const auto& peer) {
                             const auto& peer_request = result.requests[peer.request_index];
                             return peer.process_group == candidate.process_group &&
                                    peer_request.requested.device == request.requested.device &&
                                    peer.partition->id == partition.id;
                           });
                       return group && *group == candidate.process_group && !dedicated &&
                              devices.contains(request.requested.device.value) &&
                              (devices.size() > 1U || !green_context_partition_profile_permitted(
                                                          partition, effective_profiles));
                     });
    if (invalid_residual_participant != state.execution_partitions.end()) {
      const auto devices = partition_binding_devices(invalid_residual_participant->id);
      if (devices.size() > 1U) {
        std::string tensor_devices;
        for (const auto& device_port :
             partition_obligation_devices(*invalid_residual_participant)) {
          if (devices.contains(device_port.first)) {
            tensor_devices +=
                (tensor_devices.empty() ? "" : "|") + describe_device_port(device_port);
          }
        }
        reject(candidate.request_index,
               "CUDA_GREEN_CONTEXT_PARTITION_MULTI_DEVICE",
               candidate.partition,
               "every dedicated or residual partition has at most one compute device",
               "residual_partition=" +
                   invalid_residual_participant->authored_name.value_or(
                       std::to_string(invalid_residual_participant->id.value)) +
                   ",tensor_devices=" + tensor_devices);
      } else {
        reject(candidate.request_index,
               "CUDA_GREEN_CONTEXT_CAPABILITY_UNSUPPORTED",
               candidate.partition,
               "phase-1 profile policy admits every dedicated and residual participant",
               "residual_partition=" + std::to_string(invalid_residual_participant->id.value));
      }
      continue;
    }
    std::uint64_t total = 0U;
    std::uint32_t count = 0U;
    for (const auto& peer : candidates) {
      const auto& peer_request = result.requests[peer.request_index];
      if (peer.process_group == candidate.process_group &&
          peer_request.requested.device == request.requested.device) {
        total += peer_request.requested.sm_count;
        ++count;
      }
    }
    const bool residual_required =
        residual_work_required(candidate.process_group, request.requested.device);
    const auto quantum = std::lcm(candidate.capability.min_sm_partition_size,
                                  candidate.capability.sm_coscheduled_alignment);
    const std::uint64_t structured_sm_capacity =
        static_cast<std::uint64_t>(candidate.capability.max_structured_context_count) * quantum;
    const bool whole_device_group =
        count == 1U && !residual_required && total == candidate.capability.total_sm_count;
    if (quantum == 0U) {
      reject(candidate.request_index,
             "CUDA_GREEN_CONTEXT_CAPABILITY_UNSUPPORTED",
             candidate.partition,
             "nonzero driver-qualified structured CUDA quantum",
             "partition=" + request.authored_name + ",quantum=0");
      continue;
    }
    if (!whole_device_group && request.requested.sm_count % quantum != 0U) {
      const auto lower = (request.requested.sm_count / quantum) * quantum;
      const auto upper = lower + quantum;
      reject(candidate.request_index,
             "CUDA_GREEN_CONTEXT_SPLIT_GRANULARITY",
             candidate.partition,
             "quantum=" + std::to_string(quantum) +
                 ",device_total=" + std::to_string(candidate.capability.total_sm_count) +
                 ",evidence=" + candidate.capability.evidence_id,
             "request=" + std::to_string(request.requested.sm_count) + ",nearest_lower=" +
                 std::to_string(lower) + ",nearest_upper=" + std::to_string(upper));
      continue;
    }
    const std::uint64_t admitted_sm_capacity =
        whole_device_group ? candidate.capability.total_sm_count : structured_sm_capacity;
    const std::uint64_t remaining =
        total <= admitted_sm_capacity ? admitted_sm_capacity - total : 0U;
    const bool structured_residual = !whole_device_group && quantum != 0U && remaining >= quantum;
    const std::uint64_t required_total = total + (residual_required ? quantum : 0U);
    if (required_total > admitted_sm_capacity ||
        count + (structured_residual ? 1U : 0U) >
            candidate.capability.max_structured_context_count) {
      reject(
          candidate.request_index,
          "CUDA_GREEN_CONTEXT_SPLIT_OVERCOMMITTED",
          candidate.partition,
          "device_total=" + std::to_string(candidate.capability.total_sm_count) +
              ",structured_capacity=" + std::to_string(admitted_sm_capacity) +
              ",max_contexts=" + std::to_string(candidate.capability.max_structured_context_count),
          "requested_total=" + std::to_string(total) +
              ",required_residual=" + std::to_string(residual_required ? quantum : 0U) +
              ",materialized_contexts=" + std::to_string(count + (structured_residual ? 1U : 0U)));
      continue;
    }
    request.accepted = request.requested;
    request.disposition = PlanDisposition::kAccepted;
    request.reason = {};
    if (partition_binding_devices(candidate.partition->id).empty()) {
      result.warnings.push_back(detail::PlanLoweringDiagnostic{
          .token = "CUDA_GREEN_CONTEXT_INACTIVE_RESERVATION",
          .source = DiagnosticSource{.vertex = candidate.partition->members.empty()
                                                   ? VertexId{}
                                                   : candidate.partition->members.front()},
          .json_pointer = "/cuda_green_context_requests/" + std::to_string(candidate.request_index),
          .required = "intentional exact reservation for opaque stream-directed CUDA work",
          .provided = "partition=" + request.authored_name +
                      ",process_group=" + std::to_string(candidate.process_group.value) +
                      ",local_device=" + std::to_string(request.requested.device.value) +
                      ",stable_device=" + candidate.capability.deployment_device_id +
                      ",sm_count=" + std::to_string(request.requested.sm_count),
      });
    }
    const std::string pool_id = detail::cuda_green_context_pool_id(
        state.plan_generation, candidate.process_group, request.requested.device);
    result.assignments.push_back(CudaGreenContextAssignmentPlanRecord{
        .plan_generation = state.plan_generation,
        .run_generation = 0U,
        .partition = candidate.partition->id,
        .execution_lane = candidate.partition->execution_lane,
        .process_group = candidate.process_group,
        .device = request.requested.device,
        .pool_id = pool_id,
        .green_context_id =
            detail::cuda_green_context_partition_id(pool_id, candidate.partition->id),
        .source = CudaGreenContextAssignmentSource::kDeveloperBinding,
    });
  }

  std::sort(result.assignments.begin(),
            result.assignments.end(),
            detail::canonical_cuda_green_context_assignment_order);
  // Phase 3: realize complete structured-resource families from accepted assignments only.
  std::set<std::pair<std::uint32_t, std::int32_t>> materialized_groups;
  for (const auto& assignment : result.assignments) {
    const auto group_key = std::pair{assignment.process_group.value, assignment.device.value};
    if (!materialized_groups.insert(group_key).second) {
      continue;
    }
    const auto candidate = std::find_if(candidates.begin(), candidates.end(), [&](const auto& row) {
      const auto& request = result.requests[row.request_index];
      return row.process_group == assignment.process_group &&
             request.requested.device == assignment.device;
    });
    if (candidate == candidates.end()) {
      continue;
    }
    const auto quantum = std::lcm(candidate->capability.min_sm_partition_size,
                                  candidate->capability.sm_coscheduled_alignment);
    const auto driver_structured_sm_capacity = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(candidate->capability.max_structured_context_count) * quantum);
    std::uint32_t dedicated_sm_count = 0U;
    for (const auto& peer : result.assignments) {
      if (peer.process_group != assignment.process_group || peer.device != assignment.device) {
        continue;
      }
      const auto request =
          std::find_if(result.requests.begin(), result.requests.end(), [&](const auto& row) {
            return row.partition == peer.partition && row.accepted;
          });
      if (request == result.requests.end()) {
        continue;
      }
      dedicated_sm_count += request->accepted->sm_count;
      result.resources.push_back(CudaGreenContextResourcePlanRecord{
          .plan_generation = state.plan_generation,
          .run_generation = 0U,
          .process_group = peer.process_group,
          .device = peer.device,
          .pool_id = peer.pool_id,
          .green_context_id = peer.green_context_id,
          .accepted_sm_count = request->accepted->sm_count,
          .role = CudaGreenContextResourceRole::kDedicatedPartition,
      });
    }
    const bool whole_device_group =
        std::count_if(result.assignments.begin(),
                      result.assignments.end(),
                      [&](const auto& peer) {
                        return peer.process_group == assignment.process_group &&
                               peer.device == assignment.device;
                      }) == 1 &&
        dedicated_sm_count == candidate->capability.total_sm_count;
    const auto structured_sm_capacity =
        whole_device_group ? candidate->capability.total_sm_count : driver_structured_sm_capacity;
    const auto remaining = structured_sm_capacity - dedicated_sm_count;
    const auto structured_residual = quantum == 0U ? 0U : (remaining / quantum) * quantum;
    const bool service_residual =
        std::any_of(service_stream_requirements.begin(),
                    service_stream_requirements.end(),
                    [&](const auto& service) {
                      return service.process_group == assignment.process_group &&
                             service.device == assignment.device;
                    }) ||
        std::any_of(result.operator_stream_bindings.begin(),
                    result.operator_stream_bindings.end(),
                    [&](const auto& binding) {
                      return binding.process_group == assignment.process_group &&
                             binding.device == assignment.device &&
                             std::none_of(result.assignments.begin(),
                                          result.assignments.end(),
                                          [&](const auto& peer) {
                                            return peer.process_group == assignment.process_group &&
                                                   peer.device == assignment.device &&
                                                   peer.partition == binding.partition;
                                          });
                    });
    const std::string residual_id = detail::cuda_green_context_residual_id(assignment.pool_id);
    if (structured_residual != 0U) {
      result.resources.push_back(CudaGreenContextResourcePlanRecord{
          .plan_generation = state.plan_generation,
          .run_generation = 0U,
          .process_group = assignment.process_group,
          .device = assignment.device,
          .pool_id = assignment.pool_id,
          .green_context_id = residual_id,
          .accepted_sm_count = structured_residual,
          .role = service_residual ? CudaGreenContextResourceRole::kSharedServiceResidual
                                   : CudaGreenContextResourceRole::kReservedResidual,
      });
    }
    result.pools.push_back(CudaGreenContextPoolPlanRecord{
        .plan_generation = state.plan_generation,
        .run_generation = 0U,
        .process_group = assignment.process_group,
        .device = assignment.device,
        .deployment_device_id = candidate->capability.deployment_device_id,
        .pool_id = assignment.pool_id,
        .device_sm_count = candidate->capability.total_sm_count,
        .min_sm_partition_size = candidate->capability.min_sm_partition_size,
        .sm_coscheduled_alignment = candidate->capability.sm_coscheduled_alignment,
        .structured_sm_quantum = quantum,
        .max_structured_context_count = candidate->capability.max_structured_context_count,
        .unprovisioned_sm_count =
            candidate->capability.total_sm_count - dedicated_sm_count - structured_residual,
        .co_scheduling = CudaGreenContextCoScheduling::kPreserveDeviceAlignment,
        .workqueue_scope = CudaGreenContextWorkqueueScope::kDeviceContext,
        .workqueue_concurrency_limit = 0U,
        .sm_set_guarantee = CudaGreenContextSmSetGuarantee::kDriverDocumented,
        .materialize_before = LifecycleStage::kArm,
        .fallback_rejected = true,
    });
  }

  // Phase 4: resolve final streams and bindings only after every structured resource family is
  // complete. Fusion candidates are attached afterward from the final exact binding rows.
  std::set<std::pair<std::uint32_t, std::int32_t>> replaced_stream_groups;
  for (const auto& assignment : result.assignments) {
    const auto group_key = std::pair{assignment.process_group.value, assignment.device.value};
    if (!replaced_stream_groups.insert(group_key).second) {
      continue;
    }
    const std::string residual_id = detail::cuda_green_context_residual_id(assignment.pool_id);
    result.compute_streams.erase(
        std::remove_if(result.compute_streams.begin(),
                       result.compute_streams.end(),
                       [&](const auto& stream) {
                         if (stream.process_group == assignment.process_group &&
                             stream.device == assignment.device && !stream.green_context_id) {
                           return true;
                         }
                         return false;
                       }),
        result.compute_streams.end());
    for (const auto& peer : result.assignments) {
      if (peer.process_group != assignment.process_group || peer.device != assignment.device) {
        continue;
      }
      result.compute_streams.push_back(CudaComputeStreamPlanRecord{
          .plan_generation = state.plan_generation,
          .run_generation = 0U,
          .process_group = peer.process_group,
          .device = peer.device,
          .owner_id = "partition." + std::to_string(peer.partition.value),
          .partition = peer.partition,
          .stream_id = detail::cuda_green_context_stream_id(peer.green_context_id),
          .green_context_id = peer.green_context_id,
          .materialize_before = LifecycleStage::kArm,
      });
    }
    for (const auto& partition : state.execution_partitions) {
      if (std::any_of(result.assignments.begin(), result.assignments.end(), [&](const auto& peer) {
            return peer.process_group == assignment.process_group &&
                   peer.device == assignment.device && peer.partition == partition.id;
          })) {
        continue;
      }
      const auto process_group = partition_process_group(partition, state);
      const auto devices = partition_binding_devices(partition.id);
      if (!process_group || *process_group != assignment.process_group ||
          !devices.contains(assignment.device.value)) {
        continue;
      }
      result.compute_streams.push_back(CudaComputeStreamPlanRecord{
          .plan_generation = state.plan_generation,
          .run_generation = 0U,
          .process_group = assignment.process_group,
          .device = assignment.device,
          .owner_id = "partition." + std::to_string(partition.id.value),
          .partition = partition.id,
          .stream_id = detail::cuda_green_context_partition_stream_id(residual_id, partition.id),
          .green_context_id = residual_id,
          .materialize_before = LifecycleStage::kArm,
      });
    }
    for (const auto& service : service_stream_requirements) {
      if (service.process_group != assignment.process_group ||
          service.device != assignment.device) {
        continue;
      }
      result.compute_streams.push_back(CudaComputeStreamPlanRecord{
          .plan_generation = state.plan_generation,
          .run_generation = 0U,
          .process_group = assignment.process_group,
          .device = assignment.device,
          .owner_id = service.owner_id,
          .partition = std::nullopt,
          .stream_id = detail::cuda_green_context_service_stream_id(residual_id, service.owner_id),
          .green_context_id = residual_id,
          .materialize_before = LifecycleStage::kArm,
      });
    }

    // An explicit Green Context assignment makes every Operator in that partition CUDA-active,
    // including Operators whose ports do not otherwise require Tensor device storage. Publish the
    // final partition stream as their exact binding rather than reconstructing this join at
    // Runtime.
    for (const auto& peer : result.assignments) {
      if (peer.process_group != assignment.process_group || peer.device != assignment.device) {
        continue;
      }
      const auto partition =
          std::find_if(state.execution_partitions.begin(),
                       state.execution_partitions.end(),
                       [&](const auto& row) { return row.id == peer.partition; });
      const auto stream = std::find_if(
          result.compute_streams.begin(), result.compute_streams.end(), [&](const auto& row) {
            return row.process_group == peer.process_group && row.device == peer.device &&
                   row.partition == peer.partition;
          });
      if (partition == state.execution_partitions.end() || stream == result.compute_streams.end()) {
        continue;
      }
      for (const VertexId member : partition->members) {
        const auto vertex = std::find_if(snapshot.vertices.begin(),
                                         snapshot.vertices.end(),
                                         [&](const auto& row) { return row.id == member; });
        if (vertex == snapshot.vertices.end() || vertex->kind != VertexKind::kOperator) {
          continue;
        }
        const auto binding =
            std::find_if(result.operator_stream_bindings.begin(),
                         result.operator_stream_bindings.end(),
                         [&](const auto& row) { return row.operator_vertex == member; });
        if (binding == result.operator_stream_bindings.end()) {
          result.operator_stream_bindings.push_back(OperatorCudaStreamBindingPlanRecord{
              .plan_generation = state.plan_generation,
              .run_generation = 0U,
              .operator_vertex = member,
              .process_group = peer.process_group,
              .partition = peer.partition,
              .device = peer.device,
              .stream_id = stream->stream_id,
          });
        } else {
          binding->process_group = peer.process_group;
          binding->partition = peer.partition;
          binding->device = peer.device;
          binding->stream_id = stream->stream_id;
        }
      }
    }

    for (auto& binding : result.operator_stream_bindings) {
      if (binding.process_group != assignment.process_group ||
          binding.device != assignment.device) {
        continue;
      }
      const auto stream = std::find_if(
          result.compute_streams.begin(), result.compute_streams.end(), [&](const auto& row) {
            return row.process_group == binding.process_group && row.device == binding.device &&
                   row.partition == binding.partition;
          });
      if (stream != result.compute_streams.end()) {
        binding.stream_id = stream->stream_id;
      }
    }
  }

  for (const EdgeId edge_id : fusion_candidates) {
    const auto edge = std::find_if(state.payload_edges.begin(),
                                   state.payload_edges.end(),
                                   [&](const auto& row) { return row.edge == edge_id; });
    if (edge == state.payload_edges.end() || !edge->source || !edge->target) {
      continue;
    }
    const PortRecord* source_port = find_port(snapshot, GraphPortId{edge->source.value()});
    const PortRecord* target_port = find_port(snapshot, GraphPortId{edge->target.value()});
    if (source_port == nullptr || target_port == nullptr) {
      continue;
    }
    const auto source_binding =
        std::find_if(result.operator_stream_bindings.begin(),
                     result.operator_stream_bindings.end(),
                     [&](const auto& row) { return row.operator_vertex == source_port->owner; });
    const auto target_binding =
        std::find_if(result.operator_stream_bindings.begin(),
                     result.operator_stream_bindings.end(),
                     [&](const auto& row) { return row.operator_vertex == target_port->owner; });
    if (source_binding == result.operator_stream_bindings.end() ||
        target_binding == result.operator_stream_bindings.end() ||
        source_binding->stream_id != target_binding->stream_id) {
      continue;
    }
    const auto stream = std::find_if(
        result.compute_streams.begin(), result.compute_streams.end(), [&](const auto& row) {
          return row.stream_id == source_binding->stream_id;
        });
    if (stream != result.compute_streams.end()) {
      stream->fused_linear_edges.push_back(edge_id);
    }
  }
  for (auto& stream : result.compute_streams) {
    std::sort(stream.fused_linear_edges.begin(),
              stream.fused_linear_edges.end(),
              [](EdgeId left, EdgeId right) { return left.value < right.value; });
  }
  std::sort(
      result.pools.begin(), result.pools.end(), detail::canonical_cuda_green_context_pool_order);
  // Requests were inserted from authored bindings sorted by the exact canonical request key.
  // Preserve those indices so every diagnostic pointer continues to name the published row.
  std::sort(result.resources.begin(),
            result.resources.end(),
            detail::canonical_cuda_green_context_resource_order);
  std::sort(result.compute_streams.begin(),
            result.compute_streams.end(),
            detail::canonical_cuda_compute_stream_order);
  std::sort(result.operator_stream_bindings.begin(),
            result.operator_stream_bindings.end(),
            [](const auto& left, const auto& right) {
              return left.operator_vertex.value < right.operator_vertex.value;
            });
  return result;
}

}  // namespace holoscan::detail
