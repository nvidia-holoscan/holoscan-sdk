/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/compile.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <holoscan/logger/logger.hpp>

#include "../core/diagnostic_registry.hpp"
#include "../core/graph_compiler_access.hpp"
#include "clock_readiness_lowering.hpp"
#include "closed_island_activation_plan.hpp"
#include "compile_internal.hpp"
#include "compile_platform.hpp"
#include "compiler_diagnostic.hpp"
#include "cuda_execution_plan_validation.hpp"
#include "cuda_graph_region_lowering.hpp"
#include "cuda_service_stream_plan.hpp"
#include "deployment_plan_lowering.hpp"
#include "endpoint_catalog_lowering.hpp"
#include "endpoint_clock_health_lowering.hpp"
#include "execution_plan_contracts.hpp"
#include "execution_plan_identity.hpp"
#include "execution_plan_state.hpp"
#include "inline_partition_execution_plan.hpp"
#include "lifecycle_plan_lowering.hpp"
#include "parameter_identity.hpp"
#include "parameter_plan_lowering.hpp"
#include "partition_plan_lowering.hpp"
#include "partition_runner_plan.hpp"
#include "physical_plan_lowering.hpp"
#include "plan_artifact_serialization.hpp"
#include "plan_generation.hpp"
#include "plan_lowering_diagnostic.hpp"
#include "process_plan_lowering.hpp"
#include "profile_admission.hpp"
#include "profile_manifest_lowering.hpp"
#include "rate_plan_lowering.hpp"
#include "snapshot_lookup.hpp"
#include "temporal_plan_lowering.hpp"
#include "tensor_byte_span_derivation.hpp"
#include "tensor_memory_kind_derivation.hpp"

namespace holoscan {
namespace {

[[nodiscard]] std::uint32_t callback_partition_owner_count(
    const detail::ExecutionPlanState& state) noexcept {
  return static_cast<std::uint32_t>(std::count_if(
      state.execution_partitions.begin(),
      state.execution_partitions.end(),
      [&](const auto& partition) {
        return std::any_of(
            partition.members.begin(), partition.members.end(), [&](VertexId member) {
              return std::any_of(state.executable_factories.operators.begin(),
                                 state.executable_factories.operators.end(),
                                 [&](const auto& factory) { return factory.vertex == member; });
            });
      }));
}

using detail::append_compiler_diagnostic;

void append_error(detail::ExecutionPlanState& state, const Error& error,
                  DiagnosticSource source = {}) {
  const DiagnosticDetail evidence{.operation = DiagnosticOperation::kCompile, .error = error};
  append_compiler_diagnostic(state,
                             error.message.empty() ? error_code_name(error.code) : error.message,
                             source,
                             std::nullopt,
                             {},
                             {},
                             {},
                             &evidence);
}

[[nodiscard]] std::string bounded_placement_value(std::string_view value) {
  constexpr std::size_t kMaximumEvidenceCharacters = 128U;
  if (value.size() <= kMaximumEvidenceCharacters) {
    return std::string(value);
  }
  return "<oversized:" + std::to_string(value.size()) + ">";
}

// Resolve and diagnose the complete cold deployment binding set in one canonical pass.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] std::vector<detail::TensorDeviceBindingRecord> resolve_tensor_device_placements(
    const detail::GraphSnapshotData& snapshot, const DeploymentBinding& deployment,
    detail::ExecutionPlanState& state) {
  struct IndexedPlacement {
    const TensorOutputDevicePlacementRecord* record{};
    std::size_t authored_index{};
  };
  std::vector<IndexedPlacement> placements;
  placements.reserve(deployment.tensor_output_devices().size());
  for (std::size_t index = 0U; index < deployment.tensor_output_devices().size(); ++index) {
    placements.push_back(IndexedPlacement{
        .record = &deployment.tensor_output_devices()[index],
        .authored_index = index,
    });
  }
  std::sort(placements.begin(), placements.end(), [](const auto& left, const auto& right) {
    return std::tie(
               left.record->operator_path, left.record->output_port, left.record->device.value) <
           std::tie(
               right.record->operator_path, right.record->output_port, right.record->device.value);
  });

  std::vector<detail::TensorDeviceBindingRecord> result;
  result.reserve(placements.size());
  std::set<std::uint64_t> selected_ports;
  std::unordered_map<std::string, const VertexRecord*> operator_by_path;
  operator_by_path.reserve(snapshot.vertices.size());
  for (const auto& vertex : snapshot.vertices) {
    if (vertex.kind != VertexKind::kOperator) {
      continue;
    }
    auto operator_path = detail::canonical_operator_path(snapshot, vertex);
    if (!operator_path) {
      continue;
    }
    auto [found, inserted] = operator_by_path.emplace(std::move(*operator_path), &vertex);
    if (!inserted) {
      found->second = nullptr;
    }
  }
  std::vector<std::unordered_map<std::string_view, const PortRecord*>> ports_by_owner(
      snapshot.vertices.size());
  for (const auto& port : snapshot.ports) {
    if (port.owner && port.owner.value <= ports_by_owner.size()) {
      ports_by_owner[static_cast<std::size_t>(port.owner.value - 1U)].emplace(port.name, &port);
    }
  }
  for (const auto& indexed : placements) {
    const auto& placement = *indexed.record;
    const std::string pointer =
        "/deployment/tensor_output_devices/" + std::to_string(indexed.authored_index);
    const auto reject_placement = [&](std::string_view reason, DiagnosticSource source = {}) {
      append_compiler_diagnostic(
          state,
          "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID",
          source,
          std::nullopt,
          pointer,
          "one exact operator-owned Tensor-bearing output and nonnegative device",
          "operator_path=" + bounded_placement_value(placement.operator_path) +
              ",output_port=" + bounded_placement_value(placement.output_port) + ",device=" +
              std::to_string(placement.device.value) + ",reason=" + std::string(reason));
    };

    const auto owner_match = operator_by_path.find(placement.operator_path);
    if (owner_match == operator_by_path.end() || owner_match->second == nullptr) {
      reject_placement("unknown_or_ambiguous_operator");
      continue;
    }
    const VertexRecord* owner = owner_match->second;

    const PortRecord* output = nullptr;
    const PortRecord* named_port = nullptr;
    const auto& owner_ports = ports_by_owner[static_cast<std::size_t>(owner->id.value - 1U)];
    const auto port_match = owner_ports.find(placement.output_port);
    if (port_match != owner_ports.end()) {
      named_port = port_match->second;
      if (named_port->owner_kind == PortOwnerKind::kOperator &&
          named_port->direction == PortDirection::kOutput) {
        output = named_port;
      }
    }
    const DiagnosticSource source{
        .graph_generation = snapshot.generation,
        .graph = owner->graph,
        .vertex = owner->id,
        .port = output != nullptr ? PortId{output->id.value} : PortId{},
    };
    if (output == nullptr) {
      reject_placement(named_port == nullptr ? "unknown_output" : "not_operator_output", source);
      continue;
    }
    const bool tensor_bearing =
        output->payload_identity.kind == PayloadIdentityKind::kTensor ||
        (output->payload_identity.kind == PayloadIdentityKind::kSchema &&
         output->payload_identity.payload_kind == PayloadKind::kDescriptorBacked);
    if (!tensor_bearing) {
      reject_placement("non_tensor_output", source);
      continue;
    }
    if (!placement.device.valid()) {
      reject_placement("invalid_device", source);
      continue;
    }
    if (!selected_ports.insert(output->id.value).second) {
      reject_placement("duplicate_output", source);
      continue;
    }
    result.push_back(detail::TensorDeviceBindingRecord{
        .port = PortId{output->id.value},
        .device = placement.device,
    });
  }
  return result;
}

[[nodiscard]] bool partition_contains(const ExecutionPartitionRecord& partition,
                                      VertexId vertex) noexcept {
  return std::binary_search(partition.members.begin(),
                            partition.members.end(),
                            vertex,
                            [](VertexId left, VertexId right) { return left.value < right.value; });
}

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

// Device projection validates every Tensor storage vocabulary against one partition. Keeping the
// cases together makes omissions visible when a new storage record is added.
[[nodiscard]] std::set<std::int32_t>
partition_tensor_devices(  // NOLINT(readability-function-cognitive-complexity)
    const detail::GraphSnapshotData& snapshot, const ExecutionPartitionRecord& partition,
    const detail::ExecutionPlanState& state) {
  std::set<std::int32_t> devices;
  const auto owns_port = [&](PortId port_id) {
    const PortRecord* port = find_port(snapshot, GraphPortId{port_id.value()});
    return port != nullptr && port->owner_kind == PortOwnerKind::kOperator &&
           partition_contains(partition, port->owner);
  };
  const auto append_device = [&](DeviceId device) {
    if (device.valid()) {
      devices.insert(device.value);
    }
  };
  const auto append_service_devices = [&](const auto& service) {
    detail::for_each_tensor_service_storage(
        detail::tensor_service_contract_view(service),
        [&](detail::TensorServiceStorageView storage) { append_device(storage.device); });
  };
  const auto append_output = [&](PortId output, DeviceId device) {
    if (owns_port(output)) {
      append_device(device);
    }
  };
  for (const auto& allocation : state.tensor_allocations) {
    append_output(allocation.output, allocation.device);
  }
  for (const auto& allocation : state.tensor_field_allocations) {
    append_output(allocation.output, allocation.device);
  }
  for (const auto& pool : state.tensor_interface_pools) {
    bool owned = owns_port(pool.graph_port);
    const PortId graph_port{pool.graph_port.value()};
    for (const auto& edge : state.payload_edges) {
      if ((pool.imported && edge.source == graph_port && owns_port(edge.target)) ||
          (!pool.imported && edge.target == graph_port && owns_port(edge.source))) {
        owned = true;
      }
    }
    if (owned) {
      append_device(pool.device);
    }
  }
  for (const auto& subscriber : state.process_local_channel_subscribers) {
    if (!owns_port(subscriber.target)) {
      continue;
    }
    const auto service = std::find_if(state.process_local_channel_services.begin(),
                                      state.process_local_channel_services.end(),
                                      [&](const auto& row) {
                                        return row.source == subscriber.source &&
                                               row.service_id == subscriber.service_id;
                                      });
    if (service != state.process_local_channel_services.end()) {
      append_service_devices(*service);
    }
  }
  for (const auto& subscriber : state.cross_process_channel_subscribers) {
    if (!owns_port(subscriber.target)) {
      continue;
    }
    const auto service = std::find_if(state.cross_process_channel_services.begin(),
                                      state.cross_process_channel_services.end(),
                                      [&](const auto& row) {
                                        return row.source == subscriber.source &&
                                               row.service_id == subscriber.service_id;
                                      });
    if (service != state.cross_process_channel_services.end()) {
      append_service_devices(*service);
    }
  }
  for (const auto& service : state.cross_process_channel_services) {
    if (detail::descriptor_tensor_schema_service(service) && owns_port(service.source)) {
      append_service_devices(service);
    }
  }
  return devices;
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

struct CudaGreenContextLoweringResult {
  std::vector<CudaGreenContextRequestPlanRecord> requests;
  std::vector<CudaGreenContextAssignmentPlanRecord> assignments;
  std::vector<CudaGreenContextPoolPlanRecord> pools;
  std::vector<CudaGreenContextResourcePlanRecord> resources;
  std::vector<CudaComputeStreamPlanRecord> compute_streams;
  std::vector<PlacementProbePlanRecord> placement_probes;
  std::vector<detail::PlanLoweringDiagnostic> errors;
  std::vector<detail::PlanLoweringDiagnostic> warnings;
};

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
    const auto tensor_devices = partition_tensor_devices(snapshot, *partition, state);
    if (tensor_devices.size() > 1U) {
      reject(request_index,
             "CUDA_GREEN_CONTEXT_PARTITION_MULTI_DEVICE",
             partition,
             "at most one partition compute device",
             "resolved_tensor_devices=" + std::to_string(tensor_devices.size()));
      continue;
    }
    if (!tensor_devices.empty() && *tensor_devices.begin() != binding.placement.device.value) {
      reject(request_index,
             "CUDA_GREEN_CONTEXT_DEVICE_CONFLICT",
             partition,
             "requested device must equal existing Tensor compute authority",
             "requested=" + std::to_string(binding.placement.device.value) +
                 ",tensor=" + std::to_string(*tensor_devices.begin()));
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
    return std::any_of(
        state.execution_partitions.begin(),
        state.execution_partitions.end(),
        [&](const auto& partition) {
          const auto group = partition_process_group(partition, state);
          if (!group || *group != process_group ||
              !partition_tensor_devices(snapshot, partition, state).contains(device.value)) {
            return false;
          }
          return std::none_of(candidates.begin(), candidates.end(), [&](const auto& candidate) {
            const auto& request = result.requests[candidate.request_index];
            return candidate.process_group == process_group &&
                   candidate.partition->id == partition.id && request.requested.device == device;
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
                       const auto devices = partition_tensor_devices(snapshot, partition, state);
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
      const auto devices = partition_tensor_devices(snapshot, *invalid_residual_participant, state);
      if (devices.size() > 1U) {
        reject(candidate.request_index,
               "CUDA_GREEN_CONTEXT_PARTITION_MULTI_DEVICE",
               candidate.partition,
               "every dedicated or residual partition has at most one compute device",
               "residual_partition=" + std::to_string(invalid_residual_participant->id.value) +
                   ",resolved_tensor_devices=" + std::to_string(devices.size()));
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
    if (partition_tensor_devices(snapshot, *candidate.partition, state).empty()) {
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
        std::any_of(state.execution_partitions.begin(),
                    state.execution_partitions.end(),
                    [&](const auto& partition) {
                      const auto group = partition_process_group(partition, state);
                      return group && *group == assignment.process_group &&
                             partition_tensor_devices(snapshot, partition, state)
                                 .contains(assignment.device.value) &&
                             std::none_of(result.assignments.begin(),
                                          result.assignments.end(),
                                          [&](const auto& peer) {
                                            return peer.process_group == assignment.process_group &&
                                                   peer.device == assignment.device &&
                                                   peer.partition == partition.id;
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

  // Phase 4: replace ordinary streams only after every structured resource family is complete.
  std::set<std::pair<std::uint32_t, std::int32_t>> replaced_stream_groups;
  for (const auto& assignment : result.assignments) {
    const auto group_key = std::pair{assignment.process_group.value, assignment.device.value};
    if (!replaced_stream_groups.insert(group_key).second) {
      continue;
    }
    const std::string residual_id = detail::cuda_green_context_residual_id(assignment.pool_id);
    std::vector<EdgeId> fused_edges;
    result.compute_streams.erase(
        std::remove_if(result.compute_streams.begin(),
                       result.compute_streams.end(),
                       [&](const auto& stream) {
                         if (stream.process_group == assignment.process_group &&
                             stream.device == assignment.device && !stream.green_context_id) {
                           fused_edges.insert(fused_edges.end(),
                                              stream.fused_linear_edges.begin(),
                                              stream.fused_linear_edges.end());
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
      const auto devices = partition_tensor_devices(snapshot, partition, state);
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

    for (const EdgeId edge_id : fused_edges) {
      const auto edge = std::find_if(state.payload_edges.begin(),
                                     state.payload_edges.end(),
                                     [&](const auto& row) { return row.edge == edge_id; });
      if (edge == state.payload_edges.end()) {
        continue;
      }
      const PortRecord* source_port = find_port(snapshot, GraphPortId{edge->source.value()});
      const PortRecord* target_port = find_port(snapshot, GraphPortId{edge->target.value()});
      if (source_port == nullptr || target_port == nullptr) {
        continue;
      }
      const auto partition = std::find_if(state.execution_partitions.begin(),
                                          state.execution_partitions.end(),
                                          [&](const auto& row) {
                                            return partition_contains(row, source_port->owner) &&
                                                   partition_contains(row, target_port->owner);
                                          });
      if (partition == state.execution_partitions.end()) {
        continue;
      }
      const auto stream = std::find_if(
          result.compute_streams.begin(), result.compute_streams.end(), [&](const auto& row) {
            return row.process_group == assignment.process_group &&
                   row.device == assignment.device && row.partition == partition->id;
          });
      if (stream != result.compute_streams.end()) {
        stream->fused_linear_edges.push_back(edge_id);
      }
    }
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
  return result;
}

void publish_cuda_green_context_plan(CudaGreenContextLoweringResult&& lowered,
                                     detail::ExecutionPlanState& state) {
  for (const auto& diagnostic : lowered.errors) {
    append_compiler_diagnostic(state,
                               diagnostic.token,
                               diagnostic.source,
                               DiagnosticSeverity::kError,
                               diagnostic.json_pointer,
                               diagnostic.required,
                               diagnostic.provided);
  }
  for (const auto& diagnostic : lowered.warnings) {
    append_compiler_diagnostic(state,
                               diagnostic.token,
                               diagnostic.source,
                               holoscan::DiagnosticSeverity::kWarning,
                               diagnostic.json_pointer,
                               diagnostic.required,
                               diagnostic.provided);
  }
  state.placement_probes.insert(state.placement_probes.end(),
                                std::make_move_iterator(lowered.placement_probes.begin()),
                                std::make_move_iterator(lowered.placement_probes.end()));
  state.cuda_green_context_requests = std::move(lowered.requests);
  state.cuda_green_context_assignments = std::move(lowered.assignments);
  state.cuda_green_context_pools = std::move(lowered.pools);
  state.cuda_green_context_resources = std::move(lowered.resources);
  state.cuda_compute_streams = std::move(lowered.compute_streams);
}

[[nodiscard]] bool partition_environment_selector_present() {
  // Capture once at compile ingress. No downstream pass calls getenv, so an
  // accepted plan cannot change when the process environment changes later.
#if defined(_WIN32)
  constexpr const char* kName = "HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM";
  ::SetLastError(ERROR_SUCCESS);
  const DWORD required = ::GetEnvironmentVariableA(kName, nullptr, 0U);
  if (required == 0U) {
    return ::GetLastError() != ERROR_ENVVAR_NOT_FOUND;
  }
  return true;
#else
  return std::getenv("HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM") != nullptr;
#endif
}

void publish_lifecycle(detail::LifecyclePlanLowering&& lifecycle,
                       detail::ExecutionPlanState& state) {
  for (const auto& diagnostic : lifecycle.diagnostics) {
    append_compiler_diagnostic(state,
                               diagnostic.token,
                               diagnostic.source,
                               std::nullopt,
                               diagnostic.json_pointer,
                               diagnostic.required,
                               diagnostic.provided);
  }
  state.lifecycle_checkpoints = std::move(lifecycle.checkpoints);
  state.lifecycle_dependencies = std::move(lifecycle.dependencies);
  state.lifecycle_restart_permissions = std::move(lifecycle.restart_permissions);
  state.lifecycle_restart_targets = std::move(lifecycle.restart_targets);
  state.lifecycle_restart_closures = std::move(lifecycle.restart_closures);
  state.lifecycle_mutable_fields = std::move(lifecycle.mutable_fields);
  state.lifecycle_transaction_capacity = std::move(lifecycle.transaction_capacity);
}

void publish_catalog(detail::EndpointCatalogLoweringResult&& catalog,
                     detail::ExecutionPlanState& state) {
  for (const auto& diagnostic : catalog.errors) {
    append_compiler_diagnostic(state,
                               diagnostic.token,
                               diagnostic.source,
                               DiagnosticSeverity::kError,
                               diagnostic.json_pointer,
                               diagnostic.required,
                               diagnostic.provided);
  }
  for (const auto& diagnostic : catalog.warnings) {
    append_compiler_diagnostic(state,
                               diagnostic.token,
                               diagnostic.source,
                               holoscan::DiagnosticSeverity::kWarning,
                               diagnostic.json_pointer,
                               diagnostic.required,
                               diagnostic.provided);
  }
  state.endpoint_catalog = std::move(catalog.catalog);
}

void publish_endpoint_clock_health(detail::EndpointClockHealthLoweringResult&& health,
                                   detail::ExecutionPlanState& state) {
  for (const auto& diagnostic : health.errors) {
    append_compiler_diagnostic(state,
                               diagnostic.token,
                               diagnostic.source,
                               DiagnosticSeverity::kError,
                               diagnostic.json_pointer,
                               diagnostic.required,
                               diagnostic.provided);
  }
  state.endpoint_clock_health = std::move(health.records);
}

void publish_clocks_and_readiness(detail::ClockAndReadinessLoweringResult&& lowered,
                                  detail::ExecutionPlanState& state) {
  state.root_default_clock = lowered.root_default_clock;
  state.clocks = std::move(lowered.clocks);
  state.clock_import_resolutions = std::move(lowered.clock_import_resolutions);
  state.clock_conversions = std::move(lowered.clock_conversions);
  state.readiness = std::move(lowered.readiness);
}

void publish_temporal(detail::TemporalPlanLoweringResult&& temporal,
                      detail::ExecutionPlanState& state) {
  for (const auto& diagnostic : temporal.errors) {
    append_compiler_diagnostic(state,
                               diagnostic.token,
                               diagnostic.source,
                               DiagnosticSeverity::kError,
                               diagnostic.json_pointer,
                               diagnostic.required,
                               diagnostic.provided,
                               diagnostic.detail ? &*diagnostic.detail : nullptr);
  }
  for (const auto& diagnostic : temporal.warnings) {
    append_compiler_diagnostic(state,
                               diagnostic.token,
                               diagnostic.source,
                               holoscan::DiagnosticSeverity::kWarning,
                               diagnostic.json_pointer,
                               diagnostic.required,
                               diagnostic.provided,
                               diagnostic.detail ? &*diagnostic.detail : nullptr);
  }
  state.operator_completion_rules = std::move(temporal.operator_completion_rules);
  state.temporal_triggers = std::move(temporal.triggers);
  state.window_triggers = std::move(temporal.window_triggers);
  state.bracketed_windows = std::move(temporal.bracketed_windows);
  state.input_selections = std::move(temporal.input_selections);
  state.predicate_triggers = std::move(temporal.predicate_triggers);
  state.predicate_inputs = std::move(temporal.predicate_inputs);
  state.predicate_callables = std::move(temporal.predicate_callables);
  state.parameter_derived_terms = std::move(temporal.parameter_derived_terms);
  state.aligned_triggers = std::move(temporal.aligned_triggers);
  state.clock_triggers = std::move(temporal.clock_triggers);
  state.deadlines = std::move(temporal.deadlines);
  state.readable_triggers = std::move(temporal.readable_triggers);
  state.missing_aggregates = std::move(temporal.missing_aggregates);
}

void publish_cuda_graph_region_errors(std::span<const detail::PlanLoweringDiagnostic> errors,
                                      detail::ExecutionPlanState& state) {
  for (const auto& diagnostic : errors) {
    append_compiler_diagnostic(state,
                               diagnostic.token,
                               diagnostic.source,
                               DiagnosticSeverity::kError,
                               diagnostic.json_pointer,
                               diagnostic.required,
                               diagnostic.provided,
                               diagnostic.detail ? &*diagnostic.detail : nullptr);
  }
}

void publish_cuda_graph_regions(detail::CUDAGraphRegionLoweringResult&& lowered,
                                detail::ExecutionPlanState& state) {
  publish_cuda_graph_region_errors(lowered.errors, state);
  state.cuda_graph_capture_plans = std::move(lowered.capture_plans);
  // Capture plans are validated, but executable CUDA Graph lowering is not implemented yet. Reject
  // each plan so the kCudaGraph constraint is never silently ignored by ordinary execution.
  for (const auto& capture_plan : state.cuda_graph_capture_plans) {
    append_compiler_diagnostic(state,
                               "CUDA_GRAPH_EXECUTION_LOWERING_UNAVAILABLE",
                               DiagnosticSource{.graph = capture_plan.graph},
                               DiagnosticSeverity::kError,
                               "/graphs/" + std::to_string(capture_plan.graph.value - 1U) +
                                   "/regions/" + std::to_string(capture_plan.region.value - 1U) +
                                   "/constraints/cuda_graph",
                               "virtual_cuda_graph_operator_lowering",
                               "region_id=" + std::to_string(capture_plan.region.value) +
                                   ",region_name=" + capture_plan.region_name);
  }
}

void publish_physical(detail::PhysicalPlanLoweringResult&& physical,
                      detail::ExecutionPlanState& state) {
  for (const auto& diagnostic : physical.errors) {
    append_compiler_diagnostic(state,
                               diagnostic.token,
                               diagnostic.source,
                               DiagnosticSeverity::kError,
                               diagnostic.json_pointer,
                               diagnostic.required,
                               diagnostic.provided,
                               diagnostic.detail ? &*diagnostic.detail : nullptr);
  }
  detail::publish_physical_plan_records(std::move(physical), state);
}

[[nodiscard]] bool has_error_diagnostic(const detail::ExecutionPlanState& state) noexcept {
  return std::any_of(state.diagnostics.begin(), state.diagnostics.end(), [](const auto& item) {
    return item.severity == DiagnosticSeverity::kError;
  });
}

void reject_unsupported_staged_inputs(const CompileOptions& options,
                                      detail::ExecutionPlanState& state) {
  // The public option is retained for the final converter lowerer, but the
  // foundation cannot yet prove a converter-free graph. Reject the one
  // non-default request instead of accepting a hash-only policy switch.
  if (!options.auto_insert_converters) {
    append_compiler_diagnostic(state, "CONVERTER_AUTO_INSERTION_DISABLE_UNSUPPORTED");
  }
}

[[nodiscard]] bool clock_health_qualification_shape_valid(
    const detail::RuntimeEndpointClockHealthQualification& qualification) noexcept {
  constexpr std::size_t kMaxEvidenceReferenceBytes = 1024U;
  const std::array<std::chrono::nanoseconds, 4U> terms{
      qualification.monitor_period,
      qualification.observation_to_call,
      qualification.delivery_delay,
      qualification.jitter_margin,
  };
  if (terms.front().count() <= 0 || qualification.evidence_reference.empty() ||
      qualification.evidence_reference.size() > kMaxEvidenceReferenceBytes ||
      !qualification.bounded_transition_delivery || !qualification.trace_loss_fail_closed) {
    return false;
  }
  std::int64_t total{};
  for (const auto term : terms) {
    if (term.count() < 0 || total > std::numeric_limits<std::int64_t>::max() - term.count()) {
      return false;
    }
    total += term.count();
  }
  return true;
}

[[nodiscard]] std::vector<detail::EndpointClockHealthQualificationRecord>
resolve_clock_health_qualifications(const detail::RuntimeCompileEvidenceView& evidence,
                                    const detail::ChannelEndpointCatalogState* endpoint_catalog,
                                    detail::ExecutionPlanState& state) {
  constexpr std::size_t kMaxEndpointQualifications = 4096U;
  std::vector<detail::EndpointClockHealthQualificationRecord> result;
  const std::size_t qualification_count =
      std::min(evidence.endpoint_clock_health.size(), kMaxEndpointQualifications);
  result.reserve(qualification_count);
  const auto& package = evidence.package_capabilities.manifest();
  std::set<std::string_view> seen_endpoints;
  if (evidence.endpoint_clock_health.size() > kMaxEndpointQualifications) {
    append_compiler_diagnostic(state, "PROFILE_EVIDENCE_UNSATISFIED");
  }

  for (std::size_t index = 0U; index < qualification_count; ++index) {
    const auto& qualification = evidence.endpoint_clock_health[index];
    const detail::ChannelEndpointRecord* endpoint{};
    if (endpoint_catalog != nullptr) {
      const auto found = std::lower_bound(
          endpoint_catalog->endpoints.begin(),
          endpoint_catalog->endpoints.end(),
          std::string_view{qualification.endpoint_name},
          [](const auto& candidate, std::string_view name) { return candidate.name < name; });
      if (found != endpoint_catalog->endpoints.end() &&
          found->name == qualification.endpoint_name) {
        endpoint = &*found;
      }
    }
    DiagnosticSource source{};
    if (endpoint != nullptr) {
      source.port = endpoint->boundary_port;
    }
    const bool duplicate = !seen_endpoints.insert(qualification.endpoint_name).second;
    const bool package_matches = qualification.package_id == package.package_id &&
                                 qualification.build_id == package.build_id &&
                                 qualification.package_digest == package.digest;
    const bool monitored_import =
        endpoint != nullptr && endpoint->direction == TopicDirection::kImport &&
        endpoint->clock_health.capability == ClockHealthCapability::kStatusReports &&
        endpoint->clock_health.requirement == ClockHealthRequirement::kFreshHealthyReport;
    if (endpoint == nullptr || duplicate || !package_matches || !monitored_import ||
        !clock_health_qualification_shape_valid(qualification)) {
      append_compiler_diagnostic(state, "PROFILE_EVIDENCE_UNSATISFIED", source);
      continue;
    }
    result.push_back(detail::EndpointClockHealthQualificationRecord{
        .endpoint = endpoint->boundary_port,
        .monitor_period = qualification.monitor_period,
        .observation_to_call = qualification.observation_to_call,
        .delivery_delay = qualification.delivery_delay,
        .jitter_margin = qualification.jitter_margin,
        .bounded_transition_delivery = qualification.bounded_transition_delivery,
        .trace_loss_fail_closed = qualification.trace_loss_fail_closed,
        .unspecified_reason_qualified = qualification.unspecified_reason_qualified,
        .evidence_reference = qualification.evidence_reference,
    });
  }
  return result;
}

}  // namespace

void detail::publish_physical_plan_records(detail::PhysicalPlanLoweringResult&& physical,
                                           detail::ExecutionPlanState& state) {
  state.ports = std::move(physical.ports);
  state.interface_port_bindings = std::move(physical.interface_bindings);
  state.payload_edges = std::move(physical.payload_edges);
  state.channel_locality = std::move(physical.channel_locality);
  state.channel_services = std::move(physical.channel_services);
  state.channel_publishers = std::move(physical.channel_publishers);
  state.channel_subscribers = std::move(physical.channel_subscribers);
  state.process_local_channel_services = std::move(physical.process_local_channel_services);
  state.process_local_channel_publishers = std::move(physical.process_local_channel_publishers);
  state.process_local_channel_subscribers = std::move(physical.process_local_channel_subscribers);
  state.cross_process_channel_services = std::move(physical.cross_process_channel_services);
  state.cross_process_channel_publishers = std::move(physical.cross_process_channel_publishers);
  state.cross_process_channel_subscribers = std::move(physical.cross_process_channel_subscribers);
  state.generic_output_loans = std::move(physical.generic_output_loans);
  state.tensor_allocations = std::move(physical.tensor_allocations);
  state.tensor_field_allocations = std::move(physical.tensor_field_allocations);
  state.tensor_interface_pools = std::move(physical.tensor_interface_pools);
  state.tensor_device_resources = std::move(physical.tensor_device_resources);
  state.cuda_compute_streams = std::move(physical.cuda_compute_streams);
  state.tensor_inline_records = std::move(physical.tensor_inline_records);
  state.imported_generation_capacities = std::move(physical.imported_generation_capacities);
  state.tensor_transfer_plans = std::move(physical.tensor_transfer_plans);
  state.pool_capacity_contributors = std::move(physical.pool_capacity_contributors);
  state.pool_capacity_dependencies = std::move(physical.pool_capacity_dependencies);
  state.provider_allocation_plans = std::move(physical.provider_allocation_plans);
  state.publication_outcomes = std::move(physical.publication_outcomes);
}

// CompileOptions is intentionally captured by value so environment and caller-
// owned option state cannot change during total plan construction.
// NOLINTBEGIN(performance-unnecessary-value-param,readability-function-cognitive-complexity)
ExecutionPlan detail::compile_with_runtime_evidence_and_process_placement(
    Graph& graph, CompileOptions options, detail::RuntimeCompileEvidenceView evidence,
    detail::ProcessPlacementBinding placement) {
  // REQ: COMPILE-001, COMPILE-002, COMPILE-009
  const bool removed_partition_environment_selector_present =
      partition_environment_selector_present();
  // All target-package decisions in one compile share this single validated
  // immutable view; no later pass re-reads packaging state or searches paths.
  const detail::RuntimePackageCapabilityView package_capabilities = evidence.package_capabilities;
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);

  auto state = std::make_shared<detail::ExecutionPlanState>();
  state->graph = detail::GraphCompilerAccess::canonical_view(snapshot);
  state->graph_identity = state->graph.identity();

  auto profile = detail::resolve_profile_admission(*snapshot, options.profile);
  const bool profile_accepted = profile.accepted();
  state->profile_manifest = std::move(profile.manifest);
  state->profile_manifest.runtime_abi_version = "holoscan.runtime_abi.v1";
  state->profile_manifest.target_triple = std::string(detail::compile_target_triple());
  state->profile_manifest.producer_build_id = "holoscan-core-5.0";
  for (const auto& token : profile.error_tokens) {
    append_compiler_diagnostic(*state, token);
  }

  reject_unsupported_staged_inputs(options, *state);

  auto process = detail::lower_process_plan(*snapshot, placement);
  for (const auto& token : process.error_tokens) {
    append_compiler_diagnostic(*state, token);
  }
  state->process_roles = std::move(process.roles);
  state->process_selectors = std::move(process.selectors);
  state->vertex_process_placements = std::move(process.vertices);
  // Lower every record under one fixed nonzero seed. Once all compile inputs
  // have been normalized into semantic records, the seed-form artifact digest
  // determines the final generation and one cold pass rebinds provenance.
  // This prevents equivalent request spellings from becoming cache identity.
  constexpr PlanGenerationId plan_generation = 1U;
  state->plan_generation = plan_generation;
  // Capture once at the common source-private ingress so public compilation and provider-injected
  // evidence compilation retain the same executable inventory after the source Graph is gone.
  state->executable_factories = detail::GraphCompilerAccess::executable_factories(graph, *snapshot);

  // Resolve valid CUDA Graph region structure before partitioning. Contract checks remain after
  // Parameter lowering so replay observes accepted generation-zero overrides.
  auto cuda_graph_regions = detail::analyze_cuda_graph_regions(*snapshot);
  publish_cuda_graph_region_errors(cuda_graph_regions.errors, *state);

  auto clocks_and_readiness =
      detail::lower_clocks_and_readiness(*snapshot, plan_generation, profile.effective_profiles());
  if (!clocks_and_readiness) {
    append_error(*state, clocks_and_readiness.error());
  } else {
    publish_clocks_and_readiness(std::move(clocks_and_readiness).value(), *state);
  }

  // Resolve generation zero before replaying temporal declarations. An
  // accepted parameter input replaces only isolated factory clones, so every
  // later setup(), contract(), param_ref(), and lifecycle pass observes the
  // same exact resolved generation without mutating the source Graph.
  auto parameters = detail::lower_parameter_plan(*snapshot,
                                                 state->executable_factories,
                                                 profile.effective_profiles(),
                                                 plan_generation,
                                                 options.parameters,
                                                 options.lifecycle_transactions);
  // Replay each temporal contract once and share the normalized result with temporal-plan
  // lowering and CUDA Graph region validation.
  Diagnostic temporal_replay_failure;
  auto normalized_contracts = detail::GraphCompilerAccess::replay_temporal_contracts(
      state->executable_factories,
      *snapshot,
      plan_generation,
      std::span<const ReadinessPlanRecord>{state->readiness},
      &temporal_replay_failure);
  if (!normalized_contracts) {
    const auto& replay_error = normalized_contracts.error();
    append_compiler_diagnostic(
        *state,
        replay_error.message.empty() ? error_code_name(replay_error.code) : replay_error.message,
        temporal_replay_failure.source,
        std::nullopt,
        {},
        {},
        {},
        temporal_replay_failure.detail ? &*temporal_replay_failure.detail : nullptr);
  } else {
    auto temporal = detail::lower_temporal_plan(*normalized_contracts,
                                                *snapshot,
                                                plan_generation,
                                                state->clocks,
                                                state->clock_conversions,
                                                state->readiness,
                                                profile.effective_profiles(),
                                                options.parameters,
                                                package_capabilities);
    publish_cuda_graph_regions(
        detail::lower_cuda_graph_regions(
            cuda_graph_regions.regions, plan_generation, *normalized_contracts),
        *state);
    publish_temporal(std::move(temporal), *state);
  }
  for (const auto& source : state->readiness) {
    if (!source.referenced) {
      append_compiler_diagnostic(*state,
                                 "READINESS_SOURCE_UNUSED",
                                 DiagnosticSource{.vertex = source.owner},
                                 holoscan::DiagnosticSeverity::kWarning);
    }
  }

  // Resolve rate provenance before lifecycle closure admission. Rate dependencies widen only
  // restart membership; they never become public lifecycle callback-order dependencies.
  auto rate = detail::lower_rate_plan(*snapshot,
                                      options.automatic_queue_sizing.headroom_ratio,
                                      state->temporal_triggers,
                                      state->input_selections,
                                      state->bracketed_windows,
                                      profile.effective_profiles(),
                                      profile.rate_evidence_requirements(),
                                      plan_generation,
                                      process.process_group_by_vertex,
                                      state->window_triggers,
                                      state->predicate_triggers);
  for (const auto& diagnostic : rate.errors) {
    append_compiler_diagnostic(*state,
                               diagnostic.token,
                               diagnostic.source,
                               std::nullopt,
                               diagnostic.json_pointer,
                               diagnostic.required,
                               diagnostic.provided);
  }
  for (const auto& diagnostic : rate.warnings) {
    append_compiler_diagnostic(*state,
                               diagnostic.token,
                               diagnostic.source,
                               holoscan::DiagnosticSeverity::kWarning,
                               diagnostic.json_pointer,
                               diagnostic.required,
                               diagnostic.provided);
  }
  state->port_rates = std::move(rate.port_rates);
  state->accumulation_horizons = std::move(rate.accumulation_horizons);
  state->hard_floor_contributors = std::move(rate.hard_floor_contributors);
  state->queue_sizing_recipes = std::move(rate.queue_sizing_recipes);

  // Parameter queue/storage facts are available before lifecycle lowering,
  // but exact fence/quiesce/rollback minima require dependency-expanded
  // restart closures. Build those closures once, finalize Parameter capacity,
  // then rebuild the lifecycle result from the final admitted Parameter set so
  // rejected capacity cannot leave stale parameter-origin restart authority.
  auto lifecycle = detail::lower_lifecycle_plan(*snapshot,
                                                profile.effective_profiles(),
                                                parameters.lifecycle_mutable_fields,
                                                parameters.capacity,
                                                options.lifecycle_transactions,
                                                plan_generation,
                                                rate.restart_dependencies);
  detail::finalize_parameter_lifecycle_capacity(
      parameters, lifecycle, options.lifecycle_transactions);
  lifecycle = detail::lower_lifecycle_plan(*snapshot,
                                           profile.effective_profiles(),
                                           parameters.lifecycle_mutable_fields,
                                           parameters.capacity,
                                           options.lifecycle_transactions,
                                           plan_generation,
                                           rate.restart_dependencies);
  for (const auto& diagnostic : parameters.diagnostics) {
    append_compiler_diagnostic(*state, diagnostic.token, diagnostic.source);
  }
  state->operator_parameters = std::move(parameters.operators);
  state->parameter_fields = std::move(parameters.fields);
  state->parameter_catalog = std::move(parameters.catalog);
  state->parameter_generation_zero_operators = std::move(parameters.generation_zero_operators);
  state->parameter_generation_zero_fields = std::move(parameters.generation_zero_fields);
  state->parameter_runtime_capacity = parameters.capacity;

  publish_lifecycle(std::move(lifecycle), *state);
  auto tensor_byte_spans = detail::lower_tensor_byte_span_derivations(
      state->executable_factories, *snapshot, plan_generation, state->parameter_fields);
  for (const auto& diagnostic : tensor_byte_spans.diagnostics) {
    detail::publish_derivation_diagnostic(*state, diagnostic);
  }
  state->tensor_byte_span_derivations = std::move(tensor_byte_spans.records);
  std::unique_ptr<detail::GraphSnapshotData> resolved_physical_snapshot;
  const detail::GraphSnapshotData* physical_snapshot = snapshot.get();
  detail::TensorMemoryKindDerivationLowering tensor_memory_kinds;
  if (detail::has_tensor_memory_kind_derivations(*snapshot)) {
    resolved_physical_snapshot = std::make_unique<detail::GraphSnapshotData>(*snapshot);
    physical_snapshot = resolved_physical_snapshot.get();
    tensor_memory_kinds = detail::lower_tensor_memory_kind_derivations(state->executable_factories,
                                                                       *resolved_physical_snapshot,
                                                                       plan_generation,
                                                                       state->parameter_fields);
  }
  for (const auto& diagnostic : tensor_memory_kinds.diagnostics) {
    detail::publish_derivation_diagnostic(*state, diagnostic);
  }
  state->tensor_memory_kind_derivations = std::move(tensor_memory_kinds.records);
  auto tensor_device_bindings =
      resolve_tensor_device_placements(*physical_snapshot, options.deployment, *state);
  tensor_device_bindings.insert(tensor_device_bindings.end(),
                                evidence.tensor_device_bindings.begin(),
                                evidence.tensor_device_bindings.end());

  detail::finalize_temporal_queue_capacities(*physical_snapshot,
                                             rate.queue_sizings,
                                             state->input_selections,
                                             state->bracketed_windows,
                                             state->aligned_triggers,
                                             state->predicate_inputs);

  // Plan one final cover only after generation-zero Parameter, selected temporal/readiness,
  // lifecycle, derived Tensor-byte, process, and hard device facts are immutable.
  // CUDA Graph execution lowering will replace each accepted region with its virtual operator in
  // the Graph projection passed to lower_partition_plan().
  DiagnosticSource partition_failure_source;
  auto partition = detail::lower_partition_plan(*physical_snapshot,
                                                options.profile,
                                                profile.effective_profiles(),
                                                removed_partition_environment_selector_present,
                                                process.process_group_by_vertex,
                                                options.deployment,
                                                tensor_device_bindings,
                                                *state,
                                                &partition_failure_source);
  if (!partition) {
    append_error(*state, partition.error(), partition_failure_source);
  }
  auto partition_roots = detail::validate_partition_execution_roots(*physical_snapshot, *state);
  if (!partition_roots) {
    append_error(*state, partition_roots.error());
  }

  auto deployment = detail::lower_deployment_plan(*physical_snapshot,
                                                  options.deployment,
                                                  profile.effective_profiles(),
                                                  state->execution_partitions);
  for (const auto& token : deployment.error_tokens) {
    append_compiler_diagnostic(*state, token);
  }
  for (const auto& token : deployment.warning_tokens) {
    append_compiler_diagnostic(*state, token, {}, holoscan::DiagnosticSeverity::kWarning);
  }
  state->placement_probes = std::move(deployment.placement_probes);
  state->partition_placements = std::move(deployment.placements);
  state->execution_lanes = std::move(deployment.lanes);
  state->scheduler_plan.execution_lane_count =
      static_cast<std::uint32_t>(state->execution_lanes.size());
  const std::uint32_t partition_owner_count = callback_partition_owner_count(*state);
  state->scheduler_plan.max_concurrent_activations = partition_owner_count;
  state->scheduler_plan.activation_overlap = partition_owner_count > 1U;
  state->scheduler_plan.worker_count = partition_owner_count;
  state->scheduler_plan.intra_path_parallelism = partition_owner_count == 0U ? 0U : 1U;

  for (const auto& source : state->readiness) {
    if (!source.referenced || source.kind != ReadinessSourceKind::kSpin || !source.owner ||
        source.owner.value > deployment.lane_by_vertex.size()) {
      continue;
    }
    const std::uint32_t lane =
        deployment.lane_by_vertex[static_cast<std::size_t>(source.owner.value - 1U)];
    const auto accepted_lane = std::find_if(
        state->execution_lanes.begin(), state->execution_lanes.end(), [lane](const auto& item) {
          return item.execution_lane == lane;
        });
    if (accepted_lane == state->execution_lanes.end() ||
        accepted_lane->accepted_placement.wake_mode != WakeMode::kBusyPoll) {
      append_compiler_diagnostic(*state, "SPIN_UNPLACED", DiagnosticSource{.vertex = source.owner});
    }
  }

  auto endpoint_catalog = detail::lower_endpoint_catalog(*physical_snapshot,
                                                         options.deployment,
                                                         profile.effective_profiles(),
                                                         options.endpoint_catalog,
                                                         state->clocks,
                                                         state->clock_conversions,
                                                         state->tensor_byte_span_derivations,
                                                         rate.queue_sizings);
  const auto* endpoint_catalog_state =
      detail::ChannelEndpointCatalogAccess::state(endpoint_catalog.catalog);
  const auto catalog_endpoints =
      endpoint_catalog_state != nullptr
          ? std::span<const detail::ChannelEndpointRecord>{endpoint_catalog_state->endpoints}
          : std::span<const detail::ChannelEndpointRecord>{};
  state->force_process_local_holomq_event_wake = evidence.force_holomq_process_local_event_wake;
  publish_physical(detail::lower_physical_plan(*physical_snapshot,
                                               plan_generation,
                                               profile.effective_profiles(),
                                               package_capabilities,
                                               state->input_selections,
                                               state->bracketed_windows,
                                               catalog_endpoints,
                                               state->vertex_process_placements,
                                               state->lifecycle_restart_targets,
                                               state->execution_partitions,
                                               tensor_device_bindings,
                                               evidence.force_holomq_process_local_inline,
                                               evidence.force_holomq_process_local_event_wake,
                                               state->tensor_byte_span_derivations,
                                               rate.queue_sizings,
                                               state->queue_sizing_recipes),
                   *state);
  for (const auto& derivation : state->tensor_byte_span_derivations) {
    bool allocation_found{};
    bool allocation_matches{};
    if (derivation.field_id) {
      const auto allocation = std::find_if(state->tensor_field_allocations.begin(),
                                           state->tensor_field_allocations.end(),
                                           [&](const auto& item) {
                                             return item.output == derivation.output &&
                                                    item.field_id == derivation.field_id &&
                                                    item.tensor_index == derivation.tensor_index;
                                           });
      if (allocation != state->tensor_field_allocations.end()) {
        allocation_found = true;
        allocation_matches = allocation->bounds.max_byte_span == derivation.resolved_max_byte_span;
      } else {
        const auto singular =
            std::find_if(state->tensor_allocations.begin(),
                         state->tensor_allocations.end(),
                         [&](const auto& item) { return item.output == derivation.output; });
        allocation_found = singular != state->tensor_allocations.end();
        allocation_matches =
            allocation_found && singular->bounds.max_byte_span == derivation.resolved_max_byte_span;
      }
    } else {
      const auto allocation =
          std::find_if(state->tensor_allocations.begin(),
                       state->tensor_allocations.end(),
                       [&](const auto& item) { return item.output == derivation.output; });
      allocation_found = allocation != state->tensor_allocations.end();
      allocation_matches =
          allocation_found && allocation->bounds.max_byte_span == derivation.resolved_max_byte_span;
    }
    if (allocation_found && !allocation_matches) {
      append_compiler_diagnostic(*state,
                                 "PHYSICAL_TENSOR_BYTE_SPAN_CONTRACT_CONFLICT",
                                 DiagnosticSource{.port = derivation.output});
    }
  }
  for (const auto& derivation : state->tensor_memory_kind_derivations) {
    bool allocation_found{};
    bool allocation_matches{};
    if (derivation.field_id) {
      const auto allocation = std::find_if(state->tensor_field_allocations.begin(),
                                           state->tensor_field_allocations.end(),
                                           [&](const auto& item) {
                                             return item.output == derivation.output &&
                                                    item.field_id == derivation.field_id &&
                                                    item.tensor_index == derivation.tensor_index;
                                           });
      if (allocation != state->tensor_field_allocations.end()) {
        allocation_found = true;
        allocation_matches = allocation->memory_kind == derivation.resolved_memory_kind;
      } else {
        const auto singular =
            std::find_if(state->tensor_allocations.begin(),
                         state->tensor_allocations.end(),
                         [&](const auto& item) { return item.output == derivation.output; });
        allocation_found = singular != state->tensor_allocations.end();
        allocation_matches =
            allocation_found && singular->memory_kind == derivation.resolved_memory_kind;
      }
    } else {
      const auto allocation =
          std::find_if(state->tensor_allocations.begin(),
                       state->tensor_allocations.end(),
                       [&](const auto& item) { return item.output == derivation.output; });
      allocation_found = allocation != state->tensor_allocations.end();
      allocation_matches =
          allocation_found && allocation->memory_kind == derivation.resolved_memory_kind;
    }
    if (allocation_found && !allocation_matches) {
      append_compiler_diagnostic(*state,
                                 "PHYSICAL_TENSOR_MEMORY_KIND_CONTRACT_CONFLICT",
                                 DiagnosticSource{.port = derivation.output});
    }
  }
  const auto cuda_service_streams = detail::lower_cuda_service_stream_requirements(*state);
  publish_cuda_green_context_plan(lower_cuda_green_context_plan(*physical_snapshot,
                                                                options.deployment,
                                                                evidence,
                                                                profile.effective_profiles(),
                                                                cuda_service_streams,
                                                                *state),
                                  *state);
  detail::apply_process_edge_locality(
      process.locality_by_edge, state->payload_edges, state->channel_locality);
  state->closed_island_activations = detail::derive_closed_island_activation_plan(*state);
  const auto clock_health_qualifications =
      resolve_clock_health_qualifications(evidence, endpoint_catalog_state, *state);
  publish_endpoint_clock_health(detail::lower_endpoint_clock_health(*physical_snapshot,
                                                                    plan_generation,
                                                                    state->clocks,
                                                                    state->clock_conversions,
                                                                    profile.effective_profiles(),
                                                                    state->input_selections,
                                                                    clock_health_qualifications),
                                *state);
  publish_catalog(std::move(endpoint_catalog), *state);

  auto manifest = detail::finalize_profile_manifest(options, *state, package_capabilities);
  if (!manifest) {
    append_error(*state, manifest.error());
  }

  state->admitted = profile_accepted && !has_error_diagnostic(*state);
  if (auto runners = detail::derive_partition_runner_plan(*state); runners) {
    state->partition_runners = std::move(*runners);
  } else {
    append_compiler_diagnostic(*state, "PARTITION_RESOURCE_REALIZATION_FAILED");
    state->admitted = false;
    state->partition_runners.clear();
  }
  state->inline_partition_execution = detail::derive_inline_partition_execution_plan(*state);
  auto canonicalized = detail::finalize_plan_artifact(*state);
  if (!canonicalized) {
    // Artifact construction is a cold deterministic pass. Preserve a total
    // rejected result whenever its failure still permits bounded state.
    append_error(*state, canonicalized.error());
    state->admitted = false;
    // Remove derived execution records before rebuilding the rejected artifact.
    state->partition_runners.clear();
    state->inline_partition_execution.reset();
    state->cuda_graph_capture_plans.clear();
    auto retry = detail::finalize_plan_artifact(*state);
    if (!retry) {
      throw RuntimeError(retry.error());
    }
  }

  const PlanGenerationId resolved_generation = detail::derive_plan_generation_id(*state);
  detail::rebind_plan_generation(*state, resolved_generation);
  auto finalized = detail::finalize_plan_artifact(*state);
  if (!finalized) {
    // Rebinding changes only fixed-width provenance values, so failure here is
    // a process-resource failure rather than an admission disposition.
    throw RuntimeError(finalized.error());
  }

  HOLOSCAN_LOG_TRACE("Graph compile completed: admitted={} generation={} diagnostics={}",
                     state->admitted,
                     state->graph_identity.version,
                     state->diagnostics.size());
  return detail::ExecutionPlanAccess::make(std::move(state));
}
// NOLINTEND(performance-unnecessary-value-param,readability-function-cognitive-complexity)

ExecutionPlan detail::compile_with_runtime_evidence(Graph& graph, CompileOptions options,
                                                    detail::RuntimeCompileEvidenceView evidence) {
  return detail::compile_with_runtime_evidence_and_process_placement(
      graph, std::move(options), std::move(evidence), {});
}

ExecutionPlan detail::compile_with_process_placement(
    Graph& graph, CompileOptions options,
    detail::ProcessPlacementBinding placement) {  // NOLINT(performance-unnecessary-value-param)
  return detail::compile_with_runtime_evidence_and_process_placement(
      graph, std::move(options), detail::runtime_compile_evidence_view(), std::move(placement));
}

ExecutionPlan compile(Graph& graph,
                      CompileOptions options) {  // NOLINT(performance-unnecessary-value-param)
  // Capture the generated provider view exactly once before any compiler pass.
  return detail::compile_with_runtime_evidence(
      graph, std::move(options), detail::runtime_compile_evidence_view());
}

ExecutionPlan detail::compile_with_process_local_holomq(
    Graph& graph, CompileOptions options) {  // NOLINT(performance-unnecessary-value-param)
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.force_holomq_process_local_inline = true;
  return detail::compile_with_runtime_evidence(graph, std::move(options), std::move(evidence));
}

ExecutionPlan detail::compile_with_process_local_holomq_event_wake(
    Graph& graph, CompileOptions options) {  // NOLINT(performance-unnecessary-value-param)
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.force_holomq_process_local_inline = true;
  evidence.force_holomq_process_local_event_wake = true;
  return detail::compile_with_runtime_evidence(graph, std::move(options), std::move(evidence));
}

}  // namespace holoscan
