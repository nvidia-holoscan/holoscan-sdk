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
#include <unordered_set>
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

#include "../core/graph_compiler_access.hpp"
#include "clock_readiness_lowering.hpp"
#include "closed_island_activation_plan.hpp"
#include "compile_internal.hpp"
#include "compile_platform.hpp"
#include "compiler_diagnostic.hpp"
#include "cuda_execution_plan_validation.hpp"
#include "cuda_graph_execution_lowering.hpp"
#include "cuda_graph_region_lowering.hpp"
#include "cuda_green_context_lowering.hpp"
#include "cuda_service_stream_plan.hpp"
#include "deployment_plan_lowering.hpp"
#include "endpoint_catalog_lowering.hpp"
#include "endpoint_clock_health_lowering.hpp"
#include "execution_plan_contracts.hpp"
#include "execution_plan_state.hpp"
#include "inline_partition_execution_plan.hpp"
#include "lifecycle_plan_lowering.hpp"
#include "parameter_plan_lowering.hpp"
#include "partition_plan_lowering.hpp"
#include "partition_runner_plan.hpp"
#include "physical_plan_lowering.hpp"
#include "physical_topology_lowering.hpp"
#include "plan_artifact_serialization.hpp"
#include "plan_generation.hpp"
#include "plan_lowering_diagnostic.hpp"
#include "process_local_channel_plan_validation.hpp"
#include "process_plan_lowering.hpp"
#include "profile_admission.hpp"
#include "profile_manifest_lowering.hpp"
#include "rate_plan_lowering.hpp"
#include "snapshot_lookup.hpp"
#include "support_thread_affinity_contracts.hpp"
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

void publish_operator_hosts(const detail::GraphSnapshotData& snapshot,
                            PlanGenerationId plan_generation, detail::ExecutionPlanState& state) {
  state.operator_hosts.clear();
  state.operator_hosts.reserve(state.executable_factories.operators.size());
  for (const auto& factory : state.executable_factories.operators) {
    const auto vertex = std::find_if(
        snapshot.vertices.begin(), snapshot.vertices.end(), [&](const auto& candidate) {
          return candidate.id == factory.vertex;
        });
    if (vertex == snapshot.vertices.end()) {
      continue;
    }
    state.operator_hosts.push_back(OperatorHostPlanRecord{
        .owner = OperatorRef{.id = factory.operator_id,
                             .plan_generation = plan_generation,
                             .run_generation = 0U},
        .vertex = factory.vertex,
        .kind = vertex->operator_host,
        .serialization_domain = vertex->serialization_domain,
    });
  }
}

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

void append_clock_error(detail::ExecutionPlanState& state,
                        const detail::GraphSnapshotData& snapshot, const Error& error) {
  if (error.message != "CLOCK_IMPORT_UNBOUND") {
    append_error(state, error);
    return;
  }
  const DiagnosticDetail evidence{.operation = DiagnosticOperation::kCompile, .error = error};
  bool diagnosed = false;
  // Report missing direct bindings, not every descendant whose resolution traverses them.
  for (std::size_t index = 0U; index < snapshot.clock_imports.size(); ++index) {
    const auto& clock_import = snapshot.clock_imports[index];
    const bool bound = std::any_of(
        snapshot.clock_bindings.begin(), snapshot.clock_bindings.end(), [&](const auto& binding) {
          return binding.child_graph == clock_import.graph &&
                 binding.child_clock == clock_import.key;
        });
    const auto* graph = detail::find_graph(snapshot, clock_import.graph);
    if (bound || graph == nullptr) {
      continue;
    }
    const auto* parent = graph->parent ? detail::find_graph(snapshot, *graph->parent) : nullptr;
    const std::string dependency = "clock import \"" + std::string(clock_import.key.name()) +
                                   "\" in graph \"" + detail::canonical_graph_path(*graph) + "\"";
    const std::string required =
        parent != nullptr
            ? "Bind " + dependency + " using bind_clock() in the immediate parent graph \"" +
                  detail::canonical_graph_path(*parent) + "\""
            : "A root graph has no parent to bind its imports; declare a concrete clock using "
              "add_clock() and select it using set_default_clock()";
    append_compiler_diagnostic(
        state,
        error.message,
        DiagnosticSource{.graph_generation = snapshot.generation, .graph = clock_import.graph},
        std::nullopt,
        "/clock_imports/" + std::to_string(index),
        required,
        dependency + " was declared using import_clock() but has no parent binding",
        &evidence);
    diagnosed = true;
  }
  if (!diagnosed) {
    append_error(state, error);
  }
}

[[nodiscard]] std::string bounded_placement_value(std::string_view value) {
  constexpr std::size_t kMaximumEvidenceCharacters = 128U;
  if (value.size() <= kMaximumEvidenceCharacters) {
    return std::string(value);
  }
  return "<oversized:" + std::to_string(value.size()) + ">";
}

void apply_opaque_object_payload_policy(const detail::GraphSnapshotData& snapshot,
                                        OpaqueObjectPayloadPolicy policy,
                                        detail::ExecutionPlanState& state) {
  bool reject = false;
  switch (policy) {
    case OpaqueObjectPayloadPolicy::kAllow:
      return;
    case OpaqueObjectPayloadPolicy::kWarn:
      break;
    case OpaqueObjectPayloadPolicy::kReject:
      reject = true;
      break;
  }
  if (policy != OpaqueObjectPayloadPolicy::kAllow && policy != OpaqueObjectPayloadPolicy::kWarn &&
      policy != OpaqueObjectPayloadPolicy::kReject) {
    append_compiler_diagnostic(state, "OPAQUE_OBJECT_PAYLOAD_POLICY_INVALID");
    return;
  }
  for (const auto& edge : snapshot.edges) {
    const PortRecord* const source = detail::find_port(snapshot, edge.source);
    if (source == nullptr || !detail::is_py_object_payload_identity(source->payload_identity) ||
        !source->payload_identity.schema) {
      continue;
    }
    append_compiler_diagnostic(
        state,
        reject ? "PYOBJECT_PAYLOAD_REJECTED" : "PYOBJECT_PAYLOAD_ADVISORY",
        DiagnosticSource{.vertex = source->owner, .port = PortId{source->id.value}},
        reject ? DiagnosticSeverity::kError : DiagnosticSeverity::kWarning,
        "/edges/" + std::to_string(edge.id.value) + "/payload_identity",
        {},
        source->payload_identity.schema->canonical_name);
  }
}

// Return the serialization-domain split policy that partition lowering applies. A value outside the
// closed enumeration rejects the plan, and lowering continues with the default so the rest of the
// rejected plan's diagnostics stay deterministic.
[[nodiscard]] SerializationDomainSplitPolicy admitted_serialization_domain_split_policy(
    const SerializationDomainSplitPolicy policy, detail::ExecutionPlanState& state) {
  switch (policy) {
    case SerializationDomainSplitPolicy::kWarn:
    case SerializationDomainSplitPolicy::kAllow:
    case SerializationDomainSplitPolicy::kReject:
      return policy;
  }
  append_compiler_diagnostic(state, "SERIALIZATION_DOMAIN_SPLIT_POLICY_INVALID");
  return SerializationDomainSplitPolicy::kWarn;
}

// A bound pinned-host Tensor only receives its planned device through a staging pool, which a
// require-exportable transfer policy never creates. The check is per field, so a copied field
// cannot hide an unstaged one. CUDA fields with that policy are rejected earlier, during topic
// lowering, whether or not a binding exists.
[[nodiscard]] bool imported_topic_has_unstaged_pinned_tensor(const ExternalTopicRecord& topic) {
  const auto unstaged = [](const TensorBoundaryContractRecord& tensor) {
    return tensor.memory_kind == MemoryKind::kPinnedHost &&
           tensor.foreign_emit_policy == InterfaceTensorEmitPolicy::kRequireExportable;
  };
  return (topic.tensor && unstaged(*topic.tensor)) ||
         std::any_of(topic.tensor_fields.begin(),
                     topic.tensor_fields.end(),
                     [&](const auto& field) { return unstaged(field.tensor); });
}

// Resolve imported-topic device placements onto each topic's Graph-side "external" output port.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void resolve_imported_topic_device_placements(
    const detail::GraphSnapshotData& snapshot, const DeploymentBinding& deployment,
    detail::ExecutionPlanState& state, std::set<std::uint64_t>& selected_ports,
    std::vector<detail::TensorDeviceBindingRecord>& result) {
  const auto records = deployment.imported_topic_devices();
  if (records.empty()) {
    return;
  }
  // Sort indices rather than records so output is independent of authoring order while
  // diagnostics keep the authored index.
  std::vector<std::size_t> order(records.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
    return std::tie(records[left].endpoint_name, records[left].device.value) <
           std::tie(records[right].endpoint_name, records[right].device.value);
  });

  // Index the topics once by name; a null entry marks a name shared by several imported topics.
  std::unordered_map<std::string, const VertexRecord*> imported_by_name;
  std::unordered_set<std::string_view> exported_names;
  for (const auto& vertex : snapshot.vertices) {
    if (vertex.kind == VertexKind::kImportedTopic) {
      auto [found, inserted] = imported_by_name.emplace(vertex.name, &vertex);
      if (!inserted) {
        found->second = nullptr;
      }
    } else if (vertex.kind == VertexKind::kExportedTopic) {
      exported_names.insert(vertex.name);
    }
  }
  // Resolve each binding to the topic's own output port or reject it with one diagnostic.
  for (const std::size_t authored_index : order) {
    const auto& placement = records[authored_index];
    const auto reject_placement =
        [&](std::string_view reason, DiagnosticSource source = {}, std::string_view required = {}) {
          append_compiler_diagnostic(
              state,
              "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID",
              source,
              std::nullopt,
              "/deployment/imported_topic_devices/" + std::to_string(authored_index),
              required.empty()
                  ? std::string{"one exact imported Tensor-bearing topic endpoint and nonnegative "
                                "device"}
                  : std::string(required),
              "endpoint_name=" + bounded_placement_value(placement.endpoint_name) + ",device=" +
                  std::to_string(placement.device.value) + ",reason=" + std::string(reason));
        };

    // Select the imported topic by endpoint name, telling a missing name from an exported one.
    const auto owner_match = imported_by_name.find(placement.endpoint_name);
    if (owner_match == imported_by_name.end()) {
      reject_placement(exported_names.contains(placement.endpoint_name) ? "exported_topic"
                                                                        : "unknown_imported_topic");
      continue;
    }
    if (owner_match->second == nullptr) {
      reject_placement("ambiguous_imported_topic",
                       {},
                       "one imported topic per key; set TopicOptions::endpoint_name to tell "
                       "topics that share a URI apart");
      continue;
    }
    // Find the topic's Graph-side output port, which carries the device binding.
    const VertexRecord& owner = *owner_match->second;
    const auto output = std::find_if(snapshot.ports.begin(), snapshot.ports.end(), [&](auto& port) {
      return port.owner == owner.id && port.owner_kind == PortOwnerKind::kImportedTopic &&
             port.direction == PortDirection::kOutput;
    });
    if (output == snapshot.ports.end()) {
      reject_placement(
          "unknown_output",
          DiagnosticSource{
              .graph_generation = snapshot.generation, .graph = owner.graph, .vertex = owner.id});
      continue;
    }
    const DiagnosticSource source{
        .graph_generation = snapshot.generation,
        .graph = owner.graph,
        .vertex = owner.id,
        .port = PortId{output->id.value},
    };
    // Only Tensor-bearing payloads have device storage to place.
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
    // Without a copy-capable transfer policy there is no staging pool to place.
    const auto topic = std::find_if(snapshot.external_topics.begin(),
                                    snapshot.external_topics.end(),
                                    [&](const auto& item) { return item.vertex == owner.id; });
    if (topic != snapshot.external_topics.end() &&
        imported_topic_has_unstaged_pinned_tensor(*topic)) {
      reject_placement("requires_copy_capable_transfer",
                       source,
                       "each kPinnedHost Tensor of a bound topic with a copy-capable transfer "
                       "policy such as kCopyIfNeeded");
      continue;
    }
    // A port already claimed by an earlier binding, of either kind, is a duplicate.
    if (!selected_ports.insert(output->id.value).second) {
      reject_placement("duplicate_output", source);
      continue;
    }
    // Emit the same record an operator placement produces so lowering treats both alike.
    result.push_back(detail::TensorDeviceBindingRecord{
        .port = PortId{output->id.value},
        .device = placement.device,
    });
  }
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
  resolve_imported_topic_device_placements(snapshot, deployment, state, selected_ports, result);
  return result;
}

void publish_cuda_green_context_plan(detail::CudaGreenContextLoweringResult&& lowered,
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
  state.operator_cuda_stream_bindings = std::move(lowered.operator_stream_bindings);
}

// Freeze per-slot release-event inventories only after Green Context lowering has published the
// final exact Operator bindings. Explicit Green placement can make an otherwise host-only
// Operator CUDA-active, so pre-replacement Tensor obligations are not a complete lifetime domain.
void finalize_tensor_release_stream_capacities(detail::ExecutionPlanState& state) {
  const auto allocating_producer = [&](PortId output) {
    const auto port = std::ranges::find(state.ports, output, &PortPlanRecord::id);
    return port == state.ports.end() ? VertexId{} : port->owner;
  };
  const auto derive = [&](PortId output,
                          std::optional<std::uint16_t>
                              tensor_index,
                          ProcessGroupId allocator_process_group,
                          DeviceId device) {
    return detail::tensor_output_release_stream_capacity(state.process_local_channel_services,
                                                         state.cross_process_channel_services,
                                                         state.operator_cuda_stream_bindings,
                                                         allocating_producer(output),
                                                         output,
                                                         tensor_index,
                                                         allocator_process_group,
                                                         device);
  };
  const auto reject_overflow = [&](PortId output, std::string_view json_pointer) {
    const DiagnosticDetail diagnostic{
        .operation = DiagnosticOperation::kCompile,
        .error = Error{ErrorCode::kResourceExhausted,
                       "CUDA release stream capacity exceeds the representable bound"},
        .mismatches = {{.field = "release_stream_capacity",
                        .provided = "overflow",
                        .required = "representable uint32_t capacity"}},
    };
    append_compiler_diagnostic(state,
                               "TENSOR_RELEASE_STREAM_CAPACITY_MISMATCH",
                               DiagnosticSource{.port = output},
                               DiagnosticSeverity::kError,
                               json_pointer,
                               "checked final process/device forwarding and descriptor-staging "
                               "stream bound",
                               "overflow",
                               &diagnostic);
  };
  for (auto& pool : state.tensor_allocations) {
    const auto capacity =
        derive(pool.output, std::nullopt, pool.allocator_process_group, pool.device);
    if (capacity) {
      pool.release_stream_capacity = *capacity;
    } else {
      reject_overflow(pool.output,
                      "/tensor_allocations/" + std::to_string(pool.output.value()) +
                          "/release_stream_capacity");
    }
  }
  for (auto& pool : state.tensor_field_allocations) {
    const auto capacity =
        derive(pool.output, pool.tensor_index, pool.allocator_process_group, pool.device);
    if (capacity) {
      pool.release_stream_capacity = *capacity;
    } else {
      reject_overflow(pool.output,
                      "/tensor_field_allocations/" + std::to_string(pool.output.value()) +
                          "/release_stream_capacity");
    }
  }
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
  state.max_start_delays = std::move(temporal.max_start_delays);
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
}

void publish_cuda_graph_execution_regions(detail::CUDAGraphExecutionLoweringResult&& lowered,
                                          detail::ExecutionPlanState& state) {
  publish_cuda_graph_region_errors(lowered.errors, state);
  state.cuda_graph_execution_regions = std::move(lowered.regions);
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
  state.operator_cuda_stream_bindings = std::move(physical.operator_cuda_stream_bindings);
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
  state->cuda_compute_stream_policy = options.cuda_execution.compute_stream_policy;
  CudaComputeStreamPolicy lowering_cuda_compute_stream_policy = state->cuda_compute_stream_policy;
  switch (state->cuda_compute_stream_policy) {
    case CudaComputeStreamPolicy::kPerExecutionPartition:
    case CudaComputeStreamPolicy::kSharedProcessDevice:
      break;
    default: {
      const DiagnosticDetail detail{
          .operation = DiagnosticOperation::kCompile,
          .error = Error{ErrorCode::kInvalidArgument, "unknown CUDA compute stream policy"},
          .mismatches = {{.field = "compute_stream_policy",
                          .provided = std::to_string(
                              static_cast<std::uint32_t>(state->cuda_compute_stream_policy)),
                          .required = "kPerExecutionPartition or kSharedProcessDevice"}},
      };
      append_compiler_diagnostic(
          *state, "CUDA_STREAM_POLICY_VIOLATION", {}, std::nullopt, {}, {}, {}, &detail);
    }
      // Retain the rejected value for exact inspection and identity. Lower the rest of the
      // rejected plan with a valid value so diagnostics remain deterministic and memory-safe.
      lowering_cuda_compute_stream_policy = CudaComputeStreamPolicy::kPerExecutionPartition;
      break;
  }
  state->graph = detail::GraphCompilerAccess::canonical_view(snapshot);
  state->graph_identity = state->graph.identity();
  apply_opaque_object_payload_policy(*snapshot, options.opaque_object_payloads, *state);
  const SerializationDomainSplitPolicy serialization_domain_splits =
      admitted_serialization_domain_split_policy(options.serialization_domain_splits, *state);

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
  publish_operator_hosts(*snapshot, plan_generation, *state);

  // Resolve valid CUDA Graph region structure before partitioning. Contract checks remain after
  // Parameter lowering so replay observes accepted generation-zero overrides.
  auto cuda_graph_regions = detail::analyze_cuda_graph_regions(*snapshot);
  publish_cuda_graph_region_errors(cuda_graph_regions.errors, *state);

  auto clocks_and_readiness =
      detail::lower_clocks_and_readiness(*snapshot, plan_generation, profile.effective_profiles());
  if (!clocks_and_readiness) {
    append_clock_error(*state, *snapshot, clocks_and_readiness.error());
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
    for (const auto& replay : *normalized_contracts) {
      for (const auto& port : replay.ports) {
        if (port.direction == PortDirection::kOutput && port.runtime_type_ops != nullptr &&
            port.runtime_type_ops->codec_family ==
                detail::RuntimePayloadCodecFamily::kDescriptorTensorSchema) {
          state->executable_factories.schema_codecs.push_back({port.id, port.runtime_type_ops});
        }
      }
    }
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

  const auto tensor_device_inheritance =
      detail::inherit_pass_through_tensor_devices(*physical_snapshot,
                                                  options.deployment,
                                                  state->vertex_process_placements,
                                                  tensor_device_bindings);

  detail::finalize_temporal_queue_capacities(*physical_snapshot,
                                             rate.queue_sizings,
                                             state->input_selections,
                                             state->bracketed_windows,
                                             state->aligned_triggers,
                                             state->predicate_inputs);

  // Plan one final cover only after generation-zero Parameter, selected temporal/readiness,
  // lifecycle, derived Tensor-byte, process, and hard device facts are immutable.
  // CUDA Graph execution lowering represents each accepted region as one fused operator.
  DiagnosticSource partition_failure_source;
  auto partition = detail::lower_partition_plan(*physical_snapshot,
                                                options.profile,
                                                profile.effective_profiles(),
                                                removed_partition_environment_selector_present,
                                                process.process_group_by_vertex,
                                                options.deployment,
                                                serialization_domain_splits,
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
  for (const auto& diagnostic : deployment.errors) {
    append_compiler_diagnostic(*state,
                               diagnostic.token,
                               diagnostic.source,
                               holoscan::DiagnosticSeverity::kError,
                               diagnostic.json_pointer,
                               diagnostic.required,
                               diagnostic.provided,
                               diagnostic.detail ? &*diagnostic.detail : nullptr);
  }
  for (const auto& token : deployment.warning_tokens) {
    append_compiler_diagnostic(*state, token, {}, holoscan::DiagnosticSeverity::kWarning);
  }
  state->placement_probes = std::move(deployment.placement_probes);
  state->partition_placements = std::move(deployment.placements);
  state->support_thread_affinities = std::move(deployment.support_thread_affinities);
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
  state->force_process_local_pump_delivery = evidence.force_process_local_pump_delivery;
  state->fanout_experiment = evidence.fanout_experiment;
  publish_physical(
      detail::lower_physical_plan(
          *physical_snapshot,
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
          state->queue_sizing_recipes,
          lowering_cuda_compute_stream_policy,
          evidence.fanout_experiment == detail::ProcessLocalFanoutExperiment::kPostComputeCut,
          detail::process_local_emit_admission_scope(*state),
          evidence.force_holomq_process_local_fanout,
          state->lifecycle_restart_closures,
          tensor_device_inheritance),
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
  publish_cuda_green_context_plan(
      detail::lower_cuda_green_context_plan(*physical_snapshot,
                                            options.deployment,
                                            evidence,
                                            profile.effective_profiles(),
                                            cuda_service_streams,
                                            *state),
      *state);
  finalize_tensor_release_stream_capacities(*state);
  // Partition, physical Tensor storage, and final CUDA stream/context identities are immutable at
  // this point. Convert each validated region into the fused-operator plan consumed by Runtime.
  publish_cuda_graph_execution_regions(
      detail::finalize_cuda_graph_execution_regions(snapshot->generation, *state), *state);
  auto cuda_graph_channel_backup = detail::remove_cuda_graph_internal_runtime_channels(*state);
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

  if (!detail::valid_process_local_fanout_experiment(*state)) {
    append_error(*state,
                 Error{ErrorCode::kInvalidArgument,
                       "Private fanout experiment requires a closed, single-process, blocking "
                       "scalar/POD, generated-schema, raw Tensor, or descriptor-backed schema "
                       "graph with separate partitions, no eventful services and no restart "
                       "authority"});
  }
  state->admitted = profile_accepted && !has_error_diagnostic(*state);
  if (auto runners = detail::derive_partition_runner_plan(*state); runners) {
    state->partition_runners = std::move(*runners);
  } else {
    append_compiler_diagnostic(*state, "PARTITION_RESOURCE_REALIZATION_FAILED");
    state->admitted = false;
    state->partition_runners.clear();
  }
  if (const auto conflict = detail::find_inline_partition_controller_affinity_conflict(*state)) {
    append_compiler_diagnostic(
        *state,
        "THREAD_PLACEMENT_REJECTED",
        {},
        std::nullopt,
        "/deployment/support_thread_affinities/controller",
        "A controller and the sole partition runner in one process must use the same exact CPU "
        "affinity because they share one Runtime thread",
        "process_group=" + std::to_string(conflict->process_group.value) +
            ",partition=" + std::to_string(conflict->partition.value) +
            ",execution_lane=" + std::to_string(conflict->execution_lane) + ",controller=" +
            detail::support_thread_affinity_summary(conflict->controller_affinity) +
            ",partition_affinity=" +
            detail::support_thread_affinity_summary(conflict->partition_affinity));
    state->admitted = false;
    state->partition_runners.clear();
  }
  state->inline_partition_execution = detail::derive_inline_partition_execution_plan(*state);
  auto canonicalized = detail::finalize_plan_artifact(*state);
  if (!canonicalized) {
    // Recover from inconsistent compiler-generated records. Preserve a rejected result whenever
    // the remaining state can still be finalized.
    append_error(*state, canonicalized.error());
    state->admitted = false;
    // Restore the ordinary channel plan, then clear or rebuild the general execution records that
    // were derived while CUDA Graph regions were fused.
    detail::discard_cuda_graph_capture_and_execution_plan(*state,
                                                          std::move(cuda_graph_channel_backup));
    state->partition_runners.clear();
    state->inline_partition_execution.reset();
    state->closed_island_activations = detail::derive_closed_island_activation_plan(*state);
    auto retry = detail::finalize_plan_artifact(*state);
    if (!retry) {
      throw RuntimeError(retry.error());
    }
  }

  const PlanGenerationId resolved_generation = detail::derive_plan_generation_id(*state);
  detail::rebind_plan_generation(*state, resolved_generation);
  auto finalized = detail::finalize_plan_artifact(*state);
  if (!finalized) {
    // Final rebinding must preserve every cross-record identity. Failure here indicates an
    // inconsistent compiler-generated plan.
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

ExecutionPlan detail::compile_with_process_local_holomq_fanout(
    Graph& graph, CompileOptions options) {  // NOLINT(performance-unnecessary-value-param)
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.force_holomq_process_local_fanout = true;
  return detail::compile_with_runtime_evidence(graph, std::move(options), std::move(evidence));
}

ExecutionPlan detail::compile_with_process_local_holomq_event_wake(
    Graph& graph, CompileOptions options) {  // NOLINT(performance-unnecessary-value-param)
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.force_holomq_process_local_inline = true;
  evidence.force_holomq_process_local_event_wake = true;
  return detail::compile_with_runtime_evidence(graph, std::move(options), std::move(evidence));
}

ExecutionPlan detail::compile_with_process_local_pump_delivery(
    Graph& graph, CompileOptions options,  // NOLINT(performance-unnecessary-value-param)
    const bool force_holomq_process_local_inline) {
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.force_process_local_pump_delivery = true;
  evidence.force_holomq_process_local_inline = force_holomq_process_local_inline;
  // Pump delivery compares HoloMQ realizations, so fanout families stay on HoloMQ.
  evidence.force_holomq_process_local_fanout = true;
  return detail::compile_with_runtime_evidence(graph, std::move(options), std::move(evidence));
}

ExecutionPlan detail::compile_with_process_local_fanout_experiment(
    Graph& graph, CompileOptions options, const ProcessLocalFanoutExperiment experiment,
    const bool force_event_wake) {
  auto evidence = detail::runtime_compile_evidence_view();
  evidence.fanout_experiment = experiment;
  evidence.force_holomq_process_local_event_wake = force_event_wake;
  // The experiment measures HoloMQ fanout, so fanout families stay on HoloMQ.
  evidence.force_holomq_process_local_fanout = true;
  return detail::compile_with_runtime_evidence(graph, std::move(options), std::move(evidence));
}

}  // namespace holoscan
