/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda_graph_region_lowering.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "execution_plan_lookup.hpp"
#include "execution_plan_state.hpp"
#include "snapshot_lookup.hpp"

namespace holoscan::detail {
namespace {

// Keeps a plan edge with its resolved ports for Tensor and boundary validation.
struct ResolvedEdge {
  CUDAGraphCaptureEdgeRecord record;
  const PortRecord* source_port{};
  const PortRecord* target_port{};
};

// Stores the member order and internal edges accepted by topology validation.
struct ResolvedRegionTopology {
  std::vector<VertexId> order;
  std::vector<CUDAGraphCaptureEdgeRecord> edges;
};

// Returns either the accepted topology or the exact graph property that prevented resolution.
struct RegionTopologyResolution {
  std::optional<ResolvedRegionTopology> topology;
  VertexId invalid_member{};
  std::string actual;
};

// Returns either an accepted member contract or the exact unsupported contract property.
struct CaptureContractResolution {
  std::optional<CUDAGraphCaptureMemberContractRecord> contract;
  std::string actual;
  PortId diagnostic_port{};
};

// Holds the complete resolved edge set or the first edge that could not be resolved.
struct GraphEdgeResolution {
  std::vector<ResolvedEdge> edges;
  EdgeId unresolved_edge{};
};

// Keeps each marked region with its resolved operator membership so lowering resolves it once.
struct RegionMembershipResolution {
  const RegionRecord* region{};
  std::optional<std::vector<VertexId>> members;
  bool embedded_graph_region{};
  bool crosses_graph_boundary{};
};

[[nodiscard]] bool has_cuda_graph_constraint(const RegionRecord& region) noexcept {
  return std::any_of(region.constraints.begin(), region.constraints.end(), [](const auto& item) {
    return item.kind() == RegionConstraintKind::kCudaGraph;
  });
}

[[nodiscard]] std::string_view region_constraint_kind_name(RegionConstraintKind kind) noexcept {
  switch (kind) {
    case RegionConstraintKind::kStaticTopology:
      return "static_topology";
    case RegionConstraintKind::kBoundedMemory:
      return "bounded_memory";
    case RegionConstraintKind::kDeviceCompatibleOperators:
      return "device_compatible_operators";
    case RegionConstraintKind::kLatencyBudget:
      return "latency_budget";
    case RegionConstraintKind::kCudaGraph:
      return "cuda_graph";
  }
  return "unknown";
}

[[nodiscard]] std::string other_constraint_evidence(const RegionRecord& region) {
  std::string result{"constraint_count=" + std::to_string(region.constraints.size()) +
                     ";other_constraints="};
  bool first = true;
  for (const auto& constraint : region.constraints) {
    if (constraint.kind() == RegionConstraintKind::kCudaGraph) {
      continue;
    }
    if (!first) {
      result.push_back(',');
    }
    result += region_constraint_kind_name(constraint.kind());
    first = false;
  }
  return result;
}

[[nodiscard]] std::string region_pointer(GraphId graph, RegionId region) {
  return "/graphs/" + std::to_string(graph.value - 1U) + "/regions/" +
         std::to_string(region.value - 1U) + "/constraints/cuda_graph";
}

void add_region_error(std::vector<PlanLoweringDiagnostic>& errors, GraphId graph, RegionId region,
                      std::string_view region_name, std::string_view rule, std::string actual,
                      VertexId vertex = {}, PortId port = {}) {
  errors.push_back(PlanLoweringDiagnostic{
      .token = "CUDA_GRAPH_REGION_INVALID",
      .source = DiagnosticSource{.graph = graph, .vertex = vertex, .port = port},
      .json_pointer = region_pointer(graph, region),
      .required = std::string(rule),
      .provided = "region_id=" + std::to_string(region.value) +
                  ",region_name=" + std::string(region_name) + ",actual=" + std::move(actual),
  });
}

void add_region_error(std::vector<PlanLoweringDiagnostic>& errors, const RegionRecord& region,
                      std::string_view rule, std::string actual, VertexId vertex = {},
                      PortId port = {}) {
  add_region_error(
      errors, region.graph, region.id, region.name, rule, std::move(actual), vertex, port);
}

void add_region_error(std::vector<PlanLoweringDiagnostic>& errors,
                      const CUDAGraphRegionAnalysisRecord& region, std::string_view rule,
                      std::string actual, VertexId vertex = {}, PortId port = {}) {
  add_region_error(errors,
                   region.graph,
                   region.region,
                   region.region_name,
                   rule,
                   std::move(actual),
                   vertex,
                   port);
}

void add_graph_edge_error(std::vector<PlanLoweringDiagnostic>& errors,
                          const GraphSnapshotData& snapshot, EdgeId edge) {
  // This diagnostic means region analysis could not resolve an application edge. Executable CUDA
  // Graph lowering has its own temporary CUDA_GRAPH_EXECUTION_LOWERING_UNAVAILABLE diagnostic.
  errors.push_back(PlanLoweringDiagnostic{
      .token = "CUDA_GRAPH_REGION_ANALYSIS_UNAVAILABLE",
      .source = DiagnosticSource{.graph_generation = snapshot.generation},
      .json_pointer = "/edges/" + std::to_string(edge.value - 1U),
      .required = std::string(cuda_graph_analysis_rule::kResolvedGraphEdges),
      .provided = "edge_id=" + std::to_string(edge.value),
  });
}

// Resolve the region's directly listed operators in deterministic order. Embedded Graph members
// and operators owned by another Graph are not valid direct membership.
[[nodiscard]] std::optional<std::vector<VertexId>> resolve_direct_members(
    const GraphSnapshotData& snapshot, const RegionRecord& region) {
  if (!region.graph_members.empty()) {
    return std::nullopt;
  }
  std::vector<VertexId> members = region.members;
  std::sort(members.begin(), members.end(), [](VertexId left, VertexId right) {
    return left.value < right.value;
  });
  members.erase(std::unique(members.begin(), members.end()), members.end());
  for (const VertexId member : members) {
    const VertexRecord* vertex = find_vertex(snapshot, member);
    if (vertex == nullptr || vertex->kind != VertexKind::kOperator ||
        vertex->graph != region.graph) {
      return std::nullopt;
    }
  }
  return members;
}

// The first implementation accepts only direct root-Graph membership. Embedded Graph membership
// and cross-Graph members can be explored after nested execution lowering is defined.
[[nodiscard]] bool crosses_graph_boundary(const GraphSnapshotData& snapshot,
                                          const RegionRecord& region) noexcept {
  if (!region.graph_members.empty()) {  // embedded Graph added as a region member
    return true;
  }
  return std::any_of(region.members.begin(), region.members.end(), [&](VertexId member) {
    const VertexRecord* vertex = find_vertex(snapshot, member);
    return vertex != nullptr && vertex->graph != region.graph;
  });
}

// Resolve interface aliases at both endpoints and record the connected operators and ports.
[[nodiscard]] std::optional<ResolvedEdge> resolve_edge(const GraphSnapshotData& snapshot,
                                                       const EdgeRecord& edge) {
  const GraphPortId source_id = resolve_interface_port(snapshot, edge.source);
  const GraphPortId target_id = resolve_interface_port(snapshot, edge.target);
  const PortRecord* source = find_port(snapshot, source_id);
  const PortRecord* target = find_port(snapshot, target_id);
  if (source == nullptr || target == nullptr || source->direction != PortDirection::kOutput ||
      target->direction != PortDirection::kInput) {
    return std::nullopt;
  }
  // Canonical snapshot ports and plan ports are densely numbered in the same order. Preserve that
  // numeric identity when the pointer-free capture plan changes GraphPortId to PortId.
  return ResolvedEdge{
      .record = {.edge = edge.id,
                 .source = source->owner,
                 .source_port = PortId{source->id.value},
                 .target = target->owner,
                 .target_port = PortId{target->id.value}},
      .source_port = source,
      .target_port = target,
  };
}

// Resolve every application edge once. A failure blocks CUDA Graph analysis at graph scope.
[[nodiscard]] GraphEdgeResolution resolve_graph_edges(const GraphSnapshotData& snapshot) {
  GraphEdgeResolution result;
  result.edges.reserve(snapshot.edges.size());
  for (const EdgeRecord& edge : snapshot.edges) {
    auto resolved = resolve_edge(snapshot, edge);
    if (!resolved) {
      result.unresolved_edge = edge.id;
      result.edges.clear();
      return result;
    }
    result.edges.push_back(std::move(*resolved));
  }
  return result;
}

[[nodiscard]] bool is_tensor_edge(const ResolvedEdge& edge) noexcept {
  return edge.source_port->payload_identity.kind == PayloadIdentityKind::kTensor &&
         edge.target_port->payload_identity.kind == PayloadIdentityKind::kTensor;
}

[[nodiscard]] std::string_view payload_kind_name(PayloadIdentityKind kind) noexcept {
  switch (kind) {
    case PayloadIdentityKind::kScalar:
      return "scalar";
    case PayloadIdentityKind::kTensor:
      return "tensor";
    case PayloadIdentityKind::kSchema:
      return "schema";
    case PayloadIdentityKind::kPod:
      return "pod";
  }
  return "unknown";
}

[[nodiscard]] std::string_view trigger_kind_name(TemporalTriggerKind kind) noexcept {
  switch (kind) {
    case TemporalTriggerKind::kNone:
      return "none";
    case TemporalTriggerKind::kAnyInput:
      return "any_input";
    case TemporalTriggerKind::kEach:
      return "each";
    case TemporalTriggerKind::kClock:
      return "clock";
    case TemporalTriggerKind::kReadable:
      return "readable";
    case TemporalTriggerKind::kNotified:
      return "notified";
    case TemporalTriggerKind::kDoorbell:
      return "doorbell";
    case TemporalTriggerKind::kSpin:
      return "spin";
    case TemporalTriggerKind::kAligned:
      return "aligned";
    case TemporalTriggerKind::kWindow:
      return "window";
    case TemporalTriggerKind::kPredicate:
      return "predicate";
  }
  return "unknown";
}

// Detect more than one connection with the same source and target operators.
[[nodiscard]] const ResolvedEdge* find_parallel_operator_edge(
    std::span<const ResolvedEdge> edges) noexcept {
  for (std::size_t left = 0U; left < edges.size(); ++left) {
    for (std::size_t right = left + 1U; right < edges.size(); ++right) {
      if (edges[left].record.source == edges[right].record.source &&
          edges[left].record.target == edges[right].record.target) {
        return &edges[right];
      }
    }
  }
  return nullptr;
}

// The current implementation accepts one connected, single-source, single-sink linear sequence.
// Extend this function to resolve additional supported topologies, including DAGs.
[[nodiscard]] RegionTopologyResolution resolve_region_topology(
    std::span<const VertexId> members, std::span<const ResolvedEdge> edges) {
  std::unordered_map<std::uint64_t, std::vector<std::size_t>> outgoing;
  std::unordered_map<std::uint64_t, std::size_t> indegree;
  for (std::size_t index = 0; index < edges.size(); ++index) {
    outgoing[edges[index].record.source.value].push_back(index);
    ++indegree[edges[index].record.target.value];
  }

  // Identify the region endpoints from internal edge degrees.
  std::vector<VertexId> sources;
  std::vector<VertexId> sinks;
  for (const VertexId member : members) {
    if (outgoing[member.value].size() > 1U) {
      return {.invalid_member = member,
              .actual = "outgoing_internal_edges=" + std::to_string(outgoing[member.value].size())};
    }
    if (indegree[member.value] > 1U) {
      return {.invalid_member = member,
              .actual = "incoming_internal_edges=" + std::to_string(indegree[member.value])};
    }
    if (indegree[member.value] == 0U) {
      sources.push_back(member);
    }
    if (outgoing[member.value].empty()) {
      sinks.push_back(member);
    }
  }
  if (sources.size() != 1U || sinks.size() != 1U || edges.size() + 1U != members.size()) {
    return {.actual = "sources=" + std::to_string(sources.size()) + ";sinks=" +
                      std::to_string(sinks.size()) + ";members=" + std::to_string(members.size()) +
                      ";internal_edges=" + std::to_string(edges.size())};
  }

  ResolvedRegionTopology topology;
  topology.order.reserve(members.size());
  topology.edges.reserve(edges.size());
  VertexId current = sources.front();
  // Follow the only outgoing edge from the source. The outgoing- and incoming-degree checks
  // reject fan-out and joins; the final member-count check rejects cycles and disconnected sets.
  while (current && topology.order.size() < members.size()) {
    topology.order.push_back(current);
    const auto& next_edges = outgoing[current.value];
    if (next_edges.empty()) {
      break;
    }
    if (next_edges.size() != 1U) {
      return {.invalid_member = current,
              .actual = "outgoing_internal_edges=" + std::to_string(next_edges.size())};
    }
    const ResolvedEdge& edge = edges[next_edges.front()];
    if (indegree[edge.record.target.value] != 1U) {
      return {.invalid_member = edge.record.target,
              .actual =
                  "incoming_internal_edges=" + std::to_string(indegree[edge.record.target.value])};
    }
    topology.edges.push_back(edge.record);
    current = edge.record.target;
  }
  if (topology.order.size() != members.size() || topology.order.back() != sinks.front()) {
    return {.actual = "ordered_members=" + std::to_string(topology.order.size()) +
                      ";members=" + std::to_string(members.size())};
  }
  return {.topology = std::move(topology)};
}

[[nodiscard]] const GraphCompilerAccess::TemporalReplay* find_contract(
    std::span<const GraphCompilerAccess::TemporalReplay> contracts, VertexId owner) noexcept {
  const auto found = std::find_if(
      contracts.begin(), contracts.end(), [&](const auto& item) { return item.owner == owner; });
  return found == contracts.end() ? nullptr : &*found;
}

[[nodiscard]] const GraphCompilerAccess::TemporalReplayPort* find_input(
    const GraphCompilerAccess::TemporalReplay& replay, std::string_view name) noexcept {
  const auto found = std::find_if(replay.ports.begin(), replay.ports.end(), [&](const auto& port) {
    return port.direction == PortDirection::kInput && port.name == name;
  });
  return found == replay.ports.end() ? nullptr : &*found;
}

[[nodiscard]] CUDAGraphCaptureMemberContractRecord make_contract_record(
    const GraphCompilerAccess::TemporalReplay& replay, PlanGenerationId plan_generation,
    TemporalRecordTriggerKind trigger, PortId trigger_input = {}) noexcept {
  return CUDAGraphCaptureMemberContractRecord{
      .owner = OperatorRef{.id = replay.operator_id, .plan_generation = plan_generation},
      .trigger = trigger,
      .trigger_input = trigger_input,
      .synthesized_default = replay.contract.synthesized_default,
  };
}

// Validate the normalized trigger against the member's region role and create its capture-plan
// record in one pass. Deadlines are checked separately by the caller.
[[nodiscard]] CaptureContractResolution capture_contract_record(
    const GraphCompilerAccess::TemporalReplay& replay, PlanGenerationId plan_generation,
    bool inputless_source, PortId expected_input) {
  if (!replay.contract.inputs.empty()) {  // explicit input-selection rules are deferred
    return {.actual = "input_selectors=" + std::to_string(replay.contract.inputs.size())};
  }

  if (inputless_source) {  // region source without an ingress edge
    // Normal compilation derives both facts from the same resolved edges, so they cannot differ.
    // Keep this check to reject inconsistent replay data if that invariant is ever broken.
    if (replay.contract.connected_input_count != 0U) {
      return {.actual = "inputless_source_connected_inputs=" +
                        std::to_string(replay.contract.connected_input_count)};
    }
    switch (replay.contract.trigger.kind) {
      case TemporalTriggerKind::kClock:
        return {.contract = make_contract_record(
                    replay, plan_generation, TemporalRecordTriggerKind::kClock)};
      case TemporalTriggerKind::kReadable:
        return {.contract = make_contract_record(
                    replay, plan_generation, TemporalRecordTriggerKind::kReadable)};
      case TemporalTriggerKind::kNotified:
        return {.contract = make_contract_record(
                    replay, plan_generation, TemporalRecordTriggerKind::kNotified)};
      case TemporalTriggerKind::kDoorbell:
        return {.contract = make_contract_record(
                    replay, plan_generation, TemporalRecordTriggerKind::kDoorbell)};
      case TemporalTriggerKind::kSpin:
        return {.contract = make_contract_record(
                    replay, plan_generation, TemporalRecordTriggerKind::kSpin)};
      case TemporalTriggerKind::kAligned:
      case TemporalTriggerKind::kWindow:
      case TemporalTriggerKind::kPredicate:
        // These triggers need multi-input or per-activation state not represented by this plan.
        return {.actual = "inputless_source_trigger=" +
                          std::string(trigger_kind_name(replay.contract.trigger.kind))};
      default:
        return {.actual = "inputless_source_trigger=" +
                          std::string(trigger_kind_name(replay.contract.trigger.kind))};
    }
  }

  if (replay.contract.connected_input_count != 1U) {  // member with an incoming edge
    return {.actual = "input_driven_member_connected_inputs=" +
                      std::to_string(replay.contract.connected_input_count)};
  }
  // This plan records one incoming dependency. Aligned, window, and predicate activation remain
  // deferred until their multi-input or per-activation state can be represented.
  if (replay.contract.trigger.kind == TemporalTriggerKind::kAnyInput) {
    return {.contract = make_contract_record(
                replay, plan_generation, TemporalRecordTriggerKind::kAnyInput)};
  }
  if (replay.contract.trigger.kind != TemporalTriggerKind::kEach) {
    return {.actual = "input_driven_member_trigger=" +
                      std::string(trigger_kind_name(replay.contract.trigger.kind))};
  }
  const auto* input = find_input(replay, replay.contract.trigger.port_name);
  if (input == nullptr) {  // OnEach must name a declared input
    return {.actual = "each_input=" + replay.contract.trigger.port_name +
                      ";selected_port_id=unresolved" +
                      ";expected_port_id=" + std::to_string(expected_input.value())};
  }
  if (input->id != expected_input) {  // OnEach must name the topology input
    return {.actual = "each_input=" + replay.contract.trigger.port_name +
                      ";selected_port_id=" + std::to_string(input->id.value()) +
                      ";expected_port_id=" + std::to_string(expected_input.value()),
            .diagnostic_port = input->id};
  }
  return {.contract = make_contract_record(
              replay, plan_generation, TemporalRecordTriggerKind::kEach, input->id)};
}

[[nodiscard]] OperatorRef operator_ref(const GraphCompilerAccess::TemporalReplay& replay,
                                       PlanGenerationId generation) noexcept {
  return OperatorRef{.id = replay.operator_id, .plan_generation = generation};
}

[[nodiscard]] bool valid_capture_plan_edge(const GraphSnapshotData& snapshot,
                                           const CUDAGraphCaptureEdgeRecord& edge) noexcept {
  const EdgeRecord* graph_edge = find_edge(snapshot, edge.edge);
  if (graph_edge == nullptr) {
    return false;
  }
  const GraphPortId source_id = resolve_interface_port(snapshot, graph_edge->source);
  const GraphPortId target_id = resolve_interface_port(snapshot, graph_edge->target);
  const PortRecord* source = find_port(snapshot, source_id);
  const PortRecord* target = find_port(snapshot, target_id);
  // Snapshot GraphPortId and capture-plan PortId preserve the same canonical numeric ID. Compare
  // those values while rechecking the recorded operators, ports, and Tensor payloads.
  return source != nullptr && target != nullptr && source->owner == edge.source &&
         target->owner == edge.target && source->id.value == edge.source_port.value() &&
         target->id.value == edge.target_port.value() &&
         source->payload_identity.kind == PayloadIdentityKind::kTensor &&
         target->payload_identity.kind == PayloadIdentityKind::kTensor;
}

// Recheck the reduced contract using the same source and input-driven rules used during lowering.
[[nodiscard]] bool valid_capture_plan_contract(const CUDAGraphCaptureMemberContractRecord& contract,
                                               bool source, bool has_ingress,
                                               PortId expected_input) noexcept {
  if (source && !has_ingress) {
    const bool supported = contract.trigger == TemporalRecordTriggerKind::kClock ||
                           contract.trigger == TemporalRecordTriggerKind::kReadable ||
                           contract.trigger == TemporalRecordTriggerKind::kNotified ||
                           contract.trigger == TemporalRecordTriggerKind::kDoorbell ||
                           contract.trigger == TemporalRecordTriggerKind::kSpin;
    return supported && !contract.trigger_input && !contract.synthesized_default;
  }
  if (contract.trigger == TemporalRecordTriggerKind::kEach) {
    return contract.trigger_input == expected_input && !contract.synthesized_default;
  }
  return contract.trigger == TemporalRecordTriggerKind::kAnyInput && !contract.trigger_input;
}

// The temporal plan is the authority for normalized scheduling contracts. Require the reduced
// CUDA Graph copy to describe that exact trigger instead of accepting any shape-compatible form.
[[nodiscard]] bool capture_contract_matches_temporal_plan(
    const ExecutionPlanState& state,
    const CUDAGraphCaptureMemberContractRecord& contract) noexcept {
  const TemporalTriggerPlanRecord* trigger =
      find_unique_temporal_trigger_by_owner(state, contract.owner.id);
  if (trigger == nullptr || trigger->owner != contract.owner) {
    return false;
  }

  bool synthesized_default = false;
  switch (trigger->origin) {
    case TemporalRecordOrigin::kAuthored:
      break;
    case TemporalRecordOrigin::kCoreSynthesized:
      synthesized_default = true;
      break;
    default:
      return false;
  }
  return trigger->kind == contract.trigger && trigger->trigger_input == contract.trigger_input &&
         synthesized_default == contract.synthesized_default;
}

// Completion is an Operator-owned policy, not part of the reduced capture trigger. When a capture
// member owns a completion rule, require the only placement admitted by the temporal compiler: the
// inputless region source. The common OperatorRef is the explicit join between the two plan views.
[[nodiscard]] bool valid_capture_member_completion(const ExecutionPlanState& state,
                                                   const OperatorRef owner, bool source,
                                                   bool has_ingress) noexcept {
  const OperatorCompletionPlanRecord* matched = nullptr;
  for (const auto& rule : state.operator_completion_rules) {
    if (rule.owner != owner) {
      continue;
    }
    if (matched != nullptr) {
      return false;
    }
    matched = &rule;
  }
  return matched == nullptr ||
         (source && !has_ingress && matched->kind == OperatorCompletionRuleKind::kComputeCalls);
}

}  // namespace

CUDAGraphRegionAnalysisResult analyze_cuda_graph_regions(const GraphSnapshotData& snapshot) {
  CUDAGraphRegionAnalysisResult result;
  std::unordered_map<std::uint64_t, std::size_t> member_use_count;
  std::vector<RegionMembershipResolution> region_memberships;
  region_memberships.reserve(snapshot.regions.size());
  const GraphRecord* root_graph = find_root_graph(snapshot);
  // Count all CUDA Graph memberships first so overlap diagnostics are independent of region order.
  for (const RegionRecord& region : snapshot.regions) {
    if (!has_cuda_graph_constraint(region)) {
      continue;
    }
    if (root_graph == nullptr || region.graph != root_graph->id) {
      region_memberships.push_back({.region = &region, .embedded_graph_region = true});
      continue;
    }
    const bool crosses_boundary = crosses_graph_boundary(snapshot, region);
    if (crosses_boundary) {
      region_memberships.push_back(
          {.region = &region, .members = std::nullopt, .crosses_graph_boundary = true});
      continue;
    }
    auto members = resolve_direct_members(snapshot, region);
    if (!members) {
      region_memberships.push_back({.region = &region, .members = std::nullopt});
      continue;
    }
    for (const VertexId member : *members) {
      ++member_use_count[member.value];
    }
    region_memberships.push_back({.region = &region, .members = std::move(members)});
  }
  if (region_memberships.empty()) {
    return result;
  }

  const GraphEdgeResolution graph_edges = resolve_graph_edges(snapshot);
  if (graph_edges.unresolved_edge) {
    add_graph_edge_error(result.errors, snapshot, graph_edges.unresolved_edge);
    canonicalize_plan_lowering_diagnostics(result.errors);
    return result;
  }

  // Validate each marked region and retain the structure needed after contract replay.
  for (const RegionMembershipResolution& membership : region_memberships) {
    const RegionRecord& region = *membership.region;
    const auto& resolved_members = membership.members;
    // CUDA Graph lowering is an exact execution policy. Until interactions with the other region
    // constraints have explicit semantics, reject every combination instead of silently dropping
    // or partially applying one of the authored requirements.
    if (region.constraints.size() != 1U) {
      add_region_error(result.errors,
                       region,
                       cuda_graph_region_rule::kExclusiveCUDAGraphConstraint,
                       other_constraint_evidence(region));
      continue;
    }
    if (membership.embedded_graph_region) {
      const std::string root_graph_id =
          root_graph == nullptr ? "missing" : std::to_string(root_graph->id.value);
      add_region_error(result.errors,
                       region,
                       cuda_graph_region_rule::kRootGraphRegion,
                       "region_graph_id=" + std::to_string(region.graph.value) +
                           ";root_graph_id=" + root_graph_id);
      continue;
    }
    if (membership.crosses_graph_boundary) {
      add_region_error(result.errors,
                       region,
                       cuda_graph_region_rule::kDirectOperatorMembersInRegionGraph,
                       "embedded_graph_members=" + std::to_string(region.graph_members.size()) +
                           ";direct_operator_members=" + std::to_string(region.members.size()));
      continue;
    }
    if (!resolved_members || resolved_members->empty()) {
      add_region_error(
          result.errors,
          region,
          cuda_graph_region_rule::kNonemptyResolvedOperatorMembership,
          resolved_members ? "resolved_operator_members=0" : "operator_membership=unresolved");
      continue;
    }
    bool invalid = false;
    // Confirm that this region exclusively owns each member and that every member explicitly
    // permits its compute work to be included in CUDA Graph capture.
    for (const VertexId member : *resolved_members) {
      // One operator cannot be captured by two independently lowered CUDA Graph regions.
      if (member_use_count[member.value] > 1U) {
        add_region_error(
            result.errors,
            region,
            cuda_graph_region_rule::kDisjointCUDAGraphRegions,
            "cuda_graph_region_memberships=" + std::to_string(member_use_count[member.value]),
            member);
        invalid = true;
        break;
      }
      // Capture eligibility is declared through OperatorSpec and is required for every member.
      const VertexRecord* vertex = find_vertex(snapshot, member);
      if (vertex == nullptr || !vertex->cuda_graph_capture_eligible) {
        add_region_error(
            result.errors,
            region,
            cuda_graph_region_rule::kCaptureEligibleOperator,
            vertex == nullptr ? "operator=unresolved" : "cuda_graph_capture_eligible=false",
            member);
        invalid = true;
        break;
      }
    }
    if (invalid) {
      continue;
    }

    std::vector<ResolvedEdge> internal;
    std::vector<ResolvedEdge> ingress;
    std::vector<ResolvedEdge> egress;
    const auto contains = [&](VertexId vertex) {
      return std::binary_search(
          resolved_members->begin(), resolved_members->end(), vertex, [](auto left, auto right) {
            return left.value < right.value;
          });
    };
    // Classify the shared resolved edges as internal, ingress, egress, or unrelated.
    for (const ResolvedEdge& edge : graph_edges.edges) {
      const bool source_member = contains(edge.record.source);
      const bool target_member = contains(edge.record.target);
      if (source_member && target_member) {
        internal.push_back(edge);
      } else if (!source_member && target_member) {
        ingress.push_back(edge);
      } else if (source_member && !target_member) {
        egress.push_back(edge);
      }
    }
    const auto invalid_internal_edge = std::find_if(
        internal.begin(), internal.end(), [](const auto& edge) { return !is_tensor_edge(edge); });
    if (invalid_internal_edge != internal.end()) {
      add_region_error(result.errors,
                       region,
                       cuda_graph_region_rule::kTensorInternalEdges,
                       "edge_id=" + std::to_string(invalid_internal_edge->record.edge.value) +
                           ";source_payload=" +
                           std::string(payload_kind_name(
                               invalid_internal_edge->source_port->payload_identity.kind)) +
                           ";target_payload=" +
                           std::string(payload_kind_name(
                               invalid_internal_edge->target_port->payload_identity.kind)),
                       invalid_internal_edge->record.source,
                       invalid_internal_edge->record.source_port);
      continue;
    }
    const ResolvedEdge* parallel_edge = find_parallel_operator_edge(internal);
    if (parallel_edge != nullptr) {
      add_region_error(
          result.errors,
          region,
          cuda_graph_region_rule::kSingleEdgePerOperatorHop,
          "parallel_edge_id=" + std::to_string(parallel_edge->record.edge.value) +
              ";source_vertex_id=" + std::to_string(parallel_edge->record.source.value) +
              ";target_vertex_id=" + std::to_string(parallel_edge->record.target.value),
          parallel_edge->record.source,
          parallel_edge->record.source_port);
      continue;
    }

    auto topology = resolve_region_topology(*resolved_members, internal);
    if (!topology.topology) {
      add_region_error(result.errors,
                       region,
                       cuda_graph_region_rule::kSingleLinearOperatorChain,
                       std::move(topology.actual),
                       topology.invalid_member);
      continue;
    }
    const auto& order = topology.topology->order;

    if (ingress.size() > 1U || egress.size() > 1U) {
      add_region_error(result.errors,
                       region,
                       cuda_graph_region_rule::kAtMostOneRegionIngressAndEgress,
                       "ingress_edges=" + std::to_string(ingress.size()) +
                           ";egress_edges=" + std::to_string(egress.size()));
      continue;
    }
    // Under the current topology rules, ingress may enter only the resolved source and egress may
    // leave only the resolved sink.
    if (!ingress.empty() && ingress.front().record.target != order.front()) {
      add_region_error(
          result.errors,
          region,
          cuda_graph_region_rule::kRegionEdgesOnlyAtSourceAndSink,
          "ingress_target_vertex_id=" + std::to_string(ingress.front().record.target.value) +
              ";source_vertex_id=" + std::to_string(order.front().value),
          ingress.front().record.target,
          ingress.front().record.target_port);
      continue;
    }
    if (!egress.empty() && egress.front().record.source != order.back()) {
      add_region_error(
          result.errors,
          region,
          cuda_graph_region_rule::kRegionEdgesOnlyAtSourceAndSink,
          "egress_source_vertex_id=" + std::to_string(egress.front().record.source.value) +
              ";sink_vertex_id=" + std::to_string(order.back().value),
          egress.front().record.source,
          egress.front().record.source_port);
      continue;
    }
    const ResolvedEdge* invalid_boundary_edge = nullptr;
    if (!ingress.empty() && !is_tensor_edge(ingress.front())) {
      invalid_boundary_edge = &ingress.front();
    } else if (!egress.empty() && !is_tensor_edge(egress.front())) {
      invalid_boundary_edge = &egress.front();
    }
    if (invalid_boundary_edge != nullptr) {
      add_region_error(result.errors,
                       region,
                       cuda_graph_region_rule::kTensorRegionBoundaryEdges,
                       "edge_id=" + std::to_string(invalid_boundary_edge->record.edge.value) +
                           ";source_payload=" +
                           std::string(payload_kind_name(
                               invalid_boundary_edge->source_port->payload_identity.kind)) +
                           ";target_payload=" +
                           std::string(payload_kind_name(
                               invalid_boundary_edge->target_port->payload_identity.kind)),
                       invalid_boundary_edge->record.source,
                       invalid_boundary_edge->record.source_port);
      continue;
    }

    std::optional<CUDAGraphCaptureEdgeRecord> accepted_ingress;
    if (!ingress.empty()) {
      accepted_ingress = ingress.front().record;
    }
    std::optional<CUDAGraphCaptureEdgeRecord> accepted_egress;
    if (!egress.empty()) {
      accepted_egress = egress.front().record;
    }
    result.regions.push_back(CUDAGraphRegionAnalysisRecord{
        .graph = region.graph,
        .region = region.id,
        .region_name = region.name,
        .order = std::move(topology.topology->order),
        .internal_edges = std::move(topology.topology->edges),
        .ingress = accepted_ingress,
        .egress = accepted_egress,
    });
  }

  canonicalize_plan_lowering_diagnostics(result.errors);
  return result;
}

CUDAGraphRegionLoweringResult lower_cuda_graph_regions(
    std::span<const CUDAGraphRegionAnalysisRecord> regions, PlanGenerationId plan_generation,
    std::span<const GraphCompilerAccess::TemporalReplay> normalized_contracts) {
  CUDAGraphRegionLoweringResult result;
  for (const CUDAGraphRegionAnalysisRecord& region : regions) {
    std::vector<CUDAGraphCaptureMemberRecord> members;
    std::vector<CUDAGraphCaptureMemberContractRecord> contracts;
    members.reserve(region.order.size());
    contracts.reserve(region.order.size());
    bool invalid = false;
    // Validate each normalized contract against the member's topology role and expected input.
    for (std::size_t index = 0; index < region.order.size(); ++index) {
      const auto* replay = find_contract(normalized_contracts, region.order[index]);
      if (replay == nullptr) {
        add_region_error(result.errors,
                         region,
                         cuda_graph_region_rule::kNormalizedMemberContract,
                         "normalized_contract=missing",
                         region.order[index]);
        invalid = true;
        break;
      }
      // Deadline handling is not yet supported. We will explore this in a
      // future release.
      if (replay->contract.deadline || replay->contract.deadline_dependency) {
        std::string deadline_source = "deadline=";
        if (replay->contract.deadline && replay->contract.deadline_dependency) {
          deadline_source += "literal_and_parameter";
        } else if (replay->contract.deadline_dependency) {
          deadline_source += "parameter";
        } else {
          deadline_source += "literal";
        }
        add_region_error(result.errors,
                         region,
                         cuda_graph_region_rule::kNoMemberDeadline,
                         std::move(deadline_source),
                         region.order[index]);
        invalid = true;
        break;
      }
      // Completion remains an Operator-owned rule outside the reduced capture trigger. Reject it
      // here unless the member has the same zero-input producer shape admitted by temporal
      // lowering, so an invalid placement cannot leave a published CUDA Graph capture recipe.
      if (replay->contract.compute_calls) {
        const bool source_member = index == 0U;
        const bool has_input =
            std::any_of(replay->ports.begin(), replay->ports.end(), [](const auto& port) {
              return port.direction == PortDirection::kInput;
            });
        const bool has_output =
            std::any_of(replay->ports.begin(), replay->ports.end(), [](const auto& port) {
              return port.direction == PortDirection::kOutput;
            });
        if (!source_member || region.ingress || has_input || !has_output) {
          std::string placement;
          if (!source_member) {
            placement = "captured_non_source";
          } else if (region.ingress) {
            placement = "input_driven_source";
          } else if (has_input) {
            placement = "source_declares_input";
          } else {
            placement = "outputless_source";
          }
          add_region_error(result.errors,
                           region,
                           cuda_graph_region_rule::kSupportedMemberContract,
                           "completion=compute_calls;placement=" + placement,
                           region.order[index]);
          invalid = true;
          break;
        }
      }
      // Select the input that this member's contract must reference. The source uses the region
      // ingress, when present; every later member uses its incoming internal edge's target port.
      PortId expected_input{};
      if (index == 0U) {       // source member
        if (region.ingress) {  // source has a region ingress
          expected_input = region.ingress->target_port;
        }
      } else {  // intermediate or sink member
        expected_input = region.internal_edges[index - 1U].target_port;
      }
      const bool inputless_source = index == 0U && !region.ingress;
      auto contract =
          capture_contract_record(*replay, plan_generation, inputless_source, expected_input);
      if (!contract.contract) {
        const PortId diagnostic_port =
            contract.diagnostic_port ? contract.diagnostic_port : expected_input;
        add_region_error(result.errors,
                         region,
                         cuda_graph_region_rule::kSupportedMemberContract,
                         std::move(contract.actual),
                         region.order[index],
                         diagnostic_port);
        invalid = true;
        break;
      }
      members.push_back(CUDAGraphCaptureMemberRecord{
          .operator_ref = operator_ref(*replay, plan_generation), .vertex = region.order[index]});
      contracts.push_back(std::move(*contract.contract));
    }
    if (invalid) {
      continue;
    }

    result.capture_plans.push_back(CUDAGraphCapturePlanRecord{
        .graph = region.graph,
        .region = region.region,
        .region_name = region.region_name,
        .source = members.front().operator_ref,
        .sink = members.back().operator_ref,
        .members = std::move(members),
        .internal_edges = region.internal_edges,
        .ingress = region.ingress,
        .egress = region.egress,
        .contracts = std::move(contracts),
    });
  }

  canonicalize_plan_lowering_diagnostics(result.errors);
  return result;
}

bool valid_cuda_graph_capture_plans(const ExecutionPlanState& state) {
  // Capture plans refer to canonical Graph identities, so validation requires the immutable
  // snapshot retained by the ExecutionPlan.
  const auto snapshot = GraphCompilerAccess::retain_snapshot(state.graph);
  if (!snapshot && !state.cuda_graph_capture_plans.empty()) {
    return false;
  }
  if (state.cuda_graph_capture_plans.empty()) {
    return true;
  }
  // Re-derive every structurally accepted region once from the immutable Graph snapshot. A stored
  // optional boundary cannot prove that an omitted boundary was genuinely absent; exact comparison
  // with this independent analysis makes both presence and absence fail closed.
  const CUDAGraphRegionAnalysisResult analyzed = analyze_cuda_graph_regions(*snapshot);
  // The executable factory inventory is the authority that maps each plan-bound Operator ID back
  // to its canonical Graph vertex. Duplicate or incomplete mappings invalidate the plan.
  std::unordered_map<OperatorInstanceId, VertexId> operator_vertices;
  operator_vertices.reserve(state.executable_factories.operators.size());
  for (const auto& factory : state.executable_factories.operators) {
    if (factory.operator_id == 0U || !factory.vertex ||
        !operator_vertices.emplace(factory.operator_id, factory.vertex).second) {
      return false;
    }
  }
  std::optional<std::pair<std::uint64_t, std::uint64_t>> previous_plan_key;
  for (const auto& capture_plan : state.cuda_graph_capture_plans) {
    // Require a complete region identity, one contract per member, and source/sink references that
    // agree with the topology order stored in the plan.
    if (!capture_plan.graph || !capture_plan.region || capture_plan.region_name.empty() ||
        capture_plan.members.empty() ||
        capture_plan.source != capture_plan.members.front().operator_ref ||
        capture_plan.sink != capture_plan.members.back().operator_ref ||
        capture_plan.contracts.size() != capture_plan.members.size()) {
      return false;
    }

    // The currently supported topology produces one internal edge fewer than its member count.
    if (capture_plan.internal_edges.size() + 1U != capture_plan.members.size()) {
      return false;
    }

    // Lowering emits plans in canonical (Graph ID, Region ID) order. Strictly increasing keys
    // preserve deterministic artifacts and reject duplicate records for the same region.
    const auto current_plan_key = std::pair{capture_plan.graph.value, capture_plan.region.value};
    if (previous_plan_key && current_plan_key <= *previous_plan_key) {
      return false;
    }
    previous_plan_key = current_plan_key;

    // Match the plan only to a region that independently passed the same structural analysis used
    // by lowering. Exact topology comparison below prevents a copied plan from deleting or
    // substituting a real ingress, egress, internal edge, or member.
    const auto expected =
        std::find_if(analyzed.regions.begin(), analyzed.regions.end(), [&](const auto& candidate) {
          return candidate.graph == capture_plan.graph && candidate.region == capture_plan.region &&
                 candidate.region_name == capture_plan.region_name;
        });
    if (expected == analyzed.regions.end() ||
        capture_plan.internal_edges != expected->internal_edges ||
        capture_plan.ingress != expected->ingress || capture_plan.egress != expected->egress ||
        capture_plan.members.size() != expected->order.size()) {
      return false;
    }
    for (std::size_t index = 0; index < capture_plan.members.size(); ++index) {
      const auto& member = capture_plan.members[index];
      // The member, its contract, and the current plan must use one plan-bound Operator identity.
      if (!member.operator_ref.plan_bound() ||
          member.operator_ref.plan_generation != state.plan_generation || !member.vertex ||
          member.vertex != expected->order[index] ||
          capture_plan.contracts[index].owner != member.operator_ref) {
        return false;
      }
      const auto operator_vertex = operator_vertices.find(member.operator_ref.id);
      // Verify the stored Operator-to-vertex relation against the executable factory inventory.
      if (operator_vertex == operator_vertices.end() || operator_vertex->second != member.vertex) {
        return false;
      }
      // Recheck the capture permission recorded from the member's OperatorSpec declaration.
      const VertexRecord* member_record = find_vertex(*snapshot, member.vertex);
      if (member_record == nullptr || !member_record->cuda_graph_capture_eligible) {
        return false;
      }
      // Reconstruct the input expected by the reduced contract from the stored topology.
      PortId expected_input{};
      if (index == 0U) {  // source member
        if (capture_plan.ingress) {
          expected_input = capture_plan.ingress->target_port;
        }
      } else {  // intermediate or sink member
        expected_input = capture_plan.internal_edges[index - 1U].target_port;
      }
      if (!valid_capture_plan_contract(capture_plan.contracts[index],
                                       index == 0U,
                                       capture_plan.ingress.has_value(),
                                       expected_input) ||
          !capture_contract_matches_temporal_plan(state, capture_plan.contracts[index]) ||
          !valid_capture_member_completion(state,
                                           capture_plan.contracts[index].owner,
                                           index == 0U,
                                           capture_plan.ingress.has_value())) {
        return false;
      }
      if (index < capture_plan.internal_edges.size()) {
        const auto& edge = capture_plan.internal_edges[index];
        // The current plan layout stores each internal edge beside its source member. Validate
        // both that relation and the edge's canonical snapshot endpoints.
        if (edge.source != member.vertex ||
            edge.target != capture_plan.members[index + 1U].vertex ||
            !valid_capture_plan_edge(*snapshot, edge)) {
          return false;
        }
      }
    }
    const VertexId source_vertex = capture_plan.members.front().vertex;
    const VertexId sink_vertex = capture_plan.members.back().vertex;
    // Boundary edges must attach to the stored source and sink and still resolve to their recorded
    // Tensor endpoints in the canonical snapshot.
    if ((capture_plan.ingress && (capture_plan.ingress->target != source_vertex ||
                                  !valid_capture_plan_edge(*snapshot, *capture_plan.ingress))) ||
        (capture_plan.egress && (capture_plan.egress->source != sink_vertex ||
                                 !valid_capture_plan_edge(*snapshot, *capture_plan.egress)))) {
      return false;
    }
  }
  return true;
}

}  // namespace holoscan::detail
