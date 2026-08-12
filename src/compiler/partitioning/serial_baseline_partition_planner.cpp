/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "serial_baseline_partition_planner.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace holoscan::detail::partitioning {
namespace {

[[nodiscard]] unexpected<Error> planner_error(ErrorCode code, std::string_view token) {
  return make_unexpected(Error{code, std::string(token)});
}

void append_boundary_reasons(PartitionBoundaryMask mask,
                             std::vector<PartitionDecisionReason>& reasons) {
  if (mask.contains(PartitionBoundary::kProcess)) {
    reasons.push_back(PartitionDecisionReason::kProcessBoundary);
  }
  if (mask.contains(PartitionBoundary::kExternalEndpoint)) {
    reasons.push_back(PartitionDecisionReason::kExternalEndpointBoundary);
  }
  if (mask.contains(PartitionBoundary::kDeviceTransfer)) {
    reasons.push_back(PartitionDecisionReason::kDeviceTransferIncompatibility);
  }
  if (mask.contains(PartitionBoundary::kRegionBackend)) {
    reasons.push_back(PartitionDecisionReason::kRegionBackendAdmission);
  }
  if (mask.contains(PartitionBoundary::kExplicitThreadPlacement)) {
    reasons.push_back(PartitionDecisionReason::kExplicitThreadPlacement);
  }
  if (mask.contains(PartitionBoundary::kLifecycle)) {
    reasons.push_back(PartitionDecisionReason::kLifecycleIsolation);
  }
  if (mask.contains(PartitionBoundary::kReadiness)) {
    reasons.push_back(PartitionDecisionReason::kReadinessIsolation);
  }
}

}  // namespace

PartitionPlannerDescriptor SerialBaselinePartitionPlanner::descriptor() const noexcept {
  return PartitionPlannerDescriptor{
      serial_baseline_algorithm(),
      kPartitionPlannerInterfaceVersion,
      AlgorithmMaturity::kProduction,
      PartitionPlannerCapabilities{
          static_cast<std::uint32_t>(PartitionPlannerCapability::kExecutionDomains) |
          static_cast<std::uint32_t>(PartitionPlannerCapability::kBoundaryMasks)}};
}

expected<PartitionProposal, Error> SerialBaselinePartitionPlanner::propose(
    const PartitionProblem& problem) const {
  if (!problem.available_capabilities.contains(descriptor().required_capabilities)) {
    return planner_error(ErrorCode::kNotSupported, "PARTITION_PLANNER_CAPABILITY_UNSUPPORTED");
  }
  if (problem.vertices.size() > static_cast<std::size_t>(problem.max_partitions) &&
      problem.max_partitions == 0U) {
    return planner_error(ErrorCode::kResourceExhausted, "PARTITION_LIMIT_EXCEEDED");
  }

  std::vector<VertexIndex> canonical_vertices(problem.vertices.size());
  for (std::size_t index = 0; index < canonical_vertices.size(); ++index) {
    canonical_vertices[index] = static_cast<VertexIndex>(index);
  }
  std::sort(
      canonical_vertices.begin(), canonical_vertices.end(), [&](VertexIndex lhs, VertexIndex rhs) {
        const auto lhs_id = problem.vertices[lhs].id.value;
        const auto rhs_id = problem.vertices[rhs].id.value;
        return lhs_id != rhs_id ? lhs_id < rhs_id : lhs < rhs;
      });

  std::map<std::uint64_t, CandidatePartitionId> domain_partitions;
  PartitionProposal proposal;
  proposal.algorithm = serial_baseline_algorithm();
  proposal.assignments.reserve(problem.vertices.size());
  proposal.decisions.reserve(problem.vertices.size());

  for (const VertexIndex vertex_index : canonical_vertices) {
    const auto& vertex = problem.vertices[vertex_index];
    auto [domain, inserted] = domain_partitions.try_emplace(vertex.execution_domain.value);
    if (inserted) {
      const auto next = static_cast<std::uint64_t>(domain_partitions.size());
      if (next > problem.max_partitions || next > problem.vertices.size()) {
        return planner_error(ErrorCode::kResourceExhausted, "PARTITION_LIMIT_EXCEEDED");
      }
      domain->second = CandidatePartitionId{static_cast<std::uint32_t>(next)};
    }

    PartitionBoundaryMask mandatory = vertex.boundary_before;
    for (const EdgeIndex edge_index : vertex.incoming_edges) {
      if (edge_index >= problem.edges.size()) {
        return planner_error(ErrorCode::kInvalidArgument, "PARTITION_PROPOSAL_CONTRACT_INVALID");
      }
      mandatory.add(problem.edges[edge_index].mandatory_boundaries);
    }

    std::vector<PartitionDecisionReason> reasons;
    append_boundary_reasons(mandatory, reasons);
    const auto decision = !reasons.empty() ? PartitionDecisionKind::kBoundarySplit
                                           : (inserted ? PartitionDecisionKind::kStart
                                                       : PartitionDecisionKind::kJoin);
    proposal.assignments.push_back(VertexPartitionAssignment{vertex_index, domain->second});
    proposal.decisions.push_back(
        PartitionDecisionProposal{vertex_index, domain->second, decision, std::move(reasons)});
  }

  return proposal;
}

std::unique_ptr<PartitionPlanner> make_serial_baseline_partition_planner() {
  return std::make_unique<SerialBaselinePartitionPlanner>();
}

}  // namespace holoscan::detail::partitioning
