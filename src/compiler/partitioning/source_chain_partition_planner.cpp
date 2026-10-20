/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "source_chain_partition_planner.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "partition_canonicalization.hpp"

namespace holoscan::detail::partitioning {
namespace {

[[nodiscard]] unexpected<Error> planner_error(ErrorCode code, std::string_view token) {
  return make_unexpected(Error{code, std::string(token)});
}

}  // namespace

PartitionPlannerDescriptor SourceChainPartitionPlanner::descriptor() const noexcept {
  return PartitionPlannerDescriptor{source_chain_algorithm(),
                                    kPartitionPlannerInterfaceVersion,
                                    AlgorithmMaturity::kProduction,
                                    kAllPartitionPlannerCapabilities};
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<PartitionProposal, Error> SourceChainPartitionPlanner::propose(
    const PartitionProblem& problem) const {
  if (!problem.available_capabilities.contains(descriptor().required_capabilities)) {
    return planner_error(ErrorCode::kNotSupported, "PARTITION_PLANNER_CAPABILITY_UNSUPPORTED");
  }
  if (problem.vertices.size() > std::numeric_limits<std::uint32_t>::max()) {
    return planner_error(ErrorCode::kResourceExhausted, "PARTITION_LIMIT_EXCEEDED");
  }

  const std::size_t vertex_count = problem.vertices.size();
  std::vector<std::optional<CandidatePartitionId>> authored_partition_by_vertex(vertex_count);
  for (std::size_t constraint_index = 0U; constraint_index < problem.authored_constraints.size();
       ++constraint_index) {
    const auto& constraint = problem.authored_constraints[constraint_index];
    if (!constraint.id || constraint.members.empty() ||
        constraint_index >= std::numeric_limits<std::uint32_t>::max()) {
      return planner_error(ErrorCode::kInvalidArgument, "PARTITION_PROPOSAL_CONTRACT_INVALID");
    }
    const CandidatePartitionId candidate{static_cast<std::uint32_t>(constraint_index + 1U)};
    for (const VertexIndex member : constraint.members) {
      if (member >= vertex_count || authored_partition_by_vertex[member].has_value()) {
        return planner_error(ErrorCode::kInvalidArgument, "PARTITION_PROPOSAL_CONTRACT_INVALID");
      }
      authored_partition_by_vertex[member] = candidate;
    }
  }
  std::vector<std::vector<VertexIndex>> predecessors(vertex_count);
  std::vector<std::vector<VertexIndex>> successors(vertex_count);
  std::vector<std::vector<EdgeIndex>> incoming_edges(vertex_count);
  for (std::size_t index = 0; index < problem.edges.size(); ++index) {
    const auto& edge = problem.edges[index];
    if (edge.source >= vertex_count || edge.target >= vertex_count ||
        index > std::numeric_limits<std::uint32_t>::max()) {
      return planner_error(ErrorCode::kInvalidArgument, "PARTITION_PROPOSAL_CONTRACT_INVALID");
    }
    predecessors[edge.target].push_back(edge.source);
    successors[edge.source].push_back(edge.target);
    incoming_edges[edge.target].push_back(static_cast<EdgeIndex>(index));
  }

  const auto vertex_less = [&](VertexIndex lhs, VertexIndex rhs) {
    const auto lhs_id = problem.vertices[lhs].id.value;
    const auto rhs_id = problem.vertices[rhs].id.value;
    return lhs_id != rhs_id ? lhs_id < rhs_id : lhs < rhs;
  };
  for (VertexIndex vertex = 0; vertex < vertex_count; ++vertex) {
    sort_and_unique(predecessors[vertex], vertex_less);
    sort_and_unique(successors[vertex], vertex_less);
  }

  std::vector<std::uint32_t> in_degree(vertex_count);
  for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
    in_degree[vertex] = static_cast<std::uint32_t>(predecessors[vertex].size());
  }
  const auto ready_after = [&](VertexIndex lhs, VertexIndex rhs) { return vertex_less(rhs, lhs); };
  std::priority_queue<VertexIndex, std::vector<VertexIndex>, decltype(ready_after)> ready(
      ready_after);
  for (VertexIndex vertex = 0; vertex < vertex_count; ++vertex) {
    if (in_degree[vertex] == 0U) {
      ready.push(vertex);
    }
  }

  std::vector<VertexIndex> topological_order;
  topological_order.reserve(vertex_count);
  while (!ready.empty()) {
    const VertexIndex vertex = ready.top();
    ready.pop();
    topological_order.push_back(vertex);
    for (const VertexIndex successor : successors[vertex]) {
      if (--in_degree[successor] == 0U) {
        ready.push(successor);
      }
    }
  }
  if (topological_order.size() != vertex_count) {
    return planner_error(ErrorCode::kInvalidArgument, "PARTITION_PROPOSAL_CONTRACT_INVALID");
  }

  std::vector<CandidatePartitionId> partition_by_vertex(vertex_count);
  CandidateStructuralTracker structural_tracker(vertex_count);
  std::vector<ExecutionDomainKey> partition_domains(problem.authored_constraints.size());
  for (std::size_t constraint_index = 0U; constraint_index < problem.authored_constraints.size();
       ++constraint_index) {
    const auto& constraint = problem.authored_constraints[constraint_index];
    partition_domains[constraint_index] =
        problem.vertices[constraint.members.front()].execution_domain;
  }
  partition_domains.reserve(vertex_count);
  std::map<std::uint64_t, std::vector<CandidatePartitionId>> partitions_by_serialization_domain;
  CandidateQuotientGraph quotient_graph;
  PartitionProposal proposal;
  proposal.algorithm = source_chain_algorithm();
  proposal.assignments.reserve(vertex_count);
  proposal.decisions.reserve(vertex_count);

  for (const VertexIndex vertex_index : topological_order) {
    const auto& vertex = problem.vertices[vertex_index];
    std::vector<CandidatePartitionId> predecessor_partitions;
    predecessor_partitions.reserve(predecessors[vertex_index].size());
    for (const VertexIndex predecessor : predecessors[vertex_index]) {
      const CandidatePartitionId partition = partition_by_vertex[predecessor];
      if (!partition) {
        return planner_error(ErrorCode::kInvalidArgument, "PARTITION_PROPOSAL_CONTRACT_INVALID");
      }
      predecessor_partitions.push_back(partition);
    }
    std::sort(
        predecessor_partitions.begin(),
        predecessor_partitions.end(),
        [](CandidatePartitionId lhs, CandidatePartitionId rhs) { return lhs.value < rhs.value; });
    predecessor_partitions.erase(
        std::unique(predecessor_partitions.begin(), predecessor_partitions.end()),
        predecessor_partitions.end());

    if (authored_partition_by_vertex[vertex_index].has_value()) {
      const CandidatePartitionId selected_partition = *authored_partition_by_vertex[vertex_index];
      partition_by_vertex[vertex_index] = selected_partition;
      structural_tracker.add_vertex(partition_by_vertex,
                                    selected_partition,
                                    predecessors[vertex_index],
                                    static_cast<bool>(vertex.serialization_domain));
      proposal.assignments.push_back(VertexPartitionAssignment{vertex_index, selected_partition});
      proposal.decisions.push_back(PartitionDecisionProposal{
          vertex_index,
          selected_partition,
          PartitionDecisionKind::kAuthored,
          {PartitionDecisionReason::kAuthoredPartitionBoundary},
      });
      append_candidate_predecessor_edges(
          quotient_graph, selected_partition, predecessor_partitions);
      continue;
    }

    PartitionBoundaryMask mandatory = vertex.boundary_before;
    for (const EdgeIndex edge_index : incoming_edges[vertex_index]) {
      mandatory.add(problem.edges[edge_index].mandatory_boundaries);
    }

    std::vector<PartitionDecisionReason> reasons;
    append_boundary_reasons(mandatory, reasons);
    if (predecessors[vertex_index].empty()) {
      reasons.push_back(PartitionDecisionReason::kSourceRoot);
    } else if (vertex.starts_execution_chain) {
      reasons.push_back(PartitionDecisionReason::kIndependentReadinessRoot);
    }
    if (predecessor_partitions.size() > 1U) {
      reasons.push_back(PartitionDecisionReason::kFanIn);
    }
    if (predecessors[vertex_index].size() == 1U &&
        successors[predecessors[vertex_index].front()].size() > 1U) {
      reasons.push_back(PartitionDecisionReason::kFanOutLeg);
    }
    if (std::any_of(predecessor_partitions.begin(),
                    predecessor_partitions.end(),
                    [&](CandidatePartitionId candidate) {
                      return candidate.value <= problem.authored_constraints.size();
                    })) {
      reasons.push_back(PartitionDecisionReason::kAuthoredPartitionBoundary);
    }

    CandidatePartitionId selected_partition;
    PartitionDecisionKind decision = PartitionDecisionKind::kStart;
    if (vertex.serialization_domain) {
      reasons.push_back(PartitionDecisionReason::kSerializationDomain);
      const auto existing =
          partitions_by_serialization_domain.find(vertex.serialization_domain.value);
      if (existing != partitions_by_serialization_domain.end()) {
        for (const CandidatePartitionId candidate : existing->second) {
          selected_partition = candidate;
          bool can_join =
              selected_partition.value <= partition_domains.size() &&
              partition_domains[selected_partition.value - 1U] == vertex.execution_domain &&
              !candidate_join_would_cycle(
                  quotient_graph, selected_partition, predecessor_partitions);
          if (can_join) {
            for (const EdgeIndex edge_index : incoming_edges[vertex_index]) {
              const auto& edge = problem.edges[edge_index];
              if (partition_by_vertex[edge.source] == selected_partition &&
                  (!vertex.boundary_before.empty() || !edge.mandatory_boundaries.empty() ||
                   !edge.same_partition_collapse_legal || !edge.same_stream_fifo_legal)) {
                can_join = false;
                break;
              }
            }
          }
          if (can_join && structural_tracker.has_non_serialized_member(candidate) &&
              !structural_tracker.would_form_chain_with_vertex(
                  partition_by_vertex, candidate, predecessors[vertex_index])) {
            can_join = false;
          }
          if (can_join) {
            break;
          }
          selected_partition = {};
        }
      }
      if (!selected_partition) {
        if (partition_domains.size() >= problem.max_partitions ||
            partition_domains.size() >= vertex_count) {
          return planner_error(ErrorCode::kResourceExhausted, "PARTITION_LIMIT_EXCEEDED");
        }
        selected_partition =
            CandidatePartitionId{static_cast<std::uint32_t>(partition_domains.size() + 1U)};
        partition_domains.push_back(vertex.execution_domain);
        partitions_by_serialization_domain[vertex.serialization_domain.value].push_back(
            selected_partition);
      }
      decision = PartitionDecisionKind::kSerializationDomain;
    } else if (!reasons.empty()) {
      if (partition_domains.size() >= problem.max_partitions ||
          partition_domains.size() >= vertex_count) {
        return planner_error(ErrorCode::kResourceExhausted, "PARTITION_LIMIT_EXCEEDED");
      }
      selected_partition =
          CandidatePartitionId{static_cast<std::uint32_t>(partition_domains.size() + 1U)};
      partition_domains.push_back(vertex.execution_domain);
      const auto has_reason = [&](PartitionDecisionReason reason) {
        return std::find(reasons.begin(), reasons.end(), reason) != reasons.end();
      };
      const bool fan_in = has_reason(PartitionDecisionReason::kFanIn);
      const bool fan_out = has_reason(PartitionDecisionReason::kFanOutLeg);
      const bool follows_authored = has_reason(PartitionDecisionReason::kAuthoredPartitionBoundary);
      if (!mandatory.empty() || (follows_authored && !fan_in && !fan_out)) {
        decision = PartitionDecisionKind::kBoundarySplit;
      } else if (fan_in) {
        decision = PartitionDecisionKind::kFanInSplit;
      } else if (fan_out) {
        decision = PartitionDecisionKind::kFanOutLegSplit;
      }
    } else {
      bool can_join = predecessor_partitions.size() == 1U &&
                      predecessor_partitions.front().value > problem.authored_constraints.size();
      if (can_join) {
        selected_partition = predecessor_partitions.front();
        can_join = selected_partition.value <= partition_domains.size() &&
                   partition_domains[selected_partition.value - 1U] == vertex.execution_domain;
      }
      if (can_join) {
        for (const EdgeIndex edge_index : incoming_edges[vertex_index]) {
          const auto& edge = problem.edges[edge_index];
          if (partition_by_vertex[edge.source] != selected_partition ||
              !edge.same_partition_collapse_legal || !edge.same_stream_fifo_legal ||
              !edge.mandatory_boundaries.empty()) {
            can_join = false;
            break;
          }
        }
      }
      if (can_join && !structural_tracker.would_form_chain_with_vertex(
                          partition_by_vertex, selected_partition, predecessors[vertex_index])) {
        can_join = false;
      }

      if (can_join) {
        decision = PartitionDecisionKind::kJoin;
        reasons.push_back(PartitionDecisionReason::kLinearChainJoin);
      } else {
        if (partition_domains.size() >= problem.max_partitions ||
            partition_domains.size() >= vertex_count) {
          return planner_error(ErrorCode::kResourceExhausted, "PARTITION_LIMIT_EXCEEDED");
        }
        selected_partition =
            CandidatePartitionId{static_cast<std::uint32_t>(partition_domains.size() + 1U)};
        partition_domains.push_back(vertex.execution_domain);
        decision = PartitionDecisionKind::kCollapseRejected;
        reasons.push_back(PartitionDecisionReason::kCollapseRejected);
      }
    }

    partition_by_vertex[vertex_index] = selected_partition;
    structural_tracker.add_vertex(partition_by_vertex,
                                  selected_partition,
                                  predecessors[vertex_index],
                                  static_cast<bool>(vertex.serialization_domain));
    proposal.assignments.push_back(VertexPartitionAssignment{vertex_index, selected_partition});
    proposal.decisions.push_back(
        PartitionDecisionProposal{vertex_index, selected_partition, decision, std::move(reasons)});
    append_candidate_predecessor_edges(quotient_graph, selected_partition, predecessor_partitions);
  }

  return proposal;
}

std::unique_ptr<PartitionPlanner> make_source_chain_partition_planner() {
  return std::make_unique<SourceChainPartitionPlanner>();
}

}  // namespace holoscan::detail::partitioning
