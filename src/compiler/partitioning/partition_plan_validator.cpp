/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "partition_plan_validator.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <queue>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace holoscan::detail::partitioning {
namespace {

[[nodiscard]] unexpected<Error> validation_error(ErrorCode code, std::string_view token) {
  return make_unexpected(Error{code, std::string(token)});
}

[[nodiscard]] bool same_algorithm(const AlgorithmKey& lhs, const AlgorithmKey& rhs) noexcept {
  return lhs.id == rhs.id && lhs.version == rhs.version;
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

[[nodiscard]] bool reasons_are_canonical(
    const std::vector<PartitionDecisionReason>& reasons) noexcept {
  using ReasonValue = std::underlying_type_t<PartitionDecisionReason>;
  ReasonValue previous{};
  bool first = true;
  for (const PartitionDecisionReason reason : reasons) {
    const ReasonValue value = static_cast<ReasonValue>(reason);
    if (value > static_cast<ReasonValue>(PartitionDecisionReason::kLinearChainJoin) ||
        (!first && value <= previous)) {
      return false;
    }
    previous = value;
    first = false;
  }
  return true;
}

template <typename T, typename Compare>
void sort_and_unique(std::vector<T>& values, Compare compare) {
  std::sort(values.begin(), values.end(), compare);
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

}  // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<ValidatedPartitionPlan, Error> validate_partition_proposal(
    const PartitionProblem& problem, const PartitionPlannerDescriptor& selected_planner,
    const PartitionProposal& proposal) {
  if (selected_planner.interface_version != kPartitionPlannerInterfaceVersion) {
    return validation_error(ErrorCode::kNotSupported,
                            "PARTITION_PLANNER_INTERFACE_VERSION_UNSUPPORTED");
  }
  if (!problem.available_capabilities.contains(selected_planner.required_capabilities)) {
    return validation_error(ErrorCode::kNotSupported, "PARTITION_PLANNER_CAPABILITY_UNSUPPORTED");
  }
  if (!same_algorithm(selected_planner.algorithm, proposal.algorithm)) {
    return validation_error(ErrorCode::kIncompatible, "PARTITION_PROPOSAL_ALGORITHM_MISMATCH");
  }

  const std::size_t vertex_count = problem.vertices.size();
  if (vertex_count > std::numeric_limits<std::uint32_t>::max()) {
    return validation_error(ErrorCode::kResourceExhausted, "PARTITION_LIMIT_EXCEEDED");
  }
  if (proposal.assignments.size() != vertex_count || proposal.decisions.size() != vertex_count) {
    return validation_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
  }

  std::vector<std::vector<VertexIndex>> predecessors(vertex_count);
  std::vector<std::vector<VertexIndex>> successors(vertex_count);
  std::vector<std::vector<EdgeIndex>> incoming_edges(vertex_count);
  std::vector<VertexId> vertex_ids;
  vertex_ids.reserve(vertex_count);
  for (const auto& vertex : problem.vertices) {
    if (!vertex.id) {
      return validation_error(ErrorCode::kInvalidArgument, "PARTITION_PROPOSAL_CONTRACT_INVALID");
    }
    vertex_ids.push_back(vertex.id);
  }
  std::sort(vertex_ids.begin(), vertex_ids.end(), [](VertexId lhs, VertexId rhs) {
    return lhs.value < rhs.value;
  });
  if (std::adjacent_find(vertex_ids.begin(), vertex_ids.end()) != vertex_ids.end()) {
    return validation_error(ErrorCode::kInvalidArgument, "PARTITION_PROPOSAL_CONTRACT_INVALID");
  }

  for (std::size_t edge_index = 0; edge_index < problem.edges.size(); ++edge_index) {
    const auto& edge = problem.edges[edge_index];
    if (!edge.id || edge.source >= vertex_count || edge.target >= vertex_count ||
        edge_index > std::numeric_limits<std::uint32_t>::max()) {
      return validation_error(ErrorCode::kInvalidArgument, "PARTITION_PROPOSAL_CONTRACT_INVALID");
    }
    predecessors[edge.target].push_back(edge.source);
    successors[edge.source].push_back(edge.target);
    incoming_edges[edge.target].push_back(static_cast<EdgeIndex>(edge_index));
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

  std::vector<std::optional<CandidatePartitionId>> assignment_by_vertex(vertex_count);
  std::map<std::uint32_t, std::vector<VertexIndex>> members_by_partition;
  for (const auto& assignment : proposal.assignments) {
    if (assignment.vertex >= vertex_count || !assignment.partition ||
        assignment_by_vertex[assignment.vertex].has_value()) {
      return validation_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
    }
    assignment_by_vertex[assignment.vertex] = assignment.partition;
    members_by_partition[assignment.partition.value].push_back(assignment.vertex);
  }
  if (std::any_of(assignment_by_vertex.begin(),
                  assignment_by_vertex.end(),
                  [](const auto& assignment) { return !assignment.has_value(); })) {
    return validation_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
  }
  if (members_by_partition.size() > vertex_count ||
      members_by_partition.size() > problem.max_partitions) {
    return validation_error(ErrorCode::kResourceExhausted, "PARTITION_LIMIT_EXCEEDED");
  }

  std::vector<const PartitionDecisionProposal*> decision_by_vertex(vertex_count, nullptr);
  for (const auto& decision : proposal.decisions) {
    if (decision.vertex >= vertex_count || decision_by_vertex[decision.vertex] != nullptr ||
        !decision.partition ||
        decision.partition != assignment_by_vertex[decision.vertex].value() ||
        !reasons_are_canonical(decision.reasons) ||
        static_cast<std::underlying_type_t<PartitionDecisionKind>>(decision.decision) >
            static_cast<std::underlying_type_t<PartitionDecisionKind>>(
                PartitionDecisionKind::kCollapseRejected)) {
      return validation_error(ErrorCode::kInvalidArgument, "PARTITION_DECISION_REASON_INVALID");
    }
    decision_by_vertex[decision.vertex] = &decision;
  }
  if (std::any_of(decision_by_vertex.begin(), decision_by_vertex.end(), [](const auto* decision) {
        return decision == nullptr;
      })) {
    return validation_error(ErrorCode::kInvalidArgument, "PARTITION_PROPOSAL_CONTRACT_INVALID");
  }

  for (const auto& edge : problem.edges) {
    const auto source_partition = assignment_by_vertex[edge.source].value();
    const auto target_partition = assignment_by_vertex[edge.target].value();
    if (source_partition != target_partition) {
      continue;
    }
    if (!edge.mandatory_boundaries.empty() ||
        !problem.vertices[edge.target].boundary_before.empty()) {
      return validation_error(ErrorCode::kIncompatible, "PARTITION_BOUNDARY_CROSSED");
    }
    if (!edge.same_partition_collapse_legal || !edge.same_stream_fifo_legal) {
      return validation_error(ErrorCode::kIncompatible, "PARTITION_COLLAPSE_ILLEGAL");
    }
  }

  // A candidate partition is one execution domain. Boundary masks decide where
  // that domain must split; they never authorize collapsing two normalized
  // domains merely because no direct edge happens to connect their members.
  std::map<std::uint32_t, ExecutionDomainKey> domain_by_partition;
  for (VertexIndex vertex = 0; vertex < vertex_count; ++vertex) {
    const std::uint32_t candidate = assignment_by_vertex[vertex]->value;
    const auto [entry, inserted] =
        domain_by_partition.try_emplace(candidate, problem.vertices[vertex].execution_domain);
    if (!inserted && entry->second != problem.vertices[vertex].execution_domain) {
      return validation_error(ErrorCode::kIncompatible, "PARTITION_COLLAPSE_ILLEGAL");
    }
  }

  // Contracting a DAG can introduce a cycle when one candidate partition contains non-contiguous
  // graph segments. Recheck the quotient before the candidate cover gains compiler authority.
  std::map<std::uint32_t, std::size_t> quotient_index;
  for (const auto& [candidate, members] : members_by_partition) {
    static_cast<void>(members);
    quotient_index.emplace(candidate, quotient_index.size());
  }
  std::vector<std::vector<std::size_t>> quotient_successors(quotient_index.size());
  for (const auto& edge : problem.edges) {
    const auto source_candidate = assignment_by_vertex[edge.source].value().value;
    const auto target_candidate = assignment_by_vertex[edge.target].value().value;
    if (source_candidate != target_candidate) {
      quotient_successors[quotient_index.at(source_candidate)].push_back(
          quotient_index.at(target_candidate));
    }
  }
  std::vector<std::uint32_t> quotient_in_degree(quotient_index.size());
  for (auto& successors_for_partition : quotient_successors) {
    std::sort(successors_for_partition.begin(), successors_for_partition.end());
    successors_for_partition.erase(
        std::unique(successors_for_partition.begin(), successors_for_partition.end()),
        successors_for_partition.end());
    for (const std::size_t successor : successors_for_partition) {
      ++quotient_in_degree[successor];
    }
  }
  std::queue<std::size_t> quotient_ready;
  for (std::size_t index = 0; index < quotient_in_degree.size(); ++index) {
    if (quotient_in_degree[index] == 0U) {
      quotient_ready.push(index);
    }
  }
  std::size_t visited_partitions = 0;
  while (!quotient_ready.empty()) {
    const auto partition = quotient_ready.front();
    quotient_ready.pop();
    ++visited_partitions;
    for (const std::size_t successor : quotient_successors[partition]) {
      if (--quotient_in_degree[successor] == 0U) {
        quotient_ready.push(successor);
      }
    }
  }
  if (visited_partitions != quotient_index.size()) {
    return validation_error(ErrorCode::kIncompatible, "PARTITION_QUOTIENT_CYCLE");
  }

  struct ExpectedDecision {
    PartitionDecisionKind kind{PartitionDecisionKind::kStart};
    std::vector<PartitionDecisionReason> reasons;
  };
  std::vector<std::uint32_t> expected_partition_by_vertex(vertex_count);
  std::vector<ExpectedDecision> expected_decisions(vertex_count);

  const auto mandatory_for = [&](VertexIndex vertex) {
    PartitionBoundaryMask mandatory = problem.vertices[vertex].boundary_before;
    for (const EdgeIndex edge_index : incoming_edges[vertex]) {
      mandatory.add(problem.edges[edge_index].mandatory_boundaries);
    }
    return mandatory;
  };

  const bool serial_baseline = selected_planner.algorithm.id == kSerialBaselineAlgorithmId &&
                               selected_planner.algorithm.version == kPartitionAlgorithmVersion;
  const bool source_chain = selected_planner.algorithm.id == kSourceChainAlgorithmId &&
                            selected_planner.algorithm.version == kPartitionAlgorithmVersion;
  if (!serial_baseline && !source_chain) {
    return validation_error(ErrorCode::kNotSupported, "PARTITION_PLANNER_NOT_FOUND");
  }

  if (serial_baseline) {
    // Reconstruct the coarsest domain cover from compiler-normalized facts.
    // Proposal-local candidate ids may differ, but their equivalence classes
    // and the first/start versus subsequent/join decisions may not.
    std::map<std::uint64_t, std::uint32_t> partition_by_domain;
    std::vector<VertexIndex> canonical_vertices(vertex_count);
    for (VertexIndex vertex = 0; vertex < vertex_count; ++vertex) {
      canonical_vertices[vertex] = vertex;
    }
    std::sort(canonical_vertices.begin(), canonical_vertices.end(), vertex_less);
    for (const VertexIndex vertex : canonical_vertices) {
      const auto [entry, inserted] = partition_by_domain.try_emplace(
          problem.vertices[vertex].execution_domain.value,
          static_cast<std::uint32_t>(partition_by_domain.size() + 1U));
      expected_partition_by_vertex[vertex] = entry->second;
      auto& expected = expected_decisions[vertex];
      const PartitionBoundaryMask mandatory = mandatory_for(vertex);
      append_boundary_reasons(mandatory, expected.reasons);
      expected.kind = !mandatory.empty() ? PartitionDecisionKind::kBoundarySplit
                                         : (inserted ? PartitionDecisionKind::kStart
                                                     : PartitionDecisionKind::kJoin);
    }
  } else {
    // Reconstruct source-chain legality without trusting the planner's claimed
    // decision kind or reasons. In particular, a proposal cannot manufacture a
    // collapse-rejected reason to split a legal join, or call an edge-crossing
    // assignment a join into an unrelated partition.
    std::vector<std::uint32_t> in_degree(vertex_count);
    for (VertexIndex vertex = 0; vertex < vertex_count; ++vertex) {
      in_degree[vertex] = static_cast<std::uint32_t>(predecessors[vertex].size());
    }
    const auto ready_after = [&](VertexIndex lhs, VertexIndex rhs) {
      return vertex_less(rhs, lhs);
    };
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
      return validation_error(ErrorCode::kInvalidArgument, "PARTITION_PROPOSAL_CONTRACT_INVALID");
    }

    std::vector<ExecutionDomainKey> expected_partition_domains;
    expected_partition_domains.reserve(vertex_count);
    for (const VertexIndex vertex : topological_order) {
      std::vector<std::uint32_t> predecessor_partitions;
      predecessor_partitions.reserve(predecessors[vertex].size());
      for (const VertexIndex predecessor : predecessors[vertex]) {
        predecessor_partitions.push_back(expected_partition_by_vertex[predecessor]);
      }
      std::sort(predecessor_partitions.begin(), predecessor_partitions.end());
      predecessor_partitions.erase(
          std::unique(predecessor_partitions.begin(), predecessor_partitions.end()),
          predecessor_partitions.end());

      auto& expected = expected_decisions[vertex];
      const PartitionBoundaryMask mandatory = mandatory_for(vertex);
      append_boundary_reasons(mandatory, expected.reasons);
      if (predecessors[vertex].empty()) {
        expected.reasons.push_back(PartitionDecisionReason::kSourceRoot);
      } else if (problem.vertices[vertex].starts_execution_chain) {
        expected.reasons.push_back(PartitionDecisionReason::kIndependentReadinessRoot);
      }
      if (predecessor_partitions.size() > 1U) {
        expected.reasons.push_back(PartitionDecisionReason::kFanIn);
      }
      if (predecessors[vertex].size() == 1U &&
          successors[predecessors[vertex].front()].size() > 1U) {
        expected.reasons.push_back(PartitionDecisionReason::kFanOutLeg);
      }

      if (!expected.reasons.empty()) {
        if (expected_partition_domains.size() >= problem.max_partitions ||
            expected_partition_domains.size() >= vertex_count) {
          return validation_error(ErrorCode::kResourceExhausted, "PARTITION_LIMIT_EXCEEDED");
        }
        expected_partition_by_vertex[vertex] =
            static_cast<std::uint32_t>(expected_partition_domains.size() + 1U);
        expected_partition_domains.push_back(problem.vertices[vertex].execution_domain);
        if (!mandatory.empty()) {
          expected.kind = PartitionDecisionKind::kBoundarySplit;
        } else if (std::find(expected.reasons.begin(),
                             expected.reasons.end(),
                             PartitionDecisionReason::kFanIn) != expected.reasons.end()) {
          expected.kind = PartitionDecisionKind::kFanInSplit;
        } else if (std::find(expected.reasons.begin(),
                             expected.reasons.end(),
                             PartitionDecisionReason::kFanOutLeg) != expected.reasons.end()) {
          expected.kind = PartitionDecisionKind::kFanOutLegSplit;
        }
        continue;
      }

      bool can_join = predecessor_partitions.size() == 1U;
      std::uint32_t selected_partition{};
      if (can_join) {
        selected_partition = predecessor_partitions.front();
        can_join = selected_partition > 0U &&
                   selected_partition <= expected_partition_domains.size() &&
                   expected_partition_domains[selected_partition - 1U] ==
                       problem.vertices[vertex].execution_domain;
      }
      if (can_join) {
        for (const EdgeIndex edge_index : incoming_edges[vertex]) {
          const auto& edge = problem.edges[edge_index];
          if (expected_partition_by_vertex[edge.source] != selected_partition ||
              !edge.same_partition_collapse_legal || !edge.same_stream_fifo_legal ||
              !edge.mandatory_boundaries.empty()) {
            can_join = false;
            break;
          }
        }
      }

      if (can_join) {
        expected_partition_by_vertex[vertex] = selected_partition;
        expected.kind = PartitionDecisionKind::kJoin;
        expected.reasons.push_back(PartitionDecisionReason::kLinearChainJoin);
      } else {
        if (expected_partition_domains.size() >= problem.max_partitions ||
            expected_partition_domains.size() >= vertex_count) {
          return validation_error(ErrorCode::kResourceExhausted, "PARTITION_LIMIT_EXCEEDED");
        }
        expected_partition_by_vertex[vertex] =
            static_cast<std::uint32_t>(expected_partition_domains.size() + 1U);
        expected_partition_domains.push_back(problem.vertices[vertex].execution_domain);
        expected.kind = PartitionDecisionKind::kCollapseRejected;
        expected.reasons.push_back(PartitionDecisionReason::kCollapseRejected);
      }
    }
  }

  // Candidate ids are proposal-local, so compare the independently derived
  // partition equivalence relation in both directions rather than comparing
  // numeric ids. This rejects both illegal merges and gratuitous splits.
  std::map<std::uint32_t, std::uint32_t> candidate_by_expected_partition;
  std::map<std::uint32_t, std::uint32_t> expected_partition_by_candidate;
  for (VertexIndex vertex = 0; vertex < vertex_count; ++vertex) {
    const std::uint32_t expected_partition = expected_partition_by_vertex[vertex];
    const std::uint32_t candidate = assignment_by_vertex[vertex]->value;
    const auto [expected_entry, expected_inserted] =
        candidate_by_expected_partition.try_emplace(expected_partition, candidate);
    const auto [candidate_entry, candidate_inserted] =
        expected_partition_by_candidate.try_emplace(candidate, expected_partition);
    if ((!expected_inserted && expected_entry->second != candidate) ||
        (!candidate_inserted && candidate_entry->second != expected_partition)) {
      return validation_error(ErrorCode::kInvalidArgument, "PARTITION_COVER_INVALID");
    }
    const auto& decision = *decision_by_vertex[vertex];
    if (decision.decision != expected_decisions[vertex].kind ||
        decision.reasons != expected_decisions[vertex].reasons) {
      return validation_error(ErrorCode::kInvalidArgument, "PARTITION_DECISION_REASON_INVALID");
    }
  }

  struct CandidateMembers {
    std::uint32_t candidate{};
    std::vector<VertexId> members;
  };
  std::vector<CandidateMembers> canonical_candidates;
  canonical_candidates.reserve(members_by_partition.size());
  for (auto& [candidate, member_indices] : members_by_partition) {
    CandidateMembers output{candidate, {}};
    output.members.reserve(member_indices.size());
    for (const VertexIndex vertex : member_indices) {
      output.members.push_back(problem.vertices[vertex].id);
    }
    std::sort(output.members.begin(), output.members.end(), [](VertexId lhs, VertexId rhs) {
      return lhs.value < rhs.value;
    });
    canonical_candidates.push_back(std::move(output));
  }
  std::sort(canonical_candidates.begin(),
            canonical_candidates.end(),
            [](const CandidateMembers& lhs, const CandidateMembers& rhs) {
              return std::lexicographical_compare(
                  lhs.members.begin(),
                  lhs.members.end(),
                  rhs.members.begin(),
                  rhs.members.end(),
                  [](VertexId left, VertexId right) { return left.value < right.value; });
            });

  std::map<std::uint32_t, std::uint32_t> canonical_id_by_candidate;
  ValidatedPartitionPlan result;
  result.algorithm = selected_planner.algorithm;
  result.partitions.reserve(canonical_candidates.size());
  for (std::size_t index = 0; index < canonical_candidates.size(); ++index) {
    const std::uint32_t canonical_id = static_cast<std::uint32_t>(index + 1U);
    canonical_id_by_candidate.emplace(canonical_candidates[index].candidate, canonical_id);
    result.partitions.push_back(
        CanonicalPartition{canonical_id, std::move(canonical_candidates[index].members)});
  }

  result.decisions.reserve(vertex_count);
  for (const auto& decision : proposal.decisions) {
    result.decisions.push_back(
        CanonicalPartitionDecision{problem.vertices[decision.vertex].id,
                                   canonical_id_by_candidate.at(decision.partition.value),
                                   decision.decision,
                                   decision.reasons});
  }
  std::sort(result.decisions.begin(), result.decisions.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.vertex.value < rhs.vertex.value;
  });
  return result;
}

}  // namespace holoscan::detail::partitioning
