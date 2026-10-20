/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <vector>

#include "partition_types.hpp"

namespace holoscan::detail::partitioning {

using CandidateQuotientGraph = std::map<std::uint32_t, std::vector<std::uint32_t>>;

/// Return whether one already-built candidate partition can reach another.
inline bool candidate_partition_reaches(const CandidateQuotientGraph& graph,
                                        CandidatePartitionId source, CandidatePartitionId target) {
  if (source == target) {
    return true;
  }
  std::vector<std::uint32_t> pending{source.value};
  std::vector<std::uint32_t> visited;
  while (!pending.empty()) {
    const std::uint32_t current = pending.back();
    pending.pop_back();
    if (std::find(visited.begin(), visited.end(), current) != visited.end()) {
      continue;
    }
    visited.push_back(current);
    const auto successors = graph.find(current);
    if (successors == graph.end()) {
      continue;
    }
    for (const std::uint32_t successor : successors->second) {
      if (successor == target.value) {
        return true;
      }
      pending.push_back(successor);
    }
  }
  return false;
}

/// Return whether adding predecessor edges into a candidate would cycle the quotient graph.
inline bool candidate_join_would_cycle(
    const CandidateQuotientGraph& graph, CandidatePartitionId candidate,
    const std::vector<CandidatePartitionId>& predecessor_partitions) {
  return std::any_of(predecessor_partitions.begin(),
                     predecessor_partitions.end(),
                     [&](CandidatePartitionId predecessor) {
                       return predecessor != candidate &&
                              candidate_partition_reaches(graph, candidate, predecessor);
                     });
}

/// Add the distinct crossing predecessor edges for one newly assigned vertex.
inline void append_candidate_predecessor_edges(
    CandidateQuotientGraph& graph, CandidatePartitionId candidate,
    const std::vector<CandidatePartitionId>& predecessor_partitions) {
  graph.try_emplace(candidate.value);
  for (const CandidatePartitionId predecessor : predecessor_partitions) {
    if (predecessor == candidate) {
      continue;
    }
    auto& predecessor_successors = graph[predecessor.value];
    predecessor_successors.push_back(candidate.value);
    std::sort(predecessor_successors.begin(), predecessor_successors.end());
    predecessor_successors.erase(
        std::unique(predecessor_successors.begin(), predecessor_successors.end()),
        predecessor_successors.end());
  }
}

/// Track candidate chain structure incrementally during a topological traversal.
class CandidateStructuralTracker {
 public:
  explicit CandidateStructuralTracker(std::size_t vertex_count)
      : internal_out_degree_(vertex_count) {}

  /// Return whether a candidate already contains an ordinary non-serialized member.
  [[nodiscard]] bool has_non_serialized_member(CandidatePartitionId candidate) const {
    const auto found = structure_by_candidate_.find(candidate.value);
    return found != structure_by_candidate_.end() && found->second.has_non_serialized_member;
  }

  /// Return whether adding one vertex would leave a connected, non-branching chain.
  [[nodiscard]] bool would_form_chain_with_vertex(
      std::span<const CandidatePartitionId> partition_by_vertex, CandidatePartitionId candidate,
      std::span<const VertexIndex> predecessors) const {
    CandidateStructure prospective;
    const auto found = structure_by_candidate_.find(candidate.value);
    if (found != structure_by_candidate_.end()) {
      prospective = found->second;
    }
    std::size_t internal_in_degree = 0U;
    for (const VertexIndex predecessor : predecessors) {
      if (partition_by_vertex[predecessor] == candidate) {
        ++internal_in_degree;
        prospective.degree_valid &= internal_out_degree_[predecessor] == 0U;
      }
    }
    prospective.degree_valid &= internal_in_degree <= 1U;
    prospective.simple_edge_count += internal_in_degree;
    ++prospective.member_count;
    return forms_chain(prospective);
  }

  /// Record one assigned vertex after its candidate has been selected.
  void add_vertex(std::span<const CandidatePartitionId> partition_by_vertex,
                  CandidatePartitionId candidate, std::span<const VertexIndex> predecessors,
                  bool serialized) {
    auto& structure = structure_by_candidate_[candidate.value];
    std::size_t internal_in_degree = 0U;
    for (const VertexIndex predecessor : predecessors) {
      if (partition_by_vertex[predecessor] == candidate) {
        ++internal_in_degree;
        structure.degree_valid &= internal_out_degree_[predecessor] == 0U;
        ++internal_out_degree_[predecessor];
      }
    }
    structure.degree_valid &= internal_in_degree <= 1U;
    structure.simple_edge_count += internal_in_degree;
    ++structure.member_count;
    structure.has_non_serialized_member |= !serialized;
  }

  /// Return whether the complete candidate is a connected, non-branching chain.
  [[nodiscard]] bool forms_chain(CandidatePartitionId candidate) const {
    const auto found = structure_by_candidate_.find(candidate.value);
    return found != structure_by_candidate_.end() && forms_chain(found->second);
  }

 private:
  struct CandidateStructure {
    std::size_t member_count{};
    std::size_t simple_edge_count{};
    bool degree_valid{true};
    bool has_non_serialized_member{};
  };

  [[nodiscard]] static bool forms_chain(const CandidateStructure& structure) {
    // The source graph is a DAG. With internal in/out degree at most one, every component is a
    // path, so n - 1 distinct internal edges also proves that exactly one component remains.
    return structure.degree_valid && structure.member_count != 0U &&
           structure.simple_edge_count + 1U == structure.member_count;
  }

  std::map<std::uint32_t, CandidateStructure> structure_by_candidate_;
  std::vector<std::size_t> internal_out_degree_;
};

/// Append mandatory boundary reasons in their canonical decision order.
inline void append_boundary_reasons(PartitionBoundaryMask mask,
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

/// Sort values with the supplied canonical comparator and remove duplicates.
template <typename T, typename Compare>
void sort_and_unique(std::vector<T>& values, Compare compare) {
  std::sort(values.begin(), values.end(), compare);
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

}  // namespace holoscan::detail::partitioning
