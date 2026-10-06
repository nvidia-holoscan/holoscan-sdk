/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

#include "compiler/partitioning/partition_plan_validator.hpp"
#include "compiler/partitioning/serial_baseline_partition_planner.hpp"
#include "compiler/partitioning/source_chain_partition_planner.hpp"

namespace holoscan::test {

struct PartitionProblemStorage {
  explicit PartitionProblemStorage(
      std::size_t vertex_count,
      std::initializer_list<std::pair<std::uint32_t, std::uint32_t>> edge_pairs = {})
      : incoming(vertex_count),
        outgoing(vertex_count),
        device_obligations(vertex_count),
        source_regions(vertex_count) {
    edges.reserve(edge_pairs.size());
    std::uint32_t edge_id = 1U;
    for (const auto [source, target] : edge_pairs) {
      const auto edge_index = static_cast<detail::partitioning::EdgeIndex>(edges.size());
      outgoing.at(source).push_back(edge_index);
      incoming.at(target).push_back(edge_index);
      edges.push_back(detail::partitioning::PartitionEdgeView{
          EdgeId{edge_id++}, source, target, {}, true, true});
    }

    vertices.reserve(vertex_count);
    for (std::uint32_t index = 0; index < vertex_count; ++index) {
      vertices.push_back(detail::partitioning::PartitionVertexView{
          VertexId{index + 1U},
          index,
          incoming[index],
          outgoing[index],
          false,
          detail::partitioning::ExecutionDomainKey{1U},
          {},
          device_obligations[index],
          {},
          source_regions[index],
      });
    }
  }

  [[nodiscard]] detail::partitioning::PartitionProblem problem(
      std::uint32_t max_partitions = UINT32_MAX) const {
    return detail::partitioning::PartitionProblem{
        vertices,
        edges,
        {},
        max_partitions,
        detail::partitioning::kAllPartitionPlannerCapabilities,
    };
  }

  std::vector<std::vector<detail::partitioning::EdgeIndex>> incoming;
  std::vector<std::vector<detail::partitioning::EdgeIndex>> outgoing;
  std::vector<std::vector<std::int32_t>> device_obligations;
  std::vector<std::vector<RegionId>> source_regions;
  std::vector<detail::partitioning::PartitionVertexView> vertices;
  std::vector<detail::partitioning::PartitionEdgeView> edges;
};

[[nodiscard]] inline expected<detail::partitioning::ValidatedPartitionPlan, Error> run_source_chain(
    const detail::partitioning::PartitionProblem& problem) {
  detail::partitioning::SourceChainPartitionPlanner planner;
  auto proposal = planner.propose(problem);
  if (!proposal) {
    return forward_error(std::move(proposal));
  }
  return detail::partitioning::validate_partition_proposal(
      problem, planner.descriptor(), proposal.value());
}

[[nodiscard]] inline expected<detail::partitioning::ValidatedPartitionPlan, Error>
run_serial_baseline(const detail::partitioning::PartitionProblem& problem) {
  detail::partitioning::SerialBaselinePartitionPlanner planner;
  auto proposal = planner.propose(problem);
  if (!proposal) {
    return forward_error(std::move(proposal));
  }
  return detail::partitioning::validate_partition_proposal(
      problem, planner.descriptor(), proposal.value());
}

}  // namespace holoscan::test
