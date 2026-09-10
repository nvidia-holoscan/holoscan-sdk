/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>

#include "../core/graph_snapshot.hpp"
#include "execution_plan_state.hpp"
#include "partitioning/partition_types.hpp"

namespace holoscan::detail {

struct TensorDeviceBindingRecord;

/// Own the vectors borrowed by one normalized private partition-planner problem.
struct PartitionProblemStorage {
  std::vector<std::vector<partitioning::EdgeIndex>> incoming;
  std::vector<std::vector<partitioning::EdgeIndex>> outgoing;
  std::vector<std::vector<std::int32_t>> device_obligations;
  std::vector<std::vector<RegionId>> source_regions;
  std::vector<partitioning::PartitionVertexView> vertices;
  std::vector<partitioning::PartitionEdgeView> edges;
  std::vector<std::vector<partitioning::VertexIndex>> authored_constraint_members;
  std::vector<partitioning::AuthoredPartitionConstraintView> authored_constraints;
  std::unordered_map<std::uint64_t, partitioning::VertexIndex> executable_index;
};

/// Return a partition-lowering error with one stable diagnostic token.
[[nodiscard]] inline unexpected<Error> partition_lowering_error(ErrorCode code,
                                                                std::string_view token) {
  return make_unexpected(Error{code, std::string(token)});
}

/// Return whether a graph vertex participates in the executable partition cover.
[[nodiscard]] constexpr bool partition_executable(VertexKind kind) noexcept {
  return kind == VertexKind::kOperator || kind == VertexKind::kImportedTopic ||
         kind == VertexKind::kExportedTopic;
}

/// Build the storage and borrowed views for one normalized planner problem.
[[nodiscard]] expected<PartitionProblemStorage, Error> make_partition_problem_storage(
    const GraphSnapshotData& snapshot, std::span<const std::uint32_t> process_group_by_vertex,
    const DeploymentBinding& deployment,
    std::span<const TensorDeviceBindingRecord> tensor_device_bindings,
    const ExecutionPlanState& plan);

/// Add exact authored constraints to a normalized planner problem.
[[nodiscard]] expected<void, Error> append_authored_partition_constraints(
    const GraphSnapshotData& snapshot, PartitionProblemStorage& storage);

/// Return whether the selected temporal contract gives one Operator an independent root.
[[nodiscard]] bool has_independent_temporal_source(const ExecutionPlanState& plan, VertexId owner);

}  // namespace holoscan::detail
