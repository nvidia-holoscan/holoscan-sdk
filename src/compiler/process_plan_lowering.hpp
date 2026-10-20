/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <holoscan/core/execution_plan.hpp>

#include "core/graph_snapshot.hpp"
#include "process_placement_internal.hpp"
#include "process_plan_records.hpp"

namespace holoscan::detail {

/// @brief Preserve normalized process-placement records and private joins.
struct ProcessPlanLoweringResult {
  std::vector<ProcessRolePlanRecord> roles;                ///< Canonical deployment roles.
  std::vector<ProcessSelectorPlanRecord> selectors;        ///< Canonical selector resolutions.
  std::vector<VertexProcessPlacementPlanRecord> vertices;  ///< Total vertex placement map.
  std::vector<std::uint32_t> process_group_by_vertex;      ///< VertexId-indexed private join.
  std::vector<ChannelLocalityKind> locality_by_edge;       ///< EdgeId-indexed private join.
  std::vector<std::string> error_tokens;                   ///< Stable structural diagnostics.
};

/// @brief Normalize semantic process placement without starting or probing a process.
/// @param snapshot Canonical pointer-free Graph snapshot.
/// @param placement Deep-owned source-private placement bindings.
/// @return Total role, selector, vertex-group, and internal-edge locality evidence.
/// @note Runtime identity sealing and process/adapter construction belong to later components.
[[nodiscard]] ProcessPlanLoweringResult lower_process_plan(
    const GraphSnapshotData& snapshot, const ProcessPlacementBinding& placement);

/// @brief Apply compile-derived internal process locality to physical edge records.
/// @param locality_by_edge Canonical EdgeId-indexed locality projection.
/// @param payload_edges Mutable cold physical edge records.
/// @param channel_locality Mutable cold locality records.
/// @note External import/export classifications take precedence over process-group locality.
void apply_process_edge_locality(std::span<const ChannelLocalityKind> locality_by_edge,
                                 std::span<PayloadEdgePlanRecord> payload_edges,
                                 std::span<ChannelLocalityRecord> channel_locality) noexcept;

}  // namespace holoscan::detail
