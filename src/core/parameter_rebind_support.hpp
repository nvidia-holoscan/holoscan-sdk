/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <span>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/parameter_schema.hpp>

#include "graph_snapshot.hpp"
#include "runtime/executable_factory_inventory.hpp"

namespace holoscan::detail {

/// @brief Own the private compose-once evidence consumed by optional Parameter tooling.
struct ParameterRebindGraphEvidence {
  std::shared_ptr<const GraphSnapshotData> snapshot;
  ExecutableFactoryInventory factories;
  ParamSchemaHash admission_context_digest{};
};

/// @brief Capture the exact standard-profile compiler inputs used by a target-bound override set.
/// @param graph Mutable authoring Graph; lazy composition may run before the snapshot is captured.
/// @return Owning canonical snapshot, cloned factories, and standard admission-context digest.
/// @note This is a non-installed source-private seam exported only between Holoscan package DSOs.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<ParameterRebindGraphEvidence, Error>
capture_parameter_rebind_graph(Graph& graph) noexcept;

/// @brief Capture standard-profile evidence while cloning only selected canonical Operators.
/// @param graph Mutable authoring Graph that owns the selected factories.
/// @param snapshot Canonical snapshot used to resolve the explicit target scope.
/// @param selected_operators Non-empty canonical Operator vertex set.
/// @return Evidence retaining only selected Operator factories and no clock factories.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<ParameterRebindGraphEvidence, Error>
capture_parameter_rebind_graph(Graph& graph, std::shared_ptr<const GraphSnapshotData> snapshot,
                               std::span<const VertexId> selected_operators) noexcept;

}  // namespace holoscan::detail
