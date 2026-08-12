/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_core_export.h>

#include <holoscan/core/compile.hpp>

#include "process_placement_internal.hpp"
#include "runtime_package_capabilities.hpp"

namespace holoscan::detail {

struct ExecutionPlanState;
struct PhysicalPlanLoweringResult;

/// @brief Move every physical lowering record family into an execution-plan state.
/// @note This private seam is shared by production publication and compiler conformance tests.
HOLOSCAN_CORE_EXPORT void publish_physical_plan_records(PhysicalPlanLoweringResult&& physical,
                                                        ExecutionPlanState& state);

/// @brief Compile with one already captured immutable Runtime evidence view.
/// @param graph Editable Graph whose composed generation is snapshotted.
/// @param options Owned public compile inputs.
/// @param evidence Source-private package and qualification evidence captured once.
/// @return Accepted or rejected immutable plan with structural diagnostics.
/// @throws RuntimeError under the same process-resource conditions as
/// `holoscan::compile()`.
/// @note This source-private entry point exists for generated provider integration
/// and conformance tests. It is not installed as public SDK API.
[[nodiscard]] HOLOSCAN_CORE_EXPORT ExecutionPlan compile_with_runtime_evidence(
    Graph& graph, CompileOptions options, RuntimeCompileEvidenceView evidence);

/// @brief Compile with source-private process placement used by conformance tests and benchmarks.
[[nodiscard]] HOLOSCAN_CORE_EXPORT ExecutionPlan compile_with_process_placement(
    Graph& graph, CompileOptions options, ProcessPlacementBinding placement);

/// @brief Compile with both source-private Runtime evidence and process placement.
[[nodiscard]] HOLOSCAN_CORE_EXPORT ExecutionPlan
compile_with_runtime_evidence_and_process_placement(Graph& graph, CompileOptions options,
                                                    RuntimeCompileEvidenceView evidence,
                                                    ProcessPlacementBinding placement);

/// @brief Compile while conservatively retaining HoloMQ for process-local inline edges.
/// @param graph Editable Graph whose composed generation is snapshotted.
/// @param options Owned public compile inputs.
/// @return Accepted or rejected immutable plan with structural diagnostics.
/// @throws RuntimeError under the same process-resource conditions as
/// `holoscan::compile()`.
/// @note This source-private entry point exists only for exact conformance and
/// performance comparisons. It is not installed as public SDK API.
[[nodiscard]] HOLOSCAN_CORE_EXPORT ExecutionPlan
compile_with_process_local_holomq(Graph& graph, CompileOptions options);

}  // namespace holoscan::detail
