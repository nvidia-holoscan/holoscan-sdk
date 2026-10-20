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

/// @brief Compile with drop-oldest fanout families kept on HoloMQ.
///
/// Ordinary compile() delivers the process-local members of a drop-oldest fanout family by direct
/// retention, including a family with cross-process or exported members. This source-private entry
/// point keeps such families on HoloMQ, as before direct fanout existed, while fanout-one edges
/// keep direct retention, for exact A/B checks.
/// @param graph Editable Graph whose composed generation is snapshotted.
/// @param options Owned public compile inputs.
/// @return Accepted or rejected immutable plan with structural diagnostics.
[[nodiscard]] HOLOSCAN_CORE_EXPORT ExecutionPlan
compile_with_process_local_holomq_fanout(Graph& graph, CompileOptions options);

/// @brief Compile while conservatively retaining process-local HoloMQ event attachment.
/// @param graph Editable Graph whose composed generation is snapshotted.
/// @param options Owned public compile inputs.
/// @return Accepted or rejected immutable plan with structural diagnostics.
/// @throws RuntimeError under the same process-resource conditions as
/// `holoscan::compile()`.
/// @note This source-private entry point exists only for exact conformance and
/// performance comparisons. It is not installed as public SDK API.
[[nodiscard]] HOLOSCAN_CORE_EXPORT ExecutionPlan
compile_with_process_local_holomq_event_wake(Graph& graph, CompileOptions options);

/// @brief Compile with every process-local HoloMQ service served by the Channel pump.
///
/// Ordinary compile() delivers each eligible service inside the producer's emit(). This
/// source-private entry point keeps the earlier pump delivery for exact A/B comparisons.
/// @param force_holomq_process_local_inline Also route direct-retention edges through HoloMQ, as
///        compile_with_process_local_holomq() does.
[[nodiscard]] HOLOSCAN_CORE_EXPORT ExecutionPlan compile_with_process_local_pump_delivery(
    Graph& graph, CompileOptions options, bool force_holomq_process_local_inline = false);

/// Compile the restricted, source-private split-partition fanout experiment.
/// Unsupported graphs are rejected; ordinary compile() never selects this mode.
/// force_event_wake exercises incompatible-control admission in source-tree tests.
[[nodiscard]] HOLOSCAN_CORE_EXPORT ExecutionPlan compile_with_process_local_fanout_experiment(
    Graph& graph, CompileOptions options, ProcessLocalFanoutExperiment experiment,
    bool force_event_wake = false);

}  // namespace holoscan::detail
