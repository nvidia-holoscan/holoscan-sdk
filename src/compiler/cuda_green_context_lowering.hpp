/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Declares the admission of a deployment's CUDA Green Context bindings into plan records.

#pragma once

#include <optional>
#include <span>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/profile.hpp>
#include "../core/graph_snapshot.hpp"
#include "cuda_service_stream_plan.hpp"
#include "execution_plan_state.hpp"
#include "plan_lowering_diagnostic.hpp"
#include "runtime_package_capabilities.hpp"

namespace holoscan::detail {

/// Hold the CUDA Green Context and stream plan records, placement probes, and diagnostics that
/// lower_cuda_green_context_plan() produces.
struct CudaGreenContextLoweringResult {
  std::vector<CudaGreenContextRequestPlanRecord> requests;
  std::vector<CudaGreenContextAssignmentPlanRecord> assignments;
  std::vector<CudaGreenContextPoolPlanRecord> pools;
  std::vector<CudaGreenContextResourcePlanRecord> resources;
  std::vector<CudaComputeStreamPlanRecord> compute_streams;
  std::vector<OperatorCudaStreamBindingPlanRecord> operator_stream_bindings;
  std::vector<PlacementProbePlanRecord> placement_probes;
  std::vector<PlanLoweringDiagnostic> errors;
  std::vector<PlanLoweringDiagnostic> warnings;
};

/// Admit the deployment's CUDA Green Context bindings against the device capability evidence and
/// lower them into pool, context, and stream plan records.
[[nodiscard]] CudaGreenContextLoweringResult lower_cuda_green_context_plan(
    const GraphSnapshotData& snapshot, const DeploymentBinding& deployment,
    const RuntimeCompileEvidenceView& evidence,
    std::span<const std::optional<Profile>> effective_profiles,
    std::span<const CudaServiceStreamRequirement> service_stream_requirements,
    const ExecutionPlanState& state);

}  // namespace holoscan::detail
