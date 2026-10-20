/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Declares the checks that create and validate a capture plan for each CUDA Graph region.

#pragma once

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/graph_compiler_access.hpp"
#include "core/graph_snapshot.hpp"
#include "execution_plan_state.hpp"
#include "plan_lowering_diagnostic.hpp"

namespace holoscan::detail {

/// Stable values written to graph-scoped CUDA Graph analysis diagnostics.
namespace cuda_graph_analysis_rule {
inline constexpr std::string_view kResolvedGraphEdges{"resolved_graph_edges"};
}  // namespace cuda_graph_analysis_rule

/// Stable values written to Diagnostic::required when CUDA Graph region validation fails.
namespace cuda_graph_region_rule {
inline constexpr std::string_view kNonemptyResolvedOperatorMembership{
    "nonempty_resolved_operator_membership"};
inline constexpr std::string_view kRootGraphRegion{"root_graph_region"};
inline constexpr std::string_view kDirectOperatorMembersInRegionGraph{
    "direct_operator_members_in_region_graph"};
inline constexpr std::string_view kDisjointCUDAGraphRegions{"disjoint_cuda_graph_regions"};
inline constexpr std::string_view kExclusiveCUDAGraphConstraint{"exclusive_cuda_graph_constraint"};
inline constexpr std::string_view kCaptureEligibleOperator{"capture_eligible_operator"};
inline constexpr std::string_view kTensorInternalEdges{"tensor_internal_edges"};
inline constexpr std::string_view kSingleEdgePerOperatorHop{"single_edge_per_operator_hop"};
inline constexpr std::string_view kSingleLinearOperatorChain{"single_linear_operator_chain"};
inline constexpr std::string_view kAtMostOneRegionIngressAndEgress{
    "at_most_one_region_ingress_and_egress"};
inline constexpr std::string_view kRegionEdgesOnlyAtSourceAndSink{
    "region_edges_only_at_source_and_sink"};
inline constexpr std::string_view kTensorRegionBoundaryEdges{"tensor_region_boundary_edges"};
inline constexpr std::string_view kNormalizedMemberContract{"normalized_member_contract"};
inline constexpr std::string_view kNoMemberMaxStartDelay{"no_member_max_start_delay"};
inline constexpr std::string_view kSupportedMemberContract{"supported_member_contract"};
}  // namespace cuda_graph_region_rule

/// Hold validated CUDA Graph capture plans and stable admission errors.
struct CUDAGraphRegionLoweringResult {
  std::vector<CUDAGraphCapturePlanRecord> capture_plans;
  std::vector<PlanLoweringDiagnostic> errors;
};

/// Hold one region whose membership, topology, and edges passed pre-partition validation.
struct CUDAGraphRegionAnalysisRecord {
  GraphId graph{};
  RegionId region{};
  std::string region_name;
  std::vector<VertexId> order;
  std::vector<CUDAGraphCaptureEdgeRecord> internal_edges;
  std::optional<CUDAGraphCaptureEdgeRecord> ingress;
  std::optional<CUDAGraphCaptureEdgeRecord> egress;
};

/// Hold accepted region structures and stable errors produced before partitioning.
struct CUDAGraphRegionAnalysisResult {
  std::vector<CUDAGraphRegionAnalysisRecord> regions;
  std::vector<PlanLoweringDiagnostic> errors;
};

/// Validate CUDA Graph region membership, topology, edges, and capture eligibility.
[[nodiscard]] CUDAGraphRegionAnalysisResult analyze_cuda_graph_regions(
    const GraphSnapshotData& snapshot);

/// Validate normalized contracts and create capture plans for structurally accepted regions.
[[nodiscard]] CUDAGraphRegionLoweringResult lower_cuda_graph_regions(
    std::span<const CUDAGraphRegionAnalysisRecord> regions, PlanGenerationId plan_generation,
    std::span<const GraphCompilerAccess::TemporalReplay> normalized_contracts);

/// Validate all references in published CUDA Graph capture plans.
[[nodiscard]] bool valid_cuda_graph_capture_plans(const ExecutionPlanState& state);

}  // namespace holoscan::detail
