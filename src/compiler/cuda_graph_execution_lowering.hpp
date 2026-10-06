/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Converts each topology-validated CUDA Graph region into one fused execution-plan operator after
// verifying its partition, process, CUDA device, stream, and fixed Tensor storage. It also owns the
// compiler helpers that remove and restore the region's ordinary Runtime channels.

#pragma once

#include <optional>
#include <string_view>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "execution_plan_state.hpp"
#include "plan_lowering_diagnostic.hpp"

namespace holoscan::detail {

/// Stable values written to Diagnostic::required when fused-region admission fails.
namespace cuda_graph_execution_rule {
inline constexpr std::string_view kNonemptyCapturePlan{"nonempty_capture_plan"};
inline constexpr std::string_view kSingleExecutionPartition{"single_execution_partition"};
inline constexpr std::string_view kSingleProcessGroup{"single_process_group"};
inline constexpr std::string_view kSingleCUDADevice{"single_cuda_device"};
inline constexpr std::string_view kSourceCUDAStream{"source_cuda_stream"};
inline constexpr std::string_view kFixedTensorBuffer{"fixed_tensor_buffer"};
inline constexpr std::string_view kCompleteTensorShapeBounds{"complete_tensor_shape_bounds"};
inline constexpr std::string_view kImmutableCaptureState{"immutable_capture_state"};
inline constexpr std::string_view kExactTensorShape{"exact_tensor_shape"};
inline constexpr std::string_view kRuntimePoolTensorOutput{"runtime_pool_tensor_output"};
inline constexpr std::string_view kSingleTensorEmit{"single_tensor_emit"};
inline constexpr std::string_view kSingleFixedTensorSlot{"single_fixed_tensor_slot"};
inline constexpr std::string_view kExclusiveIngressOutput{"exclusive_ingress_output"};
inline constexpr std::string_view kInternalCUDADeviceBuffer{"internal_cuda_device_buffer"};
inline constexpr std::string_view kCaptureCompatibleBoundaryBuffer{
    "capture_compatible_boundary_buffer"};
}  // namespace cuda_graph_execution_rule

/// Hold physically admitted fused-region plans and stable admission errors.
struct CUDAGraphExecutionLoweringResult {
  std::vector<CUDAGraphExecutionRegionPlanRecord> regions;
  std::vector<PlanLoweringDiagnostic> errors;
};

/// Preserve the ordinary process-local channels removed by fused-region lowering.
struct CUDAGraphRuntimeChannelPlanBackup {
  std::vector<ProcessLocalChannelServicePlanRecord> services;
  std::vector<ProcessLocalChannelPublisherPlanRecord> publishers;
  std::vector<ProcessLocalChannelSubscriberPlanRecord> subscribers;
};

/// Replace each validated CUDA Graph region with one planned fused operator.
///
/// This pass runs after partition, Tensor allocation, CUDA stream, and Green Context lowering. It
/// verifies those final facts and records the exact stream and stable buffers Runtime must use.
/// Device-backed Tensor buffers and partition-owned operator streams must resolve to one CUDA
/// device; conflicting devices reject the region. For an edge-less operator, the partition-owned
/// stream establishes the device. Process-wide and service streams do not establish its placement.
/// `graph_generation` supplies diagnostic provenance and does not affect admission.
[[nodiscard]] CUDAGraphExecutionLoweringResult finalize_cuda_graph_execution_regions(
    GraphGeneration graph_generation, const ExecutionPlanState& state);

/// Validate all references and physical facts in published fused-region plans.
/// Every capture plan must map to one fused region or matching published rejection diagnostics.
[[nodiscard]] bool valid_cuda_graph_execution_regions(const ExecutionPlanState& state);

/// Remove ordinary Runtime channels for fused-region edges and return the original channel plan.
///
/// An empty result leaves the state unchanged. This occurs when no region was admitted or retained
/// channel records violate the validated region topology.
[[nodiscard]] std::optional<CUDAGraphRuntimeChannelPlanBackup>
remove_cuda_graph_internal_runtime_channels(ExecutionPlanState& state);

/// Discard CUDA Graph capture and execution plans and restore channels removed by region fusion.
void discard_cuda_graph_capture_and_execution_plan(
    ExecutionPlanState& state, std::optional<CUDAGraphRuntimeChannelPlanBackup>&& channel_backup);

}  // namespace holoscan::detail
