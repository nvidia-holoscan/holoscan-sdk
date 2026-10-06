/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Connects CUDA Graph region members to fixed addresses supplied by one-slot Tensor RuntimePools
// while their compute() methods are captured. Steady-state input and output use normal RuntimePool
// loans, publication, and retirement.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <span>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "runtime_tensor_resources.hpp"

namespace holoscan::detail {

struct ExecutionPlanState;
struct RuntimeOperatorInstance;

/// Identifies the region being constructed or captured when a runtime operation fails.
struct CUDAGraphRuntimeRegionIdentity {
  GraphId graph{};
  RegionId region{};
  VertexId fused_operator{};
};

/// Reports whether every declared member output was produced after capture bindings were restored.
struct CUDAGraphCaptureEndState {
  bool outputs_complete{};
};

/// Gives capture-time Tensor bindings access to the one-slot RuntimePools owned by RunSession.
struct RuntimeCUDAGraphTensorPoolBinding {
  void* state{};  ///< RuntimeTensorResources passed to each callback.
  expected<RuntimeFixedTensorAddress, Error> (*fixed_output_address)(void*, PortId) noexcept {};
  expected<bool, Error> (*fixed_output_available)(void*, PortId) noexcept {};
  bool (*restore_output_binding)(void*, PortId) noexcept {};  ///< Restore the RuntimePool binding.
};

/// Temporarily replaces Channel bindings during capture and enforces each fixed ingress address
/// during steady-state execution.
struct RuntimeCUDAGraphChannelBindings {
  void* state{};
  bool (*restore_input)(void*, PortId) noexcept {};
  bool (*restore_output)(void*, PortId) noexcept {};
  bool (*require_fixed_tensor)(void*, PortId, const void*,
                               std::span<const std::uint64_t>) noexcept {};
  bool (*clear_fixed_tensor)(void*, PortId, const void*) noexcept {};
};

[[nodiscard]] std::optional<std::size_t> cuda_graph_contiguous_byte_span(
    const TensorLoanRequest& request) noexcept;

class RuntimeCUDAGraphTensorResources final {
 public:
  RuntimeCUDAGraphTensorResources(const RuntimeCUDAGraphTensorResources&) = delete;
  RuntimeCUDAGraphTensorResources& operator=(const RuntimeCUDAGraphTensorResources&) = delete;
  RuntimeCUDAGraphTensorResources(RuntimeCUDAGraphTensorResources&&) = delete;
  RuntimeCUDAGraphTensorResources& operator=(RuntimeCUDAGraphTensorResources&&) = delete;
  ~RuntimeCUDAGraphTensorResources();

  /// Resolve the one-slot RuntimePool address for every planned region edge.
  [[nodiscard]] static expected<std::unique_ptr<RuntimeCUDAGraphTensorResources>, Error> create(
      const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
      RuntimeCUDAGraphTensorPoolBinding tensor_pools, ProcessGroupId process_group,
      RuntimeCUDAGraphChannelBindings channel_bindings = {},
      CUDAGraphRuntimeRegionIdentity* failure_region = nullptr);

  /// Bind region inputs to fixed buffers before stream capture starts.
  [[nodiscard]] expected<void, Error> begin_capture(GraphId graph, RegionId region,
                                                    cudaStream_t stream) noexcept;

  /// Verify that every input needed by this member was produced before compute() is invoked.
  [[nodiscard]] expected<void, Error> prepare_capture_member(VertexId member) noexcept;

  /// Restore ordinary input and output bindings after capture.
  [[nodiscard]] expected<CUDAGraphCaptureEndState, Error> end_capture(GraphId graph,
                                                                      RegionId region) noexcept;

  /// Acquire the ordinary one-slot RuntimePool output and begin its CUDA write phase.
  [[nodiscard]] expected<void, Error> begin_launch(std::size_t region_index,
                                                   cudaStream_t stream) noexcept;

  /// Return whether a region with an egress can acquire its fixed output slot.
  /// Regions without an egress are always ready.
  [[nodiscard]] expected<bool, Error> output_available(std::size_t region_index) noexcept;

  /// Commit and publish the RuntimePool output after cudaGraphLaunch has been enqueued.
  [[nodiscard]] expected<void, Error> finish_launch(std::size_t region_index) noexcept;

  /// Abandon an acquired output when cudaGraphLaunch fails.
  void cancel_launch(std::size_t region_index) noexcept;

  /// Remove callbacks and fixed-address requirements before their RunSession owners are destroyed.
  void detach_session_state() noexcept;

  /// Verify the cold-built positional binding shared with CUDA Graph region resources.
  [[nodiscard]] bool matches(std::size_t region_index, VertexId fused_operator) const noexcept;

  [[nodiscard]] bool has_egress(std::size_t region_index) const noexcept;

  [[nodiscard]] std::size_t size() const noexcept;

 private:
  class Impl;
  explicit RuntimeCUDAGraphTensorResources(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::detail
