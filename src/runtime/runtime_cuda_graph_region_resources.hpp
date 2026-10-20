/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>

#include "runtime_cuda_graph_tensor_resources.hpp"

namespace holoscan::detail {

struct ExecutionPlanState;
struct RuntimeCudaExecutionDiagnostic;
struct RuntimeOperatorInstance;
class RuntimeCudaExecutionTestControl;
class PortBinding;
class RuntimeCUDAGraphRegionResources;

/// Runtime implementation of the fused operator produced for one CUDA Graph region. Its only
/// steady-state operation is launching the executable graph captured during RunSession startup.
class CUDAGraphRegionOperator final : public Operator<> {
 public:
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

 private:
  CUDAGraphRegionOperator(RuntimeCUDAGraphRegionResources* resources,
                          std::size_t region_index) noexcept;

  RuntimeCUDAGraphRegionResources* resources_{};
  std::size_t region_index_{};

  friend class RuntimeCUDAGraphRegionResources;
};

/// Own the native graph, executable, and member bindings for every CUDA Graph region in one
/// process-local RunSession. Capture uses a private stream in the source operator's CUDA context;
/// replay uses the source operator's compiler-selected stream.
class RuntimeCUDAGraphRegionResources final {
 public:
  using CaptureMemberInvoke = expected<void, Error> (*)(void*, RuntimeOperatorInstance&, DeviceId,
                                                        cudaStream_t) noexcept;

  RuntimeCUDAGraphRegionResources(const RuntimeCUDAGraphRegionResources&) = delete;
  RuntimeCUDAGraphRegionResources& operator=(const RuntimeCUDAGraphRegionResources&) = delete;
  RuntimeCUDAGraphRegionResources(RuntimeCUDAGraphRegionResources&&) = delete;
  RuntimeCUDAGraphRegionResources& operator=(RuntimeCUDAGraphRegionResources&&) = delete;
  ~RuntimeCUDAGraphRegionResources();

  /// Join the compiler's capture and execution records to reconstructed operators and designated
  /// replay streams. Construction resolves bindings; `capture_all()` creates private capture
  /// streams and performs native capture.
  [[nodiscard]] static expected<std::unique_ptr<RuntimeCUDAGraphRegionResources>, Error> create(
      const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
      RuntimeCUDAGraphTensorPoolBinding tensor_pools, ProcessGroupId process_group,
      RuntimeCUDAGraphChannelBindings channel_bindings = {},
      CUDAGraphRuntimeRegionIdentity* failure_region = nullptr,
      std::shared_ptr<RuntimeCudaExecutionTestControl> test_control = {});

  /// Construct a region that has no Tensor edges.
  [[nodiscard]] static expected<std::unique_ptr<RuntimeCUDAGraphRegionResources>, Error> create(
      const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
      ProcessGroupId process_group);

  /// Capture every local region once on a private stream by invoking its original operators in
  /// compiler-frozen order.
  [[nodiscard]] expected<void, Error> capture_all(
      void* state, CaptureMemberInvoke invoke,
      CUDAGraphRuntimeRegionIdentity* failure_region = nullptr);

  /// Gate each fused operator's existing readiness source on availability of its fixed egress
  /// slot. The original binding retains its witness while downstream still owns the prior output.
  [[nodiscard]] expected<void, Error> bind_output_readiness(
      std::vector<std::unique_ptr<PortBinding>>& bindings) noexcept;

  /// Verify that every local region with an egress had its output-availability check attached by
  /// the partition that owns its fused operator.
  [[nodiscard]] expected<void, Error> validate_output_readiness_bindings() const noexcept;

  /// Enqueue the captured region on its designated stream and publish its asynchronously produced
  /// egress Tensor using the normal Tensor readiness mechanism.
  [[nodiscard]] expected<void, Error> launch(VertexId fused_operator) noexcept;

  /// Synchronize submitted launches before dependent operators or Tensor storage are released.
  /// Retain the failing device, stream, and CUDA status in `diagnostic` when provided.
  [[nodiscard]] expected<void, Error> quiesce(
      RuntimeCudaExecutionDiagnostic* diagnostic = nullptr) noexcept;

  /// Destroy captured executables, graph objects, and private capture streams after quiescence.
  /// The destructor retains best-effort cleanup only for partial construction and fallback paths.
  [[nodiscard]] expected<void, Error> shutdown(
      RuntimeCudaExecutionDiagnostic* diagnostic = nullptr) noexcept;

  /// Remove callbacks and Channel requirements before their RunSession owners are destroyed.
  void detach_session_state() noexcept;

  [[nodiscard]] std::size_t size() const noexcept;

 private:
  class Impl;
  explicit RuntimeCUDAGraphRegionResources(std::unique_ptr<Impl> impl) noexcept;

  [[nodiscard]] static expected<std::unique_ptr<RuntimeCUDAGraphRegionResources>, Error>
  create_impl(const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
              RuntimeCUDAGraphTensorPoolBinding tensor_pools, ProcessGroupId process_group,
              RuntimeCUDAGraphChannelBindings channel_bindings,
              CUDAGraphRuntimeRegionIdentity* failure_region,
              std::shared_ptr<RuntimeCudaExecutionTestControl> test_control);

  [[nodiscard]] expected<void, Error> launch(std::size_t region_index,
                                             cudaStream_t stream) noexcept;

  std::unique_ptr<Impl> impl_;

  friend class CUDAGraphRegionOperator;
};

}  // namespace holoscan::detail
