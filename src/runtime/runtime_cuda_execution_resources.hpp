/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan_ids.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/memory_kind.hpp>
#include <holoscan/core/operator_ref.hpp>

#include <holomq/cuda/stream_lifetime.hpp>

namespace holoscan::detail {

struct ExecutionPlanState;
struct RuntimeOperatorInstance;
struct RuntimeCudaExecutionDiagnostic;
class RuntimeTensorOutputTestControl;

/// Source-tree failure boundary for transactional CUDA execution materialization tests.
enum class RuntimeCudaExecutionFailurePoint : std::uint8_t {
  kClaim = 0,
  kProbe = 1,
  kSplit = 2,
  kContext = 3,
  kStream = 4,
  kIdentity = 5,
  kBinding = 6,
  kPublish = 7,
  kComputeDeviceRestore = 8,
  kCUDAGraphCaptureEnd = 9,
  kCUDAGraphExecDestroy = 10,
  kCUDAGraphDestroy = 11,
  kCUDAGraphCaptureStreamDestroy = 12,
  kCUDAGraphQuiesce = 13,
  kCUDAGraphLaunch = 14,
};

/// Source-tree observation points around CUDA Graph failure retention and reporting.
enum class RuntimeCudaExecutionObserverPoint : std::uint8_t {
  kFailureDiagnostic = 0,
  kQuiesceFailureRetention = 1,
  kQuiesceFailureRelease = 2,
  kQuiesceFailureReturn = 3,
};

/// Inject one source-tree failure without changing immutable plan identity.
class RuntimeCudaExecutionTestControl final {
 public:
  using FailureObserver = void (*)(void*) noexcept;

  RuntimeCudaExecutionTestControl(RuntimeCudaExecutionFailurePoint point,
                                  std::size_t occurrence = 0U) noexcept
      : point_(point), occurrence_(occurrence) {}

  void set_failure_observer(RuntimeCudaExecutionObserverPoint point, void* state,
                            FailureObserver observer) noexcept {
    observer_point_ = point;
    observer_state_ = state;
    observer_ = observer;
  }

  void notify_failure_observer(RuntimeCudaExecutionObserverPoint point) const noexcept {
    if (observer_ != nullptr && observer_point_ == point) {
      observer_(observer_state_);
    }
  }

  [[nodiscard]] bool should_fail(RuntimeCudaExecutionFailurePoint point) noexcept {
    if (point != point_) {
      return false;
    }
    return visits_.fetch_add(1U, std::memory_order_relaxed) == occurrence_;
  }

  [[nodiscard]] std::size_t visits() const noexcept {
    return visits_.load(std::memory_order_relaxed);
  }

 private:
  RuntimeCudaExecutionFailurePoint point_;
  std::size_t occurrence_{};
  std::atomic_size_t visits_{};
  RuntimeCudaExecutionObserverPoint observer_point_{
      RuntimeCudaExecutionObserverPoint::kFailureDiagnostic};
  void* observer_state_{};
  FailureObserver observer_{};
};

/// Own all process-local CUDA compute streams independently of Tensor capacity resources.
class HOLOSCAN_CORE_EXPORT RuntimeCudaExecutionResources final {
 public:
  class Impl;

  RuntimeCudaExecutionResources(const RuntimeCudaExecutionResources&) = delete;
  RuntimeCudaExecutionResources& operator=(const RuntimeCudaExecutionResources&) = delete;
  RuntimeCudaExecutionResources(RuntimeCudaExecutionResources&&) = delete;
  RuntimeCudaExecutionResources& operator=(RuntimeCudaExecutionResources&&) = delete;
  ~RuntimeCudaExecutionResources();

  /// Validate and materialize the current process's immutable CUDA execution plan.
  [[nodiscard]] static expected<std::unique_ptr<RuntimeCudaExecutionResources>, Error> create(
      const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
      RunGenerationId run_generation, ProcessGroupId process_group,
      std::shared_ptr<RuntimeTensorOutputTestControl> test_control = {},
      std::shared_ptr<RuntimeCudaExecutionTestControl> cuda_test_control = {},
      RuntimeCudaExecutionDiagnostic* diagnostic = nullptr);

  /// Synchronize every runtime-owned CUDA stream while dependent owners remain alive.
  [[nodiscard]] expected<void, Error> quiesce(RuntimeCudaExecutionDiagnostic* diagnostic = nullptr);

  /// Release native owners after successful dependency-safe stream quiescence.
  [[nodiscard]] expected<void, Error> shutdown(
      RuntimeCudaExecutionDiagnostic* diagnostic = nullptr);

  /// Quarantine this RunSession's device claims after a failure in a dependent CUDA owner.
  /// The complete owner closure must remain alive after this call.
  void quarantine_after_external_failure() noexcept;

  /// Resolve the exact partition stream, or the ordinary process/device stream when unpartitioned.
  [[nodiscard]] cudaStream_t partition_stream(ProcessGroupId process_group, DeviceId device,
                                              ExecutionPartitionId partition) const noexcept;

  /// Resolve a compiler service stream by its complete stable owner identity.
  [[nodiscard]] cudaStream_t service_stream(ProcessGroupId process_group, DeviceId device,
                                            std::string_view owner_id) const noexcept;

  /// Resolve a compiler compute stream by its complete retained identity.
  [[nodiscard]] cudaStream_t compute_stream(ProcessGroupId process_group, DeviceId device,
                                            std::string_view stream_id) const noexcept;

  /// Resolve the ordinary process/device stream on a device without Green Context placement.
  [[nodiscard]] cudaStream_t ordinary_stream(ProcessGroupId process_group,
                                             DeviceId device) const noexcept;

  /// Cold-path lookup of the control block for an SDK-owned native stream.
  [[nodiscard]] std::shared_ptr<const holomq::cuda::StreamLifetime> stream_lifetime(
      cudaStream_t stream) const noexcept;

 private:
  explicit RuntimeCudaExecutionResources(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::detail
