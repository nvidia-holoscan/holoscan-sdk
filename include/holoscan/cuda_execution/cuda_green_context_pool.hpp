/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

// CUDA Green Context resources arrived in CUDA 12.4, but the stream-creation API required by the
// usable execution contract arrived in CUDA 12.5. Guard the complete feature accordingly.
#ifndef HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12050)
#define HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE 1
#else
#define HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE 0
#endif
#endif

namespace holoscan {

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE

namespace cuda_execution_detail {
struct CudaGreenContextDriver;
}  // namespace cuda_execution_detail

/**
 * @brief Native experimental counterpart of nvidia::gxf::CudaGreenContextPool.
 *
 * Partitions a single CUDA device's SMs into one or more CUDA green contexts and owns the resulting
 * CUgreenCtx / CUcontext handles. Unlike the GXF resource, it has no dependency on GXF or
 * holoscan::core: configuration is supplied through a plain Config, fallible operations return
 * expected<T, Error>, and there is no Component lifecycle.
 *
 * There is no CCCL RAII type for green contexts (as there is cuda::stream for CudaStreamPool), so
 * the pool owns the raw CUgreenCtx handles directly and destroys them with cuGreenCtxDestroy in
 * deinitialize_pool(). Accessors hand out non-owning CUgreenCtx / CUcontext views, mirroring the
 * way CudaGreenContext borrows from the pool.
 */
class CudaGreenContextPool {
 public:
  /// @brief Configure one experimental low-level Green Context pool.
  struct Config {
    std::int32_t device_id{};  ///< Explicit nonnegative process-local CUDA device ordinal.
    /// Number of explicitly requested groups; zero requests one whole-device group.
    std::uint32_t num_partitions{};
    /// Exact SM count per requested group; each count must satisfy driver-reported granularity.
    std::vector<int32_t> sms_per_partition = {0};
    /// Default group index, or the final group when absent.
    std::optional<int32_t> default_context;
    /// Stable observability identifier owned by this value.
    std::string nvtx_identifier = "cuda_green_context_pool";
  };

  /// @brief Construct a pool with one whole-device group on device zero.
  HOLOSCAN_CORE_EXPORT CudaGreenContextPool();
  /// @brief Construct an uninitialized pool from an owned configuration.
  /// @param config Exact device and structured group request.
  HOLOSCAN_CORE_EXPORT explicit CudaGreenContextPool(Config config);
  CudaGreenContextPool(const CudaGreenContextPool&) = delete;
  CudaGreenContextPool& operator=(const CudaGreenContextPool&) = delete;
  /// @brief Destroy every owned Green Context in reverse creation order.
  HOLOSCAN_CORE_EXPORT ~CudaGreenContextPool();

  /// @brief Initialize the pool or throw RuntimeError on failure.
  /// @throws RuntimeError if configuration, CUDA capability probing, splitting, or creation fails.
  HOLOSCAN_CORE_EXPORT void initialize();
  /// @brief Initialize every requested structured group exactly once.
  /// @return Success or a typed cold-path CUDA/configuration error.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> initialize_pool();
  /// @brief Destroy every owned Green Context and reset the pool.
  /// @return Success or the first native destruction error.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> deinitialize_pool();

  /// @brief Return the borrowed Green Context handle for one group index.
  /// @param index Zero-based materialized group index.
  /// @return Borrowed handle or an out-of-range/uninitialized error.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<CUgreenCtx, Error> green_context(
      uint32_t index) const;
  /// @brief Return the borrowed converted CUDA context for one group index.
  /// @param index Zero-based materialized group index.
  /// @return Borrowed converted context or an out-of-range/uninitialized error.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<CUcontext, Error> cuda_context(uint32_t index) const;
  /// @brief Return the accepted SM count for one structured group.
  /// @param index Zero-based materialized group index.
  /// @return Accepted SM count or an out-of-range/uninitialized error.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<uint32_t, Error> partition_sms(uint32_t index) const;
  /// @brief Return the configured or derived default group index.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<uint32_t, Error> default_context_index() const;
  /// @brief Return the borrowed default Green Context handle.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<CUgreenCtx, Error> default_context() const;
  /// @brief Return the driver-reported device SM count.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<uint32_t, Error> device_total_sms() const;
  /// @brief Return the driver-reported minimum structured partition size.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<uint32_t, Error> min_sm_partition_size() const;
  /// @brief Return the driver-reported SM co-scheduling alignment.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<uint32_t, Error> sm_coscheduled_alignment() const;
  /// @brief Return the driver-qualified maximum number of minimum-size structured groups.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<uint32_t, Error> max_structured_context_count() const;

  /// @brief Return the number of materialized Green Context groups.
  /// @return Materialized group count, or zero before initialization.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT size_t num_partitions() const;
  /// @brief Return the owned cold configuration.
  /// @return Immutable pool configuration retained by this object.
  [[nodiscard]] const Config& config() const noexcept { return config_; }

 private:
  enum struct AllocatorStage : uint8_t {
    kUninitialized = 0,
    kInitializationInProgress = 1,
    kInitialized = 2,
    kDeinitializationInProgress = 3,
  };

  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> load_cuda_driver();
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> reserve_green_contexts();
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> destroy_green_contexts();

  Config config_;
  mutable std::mutex mutex_;
  AllocatorStage stage_ = AllocatorStage::kUninitialized;

  std::vector<CUgreenCtx> green_contexts_;
  std::vector<CUcontext> cuda_contexts_;
  std::vector<int32_t> resolved_sms_per_partition_;
  uint32_t resolved_num_partitions_ = 0;
  uint32_t max_sm_count_ = 0;
  uint32_t min_sm_partition_size_ = 0;
  uint32_t sm_coscheduled_alignment_ = 0;
  uint32_t structured_sm_quantum_ = 0;
  uint32_t max_structured_context_count_ = 0;
  int32_t dev_id_ = -1;
  uint32_t default_index_ = 0;

  std::shared_ptr<const cuda_execution_detail::CudaGreenContextDriver> driver_;
};

#endif  // HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE

}  // namespace holoscan
