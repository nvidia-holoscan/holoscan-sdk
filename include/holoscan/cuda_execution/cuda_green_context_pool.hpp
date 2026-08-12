/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

// CUDA green-context driver APIs (CUgreenCtx, cuGreenCtxCreate, cuDevSmResourceSplitByCount, ...)
// were introduced in CUDA 12.4. Guard so the experimental library still builds against older CUDA
// toolkits, where these native types are simply unavailable.
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12040)
#define HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE 1
#else
#define HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE 0
#endif

namespace holoscan {

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE

// Lazily-loaded CUDA driver entry points used to create and query green contexts. Defined in the
// translation unit so the public header does not expose the driver function-pointer surface.
struct CudaDriverFunctionTable;

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
  struct Config {
    int32_t device_id = 0;
    // Flags forwarded to cuDevSmResourceSplitByCount (matches the GXF default).
    uint32_t green_context_flags = cudaStreamNonBlocking;
    // Number of explicitly requested partitions. 0 means "one partition spanning the whole device".
    uint32_t num_partitions = 0;
    // Minimum SMs per partition; requested partition sizes must be a multiple of this.
    uint32_t min_sm_count = 2;
    // SM count for each requested partition (size must be >= num_partitions when num_partitions >
    // 0).
    std::vector<int32_t> sms_per_partition = {0};
    // Partition index used as the default context. If unset, the last partition is used.
    std::optional<int32_t> default_context;
    std::string nvtx_identifier = "cuda_green_context_pool";
  };

  // Constructor and destructor are defined out-of-line so the pimpl'd CudaDriverFunctionTable can
  // stay incomplete in this header.
  HOLOSCAN_CORE_EXPORT CudaGreenContextPool();
  HOLOSCAN_CORE_EXPORT explicit CudaGreenContextPool(Config config);
  CudaGreenContextPool(const CudaGreenContextPool&) = delete;
  CudaGreenContextPool& operator=(const CudaGreenContextPool&) = delete;
  HOLOSCAN_CORE_EXPORT ~CudaGreenContextPool();

  // Initialize and throw RuntimeError on failure (parallels CudaStreamPool / CudaEventPool).
  HOLOSCAN_CORE_EXPORT void initialize();
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> initialize_pool();
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> deinitialize_pool();

  // Green context handle for a given partition index.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<CUgreenCtx, Error> green_context(
      uint32_t index) const;
  // Primary CUcontext associated with a partition's green context.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<CUcontext, Error> cuda_context(uint32_t index) const;
  // SM count of a given partition.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<uint32_t, Error> partition_sms(uint32_t index) const;
  // Index of the default partition.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<uint32_t, Error> default_context_index() const;
  // Green context handle of the default partition.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<CUgreenCtx, Error> default_context() const;
  // Total SM count of the device backing this pool.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<uint32_t, Error> device_total_sms() const;

  [[nodiscard]] HOLOSCAN_CORE_EXPORT size_t num_partitions() const;
  [[nodiscard]] const Config& config() const noexcept { return config_; }

 private:
  enum struct AllocatorStage : uint8_t {
    kUninitialized = 0,
    kInitializationInProgress = 1,
    kInitialized = 2,
    kDeinitializationInProgress = 3,
  };

  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> load_cuda_driver();
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> reserve_green_contexts(
      int32_t compute_capability_major);
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> destroy_green_contexts();

  Config config_;
  mutable std::mutex mutex_;
  AllocatorStage stage_ = AllocatorStage::kUninitialized;

  std::vector<CUgreenCtx> green_contexts_;
  std::vector<CUcontext> cuda_contexts_;
  std::vector<int32_t> resolved_sms_per_partition_;
  uint32_t resolved_num_partitions_ = 0;
  uint32_t max_sm_count_ = 0;
  int32_t dev_id_ = -1;
  uint32_t default_index_ = 0;

  std::unique_ptr<CudaDriverFunctionTable> driver_;
};

#endif  // HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE

}  // namespace holoscan
