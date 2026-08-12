/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda/devices>
#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
// Provides the CUgreenCtx type and HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE so the pool
// can optionally create streams inside a CUDA green context.
#include <holoscan/cuda_execution/cuda_green_context_pool.hpp>

namespace holoscan {

/**
 * @brief Native CUDA stream pool using CCCL cuda::stream ownership.
 *
 * The pool does not create framework entities or stream-id components. Callers receive move-only
 * leases that return streams to the pool when destroyed. If a lease outlives the owning pool during
 * experimental teardown, release drops the stream instead of dereferencing the destroyed pool. A
 * production version may still move full shared pool state behind the lease if cross-owner reuse is
 * required.
 */
class CudaStreamPool {
 public:
  struct Config {
    int32_t device_id = 0;
    uint32_t stream_flags = cudaStreamNonBlocking;
    int32_t stream_priority = ::cuda::stream::default_priority;
    uint32_t reserved_size = 1;
    uint32_t max_size = 0;  // 0 means unlimited
    std::string nvtx_identifier = "cuda_stream_pool";
    // When non-null, streams are created in this application-provided CUDA context: the context is
    // made current (via the driver's cuCtxPushCurrent) for the duration of stream creation and the
    // previous context is restored afterward. The pool only borrows the context and never destroys
    // it; the caller owns its lifetime and must keep it valid for as long as the pool exists. Use
    // this for an externally-managed context, such as one shared with a graphics API for
    // simultaneous CUDA/graphics work. Requires CUDA 12.3+ at runtime and is mutually exclusive
    // with green_context. When null, streams are created on the device normally.
    CUcontext cuda_context = nullptr;
#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
    // When non-null, streams are created inside this CUDA green context via cuGreenCtxStreamCreate
    // (obtain it from CudaGreenContextPool::green_context()/default_context() or
    // CudaGreenContext::green_context()). When null, streams are created on the device normally.
    CUgreenCtx green_context = nullptr;
#endif
  };

  struct StreamEntry;

  class StreamLease {
   public:
    StreamLease() = default;
    StreamLease(const StreamLease&) = delete;
    StreamLease& operator=(const StreamLease&) = delete;
    HOLOSCAN_CORE_EXPORT StreamLease(StreamLease&& other) noexcept;
    HOLOSCAN_CORE_EXPORT StreamLease& operator=(StreamLease&& other) noexcept;
    HOLOSCAN_CORE_EXPORT ~StreamLease();

    [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> release();

    [[nodiscard]] HOLOSCAN_CORE_EXPORT ::cuda::stream_ref stream() const;
    [[nodiscard]] HOLOSCAN_CORE_EXPORT cudaStream_t native_handle() const;
    [[nodiscard]] explicit operator bool() const noexcept { return entry_ != nullptr; }

   private:
    friend class CudaStreamPool;
    StreamLease(CudaStreamPool* pool, std::shared_ptr<StreamEntry> entry,
                std::weak_ptr<bool> pool_alive);

    CudaStreamPool* pool_ = nullptr;
    std::shared_ptr<StreamEntry> entry_;
    std::weak_ptr<bool> pool_alive_;
  };

  struct StreamEntry {
    cuda::stream stream;
    std::string name;
  };

  CudaStreamPool() = default;
  HOLOSCAN_CORE_EXPORT explicit CudaStreamPool(Config config);
  HOLOSCAN_CORE_EXPORT CudaStreamPool(int32_t device_id, uint32_t stream_flags,
                                      int32_t stream_priority, uint32_t reserved_size,
                                      uint32_t max_size,
                                      std::string nvtx_identifier = "cuda_stream_pool");
  HOLOSCAN_CORE_EXPORT ~CudaStreamPool();

  HOLOSCAN_CORE_EXPORT void initialize();
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> initialize_pool();
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> deinitialize_pool();

  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<StreamLease, Error> acquire();
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> release(
      std::shared_ptr<StreamEntry> entry);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT size_t available() const;
  [[nodiscard]] HOLOSCAN_CORE_EXPORT size_t allocated() const;
  [[nodiscard]] size_t capacity() const noexcept { return config_.max_size; }
  [[nodiscard]] const Config& config() const noexcept { return config_; }

 private:
  enum struct AllocatorStage : uint8_t {
    kUninitialized = 0,
    kInitializationInProgress = 1,
    kInitialized = 2,
    kDeinitializationInProgress = 3,
  };

  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<std::shared_ptr<StreamEntry>, Error>
  create_stream_entry(std::string name);
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> reserve_streams();

  Config config_;
  mutable std::mutex mutex_;
  std::vector<std::shared_ptr<StreamEntry>> reserved_streams_;
  std::vector<StreamEntry*> allocated_streams_;
  std::shared_ptr<bool> alive_token_ = std::make_shared<bool>(true);
  AllocatorStage stage_ = AllocatorStage::kUninitialized;
  uint32_t next_stream_index_ = 0;
};

}  // namespace holoscan
