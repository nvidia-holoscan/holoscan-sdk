/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/memory/cuda_stream_allocator.hpp>
#include <holoscan/memory/memory_storage_type.hpp>

namespace holoscan {

/**
 * @brief Fixed-size block memory pool independent of GXF entities and parameters.
 */
class HOLOSCAN_CORE_EXPORT BlockMemoryPool final : public CudaStreamAllocator {
 public:
  struct Config {
    MemoryStorageType storage_type = MemoryStorageType::kHost;
    uint64_t block_size = 0;
    uint64_t num_blocks = 0;
    int32_t device_id = 0;
    // Enables allocation-path counters such as high-water and failed acquisitions. Leave false
    // for deterministic hot paths; stats() still snapshots current usage when called.
    bool enable_stats = false;
  };

  struct Stats {
    uint64_t capacity = 0;
    uint64_t available = 0;
    uint64_t used = 0;
    uint64_t deferred = 0;
    uint64_t high_water_used = 0;
    uint64_t failed_acquisitions = 0;
    bool initialized = false;
    bool counters_enabled = false;
  };

  BlockMemoryPool() = default;
  HOLOSCAN_CORE_EXPORT explicit BlockMemoryPool(Config config);
  HOLOSCAN_CORE_EXPORT BlockMemoryPool(MemoryStorageType storage_type, uint64_t block_size,
                                       uint64_t num_blocks, int32_t device_id = 0);
  HOLOSCAN_CORE_EXPORT ~BlockMemoryPool() override;

  BlockMemoryPool(const BlockMemoryPool&) = delete;
  BlockMemoryPool& operator=(const BlockMemoryPool&) = delete;
  BlockMemoryPool(BlockMemoryPool&&) = delete;
  BlockMemoryPool& operator=(BlockMemoryPool&&) = delete;

  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> initialize_pool();
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> deinitialize_pool();

  [[nodiscard]] expected<void, Error> is_available(uint64_t size,
                                                   MemoryStorageType storage_type) override;
  [[nodiscard]] expected<std::byte*, Error> allocate(uint64_t size,
                                                     MemoryStorageType storage_type) override;
  [[nodiscard]] expected<void, Error> free(std::byte* pointer) override;
  [[nodiscard]] uint64_t block_size(MemoryStorageType storage_type) const override;

  [[nodiscard]] MemoryStorageType storage_type() const noexcept { return config_.storage_type; }
  [[nodiscard]] uint64_t num_blocks() const noexcept { return config_.num_blocks; }
  [[nodiscard]] const Config& config() const noexcept { return config_; }
  [[nodiscard]] HOLOSCAN_CORE_EXPORT Stats stats();

  // Reports the device ordinal that should appear in DLPack metadata. Only device-resident
  // (kDLCUDA) allocations carry a meaningful ordinal; CUDA pinned host memory is kDLCUDAHost with
  // device_id 0 by convention, managed memory is kDLCUDAManaged with device_id 0 by convention,
  // and system memory is kDLCPU. The pool binds CUDA allocations to config_.device_id
  // (cudaSetDevice runs during initialization).
  [[nodiscard]] int32_t device_id(MemoryStorageType storage_type) const override {
    switch (storage_type) {
      case MemoryStorageType::kDevice:
        return config_.device_id;
      default:
        return kUnknownDeviceId;
    }
  }

 protected:
  [[nodiscard]] expected<std::byte*, Error> allocate_async_impl(uint64_t size,
                                                                MemoryStorageType storage_type,
                                                                cudaStream_t stream) override;
  [[nodiscard]] expected<void, Error> free_async_impl(std::byte* pointer,
                                                      cudaStream_t stream) override;

 private:
  enum struct AllocatorStage : uint8_t {
    kUninitialized = 0,
    kInitializationInProgress = 1,
    kInitialized = 2,
    kDeinitializationInProgress = 3,
  };

  class FixedPoolUint64;

  struct DeferredBlock {
    uint64_t index = 0;

    DeferredBlock() = default;
    explicit DeferredBlock(uint64_t block_index) : index(block_index) {}
  };

  [[nodiscard]] HOLOSCAN_CORE_EXPORT int64_t pointer_to_index(std::byte* pointer) const;
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> initialize_block_events();
  HOLOSCAN_CORE_EXPORT void destroy_block_events();
  HOLOSCAN_CORE_EXPORT void reclaim_completed_deferred_blocks();
  [[nodiscard]] HOLOSCAN_CORE_EXPORT uint64_t checked_out_blocks_locked() const;
  HOLOSCAN_CORE_EXPORT void update_high_water_locked();
  HOLOSCAN_CORE_EXPORT void record_failed_acquisition();
  static const char* allocator_stage_str(AllocatorStage stage);

  Config config_;
  std::byte* pointer_ = nullptr;
  std::unique_ptr<FixedPoolUint64> stack_;
  mutable std::mutex stack_mutex_;
  std::atomic<AllocatorStage> stage_{AllocatorStage::kUninitialized};
  std::vector<DeferredBlock> deferred_blocks_;
  std::vector<cudaEvent_t> block_events_;
  std::vector<uint8_t> block_deferred_;
  std::atomic<uint64_t> high_water_used_{0};
  std::atomic<uint64_t> failed_acquisitions_{0};
};

}  // namespace holoscan
