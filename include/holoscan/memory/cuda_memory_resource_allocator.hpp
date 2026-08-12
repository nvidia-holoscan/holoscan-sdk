/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <cuda_runtime.h>
#include <cuda/memory_resource>
#include <cuda/stream>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/memory/cuda_stream_allocator.hpp>

namespace holoscan {

/**
 * @brief Native experimental allocator API backed by CCCL CUDA memory pools.
 *
 * Device allocations come from CCCL/libcu++ memory pool resources. Pinned-host allocations use
 * CCCL's CUDA pinned-memory resource available with CUDA 12.x.
 */
class HOLOSCAN_CORE_EXPORT CudaMemoryResourceAllocator : public CudaStreamAllocator {
 public:
  struct Config {
    std::string device_memory_initial_size;
    std::string device_memory_max_size;
    std::string host_memory_initial_size;
    std::string host_memory_max_size;
    int32_t device_id = 0;
  };

  struct Stats {
    size_t device_pool_reserved_bytes = 0;
    size_t device_pool_used_bytes = 0;
    size_t device_pool_configured_max_bytes = 0;
    size_t device_pool_available_bytes = 0;
    size_t host_used_bytes = 0;
    size_t host_configured_max_bytes = 0;
    size_t host_available_bytes = 0;
    size_t allocation_count = 0;
    bool initialized = false;
  };

  HOLOSCAN_CORE_EXPORT CudaMemoryResourceAllocator();
  HOLOSCAN_CORE_EXPORT explicit CudaMemoryResourceAllocator(Config config);
  HOLOSCAN_CORE_EXPORT CudaMemoryResourceAllocator(std::string_view device_memory_initial_size,
                                                   std::string_view device_memory_max_size,
                                                   std::string_view host_memory_initial_size,
                                                   std::string_view host_memory_max_size,
                                                   int32_t device_id = 0);
  HOLOSCAN_CORE_EXPORT ~CudaMemoryResourceAllocator() override;

  HOLOSCAN_CORE_EXPORT void initialize();

  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> initialize_pool();
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> deinitialize_pool();

  [[nodiscard]] expected<void, Error> is_available(uint64_t size,
                                                   MemoryStorageType storage_type) override;
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> is_available(uint64_t size);
  [[nodiscard]] expected<std::byte*, Error> allocate(uint64_t size,
                                                     MemoryStorageType storage_type) override;
  [[nodiscard]] expected<void, Error> free(std::byte* pointer) override;
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<size_t, Error> pool_size(
      MemoryStorageType storage_type) const;
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<Stats, Error> stats() const;

  [[nodiscard]] const Config& config() const noexcept { return config_; }

  // Reports the device ordinal that should appear in DLPack metadata. Only device-resident
  // (kDLCUDA) allocations carry a meaningful ordinal; pinned host memory is kDLCUDAHost with
  // device_id 0 by convention, and managed memory is kDLCUDAManaged with device_id 0 by convention.
  // Pools are bound to config_.device_id.
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

  struct AllocationInfo {
    size_t size = 0;
    MemoryStorageType storage_type = MemoryStorageType::kSystem;
  };

  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<size_t, Error> parse_storage_size(
      std::string_view text) const;
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<size_t, Error> device_available_bytes() const;
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<size_t, Error> host_available_bytes() const;
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> ensure_available(
      uint64_t size, MemoryStorageType storage_type) const;
  static const char* allocator_stage_str(AllocatorStage stage);

  Config config_;
  AllocatorStage stage_ = AllocatorStage::kUninitialized;
  cudaStream_t stream_ = nullptr;
  size_t device_max_memory_pool_size_ = 0;
  size_t host_max_memory_pool_size_ = 0;

  std::unique_ptr<cuda::device_memory_pool> device_pool_;
  std::unique_ptr<cuda::mr::legacy_pinned_memory_resource> host_resource_;

  mutable std::shared_mutex shared_mutex_;
  std::unordered_map<std::byte*, AllocationInfo> pool_map_;
  size_t host_used_bytes_ = 0;
};

}  // namespace holoscan
