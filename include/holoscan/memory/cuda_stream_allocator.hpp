/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda/stream>

#include <cstddef>
#include <cstdint>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/memory/allocator.hpp>
#include <holoscan/memory/memory_storage_type.hpp>

namespace holoscan {

/**
 * @brief CUDA-specific allocator extension for stream-ordered allocation and release.
 *
 * Keeping this separate from Allocator lets CPU/system allocators avoid CUDA stream concepts while
 * CUDA-backed Tensor storage can still pass CCCL stream references through the native Holoscan API.
 */
class HOLOSCAN_CORE_EXPORT CudaStreamAllocator : public Allocator {
 public:
  ~CudaStreamAllocator() override = default;

  [[nodiscard]] expected<std::byte*, Error> allocate_async(uint64_t size,
                                                           MemoryStorageType storage_type,
                                                           cudaStream_t stream) {
    return allocate_async_impl(size, storage_type, stream);
  }

  [[nodiscard]] expected<std::byte*, Error> allocate_async(uint64_t size,
                                                           MemoryStorageType storage_type,
                                                           ::cuda::stream_ref stream) {
    return allocate_async(size, storage_type, stream.get());
  }

  [[nodiscard]] expected<void, Error> free_async(std::byte* pointer, cudaStream_t stream) {
    return free_async_impl(pointer, stream);
  }

  [[nodiscard]] expected<void, Error> free_async(std::byte* pointer, ::cuda::stream_ref stream) {
    return free_async(pointer, stream.get());
  }

 protected:
  [[nodiscard]] virtual expected<std::byte*, Error> allocate_async_impl(
      uint64_t size, MemoryStorageType storage_type, cudaStream_t stream) = 0;
  [[nodiscard]] virtual expected<void, Error> free_async_impl(std::byte* pointer,
                                                              cudaStream_t stream) = 0;
};

}  // namespace holoscan
