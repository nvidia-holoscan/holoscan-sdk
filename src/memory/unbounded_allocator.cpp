/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/memory/unbounded_allocator.hpp>

#include <holoscan/core/error_helpers.hpp>

#include <cuda_runtime_api.h>

#include <new>

#include "core/diagnostic_reporting.hpp"

namespace holoscan {
namespace {

using error_detail::cuda_runtime_error;

}  // namespace

expected<void, Error> UnboundedAllocator::is_available(uint64_t size,
                                                       MemoryStorageType storage_type) {
  (void)size;
  switch (storage_type) {
    case MemoryStorageType::kHost:
    case MemoryStorageType::kDevice:
    case MemoryStorageType::kCudaManaged:
    case MemoryStorageType::kSystem:
      return {};
    default:
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "Invalid memory storage type"});
  }
}

expected<std::byte*, Error> UnboundedAllocator::allocate(uint64_t size,
                                                         MemoryStorageType storage_type) {
  auto available = is_available(size, storage_type);
  if (!available) {
    return forward_error(available);
  }

  if (size == 0) {
    size = 1;
  }

  void* pointer = nullptr;
  switch (storage_type) {
    case MemoryStorageType::kHost: {
      const cudaError_t error = cudaMallocHost(&pointer, size);
      if (error != cudaSuccess) {
        HOLOSCAN_LOG_AUTOMATIC_ERROR("Failure in cudaMallocHost: {}", cudaGetErrorString(error));
        return make_unexpected<Error>(cuda_runtime_error("cudaMallocHost failed", error));
      }
      const std::lock_guard<std::mutex> lock(mutex_);
      cuda_host_blocks_.insert(static_cast<std::byte*>(pointer));
      break;
    }
    case MemoryStorageType::kDevice: {
      const cudaError_t error = cudaMalloc(&pointer, size);
      if (error != cudaSuccess) {
        HOLOSCAN_LOG_AUTOMATIC_ERROR("Failure in cudaMalloc: {}", cudaGetErrorString(error));
        return make_unexpected<Error>(cuda_runtime_error("cudaMalloc failed", error));
      }
      const std::lock_guard<std::mutex> lock(mutex_);
      cuda_blocks_.insert(static_cast<std::byte*>(pointer));
      break;
    }
    case MemoryStorageType::kCudaManaged: {
      const cudaError_t error = cudaMallocManaged(&pointer, size, cudaMemAttachGlobal);
      if (error != cudaSuccess) {
        HOLOSCAN_LOG_AUTOMATIC_ERROR("Failure in cudaMallocManaged: {}", cudaGetErrorString(error));
        return make_unexpected<Error>(cuda_runtime_error("cudaMallocManaged failed", error));
      }
      const std::lock_guard<std::mutex> lock(mutex_);
      cuda_blocks_.insert(static_cast<std::byte*>(pointer));
      break;
    }
    case MemoryStorageType::kSystem:
      pointer = new (std::nothrow) std::byte[size];
      if (pointer == nullptr) {
        return make_unexpected(Error{ErrorCode::kFailure, "System memory allocation failed"});
      }
      {
        const std::lock_guard<std::mutex> lock(mutex_);
        system_blocks_.insert(static_cast<std::byte*>(pointer));
      }
      break;
    default:
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "Invalid memory storage type"});
  }

  return static_cast<std::byte*>(pointer);
}

expected<void, Error> UnboundedAllocator::free(std::byte* pointer) {
  const std::lock_guard<std::mutex> lock(mutex_);

  const auto cuda_block = cuda_blocks_.find(pointer);
  if (cuda_block != cuda_blocks_.end()) {
    cuda_blocks_.erase(cuda_block);
    const cudaError_t error = cudaFree(pointer);
    if (error != cudaSuccess) {
      HOLOSCAN_LOG_AUTOMATIC_ERROR("Failure in cudaFree: {}", cudaGetErrorString(error));
      return make_unexpected<Error>(cuda_runtime_error("cudaFree failed", error));
    }
    return {};
  }

  const auto system_block = system_blocks_.find(pointer);
  if (system_block != system_blocks_.end()) {
    system_blocks_.erase(system_block);
    delete[] pointer;
    return {};
  }

  const auto cuda_host_block = cuda_host_blocks_.find(pointer);
  if (cuda_host_block != cuda_host_blocks_.end()) {
    cuda_host_blocks_.erase(cuda_host_block);
    const cudaError_t error = cudaFreeHost(pointer);
    if (error != cudaSuccess) {
      HOLOSCAN_LOG_AUTOMATIC_ERROR("Failure in cudaFreeHost: {}", cudaGetErrorString(error));
      return make_unexpected<Error>(cuda_runtime_error("cudaFreeHost failed", error));
    }
    return {};
  }

  HOLOSCAN_LOG_AUTOMATIC_ERROR("UnboundedAllocator cannot free unrecognized pointer");
  return make_unexpected(Error{ErrorCode::kInvalidArgument, "Unrecognized allocation pointer"});
}

}  // namespace holoscan
