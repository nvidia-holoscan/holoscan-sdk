/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>

#include <holoscan/memory/cuda_stream_allocator.hpp>

namespace holoscan::test {

class StreamRecordingAllocator : public CudaStreamAllocator {
 public:
  expected<void, Error> is_available([[maybe_unused]] uint64_t size,
                                     [[maybe_unused]] MemoryStorageType storage_type) override {
    return {};
  }

  expected<std::byte*, Error> allocate(uint64_t size, MemoryStorageType storage_type) override {
    last_storage_type = storage_type;
    ++allocate_count;
    return new std::byte[size == 0 ? 1 : size];
  }

  expected<void, Error> free(std::byte* pointer) override {
    return free_async_impl(pointer, nullptr);
  }

 protected:
  expected<std::byte*, Error> allocate_async_impl(uint64_t size, MemoryStorageType storage_type,
                                                  cudaStream_t stream) override {
    allocated_stream = stream;
    return allocate(size, storage_type);
  }

  expected<void, Error> free_async_impl(std::byte* pointer, cudaStream_t stream) override {
    released_stream = stream;
    ++free_count;
    delete[] pointer;
    return {};
  }

 public:
  int allocate_count = 0;
  int free_count = 0;
  void* allocated_stream = nullptr;
  void* released_stream = nullptr;
  MemoryStorageType last_storage_type = MemoryStorageType::kHost;
};

}  // namespace holoscan::test
