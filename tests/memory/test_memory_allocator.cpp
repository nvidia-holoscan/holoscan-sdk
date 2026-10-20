/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>
#include <new>
#include <string>
#include <type_traits>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cuda/stream>

#include "../common/cuda_test_utils.hpp"

#include <holoscan/memory/allocator.hpp>
#include <holoscan/memory/block_memory_pool.hpp>
#include <holoscan/memory/cuda_stream_allocator.hpp>

namespace holoscan {
using ::holoscan::test::create_non_blocking_cuda_stream;
using ::holoscan::test::skip_without_cuda_device;
using ::holoscan::test::UniqueCudaStream;

namespace {

static_assert(std::is_base_of_v<Allocator, CudaStreamAllocator>);
static_assert(std::is_base_of_v<Allocator, BlockMemoryPool>);
static_assert(std::is_base_of_v<CudaStreamAllocator, BlockMemoryPool>);

// Minimal Allocator subclass for API-contract tests: system-only, fixed 64-byte capacity,
// heap-backed. Verifies that the expected<>-returning interface works without GXF lifecycle
// dependencies. Representative test: MemoryAllocator.FacadeUsesExpectedAndDoesNotInheritComponent.
class TestAllocator final : public Allocator {
 public:
  expected<void, Error> is_available(uint64_t size, MemoryStorageType storage_type) override {
    if (storage_type != MemoryStorageType::kSystem) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "Only system storage is supported"});
    }
    if (size > max_size) {
      return make_unexpected(
          Error{ErrorCode::kFailure, "Requested allocation exceeds test capacity"});
    }
    return {};
  }

  expected<std::byte*, Error> allocate(uint64_t size, MemoryStorageType storage_type) override {
    auto available = is_available(size, storage_type);
    if (!available) {
      return forward_error(available);
    }
    auto* pointer = new (std::nothrow) std::byte[size == 0 ? 1 : size];
    if (pointer == nullptr) {
      return make_unexpected(Error{ErrorCode::kFailure, "System allocation failed"});
    }
    return pointer;
  }

  expected<void, Error> free(std::byte* pointer) override {
    delete[] pointer;
    return {};
  }

  uint64_t block_size(MemoryStorageType storage_type) const override {
    return storage_type == MemoryStorageType::kSystem ? max_size : 0;
  }

  uint64_t max_size = 64;
};

// Minimal CudaStreamAllocator subclass that records the stream handle passed to async overloads.
// Verifies that stream_ref overloads forward the raw cudaStream_t without conversion loss.
// Representative test: MemoryAllocator.CudaStreamRefOverloadsForwardRawStream.
class TestCudaStreamAllocator final : public CudaStreamAllocator {
 public:
  expected<void, Error> is_available(uint64_t size, MemoryStorageType storage_type) override {
    if (storage_type != MemoryStorageType::kDevice) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "Only device storage is supported"});
    }
    if (size > max_size) {
      return make_unexpected(
          Error{ErrorCode::kFailure, "Requested allocation exceeds test capacity"});
    }
    return {};
  }

  expected<std::byte*, Error> allocate(uint64_t size, MemoryStorageType storage_type) override {
    return allocate_common(size, storage_type);
  }

  expected<void, Error> free(std::byte* pointer) override {
    delete[] pointer;
    return {};
  }

  expected<std::byte*, Error> allocate_async_impl(uint64_t size, MemoryStorageType storage_type,
                                                  cudaStream_t stream) override {
    last_stream = stream;
    ++allocate_async_count;
    return allocate_common(size, storage_type);
  }

  expected<void, Error> free_async_impl(std::byte* pointer, cudaStream_t stream) override {
    last_stream = stream;
    ++free_async_count;
    return free(pointer);
  }

  uint64_t max_size = 64;
  int allocate_async_count = 0;
  int free_async_count = 0;
  cudaStream_t last_stream = nullptr;

 private:
  expected<std::byte*, Error> allocate_common(uint64_t size, MemoryStorageType storage_type) {
    auto available = is_available(size, storage_type);
    if (!available) {
      return forward_error(available);
    }
    auto* pointer = new (std::nothrow) std::byte[size == 0 ? 1 : size];
    if (pointer == nullptr) {
      return make_unexpected(Error{ErrorCode::kFailure, "System allocation failed"});
    }
    return pointer;
  }
};

}  // namespace

// Pins the native allocator candidate shape: expected-returning allocation over Holoscan storage
// types, with no GXF Component or registrar lifecycle dependency.
TEST(MemoryAllocator, FacadeUsesExpectedAndDoesNotInheritComponent) {
  TestAllocator allocator;

  auto available = allocator.is_available(16, MemoryStorageType::kSystem);
  EXPECT_TRUE(available) << runtime_error_message(available.error());
  EXPECT_EQ(allocator.block_size(MemoryStorageType::kSystem), 64);

  auto allocated = allocator.allocate(16, MemoryStorageType::kSystem);
  ASSERT_TRUE(allocated) << runtime_error_message(allocated.error());
  EXPECT_TRUE(allocator.free(allocated.value()));

  auto wrong_storage = allocator.allocate(16, MemoryStorageType::kDevice);
  EXPECT_FALSE(wrong_storage);
}

// Pins the CUDA-specific extension shape: stream-aware allocation/free accepts CCCL stream_ref
// without forcing CUDA stream concepts into the base allocator interface.
TEST(MemoryAllocator, CudaStreamRefOverloadsForwardRawStream) {
  // Key setup: synthetic cudaStream_t cast from a pointer-sized constant; wrapped in stream_ref
  //   to exercise the overload without requiring a real CUDA device.
  // Key assertion: last_stream matches the original handle after both allocate_async and
  // free_async.
  TestCudaStreamAllocator allocator;
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(uintptr_t{0x1234});
  const ::cuda::stream_ref stream_ref(stream);

  auto allocated = allocator.allocate_async(16, MemoryStorageType::kDevice, stream_ref);
  ASSERT_TRUE(allocated) << runtime_error_message(allocated.error());
  EXPECT_EQ(allocator.allocate_async_count, 1);
  EXPECT_EQ(allocator.last_stream, stream);

  auto freed = allocator.free_async(allocated.value(), stream_ref);
  EXPECT_TRUE(freed) << runtime_error_message(freed.error());
  EXPECT_EQ(allocator.free_async_count, 1);
  EXPECT_EQ(allocator.last_stream, stream);
}

// Verifies CCCL stream_ref overloads preserve the native CUDA stream handle from a real stream,
// matching the behavior expected when runtime-owned stream pools hand streams to allocators.
TEST(MemoryAllocator, CudaStreamRefOverloadsForwardCreatedStream) {
  skip_without_cuda_device();

  UniqueCudaStream stream;
  ASSERT_EQ(create_non_blocking_cuda_stream(stream), cudaSuccess);
  const ::cuda::stream_ref stream_ref(stream.get());

  TestCudaStreamAllocator allocator;
  auto allocated = allocator.allocate_async(16, MemoryStorageType::kDevice, stream_ref);
  ASSERT_TRUE(allocated) << runtime_error_message(allocated.error());
  EXPECT_EQ(allocator.allocate_async_count, 1);
  EXPECT_EQ(allocator.last_stream, stream.get());

  auto freed = allocator.free_async(allocated.value(), stream_ref);
  EXPECT_TRUE(freed) << runtime_error_message(freed.error());
  EXPECT_EQ(allocator.free_async_count, 1);
  EXPECT_EQ(allocator.last_stream, stream.get());
}

}  // namespace holoscan
