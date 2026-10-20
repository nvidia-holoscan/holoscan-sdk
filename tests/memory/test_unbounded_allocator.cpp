/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <string>
#include <type_traits>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "../common/cuda_test_utils.hpp"

#include <holoscan/memory/allocator.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>

namespace holoscan {
using ::holoscan::test::skip_without_cuda_device;

namespace {

static_assert(std::is_base_of_v<Allocator, UnboundedAllocator>);

}  // namespace

// Verifies the first concrete native allocator implementation uses the facade directly, without
// routing through the GXF-compatible Allocator hierarchy.
TEST(UnboundedAllocator, AllocatesAndFreesSystemMemory) {
  UnboundedAllocator allocator;
  EXPECT_TRUE(allocator.is_available(128, MemoryStorageType::kSystem));

  auto allocated = allocator.allocate(128, MemoryStorageType::kSystem);
  ASSERT_TRUE(allocated) << runtime_error_message(allocated.error());
  ASSERT_NE(allocated.value(), nullptr);
  allocated.value()[0] = std::byte{0x1};
  allocated.value()[127] = std::byte{0x2};
  EXPECT_TRUE(allocator.free(allocated.value()));

  auto zero_byte = allocator.allocate(0, MemoryStorageType::kSystem);
  ASSERT_TRUE(zero_byte) << runtime_error_message(zero_byte.error());
  ASSERT_NE(zero_byte.value(), nullptr);
  EXPECT_TRUE(allocator.free(zero_byte.value()));
}

// Verifies the allocator tracks its own allocations and rejects foreign pointers, preventing
// callers from accidentally double-freeing or interleaving pointers from different allocators.
TEST(UnboundedAllocator, RejectsUnrecognizedPointer) {
  UnboundedAllocator allocator;
  std::byte foreign{};

  // Intentionally pass a foreign stack pointer to verify it is rejected without deallocation.
  // NOLINTNEXTLINE(clang-analyzer-unix.Malloc)
  auto freed = allocator.free(&foreign);
  ASSERT_FALSE(freed);
  EXPECT_EQ(freed.error().code, ErrorCode::kInvalidArgument);
}

// Verifies unsupported storage-type values fail through expected rather than falling through to an
// allocator implementation path.
TEST(UnboundedAllocator, RejectsInvalidStorageType) {
  UnboundedAllocator allocator;
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  const auto invalid_storage = static_cast<MemoryStorageType>(255);

  auto available = allocator.is_available(16, invalid_storage);
  EXPECT_FALSE(available);

  auto allocated = allocator.allocate(16, invalid_storage);
  ASSERT_FALSE(allocated);
  EXPECT_NE(runtime_error_message(allocated.error()).find("Invalid memory storage type"),
            std::string::npos);
}

// Verifies the native unbounded allocator can allocate device memory for bring-up tests. This path
// dynamically calls cudaMalloc and is not deterministic-profile evidence.
TEST(UnboundedAllocator, AllocatesCudaMemoryWhenDeviceAvailable) {
  skip_without_cuda_device();

  UnboundedAllocator allocator;
  auto allocated = allocator.allocate(64, MemoryStorageType::kDevice);
  ASSERT_TRUE(allocated) << runtime_error_message(allocated.error());
  ASSERT_NE(allocated.value(), nullptr);
  EXPECT_TRUE(allocator.free(allocated.value()));
}

// Verifies the native unbounded allocator can allocate CUDA pinned host memory for simple transfer
// tests, while keeping this dynamic path separate from bounded pool evidence.
TEST(UnboundedAllocator, AllocatesPinnedHostMemoryWhenDeviceAvailable) {
  skip_without_cuda_device();

  UnboundedAllocator allocator;
  auto allocated = allocator.allocate(64, MemoryStorageType::kHost);
  ASSERT_TRUE(allocated) << runtime_error_message(allocated.error());
  ASSERT_NE(allocated.value(), nullptr);

  // Key setup: allocate kHost memory, then query cudaPointerAttributes for the result pointer.
  // Key assertion: attributes.type == cudaMemoryTypeHost confirms CUDA pinned-host backing.
  cudaPointerAttributes attributes{};
  ASSERT_EQ(cudaPointerGetAttributes(&attributes, allocated.value()), cudaSuccess);
  EXPECT_EQ(attributes.type, cudaMemoryTypeHost);

  allocated.value()[0] = std::byte{0x3};
  allocated.value()[63] = std::byte{0x4};

  EXPECT_TRUE(allocator.free(allocated.value()));
}

// Verifies the native unbounded allocator can allocate CUDA managed memory for unconstrained
// bring-up cases that need a single CPU/GPU-visible pointer.
TEST(UnboundedAllocator, AllocatesManagedMemoryWhenDeviceAvailable) {
  skip_without_cuda_device();

  UnboundedAllocator allocator;
  auto allocated = allocator.allocate(64, MemoryStorageType::kCudaManaged);
  ASSERT_TRUE(allocated) << runtime_error_message(allocated.error());
  ASSERT_NE(allocated.value(), nullptr);

  allocated.value()[0] = std::byte{0x5};
  allocated.value()[63] = std::byte{0x6};

  EXPECT_TRUE(allocator.free(allocated.value()));
}

}  // namespace holoscan
