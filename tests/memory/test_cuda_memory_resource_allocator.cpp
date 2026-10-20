/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include "../common/cuda_test_utils.hpp"

#include <holoscan/memory/cuda_memory_resource_allocator.hpp>

namespace holoscan {
using ::holoscan::test::create_non_blocking_cuda_stream;
using ::holoscan::test::skip_without_cuda_device;
using ::holoscan::test::UniqueCudaStream;

// Verifies pure configuration errors are reported before CUDA resources are created. A failed size
// parse should leave the allocator uninitialized and reusable for a clean retry/deinitialize path.
TEST(CudaMemoryResourceAllocator, InvalidSizeFailsBeforeCudaResourceCreation) {
  CudaMemoryResourceAllocator allocator(
      CudaMemoryResourceAllocator::Config{"not-a-size", "2MB", "1MB", "2MB", 0});

  auto first = allocator.initialize_pool();
  EXPECT_FALSE(first);
  EXPECT_NE(runtime_error_message(first.error()).find("Storage size value is not a number"),
            std::string::npos);

  auto stats = allocator.stats();
  ASSERT_TRUE(stats) << runtime_error_message(stats.error());
  EXPECT_FALSE(stats.value().initialized);

  auto second = allocator.initialize_pool();
  EXPECT_FALSE(second);
  EXPECT_NE(runtime_error_message(second.error()).find("Storage size value is not a number"),
            std::string::npos);

  EXPECT_TRUE(allocator.deinitialize_pool());
}

// Verifies initial/max pool-size consistency is checked before CUDA setup. A max smaller than the
// initial reservation should fail as configuration, not as a later CCCL construction error.
TEST(CudaMemoryResourceAllocator, RejectsMaxSizeSmallerThanInitialBeforeCudaSetup) {
  CudaMemoryResourceAllocator allocator(
      CudaMemoryResourceAllocator::Config{"2MB", "1MB", "1MB", "2MB", 0});

  auto result = allocator.initialize_pool();
  EXPECT_FALSE(result);
  EXPECT_NE(runtime_error_message(result.error()).find("Device memory max size must be greater"),
            std::string::npos);

  auto stats = allocator.stats();
  ASSERT_TRUE(stats) << runtime_error_message(stats.error());
  EXPECT_FALSE(stats.value().initialized);

  EXPECT_TRUE(allocator.deinitialize_pool());
}

// Verifies the native stream-aware allocator API reaches the CCCL-backed device allocation path and
// can be used with CUDA work on the same stream.
TEST(CudaMemoryResourceAllocator, GenericStreamApiAllocatesDeviceMemory) {
  skip_without_cuda_device();

  UniqueCudaStream stream;
  ASSERT_EQ(create_non_blocking_cuda_stream(stream), cudaSuccess);

  CudaMemoryResourceAllocator allocator(
      CudaMemoryResourceAllocator::Config{"1MB", "2MB", "1MB", "2MB", 0});
  ASSERT_TRUE(allocator.initialize_pool());

  auto maybe_pointer = allocator.allocate_async(256, MemoryStorageType::kDevice, stream.get());
  ASSERT_TRUE(maybe_pointer) << runtime_error_message(maybe_pointer.error());
  ASSERT_NE(maybe_pointer.value(), nullptr);

  ASSERT_EQ(cudaMemsetAsync(maybe_pointer.value(), 0x5A, 256, stream.get()), cudaSuccess);

  auto free_result = allocator.free_async(maybe_pointer.value(), stream.get());
  ASSERT_TRUE(free_result) << runtime_error_message(free_result.error());
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);

  EXPECT_TRUE(allocator.deinitialize_pool());
}

// Verifies the Holoscan-facing allocator capacity is enforced before asking CCCL to allocate:
// oversized device and pinned-host requests fail predictably and do not enter the allocation map.
TEST(CudaMemoryResourceAllocator, RejectsAllocationsBeyondConfiguredMax) {
  skip_without_cuda_device();

  UniqueCudaStream stream;
  ASSERT_EQ(create_non_blocking_cuda_stream(stream), cudaSuccess);

  CudaMemoryResourceAllocator allocator(
      CudaMemoryResourceAllocator::Config{"1MB", "1MB", "1MB", "1MB", 0});
  ASSERT_TRUE(allocator.initialize_pool());

  auto too_large_device =
      allocator.allocate_async(2 * 1024 * 1024, MemoryStorageType::kDevice, stream.get());
  EXPECT_FALSE(too_large_device);
  EXPECT_NE(
      runtime_error_message(too_large_device.error()).find("Requested memory is not available"),
      std::string::npos);

  auto too_large_host = allocator.allocate(2 * 1024 * 1024, MemoryStorageType::kHost);
  EXPECT_FALSE(too_large_host);
  EXPECT_NE(runtime_error_message(too_large_host.error()).find("Requested memory is not available"),
            std::string::npos);

  auto stats = allocator.stats();
  ASSERT_TRUE(stats) << runtime_error_message(stats.error());
  EXPECT_EQ(stats.value().allocation_count, 0);
  EXPECT_EQ(stats.value().host_used_bytes, 0);

  EXPECT_TRUE(allocator.deinitialize_pool());
}

// Verifies the CCCL-backed allocator's pinned-host path allocates CUDA-visible host memory and
// tracks host usage separately from the device memory pool.
TEST(CudaMemoryResourceAllocator, AllocatesPinnedHostMemoryAndTracksUsage) {
  skip_without_cuda_device();

  CudaMemoryResourceAllocator allocator(
      CudaMemoryResourceAllocator::Config{"1MB", "2MB", "1MB", "2MB", 0});
  ASSERT_TRUE(allocator.initialize_pool());

  auto maybe_pointer = allocator.allocate(256, MemoryStorageType::kHost);
  ASSERT_TRUE(maybe_pointer) << runtime_error_message(maybe_pointer.error());
  ASSERT_NE(maybe_pointer.value(), nullptr);

  cudaPointerAttributes attributes{};
  ASSERT_EQ(cudaPointerGetAttributes(&attributes, maybe_pointer.value()), cudaSuccess);
  EXPECT_EQ(attributes.type, cudaMemoryTypeHost);

  maybe_pointer.value()[0] = std::byte{0xA5};
  maybe_pointer.value()[255] = std::byte{0x5A};

  auto maybe_used = allocator.pool_size(MemoryStorageType::kHost);
  ASSERT_TRUE(maybe_used) << runtime_error_message(maybe_used.error());
  EXPECT_EQ(maybe_used.value(), 256);

  auto free_result = allocator.free(maybe_pointer.value());
  ASSERT_TRUE(free_result) << runtime_error_message(free_result.error());

  maybe_used = allocator.pool_size(MemoryStorageType::kHost);
  ASSERT_TRUE(maybe_used) << runtime_error_message(maybe_used.error());
  EXPECT_EQ(maybe_used.value(), 0);

  EXPECT_TRUE(allocator.deinitialize_pool());
}

// Verifies the CCCL-backed allocator exposes a single read-only statistics snapshot for runtime
// health checks without requiring callers to query device and host pool paths separately.
TEST(CudaMemoryResourceAllocator, StatsReportDeviceAndHostPoolUsage) {
  skip_without_cuda_device();

  UniqueCudaStream stream;
  ASSERT_EQ(create_non_blocking_cuda_stream(stream), cudaSuccess);

  CudaMemoryResourceAllocator allocator(
      CudaMemoryResourceAllocator::Config{"1MB", "2MB", "1MB", "2MB", 0});
  auto before_init = allocator.stats();
  ASSERT_TRUE(before_init) << runtime_error_message(before_init.error());
  EXPECT_FALSE(before_init.value().initialized);

  ASSERT_TRUE(allocator.initialize_pool());

  auto initial = allocator.stats();
  ASSERT_TRUE(initial) << runtime_error_message(initial.error());
  EXPECT_TRUE(initial.value().initialized);
  EXPECT_EQ(initial.value().device_pool_configured_max_bytes, 2 * 1024 * 1024);
  EXPECT_EQ(initial.value().host_configured_max_bytes, 2 * 1024 * 1024);
  EXPECT_EQ(initial.value().host_used_bytes, 0);
  EXPECT_EQ(initial.value().host_available_bytes, 2 * 1024 * 1024);
  EXPECT_EQ(initial.value().allocation_count, 0);

  // Key setup: one async device allocation and one synchronous host allocation are outstanding.
  // Key assertion: allocation_count, host_used_bytes, and device_pool_used_bytes all update in
  //   the combined stats snapshot; values return to zero after both are freed.
  auto device_pointer = allocator.allocate_async(512, MemoryStorageType::kDevice, stream.get());
  ASSERT_TRUE(device_pointer) << runtime_error_message(device_pointer.error());
  auto host_pointer = allocator.allocate(256, MemoryStorageType::kHost);
  ASSERT_TRUE(host_pointer) << runtime_error_message(host_pointer.error());

  auto in_use = allocator.stats();
  ASSERT_TRUE(in_use) << runtime_error_message(in_use.error());
  EXPECT_TRUE(in_use.value().initialized);
  EXPECT_GE(in_use.value().device_pool_used_bytes, 512);
  EXPECT_LE(in_use.value().device_pool_used_bytes, in_use.value().device_pool_configured_max_bytes);
  EXPECT_GE(in_use.value().device_pool_reserved_bytes, in_use.value().device_pool_used_bytes);
  EXPECT_EQ(in_use.value().host_used_bytes, 256);
  EXPECT_EQ(in_use.value().host_available_bytes, 2 * 1024 * 1024 - 256);
  EXPECT_EQ(in_use.value().allocation_count, 2);

  auto free_device = allocator.free_async(device_pointer.value(), stream.get());
  ASSERT_TRUE(free_device) << runtime_error_message(free_device.error());
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  auto free_host = allocator.free(host_pointer.value());
  ASSERT_TRUE(free_host) << runtime_error_message(free_host.error());

  auto after_free = allocator.stats();
  ASSERT_TRUE(after_free) << runtime_error_message(after_free.error());
  EXPECT_EQ(after_free.value().host_used_bytes, 0);
  EXPECT_EQ(after_free.value().allocation_count, 0);

  EXPECT_TRUE(allocator.deinitialize_pool());
}

// Verifies lifecycle teardown returns outstanding device and pinned-host allocations before
// clearing bookkeeping. This catches leaks from callers that shut down without explicitly freeing
// every allocation, while still treating that path as a warning-worthy lifecycle cleanup fallback.
TEST(CudaMemoryResourceAllocator, DeinitializeReleasesOutstandingAllocations) {
  skip_without_cuda_device();

  CudaMemoryResourceAllocator allocator(
      CudaMemoryResourceAllocator::Config{"1MB", "2MB", "1MB", "2MB", 0});
  ASSERT_TRUE(allocator.initialize_pool());

  // Key setup: device and host allocations are left outstanding when deinitialize_pool() runs.
  // Key assertion: stats show uninitialized state with zero host_used_bytes and allocation_count
  //   after deinitialize, confirming the fallback cleanup path reclaims both allocation types.
  auto device_pointer = allocator.allocate(512, MemoryStorageType::kDevice);
  ASSERT_TRUE(device_pointer) << runtime_error_message(device_pointer.error());
  auto host_pointer = allocator.allocate(256, MemoryStorageType::kHost);
  ASSERT_TRUE(host_pointer) << runtime_error_message(host_pointer.error());

  auto in_use = allocator.stats();
  ASSERT_TRUE(in_use) << runtime_error_message(in_use.error());
  EXPECT_TRUE(in_use.value().initialized);
  EXPECT_EQ(in_use.value().host_used_bytes, 256);
  EXPECT_EQ(in_use.value().allocation_count, 2);

  auto deinitialized = allocator.deinitialize_pool();
  EXPECT_TRUE(deinitialized) << runtime_error_message(deinitialized.error());

  auto after = allocator.stats();
  ASSERT_TRUE(after) << runtime_error_message(after.error());
  EXPECT_FALSE(after.value().initialized);
  EXPECT_EQ(after.value().host_used_bytes, 0);
  EXPECT_EQ(after.value().allocation_count, 0);
}

}  // namespace holoscan
