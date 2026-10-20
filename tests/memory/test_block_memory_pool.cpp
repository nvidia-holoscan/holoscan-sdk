/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "../common/allocation_counter.hpp"
#include "../common/cuda_test_utils.hpp"

#include <holoscan/memory/block_memory_pool.hpp>

namespace holoscan {
using ::holoscan::test::blocking_host_callback;
using ::holoscan::test::BlockingHostState;
using ::holoscan::test::create_non_blocking_cuda_stream;
using ::holoscan::test::release_blocking_host;
using ::holoscan::test::ScopedAllocationCounter;
using ::holoscan::test::skip_without_cuda_device;
using ::holoscan::test::UniqueCudaStream;
using ::holoscan::test::wait_until_started;

// Verifies the fixed-size system-memory pool checks out unique blocks, reports exhaustion, and
// reuses a returned block without allocating beyond the configured capacity.
TEST(BlockMemoryPool, AllocatesAndReusesSystemBlocks) {
  BlockMemoryPool pool(MemoryStorageType::kSystem, 64, 2);
  auto init = pool.initialize_pool();
  ASSERT_TRUE(init) << runtime_error_message(init.error());

  auto first = pool.allocate(32, MemoryStorageType::kSystem);
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  auto second = pool.allocate(64, MemoryStorageType::kSystem);
  ASSERT_TRUE(second) << runtime_error_message(second.error());
  EXPECT_NE(first.value(), second.value());

  auto unavailable = pool.allocate(1, MemoryStorageType::kSystem);
  EXPECT_FALSE(unavailable);

  const auto first_address = reinterpret_cast<std::uintptr_t>(first.value());
  auto free_first = pool.free(first.value());
  EXPECT_TRUE(free_first) << runtime_error_message(free_first.error());
  auto third = pool.allocate(1, MemoryStorageType::kSystem);
  ASSERT_TRUE(third) << runtime_error_message(third.error());
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(third.value()), first_address);

  EXPECT_TRUE(pool.free(second.value()));
  EXPECT_TRUE(pool.free(third.value()));
  EXPECT_TRUE(pool.deinitialize_pool());
}

// Verifies the constrained fixed-block path has no steady-state heap allocation after pool
// initialization: allocation/free only checks out and returns preallocated blocks.
TEST(BlockMemoryPool, SystemBlockAcquireReleaseHotPathDoesNotAllocate) {
  BlockMemoryPool pool(MemoryStorageType::kSystem, 64, 2);
  auto init = pool.initialize_pool();
  ASSERT_TRUE(init) << runtime_error_message(init.error());

  // Warm first use outside the counted window so the assertion is about the fixed-pool hot path.
  auto warmup = pool.allocate(64, MemoryStorageType::kSystem);
  ASSERT_TRUE(warmup) << runtime_error_message(warmup.error());
  ASSERT_TRUE(pool.free(warmup.value()));

  {
    const ScopedAllocationCounter allocations;
    for (int iteration = 0; iteration < 16; ++iteration) {
      auto first_result = pool.allocate(64, MemoryStorageType::kSystem);
      ASSERT_TRUE(first_result) << runtime_error_message(first_result.error());
      auto* first = first_result.value();

      auto second_result = pool.allocate(1, MemoryStorageType::kSystem);
      ASSERT_TRUE(second_result) << runtime_error_message(second_result.error());
      auto* second = second_result.value();

      const auto free_first = pool.free(first);
      ASSERT_TRUE(free_first) << runtime_error_message(free_first.error());
      const auto free_second = pool.free(second);
      ASSERT_TRUE(free_second) << runtime_error_message(free_second.error());
    }
    EXPECT_EQ(allocations.count(), 0);
  }

  EXPECT_TRUE(pool.deinitialize_pool());
}

// Verifies the pool rejects requests that do not match its configured storage class or block size.
TEST(BlockMemoryPool, RejectsWrongStorageTypeAndOversizedAllocation) {
  BlockMemoryPool pool(MemoryStorageType::kSystem, 64, 1);
  ASSERT_TRUE(pool.initialize_pool());

  auto wrong_type = pool.allocate(16, MemoryStorageType::kDevice);
  EXPECT_FALSE(wrong_type);

  auto too_large = pool.allocate(65, MemoryStorageType::kSystem);
  EXPECT_FALSE(too_large);

  EXPECT_TRUE(pool.deinitialize_pool());
}

// Verifies availability checks match fixed-block allocation semantics: one allocation can only
// consume one block, so a request larger than block_size is unavailable even when multiple blocks
// are free.
TEST(BlockMemoryPool, AvailabilityRejectsOversizedSingleAllocation) {
  BlockMemoryPool pool(MemoryStorageType::kSystem, 64, 2);
  ASSERT_TRUE(pool.initialize_pool());

  EXPECT_TRUE(pool.is_available(64, MemoryStorageType::kSystem));
  auto too_large = pool.is_available(65, MemoryStorageType::kSystem);
  EXPECT_FALSE(too_large);

  EXPECT_TRUE(pool.deinitialize_pool());
}

// Verifies observability for fixed-pool pressure: checked-out blocks, high-water usage, and
// failed acquisitions are visible without waiting for deinitialization warnings.
TEST(BlockMemoryPool, StatsTrackCapacityUsageHighWaterAndFailedAcquisitions) {
  // Key setup: stats-enabled 2-block pool; allocate both blocks, exhaust, then free one block.
  // Key assertion: used, available, high_water_used, and failed_acquisitions all track correctly.
  BlockMemoryPool::Config config;
  config.storage_type = MemoryStorageType::kSystem;
  config.block_size = 64;
  config.num_blocks = 2;
  config.enable_stats = true;
  BlockMemoryPool pool(config);
  auto before_init = pool.stats();
  EXPECT_FALSE(before_init.initialized);
  EXPECT_TRUE(before_init.counters_enabled);
  EXPECT_EQ(before_init.capacity, 2);
  EXPECT_EQ(before_init.available, 0);

  ASSERT_TRUE(pool.initialize_pool());
  auto initial = pool.stats();
  EXPECT_TRUE(initial.initialized);
  EXPECT_EQ(initial.capacity, 2);
  EXPECT_EQ(initial.available, 2);
  EXPECT_EQ(initial.used, 0);
  EXPECT_EQ(initial.deferred, 0);
  EXPECT_EQ(initial.high_water_used, 0);
  EXPECT_EQ(initial.failed_acquisitions, 0);
  EXPECT_TRUE(initial.counters_enabled);

  auto first = pool.allocate(64, MemoryStorageType::kSystem);
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  auto after_first = pool.stats();
  EXPECT_EQ(after_first.available, 1);
  EXPECT_EQ(after_first.used, 1);
  EXPECT_EQ(after_first.high_water_used, 1);

  auto second = pool.allocate(64, MemoryStorageType::kSystem);
  ASSERT_TRUE(second) << runtime_error_message(second.error());
  auto after_second = pool.stats();
  EXPECT_EQ(after_second.available, 0);
  EXPECT_EQ(after_second.used, 2);
  EXPECT_EQ(after_second.high_water_used, 2);

  auto available = pool.allocate(1, MemoryStorageType::kSystem);
  EXPECT_FALSE(available);
  auto exhausted = pool.stats();
  EXPECT_EQ(exhausted.failed_acquisitions, 1);
  EXPECT_EQ(exhausted.used, 2);
  EXPECT_EQ(exhausted.high_water_used, 2);

  EXPECT_TRUE(pool.free(first.value()));
  auto after_free = pool.stats();
  EXPECT_EQ(after_free.available, 1);
  EXPECT_EQ(after_free.used, 1);
  EXPECT_EQ(after_free.high_water_used, 2);

  EXPECT_TRUE(pool.free(second.value()));
  EXPECT_TRUE(pool.deinitialize_pool());
}

// Verifies a CUDA-device block freed with a stream is not returned to the fixed pool until the
// event recorded on that stream has completed.
TEST(BlockMemoryPool, StreamFreeDefersDeviceBlockReuseUntilEventCompletes) {
  skip_without_cuda_device();

  UniqueCudaStream stream;
  ASSERT_EQ(create_non_blocking_cuda_stream(stream), cudaSuccess);

  BlockMemoryPool pool(MemoryStorageType::kDevice, 64, 1);
  ASSERT_TRUE(pool.initialize_pool());

  // Key setup: a blocking CUDA host function stalls the stream so the deferred-free CUDA event
  //   has not yet completed when the second allocate() is attempted.
  // Key assertion: pool block is unavailable while the callback blocks; reusable after sync.
  // Allocation is a fixed-pool block pop. Stream ordering matters at release:
  // free_async(pointer, stream) below delays reuse until outstanding CUDA work completes.
  auto first = pool.allocate(64, MemoryStorageType::kDevice);
  ASSERT_TRUE(first) << runtime_error_message(first.error());

  BlockingHostState state;
  ASSERT_EQ(cudaLaunchHostFunc(stream.get(), blocking_host_callback, &state), cudaSuccess);
  if (!wait_until_started(state)) {
    release_blocking_host(state);
    EXPECT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
    EXPECT_TRUE(pool.free(first.value()));
    EXPECT_TRUE(pool.deinitialize_pool());
    FAIL() << "Timed out waiting for CUDA host function to start";
  }

  auto deferred_free = pool.free_async(first.value(), stream.get());
  EXPECT_TRUE(deferred_free) << runtime_error_message(deferred_free.error());

  auto unavailable = pool.allocate(1, MemoryStorageType::kDevice);
  EXPECT_FALSE(unavailable);

  release_blocking_host(state);
  EXPECT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);

  auto second = pool.allocate(1, MemoryStorageType::kDevice);
  ASSERT_TRUE(second) << runtime_error_message(second.error());
  EXPECT_EQ(second.value(), first.value());

  EXPECT_TRUE(pool.free(second.value()));
  EXPECT_TRUE(pool.deinitialize_pool());
}

// Verifies repeated stream-deferred free/reallocate cycles work with the pool's per-block event
// path. Event reuse itself follows from the implementation creating events only at initialization.
TEST(BlockMemoryPool, StreamFreeWorksAcrossRepeatedDeferredCycles) {
  skip_without_cuda_device();

  UniqueCudaStream stream;
  ASSERT_EQ(create_non_blocking_cuda_stream(stream), cudaSuccess);

  BlockMemoryPool pool(MemoryStorageType::kDevice, 64, 1);
  ASSERT_TRUE(pool.initialize_pool());

  for (int cycle = 0; cycle < 4; ++cycle) {
    auto block = pool.allocate(64, MemoryStorageType::kDevice);
    ASSERT_TRUE(block) << runtime_error_message(block.error());
    auto released = pool.free_async(block.value(), stream.get());
    EXPECT_TRUE(released) << runtime_error_message(released.error());
    EXPECT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  }

  auto block = pool.allocate(64, MemoryStorageType::kDevice);
  ASSERT_TRUE(block) << runtime_error_message(block.error());

  EXPECT_TRUE(pool.free(block.value()));
  EXPECT_TRUE(pool.deinitialize_pool());
}

// Verifies a failed release-fence record does not make a device block available for reuse. If the
// CUDA event cannot be recorded, the block must remain checked out so callers can retry or fail
// safely instead of reusing memory while unknown stream work may still be pending.
// REQ: MEM-BLOCK-012
TEST(BlockMemoryPool, StreamFreeRecordFailureKeepsDeviceBlockCheckedOut) {
  skip_without_cuda_device();

  BlockMemoryPool pool(MemoryStorageType::kDevice, 64, 1);
  ASSERT_TRUE(pool.initialize_pool());

  // Key setup: a stream from another context. The pool's events belong to the current context, so
  // recording one on this stream fails with a defined error. A destroyed stream handle would also
  // fail, but using one is undefined and can crash in the driver.
  CUcontext pool_context{};
  ASSERT_EQ(cuCtxGetCurrent(&pool_context), CUDA_SUCCESS);
  ASSERT_NE(pool_context, nullptr);
  CUdevice device{};
  ASSERT_EQ(cuCtxGetDevice(&device), CUDA_SUCCESS);
  struct OtherContextStream {
    CUcontext pool_context{};
    CUcontext context{};
    CUstream stream{};
    ~OtherContextStream() {
      if (stream != nullptr) {
        static_cast<void>(cuCtxSetCurrent(context));
        static_cast<void>(cuStreamDestroy(stream));
      }
      if (context != nullptr) {
        static_cast<void>(cuCtxDestroy(context));
      }
      static_cast<void>(cuCtxSetCurrent(pool_context));
    }
  } other{.pool_context = pool_context};
#if CUDA_VERSION >= 13000
  ASSERT_EQ(cuCtxCreate(&other.context, nullptr, 0U, device), CUDA_SUCCESS);
#else
  ASSERT_EQ(cuCtxCreate(&other.context, 0U, device), CUDA_SUCCESS);
#endif
  // cuCtxCreate pushes the new context; balance the push before selecting contexts explicitly.
  CUcontext popped{};
  ASSERT_EQ(cuCtxPopCurrent(&popped), CUDA_SUCCESS);
  ASSERT_EQ(cuCtxSetCurrent(other.context), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamCreate(&other.stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  ASSERT_EQ(cuCtxSetCurrent(pool_context), CUDA_SUCCESS);

  auto block = pool.allocate(64, MemoryStorageType::kDevice);
  ASSERT_TRUE(block) << runtime_error_message(block.error());

  auto failed_release = pool.free_async(block.value(), other.stream);
  EXPECT_FALSE(failed_release);
  EXPECT_NE(runtime_error_message(failed_release.error()).find("cudaEventRecord failed"),
            std::string::npos);
  (void)cudaGetLastError();

  auto available = pool.allocate(1, MemoryStorageType::kDevice);
  EXPECT_FALSE(available);

  EXPECT_TRUE(pool.free(block.value()));
  EXPECT_TRUE(pool.deinitialize_pool());
}

// Verifies stream-deferred releases are reported separately from blocks still owned by callers.
TEST(BlockMemoryPool, StatsReportDeferredDeviceBlocks) {
  skip_without_cuda_device();

  UniqueCudaStream stream;
  ASSERT_EQ(create_non_blocking_cuda_stream(stream), cudaSuccess);

  // Key setup: blocking host callback stalls the CUDA stream so free_async defers completion.
  // Key assertion: deferred counter increments while callback blocks; clears after stream sync.
  BlockMemoryPool::Config config;
  config.storage_type = MemoryStorageType::kDevice;
  config.block_size = 64;
  config.num_blocks = 1;
  config.enable_stats = true;
  BlockMemoryPool pool(config);
  ASSERT_TRUE(pool.initialize_pool());

  auto first = pool.allocate(64, MemoryStorageType::kDevice);
  ASSERT_TRUE(first) << runtime_error_message(first.error());

  BlockingHostState state;
  ASSERT_EQ(cudaLaunchHostFunc(stream.get(), blocking_host_callback, &state), cudaSuccess);
  if (!wait_until_started(state)) {
    release_blocking_host(state);
    EXPECT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
    EXPECT_TRUE(pool.free(first.value()));
    EXPECT_TRUE(pool.deinitialize_pool());
    FAIL() << "Timed out waiting for CUDA host function to start";
  }

  auto deferred_free = pool.free_async(first.value(), stream.get());
  EXPECT_TRUE(deferred_free) << runtime_error_message(deferred_free.error());

  auto pending = pool.stats();
  EXPECT_EQ(pending.capacity, 1);
  EXPECT_EQ(pending.available, 0);
  EXPECT_EQ(pending.used, 0);
  EXPECT_EQ(pending.deferred, 1);
  EXPECT_EQ(pending.high_water_used, 1);

  release_blocking_host(state);
  EXPECT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);

  auto complete = pool.stats();
  EXPECT_EQ(complete.available, 1);
  EXPECT_EQ(complete.used, 0);
  EXPECT_EQ(complete.deferred, 0);

  EXPECT_TRUE(pool.deinitialize_pool());
}

// Verifies unreleased blocks are observable before shutdown so callers can fail a lifecycle
// health check instead of discovering leaked loans only from deinitialization logs.
TEST(BlockMemoryPool, StatsReportUnreleasedBlocksBeforeShutdown) {
  BlockMemoryPool pool(MemoryStorageType::kSystem, 64, 1);
  ASSERT_TRUE(pool.initialize_pool());

  auto block = pool.allocate(64, MemoryStorageType::kSystem);
  ASSERT_TRUE(block) << runtime_error_message(block.error());

  auto stats = pool.stats();
  EXPECT_EQ(stats.available, 0);
  EXPECT_EQ(stats.used, 1);
  EXPECT_EQ(stats.deferred, 0);

  EXPECT_TRUE(pool.deinitialize_pool());
}

// Verifies the pool rejects a pointer that was not checked out from the pool.
TEST(BlockMemoryPool, RejectsInvalidFree) {
  BlockMemoryPool pool(MemoryStorageType::kSystem, 64, 1);
  ASSERT_TRUE(pool.initialize_pool());

  std::array<std::byte, 64> external{};
  // NOLINTNEXTLINE(clang-analyzer-unix.Malloc)
  auto result = pool.free(external.data());
  EXPECT_FALSE(result);

  EXPECT_TRUE(pool.deinitialize_pool());
}

}  // namespace holoscan
