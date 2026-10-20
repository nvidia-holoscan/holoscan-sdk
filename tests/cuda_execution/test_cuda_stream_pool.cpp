/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <string>
#include <utility>

#include "../common/allocation_counter.hpp"
#include "../common/cuda_test_utils.hpp"

#include <holoscan/cuda_execution/cuda_stream_pool.hpp>

#include "cuda_execution/cuda_driver_entry_points.hpp"

namespace holoscan {
using ::holoscan::test::invalid_cuda_device_id;
using ::holoscan::test::ScopedAllocationCounter;
using ::holoscan::test::skip_without_cuda_device;

// Verifies acquire() fails cleanly before the stream pool is initialized.
TEST(CudaStreamPool, AcquireFailsBeforeInitialize) {
  CudaStreamPool pool(0, cudaStreamNonBlocking, 0, 1, 1);

  auto maybe_lease = pool.acquire();
  EXPECT_FALSE(maybe_lease);
}

// Verifies callers that do not want Component::initialize() exceptions can initialize the stream
// pool through an expected-returning API and repeat initialization harmlessly.
TEST(CudaStreamPool, InitializePoolReturnsExpectedAndIsIdempotent) {
  skip_without_cuda_device();

  CudaStreamPool pool(0, cudaStreamNonBlocking, 0, 1, 1);
  auto first = pool.initialize_pool();
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  EXPECT_EQ(pool.available(), 1);
  auto second = pool.initialize_pool();
  EXPECT_TRUE(second) << runtime_error_message(second.error());
  EXPECT_EQ(pool.available(), 1);
}

// Verifies explicit deinitialize() invalidates outstanding leases without letting those old
// leases return streams to a later reinitialized pool.
//
// Lifecycle:
//   pool --acquire--> old_lease
//   pool --deinitialize--> (generation bumped; old_lease now stale)
//   pool --reinitialize--> pool --acquire--> new_lease
//   old_lease.release() -> no-op on allocated count (stale generation)
//   new_lease.release() -> decrements allocated to zero
TEST(CudaStreamPool, DeinitializeInvalidatesOutstandingLeasesAndAllowsReinitialize) {
  skip_without_cuda_device();

  CudaStreamPool pool(0, cudaStreamNonBlocking, 0, 1, 1);
  ASSERT_TRUE(pool.initialize_pool());

  // Key setup: acquire a lease before deinitialize so it outlives the pool's generation change.
  auto first = pool.acquire();
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  auto old_lease = std::move(first.value());
  EXPECT_EQ(pool.allocated(), 1);

  auto deinitialized = pool.deinitialize_pool();
  ASSERT_TRUE(deinitialized) << runtime_error_message(deinitialized.error());
  EXPECT_EQ(pool.available(), 0);
  EXPECT_EQ(pool.allocated(), 0);
  EXPECT_FALSE(pool.acquire());

  ASSERT_TRUE(pool.initialize_pool());
  auto second = pool.acquire();
  ASSERT_TRUE(second) << runtime_error_message(second.error());
  auto new_lease = std::move(second.value());
  EXPECT_EQ(pool.allocated(), 1);

  // Key assertion: releasing the stale old_lease does not affect the new-generation count;
  // only new_lease.release() decrements allocated.
  EXPECT_TRUE(old_lease.release());
  EXPECT_EQ(pool.allocated(), 1);
  EXPECT_TRUE(new_lease.release());
  EXPECT_EQ(pool.allocated(), 0);
}

// Verifies CUDA device-selection failures are surfaced as RuntimeError values instead of requiring
// callers to catch exceptions from the stream pool initialization path.
TEST(CudaStreamPool, InitializePoolReportsInvalidDeviceWithoutThrowing) {
  skip_without_cuda_device();

  CudaStreamPool pool(invalid_cuda_device_id(), cudaStreamNonBlocking, 0, 1, 1);
  auto result = pool.initialize_pool();
  ASSERT_FALSE(result);
  EXPECT_NE(runtime_error_message(result.error()).find("Failed to set CUDA stream pool device"),
            std::string::npos);
  (void)cudaGetLastError();
}

// Verifies capacity configuration is explicit: max_size can be zero for unbounded growth, but a
// nonzero max smaller than the reserved count is rejected before CUDA streams are created.
TEST(CudaStreamPool, RejectsMaxSizeSmallerThanReservedBeforeCudaSetup) {
  CudaStreamPool pool(0, cudaStreamNonBlocking, 0, 2, 1);

  auto result = pool.initialize_pool();
  EXPECT_FALSE(result);
  EXPECT_NE(runtime_error_message(result.error()).find("max_size must be zero or at least"),
            std::string::npos);
  EXPECT_EQ(pool.available(), 0);
}

// Verifies the pool returns reserved streams, enforces max_size, and makes an explicitly released
// lease available for reuse.
TEST(CudaStreamPool, AcquireReleaseAndCapacity) {
  skip_without_cuda_device();

  CudaStreamPool pool(0, cudaStreamNonBlocking, 0, 1, 1);
  ASSERT_NO_THROW(pool.initialize());
  EXPECT_EQ(pool.available(), 1);
  EXPECT_EQ(pool.allocated(), 0);

  auto maybe_lease = pool.acquire();
  ASSERT_TRUE(maybe_lease) << runtime_error_message(maybe_lease.error());
  auto lease = std::move(maybe_lease.value());
  EXPECT_TRUE(static_cast<bool>(lease));
  EXPECT_NE(lease.native_handle(), nullptr);
  EXPECT_EQ(pool.available(), 0);
  EXPECT_EQ(pool.allocated(), 1);

  auto second_lease = pool.acquire();
  EXPECT_FALSE(second_lease);

  auto release_result = lease.release();
  EXPECT_TRUE(release_result) << runtime_error_message(release_result.error());
  EXPECT_EQ(pool.available(), 1);
  EXPECT_EQ(pool.allocated(), 0);
}

// Verifies the pool can grow beyond reserved_size until max_size is reached.
TEST(CudaStreamPool, GrowsBeyondReservedStreamsUntilCapacity) {
  skip_without_cuda_device();

  CudaStreamPool pool(0, cudaStreamNonBlocking, 0, 1, 2);
  ASSERT_NO_THROW(pool.initialize());
  EXPECT_EQ(pool.available(), 1);
  EXPECT_EQ(pool.allocated(), 0);

  auto first = pool.acquire();
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  auto first_lease = std::move(first.value());
  EXPECT_EQ(pool.available(), 0);
  EXPECT_EQ(pool.allocated(), 1);

  auto second = pool.acquire();
  ASSERT_TRUE(second) << runtime_error_message(second.error());
  auto second_lease = std::move(second.value());
  EXPECT_EQ(pool.available(), 0);
  EXPECT_EQ(pool.allocated(), 2);

  auto third = pool.acquire();
  EXPECT_FALSE(third);

  EXPECT_TRUE(first_lease.release());
  EXPECT_TRUE(second_lease.release());
  EXPECT_EQ(pool.available(), 2);
  EXPECT_EQ(pool.allocated(), 0);
}

// Verifies a bounded, fully reserved stream pool has no steady-state host allocation after
// initialization: acquire/release only moves preallocated leases through pool bookkeeping.
TEST(CudaStreamPool, ReservedAcquireReleaseHotPathDoesNotAllocate) {
  skip_without_cuda_device();

  CudaStreamPool pool(0, cudaStreamNonBlocking, 0, 2, 2);
  ASSERT_TRUE(pool.initialize_pool());

  // Warm first use outside the counted window so the assertion covers the steady-state path.
  auto warmup = pool.acquire();
  ASSERT_TRUE(warmup) << runtime_error_message(warmup.error());
  ASSERT_TRUE(warmup.value().release());

  bool ok = true;
  std::size_t allocation_count = 0;
  {
    // Key setup: ScopedAllocationCounter wraps 16 full acquire/release cycles; the pool has
    // reserved_size == max_size == 2, so all leases come from preallocated slots.
    const ScopedAllocationCounter allocations;
    for (int iteration = 0; iteration < 16; ++iteration) {
      auto first = pool.acquire();
      if (!first) {
        ok = false;
        break;
      }
      auto first_lease = std::move(first.value());

      auto second = pool.acquire();
      if (!second) {
        ok = false;
        break;
      }
      auto second_lease = std::move(second.value());

      auto release_first = first_lease.release();
      auto release_second = second_lease.release();
      if (!release_first || !release_second) {
        ok = false;
        break;
      }
    }
    allocation_count = allocations.count();
  }

  EXPECT_TRUE(ok);
  // Key assertion: no host allocations during steady-state acquire/release (allocation_count == 0).
  EXPECT_EQ(allocation_count, 0);
  EXPECT_EQ(pool.available(), 2);
  EXPECT_EQ(pool.allocated(), 0);
}

// Verifies a lease returned by destructor is available again without requiring explicit release.
TEST(CudaStreamPool, LeaseDestructorReturnsStreamToPool) {
  skip_without_cuda_device();

  CudaStreamPool pool(0, cudaStreamNonBlocking, 0, 1, 2);
  ASSERT_NO_THROW(pool.initialize());

  {
    auto maybe_lease = pool.acquire();
    ASSERT_TRUE(maybe_lease) << runtime_error_message(maybe_lease.error());
    auto lease = std::move(maybe_lease.value());
    EXPECT_EQ(pool.allocated(), 1);
  }

  EXPECT_EQ(pool.allocated(), 0);
  EXPECT_EQ(pool.available(), 1);
}

// Verifies move-assignment releases the destination's previous stream and transfers ownership of
// the source stream without leaking an allocation.
TEST(CudaStreamPool, LeaseMoveAssignmentReleasesPreviousLease) {
  skip_without_cuda_device();

  CudaStreamPool pool(0, cudaStreamNonBlocking, 0, 2, 2);
  ASSERT_NO_THROW(pool.initialize());

  auto first = pool.acquire();
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  auto first_lease = std::move(first.value());
  auto* const first_handle = first_lease.native_handle();

  auto second = pool.acquire();
  ASSERT_TRUE(second) << runtime_error_message(second.error());
  auto second_lease = std::move(second.value());
  auto* const second_handle = second_lease.native_handle();
  ASSERT_NE(first_handle, second_handle);
  EXPECT_EQ(pool.available(), 0);
  EXPECT_EQ(pool.allocated(), 2);

  // Key assertion: move assignment releases first_lease (available +1) and adopts second_lease;
  // pool.available() rises to 1 and first_lease now holds the second stream's handle.
  first_lease = std::move(second_lease);
  EXPECT_TRUE(static_cast<bool>(first_lease));
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_FALSE(static_cast<bool>(second_lease));
  EXPECT_EQ(first_lease.native_handle(), second_handle);
  EXPECT_EQ(pool.available(), 1);
  EXPECT_EQ(pool.allocated(), 1);

  EXPECT_TRUE(first_lease.release());
  EXPECT_EQ(pool.available(), 2);
  EXPECT_EQ(pool.allocated(), 0);
}

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
// Verifies the pool rejects configuring both a green context and a custom CUDA context. The check
// runs before any CUDA resource is created, so it needs no GPU and uses sentinel handles.
TEST(CudaStreamPool, RejectsBothGreenAndCustomContext) {
  int green_sentinel = 0;
  int context_sentinel = 0;
  CudaStreamPool::Config config;
  config.reserved_size = 1;
  config.max_size = 1;
  config.green_context = reinterpret_cast<CUgreenCtx>(&green_sentinel);
  config.cuda_context = reinterpret_cast<CUcontext>(&context_sentinel);
  CudaStreamPool pool(config);

  auto result = pool.initialize_pool();
  EXPECT_FALSE(result);
  EXPECT_NE(runtime_error_message(result.error()).find("cannot set both"), std::string::npos);
  EXPECT_EQ(pool.available(), 0);
}
#endif

// Verifies out-of-order teardown is safe: if a stream lease survives the pool, later
// release destroys the stream object it owns instead of dereferencing the destroyed pool.
//
// Ownership:
//   pool(inner scope) --acquire--> lease(outer scope)
//   ~pool()            <- pool destroyed while lease is still held
//   lease.release()   -> destroys the CUDA stream directly; pool pointer not dereferenced
TEST(CudaStreamPool, LeaseReleaseAfterPoolDestructionDoesNotTouchPool) {
  skip_without_cuda_device();

  CudaStreamPool::StreamLease lease;
  {
    // Key setup: lease is declared outside pool's scope so it outlives pool destruction.
    CudaStreamPool pool(0, cudaStreamNonBlocking, 0, 1, 1);
    ASSERT_TRUE(pool.initialize_pool());

    auto maybe_lease = pool.acquire();
    ASSERT_TRUE(maybe_lease) << runtime_error_message(maybe_lease.error());
    lease = std::move(maybe_lease.value());
    EXPECT_TRUE(static_cast<bool>(lease));
    EXPECT_EQ(pool.allocated(), 1);
  }

  auto release_result = lease.release();
  EXPECT_TRUE(release_result) << runtime_error_message(release_result.error());
  EXPECT_FALSE(static_cast<bool>(lease));
}

// The custom-context path is realized only with cudaGetDriverEntryPointByVersion (CUDA 12.3+),
// matching HOLOSCAN_CUDA_DRIVER_ENTRY_POINTS_AVAILABLE in the library source.
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12030)
// Verifies Config::cuda_context creates streams in an application-provided CUcontext. The device
// primary context is used as the provided context because a stream created in it reports that same
// context via cuStreamGetCtx (a green-context-mirrored context would report a different underlying
// context).
//
// Context chain:
//   cudaSetDevice(0) + cudaFree(nullptr) -> primary context activated
//   provided_context ---> CudaStreamPool(Config::cuda_context=provided_context)
//   stream_pool --acquire--> lease --cuStreamGetCtx--> stream_ctx == provided_context
TEST(CudaStreamPool, CreatesStreamsInProvidedContext) {
  skip_without_cuda_device();

  // Resolve the driver entry points needed to capture and verify the context. The version argument
  // selects the driver-function ABI, so keep it aligned with these function-pointer signatures.
  // Requesting CUDA 13 for cuStreamGetCtx can resolve cuStreamGetCtx_v2, whose third CUgreenCtx*
  // parameter is incompatible with the legacy two-argument pointer used here.
  CUresult (*ctx_get_current)(CUcontext*) = nullptr;
  CUresult (*stream_get_ctx)(CUstream, CUcontext*) = nullptr;
  cudaDriverEntryPointQueryResult driver_status{};
  constexpr auto kAbi = cuda_execution_detail::kCUDADriverABIVersion12030;
  cudaGetDriverEntryPointByVersion("cuCtxGetCurrent",
                                   reinterpret_cast<void**>(&ctx_get_current),
                                   kAbi,
                                   cudaEnableDefault,
                                   &driver_status);
  cudaGetDriverEntryPointByVersion("cuStreamGetCtx",
                                   reinterpret_cast<void**>(&stream_get_ctx),
                                   kAbi,
                                   cudaEnableDefault,
                                   &driver_status);
  if (ctx_get_current == nullptr || stream_get_ctx == nullptr) {
    GTEST_SKIP() << "Required CUDA driver entry points are unavailable";
  }

  // Initialize and capture the device primary context as the application-provided context.
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);  // forces primary-context creation/activation
  CUcontext provided_context = nullptr;
  ASSERT_EQ(ctx_get_current(&provided_context), CUDA_SUCCESS);
  ASSERT_NE(provided_context, nullptr);

  // Key setup: pass provided_context as Config::cuda_context to route stream creation through it.
  CudaStreamPool::Config stream_config;
  stream_config.reserved_size = 1;
  stream_config.max_size = 1;
  stream_config.cuda_context = provided_context;  // custom-context path, not green_context
  CudaStreamPool stream_pool(stream_config);
  ASSERT_TRUE(stream_pool.initialize_pool());

  auto maybe_lease = stream_pool.acquire();
  ASSERT_TRUE(maybe_lease) << runtime_error_message(maybe_lease.error());
  auto lease = std::move(maybe_lease.value());
  ASSERT_NE(lease.native_handle(), nullptr);

  CUcontext stream_ctx = nullptr;
  ASSERT_EQ(stream_get_ctx(lease.native_handle(), &stream_ctx), CUDA_SUCCESS);
  // Key assertion: the stream's context matches the application-provided context, not a default.
  EXPECT_EQ(stream_ctx, provided_context);
}

// Verifies a provided context whose device does not match Config::device_id is rejected. Requires a
// second device so device_id=1 is valid while the context belongs to device 0.
//
// Cross-device mismatch:
//   device 0 primary context --provided to--> CudaStreamPool(Config::device_id=1)
//   initialize_pool() -> kInvalidArgument ("configured for device")
TEST(CudaStreamPool, RejectsContextOnDifferentDevice) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 2) {
    GTEST_SKIP() << "Requires at least two CUDA devices";
  }

  CUresult (*ctx_get_current)(CUcontext*) = nullptr;
  cudaDriverEntryPointQueryResult driver_status{};
  cudaGetDriverEntryPointByVersion("cuCtxGetCurrent",
                                   reinterpret_cast<void**>(&ctx_get_current),
                                   cuda_execution_detail::kCUDADriverABIVersion12030,
                                   cudaEnableDefault,
                                   &driver_status);
  if (ctx_get_current == nullptr) {
    GTEST_SKIP() << "cuCtxGetCurrent is unavailable";
  }

  // Key setup: capture device 0's primary context, then configure the pool for device 1.
  // Capture device 0's primary context.
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  CUcontext device0_context = nullptr;
  ASSERT_EQ(ctx_get_current(&device0_context), CUDA_SUCCESS);
  ASSERT_NE(device0_context, nullptr);

  // Configure the pool for device 1 but hand it device 0's context.
  CudaStreamPool::Config config;
  config.device_id = 1;
  config.reserved_size = 1;
  config.max_size = 1;
  config.cuda_context = device0_context;
  CudaStreamPool pool(config);

  auto result = pool.initialize_pool();
  EXPECT_FALSE(result);
  // Key assertion: the pool detects the device mismatch before creating any CUDA resource.
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
  EXPECT_NE(runtime_error_message(result.error()).find("configured for device"), std::string::npos);
  (void)cudaGetLastError();
}
#else
// Keep a visible record when the toolkit predates the driver-entry-point API the custom-context
// path relies on.
TEST(CudaStreamPool, CreatesStreamsInProvidedContext) {
  GTEST_SKIP() << "Config::cuda_context requires CUDA 12.3 or newer";
}
#endif

}  // namespace holoscan
