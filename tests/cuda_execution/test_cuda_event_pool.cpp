/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cuda/stream>

#include <string>
#include <utility>

#include "../common/allocation_counter.hpp"
#include "../common/cuda_test_utils.hpp"

#include <holoscan/cuda_execution/cuda_event_pool.hpp>

#include "cuda_execution/cuda_driver_entry_points.hpp"

namespace holoscan {
using ::holoscan::test::create_non_blocking_cuda_stream;
using ::holoscan::test::invalid_cuda_device_id;
using ::holoscan::test::ScopedAllocationCounter;
using ::holoscan::test::skip_without_cuda_device;
using ::holoscan::test::UniqueCudaStream;

// Verifies acquire() fails cleanly before the event pool is initialized.
TEST(CudaEventPool, AcquireFailsBeforeInitialize) {
  CudaEventPool pool(0, cuda::event_flags::none, 1, 1);

  auto maybe_lease = pool.acquire();
  EXPECT_FALSE(maybe_lease);
}

// Verifies expected-returning operations fail cleanly on a default lease instead of dereferencing
// an absent CUDA event. This keeps moved-from/default lease handling in the RuntimeError path.
TEST(CudaEventPool, DefaultLeaseOperationsReportInvalidLease) {
  const CudaEventPool::EventLease lease;

  auto synchronize = lease.synchronize();
  EXPECT_FALSE(synchronize);
  EXPECT_NE(runtime_error_message(synchronize.error()).find("lease is invalid"), std::string::npos);

  auto done = lease.is_done();
  EXPECT_FALSE(done);
  EXPECT_NE(runtime_error_message(done.error()).find("lease is invalid"), std::string::npos);
}

// Verifies callers that do not want Component::initialize() exceptions can initialize the event
// pool through an expected-returning API and repeat initialization harmlessly.
TEST(CudaEventPool, InitializePoolReturnsExpectedAndIsIdempotent) {
  skip_without_cuda_device();

  CudaEventPool pool(0, cuda::event_flags::none, 1, 1);
  auto first = pool.initialize_pool();
  ASSERT_TRUE(first) << runtime_error_message(first.error());
  EXPECT_EQ(pool.available(), 1);
  auto second = pool.initialize_pool();
  EXPECT_TRUE(second) << runtime_error_message(second.error());
  EXPECT_EQ(pool.available(), 1);
}

// Verifies explicit deinitialize() invalidates outstanding leases without letting those old leases
// return events to a later reinitialized pool.
//
// Lifecycle:
//   pool --acquire--> old_lease
//   pool --deinitialize--> (generation bumped; old_lease now stale)
//   pool --reinitialize--> pool --acquire--> new_lease
//   old_lease.release() -> no-op on allocated count (stale generation)
//   new_lease.release() -> decrements allocated to zero
TEST(CudaEventPool, DeinitializeInvalidatesOutstandingLeasesAndAllowsReinitialize) {
  skip_without_cuda_device();

  CudaEventPool pool(0, cuda::event_flags::none, 1, 1);
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

// Verifies CCCL event construction failures are surfaced as RuntimeError values instead of
// requiring callers to catch exceptions from the event pool initialization path.
TEST(CudaEventPool, InitializePoolReportsInvalidDeviceWithoutThrowing) {
  skip_without_cuda_device();

  CudaEventPool pool(invalid_cuda_device_id(), cuda::event_flags::none, 1, 1);
  auto result = pool.initialize_pool();
  ASSERT_FALSE(result);
  EXPECT_NE(runtime_error_message(result.error()).find("Invalid CUDA event pool device id"),
            std::string::npos);
  (void)cudaGetLastError();
}

// Verifies capacity configuration is explicit: max_size can be zero for unbounded growth, but a
// nonzero max smaller than the reserved count is rejected before CUDA events are created.
TEST(CudaEventPool, RejectsMaxSizeSmallerThanReservedBeforeCudaSetup) {
  CudaEventPool pool(0, cuda::event_flags::none, 2, 1);

  auto result = pool.initialize_pool();
  EXPECT_FALSE(result);
  EXPECT_NE(runtime_error_message(result.error()).find("max_size must be zero or at least"),
            std::string::npos);
  EXPECT_EQ(pool.available(), 0);
}

// Verifies the pool returns reserved events, enforces max_size, and makes an explicitly released
// lease available for reuse.
TEST(CudaEventPool, AcquireReleaseAndCapacity) {
  skip_without_cuda_device();

  CudaEventPool pool(0, cuda::event_flags::none, 1, 1);
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
TEST(CudaEventPool, GrowsBeyondReservedEventsUntilCapacity) {
  skip_without_cuda_device();

  CudaEventPool pool(0, cuda::event_flags::none, 1, 2);
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

// Verifies a bounded, fully reserved event pool has no steady-state host allocation after
// initialization: acquire/release only moves preallocated leases through pool bookkeeping.
TEST(CudaEventPool, ReservedAcquireReleaseHotPathDoesNotAllocate) {
  skip_without_cuda_device();

  CudaEventPool pool(0, cuda::event_flags::none, 2, 2);
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
TEST(CudaEventPool, LeaseDestructorReturnsEventToPool) {
  skip_without_cuda_device();

  CudaEventPool pool(0, cuda::event_flags::none, 1, 2);
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

// Verifies move-assignment releases the destination's previous event and transfers ownership of
// the source event without leaking an allocation.
TEST(CudaEventPool, LeaseMoveAssignmentReleasesPreviousLease) {
  skip_without_cuda_device();

  CudaEventPool pool(0, cuda::event_flags::none, 2, 2);
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
  // pool.available() rises to 1 and first_lease now holds the second event's handle.
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

// Verifies out-of-order teardown is safe: if an event lease survives the pool, later
// release destroys the event object it owns instead of dereferencing the destroyed pool.
//
// Ownership:
//   pool(inner scope) --acquire--> lease(outer scope)
//   ~pool()            <- pool destroyed while lease is still held
//   lease.release()   -> destroys the CUDA event directly; pool pointer not dereferenced
TEST(CudaEventPool, LeaseReleaseAfterPoolDestructionDoesNotTouchPool) {
  skip_without_cuda_device();

  CudaEventPool::EventLease lease;
  {
    // Key setup: lease is declared outside pool's scope so it outlives pool destruction.
    CudaEventPool pool(0, cuda::event_flags::none, 1, 1);
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

// Verifies the lease exposes the intended CCCL event_ref operations: record on one stream, make
// another stream wait, and query completion through the same borrowed event view.
//
// Stream coordination:
//   producer_stream --record(event)--> event_lease
//   event_lease --wait(event)--> consumer_stream
//   consumer_stream.synchronize() -> event is done (is_done() returns true)
TEST(CudaEventPool, LeaseRecordsAndWaitsThroughEventRef) {
  skip_without_cuda_device();

  CudaEventPool pool(0, cuda::event_flags::none, 1, 1);
  ASSERT_NO_THROW(pool.initialize());

  auto maybe_lease = pool.acquire();
  ASSERT_TRUE(maybe_lease) << runtime_error_message(maybe_lease.error());
  auto lease = std::move(maybe_lease.value());
  ASSERT_NE(lease.native_handle(), nullptr);

  // Key setup: two independent non-blocking streams (producer/consumer) and one event lease.
  UniqueCudaStream producer_stream;
  ASSERT_EQ(create_non_blocking_cuda_stream(producer_stream), cudaSuccess);
  UniqueCudaStream consumer_stream;
  ASSERT_EQ(create_non_blocking_cuda_stream(consumer_stream), cudaSuccess);

  auto record_result = lease.record(::cuda::stream_ref(producer_stream.get()));
  ASSERT_TRUE(record_result) << runtime_error_message(record_result.error());
  auto wait_result = lease.wait(::cuda::stream_ref(consumer_stream.get()));
  ASSERT_TRUE(wait_result) << runtime_error_message(wait_result.error());
  ASSERT_EQ(cudaStreamSynchronize(consumer_stream.get()), cudaSuccess);

  // Key assertion: after consumer_stream synchronizes, the event must report done via is_done().
  auto done = lease.is_done();
  ASSERT_TRUE(done) << runtime_error_message(done.error());
  EXPECT_TRUE(done.value());
}

// The custom-context path is realized only with cudaGetDriverEntryPointByVersion (CUDA 12.3+),
// matching HOLOSCAN_CUDA_DRIVER_ENTRY_POINTS_AVAILABLE in the library source.
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12030)
// Verifies Config::cuda_context creates events in an application-provided CUcontext. There is no
// driver API to query an event's context (unlike cuStreamGetCtx for streams), so the event is
// verified functionally: it records on a stream and reports completion. The device primary context
// is used as the provided context.
//
// Context chain:
//   cudaSetDevice(0) + cudaFree(nullptr) -> primary context activated
//   provided_context ---> CudaEventPool(Config::cuda_context=provided_context)
//   event_pool --acquire--> lease -> functionally verified: record + synchronize + is_done()
TEST(CudaEventPool, CreatesEventsInProvidedContext) {
  skip_without_cuda_device();

  // Resolve cuCtxGetCurrent to capture the device primary context. The version argument selects
  // the driver-function ABI, so keep it aligned with this function-pointer signature.
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

  // Initialize and capture the device primary context as the application-provided context.
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);  // forces primary-context creation/activation
  CUcontext provided_context = nullptr;
  ASSERT_EQ(ctx_get_current(&provided_context), CUDA_SUCCESS);
  ASSERT_NE(provided_context, nullptr);

  CudaEventPool::Config config;
  config.reserved_size = 1;
  config.max_size = 1;
  config.cuda_context = provided_context;  // custom-context path
  CudaEventPool pool(config);
  ASSERT_TRUE(pool.initialize_pool());

  auto maybe_lease = pool.acquire();
  ASSERT_TRUE(maybe_lease) << runtime_error_message(maybe_lease.error());
  auto lease = std::move(maybe_lease.value());
  ASSERT_NE(lease.native_handle(), nullptr);

  // The event created in the provided context must be usable: record on a stream and report done.
  UniqueCudaStream stream;
  ASSERT_EQ(create_non_blocking_cuda_stream(stream), cudaSuccess);
  auto record_result = lease.record(::cuda::stream_ref(stream.get()));
  ASSERT_TRUE(record_result) << runtime_error_message(record_result.error());
  auto synchronize = lease.synchronize();
  ASSERT_TRUE(synchronize) << runtime_error_message(synchronize.error());
  // Key assertion: the event created in the provided context is functional; is_done() returns true
  // after synchronize(), confirming the event was usable within that context.
  auto done = lease.is_done();
  ASSERT_TRUE(done) << runtime_error_message(done.error());
  EXPECT_TRUE(done.value());
}
#else
// Keep a visible record when the toolkit predates the driver-entry-point API the custom-context
// path relies on.
TEST(CudaEventPool, CreatesEventsInProvidedContext) {
  GTEST_SKIP() << "Config::cuda_context requires CUDA 12.3 or newer";
}
#endif

}  // namespace holoscan
