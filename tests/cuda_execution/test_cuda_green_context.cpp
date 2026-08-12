/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <string>
#include <utility>

#include "../common/cuda_test_utils.hpp"

#include <holoscan/cuda_execution/cuda_green_context.hpp>
#include <holoscan/cuda_execution/cuda_green_context_pool.hpp>
#include <holoscan/cuda_execution/cuda_stream_pool.hpp>

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE

namespace holoscan {
using ::holoscan::test::invalid_cuda_device_id;
using ::holoscan::test::skip_without_cuda_device;

namespace {

// Green-context partitioning depends on GPU architecture (Volta+) and driver support, and requires
// the device SM count to be divisible by min_sm_count. When the platform cannot satisfy the
// requested split, skip the test rather than failing.
// Representative tests: MultiPartitionSplitMatchesRequest, StreamPoolCreatesStreamsInGreenContext.
#define INITIALIZE_OR_SKIP(pool)                                      \
  do {                                                                \
    auto init_result = (pool).initialize_pool();                      \
    if (!init_result) {                                               \
      GTEST_SKIP() << "Green contexts unavailable on this platform: " \
                   << runtime_error_message(init_result.error());     \
    }                                                                 \
  } while (0)

}  // namespace

// Verifies accessors fail cleanly before the pool is initialized rather than returning garbage
// handles.
TEST(CudaGreenContextPool, AccessorsFailBeforeInitialize) {
  const CudaGreenContextPool pool;

  EXPECT_FALSE(pool.green_context(0));
  EXPECT_FALSE(pool.cuda_context(0));
  EXPECT_FALSE(pool.default_context_index());
  EXPECT_FALSE(pool.device_total_sms());
  EXPECT_EQ(pool.num_partitions(), 0u);
}

// Verifies an invalid device id is surfaced as a RuntimeError instead of throwing.
TEST(CudaGreenContextPool, InitializeReportsInvalidDevice) {
  skip_without_cuda_device();

  CudaGreenContextPool::Config config;
  config.device_id = invalid_cuda_device_id();
  CudaGreenContextPool pool(config);

  auto result = pool.initialize_pool();
  EXPECT_FALSE(result);
  (void)cudaGetLastError();
}

// Verifies the default configuration produces a single partition spanning the whole device.
//
// Partition layout (num_partitions == 0 defaults to 1):
//   [0: all device SMs] <- sole partition; default_context_index == 0
TEST(CudaGreenContextPool, DefaultConfigSinglePartitionSpansDevice) {
  skip_without_cuda_device();

  CudaGreenContextPool pool;  // num_partitions == 0 => one partition over the entire device
  INITIALIZE_OR_SKIP(pool);

  ASSERT_EQ(pool.num_partitions(), 1u);

  auto total_sms = pool.device_total_sms();
  ASSERT_TRUE(total_sms) << runtime_error_message(total_sms.error());

  auto partition_sms = pool.partition_sms(0);
  ASSERT_TRUE(partition_sms) << runtime_error_message(partition_sms.error());
  EXPECT_EQ(partition_sms.value(), total_sms.value());

  auto default_index = pool.default_context_index();
  ASSERT_TRUE(default_index) << runtime_error_message(default_index.error());
  EXPECT_EQ(default_index.value(), 0u);

  auto green_context = pool.green_context(0);
  ASSERT_TRUE(green_context) << runtime_error_message(green_context.error());
  EXPECT_NE(green_context.value(), nullptr);

  auto cuda_context = pool.cuda_context(0);
  ASSERT_TRUE(cuda_context) << runtime_error_message(cuda_context.error());
  EXPECT_NE(cuda_context.value(), nullptr);

  // Out-of-range partition is rejected.
  EXPECT_FALSE(pool.green_context(1));
}

// Verifies an explicit multi-partition split matches the requested SM counts and that a remaining
// partition is appended to cover the rest of the device (mirrors the GXF unit test).
//
// Partition layout (sms_per_partition = {4, 2, 4}, total_sms >= 10):
//   [0: 4 SMs] [1: 2 SMs] [2: 4 SMs] [3: remaining SMs (default, when total_sms > 10)]
//   total_sms == 10: no remainder; 3 partitions; default_context_index == 2
TEST(CudaGreenContextPool, MultiPartitionSplitMatchesRequest) {
  skip_without_cuda_device();

  // Key setup: request 3 partitions with explicit SM counts; the pool may append a remainder
  // partition when total_sms > sum(sms_per_partition).
  CudaGreenContextPool::Config config;
  config.num_partitions = 3;
  config.sms_per_partition = {4, 2, 4};
  CudaGreenContextPool pool(config);
  INITIALIZE_OR_SKIP(pool);

  for (uint32_t i = 0; i < 3; ++i) {
    auto partition_sms = pool.partition_sms(i);
    ASSERT_TRUE(partition_sms) << runtime_error_message(partition_sms.error());
    EXPECT_EQ(partition_sms.value(), static_cast<uint32_t>(config.sms_per_partition[i]));
  }

  auto total_sms = pool.device_total_sms();
  ASSERT_TRUE(total_sms) << runtime_error_message(total_sms.error());
  const uint32_t requested = 4 + 2 + 4;

  auto default_index = pool.default_context_index();
  ASSERT_TRUE(default_index) << runtime_error_message(default_index.error());
  // Key assertion: if total_sms > requested, a 4th remainder partition is appended as default;
  // otherwise exactly 3 partitions exist and the last is the default.
  if (total_sms.value() > requested) {
    // A remaining partition was appended; it is the default and holds the leftover SMs.
    ASSERT_EQ(pool.num_partitions(), 4u);
    EXPECT_EQ(default_index.value(), 3u);
    auto remaining = pool.partition_sms(default_index.value());
    ASSERT_TRUE(remaining) << runtime_error_message(remaining.error());
    EXPECT_EQ(remaining.value(), total_sms.value() - requested);
  } else {
    ASSERT_EQ(pool.num_partitions(), 3u);
    EXPECT_EQ(default_index.value(), 2u);
  }

  auto default_context = pool.default_context();
  ASSERT_TRUE(default_context) << runtime_error_message(default_context.error());
  EXPECT_NE(default_context.value(), nullptr);
}

// Verifies initialize_pool() is idempotent.
TEST(CudaGreenContextPool, InitializeIsIdempotent) {
  skip_without_cuda_device();

  CudaGreenContextPool pool;
  INITIALIZE_OR_SKIP(pool);
  const size_t partitions = pool.num_partitions();

  auto second = pool.initialize_pool();
  EXPECT_TRUE(second) << runtime_error_message(second.error());
  EXPECT_EQ(pool.num_partitions(), partitions);
}

// Verifies CudaGreenContext binds to an explicit partition index and exposes its borrowed handles.
//
// Partition pool (3 partitions: 4,2,4 SMs), default_context=0:
//   [0: 4 SMs (default)] [1: 2 SMs (target)] [2: 4 SMs]
//   CudaGreenContext::create(pool, 1) -> index()==1, handle matches pool.green_context(1)
TEST(CudaGreenContext, BindsToExplicitIndex) {
  skip_without_cuda_device();

  CudaGreenContextPool::Config config;
  config.num_partitions = 3;
  config.sms_per_partition = {4, 2, 4};
  config.default_context = 0;
  CudaGreenContextPool pool(config);
  INITIALIZE_OR_SKIP(pool);

  auto green_context = CudaGreenContext::create(pool, 1u, "test_green_context");
  ASSERT_TRUE(green_context) << runtime_error_message(green_context.error());
  EXPECT_EQ(green_context->index(), 1u);
  EXPECT_EQ(green_context->nvtx_identifier(), "test_green_context");

  auto handle = green_context->green_context();
  ASSERT_TRUE(handle) << runtime_error_message(handle.error());
  auto pool_handle = pool.green_context(1);
  ASSERT_TRUE(pool_handle) << runtime_error_message(pool_handle.error());
  // Key assertion: the CudaGreenContext's handle is the same pointer as the pool's partition 1
  // handle, confirming the binding is a borrow (not a copy or separate allocation).
  EXPECT_EQ(handle.value(), pool_handle.value());

  EXPECT_TRUE(green_context->cuda_context());
}

// Verifies a CudaGreenContext created without an explicit index resolves to the pool default.
TEST(CudaGreenContext, DefaultsToPoolDefaultContext) {
  skip_without_cuda_device();

  CudaGreenContextPool::Config config;
  config.num_partitions = 3;
  config.sms_per_partition = {4, 2, 4};
  config.default_context = 0;
  CudaGreenContextPool pool(config);
  INITIALIZE_OR_SKIP(pool);

  auto green_context = CudaGreenContext::create(pool);
  ASSERT_TRUE(green_context) << runtime_error_message(green_context.error());

  auto default_index = pool.default_context_index();
  ASSERT_TRUE(default_index) << runtime_error_message(default_index.error());
  EXPECT_EQ(green_context->index(), default_index.value());
}

// Verifies a default-constructed handle reports its unbound state instead of dereferencing null.
TEST(CudaGreenContext, DefaultHandleIsUnbound) {
  const CudaGreenContext green_context;
  EXPECT_FALSE(static_cast<bool>(green_context));
  EXPECT_FALSE(green_context.green_context());
  EXPECT_FALSE(green_context.cuda_context());
}

// Verifies CudaStreamPool creates its streams inside a chosen green context, confirmed by querying
// each stream's green context with cuStreamGetGreenCtx.
//
// Resource chain:
//   CudaGreenContextPool --default_context()--> CUgreenCtx
//   CudaStreamPool(Config::green_context=CUgreenCtx) --acquire()--> stream lease
//   cuStreamGetGreenCtx(stream) == CUgreenCtx  (verified when driver API is available)
TEST(CudaGreenContext, StreamPoolCreatesStreamsInGreenContext) {
  skip_without_cuda_device();

  CudaGreenContextPool pool;  // single partition spanning the whole device
  INITIALIZE_OR_SKIP(pool);

  auto green = pool.default_context();
  ASSERT_TRUE(green) << runtime_error_message(green.error());

  // Key setup: pass the green context to the stream pool so all streams are created within it.
  CudaStreamPool::Config stream_config;
  stream_config.reserved_size = 2;
  stream_config.max_size = 2;
  stream_config.green_context = green.value();
  CudaStreamPool stream_pool(stream_config);
  ASSERT_TRUE(stream_pool.initialize_pool());

  // Resolve cuStreamGetGreenCtx so we can confirm the stream's green context, when available.
  CUresult (*stream_get_green_ctx)(CUstream, CUgreenCtx*) = nullptr;
  int version = 0;
  cudaRuntimeGetVersion(&version);
  cudaDriverEntryPointQueryResult driver_status{};
  cudaGetDriverEntryPointByVersion("cuStreamGetGreenCtx",
                                   reinterpret_cast<void**>(&stream_get_green_ctx),
                                   version,
                                   cudaEnableDefault,
                                   &driver_status);

  auto maybe_lease = stream_pool.acquire();
  ASSERT_TRUE(maybe_lease) << runtime_error_message(maybe_lease.error());
  auto lease = std::move(maybe_lease.value());
  ASSERT_NE(lease.native_handle(), nullptr);

  if (stream_get_green_ctx != nullptr) {
    CUgreenCtx stream_green_ctx = nullptr;
    ASSERT_EQ(stream_get_green_ctx(lease.native_handle(), &stream_green_ctx), CUDA_SUCCESS);
    // Key assertion: the stream's green context matches the one passed to the pool config.
    EXPECT_EQ(stream_green_ctx, green.value());
  }
}

}  // namespace holoscan

#else  // HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE

// Keep a visible record when the toolkit is too old for the green-context driver APIs.
TEST(CudaGreenContext, RequiresCuda12_4) {
  GTEST_SKIP() << "CUDA green context APIs require CUDA 12.4 or newer";
}

#endif  // HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
