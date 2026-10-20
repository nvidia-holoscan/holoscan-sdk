/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <holoscan/cuda_execution/cuda_green_context.hpp>
#include <holoscan/cuda_execution/cuda_green_context_pool.hpp>
#include <holoscan/cuda_execution/cuda_stream_pool.hpp>

#include "../common/cuda_test_utils.hpp"

#include "cuda_execution/cuda_driver_entry_points.hpp"

#if HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE

namespace holoscan {
using ::holoscan::test::invalid_cuda_device_id;
using ::holoscan::test::skip_without_cuda_device;

namespace {

class CaptureResources final {
 public:
  explicit CaptureResources(cudaStream_t capture_stream) : capture_stream_(capture_stream) {}
  CaptureResources(const CaptureResources&) = delete;
  CaptureResources& operator=(const CaptureResources&) = delete;
  ~CaptureResources() { static_cast<void>(cleanup()); }

  [[nodiscard]] cudaError_t cleanup() noexcept {
    cudaError_t first = cudaSuccess;
    const auto record = [&first](cudaError_t result) {
      if (first == cudaSuccess && result != cudaSuccess) {
        first = result;
      }
    };
    if (capture_active) {
      cudaGraph_t abandoned{};
      record(cudaStreamEndCapture(capture_stream_, &abandoned));
      capture_active = false;
      if (abandoned != nullptr) {
        record(cudaGraphDestroy(abandoned));
      }
    }
    if (executable != nullptr) {
      record(cudaGraphExecDestroy(std::exchange(executable, nullptr)));
    }
    if (graph != nullptr) {
      record(cudaGraphDestroy(std::exchange(graph, nullptr)));
    }
    if (joined != nullptr) {
      record(cudaEventDestroy(std::exchange(joined, nullptr)));
    }
    if (readiness != nullptr) {
      record(cudaEventDestroy(std::exchange(readiness, nullptr)));
    }
    return first;
  }

  cudaEvent_t readiness{};
  cudaEvent_t joined{};
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  bool capture_active{};

 private:
  cudaStream_t capture_stream_{};
};

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
// REQ: MEM-GREENCTX-001
TEST(CudaGreenContextPool, AccessorsFailBeforeInitialize) {
  const CudaGreenContextPool pool;

  EXPECT_FALSE(pool.green_context(0));
  EXPECT_FALSE(pool.cuda_context(0));
  EXPECT_FALSE(pool.default_context_index());
  EXPECT_FALSE(pool.device_total_sms());
  EXPECT_EQ(pool.num_partitions(), 0u);
}

// Verifies an invalid device id is surfaced as an Error value instead of throwing.
// REQ: MEM-GREENCTX-002
// REQ: MEM-GREENCTX-010
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
// REQ: MEM-GREENCTX-003
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

// Verifies an explicit multi-partition split matches the requested SM counts and that a legal
// remaining structured partition is appended when driver granularity permits it.
//
// Partition layout (sms_per_partition = {4, 2, 4}, structured capacity >= 10):
//   [0: 4 SMs] [1: 2 SMs] [2: 4 SMs] [3: qualified residual SMs (default, if any)]
// REQ: MEM-GREENCTX-004
// REQ: MEM-GREENCTX-012
TEST(CudaGreenContextPool, MultiPartitionSplitMatchesRequest) {
  skip_without_cuda_device();

  // Key setup: request 3 partitions with explicit SM counts; the pool may append a structured
  // residual partition when the leftover satisfies the driver-reported quantum.
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
  auto minimum = pool.min_sm_partition_size();
  ASSERT_TRUE(minimum) << runtime_error_message(minimum.error());
  auto alignment = pool.sm_coscheduled_alignment();
  ASSERT_TRUE(alignment) << runtime_error_message(alignment.error());
  auto max_contexts = pool.max_structured_context_count();
  ASSERT_TRUE(max_contexts) << runtime_error_message(max_contexts.error());
  const uint32_t requested = 4 + 2 + 4;
  const std::uint64_t structured_capacity =
      static_cast<std::uint64_t>(*max_contexts) * std::lcm(*minimum, *alignment);
  ASSERT_LE(structured_capacity, *total_sms);

  auto default_index = pool.default_context_index();
  ASSERT_TRUE(default_index) << runtime_error_message(default_index.error());
  // Key assertion: if qualified structured capacity remains beyond the exact requests, a fourth
  // partition is appended as default; driver-unavailable tail SMs are never included.
  if (structured_capacity > requested) {
    // A structured residual was appended; it is the default and holds the eligible leftover SMs.
    ASSERT_EQ(pool.num_partitions(), 4u);
    EXPECT_EQ(default_index.value(), 3u);
    auto remaining = pool.partition_sms(default_index.value());
    ASSERT_TRUE(remaining) << runtime_error_message(remaining.error());
    EXPECT_EQ(remaining.value(), structured_capacity - requested);
  } else {
    ASSERT_EQ(pool.num_partitions(), 3u);
    EXPECT_EQ(default_index.value(), 2u);
  }

  auto default_context = pool.default_context();
  ASSERT_TRUE(default_context) << runtime_error_message(default_context.error());
  EXPECT_NE(default_context.value(), nullptr);
}

// Verifies initialize_pool() is idempotent.
// REQ: MEM-GREENCTX-005
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
// REQ: MEM-GREENCTX-006
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
// REQ: MEM-GREENCTX-007
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
// REQ: MEM-GREENCTX-008
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
  // The version argument selects the driver-function ABI, so keep it aligned with this
  // function-pointer signature introduced in CUDA 12.4.
  CUresult (*stream_get_green_ctx)(CUstream, CUgreenCtx*) = nullptr;
  cudaDriverEntryPointQueryResult driver_status{};
  cudaGetDriverEntryPointByVersion("cuStreamGetGreenCtx",
                                   reinterpret_cast<void**>(&stream_get_green_ctx),
                                   cuda_execution_detail::kCUDADriverABIVersion12040,
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

// Ordinary stream-scoped event waits remain legal CUDA Graph capture operations across the
// stream/context relationships used by Runtime Tensor readiness. This intentionally exercises no
// context-scoped cuGreenCtxRecordEvent/cuGreenCtxWaitEvent operation.
// REQ: TENSOR-STORAGE-014
TEST(CudaGreenContext, StreamEventWaitsRemainCaptureCompatibleAcrossRuntimeBoundaries) {
  skip_without_cuda_device();

  // Key setup: capture producer-to-consumer event joins across same, dedicated, residual, and
  // transfer-service Green Context stream relationships.
  CudaGreenContextPool::Config config;
  config.num_partitions = 2U;
  config.sms_per_partition = {4, 4};
  CudaGreenContextPool pool(config);
  INITIALIZE_OR_SKIP(pool);
  if (pool.num_partitions() < 3U) {
    GTEST_SKIP() << "device does not expose a residual Green Context partition";
  }
  auto residual_index = pool.default_context_index();
  ASSERT_TRUE(residual_index) << runtime_error_message(residual_index.error());
  if (*residual_index == 0U || *residual_index == 1U) {
    GTEST_SKIP() << "split did not append an independent residual Green Context";
  }
  auto dedicated_a = pool.green_context(0U);
  auto dedicated_b = pool.green_context(1U);
  auto residual = pool.green_context(*residual_index);
  ASSERT_TRUE(dedicated_a) << runtime_error_message(dedicated_a.error());
  ASSERT_TRUE(dedicated_b) << runtime_error_message(dedicated_b.error());
  ASSERT_TRUE(residual) << runtime_error_message(residual.error());

  CudaStreamPool::Config dedicated_a_config;
  dedicated_a_config.reserved_size = 2U;
  dedicated_a_config.max_size = 2U;
  dedicated_a_config.green_context = *dedicated_a;
  CudaStreamPool dedicated_a_streams(dedicated_a_config);
  ASSERT_TRUE(dedicated_a_streams.initialize_pool());
  CudaStreamPool::Config dedicated_b_config;
  dedicated_b_config.reserved_size = 1U;
  dedicated_b_config.max_size = 1U;
  dedicated_b_config.green_context = *dedicated_b;
  CudaStreamPool dedicated_b_streams(dedicated_b_config);
  ASSERT_TRUE(dedicated_b_streams.initialize_pool());
  CudaStreamPool::Config residual_config;
  residual_config.reserved_size = 2U;
  residual_config.max_size = 2U;
  residual_config.green_context = *residual;
  CudaStreamPool residual_streams(residual_config);
  ASSERT_TRUE(residual_streams.initialize_pool());

  auto dedicated_a_producer = dedicated_a_streams.acquire();
  auto dedicated_a_consumer = dedicated_a_streams.acquire();
  auto dedicated_b_consumer = dedicated_b_streams.acquire();
  auto residual_consumer = residual_streams.acquire();
  auto residual_service = residual_streams.acquire();
  ASSERT_TRUE(dedicated_a_producer);
  ASSERT_TRUE(dedicated_a_consumer);
  ASSERT_TRUE(dedicated_b_consumer);
  ASSERT_TRUE(residual_consumer);
  ASSERT_TRUE(residual_service);

  const std::array<std::pair<std::string_view, cudaStream_t>, 4U> consumers{{
      {"same-partition", dedicated_a_consumer->native_handle()},
      {"dedicated-to-dedicated", dedicated_b_consumer->native_handle()},
      {"dedicated-to-residual", residual_consumer->native_handle()},
      {"transfer-service", residual_service->native_handle()},
  }};
  for (const auto& [boundary, consumer] : consumers) {
    CaptureResources resources(dedicated_a_producer->native_handle());
    ASSERT_EQ(cudaEventCreateWithFlags(&resources.readiness, cudaEventDisableTiming), cudaSuccess)
        << boundary;
    ASSERT_EQ(cudaEventCreateWithFlags(&resources.joined, cudaEventDisableTiming), cudaSuccess)
        << boundary;
    ASSERT_EQ(cudaStreamBeginCapture(dedicated_a_producer->native_handle(),
                                     cudaStreamCaptureModeThreadLocal),
              cudaSuccess)
        << boundary;
    resources.capture_active = true;
    ASSERT_EQ(cudaEventRecord(resources.readiness, dedicated_a_producer->native_handle()),
              cudaSuccess)
        << boundary;
    ASSERT_EQ(cudaStreamWaitEvent(consumer, resources.readiness), cudaSuccess) << boundary;
    ASSERT_EQ(cudaEventRecord(resources.joined, consumer), cudaSuccess) << boundary;
    ASSERT_EQ(cudaStreamWaitEvent(dedicated_a_producer->native_handle(), resources.joined),
              cudaSuccess)
        << boundary;
    ASSERT_EQ(cudaStreamEndCapture(dedicated_a_producer->native_handle(), &resources.graph),
              cudaSuccess)
        << boundary;
    resources.capture_active = false;
    ASSERT_NE(resources.graph, nullptr) << boundary;
    ASSERT_EQ(cudaGraphInstantiate(&resources.executable, resources.graph, 0U), cudaSuccess)
        << boundary;
    ASSERT_EQ(cudaGraphLaunch(resources.executable, consumer), cudaSuccess) << boundary;
    ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess) << boundary;
    // Key assertion: every boundary instantiates, launches, and synchronizes as a CUDA Graph.
    EXPECT_EQ(resources.cleanup(), cudaSuccess) << boundary;
  }
}

}  // namespace holoscan

#else  // HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE

// Keep a visible record when the toolkit is too old for the green-context driver APIs.
TEST(CudaGreenContext, RequiresCuda12_4) {
  GTEST_SKIP() << "CUDA Green Context execution requires CUDA 12.5 or newer";
}

#endif  // HOLOSCAN_CUDA_GREEN_CONTEXT_AVAILABLE
