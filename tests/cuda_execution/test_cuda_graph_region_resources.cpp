/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "../common/allocation_counter.hpp"
#include "../common/cuda_test_utils.hpp"
#include "compiler/execution_plan_state.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/runtime_cuda_execution_diagnostic.hpp"
#include "runtime/runtime_cuda_execution_resources.hpp"
#include "runtime/runtime_cuda_graph_region_resources.hpp"

namespace holoscan::detail {
namespace {

class CUDAGraphRegionNativeTest : public testing::Test {
 protected:
  void SetUp() override {
    if (cudaGetDeviceCount(&device_count_) != cudaSuccess || device_count_ == 0) {
      GTEST_SKIP() << "No CUDA device is available";
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&device_data_, sizeof(std::uint32_t)), cudaSuccess);
  }

  void TearDown() override {
    static_cast<void>(cudaSetDevice(0));
    if (stream_ != nullptr) {
      static_cast<void>(cudaStreamSynchronize(stream_));
    }
    if (device_data_ != nullptr) {
      static_cast<void>(cudaFree(device_data_));
    }
    if (stream_ != nullptr) {
      static_cast<void>(cudaStreamDestroy(stream_));
    }
  }

  [[nodiscard]] ExecutionPlanState plan() const {
    ExecutionPlanState result;
    result.cuda_graph_capture_plans.push_back(CUDAGraphCapturePlanRecord{
        .graph = GraphId{1U},
        .region = RegionId{1U},
        .members = {{.vertex = VertexId{1U}}},
    });
    result.cuda_graph_execution_regions.push_back(CUDAGraphExecutionRegionPlanRecord{
        .graph = GraphId{1U},
        .region = RegionId{1U},
        .fused_operator = VertexId{1U},
        .partition = ExecutionPartitionId{1U},
        .execution_lane = 0U,
        .process_group = ProcessGroupId{1U},
        .device = DeviceId{0},
        .stream_id = "partition-1-device-0",
    });
    result.cuda_compute_streams.push_back(CudaComputeStreamPlanRecord{
        .process_group = ProcessGroupId{1U},
        .device = DeviceId{0},
        .partition = ExecutionPartitionId{1U},
        .stream_id = "partition-1-device-0",
    });
    return result;
  }

  [[nodiscard]] std::vector<RuntimeOperatorInstance> operators() const {
    std::vector<RuntimeOperatorInstance> result(1U);
    result.front().vertex = VertexId{1U};
    result.front().partition = ExecutionPartitionId{1U};
    result.front().execution_lane = 0U;
    result.front().compute_stream = stream_;
    result.front().compute_device = DeviceId{0};
    return result;
  }

  static expected<void, Error> capture_memset(void* state, RuntimeOperatorInstance& instance,
                                              DeviceId device, cudaStream_t stream) noexcept {
    auto* const device_data = static_cast<std::uint32_t*>(state);
    // Capture must not reuse the operator's designated steady-state stream.
    if (device_data == nullptr || device.value != 0 || stream == nullptr ||
        stream == instance.compute_stream) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const cudaError_t status = cudaMemsetAsync(device_data, 0x2A, sizeof(*device_data), stream);
    return status == cudaSuccess
               ? expected<void, Error>{}
               : expected<void, Error>{make_unexpected(Error{ErrorCode::kFailure})};
  }

  static expected<void, Error> fail_capture_after_enqueue(void* state,
                                                          RuntimeOperatorInstance& instance,
                                                          DeviceId device,
                                                          cudaStream_t stream) noexcept {
    auto enqueued = capture_memset(state, instance, device, stream);
    return enqueued ? expected<void, Error>{make_unexpected(
                          Error{ErrorCode::kFailure, "CUDA_GRAPH_TEST_CAPTURE_FAILURE"})}
                    : enqueued;
  }

  static expected<void, Error> capture_no_cuda_work(void*, RuntimeOperatorInstance&, DeviceId,
                                                    cudaStream_t) noexcept {
    return {};
  }

  int device_count_{};
  cudaStream_t stream_{};
  std::uint32_t* device_data_{};
};

// REQ: RUNTIME-059
// REQ: COMPILE-089
TEST_F(CUDAGraphRegionNativeTest, CapturesPrivatelyAndLaunchesOnSourceStream) {
  auto execution_plan = plan();
  auto runtime_operators = operators();
  auto resources = RuntimeCUDAGraphRegionResources::create(
      execution_plan, runtime_operators, ProcessGroupId{1U});
  ASSERT_TRUE(resources) << resources.error().message;

  auto captured = (*resources)->capture_all(device_data_, &capture_memset);
  ASSERT_TRUE(captured) << captured.error().message;

  ASSERT_EQ(cudaMemsetAsync(device_data_, 0, sizeof(*device_data_), stream_), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  ASSERT_TRUE((*resources)->launch(VertexId{1U}));
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  std::uint32_t observed{};
  ASSERT_EQ(cudaMemcpy(&observed, device_data_, sizeof(observed), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(observed, 0x2A2A2A2AU);
}

// REQ: RUNTIME-059
// REQ: COMPILE-089
TEST_F(CUDAGraphRegionNativeTest, EnqueuesRepeatedLaunchesOnTheSourceStream) {
  auto execution_plan = plan();
  auto runtime_operators = operators();
  auto resources = RuntimeCUDAGraphRegionResources::create(
      execution_plan, runtime_operators, ProcessGroupId{1U});
  ASSERT_TRUE(resources) << resources.error().message;
  ASSERT_TRUE((*resources)->capture_all(device_data_, &capture_memset));

  ASSERT_TRUE((*resources)->launch(VertexId{1U}));
  ASSERT_TRUE((*resources)->launch(VertexId{1U}));
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

// REQ: RUNTIME-059
TEST_F(CUDAGraphRegionNativeTest, LaunchFailureRemainsRecoverableWhenErrorTextCannotAllocate) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  auto execution_plan = plan();
  auto runtime_operators = operators();
  const auto control = std::make_shared<RuntimeCudaExecutionTestControl>(
      RuntimeCudaExecutionFailurePoint::kCUDAGraphLaunch);
  auto resources = RuntimeCUDAGraphRegionResources::create(
      execution_plan, runtime_operators, {}, ProcessGroupId{1U}, {}, nullptr, control);
  ASSERT_TRUE(resources) << resources.error().message;
  ASSERT_TRUE((*resources)->capture_all(device_data_, &capture_memset));

  const auto failed_launch = [&] {
    const test::ScopedAllocationExhaustion exhaust_allocations{0U};
    return (*resources)->launch(VertexId{1U});
  }();

  ASSERT_FALSE(failed_launch);
  EXPECT_EQ(failed_launch.error().code, ErrorCode::kFailure);
  EXPECT_TRUE(failed_launch.error().message.empty());
  // The failed launch still cancels its fixed-output claim, so the one-shot injected failure can
  // be retried normally.
  EXPECT_TRUE((*resources)->launch(VertexId{1U}));
  EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
#else
  GTEST_SKIP() << "allocation-failure containment requires language exception support";
#endif
}

// REQ: LIFECYCLE-011
// REQ: RUNTIME-059
TEST_F(CUDAGraphRegionNativeTest, QuiesceFinishesSubmittedLaunches) {
  auto execution_plan = plan();
  auto runtime_operators = operators();
  auto resources = RuntimeCUDAGraphRegionResources::create(
      execution_plan, runtime_operators, ProcessGroupId{1U});
  ASSERT_TRUE(resources) << resources.error().message;
  ASSERT_TRUE((*resources)->capture_all(device_data_, &capture_memset));

  ASSERT_EQ(cudaMemsetAsync(device_data_, 0, sizeof(*device_data_), stream_), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  test::BlockingHostState blocked_stream;
  test::BlockingHostReleaseGuard release_guard{blocked_stream};
  ASSERT_EQ(cudaLaunchHostFunc(stream_, test::blocking_host_callback, &blocked_stream),
            cudaSuccess);
  ASSERT_TRUE(test::wait_until_started(blocked_stream));
  ASSERT_TRUE((*resources)->launch(VertexId{1U}));

  // Keep the captured memset pending long enough that quiesce must wait for the launch stream.
  std::jthread release_thread([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds{100});
    test::release_blocking_host(blocked_stream);
  });
  RuntimeCudaExecutionDiagnostic diagnostic{.operation = "stale"};
  ASSERT_TRUE((*resources)->quiesce(&diagnostic));
  release_thread.join();
  release_guard.dismiss();
  EXPECT_TRUE(diagnostic.operation.empty());
  EXPECT_EQ(cudaStreamQuery(stream_), cudaSuccess);

  std::uint32_t observed{};
  ASSERT_EQ(cudaMemcpy(&observed, device_data_, sizeof(observed), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(observed, 0x2A2A2A2AU);
}

// REQ: LIFECYCLE-011
// REQ: RUNTIME-059
TEST_F(CUDAGraphRegionNativeTest, QuiesceFailureKeepsStructuralEvidenceWhenTextCannotAllocate) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  auto execution_plan = plan();
  auto runtime_operators = operators();
  const auto control = std::make_shared<RuntimeCudaExecutionTestControl>(
      RuntimeCudaExecutionFailurePoint::kCUDAGraphQuiesce);
  auto resources = RuntimeCUDAGraphRegionResources::create(
      execution_plan, runtime_operators, {}, ProcessGroupId{1U}, {}, nullptr, control);
  ASSERT_TRUE(resources) << resources.error().message;
  ASSERT_TRUE((*resources)->capture_all(device_data_, &capture_memset));
  ASSERT_TRUE((*resources)->launch(VertexId{1U}));

  RuntimeCudaExecutionDiagnostic diagnostic{.operation = "stale"};
  const auto failed_quiesce = [&] {
    const test::ScopedAllocationExhaustion exhaust_allocations{0U};
    return (*resources)->quiesce(&diagnostic);
  }();

  ASSERT_FALSE(failed_quiesce);
  EXPECT_EQ(failed_quiesce.error().code, ErrorCode::kFailure);
  EXPECT_TRUE(failed_quiesce.error().message.empty());
  EXPECT_TRUE(diagnostic.operation.empty());
  EXPECT_EQ(diagnostic.device, DeviceId{0});
  EXPECT_TRUE(diagnostic.stream_id.empty());
  EXPECT_EQ(diagnostic.graph, GraphId{1U});
  EXPECT_EQ(diagnostic.region, RegionId{1U});
  EXPECT_EQ(diagnostic.fused_operator, VertexId{1U});
  EXPECT_NE(diagnostic.driver_result, static_cast<std::int32_t>(cudaSuccess));
  EXPECT_TRUE((*resources)->quiesce());
#else
  GTEST_SKIP() << "allocation-failure containment requires language exception support";
#endif
}

// REQ: RUNTIME-059
TEST_F(CUDAGraphRegionNativeTest, CaptureFailureLeavesNoLaunchableExecutable) {
  auto execution_plan = plan();
  auto runtime_operators = operators();
  auto resources = RuntimeCUDAGraphRegionResources::create(
      execution_plan, runtime_operators, ProcessGroupId{1U});
  ASSERT_TRUE(resources) << resources.error().message;

  auto captured = (*resources)->capture_all(device_data_, &fail_capture_after_enqueue);
  ASSERT_FALSE(captured);
  EXPECT_EQ(captured.error().message, "CUDA_GRAPH_TEST_CAPTURE_FAILURE");
  auto launched = (*resources)->launch(VertexId{1U});
  ASSERT_FALSE(launched);
  EXPECT_EQ(launched.error().code, ErrorCode::kNotReady);
  EXPECT_EQ(launched.error().message, "CUDA_GRAPH_LAUNCH_STATE_INVALID");
}

// REQ: RUNTIME-059
TEST_F(CUDAGraphRegionNativeTest, RejectsCaptureThatContainsNoCudaWork) {
  auto execution_plan = plan();
  auto runtime_operators = operators();
  auto resources = RuntimeCUDAGraphRegionResources::create(
      execution_plan, runtime_operators, ProcessGroupId{1U});
  ASSERT_TRUE(resources) << resources.error().message;

  auto captured = (*resources)->capture_all(nullptr, &capture_no_cuda_work);

  ASSERT_FALSE(captured);
  EXPECT_EQ(captured.error().message, "CUDA_GRAPH_CAPTURE_EMPTY");
}

// REQ: RUNTIME-059
TEST_F(CUDAGraphRegionNativeTest, RestoresAnAmbientCUDADeviceAfterCaptureAndLaunch) {
  if (device_count_ < 2) {
    GTEST_SKIP() << "Two CUDA devices are required";
  }
  auto execution_plan = plan();
  auto runtime_operators = operators();
  auto resources = RuntimeCUDAGraphRegionResources::create(
      execution_plan, runtime_operators, ProcessGroupId{1U});
  ASSERT_TRUE(resources) << resources.error().message;
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);

  ASSERT_TRUE((*resources)->capture_all(device_data_, &capture_memset));
  int current_device{};
  ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
  EXPECT_EQ(current_device, 1);
  ASSERT_TRUE((*resources)->launch(VertexId{1U}));
  ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
  EXPECT_EQ(current_device, 1);
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
}

// REQ: LIFECYCLE-011
// REQ: RUNTIME-059
TEST_F(CUDAGraphRegionNativeTest, ReportsEveryNativeGraphShutdownFailure) {
  using Failure = RuntimeCudaExecutionFailurePoint;
  constexpr std::array<std::pair<Failure, std::string_view>, 3U> kFailures{
      std::pair{Failure::kCUDAGraphExecDestroy, "cuda_graph_exec_destroy"},
      std::pair{Failure::kCUDAGraphDestroy, "cuda_graph_destroy"},
      std::pair{Failure::kCUDAGraphCaptureStreamDestroy, "cuda_graph_capture_stream_destroy"},
  };

  for (const auto& [point, operation] : kFailures) {
    auto execution_plan = plan();
    auto runtime_operators = operators();
    const auto control = std::make_shared<RuntimeCudaExecutionTestControl>(point);
    auto resources = RuntimeCUDAGraphRegionResources::create(
        execution_plan, runtime_operators, {}, ProcessGroupId{1U}, {}, nullptr, control);
    ASSERT_TRUE(resources) << resources.error().message;
    ASSERT_TRUE((*resources)->capture_all(device_data_, &capture_memset));

    RuntimeCudaExecutionDiagnostic diagnostic;
    const auto shutdown = (*resources)->shutdown(&diagnostic);

    ASSERT_FALSE(shutdown);
    EXPECT_TRUE(shutdown.error().message.starts_with("CUDA_GRAPH_RUNTIME_SHUTDOWN_FAILED"));
    EXPECT_EQ(diagnostic.operation, operation);
    EXPECT_EQ(diagnostic.device, DeviceId{0});
    EXPECT_EQ(diagnostic.graph, GraphId{1U});
    EXPECT_EQ(diagnostic.region, RegionId{1U});
    EXPECT_EQ(diagnostic.fused_operator, VertexId{1U});
    EXPECT_NE(diagnostic.driver_result, static_cast<std::int32_t>(cudaSuccess));

    // The one-shot failure leaves only that handle for a safe explicit retry. Independent
    // handles were still destroyed during the failed call.
    EXPECT_TRUE((*resources)->shutdown());
  }
}

// REQ: LIFECYCLE-011
// REQ: RUNTIME-059
TEST_F(CUDAGraphRegionNativeTest, ShutdownFailureKeepsStructuralEvidenceWhenTextCannotAllocate) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  auto execution_plan = plan();
  auto runtime_operators = operators();
  const auto control = std::make_shared<RuntimeCudaExecutionTestControl>(
      RuntimeCudaExecutionFailurePoint::kCUDAGraphExecDestroy);
  auto resources = RuntimeCUDAGraphRegionResources::create(
      execution_plan, runtime_operators, {}, ProcessGroupId{1U}, {}, nullptr, control);
  ASSERT_TRUE(resources) << resources.error().message;
  ASSERT_TRUE((*resources)->capture_all(device_data_, &capture_memset));

  RuntimeCudaExecutionDiagnostic diagnostic{.operation = "stale"};
  const auto failed_shutdown = [&] {
    const test::ScopedAllocationExhaustion exhaust_allocations{0U};
    return (*resources)->shutdown(&diagnostic);
  }();

  ASSERT_FALSE(failed_shutdown);
  EXPECT_EQ(failed_shutdown.error().code, ErrorCode::kFailure);
  EXPECT_TRUE(failed_shutdown.error().message.empty());
  EXPECT_EQ(diagnostic.lifecycle_stage, LifecycleStage::kRelease);
  EXPECT_TRUE(diagnostic.operation.empty());
  EXPECT_EQ(diagnostic.device, DeviceId{0});
  EXPECT_TRUE(diagnostic.stream_id.empty());
  EXPECT_EQ(diagnostic.graph, GraphId{1U});
  EXPECT_EQ(diagnostic.region, RegionId{1U});
  EXPECT_EQ(diagnostic.fused_operator, VertexId{1U});
  EXPECT_NE(diagnostic.driver_result, static_cast<std::int32_t>(cudaSuccess));
  EXPECT_TRUE((*resources)->shutdown());
#else
  GTEST_SKIP() << "allocation-failure containment requires language exception support";
#endif
}

}  // namespace
}  // namespace holoscan::detail
