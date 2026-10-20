/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <memory>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <holoscan/core/tensor_output_loan.hpp>

#include "channel/tensor_stream_readiness.hpp"
#include "compiler/execution_plan_state.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

TEST(TensorStreamReadinessTest, ClosingDoesNotClaimHostReadiness) {
  auto lifetime = std::make_shared<holomq::cuda::StreamLifetime>(cudaStreamLegacy, 0);
  channel_internal::TensorStreamReadiness readiness;
  readiness.set(cudaStreamLegacy, lifetime);
  lifetime->close_admission();
  EXPECT_FALSE(readiness.host());
  EXPECT_TRUE(lifetime->publish_retired());
  EXPECT_TRUE(readiness.host());
}

TEST(TensorStreamReadinessTest, QuarantinedDoesNotClaimHostReadiness) {
  auto lifetime = std::make_shared<holomq::cuda::StreamLifetime>(cudaStreamLegacy, 0);
  channel_internal::TensorStreamReadiness readiness;
  readiness.set(cudaStreamLegacy, lifetime);
  lifetime->close_admission();
  EXPECT_TRUE(lifetime->publish_quarantined());
  EXPECT_FALSE(readiness.host());
}

TEST(TensorStreamReadinessTest, HostWaitRestoresDifferentDeviceAndNonPrimaryContext) {
  int devices{};
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
    GTEST_SKIP() << "Requires two CUDA devices";
  }
  int previous{};
  ASSERT_EQ(cudaGetDevice(&previous), cudaSuccess);
  CUcontext previous_context{};
  ASSERT_EQ(cuCtxGetCurrent(&previous_context), CUDA_SUCCESS);
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  struct StreamGuard {
    cudaStream_t stream{};
    int previous{};
    CUcontext previous_context{};
    CUcontext consumer_context{};
    ~StreamGuard() {
      static_cast<void>(cudaSetDevice(0));
      static_cast<void>(cudaStreamDestroy(stream));
      if (consumer_context != nullptr) {
        static_cast<void>(cuCtxDestroy(consumer_context));
      }
      static_cast<void>(cudaSetDevice(previous));
      static_cast<void>(cuCtxSetCurrent(previous_context));
    }
  } guard{.previous = previous, .previous_context = previous_context};
  ASSERT_EQ(cudaStreamCreateWithFlags(&guard.stream, cudaStreamNonBlocking), cudaSuccess);
  auto lifetime = std::make_shared<holomq::cuda::StreamLifetime>(guard.stream, 0);
  channel_internal::TensorStreamReadiness readiness;
  readiness.set(guard.stream, lifetime);
  CUdevice consumer_device{};
  ASSERT_EQ(cuDeviceGet(&consumer_device, 1), CUDA_SUCCESS);
#if CUDA_VERSION >= 13000
  ASSERT_EQ(cuCtxCreate(&guard.consumer_context, nullptr, 0U, consumer_device), CUDA_SUCCESS);
#else
  ASSERT_EQ(cuCtxCreate(&guard.consumer_context, 0U, consumer_device), CUDA_SUCCESS);
#endif
  // cuCtxCreate pushes the new context. Balance that push, then select it on the
  // original stack level so the test leaves the caller's context stack unchanged.
  CUcontext popped{};
  ASSERT_EQ(cuCtxPopCurrent(&popped), CUDA_SUCCESS);
  ASSERT_EQ(popped, guard.consumer_context);
  ASSERT_EQ(cuCtxSetCurrent(guard.consumer_context), CUDA_SUCCESS);
  EXPECT_TRUE(readiness.host());
  int observed{};
  ASSERT_EQ(cudaGetDevice(&observed), cudaSuccess);
  EXPECT_EQ(observed, 1);
  CUcontext observed_context{};
  ASSERT_EQ(cuCtxGetCurrent(&observed_context), CUDA_SUCCESS);
  EXPECT_EQ(observed_context, guard.consumer_context);
  lifetime->close_admission();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(guard.stream), cudaSuccess);
  EXPECT_TRUE(lifetime->publish_retired());
}

}  // namespace
}  // namespace holoscan
