/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/sync/cuda_stream_generation.hpp"
#include "cuda/context.hpp"

#include <cuda.h>

#include <gtest/gtest.h>

#include <cstdint>

namespace {

void skip_unless_cuda_available() {
  const CUresult init_result = cuInit(0);
  if (init_result != CUDA_SUCCESS) {
    GTEST_SKIP() << "CUDA driver not available";
  }

  CUdevice device{};
  if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
    GTEST_SKIP() << "No CUDA device available";
  }
}

[[nodiscard]] holomq::cuda::Context make_cuda_context() {
  CUdevice device{};
  if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
    return {};
  }
  return holomq::cuda::Context{device};
}

}  // namespace

TEST(CudaStreamGenerationTest, RejectsInvalidInputs) {
  skip_unless_cuda_available();

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  CUstream stream{};
  ASSERT_EQ(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);

  EXPECT_FALSE(holomq::stream_write_generation(nullptr, 0x1000U, 1U));
  EXPECT_FALSE(holomq::stream_wait_generation(nullptr, 0x1000U, 1U));
  EXPECT_FALSE(holomq::stream_write_generation(stream, 0U, 1U));
  EXPECT_FALSE(holomq::stream_wait_generation(stream, 0U, 1U));

  ASSERT_EQ(cuStreamDestroy(stream), CUDA_SUCCESS);
}

TEST(CudaStreamGenerationTest, WriteThenWaitOnDeviceWord) {
  skip_unless_cuda_available();

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  CUdeviceptr device_word{};
  ASSERT_EQ(cuMemAlloc(&device_word, sizeof(std::uint64_t)), CUDA_SUCCESS);

  std::uint64_t zero = 0U;
  ASSERT_EQ(cuMemcpyHtoD(device_word, &zero, sizeof(zero)), CUDA_SUCCESS);
  // A pageable HtoD copy may return after staging but before its DMA completes. Ensure the
  // initialization cannot race the nonblocking stream memory operations below.
  ASSERT_EQ(cuCtxSynchronize(), CUDA_SUCCESS);

  CUstream write_stream{};
  CUstream wait_stream{};
  ASSERT_EQ(cuStreamCreate(&write_stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamCreate(&wait_stream, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);

  ASSERT_TRUE(holomq::stream_write_generation(write_stream, device_word, 1U));
  ASSERT_TRUE(holomq::stream_wait_generation(wait_stream, device_word, 1U));

  ASSERT_EQ(cuStreamSynchronize(write_stream), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamSynchronize(wait_stream), CUDA_SUCCESS);

  std::uint64_t observed = 0U;
  ASSERT_EQ(cuMemcpyDtoH(&observed, device_word, sizeof(observed)), CUDA_SUCCESS);
  EXPECT_EQ(observed, 1U);

  ASSERT_EQ(cuStreamDestroy(write_stream), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamDestroy(wait_stream), CUDA_SUCCESS);
  ASSERT_EQ(cuMemFree(device_word), CUDA_SUCCESS);
}
