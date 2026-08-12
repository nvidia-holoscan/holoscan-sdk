/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/sync/deferred_ref_release.hpp"
#include "cuda/context.hpp"
#include "cuda/stream.hpp"

#include <cuda.h>

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <utility>

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

struct HostFnPayload {
  std::atomic<bool>* flag{};
};

void CUDAAPI set_host_fn_flag(void* user_data) {
  const auto payload = static_cast<HostFnPayload*>(user_data);
  payload->flag->store(true, std::memory_order_release);
  delete payload;
}

}  // namespace

TEST(CudaStreamTest, DestructorSynchronizesPendingHostFunction) {
  skip_unless_cuda_available();

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  std::atomic<bool> host_fn_ran{false};

  {
    holomq::cuda::Stream stream(CU_STREAM_NON_BLOCKING);
    ASSERT_TRUE(static_cast<bool>(stream));

    auto* payload = new HostFnPayload{&host_fn_ran};
    ASSERT_EQ(cuLaunchHostFunc(reinterpret_cast<CUstream>(stream.get()), set_host_fn_flag, payload),
              CUDA_SUCCESS);
    ASSERT_EQ(cuStreamQuery(reinterpret_cast<CUstream>(stream.get())), CUDA_ERROR_NOT_READY);
  }

  EXPECT_TRUE(host_fn_ran.load(std::memory_order_acquire));
}

TEST(CudaStreamTest, MoveAssignDrainsReplacedStream) {
  skip_unless_cuda_available();

  holomq::cuda::Context cuda_ctx = make_cuda_context();
  ASSERT_TRUE(static_cast<bool>(cuda_ctx));
  ASSERT_EQ(cuCtxSetCurrent(cuda_ctx.get()), CUDA_SUCCESS);

  std::atomic<bool> replaced_fn_ran{false};
  std::atomic<bool> transferred_fn_ran{false};

  holomq::cuda::Stream first(CU_STREAM_NON_BLOCKING);
  ASSERT_TRUE(static_cast<bool>(first));
  holomq::cuda::Stream second(CU_STREAM_NON_BLOCKING);
  ASSERT_TRUE(static_cast<bool>(second));

  auto* transferred_payload = new HostFnPayload{&transferred_fn_ran};
  ASSERT_EQ(cuLaunchHostFunc(
                reinterpret_cast<CUstream>(first.get()), set_host_fn_flag, transferred_payload),
            CUDA_SUCCESS);
  auto* replaced_payload = new HostFnPayload{&replaced_fn_ran};
  ASSERT_EQ(cuLaunchHostFunc(
                reinterpret_cast<CUstream>(second.get()), set_host_fn_flag, replaced_payload),
            CUDA_SUCCESS);
  ASSERT_EQ(cuStreamQuery(reinterpret_cast<CUstream>(second.get())), CUDA_ERROR_NOT_READY);

  CUstream_st* const transferred_handle = first.get();
  second = std::move(first);

  // Move assignment destroys the replaced stream, so its pending host function must have drained.
  // The transferred stream is adopted as-is; its pending work travels with the handle.
  EXPECT_TRUE(replaced_fn_ran.load(std::memory_order_acquire));
  EXPECT_FALSE(static_cast<bool>(first));
  EXPECT_EQ(second.get(), transferred_handle);

  // Draining the new owner completes the work that was pending on the transferred stream.
  second = holomq::cuda::Stream{};
  EXPECT_TRUE(transferred_fn_ran.load(std::memory_order_acquire));
}

TEST(CudaStreamTest, NullStreamPinsDeferredRefCount) {
  auto mapping_or = holomq::ShmRefCountMapping::create(1U);
  ASSERT_TRUE(mapping_or.has_value());
  holomq::ShmRefCountMappingPtr mapping = std::move(mapping_or).value();
  ASSERT_TRUE(mapping->table().try_acquire_slot(0U));

  holomq::defer_ref_release_on_stream(mapping, 0U, nullptr);
  EXPECT_EQ(mapping->table().ref_count(0U), 1U)
      << "uncertain CUDA completion must keep the slot pinned";

  mapping->table().release_ref(0U);
}
