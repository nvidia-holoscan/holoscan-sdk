/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda.h>

#include "buffer_pool/sync/stream_release.hpp"

namespace {
std::uint32_t* exit_gate{};
int cuda_failure(const char* step, CUresult result) {
  std::fprintf(stderr, "%s failed: CUDA result=%d\n", step, static_cast<int>(result));
  return 1;
}
}  // namespace

int main() {
  // Register before the worker, so this runs only after its later atexit handler returns.
  // Release the tiny test gate before the process tears down CUDA driver state.
  if (std::atexit([] {
        if (exit_gate != nullptr) {
          std::atomic_ref<std::uint32_t>(*exit_gate).store(1U, std::memory_order_release);
        }
      }) != 0) {
    std::fputs("atexit registration failed\n", stderr);
    return 1;
  }
  CUdevice device{};
  if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
    return 77;
  }
  CUcontext context{};
  auto result = cuDevicePrimaryCtxRetain(&context, device);
  if (result != CUDA_SUCCESS) {
    return cuda_failure("primary context retain", result);
  }
  result = cuCtxSetCurrent(context);
  if (result != CUDA_SUCCESS) {
    return cuda_failure("select primary context", result);
  }
  auto endpoint = holomq::EndpointStream::create();
  if (endpoint == nullptr) {
    std::fputs("endpoint stream creation failed\n", stderr);
    return 1;
  }
  std::uint32_t* word{};
  CUdeviceptr device_word{};
  result =
      cuMemHostAlloc(reinterpret_cast<void**>(&word), sizeof(*word), CU_MEMHOSTALLOC_DEVICEMAP);
  if (result != CUDA_SUCCESS) {
    return cuda_failure("allocate exit gate", result);
  }
  result = cuMemHostGetDevicePointer(&device_word, word, 0U);
  if (result != CUDA_SUCCESS) {
    return cuda_failure("map exit gate", result);
  }
  *word = 0U;
  exit_gate = word;
  result = cuStreamWaitValue32(endpoint->get(), device_word, 1U, CU_STREAM_WAIT_VALUE_EQ);
  if (result != CUDA_SUCCESS) {
    return cuda_failure("queue exit gate wait", result);
  }
  // Model a peer that has not written its ready generation. The gate remains blocked
  // throughout worker shutdown, which must return before the earlier atexit handler opens it.
  endpoint.reset();
  static_cast<void>(cuDevicePrimaryCtxRelease(device));
  std::puts("queued blocked endpoint retirement; exiting normally");
  return 0;
}
