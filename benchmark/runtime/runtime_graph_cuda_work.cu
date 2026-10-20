/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_graph_cuda_work.hpp"

#include <cstdint>

namespace holoscan::benchmarking {

namespace {

constexpr std::uint32_t kThreadsPerBlock = 256U;

__global__ void bounded_cuda_work_kernel(std::uint64_t* scratch,
                                         const std::uint64_t duration_cycles) {
  const std::uint64_t started = clock64();
  std::uint64_t value = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1U;
  while (clock64() - started < duration_cycles) {
    value = value * 2862933555777941757ULL + 3037000493ULL;
  }
  if (threadIdx.x == 0U) {
    scratch[blockIdx.x] = value;
  }
}

}  // namespace

cudaError_t launch_runtime_graph_cuda_work(cudaStream_t stream, void* scratch,
                                           const std::uint64_t duration_cycles,
                                           const std::uint32_t block_count) noexcept {
  if (stream == nullptr || scratch == nullptr || duration_cycles == 0U || block_count == 0U) {
    return cudaErrorInvalidValue;
  }
  bounded_cuda_work_kernel<<<block_count, kThreadsPerBlock, 0U, stream>>>(
      static_cast<std::uint64_t*>(scratch), duration_cycles);
  return cudaGetLastError();
}

}  // namespace holoscan::benchmarking
