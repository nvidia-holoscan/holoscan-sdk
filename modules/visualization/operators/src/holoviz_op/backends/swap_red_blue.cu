// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include "holoviz_op/backends/swap_red_blue.hpp"

namespace holoscan::visualization::backends {
namespace {

__global__ void swap_red_blue(unsigned char* data, size_t pitch, uint32_t width, uint32_t height) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  unsigned char* pixel = data + static_cast<size_t>(y) * pitch + static_cast<size_t>(x) * 4;
  const unsigned char red = pixel[0];
  pixel[0] = pixel[2];
  pixel[2] = red;
}

}  // namespace

cudaError_t launch_swap_red_blue_inplace(CUdeviceptr data, size_t pitch, uint32_t width,
                                         uint32_t height, CUstream stream) {
  const dim3 block(32, 8);
  const dim3 grid((width - 1) / block.x + 1, (height - 1) / block.y + 1);
  swap_red_blue<<<grid, block, 0, stream>>>(
      reinterpret_cast<unsigned char*>(data), pitch, width, height);
  return cudaGetLastError();
}

}  // namespace holoscan::visualization::backends
