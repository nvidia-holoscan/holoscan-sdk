// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include "holoviz_op/backends/rgb_to_rgba.hpp"

namespace holoscan::visualization::backends {
namespace {

__global__ void rgb_to_rgba(const unsigned char* source, size_t pitch, uint32_t width,
                            uint32_t height, cudaSurfaceObject_t destination) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const auto* rgb = source + static_cast<size_t>(y) * pitch + static_cast<size_t>(x) * 3;
  surf2Dwrite(make_uchar4(rgb[0], rgb[1], rgb[2], 255), destination, x * sizeof(uchar4), y);
}

}  // namespace

cudaError_t launch_rgb_to_rgba(CUdeviceptr source, size_t pitch, uint32_t width, uint32_t height,
                               CUsurfObject destination, CUstream stream) {
  const dim3 block(32, 8);
  const dim3 grid((width - 1) / block.x + 1, (height - 1) / block.y + 1);
  rgb_to_rgba<<<grid, block, 0, stream>>>(
      reinterpret_cast<const unsigned char*>(source), pitch, width, height, destination);
  return cudaGetLastError();
}

}  // namespace holoscan::visualization::backends
