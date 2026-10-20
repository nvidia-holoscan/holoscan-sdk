/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pattern.hpp"

#include <stdexcept>
#include <string>

namespace {

__global__ void band_pattern_kernel(unsigned char* dst, size_t dst_pitch, uint32_t width,
                                    uint32_t band_height, uint32_t y0, uint32_t full_height,
                                    uint32_t frame) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= band_height) {
    return;
  }

  const float u = static_cast<float>(x) / static_cast<float>(width);
  const float v = static_cast<float>(y0 + y) / static_cast<float>(full_height);
  const float t = static_cast<float>(frame) * 0.05F;

  constexpr float kTwoPi = 6.2831853F;
  float r = 0.5F + 0.5F * __sinf(kTwoPi * (u + t * 0.11F));
  float g = 0.5F + 0.5F * __sinf(kTwoPi * (v - t * 0.07F) + 2.094F);
  float b = 0.5F + 0.5F * __sinf(kTwoPi * (u + v + t * 0.05F) + 4.189F);
  if (y == 0) {
    // Band-boundary marker: the first row of every band is white.
    r = g = b = 1.0F;
  }

  uchar4* row = reinterpret_cast<uchar4*>(dst + static_cast<size_t>(y) * dst_pitch);
  row[x] = make_uchar4(static_cast<unsigned char>(r * 255.0F),
                       static_cast<unsigned char>(g * 255.0F),
                       static_cast<unsigned char>(b * 255.0F),
                       255);
}

}  // namespace

void launch_band_pattern(CUdeviceptr dst, size_t dst_pitch, uint32_t width, uint32_t band_height,
                         uint32_t y0, uint32_t full_height, uint32_t frame, CUstream stream) {
  const dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x, (band_height + block.y - 1) / block.y);
  band_pattern_kernel<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<unsigned char*>(dst), dst_pitch, width, band_height, y0, full_height, frame);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("band_pattern_kernel launch failed: ") +
                             cudaGetErrorString(error));
  }
}
