/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "convert.hpp"

#include "cuda_service.hpp"

namespace holoscan::viz {

namespace {

/**
 * Convert from R8G8B8 to R8G8B8A8 (set alpha to 0xFF)
 */
__global__ void ConvertR8G8B8ToR8G8B8A8Kernel(uint32_t width, uint32_t height, const uint8_t* src,
                                              size_t src_pitch, CUsurfObject dst_surface) {
  const uint2 launch_index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
  if ((launch_index.x >= width) || (launch_index.y >= height)) { return; }

  const size_t src_offset = launch_index.x * 3 + launch_index.y * src_pitch;

  const uchar4 data{src[src_offset + 0], src[src_offset + 1], src[src_offset + 2], 0xFF};
  surf2Dwrite(data, dst_surface, launch_index.x * sizeof(uchar4), launch_index.y);
}

}  // namespace

void ConvertR8G8B8ToR8G8B8A8(uint32_t width, uint32_t height, CUdeviceptr src, size_t src_pitch,
                             CUarray dst, CUstream stream) {
  UniqueCUsurfObject dst_surface;

  dst_surface.reset([dst] {
    CUDA_RESOURCE_DESC res_desc{};
    res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    res_desc.res.array.hArray = dst;
    CUsurfObject surf_object;
    CudaCheck(cuSurfObjectCreate(&surf_object, &res_desc));
    return surf_object;
  }());

  const dim3 block_dim(32, 32);
  const dim3 launch_grid((width + (block_dim.x - 1)) / block_dim.x,
                         (height + (block_dim.y - 1)) / block_dim.y);
  ConvertR8G8B8ToR8G8B8A8Kernel<<<launch_grid, block_dim, 0, stream>>>(
      width, height, reinterpret_cast<const uint8_t*>(src), src_pitch, dst_surface.get());
}

namespace {

/**
 * Convert from B8G8R8A8 to R8G8B8A8
 */
__global__ void ConvertB8G8R8A8ToR8G8B8A8Kernel(uint32_t width, uint32_t height, const uint8_t* src,
                                                size_t src_pitch, uint8_t* dst, size_t dst_pitch) {
  const uint2 launch_index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
  if ((launch_index.x >= width) || (launch_index.y >= height)) { return; }

  const size_t src_offset = launch_index.x * 4 + launch_index.y * src_pitch;
  const size_t dst_offset = launch_index.x * 4 + launch_index.y * dst_pitch;

  dst[dst_offset + 2] = src[src_offset + 0];
  dst[dst_offset + 1] = src[src_offset + 1];
  dst[dst_offset + 0] = src[src_offset + 2];
  dst[dst_offset + 3] = src[src_offset + 3];
}

}  // namespace

void ConvertB8G8R8A8ToR8G8B8A8(uint32_t width, uint32_t height, CUdeviceptr src, size_t src_pitch,
                               CUdeviceptr dst, size_t dst_pitch, CUstream stream) {
  const dim3 block_dim(32, 32);
  const dim3 launch_grid((width + (block_dim.x - 1)) / block_dim.x,
                         (height + (block_dim.y - 1)) / block_dim.y);
  ConvertB8G8R8A8ToR8G8B8A8Kernel<<<launch_grid, block_dim, 0, stream>>>(
      width,
      height,
      reinterpret_cast<const uint8_t*>(src),
      src_pitch,
      reinterpret_cast<uint8_t*>(dst),
      dst_pitch);
}

}  // namespace holoscan::viz
