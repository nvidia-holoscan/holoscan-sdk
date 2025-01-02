/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gen_depth_map.hpp"

#include <type_traits>

#include "cuda_service.hpp"

namespace holoscan::viz {

namespace {

template <typename T>
__global__ void GenDepthMapCoordsKernel(uint32_t width, uint32_t height, float inv_width,
                                        float inv_height, const T* src, float* dst) {
  const uint2 launch_index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
  if ((launch_index.x >= width) || (launch_index.y >= height)) { return; }

  const size_t offset = launch_index.x + launch_index.y * width;

  dst += offset * 3;

  dst[0] = float(launch_index.x) * inv_width - 0.5F;
  dst[1] = float(launch_index.y) * inv_height - 0.5F;
  if constexpr (std::is_same<T, uint8_t>::value) {
    dst[2] = float(src[offset]) / 255.F;
  } else if constexpr (std::is_same<T, float>::value) {
    dst[2] = src[offset];
  }
}

template <DepthMapRenderMode render_mode>
__global__ void GenDepthMapIndicesKernel(uint32_t width, uint32_t height, uint32_t* dst) {
  const uint2 launch_index =
      make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

  const uint32_t index = launch_index.y * width + launch_index.x;

  switch (render_mode) {
    case DepthMapRenderMode::LINES: {
      if ((launch_index.x >= width) || (launch_index.y >= height)) { return; }

      const size_t offset = launch_index.x * 4 + launch_index.y * ((width - 1) * 4 + 2);
      dst += offset;

      // line right to next column (except last column)
      if (launch_index.x != width - 1) {
        dst[0] = index;
        dst[1] = index + 1;
        dst += 2;
      }

      // line down to next row (except last row)
      if (launch_index.y != height - 1) {
        dst[0] = index;
        dst[1] = index + width;
      }
    } break;
    case DepthMapRenderMode::TRIANGLES: {
      if ((launch_index.x >= width - 1) || (launch_index.y >= height - 1)) { return; }

      const size_t offset = (launch_index.x + launch_index.y * width) * 6;
      dst += offset;

      dst[0] = index;
      dst[1] = index + 1;
      dst[2] = index + width;
      dst[3] = index + 1;
      dst[4] = index + width + 1;
      dst[5] = index + width;
    } break;
  }
}

}  // namespace

void GenDepthMapCoords(ImageFormat depth_format, uint32_t width, uint32_t height, CUdeviceptr src,
                       CUdeviceptr dst, CUstream stream) {
  const dim3 block_dim(32, 32);
  const dim3 launch_grid((width + (block_dim.x - 1)) / block_dim.x,
                         (height + (block_dim.y - 1)) / block_dim.y);

  const float inv_width = 1.F / float(width);
  const float inv_height = 1.F / float(height);

  switch (depth_format) {
    case ImageFormat::R8_UNORM:
      GenDepthMapCoordsKernel<uint8_t>
          <<<launch_grid, block_dim, 0, stream>>>(width,
                                                  height,
                                                  inv_width,
                                                  inv_height,
                                                  reinterpret_cast<const uint8_t*>(src),
                                                  reinterpret_cast<float*>(dst));
      CudaRTCheck(cudaPeekAtLastError());
      break;
    case ImageFormat::D32_SFLOAT:
      GenDepthMapCoordsKernel<float>
          <<<launch_grid, block_dim, 0, stream>>>(width,
                                                  height,
                                                  inv_width,
                                                  inv_height,
                                                  reinterpret_cast<const float*>(src),
                                                  reinterpret_cast<float*>(dst));
      CudaRTCheck(cudaPeekAtLastError());
      break;
    default:
      throw std::runtime_error("Unsupported depth map format.");
  }
}

size_t GenDepthMapIndices(DepthMapRenderMode render_mode, uint32_t width, uint32_t height,
                          CUdeviceptr dst, CUstream stream) {
  const dim3 block_dim(32, 32);
  const dim3 launch_grid((width + (block_dim.x - 1)) / block_dim.x,
                         (height + (block_dim.y - 1)) / block_dim.y);

  size_t generated_bytes;

  switch (render_mode) {
    case DepthMapRenderMode::LINES:
      GenDepthMapIndicesKernel<DepthMapRenderMode::LINES>
          <<<launch_grid, block_dim, 0, stream>>>(width, height, reinterpret_cast<uint32_t*>(dst));
      CudaRTCheck(cudaPeekAtLastError());
      generated_bytes = (width - 1) * (height - 1) * sizeof(uint32_t) * 4;
      // last column
      generated_bytes += (height - 1) * sizeof(uint32_t) * 2;
      // last row
      generated_bytes += (width - 1) * sizeof(uint32_t) * 2;
      break;
    case DepthMapRenderMode::TRIANGLES:
      GenDepthMapIndicesKernel<DepthMapRenderMode::TRIANGLES>
          <<<launch_grid, block_dim, 0, stream>>>(width, height, reinterpret_cast<uint32_t*>(dst));
      CudaRTCheck(cudaPeekAtLastError());
      generated_bytes = (width - 1) * (height - 1) * sizeof(uint32_t) * 6;
      break;
    default:
      throw std::runtime_error("Unsupported depth map render mode.");
  }

  return generated_bytes;
}

}  // namespace holoscan::viz
