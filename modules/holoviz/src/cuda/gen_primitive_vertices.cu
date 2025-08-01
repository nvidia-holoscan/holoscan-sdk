/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gen_primitive_vertices.hpp"

#include <vector>

#include "cuda_service.hpp"

namespace holoscan::viz {

namespace {

__global__ void copy_and_add_zero(uint32_t vertex_count, const float* src, float* dst) {
  const uint vertex_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertex_index >= vertex_count) {
    return;
  }

  // just copy and add zero for Z
  src += vertex_index * 2;
  dst += vertex_index * 3;
  dst[0] = src[0];
  dst[1] = src[1];
  dst[2] = 0.F;
}

// generate crosses
__global__ void gen_cross_list_vertices(uint32_t primitive_count, float aspect_ratio,
                                        const float* src, float* dst) {
  const uint primitive_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (primitive_index >= primitive_count) {
    return;
  }

  src += primitive_index * 3;
  const float x = src[0];
  const float y = src[1];
  const float sy = src[2] * 0.5F;
  const float sx = sy / aspect_ratio;

  dst += primitive_index * 12;
  dst[0] = x - sx;
  dst[1] = y;
  dst[2] = 0.F;
  dst[3] = x + sx;
  dst[4] = y;
  dst[5] = 0.F;
  dst[6] = x;
  dst[7] = y - sy;
  dst[8] = 0.F;
  dst[9] = x;
  dst[10] = y + sy;
  dst[11] = 0.F;
}
// generate rectangles
__global__ void gen_rectangle_list_vertices(uint32_t primitive_count, const float* src,
                                            float* dst) {
  const uint primitive_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (primitive_index >= primitive_count) {
    return;
  }

  src += primitive_index * 4;
  const float x0 = src[0];
  const float y0 = src[1];
  const float x1 = src[2];
  const float y1 = src[3];

  dst += primitive_index * 15;
  dst[0] = x0;
  dst[1] = y0;
  dst[2] = 0.F;
  dst[3] = x1;
  dst[4] = y0;
  dst[5] = 0.F;
  dst[6] = x1;
  dst[7] = y1;
  dst[8] = 0.F;
  dst[9] = x0;
  dst[10] = y1;
  dst[11] = 0.F;
  dst[12] = x0;
  dst[13] = y0;
  dst[14] = 0.F;
}

__global__ void gen_oval_list_vertices(uint32_t primitive_count, const float* src, float* dst) {
  const uint primitive_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (primitive_index >= primitive_count) {
    return;
  }

  src += primitive_index * 4;
  const float x = src[0];
  const float y = src[1];
  const float rx = src[2] * 0.5F;
  const float ry = src[3] * 0.5F;

  dst += primitive_index * (CIRCLE_SEGMENTS + 1) * 3;
  for (uint32_t segment = 0; segment <= CIRCLE_SEGMENTS; ++segment) {
    const float rad = (2.F * M_PI) / CIRCLE_SEGMENTS * segment;
    const float px = x + std::cos(rad) * rx;
    const float py = y + std::sin(rad) * ry;
    dst[0] = px;
    dst[1] = py;
    dst[2] = 0.F;
    dst += 3;
  }
}

}  // namespace

void gen_primitive_vertices(PrimitiveTopology topology, uint32_t primitive_count,
                            const std::vector<uint32_t>& vertex_counts, float aspect_ratio,
                            CUdeviceptr src, CUdeviceptr dst, CUstream stream) {
  const dim3 block_dim(32, 1);

  switch (topology) {
    case PrimitiveTopology::POINT_LIST:
    case PrimitiveTopology::LINE_LIST:
    case PrimitiveTopology::LINE_STRIP:
    case PrimitiveTopology::TRIANGLE_LIST: {
      if (vertex_counts.size() != 1) {
        throw std::runtime_error("Unexpected vertex count vector.");
      }
      const dim3 launch_grid((vertex_counts[0] + (block_dim.x - 1)) / block_dim.x);
      copy_and_add_zero<<<launch_grid, block_dim, 0, stream>>>(
          vertex_counts[0], reinterpret_cast<const float*>(src), reinterpret_cast<float*>(dst));
      CudaRTCheck(cudaPeekAtLastError());
    } break;
    case PrimitiveTopology::CROSS_LIST: {
      const dim3 launch_grid((primitive_count + (block_dim.x - 1)) / block_dim.x);
      gen_cross_list_vertices<<<launch_grid, block_dim, 0, stream>>>(
          primitive_count,
          aspect_ratio,
          reinterpret_cast<const float*>(src),
          reinterpret_cast<float*>(dst));
      CudaRTCheck(cudaPeekAtLastError());
    } break;
    case PrimitiveTopology::OVAL_LIST: {
      const dim3 launch_grid((primitive_count + (block_dim.x - 1)) / block_dim.x);
      gen_oval_list_vertices<<<launch_grid, block_dim, 0, stream>>>(
          primitive_count, reinterpret_cast<const float*>(src), reinterpret_cast<float*>(dst));
      CudaRTCheck(cudaPeekAtLastError());
    } break;
    case PrimitiveTopology::RECTANGLE_LIST: {
      const dim3 launch_grid((primitive_count + (block_dim.x - 1)) / block_dim.x);
      gen_rectangle_list_vertices<<<launch_grid, block_dim, 0, stream>>>(
          primitive_count, reinterpret_cast<const float*>(src), reinterpret_cast<float*>(dst));
      CudaRTCheck(cudaPeekAtLastError());
    } break;
    case PrimitiveTopology::POINT_LIST_3D:
    case PrimitiveTopology::LINE_LIST_3D:
    case PrimitiveTopology::LINE_STRIP_3D:
    case PrimitiveTopology::TRIANGLE_LIST_3D:
      if (vertex_counts.size() != 1) {
        throw std::runtime_error("Unexpected vertex count vector.");
      }
      CudaCheck(cuMemcpyAsync(dst, src, vertex_counts[0] * 3 * sizeof(float), stream));
      break;
    default:
      throw std::runtime_error("Unsupported primitive topology.");
  }

#undef CASE
}

}  // namespace holoscan::viz
