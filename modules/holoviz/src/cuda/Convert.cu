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

#include "Convert.h"

#include "CudaService.h"

namespace clara::holoviz
{

namespace
{

/**
 * Convert from R8G8B8 to R8G8B8A8 (set alpha to 0xFF)
 */
__global__ void ConvertR8G8B8ToR8G8B8A8Kernel(uint32_t width, uint32_t height, const uint8_t *src, size_t src_pitch,
                                              CUsurfObject dst_surface)
{
    const uint2 launch_index = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    if ((launch_index.x >= width) || (launch_index.y >= height))
    {
        return;
    }

    const size_t src_offset = launch_index.x * 3 + launch_index.y * src_pitch;

    const uchar4 data{src[src_offset + 0], src[src_offset + 1], src[src_offset + 2], 0xFF};
    surf2Dwrite(data, dst_surface, launch_index.x * sizeof(uchar4), launch_index.y);
}

} // namespace

void ConvertR8G8B8ToR8G8B8A8(uint32_t width, uint32_t height, CUdeviceptr src, size_t src_pitch, CUarray dst)
{
    CUsurfObject dst_surface;
    CUDA_RESOURCE_DESC res_desc{};
    res_desc.resType          = CU_RESOURCE_TYPE_ARRAY;
    res_desc.res.array.hArray = dst;
    CudaCheck(cuSurfObjectCreate(&dst_surface, &res_desc));

    const dim3 block_dim(32, 32);
    const dim3 launch_grid((width + (block_dim.x - 1)) / block_dim.x, (height + (block_dim.y - 1)) / block_dim.y);
    ConvertR8G8B8ToR8G8B8A8Kernel<<<launch_grid, block_dim>>>(width, height, reinterpret_cast<const uint8_t *>(src),
                                                              src_pitch, dst_surface);

    CudaCheck(cuSurfObjectDestroy(dst_surface));
}

} // namespace clara::holoviz
