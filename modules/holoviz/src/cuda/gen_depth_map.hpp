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

#ifndef MODULES_HOLOVIZ_SRC_CUDA_GEN_DEPTH_MAP_HPP
#define MODULES_HOLOVIZ_SRC_CUDA_GEN_DEPTH_MAP_HPP

#include <cuda.h>

#include <cstddef>
#include <cstdint>

#include "../holoviz/depth_map_render_mode.hpp"
#include "../holoviz/image_format.hpp"

namespace holoscan::viz {

/**
 * @brief Generate vertex coordinates for depth map rendering.
 *
 * This generates a three component vertex, x and y define a regular grid and run from 0...1 and
 * z is the depth value read from `src`.
 *
 * @param depth_format depth values format
 * @param width     depth map width
 * @param height    depth map height
 * @param src       memory containing depth values
 * @param dst       memory to write generated coordinates to
 * @param stream    CUDA stream to use
 */
void GenDepthMapCoords(ImageFormat depth_format, uint32_t width, uint32_t height, CUdeviceptr src,
                       CUdeviceptr dst, CUstream stream);

/**
 * @brief Generate the indices for depth map rendering
 *
 * @param render_mode   depth map render mode
 * @param width     depth map width
 * @param height    depth map height
 * @param dst       memory to write generated indices to
 * @param stream    CUDA stream to use
 * @return size_t the amount of bytes generated
 */
size_t GenDepthMapIndices(DepthMapRenderMode render_mode, uint32_t width, uint32_t height,
                          CUdeviceptr dst, CUstream stream);

}  // namespace holoscan::viz

#endif /* MODULES_HOLOVIZ_SRC_CUDA_GEN_DEPTH_MAP_HPP */
