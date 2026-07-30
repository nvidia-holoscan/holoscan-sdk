/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
