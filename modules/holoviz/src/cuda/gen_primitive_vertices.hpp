/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MODULES_HOLOVIZ_SRC_CUDA_GEN_PRIMITIVE_VERTICES_HPP
#define MODULES_HOLOVIZ_SRC_CUDA_GEN_PRIMITIVE_VERTICES_HPP

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../holoviz/primitive_topology.hpp"

namespace holoscan::viz {

/// the segment count a circle is made of
constexpr uint32_t CIRCLE_SEGMENTS = 32;

/**
 * @brief Generate vertex coordinates for geometric primitives.
 *
 * @param topology primitive topology
 * @param primitive_count     primitive count
 * @param vertex_counts  vertex counts
 * @param aspect_ratio aspect ratio
 * @param src       memory containing source coordinates
 * @param dst       memory to write generated coordinates to
 * @param stream    CUDA stream to use
 */
void gen_primitive_vertices(PrimitiveTopology topology, uint32_t primitive_count,
                            const std::vector<uint32_t>& vertex_counts, float aspect_ratio,
                            CUdeviceptr src, CUdeviceptr dst, CUstream stream);

}  // namespace holoscan::viz

#endif /* MODULES_HOLOVIZ_SRC_CUDA_GEN_PRIMITIVE_VERTICES_HPP */
