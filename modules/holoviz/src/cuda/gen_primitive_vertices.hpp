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
