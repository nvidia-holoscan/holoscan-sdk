// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "holoviz_op/types.hpp"
namespace holoscan::visualization::backends::detail {

/// Sampled image layer: one 6-vertex quad (SV_VertexID corner table, no vertex buffer)
/// placed by the layer's ViewTransform, sampling the slot texture. Single-channel (R8/R16)
/// sources replicate the red channel to gray and BGRA sources swizzle to RGBA, each via a push
/// flag; 16-bit sources need no extra handling because UNORM sampling already yields 0..1.
extern const char* const kHolovizImageSlangSource;

inline constexpr const char* kHolovizImageVs = "vertex_main";
inline constexpr const char* kHolovizImageFs = "fragment_main";
inline constexpr uint32_t kHolovizImageQuadVertices = 6;

/// float2 view_offset + float2 view_scale + float2 uv_scale + uint flags + pad (std140).
inline constexpr uint32_t kHolovizImagePushBytes = 40;
inline constexpr uint32_t kHolovizImageFlagGrayscaleReplicate = 1;
/// A BGRA source is stored in an RGBA8 slot texture (CUDA import has no BGRA array format):
/// its bytes land unchanged and the fragment shader swaps the red and blue channels. Other
/// formats use a slot texture matching their own format (R8, R16, RGBA8, RGBA16).
inline constexpr uint32_t kHolovizImageFlagSwizzleBGRA = 2;

}  // namespace holoscan::visualization::backends::detail
