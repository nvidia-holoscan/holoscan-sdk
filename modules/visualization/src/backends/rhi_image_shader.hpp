// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace holoviz::backends::detail {

/// Sampled image layer: one 6-vertex quad (SV_VertexID corner table, no vertex buffer)
/// placed by the layer's ViewTransform, sampling the slot texture. R8 sources replicate
/// the red channel to gray via a push flag.
extern const char* const kRhiImageSlangSource;

inline constexpr const char* kRhiImageVs = "vertex_main";
inline constexpr const char* kRhiImageFs = "fragment_main";
inline constexpr uint32_t kRhiImageQuadVertices = 6;

/// float2 view_offset + float2 view_scale + float2 uv_scale + uint flags + pad (std140).
inline constexpr uint32_t kRhiImagePushBytes = 40;
inline constexpr uint32_t kRhiImageFlagGrayscaleReplicate = 1;

}  // namespace holoviz::backends::detail
