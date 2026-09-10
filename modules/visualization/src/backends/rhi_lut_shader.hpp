// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace holoviz::backends::detail {

/// Segmentation-mask colorization: a nearest-sampled kR8Unorm index mask looked up in an
/// RGBA float palette (StructuredBuffer). Same quad/view-transform scheme as the image
/// shader.
extern const char* const kRhiLutSlangSource;

inline constexpr const char* kRhiLutVs = "vertex_main";
inline constexpr const char* kRhiLutFs = "fragment_main";
inline constexpr uint32_t kRhiLutQuadVertices = 6;

/// float2 view_offset + float2 view_scale + float2 uv_scale + uint lut_size + pad (std140).
inline constexpr uint32_t kRhiLutPushBytes = 40;

}  // namespace holoviz::backends::detail
