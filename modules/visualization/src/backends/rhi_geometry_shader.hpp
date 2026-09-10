// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace holoviz::backends::detail {

/// One Slang source, one vertex entry point per expansion family (all sharing the fragment
/// shader and the 32-byte push block). See rhi_geometry_shader.cpp for the expansion math.
extern const char* const kRhiGeometrySlangSource;

inline constexpr const char* kRhiGeometryVsLineList = "vertex_line_list";
inline constexpr const char* kRhiGeometryVsLineStrip = "vertex_line_strip";
inline constexpr const char* kRhiGeometryVsCross = "vertex_cross";
inline constexpr const char* kRhiGeometryVsRect = "vertex_rect";
inline constexpr const char* kRhiGeometryVsOval = "vertex_oval";
inline constexpr const char* kRhiGeometryFs = "fragment_main";

/// GPU-side expansion factors (vertices per instanced draw), mirrored by the record-time
/// draw calls. Every topology lowers to triangles so line_width_px is honored uniformly.
inline constexpr uint32_t kRhiGeometryLineVertices = 6;    // 1 square-capped quad per segment
inline constexpr uint32_t kRhiGeometryCrossVertices = 12;  // 2 arm quads per cross
inline constexpr uint32_t kRhiGeometryRectVertices = 24;   // 4 side quads (mitered ring)
inline constexpr uint32_t kRhiGeometryOvalSegments = 32;
inline constexpr uint32_t kRhiGeometryOvalVertices = kRhiGeometryOvalSegments * 6;

/// The push block below, as reflected (float4 + float2 + float + uint, std140).
inline constexpr uint32_t kRhiGeometryPushBytes = 32;

}  // namespace holoviz::backends::detail
