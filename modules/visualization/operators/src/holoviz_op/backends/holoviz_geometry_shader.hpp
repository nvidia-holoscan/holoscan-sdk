// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "holoviz_op/types.hpp"
namespace holoscan::visualization::backends::detail {

/// One Slang source, one vertex entry point per expansion family (all sharing the fragment
/// shader and the 32-byte push block). See holoviz_geometry_shader.cpp for the expansion math.
extern const char* const kHolovizGeometrySlangSource;

inline constexpr const char* kHolovizGeometryVsLineList = "vertex_line_list";
inline constexpr const char* kHolovizGeometryVsLineStrip = "vertex_line_strip";
inline constexpr const char* kHolovizGeometryVsCross = "vertex_cross";
inline constexpr const char* kHolovizGeometryVsRect = "vertex_rect";
inline constexpr const char* kHolovizGeometryVsOval = "vertex_oval";
inline constexpr const char* kHolovizGeometryFs = "fragment_main";

/// GPU-side expansion factors (vertices per instanced draw), mirrored by the record-time
/// draw calls. Every topology lowers to triangles so line_width_px is honored uniformly.
inline constexpr uint32_t kHolovizGeometryLineVertices = 6;    // 1 square-capped quad per segment
inline constexpr uint32_t kHolovizGeometryCrossVertices = 12;  // 2 arm quads per cross
inline constexpr uint32_t kHolovizGeometryRectVertices = 24;   // 4 side quads (mitered ring)
inline constexpr uint32_t kHolovizGeometryOvalSegments = 32;
inline constexpr uint32_t kHolovizGeometryOvalVertices = kHolovizGeometryOvalSegments * 6;

/// The push block below, as reflected (float4 + float2 + float + uint, std140).
inline constexpr uint32_t kHolovizGeometryPushBytes = 32;

}  // namespace holoscan::visualization::backends::detail
