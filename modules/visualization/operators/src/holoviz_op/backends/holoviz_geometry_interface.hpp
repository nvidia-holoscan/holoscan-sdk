/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared between holoviz_geometry.slang and holoviz_geometry_layer.cpp: the push-constant block,
// the vertex count each primitive expands to, and the float stride of each primitive family in
// the parameter buffer. See <holoviz/shader_interop.hpp> for the rules.
// The namespace is the one the generated holoviz_geometry.slang.hpp opens (NAMESPACE in
// operators/CMakeLists.txt), and no other shader's interface: each shader's PushConstants
// is a distinct C++ type.

#pragma once

#include <holoviz/shader_interop.hpp>

HOLOVIZ_SHADER_INTERFACE_BEGIN(holoscan::visualization::backends::detail::holoviz_geometry)

/// Every primitive lowers to square-capped quads of two triangles each.
HOLOVIZ_SHADER_CONSTANT uint kGeometryQuadVertices = 6u;
HOLOVIZ_SHADER_CONSTANT uint kGeometryLineVertices = kGeometryQuadVertices;  // one quad per segment
HOLOVIZ_SHADER_CONSTANT uint kGeometryCrossVertices = 2u * kGeometryQuadVertices;  // two arms
HOLOVIZ_SHADER_CONSTANT uint kGeometryRectVertices = 4u * kGeometryQuadVertices;   // four sides
HOLOVIZ_SHADER_CONSTANT uint kGeometryOvalSegments = 32u;
HOLOVIZ_SHADER_CONSTANT uint kGeometryOvalVertices = kGeometryOvalSegments * kGeometryQuadVertices;

/// Floats per primitive in the geometryParams storage buffer (the packing contract with
/// frame_data.hpp): what the vertex shader fetches and the host advances by per primitive.
HOLOVIZ_SHADER_CONSTANT uint kGeometryLineListFloats = 4u;   // x0, y0, x1, y1 per segment
HOLOVIZ_SHADER_CONSTANT uint kGeometryLineStripFloats = 2u;  // x, y per point
HOLOVIZ_SHADER_CONSTANT uint kGeometryCrossFloats = 3u;      // cx, cy, size
HOLOVIZ_SHADER_CONSTANT uint kGeometryRectFloats = 4u;       // two opposite corners
HOLOVIZ_SHADER_CONSTANT uint kGeometryOvalFloats = 4u;       // cx, cy, rx, ry

struct PushConstants {
  float4 color;
  float2 viewport_px;
  float line_width_px;
  uint base_float_offset;  // this batch's first float within the bound slot slice
};

HOLOVIZ_SHADER_INTERFACE_END
