/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared between holoviz_lut.slang and holoviz_lut_layer.cpp: the push-constant block and the
// palette size a kR8Unorm mask can index. See <holoviz/shader_interop.hpp> for the rules.
// The namespace is the one the generated holoviz_lut.slang.hpp opens (NAMESPACE in
// operators/CMakeLists.txt), and no other shader's interface: each shader's PushConstants
// is a distinct C++ type.

#pragma once

#include <holoviz/shader_interop.hpp>

HOLOVIZ_SHADER_INTERFACE_BEGIN(holoscan::visualization::backends::detail::holoviz_lut)

/// The layer draws one full quad of two triangles.
HOLOVIZ_SHADER_CONSTANT uint kLutQuadVertices = 6u;

/// A kR8Unorm mask texel reads back as value / (kLutMaxSize - 1), so it can address at most
/// this many palette entries; the layer rejects larger lut_size values at allocation.
HOLOVIZ_SHADER_CONSTANT uint kLutMaxSize = 256u;

struct PushConstants {
  float2 view_offset;
  float2 view_scale;
  float2 uv_scale;  // valid region of the slot texture (frame size / declared max)
  float2 uv_max;    // (frame extent - 0.5) / declared max (last valid texel center)
  uint lut_size;
  uint pad;
};

HOLOVIZ_SHADER_INTERFACE_END
