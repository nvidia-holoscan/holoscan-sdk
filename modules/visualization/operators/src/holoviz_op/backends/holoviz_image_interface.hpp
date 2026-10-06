/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared between holoviz_image.slang and holoviz_image_layer.cpp: the push-constant block and
// its flag bits. See <holoviz/shader_interop.hpp> for the rules.
// The namespace is the one the generated holoviz_image.slang.hpp opens (NAMESPACE in
// operators/CMakeLists.txt), and no other shader's interface: each shader's PushConstants
// is a distinct C++ type.

#pragma once

#include <holoviz/shader_interop.hpp>

HOLOVIZ_SHADER_INTERFACE_BEGIN(holoscan::visualization::backends::detail::holoviz_image)

/// The layer draws one full quad of two triangles.
HOLOVIZ_SHADER_CONSTANT uint kImageQuadVertices = 6u;

/// PushConstants::flags bits.
HOLOVIZ_SHADER_CONSTANT uint kImageFlagGrayscaleReplicate = 1u;  // single-channel (R8/R16) source
HOLOVIZ_SHADER_CONSTANT uint kImageFlagSwizzleBGRA = 2u;         // source bytes are B, G, R, A
HOLOVIZ_SHADER_CONSTANT uint kImageFlagNV12 = 4u;                // Y in source, CbCr in chroma
HOLOVIZ_SHADER_CONSTANT uint kImageFlagYUYV422 = 8u;             // packed Y0 Cb Y1 Cr in source

struct PushConstants {
  float2 view_offset;  // normalized top-left of the layer's view rect
  float2 view_scale;   // normalized size of the view rect
  float2 uv_scale;     // valid region of the slot texture (frame size / declared max)
  float2 uv_max;       // (frame extent - 0.5) / declared max (last valid texel center)
  uint flags;          // kImageFlag* bits
  uint pad;
  uint2 alignment_pad;      // std140: the float4s below start on a 16-byte boundary
  float4 yuv_coefficients;  // R/Cr, B/Cb, G/Cb, G/Cr
  float4 yuv_range;         // Y offset/scale, chroma offset/scale
  float2 frame_size;        // frame extent in texels
  float2 max_size;          // declared max extent in texels
};

HOLOVIZ_SHADER_INTERFACE_END
