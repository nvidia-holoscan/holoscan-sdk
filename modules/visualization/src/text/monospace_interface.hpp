/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared between text/monospace.slang and text/monospace_text_renderer.cpp: the push-constant
// block, the glyph-table entry, and every constant the char packing and glyph lookup depend on.
// See <holoviz/shader_interop.hpp> for the rules.

#pragma once

#include <holoviz/shader_interop.hpp>

HOLOVIZ_SHADER_INTERFACE_BEGIN(holoviz::text::detail)

/// Corners per glyph quad (two triangles); fill-rectangle mode draws only the first triangle.
HOLOVIZ_SHADER_CONSTANT uint kVerticesPerGlyph = 6u;
HOLOVIZ_SHADER_CONSTANT uint kFillRectangleVerticesPerGlyph = 3u;

/// ASCII codes the glyph table covers; anything else renders as kFallbackGlyphCode ('?').
/// Table slot 0 is a metrics header, so code c lives at index c - kFirstGlyphCode + 1.
HOLOVIZ_SHADER_CONSTANT uint kFirstGlyphCode = 32u;
HOLOVIZ_SHADER_CONSTANT uint kLastGlyphCode = 127u;
HOLOVIZ_SHADER_CONSTANT uint kFallbackGlyphCode = 63u;
HOLOVIZ_SHADER_CONSTANT uint kGlyphTableEntries = kLastGlyphCode - kFirstGlyphCode + 2u;

/// Characters are packed four per 32-bit word, little-endian, into uint4 words.
HOLOVIZ_SHADER_CONSTANT uint kCharsPerWord = 4u;
HOLOVIZ_SHADER_CONSTANT uint kPackedCharWords = 14u;
HOLOVIZ_SHADER_CONSTANT uint kMaxCharsPerDraw = kPackedCharWords * 4u * kCharsPerWord;  // 224

/// One glyph-table entry (StructuredBuffer element). [0] is a metrics header: uv.x is the
/// monospace advance in bake px, uv.y the bake font size in px. Quad corners are bake px
/// relative to the top-left pen (ascent pre-baked by the atlas builder).
struct GlyphInfo {
  float4 uv;    // u0, v0, u1, v1 (normalized atlas coordinates)
  float4 quad;  // x0, y0, x1, y1 (bake px, relative to the top-left pen)
};

/// std140 push block: 24-byte header, 8 bytes of alignment padding the uint4 array
/// introduces, then 14 uint4 words = 256 bytes exactly (the NVIDIA push-constant budget this
/// renderer assumes; validated at creation).
struct PushConstants {
  float2 origin_px;                      // top-left of the run, framebuffer pixels
  float2 clip_scale;                     // 2/fb_width, 2/fb_height
  float glyph_height_px;                 // rendered glyph height; scale = this / bake size
  uint rgba;                             // packed color, red in the lowest byte
  uint4 packed_chars[kPackedCharWords];  // 224 ASCII bytes, little-endian within each uint
};

HOLOVIZ_SHADER_INTERFACE_END
