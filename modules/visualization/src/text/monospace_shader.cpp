/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "text/monospace_shader.hpp"

namespace holoviz::text::detail {

// Vertex-buffer-less instanced text: draw(6, char_count) per string run. The
// vertex shader derives each glyph quad from SV_VertexID/SV_InstanceID (the
// input struct reflects zero vertex attributes, so no vertex buffer is ever
// bound), unpacks the instance's ASCII code from the push-constant words, and
// looks up its atlas UVs and quad corners in a static glyph table uploaded
// once at creation. Push constants use std140/cbuffer layout, so the char
// bytes are packed as uint4 words: 24-byte header + 8 pad + uint4[14] = 256
// bytes exactly — the NVIDIA push-constant budget this renderer assumes
// (validated at creation). Coordinates are framebuffer pixels mapped to clip
// space by clip_scale; Vulkan's y-down framebuffer matches, no flip.
const char* const kMonospaceSlangSource = R"(
Texture2D fontTexture;
SamplerState fontSampler;

// [0] is a metrics header (uv.x = monospace advance in bake px, uv.y = bake
// font size px); [1..96] are ASCII 32..127. quad corners are bake px relative
// to the top-left pen (ascent pre-baked by the atlas builder).
struct GlyphInfo {
    float4 uv;    // u0, v0, u1, v1 (normalized atlas coordinates)
    float4 quad;  // x0, y0, x1, y1 (bake px, relative to the top-left pen)
};
StructuredBuffer<GlyphInfo> glyphTable;

struct PushConstants {
    float2 origin_px;        // top-left of the run, framebuffer pixels
    float2 clip_scale;       // 2/fb_width, 2/fb_height
    float  glyph_height_px;  // rendered glyph height; scale = this / bake size
    uint   rgba;             // packed color, red in the lowest byte
    uint4  packed_chars[14]; // 224 ASCII bytes, little-endian within each uint
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// System values only: reflection skips SV_ fields, so the pipeline has zero
// vertex attributes.
struct VertexInput {
    uint vertex_id : SV_VertexID;
    uint instance_id : SV_InstanceID;
};

struct VertexOutput {
    float4 position : SV_POSITION;
    float2 uv_coord : TEXCOORD0;
    float4 color : COLOR0;
};

// The first three corners form a triangle whose axis-aligned bounding rect is
// the whole unit quad: fill-rectangle mode (VK_NV_fill_rectangle) draws only
// those three vertices and rasterizes the rect, with the affine UV/position
// plane equations extrapolating exactly to the fourth corner. Keep that
// triangle first when editing this table.
static const float2 kCorners[6] = {
    float2(0.0, 0.0), float2(1.0, 0.0), float2(1.0, 1.0),
    float2(0.0, 0.0), float2(1.0, 1.0), float2(0.0, 1.0)
};

[shader("vertex")]
VertexOutput vertex_main(VertexInput input) {
    GlyphInfo metrics = glyphTable[0];
    float scale = pc.glyph_height_px / metrics.uv.y;

    uint word_index = input.instance_id >> 2;
    uint word = pc.packed_chars[word_index >> 2][word_index & 3];
    uint code = (word >> ((input.instance_id & 3) * 8)) & 0xFFu;
    if (code < 32u || code > 127u) {
        code = 63u;  // '?'
    }
    GlyphInfo glyph = glyphTable[code - 32u + 1u];

    float2 corner = kCorners[input.vertex_id];
    float2 pen = pc.origin_px
               + float2(float(input.instance_id) * metrics.uv.x * scale, 0.0);
    float2 pos_px = pen + lerp(glyph.quad.xy, glyph.quad.zw, corner) * scale;

    VertexOutput output;
    output.position = float4(pos_px * pc.clip_scale - 1.0, 0.0, 1.0);
    output.uv_coord = lerp(glyph.uv.xy, glyph.uv.zw, corner);
    output.color = float4(
        float(pc.rgba & 0xFFu),
        float((pc.rgba >> 8) & 0xFFu),
        float((pc.rgba >> 16) & 0xFFu),
        float((pc.rgba >> 24) & 0xFFu)) / 255.0;
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_TARGET {
    return input.color * fontTexture.Sample(fontSampler, input.uv_coord);
}
)";

}  // namespace holoviz::text::detail
