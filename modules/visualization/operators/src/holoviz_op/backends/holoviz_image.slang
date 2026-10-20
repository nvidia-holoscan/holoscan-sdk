// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The quad's corners come from SV_VertexID (same table discipline as the monospace text
// shader); positions map the unit quad through the layer's ViewTransform in normalized
// [0,1] framebuffer coordinates, then to y-down clip space. uv_scale confines sampling to
// the valid region of the slot texture (frames smaller than the declared max occupy only
// its top-left); the fragment shader clamps the interpolated UV to the last valid texel
// CENTER (uv_max = (extent - 0.5)/max) so linear filtering never blends in stale texels
// beyond the frame edge, while interior pixels keep the exact 1:1 mapping.

// PushConstants and its flag bits are defined once, shared with holoviz_image_layer.cpp.
#include "holoviz_image_interface.hpp"

Texture2D sourceTexture;
Texture2D chromaTexture;
SamplerState sourceSampler;

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

struct VertexInput {
    uint vertex_id : SV_VertexID;
};

struct VertexOutput {
    float4 position : SV_POSITION;
    float2 uv_coord : TEXCOORD0;
};

static const float2 kCorners[kImageQuadVertices] = {
    float2(0.0, 0.0), float2(1.0, 0.0), float2(1.0, 1.0),
    float2(0.0, 0.0), float2(1.0, 1.0), float2(0.0, 1.0)
};

[shader("vertex")]
VertexOutput vertex_main(VertexInput input) {
    float2 corner = kCorners[input.vertex_id];
    float2 pos_norm = pc.view_offset + corner * pc.view_scale;
    VertexOutput output;
    output.position = float4(pos_norm * 2.0 - 1.0, 0.0, 1.0);
    output.uv_coord = corner * pc.uv_scale;
    return output;
}

// Decode luma before filtering: neighboring Y values occupy different packed channels.
float packed_luma(int2 pixel) {
    pixel = clamp(pixel, int2(0), int2(pc.frame_size) - 1);
    float4 pair = sourceTexture.Load(int3(pixel.x / 2, pixel.y, 0));
    return (pixel.x & 1) == 0 ? pair.r : pair.b;
}

float3 convert_yuv(float y, float2 uv) {
    y = (y - pc.yuv_range.x) * pc.yuv_range.y;
    uv = (uv - pc.yuv_range.z) * pc.yuv_range.w;
    return saturate(float3(y + pc.yuv_coefficients.x * uv.y,
                           y + dot(pc.yuv_coefficients.zw, uv),
                           y + pc.yuv_coefficients.y * uv.x));
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_TARGET {
    if ((pc.flags & kImageFlagNV12) != 0u) {
        float y = sourceTexture.Sample(sourceSampler, min(input.uv_coord, pc.uv_max)).r;
        // Centered chroma: normalized coordinates coincide, but each plane has its own edge.
        float2 chroma_max = (pc.frame_size - 1.0) / pc.max_size;
        float2 uv = chromaTexture.Sample(sourceSampler, min(input.uv_coord, chroma_max)).rg;
        return float4(convert_yuv(y, uv), 1.0);
    }
    if ((pc.flags & kImageFlagYUYV422) != 0u) {
        float2 position = clamp(input.uv_coord * pc.max_size - 0.5,
                                float2(0), pc.frame_size - 1.0);
        int2 base = int2(floor(position));
        float2 weight = frac(position);
        float y = lerp(lerp(packed_luma(base), packed_luma(base + int2(1, 0)), weight.x),
                       lerp(packed_luma(base + int2(0, 1)),
                            packed_luma(base + int2(1, 1)), weight.x), weight.y);
        float2 chroma_max = (pc.frame_size - float2(1.0, 0.5)) / pc.max_size;
        float2 uv = sourceTexture.Sample(sourceSampler, min(input.uv_coord, chroma_max)).ga;
        return float4(convert_yuv(y, uv), 1.0);
    }
    float4 texel = sourceTexture.Sample(sourceSampler, min(input.uv_coord, pc.uv_max));
    if ((pc.flags & kImageFlagGrayscaleReplicate) != 0u) {
        texel = float4(texel.rrr, 1.0);
    }
    if ((pc.flags & kImageFlagSwizzleBGRA) != 0u) {
        texel = texel.bgra;
    }
    return texel;
}
