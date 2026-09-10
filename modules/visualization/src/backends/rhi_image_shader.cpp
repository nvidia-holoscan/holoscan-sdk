// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "backends/rhi_image_shader.hpp"

namespace holoviz::backends::detail {

// The quad's corners come from SV_VertexID (same table discipline as the monospace text
// shader); positions map the unit quad through the layer's ViewTransform in normalized
// [0,1] framebuffer coordinates, then to y-down clip space. uv_scale confines sampling to
// the valid region of the slot texture (frames smaller than the declared max occupy only
// its top-left); the fragment shader clamps the interpolated UV to the last valid texel
// CENTER (uv_max = (extent - 0.5)/max) so linear filtering never blends in stale texels
// beyond the frame edge, while interior pixels keep the exact 1:1 mapping.
const char* const kRhiImageSlangSource = R"(
Texture2D sourceTexture;
SamplerState sourceSampler;

struct PushConstants {
    float2 view_offset;   // normalized top-left of the layer's view rect
    float2 view_scale;    // normalized size of the view rect
    float2 uv_scale;      // valid region of the slot texture (frame size / declared max)
    float2 uv_max;        // (frame extent - 0.5) / declared max (last valid texel center)
    uint   flags;         // kRhiImageFlagGrayscaleReplicate: single-channel source
    uint   pad;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

struct VertexInput {
    uint vertex_id : SV_VertexID;
};

struct VertexOutput {
    float4 position : SV_POSITION;
    float2 uv_coord : TEXCOORD0;
};

static const float2 kCorners[6] = {
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

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_TARGET {
    float4 texel = sourceTexture.Sample(sourceSampler, min(input.uv_coord, pc.uv_max));
    if ((pc.flags & 1u) != 0u) {
        texel = float4(texel.rrr, 1.0);
    }
    return texel;
}
)";

}  // namespace holoviz::backends::detail
