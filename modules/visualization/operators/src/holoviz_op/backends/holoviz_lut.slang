// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The mask must be sampled with a nearest filter (bound by the layer's descriptor): index
// values must never interpolate across segment boundaries. A kR8Unorm texel reads back as
// value/255, so the index is recovered by scaling and rounding, then clamped into the
// declared palette.

// PushConstants and the palette limit are defined once, shared with holoviz_lut_layer.cpp.
#include "holoviz_lut_interface.hpp"

Texture2D maskTexture;
SamplerState maskSampler;
StructuredBuffer<float4> lutTable;

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

struct VertexInput {
    uint vertex_id : SV_VertexID;
};

struct VertexOutput {
    float4 position : SV_POSITION;
    float2 uv_coord : TEXCOORD0;
};

static const float2 kCorners[kLutQuadVertices] = {
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
    float value = maskTexture.Sample(maskSampler, min(input.uv_coord, pc.uv_max)).r;
    uint index = min(uint(value * float(kLutMaxSize - 1u) + 0.5), pc.lut_size - 1u);
    return lutTable[index];
}
