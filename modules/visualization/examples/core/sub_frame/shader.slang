// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "shader_interface.hpp"

Texture2D frameTexture;
SamplerState frameSampler;

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

struct VertexOutput {
    float4 position : SV_POSITION;
    float2 uv_coord : TEXCOORD0;
};

// Fullscreen triangle from SV_VertexID — no vertex buffers. The scissor rect set by
// the example clips the draw to one sub-frame band while the UVs stay full-frame.
[shader("vertex")]
VertexOutput vertex_main(uint vertex_id : SV_VertexID) {
    float2 position = float2(vertex_id == 1 ? 3.0 : -1.0, vertex_id == 2 ? 3.0 : -1.0);

    VertexOutput output;
    output.position = float4(position, 0.0, 1.0);
    output.uv_coord = position * 0.5 + 0.5;
    return output;
}

// The scissor only bounds which destination pixels are written; the sampler footprint is
// bounded here. At a band edge a full-frame UV sits between the band's last source row
// and the next band's first, and linear filtering would blend in that neighbouring row —
// which belongs to a band CUDA may still be uploading (or has not started) this frame.
// Clamping the UV to the band's texel-centre range keeps every read inside rows this
// draw's upload wrote.
[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_TARGET {
    float2 uv = input.uv_coord;
    uv.y = clamp(uv.y, pc.uv_min_y, pc.uv_max_y);
    return frameTexture.Sample(frameSampler, uv);
}
