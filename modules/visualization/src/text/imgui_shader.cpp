/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "text/imgui_shader.hpp"

namespace holoviz::text::detail {

// Vertex layout mirrors ImDrawVert (20 bytes): float2 pos, float2 uv, packed
// RGBA8 color. The color is declared uint (kR32Uint, 4 bytes) and unpacked in
// the vertex shader so the reflected stride matches ImDrawVert exactly — a
// float4 declaration would reflect as 16 bytes. Coordinates are ImGui screen
// pixels; scale/translate push constants map them to clip space (Vulkan's
// y-down framebuffer matches ImGui's y-down convention, no flip needed).
const char* const kImGuiSlangSource = R"(
Texture2D fontTexture;
SamplerState fontSampler;

struct PushConstants {
    float2 scale;
    float2 translate;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

struct VertexInput {
    float2 position : POSITION;
    float2 uv_coord : TEXCOORD0;
    uint color : COLOR0;
};

struct VertexOutput {
    float4 position : SV_POSITION;
    float2 uv_coord : TEXCOORD0;
    float4 color : COLOR0;
};

[shader("vertex")]
VertexOutput vertex_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position * pc.scale + pc.translate, 0.0, 1.0);
    output.uv_coord = input.uv_coord;
    // Unpack IM_COL32 (red in the lowest byte) to float4.
    output.color = float4(
        float(input.color & 0xFFu),
        float((input.color >> 8) & 0xFFu),
        float((input.color >> 16) & 0xFFu),
        float((input.color >> 24) & 0xFFu)) / 255.0;
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_TARGET {
    return input.color * fontTexture.Sample(fontSampler, input.uv_coord);
}
)";

}  // namespace holoviz::text::detail
