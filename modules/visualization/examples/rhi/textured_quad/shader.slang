// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

Texture2D myTexture;
SamplerState mySampler;

struct VertexInput {
    float2 position : POSITION;
    float2 uv_coord: TEXCOORD0;
};

struct VertexOutput {
    float4 position : SV_POSITION;
    float2 uv_coord: TEXCOORD0;
};

struct RotationData {
    float angle; // rotation in radians
};

StructuredBuffer<RotationData> rotation;

struct PushConstants {
    int rotation_index;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[shader("vertex")]
VertexOutput vertex_main(VertexInput input) {
    VertexOutput output;

    float s = sin(rotation[pc.rotation_index].angle);
    float c = cos(rotation[pc.rotation_index].angle);

    float2 rotated = float2(
        input.position.x * c - input.position.y * s,
        input.position.x * s + input.position.y * c
    );

    output.position = float4(rotated, 0.0, 1.0);
    output.uv_coord = input.uv_coord;
    return output;
}

struct FragmentInput {
    float4 position : SV_POSITION;
    float2 uv_coord: TEXCOORD0;
};

[shader("fragment")]
float4 fragment_main(FragmentInput input) : SV_TARGET {
    return myTexture.Sample(mySampler, input.uv_coord);
}
