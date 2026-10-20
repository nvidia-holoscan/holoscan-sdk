// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// One binding of each bindable kind, so every bind method has a matching slot and two
// mismatched ones. Compiled at build time by holoviz_generate_shader_header(); the test binds
// against the generated ShaderBlob's reflected interface.

StructuredBuffer<float> values;
Texture2D<float4> image;
SamplerState image_sampler;

struct VSInput {
  float3 position;
};

struct VSOutput {
  float4 position : SV_Position;
};

[shader("vertex")]
VSOutput vertex_main(VSInput input) {
  VSOutput output;
  output.position = float4(input.position, 1.0);
  return output;
}

[shader("fragment")]
float4 fragment_main(VSOutput input) : SV_Target {
  return image.Sample(image_sampler, float2(0.5, 0.5)) * values[0];
}
