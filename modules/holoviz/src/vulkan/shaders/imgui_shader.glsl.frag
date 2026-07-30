/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#version 450
#extension GL_GOOGLE_include_directive : require

#include "push_constants.hpp"

// incoming
layout(location = 0) in struct
{
    vec4 color;
    vec2 texCoord;
} In;

// outgoing
layout(location = 0) out vec4 o_color;

// sampler
layout(binding = SAMPLE_BINDING_COLOR) uniform sampler2D texSampler;

// constants
layout(push_constant) uniform constants {
  layout(offset = PUSH_CONSTANT_VERTEX_SIZE) PushConstantFragment fragment;
} push_constants;

void main()
{
    vec4 color = In.color * texture(texSampler, In.texCoord);
    color.a *= push_constants.fragment.opacity;

    // discard transparent fragments
    if (color.a == 0.F)
        discard;

    o_color = color;
}
