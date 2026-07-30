/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#version 450
#extension GL_GOOGLE_include_directive : require

#include "push_constants.hpp"

// incoming
layout(location = 0) in vec2 i_position;
layout(location = 1) in vec2 i_texCoord;
layout(location = 2) in vec4 i_color;

// outgoing
layout(location = 0) out struct
{
    vec4 color;
    vec2 texCoord;
} Out;

// constants
layout(push_constant) uniform constants {
    PushConstantVertex vertex;
} push_constants;

void main()
{
    gl_PointSize = push_constants.vertex.point_size;

    Out.color    = i_color;
    Out.texCoord = i_texCoord;

    gl_Position = push_constants.vertex.matrix * vec4(i_position.x, i_position.y, 0.F, 1.0);
}
