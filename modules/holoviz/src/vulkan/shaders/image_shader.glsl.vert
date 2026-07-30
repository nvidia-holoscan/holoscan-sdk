/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#version 450
#extension GL_GOOGLE_include_directive : require

#include "push_constants.hpp"

// incoming
layout(location = 0) in vec2 i_position;

// outgoing
layout(location = 0) out vec2 o_texCoord;

// constants
layout(push_constant) uniform constants {
    PushConstantVertex vertex;
} push_constants;

void main()
{
    gl_Position = push_constants.vertex.matrix * vec4(i_position, 0.0, 1.0);
    o_texCoord  = (i_position + vec2(1.F)) * vec2(0.5F);
}
