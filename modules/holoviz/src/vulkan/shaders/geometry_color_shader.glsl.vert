/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#version 450
#extension GL_GOOGLE_include_directive : require

#include "push_constants.hpp"

// incoming
layout(location = 0) in vec3 i_position;
layout(location = 1) in vec4 i_color;

// outgoing
layout(location = 0) out vec4 o_color;

layout(push_constant) uniform constants
{
    PushConstantVertex vertex;
} pushConstants;

void main()
{
    gl_PointSize = pushConstants.vertex.point_size;
    o_color = i_color;

    gl_Position  = pushConstants.vertex.matrix * vec4(i_position, 1.0);
}
