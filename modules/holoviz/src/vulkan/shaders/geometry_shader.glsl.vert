/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#version 450
#extension GL_GOOGLE_include_directive : require

#include "push_constants.hpp"

// incoming
layout(location = 0) in vec3 i_position;

// outgoing
layout(location = 0) out vec4 o_color;

// constants
layout(push_constant) uniform constants {
    PushConstantVertex vertex;
} push_constants;

void main()
{
  gl_PointSize = push_constants.vertex.point_size;
  o_color = push_constants.vertex.color;

  gl_Position  = push_constants.vertex.matrix * vec4(i_position, 1.0);
}
