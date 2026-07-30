/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#version 450
#extension GL_GOOGLE_include_directive : require

#include "push_constants.hpp"

// incoming
layout(location = 0) in vec4 i_color;

// outgoing
layout(location = 0) out vec4 o_color;

// constants
layout(push_constant) uniform constants {
  layout(offset = PUSH_CONSTANT_VERTEX_SIZE) PushConstantFragment fragment;
} push_constants;

void main()
{
    o_color = i_color;
    o_color.a *= push_constants.fragment.opacity;
}
