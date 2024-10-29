/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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