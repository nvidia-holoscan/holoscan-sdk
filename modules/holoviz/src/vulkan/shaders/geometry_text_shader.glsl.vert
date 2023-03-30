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

// incoming
layout(location = 0) in vec3 i_position;
layout(location = 1) in vec2 i_texCoord;
layout(location = 2) in vec4 i_color;

// outgoing
layout(location = 0) out struct
{
    vec4 color;
    vec2 texCoord;
} Out;

layout(push_constant) uniform constants
{
    mat4x4 matrix;
} pushConstants;

void main()
{
    Out.color    = i_color;
    Out.texCoord = i_texCoord;

    gl_Position = pushConstants.matrix * vec4(i_position, 1.0);
}