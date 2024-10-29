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
