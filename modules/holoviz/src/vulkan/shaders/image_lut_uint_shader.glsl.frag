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
layout(location = 0) in vec2 i_texCoord;

// outgoing
layout(location = 0) out vec4 o_color;

// sampler
layout(binding = 0) uniform usampler2D texSampler;
layout(binding = 1) uniform sampler2D texSamplerLUT;

layout(push_constant) uniform constants
{
    layout(offset = 21 * 4) float opacity;
} pushConstants;

void main()
{
    const uint index = texture(texSampler, i_texCoord).x;
    o_color = textureLod(texSamplerLUT, vec2(float(index), 0.f), 0);
    o_color.a *= pushConstants.opacity;
}
