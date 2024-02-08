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
layout(location = 0) in vec2 i_texCoord;

in vec4 gl_FragCoord;

// outgoing
layout(location = 0) out vec4 o_color;

layout(depth_less) out float gl_FragDepth;

// sampler
layout(binding = SAMPLE_BINDING_COLOR) uniform sampler2D colorSampler;
layout(binding = SAMPLE_BINDING_COLOR_U) uniform usampler2D coloruSampler;
layout(binding = SAMPLE_BINDING_COLOR_S) uniform isampler2D coloriSampler;
layout(binding = SAMPLE_BINDING_LUT) uniform sampler2D lutSampler;
layout(binding = SAMPLE_BINDING_DEPTH) uniform sampler2D depthSampler;

// constants
layout(push_constant) uniform constants {
  layout(offset = PUSH_CONSTANT_VERTEX_SIZE) PushConstantFragment fragment;
} push_constants;

void main()
{
  vec4 color;
  if ((push_constants.fragment.flags & PUSH_CONSTANT_FRAGMENT_FLAG_COLOR) != 0) {
    color = texture(colorSampler, i_texCoord);
  } else {
    if ((push_constants.fragment.flags & PUSH_CONSTANT_FRAGMENT_FLAG_LUT) != 0) {
      const float index = texture(colorSampler, i_texCoord).x;
      color = textureLod(lutSampler, vec2(index, 0.f), 0);
    } else if ((push_constants.fragment.flags & PUSH_CONSTANT_FRAGMENT_FLAG_LUT_U) != 0) {
      const uint index = texture(coloruSampler, i_texCoord).x;
      color = textureLod(lutSampler, vec2(float(index), 0.f), 0);
    } else if ((push_constants.fragment.flags & PUSH_CONSTANT_FRAGMENT_FLAG_LUT_S) != 0) {
      const uint index = texture(coloriSampler, i_texCoord).x;
      color = textureLod(lutSampler, vec2(float(index), 0.f), 0);
    }
  }

  color.a *= push_constants.fragment.opacity;

  // discard transparent fragments
  if (color.a == 0.f)
      discard;

  o_color = color;

  // write depth
  float depth;
  if ((push_constants.fragment.flags & PUSH_CONSTANT_FRAGMENT_FLAG_DEPTH) != 0) {
    depth = texture(depthSampler, i_texCoord).r;
  } else {
    depth = gl_FragCoord.z;
  }
  gl_FragDepth = depth;
}
