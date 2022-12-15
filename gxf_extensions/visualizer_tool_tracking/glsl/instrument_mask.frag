#version 450

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

// Arrays in a UBO must use a constant expression for their size.
// Note: MAX_LAYERS should match the value used by the host code using this shader.
const int MAX_LAYERS = 64;

in vec2 tex_coords;

layout(binding = 0) uniform sampler2DArray mask;

layout(location = 0) uniform uint num_layers;
layout(location = 1) uniform vec3 layer_colors[MAX_LAYERS];

layout(std430, binding = 0) buffer ConfidenceBlock {
  float confidence[MAX_LAYERS];
};

layout(location = 0) out vec4 out_color;

float threshold(float v, float minV, float maxV) {
  const float range = maxV - minV;
  v = clamp(v, minV, maxV);
  v -= minV;
  v /= range;
  return v;
}

void main() {
  vec4 dst = vec4(0.0);

  const float minV = 0.3;
  const float maxV = 0.99;

  ivec3 tex_size = textureSize(mask, 0);
  vec2 tex_coord_offset = vec2(1.0 / tex_size.xy);

  for (int i = 0; i != num_layers; ++i) {
    if (confidence[i] > 0.5) {
      float s = threshold(texture(mask, vec3(tex_coords - tex_coord_offset, i)).r, minV, maxV);
      vec4 src = vec4(layer_colors[i], s * 0.7);
      dst = (1.0 - src.a) * dst + src.a * src;
    }
  }

  out_color = dst;
};
