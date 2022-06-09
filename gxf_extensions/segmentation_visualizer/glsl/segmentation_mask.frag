#version 460

/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

in vec2 tex_coord;

// Note: MAX_LUT_COLORS should match the value used by the host code using this shader.
const int MAX_LUT_COLORS = 64;
layout(location = 0) uniform uint lut_count;
layout(location = 1) uniform vec4 lut_colors[MAX_LUT_COLORS];

layout(binding = 0) uniform sampler2D image_texture;
layout(binding = 1) uniform usampler2D class_texure;

out vec4 out_color;

void main() {
  uint class_index = texture(class_texure, tex_coord).r;
  // Set maximum index to last index that can be used as an "other" class.
  class_index = min(class_index, lut_count - 1);
  vec4 class_color = lut_colors[class_index];
  vec4 image_color = texture(image_texture, tex_coord);

  vec4 dst = vec4(0.0);
  dst = (1.0 - image_color.a) * dst + image_color.a * image_color;
  dst = (1.0 - class_color.a) * dst + class_color.a * class_color;
  dst.a = 1.0;
  out_color = dst;
};
