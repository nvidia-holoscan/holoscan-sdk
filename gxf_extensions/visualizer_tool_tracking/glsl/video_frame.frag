#version 450

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

in vec2 tex_coords;
layout(binding = 0) uniform sampler2D background;
layout(location = 0) out vec4 out_color;

void main() {
  vec3 rgb = texture2D(background, tex_coords).rgb;
  out_color = vec4(rgb, 1.0);
};
