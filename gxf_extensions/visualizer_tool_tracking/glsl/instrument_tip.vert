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

layout(location = 0) in vec2 position;
layout(location = 1) in float confidence;

flat out float size;
flat out float linewidth;
flat out int instrument_idx;

void main(void) {
  // consts
  const float M_SQRT_2 = 1.4142135623730951;
  size = 30.0;
  linewidth = 4.0;

  vec2 pos = position.xy;
  // change direction of y coordinate
  pos.y = 1.0f - pos.y;
  // transform to NDC space with (0,0) in center and coordinate range [-1, 1 ] for x and y
  pos.x = 2.0f * pos.x - 1.0f;
  pos.y = 2.0f * pos.y - 1.0f;

  gl_Position = vec4(pos, 0.0, 1.0);
  instrument_idx = gl_VertexID;

  float p_size = (confidence > 0.5) ? M_SQRT_2 * size : 0.0;

  gl_PointSize = p_size;
}
