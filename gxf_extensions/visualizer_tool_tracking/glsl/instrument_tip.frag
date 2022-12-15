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
// Note: MAX_TOOLS should match the value used by the host code using this shader.
const int MAX_TOOLS = 64;

layout(location = 0) uniform vec3 tool_colors[MAX_TOOLS];

flat in float size;
flat in float linewidth;
flat in int instrument_idx;

layout(location = 0) out vec4 out_color;

void main() {
  vec2 P = gl_PointCoord.xy - vec2(0.5, 0.5);
  P *= size;

  if (abs(P.x) > 0.5 * linewidth && abs(P.y) > 0.5 * linewidth) { discard; }

  out_color = vec4(tool_colors[instrument_idx], 1.0);
}
