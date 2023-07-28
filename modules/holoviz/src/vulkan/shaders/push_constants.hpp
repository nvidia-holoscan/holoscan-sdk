/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOVIZ_SRC_VULKAN_SHADERS_PUSH_CONSTANTS_H
#define HOLOVIZ_SRC_VULKAN_SHADERS_PUSH_CONSTANTS_H

#ifdef __cplusplus
// define GLSL types when using the header in C++ code
using vec4 = std::array<float, 4>;
using mat4x4 = nvmath::mat4f;
#endif

#define SAMPLE_BINDING_COLOR 0
#define SAMPLE_BINDING_COLOR_U 1
#define SAMPLE_BINDING_LUT 2
#define SAMPLE_BINDING_DEPTH 3

struct PushConstantVertex {
  mat4x4 matrix;
  vec4 color;
  float point_size;
};

#define PUSH_CONSTANT_VERTEX_SIZE ((16 * 4) + (4 * 4) + 4)

#ifdef __cplusplus
static_assert(sizeof(PushConstantVertex) == PUSH_CONSTANT_VERTEX_SIZE);
#endif

struct PushConstantFragment {
  float opacity;
  uint flags;
};

#define PUSH_CONSTANT_FRAGMENT_SIZE (4 + 4)

#define PUSH_CONSTANT_FRAGMENT_FLAG_COLOR 1
#define PUSH_CONSTANT_FRAGMENT_FLAG_LUT 2
#define PUSH_CONSTANT_FRAGMENT_FLAG_LUT_U 4
#define PUSH_CONSTANT_FRAGMENT_FLAG_DEPTH 8

#ifdef __cplusplus
static_assert(sizeof(PushConstantFragment) == PUSH_CONSTANT_FRAGMENT_SIZE);
#endif

#endif /* HOLOVIZ_SRC_VULKAN_SHADERS_PUSH_CONSTANTS_H */
