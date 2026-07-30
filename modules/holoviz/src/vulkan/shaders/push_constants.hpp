/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#define SAMPLE_BINDING_COLOR_S 2
#define SAMPLE_BINDING_LUT 3
#define SAMPLE_BINDING_DEPTH 4

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
#define PUSH_CONSTANT_FRAGMENT_FLAG_LUT_S 8
#define PUSH_CONSTANT_FRAGMENT_FLAG_DEPTH 16

#ifdef __cplusplus
static_assert(sizeof(PushConstantFragment) == PUSH_CONSTANT_FRAGMENT_SIZE);
#endif

#endif /* HOLOVIZ_SRC_VULKAN_SHADERS_PUSH_CONSTANTS_H */
