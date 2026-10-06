/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared between shader.slang and Main.cpp: the push-constant block and the per-frame rotation
// entry the shader reads from its storage buffer. Including one header from both languages is
// how a HoloViz application keeps its CPU and GPU views of the same data in sync; see
// <holoviz/shader_interop.hpp> for the rules and the build-time checks it enables.

#pragma once

#include <holoviz/shader_interop.hpp>

HOLOVIZ_SHADER_INTERFACE_BEGIN(holoviz::examples::textured_quad)

/// One entry per swapchain image in the `rotation` StructuredBuffer.
struct RotationData {
  float angle;  // rotation in radians
};

struct PushConstants {
  int rotation_index;  // which RotationData entry this frame reads
};

HOLOVIZ_SHADER_INTERFACE_END
