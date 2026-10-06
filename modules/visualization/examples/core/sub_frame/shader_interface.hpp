/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared between shader.slang and Main.cpp: the per-band push constants. Including one header
// from both languages keeps the CPU and GPU views of the data in sync; see
// <holoviz/shader_interop.hpp> for the rules and the build-time layout checks it enables.

#pragma once

#include <holoviz/shader_interop.hpp>

HOLOVIZ_SHADER_INTERFACE_BEGIN(holoviz::examples::sub_frame)

/// Vertical sampling bounds of one band draw, in source-texture UV units at texel centres:
/// [ (y0 + 0.5) / height, (y1 - 0.5) / height ] for the band's source rows [y0, y1). The
/// fragment shader clamps its UV to this range so the linear sampler never touches a row
/// outside the band — the scissor bounds the destination pixels, but a full-frame UV at a
/// band edge would still blend in the neighbouring band's rows, which CUDA may not have
/// uploaded yet for this frame.
struct PushConstants {
  float uv_min_y;
  float uv_max_y;
};

HOLOVIZ_SHADER_INTERFACE_END
