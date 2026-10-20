/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared between imgui/shader.slang and imgui/renderer.cpp: the one definition of the
// push-constant block both sides use. See <holoviz/shader_interop.hpp> for the rules.

#pragma once

#include <holoviz/shader_interop.hpp>

HOLOVIZ_SHADER_INTERFACE_BEGIN(holoviz::imgui::detail)

/// Maps ImGui display coordinates (pixels, y-down, origin at DisplayPos) to clip space:
/// clip = position * scale + translate.
struct PushConstants {
  float2 scale;
  float2 translate;
};

HOLOVIZ_SHADER_INTERFACE_END
