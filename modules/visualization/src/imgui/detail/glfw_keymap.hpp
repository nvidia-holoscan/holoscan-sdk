/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoviz/imgui/vendor/imgui.h>

namespace holoviz::imgui::detail {

/// Map a GLFW_KEY_* code to the corresponding ImGuiKey (ImGuiKey_None for anything
/// unmapped). Mirrors imgui_impl_glfw for ImGui 1.89.3. Its own translation unit so the
/// table is host-testable without a window system.
[[nodiscard]] ImGuiKey glfw_key_to_imgui_key(int key);

}  // namespace holoviz::imgui::detail
