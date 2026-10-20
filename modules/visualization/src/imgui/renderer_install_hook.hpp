// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Sanctioned test seam for Renderer::install_input_callbacks() (the vk_instance_init_hook.hpp
// pattern): lets a test hold an installation open between the window registration and the
// renderer's cached window state, to overlap another thread's event dispatch with that gap
// deterministically. Production code never sets it. ImGui- and GLFW-header-free so test
// targets can include it directly.

#pragma once

#include <holoviz/holoviz_export.h>

namespace holoviz::imgui::detail {

using InputInstallHook = void (*)();

/// Run \p hook inside install_input_callbacks() after the window registration succeeds and
/// before the renderer publishes its cached window state, with the window-system lock held.
/// nullptr (the default) disables it.
HOLOVIZ_EXPORT void set_input_install_hook(InputInstallHook hook);

}  // namespace holoviz::imgui::detail
