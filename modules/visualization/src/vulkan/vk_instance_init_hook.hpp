// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Sanctioned test seam for the one-time Vulkan instance initialization (the
// vk_swapchain_fault_injection.hpp pattern): lets a test hold the first initializer open at a
// known point to overlap other first-time callers with it deterministically. Production code
// never sets it. Vulkan-header-free so test targets can include it directly.

#pragma once

#include <holoviz/holoviz_export.h>

namespace holoviz::vulkan {

using InstanceInitHook = void (*)();

/// Run \p hook first inside one-time instance initialization, with the window-system lock held
/// and before the initializer takes its GLFW reference. nullptr (the default) disables it. Only
/// the first initialization in a process reaches the hook.
HOLOVIZ_EXPORT void set_instance_init_hook(InstanceInitHook hook);

}  // namespace holoviz::vulkan
