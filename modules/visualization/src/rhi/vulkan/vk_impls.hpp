/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#if defined(_WIN32)
#include <windows.h>
// #define VK_USE_PLATFORM_WIN32_KHR
#endif

#include <volk.h>

// volk declares the VK_EXT_acquire_xlib_display entry points itself (with
// self-contained Display/RROutput typedefs), but the real <vulkan/vulkan.h>
// that VMA includes would react to the platform define by dragging in
// <X11/Xlib.h> — whose None/Bool/Status macros break everything downstream.
// volk has what it needs by this point; drop the define for everyone else.
#undef VK_USE_PLATFORM_XLIB_XRANDR_EXT

// volk.h must come first: it defines VK_NO_PROTOTYPES for everything below
#include <vk_mem_alloc.h>
