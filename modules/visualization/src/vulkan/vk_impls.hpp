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

// volk.h must come first: it defines VK_NO_PROTOTYPES for everything below
#include <vk_mem_alloc.h>
