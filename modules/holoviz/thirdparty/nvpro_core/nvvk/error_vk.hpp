/*
 * SPDX-FileCopyrightText: Copyright (c) 2014-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


//////////////////////////////////////////////////////////////////////////
/**
\fn nvvk::checkResult
\brief Returns true on critical error result, logs errors.

Use `NVVK_CHECK(result)` to automatically log filename/linenumber.
*/

#pragma once

#include <cassert>
#include <vulkan/vulkan_core.h>

namespace nvvk {
bool checkResult(VkResult result, const char* message = nullptr);
bool checkResult(VkResult result, const char* file, int32_t line);

#ifndef NVVK_CHECK
#define NVVK_CHECK(result) nvvk::checkResult(result, __FILE__, __LINE__)
#endif

#ifdef VULKAN_HPP
inline bool checkResult(vk::Result result, const char* message = nullptr)
{
  return checkResult((VkResult)result, message);
}
inline bool checkResult(vk::Result result, const char* file, int32_t line)
{
  return checkResult((VkResult)result, file, line);
}
#endif

}  // namespace nvvk
