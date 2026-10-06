/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoviz/errors.hpp>
#include <holoviz/expected.hpp>

#include <string>
#include "vk_impls.hpp"  // IWYU pragma: keep

namespace holoviz::vulkan {

/**
 * @brief Check the result of a Vulkan function call and log an error if it indicates a failure.
 * @param result The Vulkan result to check.
 * @param file The file where the Vulkan call was made.
 * @param line The line number where the Vulkan call was made.
 * @return true if the Vulkan call failed, false on success.
 */
[[nodiscard]] bool checkVkResult(VkResult result, const char* file, int32_t line);

std::string vk_error_string(VkResult result, const char* file, int32_t line);

#ifndef VIZ_VK_CHECK
#define VIZ_VK_CHECK(result) holoviz::vulkan::checkVkResult(result, __FILE__, __LINE__)
#endif

#ifndef VIZ_VK_FORWARD_ERROR
#define VIZ_VK_FORWARD_ERROR(result)                                                     \
  {                                                                                      \
    VkResult _vk_result = result;                                                        \
    if (VIZ_VK_CHECK(_vk_result))                                                        \
      return holoviz::make_unexpected(holoviz::Error{                                    \
          .code = ErrorCode::kVulkanError,                                               \
          .message = holoviz::vulkan::vk_error_string(_vk_result, __FILE__, __LINE__)}); \
  }
#endif

// Comma token for the __VA_OPT__ expansions below: a literal `,` inside the parentheses is
// formatted as `(, )` by clang-format, which cpplint then rejects as an extra space before `)`.
#ifndef VIZ_VK_COMMA
#define VIZ_VK_COMMA ,
#endif

// Dispatch a device-scoped Vulkan call: the VulkanDevice object (pointer-like, e.g. `device_` or
// `this`) is named once and supplies both the dispatch table (fn()) and the VkDevice first
// argument (vk_device()). Errors return through VIZ_VK_FORWARD_ERROR.
#ifndef VIZ_VK_DEVICE_FUNC
#define VIZ_VK_DEVICE_FUNC(func, device, ...) \
  VIZ_VK_FORWARD_ERROR(                       \
      (device)->fn()->func((device)->vk_device() __VA_OPT__(VIZ_VK_COMMA) __VA_ARGS__))
#endif

// Same dispatch shape for device-scoped Vulkan calls that do not return a VkResult
// (e.g. vkDestroy*, vkGet*MemoryRequirements): no error handling.
#ifndef VIZ_VK_DEVICE_FUNC_NO_ERROR
#define VIZ_VK_DEVICE_FUNC_NO_ERROR(func, device, ...) \
  (device)->fn()->func((device)->vk_device() __VA_OPT__(VIZ_VK_COMMA) __VA_ARGS__)
#endif

}  // namespace holoviz::vulkan
