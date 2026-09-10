/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vk_errors.hpp"
#include <holoviz/logger.hpp>

#include <string>

#include <vulkan/vk_enum_string_helper.h>

namespace holoviz::rhi::vulkan {

bool checkVkResult(VkResult result, const char* file, int32_t line) {
  if (result == VK_SUCCESS) {
    return false;
  }

  HOLOVIZ_LOG_ERROR("Vulkan error at {}:{} - {} ({})",
                    file,
                    line,
                    string_VkResult(result),
                    static_cast<int>(result));

  return true;
}

std::string vk_error_string(VkResult result, const char* file, int32_t line) {
  const char* error_string = string_VkResult(result);

  std::string out_str = "[holoviz] Vulkan error at " + std::string(file) + ":" +
                        std::to_string(line) + " - " +
                        std::string(error_string ? error_string : "Unknown Vulkan error") + " (" +
                        std::to_string(static_cast<int>(result)) + ")";

  return out_str;
}

}  // namespace holoviz::rhi::vulkan
