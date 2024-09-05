/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MODULES_HOLOVIZ_SRC_VULKAN_FORMAT_UTIL_HPP
#define MODULES_HOLOVIZ_SRC_VULKAN_FORMAT_UTIL_HPP

#include <optional>

#include <vulkan/vulkan.hpp>

#include "../holoviz/color_space.hpp"
#include "../holoviz/image_format.hpp"

namespace holoscan::viz {

/**
 * Get information on a format
 *
 * @param format format to get information from
 * @param channels format channels
 * @param hw_channels channels when used by Vulkan (different from `channels` for RGB8 formats)
 * @param component_size size in bytes of one component
 * @param width_divisor width divisor for multi-planar formats
 * @param height_divisor height divisor for multi-planar formats
 * @param plane image plane for multi-planar formats
 */
void format_info(ImageFormat format, uint32_t* channels, uint32_t* hw_channels,
                 uint32_t* component_size, uint32_t* width_divisor = nullptr,
                 uint32_t* height_divisor = nullptr, uint32_t plane = 0);

/**
 * Convert a ImageFormat enum to a Vulkan format enum
 *
 * @param format ImageFormat enum
 * @return vk::Format Vulkan format enum
 */
vk::Format to_vulkan_format(ImageFormat format);

/**
 * Convert a Vulkan format enum to a ImageFormat enum. If there is no matching ImageFormat then
 * the return value will not be valid.
 *
 * @param vk_format Vulkan format enum
 * @return std::optional<ImageFormat> ImageFormat enum
 */
std::optional<ImageFormat> to_image_format(vk::Format vk_format);

/**
 * Convert a ColorSpace enum to a Vulkan color space enum
 *
 * @param color_space ColorSpace enum
 * @return vk::ColorSpaceKHR Vulkan color space enum
 */
vk::ColorSpaceKHR to_vulkan_color_space(ColorSpace color_space);

/// @return true if fmt is a depth format
bool is_depth_format(ImageFormat fmt);

/// @return true if fmt is a ycpcr format
bool is_yuv_format(ImageFormat fmt);

/// @return true if fmt is multi-planar
bool is_multi_planar_format(ImageFormat fmt);

}  // namespace holoscan::viz

#endif /* MODULES_HOLOVIZ_SRC_VULKAN_FORMAT_UTIL_HPP */
