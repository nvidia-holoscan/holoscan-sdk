/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_fp16.h>
#include <fmt/format.h>

#include <cstring>
#include <optional>
#include <type_traits>
#include <vector>

#include <magic_enum/magic_enum.hpp>
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
 * @param bit_depth representative bit depth per color/luma channel
 */
void format_info(ImageFormat format, uint32_t* channels, uint32_t* hw_channels,
                 uint32_t* component_size, uint32_t* width_divisor = nullptr,
                 uint32_t* height_divisor = nullptr, uint32_t plane = 0,
                 uint32_t* bit_depth = nullptr);

/**
 * Get the bit depth of a format
 *
 * @param format format to get the bit depth from
 * @return bit depth of the format
 */
uint32_t format_bit_depth(ImageFormat format);

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

/// @return true if fmt is supported by the hardware
bool is_format_supported(vk::PhysicalDevice physical_device, ImageFormat fmt);

/// @return list of all formats supported by the hardware
std::vector<ImageFormat> get_supported_formats(vk::PhysicalDevice physical_device);

/// @return list of all available formats
const std::vector<ImageFormat>& get_formats();

/**
 * Get the alpha value for a given format when converting RGB to RGBA.
 *
 * This function returns the appropriate alpha channel value (representing full opacity)
 * for RGB formats when converting them to RGBA formats.
 *
 * @tparam T The type of the alpha value (uint8_t, uint16_t, or uint32_t)
 * @param format The image format
 * @return The alpha value appropriate for the format
 * @throws std::runtime_error if the format is not supported for the given type
 */
template <typename T>
T GetAlphaValueForFormat(ImageFormat format) {
  if constexpr (std::is_same_v<T, uint8_t>) {
    switch (format) {
      case ImageFormat::R8G8B8_UNORM:
      case ImageFormat::R8G8B8_SRGB:
        return 0xFF;
      case ImageFormat::R8G8B8_SNORM:
        return 0x7F;
      default:
        throw std::runtime_error(
            fmt::format("Unhandled format {}.", magic_enum::enum_name(format)));
    }
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    switch (format) {
      case ImageFormat::R16G16B16_UNORM:
        return 0xFFFF;
      case ImageFormat::R16G16B16_SNORM:
        return 0x7FFF;
      case ImageFormat::R16G16B16_SFLOAT:
        return __half_as_ushort(__float2half(1.0f));
      default:
        throw std::runtime_error(
            fmt::format("Unhandled format {}.", magic_enum::enum_name(format)));
    }
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    switch (format) {
      case ImageFormat::R32G32B32_SFLOAT: {
        float one = 1.0f;
        uint32_t alpha;
        static_assert(sizeof(alpha) == sizeof(one), "alpha and one have different sizes");
        std::memcpy(&alpha, &one, sizeof(alpha));
        return alpha;
      }
      default:
        throw std::runtime_error(
            fmt::format("Unhandled format {}.", magic_enum::enum_name(format)));
    }
  } else {
    static_assert(
        std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, uint32_t>,
        "GetAlphaValueForFormat only supports uint8_t, uint16_t, or uint32_t");
  }
}

}  // namespace holoscan::viz

#endif /* MODULES_HOLOVIZ_SRC_VULKAN_FORMAT_UTIL_HPP */
