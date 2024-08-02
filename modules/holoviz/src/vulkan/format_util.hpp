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

void format_info(ImageFormat format, uint32_t* src_channels, uint32_t* dst_channels,
                 uint32_t* component_size);
vk::Format to_vulkan_format(ImageFormat format);
std::optional<ImageFormat> to_image_format(vk::Format vk_format);

vk::ColorSpaceKHR to_vulkan_color_space(ColorSpace color_space);

}  // namespace holoscan::viz

#endif /* MODULES_HOLOVIZ_SRC_VULKAN_FORMAT_UTIL_HPP */
