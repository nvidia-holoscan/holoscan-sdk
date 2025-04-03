/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "format_util.hpp"

#include <vector>

namespace holoscan::viz {

void format_info(ImageFormat format, uint32_t* channels, uint32_t* hw_channels,
                 uint32_t* component_size, uint32_t* width_divisor, uint32_t* height_divisor,
                 uint32_t plane) {
  if (width_divisor) { *width_divisor = 1; }
  if (height_divisor) { *height_divisor = 1; }
  switch (format) {
    case ImageFormat::R8_UINT:
    case ImageFormat::R8_SINT:
    case ImageFormat::R8_UNORM:
    case ImageFormat::R8_SNORM:
    case ImageFormat::R8_SRGB:
      *channels = *hw_channels = 1U;
      *component_size = sizeof(uint8_t);
      break;
    case ImageFormat::R16_UINT:
    case ImageFormat::R16_SINT:
    case ImageFormat::R16_UNORM:
    case ImageFormat::R16_SNORM:
    case ImageFormat::R16_SFLOAT:
      *channels = *hw_channels = 1U;
      *component_size = sizeof(uint16_t);
      break;
    case ImageFormat::R32_UINT:
    case ImageFormat::R32_SINT:
    // packed formats are treated as single component formats
    case ImageFormat::A2B10G10R10_UNORM_PACK32:
    case ImageFormat::A2R10G10B10_UNORM_PACK32:
      *channels = *hw_channels = 1U;
      *component_size = sizeof(uint32_t);
      break;
    case ImageFormat::R32_SFLOAT:
      *channels = *hw_channels = 1U;
      *component_size = sizeof(float);
      break;
    case ImageFormat::R8G8B8_UNORM:
    case ImageFormat::R8G8B8_SNORM:
    case ImageFormat::R8G8B8_SRGB:
      *channels = 3U;
      *hw_channels = 4U;
      *component_size = sizeof(uint8_t);
      break;
    case ImageFormat::R8G8B8A8_UNORM:
    case ImageFormat::R8G8B8A8_SNORM:
    case ImageFormat::R8G8B8A8_SRGB:
    case ImageFormat::B8G8R8A8_UNORM:
    case ImageFormat::B8G8R8A8_SRGB:
    case ImageFormat::A8B8G8R8_UNORM_PACK32:
    case ImageFormat::A8B8G8R8_SRGB_PACK32:
      *channels = *hw_channels = 4U;
      *component_size = sizeof(uint8_t);
      break;
    case ImageFormat::R16G16B16A16_UNORM:
    case ImageFormat::R16G16B16A16_SNORM:
    case ImageFormat::R16G16B16A16_SFLOAT:
      *channels = *hw_channels = 4U;
      *component_size = sizeof(uint16_t);
      break;
    case ImageFormat::R32G32B32A32_SFLOAT:
      *channels = *hw_channels = 4U;
      *component_size = sizeof(float);
      break;
    case ImageFormat::D16_UNORM:
      *channels = *hw_channels = 1U;
      *component_size = sizeof(uint16_t);
      break;
    case ImageFormat::X8_D24_UNORM:
      *channels = *hw_channels = 1U;
      *component_size = sizeof(uint32_t);
      break;
    case ImageFormat::D32_SFLOAT:
      *channels = *hw_channels = 1U;
      *component_size = sizeof(uint32_t);
      break;
    case ImageFormat::Y8U8Y8V8_422_UNORM:
    case ImageFormat::U8Y8V8Y8_422_UNORM:
      *channels = *hw_channels = 2U;
      *component_size = sizeof(uint8_t);
      break;
    case ImageFormat::Y8_U8V8_2PLANE_420_UNORM:
      if (plane == 0) {
        *channels = *hw_channels = 1U;
      } else if (plane == 1) {
        *channels = *hw_channels = 2U;
        if (width_divisor) { *width_divisor = 2; }
        if (height_divisor) { *height_divisor = 2; }
      } else {
        throw std::invalid_argument("Unhandled plane index");
      }
      *component_size = sizeof(uint8_t);
      break;
    case ImageFormat::Y8_U8V8_2PLANE_422_UNORM:
      if (plane == 0) {
        *channels = *hw_channels = 1U;
      } else if (plane == 1) {
        *channels = *hw_channels = 2U;
        if (width_divisor) { *width_divisor = 2; }
      } else {
        throw std::invalid_argument("Unhandled plane index");
      }
      *component_size = sizeof(uint8_t);
      break;
    case ImageFormat::Y8_U8_V8_3PLANE_420_UNORM:
      *channels = *hw_channels = 1U;
      *component_size = sizeof(uint8_t);
      if (plane == 0) {
      } else if ((plane == 1) || (plane == 2)) {
        if (width_divisor) { *width_divisor = 2; }
        if (height_divisor) { *height_divisor = 2; }
      } else {
        throw std::invalid_argument("Unhandled plane index");
      }
      break;
    case ImageFormat::Y8_U8_V8_3PLANE_422_UNORM:
      *channels = *hw_channels = 1U;
      *component_size = sizeof(uint8_t);
      if (plane == 0) {
      } else if ((plane == 1) || (plane == 2)) {
        if (width_divisor) { *width_divisor = 2; }
      } else {
        throw std::invalid_argument("Unhandled plane index");
      }
      break;
    case ImageFormat::Y16_U16V16_2PLANE_420_UNORM:
      if (plane == 0) {
        *channels = *hw_channels = 1U;
      } else if (plane == 1) {
        *channels = *hw_channels = 2U;
        if (width_divisor) { *width_divisor = 2; }
        if (height_divisor) { *height_divisor = 2; }
      } else {
        throw std::invalid_argument("Unhandled plane index");
      }
      *component_size = sizeof(uint16_t);
      break;
    case ImageFormat::Y16_U16V16_2PLANE_422_UNORM:
      if (plane == 0) {
        *channels = *hw_channels = 1U;
      } else if (plane == 1) {
        *channels = *hw_channels = 2U;
        if (width_divisor) { *width_divisor = 2; }
      } else {
        throw std::invalid_argument("Unhandled plane index");
      }
      *component_size = sizeof(uint16_t);
      break;
    case ImageFormat::Y16_U16_V16_3PLANE_420_UNORM:
      *channels = *hw_channels = 1U;
      *component_size = sizeof(uint16_t);
      if (plane == 0) {
      } else if ((plane == 1) || (plane == 2)) {
        if (width_divisor) { *width_divisor = 2; }
        if (height_divisor) { *height_divisor = 2; }
      } else {
        throw std::invalid_argument("Unhandled plane index");
      }
      break;
    case ImageFormat::Y16_U16_V16_3PLANE_422_UNORM:
      *channels = *hw_channels = 1U;
      *component_size = sizeof(uint16_t);
      if (plane == 0) {
      } else if ((plane == 1) || (plane == 2)) {
        if (width_divisor) { *width_divisor = 2; }
      } else {
        throw std::invalid_argument("Unhandled plane index");
      }
      break;
    default:
      throw std::runtime_error("Unhandled image format.");
  }
}

vk::Format to_vulkan_format(ImageFormat format) {
  vk::Format vk_format;

  switch (format) {
    case ImageFormat::R8_UINT:
      vk_format = vk::Format::eR8Uint;
      break;
    case ImageFormat::R8_SINT:
      vk_format = vk::Format::eR8Sint;
      break;
    case ImageFormat::R8_UNORM:
      vk_format = vk::Format::eR8Unorm;
      break;
    case ImageFormat::R8_SNORM:
      vk_format = vk::Format::eR8Snorm;
      break;
    case ImageFormat::R8_SRGB:
      vk_format = vk::Format::eR8Srgb;
      break;
    case ImageFormat::R16_UINT:
      vk_format = vk::Format::eR16Uint;
      break;
    case ImageFormat::R16_SINT:
      vk_format = vk::Format::eR16Sint;
      break;
    case ImageFormat::R16_UNORM:
      vk_format = vk::Format::eR16Unorm;
      break;
    case ImageFormat::R16_SNORM:
      vk_format = vk::Format::eR16Snorm;
      break;
    case ImageFormat::R16_SFLOAT:
      vk_format = vk::Format::eR16Sfloat;
      break;
    case ImageFormat::R32_UINT:
      vk_format = vk::Format::eR32Uint;
      break;
    case ImageFormat::R32_SINT:
      vk_format = vk::Format::eR32Sint;
      break;
    case ImageFormat::R32_SFLOAT:
      vk_format = vk::Format::eR32Sfloat;
      break;
    case ImageFormat::R8G8B8_UNORM:
      // there is no rgb format in Vulkan, use rgba instead; data is converted on upload/download
      vk_format = vk::Format::eR8G8B8A8Unorm;
      break;
    case ImageFormat::R8G8B8_SNORM:
      // there is no rgb format in Vulkan, use rgba instead; data is converted on upload/download
      vk_format = vk::Format::eR8G8B8A8Snorm;
      break;
    case ImageFormat::R8G8B8_SRGB:
      // there is no rgb format in Vulkan, use rgba instead; data is converted on upload/download
      vk_format = vk::Format::eR8G8B8A8Srgb;
      break;
    case ImageFormat::R8G8B8A8_UNORM:
      vk_format = vk::Format::eR8G8B8A8Unorm;
      break;
    case ImageFormat::R8G8B8A8_SNORM:
      vk_format = vk::Format::eR8G8B8A8Snorm;
      break;
    case ImageFormat::R8G8B8A8_SRGB:
      vk_format = vk::Format::eR8G8B8A8Srgb;
      break;
    case ImageFormat::R16G16B16A16_UNORM:
      vk_format = vk::Format::eR16G16B16A16Unorm;
      break;
    case ImageFormat::R16G16B16A16_SNORM:
      vk_format = vk::Format::eR16G16B16A16Snorm;
      break;
    case ImageFormat::R16G16B16A16_SFLOAT:
      vk_format = vk::Format::eR16G16B16A16Sfloat;
      break;
    case ImageFormat::R32G32B32A32_SFLOAT:
      vk_format = vk::Format::eR32G32B32A32Sfloat;
      break;
    case ImageFormat::D16_UNORM:
      vk_format = vk::Format::eD16Unorm;
      break;
    case ImageFormat::X8_D24_UNORM:
      vk_format = vk::Format::eX8D24UnormPack32;
      break;
    case ImageFormat::D32_SFLOAT:
      vk_format = vk::Format::eD32Sfloat;
      break;
    case ImageFormat::A2B10G10R10_UNORM_PACK32:
      vk_format = vk::Format::eA2B10G10R10UnormPack32;
      break;
    case ImageFormat::A2R10G10B10_UNORM_PACK32:
      vk_format = vk::Format::eA2R10G10B10UnormPack32;
      break;
    case ImageFormat::B8G8R8A8_UNORM:
      vk_format = vk::Format::eB8G8R8A8Unorm;
      break;
    case ImageFormat::B8G8R8A8_SRGB:
      vk_format = vk::Format::eB8G8R8A8Srgb;
      break;
    case ImageFormat::A8B8G8R8_UNORM_PACK32:
      vk_format = vk::Format::eA8B8G8R8UnormPack32;
      break;
    case ImageFormat::A8B8G8R8_SRGB_PACK32:
      vk_format = vk::Format::eA8B8G8R8SrgbPack32;
      break;
    case ImageFormat::Y8U8Y8V8_422_UNORM:
      vk_format = vk::Format::eG8B8G8R8422Unorm;
      break;
    case ImageFormat::U8Y8V8Y8_422_UNORM:
      vk_format = vk::Format::eB8G8R8G8422Unorm;
      break;
    case ImageFormat::Y8_U8V8_2PLANE_420_UNORM:
      vk_format = vk::Format::eG8B8R82Plane420Unorm;
      break;
    case ImageFormat::Y8_U8V8_2PLANE_422_UNORM:
      vk_format = vk::Format::eG8B8R82Plane422Unorm;
      break;
    case ImageFormat::Y8_U8_V8_3PLANE_420_UNORM:
      vk_format = vk::Format::eG8B8R83Plane420Unorm;
      break;
    case ImageFormat::Y8_U8_V8_3PLANE_422_UNORM:
      vk_format = vk::Format::eG8B8R83Plane422Unorm;
      break;
    case ImageFormat::Y16_U16V16_2PLANE_420_UNORM:
      vk_format = vk::Format::eG16B16R162Plane420Unorm;
      break;
    case ImageFormat::Y16_U16V16_2PLANE_422_UNORM:
      vk_format = vk::Format::eG16B16R162Plane422Unorm;
      break;
    case ImageFormat::Y16_U16_V16_3PLANE_420_UNORM:
      vk_format = vk::Format::eG16B16R163Plane420Unorm;
      break;
    case ImageFormat::Y16_U16_V16_3PLANE_422_UNORM:
      vk_format = vk::Format::eG16B16R163Plane422Unorm;
      break;
    default:
      throw std::runtime_error("Unhandled image format.");
  }

  return vk_format;
}

std::optional<ImageFormat> to_image_format(vk::Format vk_format) {
  std::optional<ImageFormat> image_format;

  switch (vk_format) {
    case vk::Format::eR8Uint:
      image_format = ImageFormat::R8_UINT;
      break;
    case vk::Format::eR8Sint:
      image_format = ImageFormat::R8_SINT;
      break;
    case vk::Format::eR8Unorm:
      image_format = ImageFormat::R8_UNORM;
      break;
    case vk::Format::eR8Snorm:
      image_format = ImageFormat::R8_SNORM;
      break;
    case vk::Format::eR8Srgb:
      image_format = ImageFormat::R8_SRGB;
      break;
    case vk::Format::eR16Uint:
      image_format = ImageFormat::R16_UINT;
      break;
    case vk::Format::eR16Sint:
      image_format = ImageFormat::R16_SINT;
      break;
    case vk::Format::eR16Unorm:
      image_format = ImageFormat::R16_UNORM;
      break;
    case vk::Format::eR16Snorm:
      image_format = ImageFormat::R16_SNORM;
      break;
    case vk::Format::eR16Sfloat:
      image_format = ImageFormat::R16_SFLOAT;
      break;
    case vk::Format::eR32Uint:
      image_format = ImageFormat::R32_UINT;
      break;
    case vk::Format::eR32Sint:
      image_format = ImageFormat::R32_SINT;
      break;
    case vk::Format::eR32Sfloat:
      image_format = ImageFormat::R32_SFLOAT;
      break;
    case vk::Format::eR8G8B8A8Unorm:
      image_format = ImageFormat::R8G8B8A8_UNORM;
      break;
    case vk::Format::eR8G8B8A8Snorm:
      image_format = ImageFormat::R8G8B8A8_SNORM;
      break;
    case vk::Format::eR8G8B8A8Srgb:
      image_format = ImageFormat::R8G8B8A8_SRGB;
      break;
    case vk::Format::eR16G16B16A16Unorm:
      image_format = ImageFormat::R16G16B16A16_UNORM;
      break;
    case vk::Format::eR16G16B16A16Snorm:
      image_format = ImageFormat::R16G16B16A16_SNORM;
      break;
    case vk::Format::eR16G16B16A16Sfloat:
      image_format = ImageFormat::R16G16B16A16_SFLOAT;
      break;
    case vk::Format::eR32G32B32A32Sfloat:
      image_format = ImageFormat::R32G32B32A32_SFLOAT;
      break;
    case vk::Format::eD16Unorm:
      image_format = ImageFormat::D16_UNORM;
      break;
    case vk::Format::eX8D24UnormPack32:
      image_format = ImageFormat::X8_D24_UNORM;
      break;
    case vk::Format::eD32Sfloat:
      image_format = ImageFormat::D32_SFLOAT;
      break;
    case vk::Format::eA2B10G10R10UnormPack32:
      image_format = ImageFormat::A2B10G10R10_UNORM_PACK32;
      break;
    case vk::Format::eA2R10G10B10UnormPack32:
      image_format = ImageFormat::A2R10G10B10_UNORM_PACK32;
      break;
    case vk::Format::eB8G8R8A8Unorm:
      image_format = ImageFormat::B8G8R8A8_UNORM;
      break;
    case vk::Format::eB8G8R8A8Srgb:
      image_format = ImageFormat::B8G8R8A8_SRGB;
      break;
    case vk::Format::eA8B8G8R8UnormPack32:
      image_format = ImageFormat::A8B8G8R8_UNORM_PACK32;
      break;
    case vk::Format::eA8B8G8R8SrgbPack32:
      image_format = ImageFormat::A8B8G8R8_SRGB_PACK32;
      break;
    default:
      break;
  }

  return image_format;
}

vk::ColorSpaceKHR to_vulkan_color_space(ColorSpace color_space) {
  vk::ColorSpaceKHR vk_color_space;
  switch (color_space) {
    case ColorSpace::SRGB_NONLINEAR:
      vk_color_space = vk::ColorSpaceKHR::eSrgbNonlinear;
      break;
    case ColorSpace::EXTENDED_SRGB_LINEAR:
      vk_color_space = vk::ColorSpaceKHR::eExtendedSrgbLinearEXT;
      break;
    case ColorSpace::BT2020_LINEAR:
      vk_color_space = vk::ColorSpaceKHR::eBt2020LinearEXT;
      break;
    case ColorSpace::HDR10_ST2084:
      vk_color_space = vk::ColorSpaceKHR::eHdr10St2084EXT;
      break;
    case ColorSpace::PASS_THROUGH:
      vk_color_space = vk::ColorSpaceKHR::ePassThroughEXT;
      break;
    case ColorSpace::BT709_LINEAR:
      vk_color_space = vk::ColorSpaceKHR::eBt709LinearEXT;
      break;
    default:
      throw std::runtime_error("Unhandled color space.");
  }

  return vk_color_space;
}

bool is_depth_format(ImageFormat fmt) {
  return ((fmt == ImageFormat::D16_UNORM) || (fmt == ImageFormat::X8_D24_UNORM) ||
          (fmt == ImageFormat::D32_SFLOAT));
}

bool is_yuv_format(ImageFormat fmt) {
  switch (fmt) {
    case ImageFormat::Y8U8Y8V8_422_UNORM:
    case ImageFormat::U8Y8V8Y8_422_UNORM:
    case ImageFormat::Y8_U8V8_2PLANE_420_UNORM:
    case ImageFormat::Y8_U8V8_2PLANE_422_UNORM:
    case ImageFormat::Y8_U8_V8_3PLANE_420_UNORM:
    case ImageFormat::Y8_U8_V8_3PLANE_422_UNORM:
    case ImageFormat::Y16_U16V16_2PLANE_420_UNORM:
    case ImageFormat::Y16_U16V16_2PLANE_422_UNORM:
    case ImageFormat::Y16_U16_V16_3PLANE_420_UNORM:
    case ImageFormat::Y16_U16_V16_3PLANE_422_UNORM:
      return true;
    default:
      return false;
  }
}

bool is_multi_planar_format(ImageFormat fmt) {
  switch (fmt) {
    case ImageFormat::Y8_U8V8_2PLANE_420_UNORM:
    case ImageFormat::Y8_U8V8_2PLANE_422_UNORM:
    case ImageFormat::Y8_U8_V8_3PLANE_420_UNORM:
    case ImageFormat::Y8_U8_V8_3PLANE_422_UNORM:
    case ImageFormat::Y16_U16V16_2PLANE_420_UNORM:
    case ImageFormat::Y16_U16V16_2PLANE_422_UNORM:
    case ImageFormat::Y16_U16_V16_3PLANE_420_UNORM:
    case ImageFormat::Y16_U16_V16_3PLANE_422_UNORM:
      return true;
    default:
      return false;
  }
}

bool is_format_supported(vk::PhysicalDevice physical_device, ImageFormat fmt) {
  // First try to convert to Vulkan format
  const vk::Format vk_format = to_vulkan_format(fmt);

  // Get format properties from physical device
  const vk::FormatProperties format_properties = physical_device.getFormatProperties(vk_format);

  // Check if the format supports sampling
  // We check for sampling since that's what we typically use for texture sampling
  if (!(format_properties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImage)) {
    return false;
  }

  // For depth formats, check if they support depth attachment
  if (is_depth_format(fmt)) {
    if (!(format_properties.optimalTilingFeatures &
          vk::FormatFeatureFlagBits::eDepthStencilAttachment)) {
      return false;
    }
  }

  // For YUV formats, check if they support YUV sampling
  if (is_yuv_format(fmt)) {
    if (!(format_properties.optimalTilingFeatures &
          vk::FormatFeatureFlagBits::eSampledImageYcbcrConversionLinearFilter)) {
      return false;
    }
  }

  return true;
}

const std::vector<ImageFormat>& get_formats() {
  // Add all formats from the ImageFormat enum
  static const std::vector<ImageFormat> all_formats = {ImageFormat::R8_UINT,
                                                       ImageFormat::R8_SINT,
                                                       ImageFormat::R8_UNORM,
                                                       ImageFormat::R8_SNORM,
                                                       ImageFormat::R8_SRGB,
                                                       ImageFormat::R16_UINT,
                                                       ImageFormat::R16_SINT,
                                                       ImageFormat::R16_UNORM,
                                                       ImageFormat::R16_SNORM,
                                                       ImageFormat::R16_SFLOAT,
                                                       ImageFormat::R32_UINT,
                                                       ImageFormat::R32_SINT,
                                                       ImageFormat::R32_SFLOAT,
                                                       ImageFormat::R8G8B8_UNORM,
                                                       ImageFormat::R8G8B8_SNORM,
                                                       ImageFormat::R8G8B8_SRGB,
                                                       ImageFormat::R8G8B8A8_UNORM,
                                                       ImageFormat::R8G8B8A8_SNORM,
                                                       ImageFormat::R8G8B8A8_SRGB,
                                                       ImageFormat::R16G16B16A16_UNORM,
                                                       ImageFormat::R16G16B16A16_SNORM,
                                                       ImageFormat::R16G16B16A16_SFLOAT,
                                                       ImageFormat::R32G32B32A32_SFLOAT,
                                                       ImageFormat::D16_UNORM,
                                                       ImageFormat::X8_D24_UNORM,
                                                       ImageFormat::D32_SFLOAT,
                                                       ImageFormat::A2B10G10R10_UNORM_PACK32,
                                                       ImageFormat::A2R10G10B10_UNORM_PACK32,
                                                       ImageFormat::B8G8R8A8_UNORM,
                                                       ImageFormat::B8G8R8A8_SRGB,
                                                       ImageFormat::A8B8G8R8_UNORM_PACK32,
                                                       ImageFormat::A8B8G8R8_SRGB_PACK32,
                                                       ImageFormat::Y8U8Y8V8_422_UNORM,
                                                       ImageFormat::U8Y8V8Y8_422_UNORM,
                                                       ImageFormat::Y8_U8V8_2PLANE_420_UNORM,
                                                       ImageFormat::Y8_U8V8_2PLANE_422_UNORM,
                                                       ImageFormat::Y8_U8_V8_3PLANE_420_UNORM,
                                                       ImageFormat::Y8_U8_V8_3PLANE_422_UNORM,
                                                       ImageFormat::Y16_U16V16_2PLANE_420_UNORM,
                                                       ImageFormat::Y16_U16V16_2PLANE_422_UNORM,
                                                       ImageFormat::Y16_U16_V16_3PLANE_420_UNORM,
                                                       ImageFormat::Y16_U16_V16_3PLANE_422_UNORM};

  return all_formats;
}

std::vector<ImageFormat> get_supported_formats(vk::PhysicalDevice physical_device) {
  std::vector<ImageFormat> supported_formats;

  const std::vector<ImageFormat> all_formats = get_formats();

  // Filter only formats supported by the hardware
  for (const auto& fmt : all_formats) {
    if (is_format_supported(physical_device, fmt)) { supported_formats.push_back(fmt); }
  }

  return supported_formats;
}

}  // namespace holoscan::viz
