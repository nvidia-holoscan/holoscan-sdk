/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "framebuffer_sequence.hpp"

#include <stdexcept>
#include <vector>

#include <nvvk/commands_vk.hpp>
#include <nvvk/error_vk.hpp>
#include <nvvk/images_vk.hpp>

#include <holoscan/logger/logger.hpp>

#include "format_util.hpp"

namespace holoscan::viz {

FramebufferSequence::~FramebufferSequence() {
  if (swap_chain_) {
    swap_chain_->deinit();
    swap_chain_.reset();
  } else {
    for (auto&& color_texture : color_textures_) {
      alloc_->destroy(color_texture);
    }
    semaphores_.clear();
  }
  for (auto&& depth_texture : depth_textures_) {
    alloc_->destroy(depth_texture);
  }
}

void FramebufferSequence::init(nvvk::ResourceAllocator* alloc, vk::Device device,
                               vk::PhysicalDevice physical_device, vk::Queue queue,
                               uint32_t queue_family_index,
                               std::optional<SurfaceFormat> surface_format,
                               vk::SurfaceKHR surface) {
  alloc_ = alloc;
  device_ = device;
  physical_device_ = physical_device;
  queue_family_index_ = queue_family_index;
  surface_ = surface;
  image_count_ = 3;

  // pick the surface format
  auto surface_formats = get_surface_formats();
#ifdef _DEBUG
  HOLOSCAN_LOG_INFO("Available surface formats:");
  for (auto& syrface_format : surface_formats) {
    HOLOSCAN_LOG_INFO("    {}, {}",
                      vk::to_string(to_vulkan_format(syrface_format.image_format_)),
                      vk::to_string(to_vulkan_color_space(syrface_format.color_space_)));
  }
#endif

  bool found_surface_format = false;
  if (surface_format.has_value()) {
    // if the surface format has been specified, check if it is supported
    for (auto&& cur_surface_format : surface_formats) {
      if ((cur_surface_format.image_format_ == surface_format.value().image_format_) &&
          (cur_surface_format.color_space_ == surface_format.value().color_space_)) {
        surface_format = cur_surface_format;
        found_surface_format = true;
        break;
      }
    }
    if (!found_surface_format) {
      throw std::runtime_error(fmt::format("Surface format '{}, {}' not supported",
                                           int(surface_format.value().image_format_),
                                           int(surface_format.value().color_space_)));
    }
  } else {
    // else pick the 8-bit non-srgb image format, this matches the previous Holoviz behavior
    for (auto&& cur_surface_format : surface_formats) {
      if ((cur_surface_format.image_format_ == ImageFormat::B8G8R8A8_UNORM) ||
          (cur_surface_format.image_format_ == ImageFormat::R8G8B8A8_UNORM) ||
          (cur_surface_format.image_format_ == ImageFormat::A8B8G8R8_UNORM_PACK32)) {
        surface_format = cur_surface_format;
        found_surface_format = true;
        break;
      }
    }
    if (!found_surface_format) {
      // not found, pick the first one
      HOLOSCAN_LOG_WARN("Surface format '{}, {}' not found, using first available format '{}, {}'",
                        int(surface_format.value().image_format_),
                        int(surface_format.value().color_space_),
                        int(surface_formats[0].image_format_),
                        int(surface_formats[0].color_space_));
      surface_format = surface_formats[0];
    }
  }

  color_format_ = to_vulkan_format(surface_format.value().image_format_);
  color_space_ = to_vulkan_color_space(surface_format.value().color_space_);

  if (surface_) {
    swap_chain_.reset(
        new nvvk::SwapChain(device,
                            physical_device,
                            queue,
                            queue_family_index,
                            surface_,
                            VkFormat(color_format_),
                            VkColorSpaceKHR(color_space_),
                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT));

    // the image count might have changed when creating the swap chain
    image_count_ = swap_chain_->getImageCount();
  } else {
    // create semaphores
    for (uint32_t i = 0; i < image_count_; ++i) {
      semaphores_.push_back(device.createSemaphoreUnique({}));
    }
  }

  HOLOSCAN_LOG_INFO(
      "Using surface format '{}, {}'", vk::to_string(color_format_), vk::to_string(color_space_));

  // pick the first available depth format in order of preference
  depth_format_ = vk::Format::eUndefined;
  const vk::FormatFeatureFlagBits depth_feature =
      vk::FormatFeatureFlagBits::eDepthStencilAttachment;
  for (const auto& format :
       {vk::Format::eD32Sfloat, vk::Format::eD24UnormS8Uint, vk::Format::eD16Unorm}) {
    vk::FormatProperties format_prop;
    physical_device.getFormatProperties(format, &format_prop);
    if ((format_prop.optimalTilingFeatures & depth_feature) == depth_feature) {
      depth_format_ = format;
      break;
    }
  }
  if (depth_format_ == vk::Format::eUndefined) {
    throw std::runtime_error("Could not find a suitable depth format.");
  }

  HOLOSCAN_LOG_INFO("Using depth format '{}'", vk::to_string(depth_format_));

#ifdef _DEBUG
  if (surface_) {
    const std::vector<vk::PresentModeKHR> present_modes =
        physical_device_.getSurfacePresentModesKHR(surface_);
    HOLOSCAN_LOG_INFO("Available present modes");
    for (auto present_mode : present_modes) {
      HOLOSCAN_LOG_INFO(" {}", to_string(present_mode));
    }
  }
#endif
}

std::vector<SurfaceFormat> FramebufferSequence::get_surface_formats() const {
  std::vector<SurfaceFormat> surface_formats;
  if (surface_) {
    const std::vector<vk::SurfaceFormatKHR> vulkan_surface_formats =
        physical_device_.getSurfaceFormatsKHR(surface_);

    for (auto&& vulkan_surface_format : vulkan_surface_formats) {
      std::optional<ImageFormat> image_format = to_image_format(vulkan_surface_format.format);
      if (image_format.has_value()) {
        switch (vulkan_surface_format.colorSpace) {
          case vk::ColorSpaceKHR::eSrgbNonlinear:
            surface_formats.push_back({image_format.value(), ColorSpace::SRGB_NONLINEAR});
            break;
          case vk::ColorSpaceKHR::eExtendedSrgbLinearEXT:
            surface_formats.push_back({image_format.value(), ColorSpace::EXTENDED_SRGB_LINEAR});
            break;
          case vk::ColorSpaceKHR::eBt2020LinearEXT:
            surface_formats.push_back({image_format.value(), ColorSpace::BT2020_LINEAR});
            break;
          case vk::ColorSpaceKHR::eHdr10St2084EXT:
            surface_formats.push_back({image_format.value(), ColorSpace::HDR10_ST2084});
            break;
          case vk::ColorSpaceKHR::eBt709LinearEXT:
            surface_formats.push_back({image_format.value(), ColorSpace::BT709_LINEAR});
            break;
          default:
            HOLOSCAN_LOG_DEBUG("Unhandled Vulkan color space {}",
                               to_string(vulkan_surface_format.colorSpace));
            break;
        }
      }
    }
  } else {
    // headless rendering, add support for common formats and use pass through color space since
    // there is no display attached
    surface_formats.push_back({ImageFormat::B8G8R8A8_UNORM, ColorSpace::PASS_THROUGH});
    surface_formats.push_back({ImageFormat::B8G8R8A8_SRGB, ColorSpace::PASS_THROUGH});
    surface_formats.push_back({ImageFormat::R8G8B8A8_UNORM, ColorSpace::PASS_THROUGH});
    surface_formats.push_back({ImageFormat::R8G8B8A8_SRGB, ColorSpace::PASS_THROUGH});
    surface_formats.push_back({ImageFormat::A2B10G10R10_UNORM_PACK32, ColorSpace::PASS_THROUGH});
    surface_formats.push_back({ImageFormat::A2R10G10B10_UNORM_PACK32, ColorSpace::PASS_THROUGH});
    surface_formats.push_back({ImageFormat::R16G16B16A16_SFLOAT, ColorSpace::PASS_THROUGH});
  }
  return surface_formats;
}

std::vector<PresentMode> FramebufferSequence::get_present_modes() const {
  std::vector<PresentMode> present_modes;
  if (surface_) {
    const std::vector<vk::PresentModeKHR> vulkan_present_modes =
        physical_device_.getSurfacePresentModesKHR(surface_);

    for (auto& vulkan_present_mode : vulkan_present_modes) {
      switch (vulkan_present_mode) {
        case vk::PresentModeKHR::eFifo:
          present_modes.push_back(PresentMode::FIFO);
          break;
        case vk::PresentModeKHR::eImmediate:
          present_modes.push_back(PresentMode::IMMEDIATE);
          break;
        case vk::PresentModeKHR::eMailbox:
          present_modes.push_back(PresentMode::MAILBOX);
          break;
        default:
          // ignore
          break;
      }
    }
  }
  return present_modes;
}

void FramebufferSequence::update(uint32_t width, uint32_t height, PresentMode present_mode,
                                 vk::Extent2D* dimensions) {
  if (swap_chain_) {
    const std::vector<vk::PresentModeKHR> present_modes =
        physical_device_.getSurfacePresentModesKHR(surface_);

    vk::PresentModeKHR vk_present_mode;
    if (present_mode == PresentMode::AUTO) {
      // auto select

      // everyone must support FIFO mode
      vk_present_mode = vk::PresentModeKHR::eFifo;
      // try to find a non-blocking alternative to FIFO
      for (auto mode : present_modes) {
        if (mode == vk::PresentModeKHR::eMailbox) {
          // prefer mailbox due to no tearing
          vk_present_mode = mode;
          break;
        }
        if (mode == vk::PresentModeKHR::eImmediate) {
          // immediate mode is non-blocking, but has tearing
          vk_present_mode = mode;
        }
      }
    } else {
      switch (present_mode) {
        case PresentMode::FIFO:
          vk_present_mode = vk::PresentModeKHR::eFifo;
          break;
        case PresentMode::IMMEDIATE:
          vk_present_mode = vk::PresentModeKHR::eImmediate;
          break;
        case PresentMode::MAILBOX:
          vk_present_mode = vk::PresentModeKHR::eMailbox;
          break;
        default:
          throw std::runtime_error(fmt::format("Unhandled present mode '{}'", int(present_mode)));
      }

      auto it = std::find(present_modes.begin(), present_modes.end(), vk_present_mode);
      if (it == present_modes.end()) {
        throw std::runtime_error(
            fmt::format("Present mode {} is not supported", to_string(vk_present_mode)));
      }
    }
    HOLOSCAN_LOG_INFO("Using present mode '{}'", to_string(vk_present_mode));

    // vkCreateSwapchainKHR() randomly fails with VK_ERROR_INITIALIZATION_FAILED on driver 510
    // https://nvbugswb.nvidia.com/NvBugs5/SWBug.aspx?bugid=3612509&cmtNo=
    // Workaround: retry several times.
    uint32_t retries = 3;
    while (retries) {
      if (swap_chain_->update(static_cast<int>(width),
                              static_cast<int>(height),
                              (VkPresentModeKHR)vk_present_mode,
                              reinterpret_cast<VkExtent2D*>(dimensions))) {
        break;
      }

      --retries;
      if (retries == 0) {
        throw std::runtime_error("Failed to update swap chain.");
      }
    }

    image_count_ = swap_chain_->getImageCount();
  } else {
    for (auto&& color_texture : color_textures_) {
      alloc_->destroy(color_texture);
    }
    color_textures_.clear();

    color_textures_.resize(image_count_);
    for (uint32_t i = 0; i < image_count_; ++i) {
      const vk::ImageCreateInfo color_create_info = nvvk::makeImage2DCreateInfo(
          vk::Extent2D{width, height},
          color_format_,
          vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc);
      const nvvk::Image color_image = alloc_->createImage(color_create_info);

      const vk::ImageViewCreateInfo color_image_view_info =
          nvvk::makeImageViewCreateInfo(color_image.image, color_create_info);

      color_textures_[i] = alloc_->createTexture(color_image, color_image_view_info);
      {
        nvvk::CommandPool cmd_buf_get(device_, queue_family_index_);
        vk::CommandBuffer cmd_buf = cmd_buf_get.createCommandBuffer();

        nvvk::cmdBarrierImageLayout(cmd_buf,
                                    color_textures_[i].image,
                                    VK_IMAGE_LAYOUT_UNDEFINED,
                                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        cmd_buf_get.submitAndWait(cmd_buf);
      }
      color_textures_[i].descriptor.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
  }

  for (auto&& depth_texture : depth_textures_) {
    alloc_->destroy(depth_texture);
  }
  depth_textures_.clear();
  depth_textures_.resize(image_count_);
  for (uint32_t i = 0; i < image_count_; ++i) {
    const vk::ImageCreateInfo depth_create_info = nvvk::makeImage2DCreateInfo(
        vk::Extent2D{width, height},
        depth_format_,
        vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eTransferSrc);
    const nvvk::Image depth_image = alloc_->createImage(depth_create_info);

    const vk::ImageAspectFlags aspect_mask = vk::ImageAspectFlagBits::eDepth;
    const vk::ImageViewCreateInfo depth_image_view_info =
        nvvk::makeImage2DViewCreateInfo(depth_image.image, depth_format_, aspect_mask);

    depth_textures_[i] = alloc_->createTexture(depth_image, depth_image_view_info);
    {
      nvvk::CommandPool cmd_buf_get(device_, queue_family_index_);
      vk::CommandBuffer cmd_buf = cmd_buf_get.createCommandBuffer();

      nvvk::cmdBarrierImageLayout(cmd_buf,
                                  depth_textures_[i].image,
                                  VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                  VkImageAspectFlags(aspect_mask));

      cmd_buf_get.submitAndWait(cmd_buf);
    }
    depth_textures_[i].descriptor.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  }
}

void FramebufferSequence::acquire() {
  if (swap_chain_) {
    if (!swap_chain_->acquire()) {
      throw std::runtime_error("Failed to acquire next swap chain image.");
    }
  }

  current_image_ = (current_image_ + 1) % image_count_;
}

uint32_t FramebufferSequence::get_active_image_index() const {
  if (swap_chain_) {
    return swap_chain_->getActiveImageIndex();
  }

  return current_image_;
}

vk::ImageView FramebufferSequence::get_color_image_view(uint32_t i) const {
  if (swap_chain_) {
    return swap_chain_->getImageView(i);
  }

  if (i >= color_textures_.size()) {
    throw std::runtime_error("Invalid image view index");
  }

  return color_textures_[i].descriptor.imageView;
}

vk::ImageView FramebufferSequence::get_depth_image_view(uint32_t i) const {
  if (i >= depth_textures_.size()) {
    throw std::runtime_error("Invalid image view index");
  }

  return depth_textures_[i].descriptor.imageView;
}

void FramebufferSequence::present(vk::Queue queue) {
  if (swap_chain_) {
    swap_chain_->present(queue);
  } else {
    active_semaphore_ = semaphores_[current_image_].get();
  }
}

vk::Semaphore FramebufferSequence::get_active_read_semaphore() const {
  if (swap_chain_) {
    return swap_chain_->getActiveReadSemaphore();
  }

  return active_semaphore_;
}

vk::Semaphore FramebufferSequence::get_active_written_semaphore() const {
  if (swap_chain_) {
    return swap_chain_->getActiveWrittenSemaphore();
  }

  return semaphores_[current_image_].get();
}

vk::Image FramebufferSequence::get_active_color_image() const {
  if (swap_chain_) {
    return swap_chain_->getActiveImage();
  }

  return color_textures_[current_image_].image;
}

vk::Image FramebufferSequence::get_active_depth_image() const {
  return depth_textures_[current_image_].image;
}

void FramebufferSequence::cmd_update_barriers(vk::CommandBuffer cmd) const {
  if (swap_chain_) {
    swap_chain_->cmdUpdateBarriers(cmd);
  }
}

}  // namespace holoscan::viz
