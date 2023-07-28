/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace holoscan::viz {

FramebufferSequence::~FramebufferSequence() {
  if (swap_chain_) {
    swap_chain_->deinit();
    swap_chain_.reset();
  } else {
    for (auto&& color_texture : color_textures_) { alloc_->destroy(color_texture); }
    semaphores_.clear();
  }
  for (auto&& depth_texture : depth_textures_) { alloc_->destroy(depth_texture); }
}

void FramebufferSequence::init(nvvk::ResourceAllocator* alloc, const vk::Device& device,
                               const vk::PhysicalDevice& physical_device, vk::Queue queue,
                               uint32_t queue_family_index, vk::SurfaceKHR surface) {
  alloc_ = alloc;
  device_ = device;
  queue_family_index_ = queue_family_index;
  image_count_ = 3;

  color_format_ = vk::Format::eR8G8B8A8Unorm;

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

  if (surface) {
    // pick a preferred format or use the first available one
    const std::vector<vk::SurfaceFormatKHR> surfaceFormats =
        physical_device.getSurfaceFormatsKHR(surface);

    bool found = false;
    for (auto& f : surfaceFormats) {
      if (color_format_ == f.format) {
        found = true;
        break;
      }
    }
    if (!found) { color_format_ = surfaceFormats[0].format; }

    swap_chain_.reset(new nvvk::SwapChain());
    if (!swap_chain_->init(device,
                           physical_device,
                           queue,
                           queue_family_index,
                           surface,
                           VkFormat(color_format_),
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
      throw std::runtime_error("Failed to init swap chain.");
    }

    // the color format and image count might have changed when creating the swap chain
    color_format_ = vk::Format(swap_chain_->getFormat());
    image_count_ = swap_chain_->getImageCount();
  } else {
    // create semaphores
    for (uint32_t i = 0; i < image_count_; ++i) {
      semaphores_.push_back(device.createSemaphoreUnique({}));
    }
  }
}

void FramebufferSequence::update(uint32_t width, uint32_t height, vk::Extent2D* dimensions) {
  if (swap_chain_) {
    // vkCreateSwapchainKHR() randomly fails with VK_ERROR_INITIALIZATION_FAILED on driver 510
    // https://nvbugswb.nvidia.com/NvBugs5/SWBug.aspx?bugid=3612509&cmtNo=
    // Workaround: retry several times.
    uint32_t retries = 3;
    while (retries) {
      if (swap_chain_->update(static_cast<int>(width),
                              static_cast<int>(height),
                              reinterpret_cast<VkExtent2D*>(dimensions))) {
        break;
      }

      --retries;
      if (retries == 0) { throw std::runtime_error("Failed to update swap chain."); }
    }

    image_count_ = swap_chain_->getImageCount();
  } else {
    for (auto&& color_texture : color_textures_) { alloc_->destroy(color_texture); }
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

  for (auto&& depth_texture : depth_textures_) { alloc_->destroy(depth_texture); }
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
  if (swap_chain_) { return swap_chain_->getActiveImageIndex(); }

  return current_image_;
}

vk::ImageView FramebufferSequence::get_color_image_view(uint32_t i) const {
  if (swap_chain_) { return swap_chain_->getImageView(i); }

  if (i >= color_textures_.size()) { throw std::runtime_error("Invalid image view index"); }

  return color_textures_[i].descriptor.imageView;
}

vk::ImageView FramebufferSequence::get_depth_image_view(uint32_t i) const {
  if (i >= depth_textures_.size()) { throw std::runtime_error("Invalid image view index"); }

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
  if (swap_chain_) { return swap_chain_->getActiveReadSemaphore(); }

  return active_semaphore_;
}

vk::Semaphore FramebufferSequence::get_active_written_semaphore() const {
  if (swap_chain_) { return swap_chain_->getActiveWrittenSemaphore(); }

  return semaphores_[current_image_].get();
}

vk::Image FramebufferSequence::get_active_color_image() const {
  if (swap_chain_) { return swap_chain_->getActiveImage(); }

  return color_textures_[current_image_].image;
}

vk::Image FramebufferSequence::get_active_depth_image() const {
  return depth_textures_[current_image_].image;
}

void FramebufferSequence::cmd_update_barriers(vk::CommandBuffer cmd) const {
  if (swap_chain_) { swap_chain_->cmdUpdateBarriers(cmd); }
}

}  // namespace holoscan::viz
