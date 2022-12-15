/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nvvk/commands_vk.hpp>
#include <nvvk/error_vk.hpp>
#include <nvvk/images_vk.hpp>


namespace holoscan::viz {

bool FramebufferSequence::init(nvvk::ResourceAllocator *alloc, const nvvk::Context &vkctx,
                               VkQueue queue, VkSurfaceKHR surface) {
    color_format_ = VkFormat::VK_FORMAT_R8G8B8A8_UNORM;

    if (surface) {
        // pick a preferred format or use the first available one
        uint32_t surface_format_count;
        NVVK_CHECK(
            vkGetPhysicalDeviceSurfaceFormatsKHR(vkctx.m_physicalDevice, surface,
                                                &surface_format_count, nullptr));
        std::vector<VkSurfaceFormatKHR> surfaceFormats(surface_format_count);
        NVVK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(vkctx.m_physicalDevice, surface,
                                         &surface_format_count, surfaceFormats.data()));

        bool found = false;
        for (auto &f : surfaceFormats) {
            if (color_format_ == f.format) {
                found = true;
                break;
            }
        }
        if (!found) {
            color_format_ = surfaceFormats[0].format;
        }

        swap_chain_.reset(new nvvk::SwapChain());
        if (!swap_chain_->init(vkctx.m_device, vkctx.m_physicalDevice, queue,
                               vkctx.m_queueGCT.familyIndex, surface, color_format_,
                             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
            return false;

        // the color format might have changed when creating the swap chain
        color_format_ = swap_chain_->getFormat();
    } else {
        device_             = vkctx.m_device;
        queue_family_index_ = vkctx.m_queueGCT.familyIndex;

        image_count_ = 3;

        alloc_ = alloc;

        // create semaphores
        semaphores_.resize(image_count_);
        const VkSemaphoreCreateInfo sem_create_info = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        for (uint32_t i = 0; i < image_count_; ++i) {
            if (NVVK_CHECK(vkCreateSemaphore(device_, &sem_create_info, nullptr, &semaphores_[i])))
                return false;
        }
    }

    return true;
}

void FramebufferSequence::deinit() {
    if (swap_chain_) {
        swap_chain_->deinit();
        swap_chain_.reset();
    } else {
        for (auto &&color_texture : color_textures_) {
            alloc_->destroy(color_texture);
        }
        for (auto &&semaphore : semaphores_) {
            vkDestroySemaphore(device_, semaphore, nullptr);
        }
    }
}

bool FramebufferSequence::update(uint32_t width, uint32_t height, VkExtent2D *dimensions) {
    if (swap_chain_) {
        // vkCreateSwapchainKHR() randomly fails with VK_ERROR_INITIALIZATION_FAILED on driver 510
        // https://nvbugswb.nvidia.com/NvBugs5/SWBug.aspx?bugid=3612509&cmtNo=
        // Workaround: retry several times.
        uint32_t retries = 3;
        while (retries) {
            if (swap_chain_->update(static_cast <int>(width), static_cast <int>(height), dimensions))
                break;

            --retries;
            if (retries == 0) {
                return false;
            }
        }

        image_count_ = swap_chain_->getImageCount();
    } else {
        for (auto &&color_texture : color_textures_) {
            alloc_->destroy(color_texture);
        }
        color_textures_.clear();

        color_textures_.resize(image_count_);
        for (uint32_t i = 0; i < image_count_; ++i) {
            const VkImageCreateInfo color_create_info =
                nvvk::makeImage2DCreateInfo(VkExtent2D{width, height}, color_format_,
                                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
            const nvvk::Image image = alloc_->createImage(color_create_info);

            const VkImageViewCreateInfo image_view_info =
                  nvvk::makeImageViewCreateInfo(image.image, color_create_info);

            color_textures_[i] = alloc_->createTexture(image, image_view_info);
            {
                nvvk::CommandPool cmd_buf_get(device_, queue_family_index_);
                VkCommandBuffer cmd_buf = cmd_buf_get.createCommandBuffer();

                nvvk::cmdBarrierImageLayout(cmd_buf, color_textures_[i].image,
                                            VK_IMAGE_LAYOUT_UNDEFINED,
                                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

                cmd_buf_get.submitAndWait(cmd_buf);
            }
            color_textures_[i].descriptor.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
    }
    return true;
}

bool FramebufferSequence::acquire() {
    if (swap_chain_) {
        return swap_chain_->acquire();
    }

    current_image_ = (current_image_ + 1) % image_count_;
    return true;
}

uint32_t FramebufferSequence::get_active_image_index() const {
    if (swap_chain_) {
        return swap_chain_->getActiveImageIndex();
    }

    return current_image_;
}

VkImageView FramebufferSequence::get_image_view(uint32_t i) const {
    if (swap_chain_) {
        return swap_chain_->getImageView(i);
    }

    if (i >= color_textures_.size()) {
        throw std::runtime_error("Invalid image view index");
    }

    return color_textures_[i].descriptor.imageView;
}

void FramebufferSequence::present(VkQueue queue) {
    if (swap_chain_) {
        swap_chain_->present(queue);
    } else {
        active_semaphore_ = semaphores_[current_image_];
    }
}

VkSemaphore FramebufferSequence::get_active_read_semaphore() const {
    if (swap_chain_) {
        return swap_chain_->getActiveReadSemaphore();
    }

    return active_semaphore_;
}

VkSemaphore FramebufferSequence::get_active_written_semaphore() const {
    if (swap_chain_) {
        return swap_chain_->getActiveWrittenSemaphore();
    }

    return semaphores_[current_image_];
}

VkImage FramebufferSequence::get_active_image() const {
    if (swap_chain_) {
        return swap_chain_->getActiveImage();
    }

    return color_textures_[current_image_].image;
}

void FramebufferSequence::cmd_update_barriers(VkCommandBuffer cmd) const {
    if (swap_chain_) {
        swap_chain_->cmdUpdateBarriers(cmd);
    }
}

}  // namespace holoscan::viz
