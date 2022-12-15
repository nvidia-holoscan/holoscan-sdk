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

#ifndef HOLOSCAN_VIZ_VULKAN_FRAMEBUFFER_SEQUENCE_HPP
#define HOLOSCAN_VIZ_VULKAN_FRAMEBUFFER_SEQUENCE_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include <nvvk/context_vk.hpp>
#include <nvvk/resourceallocator_vk.hpp>
#include <nvvk/swapchain_vk.hpp>

namespace holoscan::viz {

/**
 * Handle frame buffer sequences, either offscreen or on screen (using swap chain).
 */
class FramebufferSequence {
 public:
    /**
     * Initialize.
     *
     * @param alloc resource allocator to use
     * @param vkctx context
     * @param queue queue
     * @param surface surface, if set then use a swap chain to present to a display,
     *                else render offscreen
     */
    bool init(nvvk::ResourceAllocator *alloc, const nvvk::Context &vkctx,
             VkQueue queue, VkSurfaceKHR surface);

    // triggers queue/device wait idle
    void deinit();

    /**
     * Update the framebuffer to the given size.
     *
     * @param width new framebuffer width
     * @param height new framebuffer height
     * @param [out] dimensions actual dimensions, which may differ from the requested (optional)
     */
    bool update(uint32_t width, uint32_t height, VkExtent2D *dimensions = nullptr);

    /**
     * Acquire the next image to render to
     */
    bool acquire();

    /**
     * @returns the framebuffer color format/
     */
    VkFormat get_format() const {
        return color_format_;
    }

    /**
     * @returns the image view of a color buffer
     */
    VkImageView get_image_view(uint32_t i) const;

    /**
     * @returns the color buffer count
     */
    uint32_t get_image_count() const {
        return image_count_;
    }

    /**
     * @returns the index of the current active color buffer
     */
    uint32_t get_active_image_index() const;

    /**
     * Present on provided queue. After this call the active color buffer is switched to the next one.
     *
     * @param queue queue to present on
     */
    void present(VkQueue queue);

    /**
     * Get the active read semaphore. For offscreen rendering this is identical to the written semaphore
     * of the previous frame, else it's the semaphore used to acquire the swap queue image.
     *
     * @returns active read semaphore
     */
    VkSemaphore get_active_read_semaphore() const;

    /**
     * Get the active write semaphore.
     *
     * @returns active write semaphore
     */
    VkSemaphore get_active_written_semaphore() const;

    /**
     * Get the active image.
     *
     * @returns active image
     */
    VkImage get_active_image() const;

    /**
     * Do a vkCmdPipelineBarrier for VK_IMAGE_LAYOUT_UNDEFINED to VK_IMAGE_LAYOUT_PRESENT_SRC_KHR.
     * Must apply resource transitions after update calls.
     *
     * @param cmd command buffer to use
     */
    void cmd_update_barriers(VkCommandBuffer cmd) const;

 private:
    VkDevice device_;
    uint32_t queue_family_index_;
    nvvk::ResourceAllocator *alloc_ = nullptr;  ///< Allocator for color buffers

    VkFormat color_format_;

    uint32_t image_count_;

    /// the swap chain is used if no surface is set
    std::unique_ptr<nvvk::SwapChain> swap_chain_;

    // members used when rendering offscreen
    uint32_t current_image_       = 0;
    VkSemaphore active_semaphore_ = VK_NULL_HANDLE;  ///< the active semaphore for the current image

    std::vector<nvvk::Texture> color_textures_;  // color buffers when rendering offscreen
    std::vector<VkSemaphore> semaphores_;
};

}  // namespace holoscan::viz

#endif /* HOLOSCAN_VIZ_VULKAN_FRAMEBUFFER_SEQUENCE_HPP */
