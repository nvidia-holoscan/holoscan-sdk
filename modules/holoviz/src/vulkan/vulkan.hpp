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

#ifndef HOLOSCAN_VIZ_VULKAN_VULKAN_HPP
#define HOLOSCAN_VIZ_VULKAN_VULKAN_HPP

#include <cuda.h>
#include <vulkan/vulkan_core.h>

#include <array>
#include <cstdint>
#include <list>
#include <memory>

#include "../holoviz/image_format.hpp"
#include "../window.hpp"

namespace holoscan::viz {

class Layer;

/**
 * The class is responsible for all operations regarding Vulkan.
 */
class Vulkan {
 public:
    /**
     * Construct a new Vulkan object.
     */
    Vulkan();

    /**
     * Destroy the Vulkan object.
     */
    ~Vulkan();

    struct Texture;  ///< texture object
    struct Buffer;   ///< buffer object

    /**
     * Setup Vulkan using the given window.
     *
     * @param window    window to use
     */
    void setup(Window *window);

    /**
     * Begin the transfer pass. This creates a transfer job and a command buffer.
     */
    void begin_transfer_pass();

    /**
     * End the transfer pass. This ends and submits the transfer command buffer.
     * It's an error to call end_transfer_pass()
     * without begin_transfer_pass().
     */
    void end_transfer_pass();

    /**
     * Begin the render pass. This acquires the next image to render to
     * and sets up the render command buffer.
     */
    void begin_render_pass();

    /**
     * End the render pass. Submits the render command buffer.
     */
    void end_render_pass();

    /**
     * Get the command buffer for the current frame.
     *
     * @return VkCommandBuffer
     */
    VkCommandBuffer get_command_buffer();

    /**
     * Create a texture to be used for upload from Cuda data, see ::UploadToTexture.
     * Destroy with ::DestroyTexture.
     *
     * @param width, height     size
     * @param format            texture format
     * @param filter            texture filter
     * @param normalized        if true, then texture coordinates are normalize (0...1), 
     *                             else (0...width, 0...height)
     * @return created texture object
     */
    Texture *create_texture_for_cuda_upload(uint32_t width, uint32_t height, ImageFormat format,
                                            VkFilter filter = VK_FILTER_LINEAR,
                                            bool normalized = true);

    /**
     * Create a Texture using host data. Destroy with ::DestroyTexture.
     *
     * @param width, height     size
     * @param format            texture format
     * @param data_size         data size in bytes
     * @param data              texture data
     * @param filter            texture filter
     * @param normalized        if true, then texture coordinates are normalize (0...1),
     *                             else (0...width, 0...height)
     * @return created texture object
     */
    Texture *create_texture(uint32_t width, uint32_t height, ImageFormat format, size_t data_size,
                            const void *data, VkFilter filter = VK_FILTER_LINEAR,
                            bool normalized = true);

    /**
     * Destroy a texture created with ::CreateTextureForCudaUpload or ::CreateTexture.
     *
     * @param texture   texture to destroy
     */
    void destroy_texture(Texture *texture);

    /**
     * Upload data from Cuda device memory to a texture created with ::CreateTextureForCudaUpload
     *
     * @param device_ptr    Cuda device memory
     * @param texture       texture to be updated
     * @param stream        Cuda stream
     */
    void upload_to_texture(CUdeviceptr device_ptr, Texture *texture, CUstream stream = 0);

    /**
     * Upload data from host memory to a texture created with ::CreateTexture
     *
     * @param host_ptr      data to upload in host memory
     * @param texture       texture to be updated
     */
    void upload_to_texture(const void *host_ptr, Texture *texture);

    /**
     * Draw a texture with an optional color lookup table.
     *
     * @param texture   texture to draw
     * @param lut       lookup table, can be nullptr
     * @param opacity   opacity, 0.0 is transparent, 1.0 is opaque
     */
    void draw_texture(Texture *texture, Texture *lut, float opacity);

    /**
     * Create a vertex or index buffer and initialize with data. Destroy with ::DestroyBuffer.
     *
     * @param data_size     size of the buffer in bytes
     * @param data          host size data to initialize buffer with.
     * @param usage         buffer usage
     * @return created buffer
     */
    Buffer *create_buffer(size_t data_size, const void *data, VkBufferUsageFlags usage);

    /**
     * Destroy a buffer created with ::CreateBuffer.
     *
     * @param buffer    buffer to destroy
     */
    void destroy_buffer(Buffer *buffer);

    /**
     * Draw geometry.
     *
     * @param topology      topology
     * @param count         vertex count
     * @param first         first vertex
     * @param buffer        vertex buffer
     * @param opacity       opacity, 0.0 is transparent, 1.0 is opaque
     * @param color         color
     * @param point_size    point size
     * @param line_width    line width
     */
    void draw(VkPrimitiveTopology topology, uint32_t count, uint32_t first, Buffer *buffer,
              float opacity, const std::array<float, 4> &color,
              float point_size, float line_width);

    /**
     * Draw indexed triangle list geometry. Used to draw ImGui draw list for text drawing.
     *
     * @param desc_set          descriptor set for texture atlas
     * @param vertex_buffer     vertex buffer
     * @param index_buffer      index buffer
     * @param index_type        index type
     * @param index_count       index count
     * @param first_index       first index
     * @param vertex_offset     vertex offset
     * @param opacity           opacity, 0.0 is transparent, 1.0 is opaque
     */
    void draw_indexed(VkDescriptorSet desc_set, Buffer *vertex_buffer, Buffer *index_buffer,
                      VkIndexType index_type, uint32_t index_count, uint32_t first_index,
                      uint32_t vertex_offset, float opacity);

    /**
     * Read the framebuffer and store it to cuda device memory.
     *
     * Can only be called outside of Begin()/End().
     *
     * @param fmt          image format, currently only R8G8B8A8_UNORM is supported.
     * @param buffer_size  size of the storage buffer in bytes
     * @param buffer       pointer to Cuda device memory to store the framebuffer into
     * @param stream       Cuda stream
     */
    void read_framebuffer(ImageFormat fmt, size_t buffer_size, CUdeviceptr buffer,
                                                             CUstream stream = 0);

 private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace holoscan::viz

#endif /* HOLOSCAN_VIZ_VULKAN_VULKAN_HPP */
