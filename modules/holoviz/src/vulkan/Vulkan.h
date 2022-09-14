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

#pragma once

#include "../Window.h"
#include "../holoviz/ImageFormat.h"

#include <array>
#include <cstdint>
#include <list>
#include <memory>

#include <cuda.h>
#include <vulkan/vulkan_core.h>

namespace clara::holoviz
{

class Layer;

/**
 * The class is responsible for all operations regarding Vulkan.
 */
class Vulkan
{
public:
    /**
     * Construct a new Vulkan object.
     */
    Vulkan();

    /**
     * Destroy the Vulkan object.
     */
    ~Vulkan();

    struct Texture; ///< texture object
    struct Buffer;  ///< buffer object

    /**
     * Setup Vulkan using the given window.
     *
     * @param window    window to use
     */
    void Setup(Window *window);

    /**
     * Submit the frame with the recorded layers.
     *
     * @param layers    recorded layers
     */
    void SubmitFrame(const std::list<std::unique_ptr<Layer>> &layers);

    /**
     * Get the command buffer for the current frame.
     *
     * @return VkCommandBuffer
     */
    VkCommandBuffer GetCommandBuffer();

    /**
     * Create a texure to be used for upload from Cuda data, see ::UploadToTexture. Destory with ::DestroyTexture.
     *
     * @param width, height     size
     * @param format            texture format
     * @param filter            texture filter
     * @param normalized        if true, then texture coordinates are normalize (0...1), else (0...width, 0...height)
     * @return created texture object
     */
    Texture *CreateTextureForCudaUpload(uint32_t width, uint32_t height, ImageFormat format,
                                        VkFilter filter = VK_FILTER_LINEAR, bool normalized = true);

    /**
     * Create a Texture using host data. Destory with ::DestroyTexture.
     *
     * @param width, height     size
     * @param format            texture format
     * @param data_size         data size in bytes
     * @param data              texture data
     * @param filter            texture filter
     * @param normalized        if true, then texture coordinates are normalize (0...1), else (0...width, 0...height)
     * @return created texture object
     */
    Texture *CreateTexture(uint32_t width, uint32_t height, ImageFormat format, size_t data_size, const void *data,
                           VkFilter filter = VK_FILTER_LINEAR, bool normalized = true);

    /**
     * Destroy a texture created with ::CreateTextureForCudaUpload or ::CreateTexture.
     *
     * @param texture   texture to destroy
     */
    void DestroyTexture(Texture *texture);

    /**
     * Upload data from Cuda device memory to a texture created with ::CreateTextureForCudaUpload
     *
     * @param device_ptr    Cuda device memory
     * @param texture       texture to be updated
     */
    void UploadToTexture(CUdeviceptr device_ptr, const Texture *texture);

    /**
     * Draw a texture with an otional color lookup table.
     *
     * @param texture   texture to draw
     * @param lut       lookup table, can be nullptr
     * @param opacity   opacity, 0.0 is transparent, 1.0 is opaque
     */
    void DrawTexture(const Texture *texture, const Texture *lut, float opacity);

    /**
     * Create a vertex or index buffer and initialize with data. Destroy with ::DestroyBuffer.
     *
     * @param data_size     size of the buffer in bytes
     * @param data          host size data to initialize buffer with.
     * @param usage         buffer usage
     * @return created buffer
     */
    Buffer *CreateBuffer(size_t data_size, const void *data, VkBufferUsageFlags usage);

    /**
     * Destory a buffer created with ::CreateBuffer.
     *
     * @param buffer    buffer to destroy
     */
    void DestroyBuffer(Buffer *buffer);

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
    void Draw(VkPrimitiveTopology topology, uint32_t count, uint32_t first, const Buffer *buffer, float opacity,
              const std::array<float, 4> &color, float point_size, float line_width);

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
    void DrawIndexed(VkDescriptorSet desc_set, const Buffer *vertex_buffer, const Buffer *index_buffer,
                     VkIndexType index_type, uint32_t index_count, uint32_t first_index, uint32_t vertex_offset,
                     float opacity);

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace clara::holoviz
