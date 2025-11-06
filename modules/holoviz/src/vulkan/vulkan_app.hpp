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

#ifndef MODULES_HOLOVIZ_SRC_VULKAN_VULKAN_APP_HPP
#define MODULES_HOLOVIZ_SRC_VULKAN_VULKAN_APP_HPP

#include <cuda.h>
#include <nvmath/nvmath_types.h>
#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../holoviz/display_event_type.hpp"
#include "../holoviz/image_format.hpp"
#include "../holoviz/present_mode.hpp"
#include "../holoviz/surface_format.hpp"
#include "../window.hpp"

namespace holoscan::viz {

class Layer;
class Texture;  ///< texture object
class Buffer;   ///< buffer object
class CudaService;

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

  /**
   * Setup Vulkan using the given window.
   *
   * @param window    window to use
   * @param font_path path to font file for text rendering, if not set the default font is used
   * @param font_size_in_pixels size of the font bitmaps
   */
  void setup(Window* window, const std::string& font_path, float font_size_in_pixels);

  /**
   * @return the window used by Vulkan
   */
  Window* get_window() const;

  /**
   * @return the CUDA service (created to use the same device as Vulkan)
   */
  CudaService* get_cuda_service() const;

  /**
   * @return the Vulkan device
   */
  vk::Device get_device() const;

  /**
   * Get the supported surface formats.
   *
   * @returns supported surface formats
   */
  std::vector<SurfaceFormat> get_surface_formats() const;

  /**
   * Set the surface format.
   *
   * The surface format can be changed any time.
   *
   * @param surface_format surface format
   */
  void set_surface_format(SurfaceFormat surface_format);

  /**
   * Get the supported present modes.
   *
   * @returns supported present modes
   */
  std::vector<PresentMode> get_present_modes() const;

  /**
   * Set the present mode.
   *
   * The present mode can be changed any time.
   *
   * @param present_mode present mode
   */
  void set_present_mode(PresentMode present_mode);

  /**
   * Get the supported image formats.
   *
   * @returns supported image formats
   */
  std::vector<ImageFormat> get_image_formats() const;

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
   * @return vk::CommandBuffer
   */
  vk::CommandBuffer get_command_buffer();

  /**
   * Set the viewport for subsequent draw commands.
   *
   * @param x, y              the viewport's upper left corner [0 ... 1]
   * @param width, height     the viewport's size [0 ... 1]
   */
  void set_viewport(float x, float y, float width, float height);

  /**
   * Arguments for create_texture()
   */
  struct CreateTextureArgs {
    uint32_t width_;                           //< texture width
    uint32_t height_;                          //< texture height
    ImageFormat format_;                       //< texture format
    vk::ComponentMapping component_mapping_;   //< component mapping
    vk::Filter filter_ = vk::Filter::eLinear;  //< texture filter
    bool normalized_ = true;     //< if true, then texture coordinates are normalize (0...1), else
                                 //< (0...width, 0...height)
    bool cuda_interop_ = false;  //< used for interop with CUDA
    vk::SamplerYcbcrModelConversion ycbcr_model_conversion_ =
        vk::SamplerYcbcrModelConversion::eYcbcr601;  ///< YCbCr model conversion
    vk::SamplerYcbcrRange ycbcr_range_ = vk::SamplerYcbcrRange::eItuFull;  ///< YCbCR range
    vk::ChromaLocation x_chroma_location_ =
        vk::ChromaLocation::eCositedEven;  ///< chroma location in x direction for formats which
                                           ///< are chroma downsampled in width (420 and 422)
    vk::ChromaLocation y_chroma_location_ =
        vk::ChromaLocation::eCositedEven;  ///< chroma location in y direction for formats which
                                           ///< are chroma downsampled in height (420)
  };

  /**
   * Create a Texture using host data.
   *
   * @param args arguments
   * @return created texture object
   */
  std::unique_ptr<Texture> create_texture(const CreateTextureArgs& args);

  /**
   * Upload data from host memory to a texture created with ::create_texture
   *
   * @param texture       texture to be updated
   * @param host_ptr      data in host memory to upload for the planes
   * @param row_pitch     the number of bytes between each row for the planes, if zero then data is
   * assumed to be contiguous in memory
   */
  void upload_to_texture(Texture* texture, const std::array<const void*, 3>& host_ptr,
                         const std::array<size_t, 3>& row_pitch);

  /**
   * Upload data from a Buffer to a texture
   *
   * @param texture texture to be updated
   * @param buffers data to be uploaded for each plane
   */
  void upload_to_texture(Texture* texture, const std::array<Buffer*, 3>& buffers);

  /**
   * Create a vertex or index buffer to be used for interop with CUDA, see ::upload_texture.
   *
   * @param data_size     size of the buffer in bytes
   * @param usage         buffer usage
   * @return created buffer
   */
  std::unique_ptr<Buffer> create_buffer_for_cuda_interop(size_t data_size,
                                                         vk::BufferUsageFlags usage);

  /**
   * Create a vertex or index buffer and initialize with data.
   *
   * @param data_size     size of the buffer in bytes
   * @param data          host size data to initialize buffer with or nullptr
   * @param usage         buffer usage
   * @return created buffer
   */
  std::unique_ptr<Buffer> create_buffer(size_t data_size, const void* data,
                                        vk::BufferUsageFlags usage);

  /**
   * Upload data from CUDA device memory to a buffer created with ::create_buffer_for_cuda_interop
   *
   * @param data_size   data size
   * @param device_ptr  CUDA device memory
   * @param buffer      buffer to be updated
   * @param dst_offset  offset in buffer to copy to
   * @param ext_stream  CUDA stream to use for operations
   */
  void upload_to_buffer(size_t data_size, CUdeviceptr device_ptr, Buffer* buffer, size_t dst_offset,
                        CUstream ext_stream);

  /**
   * Upload data from host memory to a buffer created with ::CreateBuffer
   *
   * @param data_size data size
   * @param data      host memory data buffer pointer
   * @param buffer    buffer to be updated
   */
  void upload_to_buffer(size_t data_size, const void* data, const Buffer* buffer);

  /**
   * Draw a texture with an optional depth texture and color lookup table.
   *
   * @param texture     texture to draw
   * @param depth_texture depth texture to draw, can be nullptr
   * @param lut         lookup table, can be nullptr
   * @param opacity     opacity, 0.0 is transparent, 1.0 is opaque
   * @param view_matrix view matrix
   */
  void draw_texture(Texture* texture, Texture* depth_texture, Texture* lut, float opacity,
                    const nvmath::mat4f& view_matrix = nvmath::mat4f(1));

  /**
   * Draw geometry.
   *
   * @param topology       topology
   * @param count          vertex count
   * @param first          first vertex
   * @param vertex_buffers vertex buffers
   * @param opacity        opacity, 0.0 is transparent, 1.0 is opaque
   * @param color          color
   * @param point_size     point size
   * @param line_width     line width
   * @param view_matrix    view matrix
   */
  void draw(vk::PrimitiveTopology topology, uint32_t count, uint32_t first,
            const std::vector<Buffer*>& vertex_buffers, float opacity,
            const std::array<float, 4>& color, float point_size, float line_width,
            const nvmath::mat4f& view_matrix = nvmath::mat4f(1));

  /**
   * Draw indexed triangle list geometry. Used to draw ImGui draw lists.
   *
   * @param desc_set          descriptor set for texture atlas
   * @param vertex_buffer     vertex buffer
   * @param index_buffer      index buffer
   * @param index_type        index type
   * @param index_count       index count
   * @param first_index       first index
   * @param vertex_offset     vertex offset
   * @param opacity           opacity, 0.0 is transparent, 1.0 is opaque
   * @param view_matrix       view matrix
   */
  void draw_imgui(vk::DescriptorSet desc_set, Buffer* vertex_buffer, Buffer* index_buffer,
                  vk::IndexType index_type, uint32_t index_count, uint32_t first_index,
                  uint32_t vertex_offset, float opacity,
                  const nvmath::mat4f& view_matrix = nvmath::mat4f(1));

  /**
   * Draw indexed geometry.
   *
   * @param topology
   * @param vertex_buffers  vertex buffers
   * @param index_buffer    index buffer
   * @param index_type      index type
   * @param index_count     index count
   * @param first_index     first index
   * @param vertex_offset   vertex offset
   * @param opacity         opacity, 0.0 is transparent, 1.0 is opaque
   * @param color           color
   * @param point_size      point size
   * @param line_width      line width
   * @param view_matrix     view matrix
   */
  void draw_indexed(vk::PrimitiveTopology topology, const std::vector<Buffer*>& vertex_buffers,
                    Buffer* index_buffer, vk::IndexType index_type, uint32_t index_count,
                    uint32_t first_index, uint32_t vertex_offset, float opacity,
                    const std::array<float, 4>& color, float point_size, float line_width,
                    const nvmath::mat4f& view_matrix = nvmath::mat4f(1));

  /**
   * Read the framebuffer and store it to cuda device memory.
   *
   * Can only be called outside of Begin()/End().
   *
   * @param fmt           image format, currently only R8G8B8A8_UNORM is supported.
   * @param width, height width and height of the region to read back, will be limited to the
   *                      framebuffer size if the framebuffer is smaller than that
   * @param buffer_size   size of the storage buffer in bytes
   * @param buffer        pointer to CUDA device memory to store the framebuffer into
   * @param ext_stream    CUDA stream to use for operations
   * @param row_pitch     the number of bytes between each row, if zero then data is assumed to be
   * contiguous in memory
   */
  void read_framebuffer(ImageFormat fmt, uint32_t width, uint32_t height, size_t buffer_size,
                        CUdeviceptr buffer, CUstream ext_stream, size_t row_pitch);

  /**
   * Block until either the present_id is greater than or equal to the current present id, or
   * timeout_ns nanoseconds passes. The present ID is initially zero and increments after each
   * present.
   *
   * @param present_id the presentation presentId to wait for
   * @param timeout_ns timeout in nanoseconds
   * @return true if the present_id is greater than or equal to the current present id, false if
   * timeout_ns nanoseconds passes
   */
  bool wait_for_present(uint64_t present_id, uint64_t timeout_ns);

  /**
   * Block until either the display_event_type is signaled, or timeout_ns nanoseconds passes.
   *
   * @param display_event_type display event type
   * @param timeout_ns timeout in nanoseconds
   * @return true if the display_event_type is signaled, false if timeout_ns nanoseconds passes
   */
  bool wait_for_display_event(DisplayEventType display_event_type, uint64_t timeout_ns);

  /**
   * @returns the counter incrementing once every time a vertical blanking period occurs on the
   * display associated window or the display selected in full screen mode or exclusive display
   * mode.
   */
  uint64_t get_vblank_counter();

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace holoscan::viz

#endif /* MODULES_HOLOVIZ_SRC_VULKAN_VULKAN_APP_HPP */
