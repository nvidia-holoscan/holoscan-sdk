/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_VIZ_LAYERS_IMAGE_LAYER_HPP
#define HOLOSCAN_VIZ_LAYERS_IMAGE_LAYER_HPP

#include <cuda.h>

#include <cstdint>
#include <memory>

#include "../holoviz/image_format.hpp"

#include "layer.hpp"

namespace holoscan::viz {

/**
 * Layer specialication for image rendering.
 */
class ImageLayer : public Layer {
 public:
  /**
   * Construct a new ImageLayer object.
   */
  ImageLayer();

  /**
   * Destroy the ImageLayer object.
   */
  ~ImageLayer();

  /**
   * Defines the image data for this layer, source is CUDA device memory.
   *
   * If the image has a alpha value it's multiplied with the layer opacity.
   *
   * If fmt is a depth format, the image will be interpreted as a depth image, and will be written
   * to the depth buffer when rendering the color image from a separate invocation of image_*() for
   * the same layer. This enables depth-compositing image layers with other Holoviz layers.
   * Supported depth formats are: D16_UNORM, X8_D24_UNORM, D32_SFLOAT.
   *
   * Supports multi-planar images (e.g. YUV), `device_ptr` and `row_pitch` specify the parameters
   * for the first plane (plane 0), `device_ptr_n` and `row_pitch_n` for subsequent planes.
   *
   * @param width         width of the image
   * @param height        height of the image
   * @param fmt           image format
   * @param device_ptr    CUDA device memory pointer
   * @param row_pitch     the number of bytes between each row, if zero then data is
   * assumed to be contiguous in memory
   * @param device_ptr_plane_1    CUDA device memory pointer for plane 1
   * @param row_pitch_1     the number of bytes between each row for plane 1, if zero then data is
   * assumed to be contiguous in memory
   * @param device_ptr_plane_2    CUDA device memory pointer for plane 2
   * @param row_pitch_2     the number of bytes between each row for plane 2, if zero then data is
   * assumed to be contiguous in memory
   */
  void image_cuda_device(uint32_t width, uint32_t height, ImageFormat fmt, CUdeviceptr device_ptr,
                         size_t row_pitch = 0, CUdeviceptr device_ptr_plane_1 = 0,
                         size_t row_pitch_plane_1 = 0, CUdeviceptr device_ptr_plane_2 = 0,
                         size_t row_pitch_plane_2 = 0);

  /**
   * Defines the image data for this layer, source is a CUDA array.
   *
   * If the image has a alpha value it's multiplied with the layer opacity.
   *
   * If fmt is a depth format, the image will be interpreted as a depth image, and will be written
   * to the depth buffer when rendering the color image from a separate invocation of image_*() for
   * the same layer. This enables depth-compositing image layers with other Holoviz layers.
   * Supported depth formats are: D16_UNORM, X8_D24_UNORM, D32_SFLOAT.
   *
   * @param fmt       image format
   * @param array     CUDA array
   */
  void image_cuda_array(ImageFormat fmt, CUarray array);

  /**
   * Defines the image data for this layer, source is host memory.
   *
   * If the image has a alpha value it's multiplied with the layer opacity.
   *
   * If fmt is a depth format, the image will be interpreted as a depth image, and will be written
   * to the depth buffer when rendering the color image from a separate invocation of image_*() for
   * the same layer. This enables depth-compositing image layers with other Holoviz layers.
   * Supported depth formats are: D16_UNORM, X8_D24_UNORM, D32_SFLOAT.
   *
   * Supports multi-planar images (e.g. YUV), `device_ptr` and `row_pitch` specify the parameters
   * for the first plane (plane 0), `device_ptr_n` and `row_pitch_n` for subsequent planes.
   *
   * @param width     width of the image
   * @param height    height of the image
   * @param fmt       image format
   * @param data      host memory pointer
   * @param row_pitch the number of bytes between each row, if zero then data is assumed to be
   * contiguous in memory
   * @param data_plane_1      host memory pointer for plane 1
   * @param row_pitch_plane_1 the number of bytes between each row for plane 1, if zero then data is
   * assumed to be contiguous in memory
   * @param data_plane_2      host memory pointer for plane 2
   * @param row_pitch_plane_2 the number of bytes between each row for plane 2, if zero then data is
   * assumed to be contiguous in memory
   */
  void image_host(uint32_t width, uint32_t height, ImageFormat fmt, const void* data,
                  size_t row_pitch = 0, const void* data_plane_1 = nullptr,
                  size_t row_pitch_plane_1 = 0, const void* data_plane_2 = nullptr,
                  size_t row_pitch_plane_2 = 0);

  /**
   * Defines the lookup table for this image layer.
   *
   * If a lookup table is used the image format has to be a single channel integer or
   * float format (e.g. ::ImageFormat::R8_UINT, ::ImageFormat::R16_UINT, ::ImageFormat::R32_UINT,
   * ::ImageFormat::R8_UNORM, ::ImageFormat::R16_UNORM, ::ImageFormat::R32_SFLOAT).
   *
   * If normalized is 'true' the function processed is as follow
   * @code{.cpp}
   *  out = lut[clamp(in, 0.0, 1.0)]
   * @endcode
   * Input image values are clamped to the range of the lookup table size: `[0.0, 1.0[`.
   *
   * If normalized is 'false' the function processed is as follow
   * @code{.cpp}
   *  out = lut[clamp(in, 0, size)]
   * @endcode
   * Input image values are clamped to the range of the lookup table size: `[0.0, size[`.
   *
   * @param size      size of the lookup table in elements
   * @param fmt       lookup table color format
   * @param data_size size of the lookup table data in bytes
   * @param data      host memory pointer to lookup table data
   * @param normalized if true then the range of the lookup table is '[0.0, 1.0[', else it is
   * `[0.0, size[`
   */
  void lut(uint32_t size, ImageFormat fmt, size_t data_size, const void* data, bool normalized);

  /**
   * Specifies how the color components of an image are mapped to the color components of the
   * output. Output components can be set to the R, G, B or A component of the input or fixed to
   * zero or one or just identical to the input.
   *
   * Default: all output components are identical to the input components
   * (ComponentSwizzle::IDENTITY).
   *
   * This can be used display an image in color formats which are not natively supported by Holoviz.
   * For example to display a BGRA image:
   * @code{.cpp}
   *  image_component_mapping(ComponentSwizzle::B, ComponentSwizzle::G, ComponentSwizzle::R,
   *    ComponentSwizzle::A);
   *  ImageHost(width, height, ImageFormat::R8G8B8A8_UNORM, bgra_data);
   * @endcode
   * or to display a single component image in gray scale:
   * @code{.cpp}
   *  image_component_mapping(ComponentSwizzle::R, ComponentSwizzle::R, ComponentSwizzle::R,
   *    ComponentSwizzle::IDENTITY);
   *  ImageHost(width, height, ImageFormat::R8_UNORM, single_component_data);
   * @endcode
   *
   * @param r, g, b, a    sets how the component values are placed in each component of the output
   */
  void image_component_mapping(ComponentSwizzle r, ComponentSwizzle g, ComponentSwizzle b,
                               ComponentSwizzle a);

  /**
   * Specifies the YUV model conversion.
   *
   * @param yuv_model_conversion YUV model conversion. Default is `YUV_601`.
   */
  void image_yuv_model_conversion(YuvModelConversion yuv_model_conversion);

  /**
   * Specifies the YUV range.
   *
   * @param yuv_range YUV range. Default is `ITU_FULL`.
   */
  void image_yuv_range(YuvRange yuv_range);

  /**
   * Defines the location of downsampled chroma component samples relative to the luma samples.
   *
   * @param x_chroma_location chroma location in x direction for formats which are chroma
   * downsampled in width (420 and 422). Default is `COSITED_EVEN`.
   * @param y_chroma_location chroma location in y direction for formats which are chroma
   * downsampled in height (420). Default is `COSITED_EVEN`.
   */
  void image_chroma_location(ChromaLocation x_chroma_location, ChromaLocation y_chroma_location);

  /// holoscan::viz::Layer virtual members
  ///@{
  bool can_be_reused(Layer& other) const override;
  void end(Vulkan* vulkan) override;
  void render(Vulkan* vulkan) override;
  ///@}

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace holoscan::viz

#endif /* HOLOSCAN_VIZ_LAYERS_IMAGE_LAYER_HPP */
