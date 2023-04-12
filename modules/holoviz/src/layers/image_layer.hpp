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
   * Defines the image data for this layer, source is Cuda device memory.
   *
   * If the image has a alpha value it's multiplied with the layer opacity.
   *
   * @param width         width of the image
   * @param height        height of the image
   * @param fmt           image format
   * @param device_ptr    Cuda device memory pointer
   */
  void image_cuda_device(uint32_t width, uint32_t height, ImageFormat fmt, CUdeviceptr device_ptr);

  /**
   * Defines the image data for this layer, source is a Cuda array.
   *
   * If the image has a alpha value it's multiplied with the layer opacity.
   *
   * @param fmt       image format
   * @param array     Cuda array
   */
  void image_cuda_array(ImageFormat fmt, CUarray array);

  /**
   * Defines the image data for this layer, source is a Cuda array.
   *
   * If the image has a alpha value it's multiplied with the layer opacity.
   *
   * @param fmt       image format
   * @param array     Cuda array
   */
  void image_host(uint32_t width, uint32_t height, ImageFormat fmt, const void* data);

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
