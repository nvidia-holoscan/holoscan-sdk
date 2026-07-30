/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOVIZ_SRC_LAYERS_GEOMETRY_LAYER_HPP
#define HOLOVIZ_SRC_LAYERS_GEOMETRY_LAYER_HPP

#include <cuda.h>

#include <cstdint>
#include <memory>

#include "layer.hpp"

#include "../holoviz/depth_map_render_mode.hpp"
#include "../holoviz/image_format.hpp"
#include "../holoviz/primitive_topology.hpp"

namespace holoscan::viz {

/**
 * Layer specialication for geometry rendering.
 */
class GeometryLayer : public Layer {
 public:
  /**
   * Construct a new GeometryLayer object.
   */
  GeometryLayer();

  /**
   * Destroy the GeometryLayer object.
   */
  ~GeometryLayer();

  /**
   * Set the color for following geometry.
   *
   * @param r,g,b,a RGBA color. Default (1.0, 1.0, 1.0, 1.0).
   */
  void color(float r, float g, float b, float a);

  /**
   * Set the line width for geometry made of lines.
   *
   * @param width line width in pixels. Default 1.0.
   */
  void line_width(float width);

  /**
   * Set the point size for geometry made of points.
   *
   * @param size  point size in pixels. Default 1.0.
   */
  void point_size(float size);

  /**
   * Draw a geometric primitive.
   *
   * @param topology          primitive topology
   * @param primitive_count   primitive count
   * @param data_size         size of the data array in floats
   * @param data              pointer to data, the format and size of the array
   *                          depends on the primitive count and topology
   */
  void primitive(PrimitiveTopology topology, uint32_t primitive_count, size_t data_size,
                 const float* data);

  /**
   * Draw a geometric primitive, source is CUDA device memory.
   *
   * @param topology          primitive topology
   * @param primitive_count   primitive count
   * @param data_size         size of the data array in floats
   * @param data              CUDA device memory pointer to data, the format and size of the array
   *                          depends on the primitive count and topology
   */
  void primitive_cuda_device(PrimitiveTopology topology, uint32_t primitive_count, size_t data_size,
                             CUdeviceptr data);

  /**
   * Draw text.
   *
   * @param x     x coordinate
   * @param y     y coordinate
   * @param size  font size
   * @param text  text to draw
   */
  void text(float x, float y, float size, const char* text);

  /**
   * Render a depth map.
   *
   * Supported depth formats are: R8_UNORM, D32_SFLOAT.
   *
   * @param render_mode       depth map render mode
   * @param width             width of the depth map
   * @param height            height of the depth map
   * @param depth_fmt         format of the depth map data
   * @param depth_device_ptr  Cuda device memory pointer holding the depth data
   * @param color_fmt         format of the color data (has to be ImageFormat::R8G8B8A8_UNORM)
   * @param color_device_ptr  Cuda device memory pointer holding the color data (optional)
   */
  void depth_map(DepthMapRenderMode render_mode, uint32_t width, uint32_t height,
                 ImageFormat depth_fmt, CUdeviceptr depth_device_ptr, ImageFormat color_fmt,
                 CUdeviceptr color_device_ptr);

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

#endif /* HOLOVIZ_SRC_LAYERS_GEOMETRY_LAYER_HPP */
