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

#ifndef HOLOVIZ_SRC_LAYERS_LAYER_HPP
#define HOLOVIZ_SRC_LAYERS_LAYER_HPP

#include <nvmath/nvmath.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace holoscan::viz {

class Vulkan;

/**
 * The base class for all layers.
 */
class Layer {
 public:
  /**
   * Layer types
   */
  enum class Type {
    Image,     ///< image layer
    Geometry,  ///< geometry layer
    ImGui      ///< ImBui layer
  };

  /**
   * Construct a new Layer object.
   *
   * @param type  layer type
   */
  explicit Layer(Type type);
  Layer() = delete;

  /**
   * Destroy the Layer object.
   */
  virtual ~Layer();

  /**
   * @returns  the layer type
   */
  Type get_type() const;

  /**
   * @returns  the layer priority
   */
  int32_t get_priority() const;

  /**
   * Set the layer priority.
   *
   * @param priority  new layer priority
   */
  void set_priority(int32_t priority);

  /**
   * @returns  the layer opacity
   */
  float get_opacity() const;

  /**
   * Set the layer opacity.
   *
   * @param opacity   new layer opacity
   */
  virtual void set_opacity(float opacity);

  /**
   * Describes a view for which a layer can be rendered.
   */
  struct View {
    /// offset of top-left corner. Top left coordinate is (0, 0) bottom right coordinate is (1, 1).
    float offset_x = 0.f, offset_y = 0.f;
    /// width and height of the layer in normalized range. 1.0 is full size.
    float width = 1.f, height = 1.f;

    /// transform matrix
    std::optional<nvmath::mat4f> matrix;
  };

  /**
   * @returns  the layer views
   */
  const std::vector<View>& get_views() const;

  /**
   * Set the layer views.
   *
   * @param views layer views to add
   */
  void set_views(const std::vector<View>& views);

  /**
   * Add a layer view.
   *
   * @param views layer view to add
   */
  void add_view(const View& view);

  /**
   * Checks if a layer can be reused (properties have to match).
   *
   * @param other layer which is to be checked for re-usability
   */
  virtual bool can_be_reused(Layer& other) const;

  /**
   * End layer construction. Upload data.
   *
   * @param vulkan    vulkan instance to use for updating data
   */
  virtual void end(Vulkan* vulkan) {}

  /**
   * Render the layer.
   *
   * @param vulkan    vulkan instance to use for drawing
   */
  virtual void render(Vulkan* vulkan) = 0;

 protected:
  const Type type_;  ///< layer type

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace holoscan::viz

#endif /* HOLOVIZ_SRC_LAYERS_LAYER_HPP */
