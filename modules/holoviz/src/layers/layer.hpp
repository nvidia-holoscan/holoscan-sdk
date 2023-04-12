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

#ifndef HOLOSCAN_VIZ_LAYERS_LAYER_HPP
#define HOLOSCAN_VIZ_LAYERS_LAYER_HPP

#include <cstdint>
#include <memory>

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

#endif /* HOLOSCAN_VIZ_LAYERS_LAYER_HPP */
