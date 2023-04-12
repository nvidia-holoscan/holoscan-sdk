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

#ifndef HOLOSCAN_VIZ_LAYERS_IM_GUI_LAYER_HPP
#define HOLOSCAN_VIZ_LAYERS_IM_GUI_LAYER_HPP

#include <cstdint>
#include <memory>

#include "layer.hpp"

namespace holoscan::viz {

/**
 * Layer specialication for ImGui rendering.
 */
class ImGuiLayer : public Layer {
 public:
  /**
   * Construct a new ImGuiLayer object.
   */
  ImGuiLayer();

  /**
   * Destroy the ImGuiLayer object.
   */
  ~ImGuiLayer();

  /// holoscan::viz::Layer virtual members
  ///@{
  void set_opacity(float opacity) override;
  void render(Vulkan* vulkan) override;
  ///@}

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace holoscan::viz

#endif /* HOLOSCAN_VIZ_LAYERS_IM_GUI_LAYER_HPP */
