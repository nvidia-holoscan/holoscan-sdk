/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
  void end(Vulkan* vulkan) override;
  void render(Vulkan* vulkan) override;
  ///@}

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace holoscan::viz

#endif /* HOLOSCAN_VIZ_LAYERS_IM_GUI_LAYER_HPP */
