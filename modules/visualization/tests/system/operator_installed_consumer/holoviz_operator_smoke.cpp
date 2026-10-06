/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <utility>

#include <holoscan/visualization/holoviz_op.hpp>

int holoviz_operator_smoke() {
  holoscan::visualization::LayerStack layers;
  layers.add_image_layer("smoke",
                         holoscan::visualization::ImageLayerDesc{
                             holoscan::visualization::ImageFormat::kR8Unorm, 1U, 1U});
  holoscan::visualization::HolovizOp op{
      std::move(layers),
      holoscan::visualization::WindowSinkDesc{.title = "installed-consumer"},
      holoscan::visualization::ClockTrigger{}};
  static_cast<void>(op.image_layer_input(0));
  return 0;
}
