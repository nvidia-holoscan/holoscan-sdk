/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <utility>

#include <holoscan/operators/holoviz/holoviz_op.hpp>

int holoviz_operator_smoke() {
  ::holoviz::LayerStack layers;
  holoscan::holoviz::HolovizOp op{std::move(layers),
                                  ::holoviz::WindowSinkDesc{.title = "installed-consumer"}};
  static_cast<void>(op.layer_input(0));
  return 0;
}
