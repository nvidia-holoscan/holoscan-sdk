// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>

#include <holoscan/visualization/operator_catalog.hpp>

namespace nb = nanobind;

// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays)
NB_MODULE(_visualization, module) {
  // This private extension initializes the provider catalog; the public facade is pure Python.
  auto registered = holoscan::visualization::register_holoviz_operator_catalog();
  if (!registered) {
    throw holoscan::RuntimeError{registered.error()};
  }
  module.doc() = "Holoscan Visualization operator-catalog initializer.";
}
// NOLINTEND(cppcoreguidelines-avoid-c-arrays)
