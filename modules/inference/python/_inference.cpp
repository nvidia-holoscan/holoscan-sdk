// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>

#include <holoscan/operators/inference/operator_catalog.hpp>

// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays)
// Nanobind fixes this initializer signature and passes its module handle by value.
// NOLINTNEXTLINE(performance-unnecessary-value-param)
NB_MODULE(_inference, module) {
  // This private extension initializes the provider catalog; the public facade is pure Python.
  auto registered = holoscan::ops::register_inference_operator_catalog();
  if (!registered) {
    throw holoscan::RuntimeError{registered.error()};
  }
  module.doc() = "Holoscan Inference operator-catalog initializer.";
}
// NOLINTEND(cppcoreguidelines-avoid-c-arrays)
