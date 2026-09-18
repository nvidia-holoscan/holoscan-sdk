/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nanobind/nanobind.h>

#include "tensor.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays)
NB_MODULE(_core, module) {
  module.doc() = "Minimal Holoscan SDK core Python bindings.";

  holoscan::init_tensor(module);
}
// NOLINTEND(cppcoreguidelines-avoid-c-arrays)
