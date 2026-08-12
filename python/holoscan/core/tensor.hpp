/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

namespace holoscan {

namespace nb = nanobind;

/// @brief Register the DLDevice, DLDataType, and Tensor bindings on the given module.
///
/// Phase 1 of the 5.x nanobind port: exposes the read-only metadata of the GXF-free
/// holoscan::Tensor. DLPack / array-interface protocols and the from-Python factories are added in
/// later phases (see h5x_plan_holoscan_tensor_python.md).
void init_tensor(nb::module_& m);

}  // namespace holoscan
