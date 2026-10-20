/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

namespace holoscan {

namespace nb = nanobind;

void init_channel(nb::module_& module);

}  // namespace holoscan
