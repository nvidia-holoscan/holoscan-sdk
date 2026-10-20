/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

namespace holoscan {

/// @brief Add private registry metadata helpers to the native module.
void init_registry(nanobind::module_& module);

}  // namespace holoscan
