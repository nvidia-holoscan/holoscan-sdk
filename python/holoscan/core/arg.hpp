/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_CORE_ARG_HPP
#define PYHOLOSCAN_CORE_ARG_HPP

#include <pybind11/pybind11.h>

#include <unordered_map>

#include <holoscan/core/arg.hpp>

namespace py = pybind11;

namespace holoscan {

void init_arg(py::module_&);

}  // namespace holoscan

#endif /* PYHOLOSCAN_CORE_ARG_HPP */
