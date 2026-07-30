/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_CORE_CLOCK_HPP
#define PYHOLOSCAN_CORE_CLOCK_HPP

#include <pybind11/pybind11.h>

#include <holoscan/core/clock.hpp>

namespace py = pybind11;

namespace holoscan {

void init_clock(py::module_&);

int64_t get_duration_ns(const py::object& duration);

}  // namespace holoscan

#endif /* PYHOLOSCAN_CORE_CLOCK_HPP */
