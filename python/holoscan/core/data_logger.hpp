/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_CORE_DATA_LOGGER_HPP
#define PYHOLOSCAN_CORE_DATA_LOGGER_HPP

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace holoscan {

void init_data_logger(py::module_& m);

}  // namespace holoscan

#endif /* PYHOLOSCAN_CORE_DATA_LOGGER_HPP */
