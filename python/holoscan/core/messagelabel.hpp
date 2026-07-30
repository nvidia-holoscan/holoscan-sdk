/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_CORE_MESSAGELABEL_HPP
#define PYHOLOSCAN_CORE_MESSAGELABEL_HPP

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace holoscan {

void init_messagelabel(py::module_&);

}  // namespace holoscan

#endif /* PYHOLOSCAN_CORE_MESSAGELABEL_HPP */
