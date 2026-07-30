/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_CORE_CORE_HPP
#define PYHOLOSCAN_CORE_CORE_HPP

#include <pybind11/pybind11.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>

namespace py = pybind11;

namespace holoscan {

void init_component(py::module_&);
void init_condition(py::module_&);
void init_metadata(py::module_&);
void init_messagelabel(py::module_&);
void init_network_context(py::module_&);
void init_resource(py::module_&);
void init_clock(py::module_&);
void init_data_logger(py::module_&);
void init_scheduler(py::module_&);
void init_executor(py::module_&);
void init_fragment(py::module_&);
void init_subgraph(py::module_&);
void init_application(py::module_&);
void init_data_flow_tracker(py::module_&);
void init_cli(py::module_&);

}  // namespace holoscan

#endif /* PYHOLOSCAN_CORE_CORE_HPP */
