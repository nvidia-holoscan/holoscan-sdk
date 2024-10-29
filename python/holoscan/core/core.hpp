/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PYHOLOSCAN_CORE_CORE_HPP
#define PYHOLOSCAN_CORE_CORE_HPP

#include <pybind11/pybind11.h>

#include <memory>
#include <unordered_map>
#include <string>
#include <vector>

#include "holoscan/core/domain/tensor.hpp"

namespace py = pybind11;

namespace holoscan {

void init_component(py::module_&);
void init_condition(py::module_&);
void init_metadata(py::module_&);
void init_network_context(py::module_&);
void init_resource(py::module_&);
void init_scheduler(py::module_&);
void init_executor(py::module_&);
void init_fragment(py::module_&);
void init_application(py::module_&);
void init_data_flow_tracker(py::module_&);
void init_cli(py::module_&);

}  // namespace holoscan

#endif /* PYHOLOSCAN_CORE_CORE_HPP */
