/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_CORE_IO_SPEC_HPP
#define PYHOLOSCAN_CORE_IO_SPEC_HPP

#include <pybind11/pybind11.h>

#include <unordered_map>

#include "holoscan/core/io_spec.hpp"

namespace py = pybind11;

namespace holoscan {

void init_io_spec(py::module_&);

static const std::unordered_map<IOSpec::IOType, const char*> io_type_namemap{
    {IOSpec::IOType::kInput, "INPUT"},
    {IOSpec::IOType::kOutput, "OUTPUT"},
};

static const std::unordered_map<IOSpec::ConnectorType, const char*> connector_type_namemap{
    {IOSpec::ConnectorType::kDefault, "DEFAULT"},
    {IOSpec::ConnectorType::kDoubleBuffer, "DOUBLE_BUFFER"},
    {IOSpec::ConnectorType::kUCX, "UCX"},
};

}  // namespace holoscan

#endif /* PYHOLOSCAN_CORE_IO_SPEC_HPP */
