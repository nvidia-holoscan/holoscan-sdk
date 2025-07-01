/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "data_logger.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <any>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "component.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/component.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/data_logger.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/resources/data_logger.hpp"
#include "kwarg_handling.hpp"
#include "data_logger_pydoc.hpp"

namespace py = pybind11;

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

namespace holoscan {

// === Python binding initialization ===

void init_data_logger(py::module_& m) {
  // DataLogger base interface
  py::class_<DataLogger, std::shared_ptr<DataLogger>> data_logger_class(
      m, "DataLogger", py::dynamic_attr(), doc::DataLogger::doc_DataLogger);

  // DataLoggerResource implementation
  py::class_<DataLoggerResource, DataLogger, Resource, std::shared_ptr<DataLoggerResource>>(
      m, "DataLoggerResource", py::dynamic_attr(), "Resource-based data logger implementation");
}

}  // namespace holoscan
