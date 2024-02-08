/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "execution_context.hpp"

#include <pybind11/pybind11.h>

#include <memory>

#include "execution_context_pydoc.hpp"
#include "holoscan/core/execution_context.hpp"

using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan {

void init_execution_context(py::module_& m) {
  py::class_<ExecutionContext, std::shared_ptr<ExecutionContext>>(
      m, "ExecutionContext", doc::ExecutionContext::doc_ExecutionContext);

  py::class_<PyExecutionContext, ExecutionContext, std::shared_ptr<PyExecutionContext>>(
      m, "PyExecutionContext", R"doc(Execution context class.)doc")
      .def_property_readonly("input", &PyExecutionContext::py_input)
      .def_property_readonly("output", &PyExecutionContext::py_output);
}
}  // namespace holoscan
