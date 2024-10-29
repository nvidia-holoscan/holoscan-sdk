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

#include <pybind11/pybind11.h>

#include <memory>

#include "./executors_pydoc.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/executors/gxf/gxf_executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/graph.hpp"

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

PYBIND11_MODULE(_executors, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Executor Python Bindings
        -------------------------------------
        .. currentmodule:: _executors
    )pbdoc";

  py::class_<gxf::GXFExecutor, Executor, std::shared_ptr<gxf::GXFExecutor>>(
      m, "GXFExecutor", doc::GXFExecutor::doc_GXFExecutor)
      .def(py::init<Fragment*>(), "app"_a, doc::GXFExecutor::doc_GXFExecutor_app);
  // Note: context property and run method are inherited from Executor
}  // PYBIND11_MODULE
}  // namespace holoscan
