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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "executor_pydoc.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/fragment.hpp"

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

class PyExecutor : public Executor {
 public:
  /* Inherit the constructors */
  using Executor::Executor;

  /* Trampolines (need one for each virtual function) */
  void run(OperatorGraph& graph) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Executor, run, graph);
  }
};

void init_executor(py::module_& m) {
  py::class_<Executor, PyExecutor, std::shared_ptr<Executor>>(
      m, "Executor", R"doc(Executor class.)doc")
      .def(py::init<Fragment*>(), "fragment"_a, doc::Executor::doc_Executor)
      .def("interrupt", &Executor::interrupt, doc::Executor::doc_interrupt)
      .def("run", &Executor::run, doc::Executor::doc_run)  // note: virtual function
      .def_property_readonly(
          "fragment", py::overload_cast<>(&Executor::fragment), doc::Executor::doc_fragment)
      .def_property("context",
                    py::overload_cast<>(&Executor::context),
                    py::overload_cast<void*>(&Executor::context),
                    doc::Executor::doc_context)
      .def_property("context_uint64",
                    py::overload_cast<>(&Executor::context_uint64),
                    py::overload_cast<uint64_t>(&Executor::context_uint64),
                    doc::Executor::doc_context_uint64);
}

}  // namespace holoscan
