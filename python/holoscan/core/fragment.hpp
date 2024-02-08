/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYBIND11_CORE_FRAGMENT_HPP
#define PYBIND11_CORE_FRAGMENT_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <set>
#include <string>
#include <utility>

#include "application_pydoc.hpp"
#include "fragment_pydoc.hpp"
#include "holoscan/core/application.hpp"
#include "holoscan/core/config.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/network_context.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/scheduler.hpp"
#include "kwarg_handling.hpp"

namespace py = pybind11;

namespace holoscan {

void init_fragment(py::module_&);

/**********************************************************
 * Define trampolines for classes with virtual functions. *
 **********************************************************
 *
 * see:
 *https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python
 *
 */

class PyFragment : public Fragment {
 public:
  /* Inherit the constructors */
  using Fragment::Fragment;

  explicit PyFragment(py::object op) : Fragment() {
    py::gil_scoped_acquire scope_guard;
    py_compose_ = py::getattr(op, "compose");
  }

  /* Trampolines (need one for each virtual function) */
  void add_operator(const std::shared_ptr<Operator>& op) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Fragment, add_operator, op);
  }
  void add_flow(const std::shared_ptr<Operator>& upstream_op,
                const std::shared_ptr<Operator>& downstream_op) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Fragment, add_flow, upstream_op, downstream_op);
  }
  void add_flow(const std::shared_ptr<Operator>& upstream_op,
                const std::shared_ptr<Operator>& downstream_op,
                std::set<std::pair<std::string, std::string>> io_map) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Fragment, add_flow, upstream_op, downstream_op, io_map);
  }

  void compose() override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    // PYBIND11_OVERRIDE(void, Fragment, compose);

    // PYBIND11_OVERRIDE doesn't work when Fragment object is created during Application::compose().
    // So we take the py::object from the constructor and call it here.
    py::gil_scoped_acquire scope_guard;
    py_compose_.operator()();
  }
  void run() override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Fragment, run);
  }

 private:
  py::object py_compose_ = py::none();
};

}  // namespace holoscan

#endif /* PYBIND11_CORE_FRAGMENT_HPP */
