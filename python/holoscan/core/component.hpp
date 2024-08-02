/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_CORE_COMPONENT_HPP
#define PYHOLOSCAN_CORE_COMPONENT_HPP

#include <pybind11/pybind11.h>

#include <list>
#include <memory>
#include <string>
#include <utility>

#include "holoscan/core/component.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/parameter.hpp"
#include "io_context.hpp"

namespace py = pybind11;

namespace holoscan {

void init_component(py::module_&);

class PyComponentSpec : public ComponentSpec {
 public:
  /* Inherit the constructors */
  using ComponentSpec::ComponentSpec;

  // Override the constructor to get the py::object for the Python class
  explicit PyComponentSpec(Fragment* fragment = nullptr, py::object component = py::none())
      : ComponentSpec(fragment), py_component_(std::move(component)) {}

  void py_param(const std::string& name, const py::object& default_value, const ParameterFlag& flag,
                const py::kwargs& kwargs);

  py::object py_component() const { return py_component_; }

  std::list<Parameter<py::object>>& py_params() { return py_params_; }

 private:
  py::object py_component_ = py::none();
  // NOTE: we use std::list instead of std::vector because we register the address of Parameter<T>
  // object to the GXF framework. The address of a std::vector element may change when the vector is
  // resized.
  std::list<Parameter<py::object>> py_params_;
};

class PyComponentBase : public ComponentBase {
 public:
  /* Inherit the constructors */
  using ComponentBase::ComponentBase;

  /* Trampolines (need one for each virtual function) */
  void initialize() override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, ComponentBase, initialize);
  }
};

class PyComponent : public Component {
 public:
  /* Inherit the constructors */
  using Component::Component;

  /* Trampolines (need one for each virtual function) */
  void initialize() override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Component, initialize);
  }
};

}  // namespace holoscan

#endif /* PYHOLOSCAN_CORE_COMPONENT_HPP */
