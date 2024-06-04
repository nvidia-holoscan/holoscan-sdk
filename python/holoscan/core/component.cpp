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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <list>

#include "component_pydoc.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/component.hpp"
#include "holoscan/core/fragment.hpp"

using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan {

class PyComponentSpec : public ComponentSpec {
 public:
  /* Inherit the constructors */
  using ComponentSpec::ComponentSpec;

  // Override the constructor to get the py::object for the Python class
  explicit PyComponentSpec(Fragment* fragment = nullptr, py::object op = py::none())
      : ComponentSpec(fragment), py_op_(op) {}

  // TOIMPROVE: Should we parse headline and description from kwargs or just
  //            add them to the function signature?
  void py_param(const std::string& name, const py::object& default_value, const ParameterFlag& flag,
                const py::kwargs& kwargs) {
    using std::string_literals::operator""s;

    bool is_receivers = false;
    std::string headline{""s};
    std::string description{""s};
    for (const auto& [name, value] : kwargs) {
      std::string param_name = name.cast<std::string>();
      if (param_name == "headline") {
        headline = value.cast<std::string>();
      } else if (param_name == "description") {
        description = value.cast<std::string>();
      } else {
        throw std::runtime_error("unsupported kwarg: "s + param_name);
      }
    }

    // Create parameter object
    py_params_.emplace_back(py_op());

    // Register parameter
    auto& parameter = py_params_.back();
    param(parameter, name.c_str(), headline.c_str(), description.c_str(), default_value, flag);
  }

  py::object py_op() const { return py_op_; }

  std::list<Parameter<py::object>>& py_params() { return py_params_; }

 private:
  py::object py_op_ = py::none();
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

void init_component(py::module_& m) {
  py::class_<ComponentSpec, std::shared_ptr<ComponentSpec>>(
      m, "ComponentSpec", R"doc(Component specification class.)doc")
      .def(py::init<Fragment*>(), "fragment"_a, doc::ComponentSpec::doc_ComponentSpec)
      .def_property_readonly("fragment",
                             py::overload_cast<>(&ComponentSpec::fragment),
                             doc::ComponentSpec::doc_fragment)
      .def_property_readonly("params", &ComponentSpec::params, doc::ComponentSpec::doc_params)
      .def_property_readonly(
          "description", &ComponentSpec::description, doc::ComponentSpec::doc_description)
      .def(
          "__repr__",
          // use py::object and obj.cast to avoid a segfault if object has not been initialized
          [](const ComponentSpec& spec) { return spec.description(); },
          R"doc(Return repr(self).)doc");

  py::enum_<ParameterFlag>(m, "ParameterFlag", doc::ParameterFlag::doc_ParameterFlag)
      .value("NONE", ParameterFlag::kNone)
      .value("OPTIONAL", ParameterFlag::kOptional)
      .value("DYNAMIC", ParameterFlag::kDynamic);

  py::class_<PyComponentSpec, ComponentSpec, std::shared_ptr<PyComponentSpec>>(
      m, "PyComponentSpec", R"doc(Operator specification class.)doc")
      .def(py::init<Fragment*, py::object>(),
           "fragment"_a,
           "op"_a = py::none(),
           doc::ComponentSpec::doc_ComponentSpec)
      .def("param",
           &PyComponentSpec::py_param,
           "Register parameter",
           "name"_a,
           "default_value"_a = py::none(),
           "flag"_a = ParameterFlag::kNone,
           doc::ComponentSpec::doc_param);

  py::class_<ComponentBase, PyComponentBase, std::shared_ptr<ComponentBase>>(
      m, "ComponentBase", doc::Component::doc_Component)
      .def(py::init<>(), doc::Component::doc_Component)
      .def_property_readonly("id", &ComponentBase::id, doc::Component::doc_id)
      .def_property_readonly("name", &ComponentBase::name, doc::Component::doc_name)
      .def_property_readonly("fragment", &ComponentBase::fragment, doc::Component::doc_fragment)
      .def("add_arg",
           py::overload_cast<const Arg&>(&ComponentBase::add_arg),
           "arg"_a,
           doc::Component::doc_add_arg_Arg)
      .def("add_arg",
           py::overload_cast<const ArgList&>(&ComponentBase::add_arg),
           "arg"_a,
           doc::Component::doc_add_arg_ArgList)
      .def_property_readonly("args", &ComponentBase::args, doc::Component::doc_args)
      .def("initialize",
           &ComponentBase::initialize,
           doc::Component::doc_initialize)  // note: virtual function
      .def_property_readonly(
          "description", &ComponentBase::description, doc::Component::doc_description)
      .def(
          "__repr__",
          [](const py::object& obj) {
            // use py::object and obj.cast to avoid a segfault if object has not been initialized
            auto component = obj.cast<std::shared_ptr<ComponentBase>>();
            if (component) { return component->description(); }
            return std::string("<Component: None>");
          },
          R"doc(Return repr(self).)doc");

  py::class_<Component, ComponentBase, PyComponent, std::shared_ptr<Component>>(
      m, "Component", doc::Component::doc_Component)
      .def(py::init<>(), doc::Component::doc_Component);
}

}  // namespace holoscan
