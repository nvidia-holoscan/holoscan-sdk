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

#include "component_pydoc.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/component.hpp"
#include "holoscan/core/fragment.hpp"

using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan {

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
