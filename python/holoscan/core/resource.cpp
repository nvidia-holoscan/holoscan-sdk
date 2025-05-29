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

#include "resource.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <utility>

#include "component.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/component.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/resource.hpp"
#include "kwarg_handling.hpp"
#include "resource_pydoc.hpp"

namespace py = pybind11;

namespace holoscan {

// Define a kwargs-based constructor that can create an ArgList
// for passing on to the variadic-template based constructor.
PyResource::PyResource(py::object resource, Fragment* fragment, const py::args& args,
                       const py::kwargs& kwargs)
    : py_resource_(std::move(resource)) {
  using std::string_literals::operator""s;

  fragment_ = fragment;

  int n_fragments = 0;
  for (const auto& item : args) {
    auto arg_value = item.cast<py::object>();
    if (py::isinstance<Fragment>(arg_value)) {
      if (n_fragments > 0) { throw std::runtime_error("multiple Fragment objects provided"); }
      fragment_ = arg_value.cast<Fragment*>();
      n_fragments += 1;
    } else {
      this->add_arg(py_object_to_arg(arg_value, ""s));
    }
  }
  for (const auto& [name, value] : kwargs) {
    auto kwarg_name = name.cast<std::string>();
    auto kwarg_value = value.cast<py::object>();
    if (kwarg_name == "name"s) {
      if (py::isinstance<py::str>(kwarg_value)) {
        name_ = kwarg_value.cast<std::string>();
      } else {
        throw std::runtime_error("name kwarg must be a string");
      }
    } else if (kwarg_name == "fragment"s) {
      if (py::isinstance<Fragment>(kwarg_value)) {
        if (n_fragments > 0) {
          throw std::runtime_error(
              "Cannot add kwarg fragment, when a Fragment was also provided positionally");
        }
        fragment_ = kwarg_value.cast<Fragment*>();
      } else {
        throw std::runtime_error("fragment kwarg must be a Fragment");
      }
    } else {
      this->add_arg(py_object_to_arg(kwarg_value, kwarg_name));
    }
  }
}

// Override spec() method
std::shared_ptr<PyComponentSpec> PyResource::py_shared_spec() {
  auto spec_ptr = spec_shared();
  return std::static_pointer_cast<PyComponentSpec>(spec_ptr);
}

/* Trampolines (need one for each virtual function) */
void PyResource::initialize() {
  /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
  PYBIND11_OVERRIDE(void, Resource, initialize);
}

void PyResource::setup(ComponentSpec& spec) {
  /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
  PYBIND11_OVERRIDE(void, Resource, setup, spec);
}

void init_resource(py::module_& m) {
  // note: added py::dynamic_attr() to allow dynamically adding attributes in a Python subclass
  py::class_<Resource, Component, PyResource, std::shared_ptr<Resource>> resource_class(
      m, "Resource", py::dynamic_attr(), doc::Resource::doc_Resource_args_kwargs);

  resource_class
      .def(py::init<py::object, Fragment*, const py::args&, const py::kwargs&>(),
           doc::Resource::doc_Resource_args_kwargs)
      .def_property(
          "name",
          py::overload_cast<>(&Resource::name, py::const_),
          [](Resource& r, const std::string& name) -> Resource& { return r.name(name); },
          doc::Resource::doc_name)
      .def_property_readonly(
          "fragment", py::overload_cast<>(&Resource::fragment), doc::Resource::doc_fragment)
      .def_property("spec",
                    &Resource::spec_shared,
                    py::overload_cast<const std::shared_ptr<ComponentSpec>&>(&Resource::spec))
      .def("setup", &Resource::setup, doc::Resource::doc_setup)  // note: virtual
      .def("initialize",
           &Resource::initialize,
           doc::Resource::doc_initialize)  // note: virtual function
      .def_property_readonly("description", &Resource::description, doc::Resource::doc_description)
      .def_property_readonly(
          "resource_type", &Resource::resource_type, doc::Resource::doc_resource_type)
      .def(
          "__repr__",
          [](const py::object& obj) {
            // use py::object and obj.cast to avoid a segfault if object has not been initialized
            auto resource = obj.cast<std::shared_ptr<Resource>>();
            if (resource) { return resource->description(); }
            return std::string("<Resource: None>");
          },
          R"doc(Return repr(self).)doc");

  py::enum_<Resource::ResourceType>(resource_class, "ResourceType")
      .value("NATIVE", Resource::ResourceType::kNative)
      .value("GXF", Resource::ResourceType::kGXF);
}

}  // namespace holoscan
