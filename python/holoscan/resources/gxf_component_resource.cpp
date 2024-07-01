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

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>

#include "./gxf_component_resource_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/resources/gxf/gxf_component_resource.hpp"

#include "../operators/operator_util.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan {

// PyGXFComponentResource trampoline class: provides override for virtual function is_available

class PyGXFComponentResource : public GXFComponentResource {
 public:
  /* Inherit the constructors */
  using GXFComponentResource::GXFComponentResource;

  // Define a constructor that fully initializes the object.
  PyGXFComponentResource(py::object component, Fragment* fragment, const std::string& gxf_typename,
                         const std::string& name, const py::kwargs& kwargs)
      : GXFComponentResource(gxf_typename.c_str()) {
    py_component_ = component;
    py_initialize_ = py::getattr(component, "initialize");  // cache the initialize method

    // We don't need to call `add_positional_condition_and_resource_args(this, args);` because
    // Holoscan resources don't accept the positional arguments for Condition and Resource.
    add_kwargs(this, kwargs);

    name_ = name;
    fragment_ = fragment;
  }

  void initialize() override {
    // Get the initialize method of the Python Resource class and call it
    py::gil_scoped_acquire scope_guard;

    // TODO(gbae): setup tracing
    // // set_py_tracing();

    // Call the initialize method of the Python Resource class
    py_initialize_.operator()();

    // Call the parent class's initialize method after invoking the Python Resource's initialize
    // method.
    GXFComponentResource::initialize();
  }

 protected:
  py::object py_component_ = py::none();
  py::object py_initialize_ = py::none();
};

/* Trampoline classes for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the resource.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the resource's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_resource<ResourceT>
 */
void init_gxf_component_resource(py::module_& m) {
  py::class_<GXFComponentResource,
             PyGXFComponentResource,
             gxf::GXFResource,
             std::shared_ptr<GXFComponentResource>>(
      m, "GXFComponentResource", doc::GXFComponentResource::doc_GXFComponentResource)
      .def(py::init<>())
      .def(py::init<py::object,
                    Fragment*,
                    const std::string&,
                    const std::string&,
                    const py::kwargs&>(),
           "component"_a,
           "fragment"_a,
           "gxf_typename"_a,
           py::kw_only(),
           "name"_a = "gxf_component"s,
           doc::GXFComponentResource::doc_GXFComponentResource)
      .def_property_readonly("gxf_typename",
                             &GXFComponentResource::gxf_typename,
                             doc::GXFComponentResource::doc_gxf_typename)
      .def("initialize",
           &GXFComponentResource::initialize,
           doc::GXFComponentResource::doc_initialize)
      .def("setup", &GXFComponentResource::setup, "spec"_a, doc::GXFComponentResource::doc_setup);
}
}  // namespace holoscan
