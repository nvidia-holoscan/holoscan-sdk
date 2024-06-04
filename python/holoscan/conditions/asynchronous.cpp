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

#include "./asynchronous_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/conditions/gxf/count.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace holoscan {

/* Trampoline classes for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the condition.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the condition's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_condition<ConditionT>
 */

class PyAsynchronousCondition : public AsynchronousCondition {
 public:
  /* Inherit the constructors */
  using AsynchronousCondition::AsynchronousCondition;

  // Define a constructor that fully initializes the object.
  explicit PyAsynchronousCondition(Fragment* fragment,
                                   const std::string& name = "noname_async_condition")
      : AsynchronousCondition() {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
  }
};

void init_asynchronous(py::module_& m) {
  py::enum_<holoscan::AsynchronousEventState>(m, "AsynchronousEventState")
      .value("READY", holoscan::AsynchronousEventState::READY)
      .value("WAIT", holoscan::AsynchronousEventState::WAIT)
      .value("EVENT_WAITING", holoscan::AsynchronousEventState::EVENT_WAITING)
      .value("EVENT_DONE", holoscan::AsynchronousEventState::EVENT_DONE)
      .value("EVENT_NEVER", holoscan::AsynchronousEventState::EVENT_NEVER);

  py::class_<AsynchronousCondition,
             PyAsynchronousCondition,
             gxf::GXFCondition,
             std::shared_ptr<AsynchronousCondition>>(
      m, "AsynchronousCondition", doc::AsynchronousCondition::doc_AsynchronousCondition)
      .def(py::init<Fragment*, const std::string&>(),
           "fragment"_a,
           "name"_a = "noname_async_condition"s,
           doc::AsynchronousCondition::doc_AsynchronousCondition_python)
      .def_property_readonly("gxf_typename",
                             &AsynchronousCondition::gxf_typename,
                             doc::AsynchronousCondition::doc_gxf_typename)
      .def_property(
          "event_state",
          py::overload_cast<>(&AsynchronousCondition::event_state, py::const_),
          py::overload_cast<holoscan::AsynchronousEventState>(&AsynchronousCondition::event_state),
          doc::AsynchronousCondition::doc_event_state)
      .def("setup", &AsynchronousCondition::setup, "spec"_a, doc::AsynchronousCondition::doc_setup);
}
}  // namespace holoscan
