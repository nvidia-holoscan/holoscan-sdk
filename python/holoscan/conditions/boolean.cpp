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

#include <pybind11/pybind11.h>

#include <memory>
#include <string>

#include "./boolean_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/conditions/gxf/boolean.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

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

class PyBooleanCondition : public BooleanCondition {
 public:
  /* Inherit the constructors */
  using BooleanCondition::BooleanCondition;

  // Define a constructor that fully initializes the object.
  PyBooleanCondition(Fragment* fragment, bool enable_tick = true,
                     const std::string& name = "noname_boolean_condition")
      : BooleanCondition(Arg{"enable_tick", enable_tick}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
  }
};

void init_boolean(py::module_& m) {
  py::class_<BooleanCondition,
             PyBooleanCondition,
             gxf::GXFCondition,
             std::shared_ptr<BooleanCondition>>(
      m, "BooleanCondition", doc::BooleanCondition::doc_BooleanCondition)
      .def(py::init<Fragment*, bool, const std::string&>(),
           "fragment"_a,
           "enable_tick"_a = true,
           "name"_a = "noname_boolean_condition"s,
           doc::BooleanCondition::doc_BooleanCondition_python)
      .def_property_readonly(
          "gxf_typename", &BooleanCondition::gxf_typename, doc::BooleanCondition::doc_gxf_typename)
      .def("enable_tick", &BooleanCondition::enable_tick, doc::BooleanCondition::doc_enable_tick)
      .def("disable_tick", &BooleanCondition::disable_tick, doc::BooleanCondition::doc_disable_tick)
      .def("check_tick_enabled",
           &BooleanCondition::check_tick_enabled,
           doc::BooleanCondition::doc_check_tick_enabled)
      .def("setup", &BooleanCondition::setup, "spec"_a, doc::BooleanCondition::doc_setup);
}  // PYBIND11_MODULE
}  // namespace holoscan
