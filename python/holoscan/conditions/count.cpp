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

#include <cstdint>
#include <memory>
#include <string>

#include "./count_pydoc.hpp"
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

class PyCountCondition : public CountCondition {
 public:
  /* Inherit the constructors */
  using CountCondition::CountCondition;

  // Define a constructor that fully initializes the object.
  PyCountCondition(Fragment* fragment, int64_t count = 1L,
                   const std::string& name = "noname_count_condition")
      : CountCondition(Arg{"count", count}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
  }
};

void init_count(py::module_& m) {
  py::class_<CountCondition, PyCountCondition, gxf::GXFCondition, std::shared_ptr<CountCondition>>(
      m, "CountCondition", doc::CountCondition::doc_CountCondition)
      .def(py::init<Fragment*, int64_t, const std::string&>(),
           "fragment"_a,
           "count"_a = 1L,
           "name"_a = "noname_count_condition"s,
           doc::CountCondition::doc_CountCondition)
      .def_property_readonly(
          "gxf_typename", &CountCondition::gxf_typename, doc::CountCondition::doc_gxf_typename)
      .def("setup", &CountCondition::setup, doc::CountCondition::doc_setup)
      .def_property("count",
                    py::overload_cast<>(&CountCondition::count),
                    py::overload_cast<int64_t>(&CountCondition::count),
                    doc::CountCondition::doc_count);
}
}  // namespace holoscan
