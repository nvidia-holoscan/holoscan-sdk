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

#include <cstdint>
#include <memory>
#include <string>

#include "./downstream_message_affordable_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/conditions/gxf/downstream_affordable.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

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

class PyDownstreamMessageAffordableCondition : public DownstreamMessageAffordableCondition {
 public:
  /* Inherit the constructors */
  using DownstreamMessageAffordableCondition::DownstreamMessageAffordableCondition;

  // Define a constructor that fully initializes the object.
  explicit PyDownstreamMessageAffordableCondition(
      Fragment* fragment, uint64_t min_size = 1L,
      std::optional<const std::string> transmitter = std::nullopt,
      const std::string& name = "noname_downstream_affordable_condition")
      : DownstreamMessageAffordableCondition(Arg{"min_size", min_size}) {
    name_ = name;
    fragment_ = fragment;
    if (transmitter.has_value()) { this->add_arg(Arg("transmitter", transmitter.value())); }
    spec_ = std::make_shared<ComponentSpec>(fragment);
    // Note "transmitter" parameter is set automatically from GXFExecutor
    setup(*spec_);
  }
};

void init_downstream_message_affordable(py::module_& m) {
  py::class_<DownstreamMessageAffordableCondition,
             PyDownstreamMessageAffordableCondition,
             gxf::GXFCondition,
             std::shared_ptr<DownstreamMessageAffordableCondition>>(
      m,
      "DownstreamMessageAffordableCondition",
      doc::DownstreamMessageAffordableCondition::doc_DownstreamMessageAffordableCondition)
      .def(py::init<Fragment*, uint64_t, std::optional<const std::string>, const std::string&>(),
           "fragment"_a,
           "min_size"_a = 1L,
           "transmitter"_a = py::none(),
           "name"_a = "noname_downstream_affordable_condition"s,
           doc::DownstreamMessageAffordableCondition::doc_DownstreamMessageAffordableCondition)
      .def_property("min_size",
                    py::overload_cast<>(&DownstreamMessageAffordableCondition::min_size),
                    py::overload_cast<uint64_t>(&DownstreamMessageAffordableCondition::min_size),
                    doc::DownstreamMessageAffordableCondition::doc_min_size)
      .def_property("transmitter",
                    py::overload_cast<>(&DownstreamMessageAffordableCondition::transmitter),
                    py::overload_cast<std::shared_ptr<Transmitter>>(
                        &DownstreamMessageAffordableCondition::transmitter),
                    doc::DownstreamMessageAffordableCondition::doc_transmitter);
}
}  // namespace holoscan
