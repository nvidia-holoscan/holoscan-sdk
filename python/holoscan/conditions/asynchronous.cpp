/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <variant>

#include "../core/component_util.hpp"
#include "./asynchronous_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/conditions/gxf/count.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/subgraph.hpp"

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

class PyAsynchronousCondition : public AsynchronousCondition {
 public:
  /* Inherit the constructors */
  using AsynchronousCondition::AsynchronousCondition;

  // Define a constructor that fully initializes the object.
  explicit PyAsynchronousCondition(const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
                                   const std::string& name = "noname_async_condition") {
    init_component_base(this, fragment_or_subgraph, name, "condition");
  }
};

void init_asynchronous(py::module_& m) {
  py::enum_<holoscan::AsynchronousEventState>(m, "AsynchronousEventState")
      .value("READY", holoscan::AsynchronousEventState::kReady)
      .value("WAIT", holoscan::AsynchronousEventState::kWait)
      .value("EVENT_WAITING", holoscan::AsynchronousEventState::kEventWaiting)
      .value("EVENT_DONE", holoscan::AsynchronousEventState::kEventDone)
      .value("EVENT_NEVER", holoscan::AsynchronousEventState::kEventNever);

  py::class_<AsynchronousCondition,
             PyAsynchronousCondition,
             gxf::GXFCondition,
             std::shared_ptr<AsynchronousCondition>>(
      m, "AsynchronousCondition", doc::AsynchronousCondition::doc_AsynchronousCondition)
      .def(py::init<std::variant<Fragment*, Subgraph*>, const std::string&>(),
           "fragment"_a,
           "name"_a = "noname_async_condition"s,
           doc::AsynchronousCondition::doc_AsynchronousCondition_python)
      .def_property(
          "event_state",
          py::overload_cast<>(&AsynchronousCondition::event_state, py::const_),
          py::overload_cast<holoscan::AsynchronousEventState>(&AsynchronousCondition::event_state),
          doc::AsynchronousCondition::doc_event_state);
}
}  // namespace holoscan
