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

#include "./message_available_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/conditions/gxf/message_available.hpp"
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

class PyMessageAvailableCondition : public MessageAvailableCondition {
 public:
  /* Inherit the constructors */
  using MessageAvailableCondition::MessageAvailableCondition;

  // Define a constructor that fully initializes the object.
  PyMessageAvailableCondition(Fragment* fragment,
                              // std::shared_ptr<gxf::GXFResource> receiver,
                              uint64_t min_size = 1UL, size_t front_stage_max_size = 1UL,
                              const std::string& name = "noname_message_available_condition")
      : MessageAvailableCondition(
            ArgList{Arg{"min_size", min_size}, Arg{"front_stage_max_size", front_stage_max_size}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    // receiver = receiver;  // e.g. DoubleBufferReceiver
    setup(*spec_.get());
  }
};

void init_message_available(py::module_& m) {
  py::class_<MessageAvailableCondition,
             PyMessageAvailableCondition,
             gxf::GXFCondition,
             std::shared_ptr<MessageAvailableCondition>>(
      m, "MessageAvailableCondition", doc::MessageAvailableCondition::doc_MessageAvailableCondition)
      .def(py::init<Fragment*, uint64_t, size_t, const std::string&>(),
           "fragment"_a,
           "min_size"_a = 1UL,
           "front_stage_max_size"_a = 1UL,
           "name"_a = "noname_message_available_condition"s,
           doc::MessageAvailableCondition::doc_MessageAvailableCondition)
      .def_property_readonly("gxf_typename",
                             &MessageAvailableCondition::gxf_typename,
                             doc::MessageAvailableCondition::doc_gxf_typename)
      .def_property("receiver",
                    py::overload_cast<>(&MessageAvailableCondition::receiver),
                    py::overload_cast<std::shared_ptr<gxf::GXFResource>>(
                        &MessageAvailableCondition::receiver),
                    doc::MessageAvailableCondition::doc_receiver)
      .def_property("min_size",
                    py::overload_cast<>(&MessageAvailableCondition::min_size),
                    py::overload_cast<size_t>(&MessageAvailableCondition::min_size),
                    doc::MessageAvailableCondition::doc_min_size)
      .def_property("front_stage_max_size",
                    py::overload_cast<>(&MessageAvailableCondition::front_stage_max_size),
                    py::overload_cast<size_t>(&MessageAvailableCondition::front_stage_max_size),
                    doc::MessageAvailableCondition::doc_front_stage_max_size)
      .def("setup", &MessageAvailableCondition::setup, doc::MessageAvailableCondition::doc_setup)
      .def("initialize",
           &MessageAvailableCondition::initialize,
           doc::MessageAvailableCondition::doc_initialize);
}
}  // namespace holoscan
