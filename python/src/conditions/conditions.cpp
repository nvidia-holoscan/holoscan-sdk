/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "./conditions_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/conditions/gxf/boolean.hpp"
#include "holoscan/core/conditions/gxf/count.hpp"
#include "holoscan/core/conditions/gxf/downstream_affordable.hpp"
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

class PyBooleanCondition : public BooleanCondition {
 public:
  /* Inherit the constructors */
  using BooleanCondition::BooleanCondition;

  // Define a constructor that fully initializes the object.
  PyBooleanCondition(Fragment* fragment, bool enable_tick = true,
                     const std::string& name = "boolean_condition")
      : BooleanCondition(Arg{"enable_tick", enable_tick}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
  }
};

class PyCountCondition : public CountCondition {
 public:
  /* Inherit the constructors */
  using CountCondition::CountCondition;

  // Define a constructor that fully initializes the object.
  PyCountCondition(Fragment* fragment, int64_t count = 1L,
                   const std::string& name = "boolean_condition")
      : CountCondition(Arg{"count", count}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
  }
};

class PyDownstreamMessageAffordableCondition : public DownstreamMessageAffordableCondition {
 public:
  /* Inherit the constructors */
  using DownstreamMessageAffordableCondition::DownstreamMessageAffordableCondition;

  // Define a constructor that fully initializes the object.
  PyDownstreamMessageAffordableCondition(Fragment* fragment,
                                         // std::shared_ptr<gxf::GXFResource> transmitter,
                                         // add transmitter here? gxf_uid_t eid,
                                         uint64_t min_size = 1L,
                                         const std::string& name = "boolean_condition")
      : DownstreamMessageAffordableCondition(Arg{"min_size", min_size}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    // transmitter_ = transmitter;  // e.g. DoubleBufferTransmitter
    setup(*spec_.get());
  }
};

class PyMessageAvailableCondition : public MessageAvailableCondition {
 public:
  /* Inherit the constructors */
  using MessageAvailableCondition::MessageAvailableCondition;

  // Define a constructor that fully initializes the object.
  PyMessageAvailableCondition(Fragment* fragment,
                              // std::shared_ptr<gxf::GXFResource> receiver,
                              size_t min_size = 1UL, size_t front_stage_max_size = 1UL,
                              const std::string& name = "boolean_condition")
      : MessageAvailableCondition(
            ArgList{Arg{"min_size", min_size}, Arg{"front_stage_max_size", front_stage_max_size}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    // receiver = receiver;  // e.g. DoubleBufferReceiver
    setup(*spec_.get());
  }
};

// End of trampoline classes for handling Python kwargs

PYBIND11_MODULE(_conditions, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _conditions
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<BooleanCondition,
             PyBooleanCondition,
             gxf::GXFCondition,
             std::shared_ptr<BooleanCondition>>(
      m, "BooleanCondition", doc::BooleanCondition::doc_BooleanCondition)
      .def(py::init<Fragment*, bool, const std::string&>(),
           "fragment"_a,
           "enable_tick"_a = true,
           "name"_a = "boolean_condition"s,
           doc::BooleanCondition::doc_BooleanCondition_python)
      .def_property_readonly(
          "gxf_typename", &BooleanCondition::gxf_typename, doc::BooleanCondition::doc_gxf_typename)
      .def("enable_tick", &BooleanCondition::enable_tick, doc::BooleanCondition::doc_enable_tick)
      .def("disable_tick", &BooleanCondition::disable_tick, doc::BooleanCondition::doc_disable_tick)
      .def("check_tick_enabled",
           &BooleanCondition::check_tick_enabled,
           doc::BooleanCondition::doc_check_tick_enabled)
      .def("setup", &BooleanCondition::setup, "spec"_a, doc::BooleanCondition::doc_setup)
      .def(
          "__repr__",
          // had to remove const here to access check_tick_enabled()
          [](BooleanCondition& cond) {
            try {
              std::string enabled = fmt::format("{}", cond.check_tick_enabled());
              // Python True, False are uppercase
              enabled[0] = toupper(enabled[0]);
              return fmt::format(
                "BooleanCondition(self, enable_tick={}, name={})", enabled, cond.name());
            } catch (const std::runtime_error& e) {
              // fallback for when initialize() has not yet been called
              return fmt::format(
                "BooleanCondition(self, enable_tick=<uninitialized>, name={})", cond.name());
            }
          },
          py::call_guard<py::gil_scoped_release>(),
          R"doc(Return repr(self).)doc");

  py::class_<CountCondition, PyCountCondition, gxf::GXFCondition, std::shared_ptr<CountCondition>>(
      m, "CountCondition", doc::CountCondition::doc_CountCondition)
      .def(py::init<Fragment*, uint64_t, const std::string&>(),
           "fragment"_a,
           "count"_a = 1L,
           "name"_a = "count_condition"s,
           doc::CountCondition::doc_CountCondition_python)
      .def_property_readonly(
          "gxf_typename", &CountCondition::gxf_typename, doc::CountCondition::doc_gxf_typename)
      .def("setup", &CountCondition::setup, doc::CountCondition::doc_setup)
      .def_property("count",
                    py::overload_cast<>(&CountCondition::count),
                    py::overload_cast<int64_t>(&CountCondition::count),
                    doc::CountCondition::doc_count)
      .def(
          "__repr__",
          // had to remove const here to access count() property getter
          [](CountCondition& cond) {
            try {
              auto cnt = cond.count();
              return fmt::format(
                "CountCondition(self, count={}, name={})", cnt, cond.name());
            } catch (const std::runtime_error& e) {
              // fallback for when initialize() has not yet been called
              return fmt::format(
                "CountCondition(self, count=<uninitialized>, name={})", cond.name());
            }
          },
          py::call_guard<py::gil_scoped_release>(),
          R"doc(Return repr(self).)doc");

  py::class_<DownstreamMessageAffordableCondition,
             PyDownstreamMessageAffordableCondition,
             gxf::GXFCondition,
             std::shared_ptr<DownstreamMessageAffordableCondition>>(
      m,
      "DownstreamMessageAffordableCondition",
      doc::DownstreamMessageAffordableCondition::doc_DownstreamMessageAffordableCondition)
      .def(py::init<Fragment*, uint64_t, const std::string&>(),
           "fragment"_a,
           "min_size"_a = 1L,
           "name"_a = "downstream_message_affordable_condition"s,
           doc::DownstreamMessageAffordableCondition::
               doc_DownstreamMessageAffordableCondition_python)
      .def_property_readonly("gxf_typename",
                             &DownstreamMessageAffordableCondition::gxf_typename,
                             doc::DownstreamMessageAffordableCondition::doc_gxf_typename)
      .def("setup",
           &DownstreamMessageAffordableCondition::setup,
           "spec"_a,
           doc::DownstreamMessageAffordableCondition::doc_setup)
      .def("initialize",
           &DownstreamMessageAffordableCondition::initialize,
           doc::DownstreamMessageAffordableCondition::doc_initialize)
      .def_property("min_size",
                    py::overload_cast<>(&DownstreamMessageAffordableCondition::min_size),
                    py::overload_cast<uint64_t>(&DownstreamMessageAffordableCondition::min_size),
                    doc::DownstreamMessageAffordableCondition::doc_min_size)
      .def_property("transmitter",
                    py::overload_cast<>(&DownstreamMessageAffordableCondition::transmitter),
                    py::overload_cast<std::shared_ptr<gxf::GXFResource>>(
                        &DownstreamMessageAffordableCondition::transmitter),
                    doc::DownstreamMessageAffordableCondition::doc_transmitter);

  py::class_<MessageAvailableCondition,
             PyMessageAvailableCondition,
             gxf::GXFCondition,
             std::shared_ptr<MessageAvailableCondition>>(
      m, "MessageAvailableCondition", doc::MessageAvailableCondition::doc_MessageAvailableCondition)
      .def(py::init<Fragment*, size_t, size_t, const std::string&>(),
           "fragment"_a,
           "min_size"_a = 1UL,
           "front_stage_max_size"_a = 1UL,
           "name"_a = "message_available_condition"s,
           doc::MessageAvailableCondition::doc_MessageAvailableCondition_python)
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
}  // PYBIND11_MODULE
}  // namespace holoscan
