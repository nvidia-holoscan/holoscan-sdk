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

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include "./expiring_message_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/conditions/gxf/expiring_message.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/resources/gxf/realtime_clock.hpp"

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

class PyExpiringMessageAvailableCondition : public ExpiringMessageAvailableCondition {
 public:
  /* Inherit the constructors */
  using ExpiringMessageAvailableCondition::ExpiringMessageAvailableCondition;

  // Define a constructor that fully initializes the object.
  PyExpiringMessageAvailableCondition(
      Fragment* fragment,
      // std::shared_ptr<gxf::GXFResource> receiver,
      int64_t max_batch_size, int64_t max_delay_ns, std::shared_ptr<Clock> clock = nullptr,
      const std::string& name = "noname_expiring_message_available_condition")
      : ExpiringMessageAvailableCondition(max_batch_size, max_delay_ns) {
    name_ = name;
    fragment_ = fragment;
    if (clock) {
      this->add_arg(Arg{"clock", clock});
    } else {
      this->add_arg(Arg{"clock", fragment_->make_resource<RealtimeClock>("realtime_clock")});
    }
    spec_ = std::make_shared<ComponentSpec>(fragment);
    // receiver = receiver;  // e.g. DoubleBufferReceiver
    setup(*spec_.get());
  }

  template <typename Rep, typename Period>
  PyExpiringMessageAvailableCondition(
      Fragment* fragment,
      // std::shared_ptr<gxf::GXFResource> receiver,
      int64_t max_batch_size, std::chrono::duration<Rep, Period> recess_period_duration,
      std::shared_ptr<Clock> clock = nullptr,
      const std::string& name = "noname_expiring_message_available_condition")
      : ExpiringMessageAvailableCondition(max_batch_size, recess_period_duration) {
    name_ = name;
    fragment_ = fragment;
    if (clock) {
      this->add_arg(Arg{"clock", clock});
    } else {
      this->add_arg(Arg{"clock", fragment_->make_resource<RealtimeClock>("realtime_clock")});
    }
    spec_ = std::make_shared<ComponentSpec>(fragment);
    // receiver = receiver;  // e.g. DoubleBufferReceiver
    setup(*spec_.get());
  }
};

void init_expiring_message_available(py::module_& m) {
  py::class_<ExpiringMessageAvailableCondition,
             PyExpiringMessageAvailableCondition,
             gxf::GXFCondition,
             std::shared_ptr<ExpiringMessageAvailableCondition>>(
      m,
      "ExpiringMessageAvailableCondition",
      doc::ExpiringMessageAvailableCondition::doc_ExpiringMessageAvailableCondition)
      // TODO: sphinx API doc build complains if more than one ExpiringMessageAvailableCondition
      //       init method has a docstring specified. For now just set the docstring for the
      //       overload using datetime.timedelta for the max_delay.
      .def(py::init<Fragment*, int64_t, int64_t, std::shared_ptr<Clock>, const std::string&>(),
           "fragment"_a,
           "max_batch_size"_a,
           "max_delay_ns"_a,
           "clock"_a = py::none(),
           "name"_a = "noname_expiring_message_available_condition"s)
      .def(py::init<Fragment*,
                    int64_t,
                    std::chrono::nanoseconds,
                    std::shared_ptr<Clock>,
                    const std::string&>(),
           "fragment"_a,
           "max_batch_size"_a,
           "max_delay_ns"_a,
           "clock"_a = py::none(),
           "name"_a = "noname_expiring_message_available_condition"s,
           doc::ExpiringMessageAvailableCondition::doc_ExpiringMessageAvailableCondition)
      .def_property_readonly("gxf_typename",
                             &ExpiringMessageAvailableCondition::gxf_typename,
                             doc::ExpiringMessageAvailableCondition::doc_gxf_typename)
      .def_property("receiver",
                    py::overload_cast<>(&ExpiringMessageAvailableCondition::receiver),
                    py::overload_cast<std::shared_ptr<gxf::GXFResource>>(
                        &ExpiringMessageAvailableCondition::receiver),
                    doc::ExpiringMessageAvailableCondition::doc_receiver)
      .def_property("max_batch_size",
                    py::overload_cast<>(&ExpiringMessageAvailableCondition::max_batch_size),
                    py::overload_cast<int64_t>(&ExpiringMessageAvailableCondition::max_batch_size),
                    doc::ExpiringMessageAvailableCondition::doc_max_batch_size)
      .def("max_delay",
           static_cast<void (ExpiringMessageAvailableCondition::*)(int64_t)>(
               &ExpiringMessageAvailableCondition::max_delay),
           doc::ExpiringMessageAvailableCondition::doc_max_delay)
      .def("max_delay",
           static_cast<void (ExpiringMessageAvailableCondition::*)(std::chrono::nanoseconds)>(
               &ExpiringMessageAvailableCondition::max_delay),
           doc::ExpiringMessageAvailableCondition::doc_max_delay)
      .def("max_delay_ns",
           &ExpiringMessageAvailableCondition::max_delay_ns,
           doc::ExpiringMessageAvailableCondition::doc_max_delay_ns)
      .def("setup",
           &ExpiringMessageAvailableCondition::setup,
           doc::ExpiringMessageAvailableCondition::doc_setup)
      .def("initialize",
           &ExpiringMessageAvailableCondition::initialize,
           doc::ExpiringMessageAvailableCondition::doc_initialize);
}
}  // namespace holoscan
