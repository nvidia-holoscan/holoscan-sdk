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

#include "./event_based_scheduler_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_component.hpp"
#include "holoscan/core/gxf/gxf_scheduler.hpp"
#include "holoscan/core/resources/gxf/clock.hpp"
#include "holoscan/core/resources/gxf/realtime_clock.hpp"
#include "holoscan/core/schedulers/gxf/event_based_scheduler.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan {

/* Trampoline classes for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the scheduler.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the scheduler's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_scheduler<SchedulerT>
 */

class PyEventBasedScheduler : public EventBasedScheduler {
 public:
  /* Inherit the constructors */
  using EventBasedScheduler::EventBasedScheduler;

  // Define a constructor that fully initializes the object.
  explicit PyEventBasedScheduler(Fragment* fragment, std::shared_ptr<Clock> clock = nullptr,
                                 int64_t worker_thread_number = 1LL, bool stop_on_deadlock = true,
                                 int64_t max_duration_ms = -1LL,
                                 int64_t stop_on_deadlock_timeout = 0LL,
                                 const std::string& name = "event_based_scheduler")
      : EventBasedScheduler(ArgList{Arg{"worker_thread_number", worker_thread_number},
                                    Arg{"stop_on_deadlock", stop_on_deadlock},
                                    Arg{"stop_on_deadlock_timeout", stop_on_deadlock_timeout}}) {
    // max_duration_ms is an optional argument in GXF. We use a negative value in this constructor
    // to indicate that the argument should not be set.
    if (max_duration_ms >= 0) { this->add_arg(Arg{"max_duration_ms", max_duration_ms}); }
    name_ = name;
    fragment_ = fragment;
    if (clock) {
      this->add_arg(Arg{"clock", clock});
    } else {
      this->add_arg(Arg{"clock", fragment_->make_resource<RealtimeClock>("realtime_clock")});
    }
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
  }
};

void init_event_based_scheduler(py::module_& m) {
  py::class_<EventBasedScheduler,
             PyEventBasedScheduler,
             gxf::GXFScheduler,
             Component,
             gxf::GXFComponent,
             std::shared_ptr<EventBasedScheduler>>(
      m, "EventBasedScheduler", doc::EventBasedScheduler::doc_EventBasedScheduler)
      .def(py::init<Fragment*,
                    std::shared_ptr<Clock>,
                    int64_t,
                    bool,
                    int64_t,
                    int64_t,
                    const std::string&>(),
           "fragment"_a,
           py::kw_only(),
           "clock"_a = py::none(),
           "worker_thread_number"_a = 1LL,
           "stop_on_deadlock"_a = true,
           "max_duration_ms"_a = -1LL,
           "stop_on_deadlock_timeout"_a = 0LL,
           "name"_a = "multithread_scheduler"s,
           doc::EventBasedScheduler::doc_EventBasedScheduler_python)
      .def_property_readonly("clock", &EventBasedScheduler::clock)
      .def_property_readonly("worker_thread_number", &EventBasedScheduler::worker_thread_number)
      .def_property_readonly("max_duration_ms", &EventBasedScheduler::max_duration_ms)
      .def_property_readonly("stop_on_deadlock", &EventBasedScheduler::stop_on_deadlock)
      .def_property_readonly("stop_on_deadlock_timeout",
                             &EventBasedScheduler::stop_on_deadlock_timeout)
      .def_property_readonly("gxf_typename",
                             &EventBasedScheduler::gxf_typename,
                             doc::EventBasedScheduler::doc_gxf_typename);
}  // PYBIND11_MODULE
}  // namespace holoscan
