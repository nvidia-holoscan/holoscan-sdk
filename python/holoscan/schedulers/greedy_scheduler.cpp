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

#include "./greedy_scheduler_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_component.hpp"
#include "holoscan/core/gxf/gxf_scheduler.hpp"
#include "holoscan/core/resources/gxf/clock.hpp"
#include "holoscan/core/resources/gxf/realtime_clock.hpp"
#include "holoscan/core/schedulers/gxf/greedy_scheduler.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

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

class PyGreedyScheduler : public GreedyScheduler {
 public:
  /* Inherit the constructors */
  using GreedyScheduler::GreedyScheduler;

  // Define a constructor that fully initializes the object.
  explicit PyGreedyScheduler(Fragment* fragment, std::shared_ptr<Clock> clock = nullptr,
                             bool stop_on_deadlock = true, int64_t max_duration_ms = -1LL,
                             double check_recession_period_ms = 5.0,
                             int64_t stop_on_deadlock_timeout = 0LL,
                             const std::string& name = "greedy_scheduler")
      : GreedyScheduler(ArgList{Arg{"stop_on_deadlock", stop_on_deadlock},
                                Arg{"check_recession_period_ms", check_recession_period_ms},
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
    HOLOSCAN_LOG_TRACE("in PyGreedyScheduler constructor");
    setup(*spec_);
  }
};
void init_greedy_scheduler(py::module_& m) {
  py::class_<GreedyScheduler,
             PyGreedyScheduler,
             gxf::GXFScheduler,
             Component,
             gxf::GXFComponent,
             std::shared_ptr<GreedyScheduler>>(
      m, "GreedyScheduler", doc::GreedyScheduler::doc_GreedyScheduler)
      .def(py::init<Fragment*,
                    std::shared_ptr<Clock>,
                    bool,
                    int64_t,
                    double,
                    int64_t,
                    const std::string&>(),
           "fragment"_a,
           py::kw_only(),
           "clock"_a = py::none(),
           "stop_on_deadlock"_a = true,
           "max_duration_ms"_a = -1LL,
           "check_recession_period_ms"_a = 0.0,
           "stop_on_deadlock_timeout"_a = 0LL,
           "name"_a = "greedy_scheduler"s,
           doc::GreedyScheduler::doc_GreedyScheduler)
      .def_property_readonly("clock", &GreedyScheduler::clock)
      .def_property_readonly("max_duration_ms", &GreedyScheduler::max_duration_ms)
      .def_property_readonly("stop_on_deadlock", &GreedyScheduler::stop_on_deadlock)
      .def_property_readonly("check_recession_period_ms",
                             &GreedyScheduler::check_recession_period_ms)
      .def_property_readonly("stop_on_deadlock_timeout",
                             &GreedyScheduler::stop_on_deadlock_timeout);
}  // PYBIND11_MODULE
}  // namespace holoscan
