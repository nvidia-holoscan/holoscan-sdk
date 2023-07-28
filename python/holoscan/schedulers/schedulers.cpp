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

#include <cstdint>
#include <memory>
#include <string>

#include "./schedulers_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_component.hpp"
#include "holoscan/core/gxf/gxf_scheduler.hpp"
#include "holoscan/core/resources/gxf/manual_clock.hpp"
#include "holoscan/core/resources/gxf/realtime_clock.hpp"
#include "holoscan/core/schedulers/gxf/greedy_scheduler.hpp"
#include "holoscan/core/schedulers/gxf/multithread_scheduler.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

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
  PyGreedyScheduler(Fragment* fragment, std::shared_ptr<Clock> clock = nullptr,
                    bool stop_on_deadlock = true, int64_t max_duration_ms = -1LL,
                    double check_recession_period_ms = 5.0, int64_t stop_on_deadlock_timeout = 0LL,
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
    setup(*spec_.get());
  }
};

class PyMultiThreadScheduler : public MultiThreadScheduler {
 public:
  /* Inherit the constructors */
  using MultiThreadScheduler::MultiThreadScheduler;

  // Define a constructor that fully initializes the object.
  explicit PyMultiThreadScheduler(Fragment* fragment, std::shared_ptr<Clock> clock = nullptr,
                                  int64_t worker_thread_number = 1LL, bool stop_on_deadlock = true,
                                  double check_recession_period_ms = 5.0,
                                  int64_t max_duration_ms = -1LL,
                                  int64_t stop_on_deadlock_timeout = 0LL,
                                  const std::string& name = "multithread_scheduler")
      : MultiThreadScheduler(ArgList{Arg{"worker_thread_number", worker_thread_number},
                                     Arg{"stop_on_deadlock", stop_on_deadlock},
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
    setup(*spec_.get());
  }
};
// End of trampoline classes for handling Python kwargs

PYBIND11_MODULE(_schedulers, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _schedulers
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
           doc::GreedyScheduler::doc_GreedyScheduler_python)
      .def_property_readonly("clock", &GreedyScheduler::clock)
      .def_property_readonly("max_duration_ms", &GreedyScheduler::max_duration_ms)
      .def_property_readonly("stop_on_deadlock", &GreedyScheduler::stop_on_deadlock)
      .def_property_readonly("check_recession_period_ms",
                             &GreedyScheduler::check_recession_period_ms)
      .def_property_readonly("stop_on_deadlock_timeout", &GreedyScheduler::stop_on_deadlock_timeout)
      .def_property_readonly(
          "gxf_typename", &GreedyScheduler::gxf_typename, doc::GreedyScheduler::doc_gxf_typename)
      .def(
          "__repr__",
          // had to remove const here to access count() property getter
          [](GreedyScheduler& scheduler) {
            try {
              auto clk = scheduler.clock()->name();
              auto maxdur = scheduler.max_duration_ms();
              auto stop = scheduler.stop_on_deadlock();
              auto crp = scheduler.check_recession_period_ms();
              auto deadlock_timeout = scheduler.stop_on_deadlock_timeout();
              return fmt::format(
                  "GreedyScheduler(self, clock={}, stop_on_deadlock={}, "
                  "check_recession_period_ms={}, max_duration_ms={}, "
                  "stop_on_deadlock_timeout={}, name={})",
                  clk,
                  stop,
                  crp,
                  maxdur,
                  deadlock_timeout,
                  scheduler.name());
            } catch (const std::runtime_error& e) {
              // fallback for when initialize() has not yet been called
              return fmt::format(
                  "GreedyScheduler(self, clock=<uninitialized>, "
                  "stop_on_deadlock=<uninitialized>, check_recession_period_ms=<uninitialized>, "
                  "max_duration_ms=<uninitialized>, stop_on_deadlock_timeout=<uninitialized>, "
                  "name={})",
                  scheduler.name());
            }
          },
          R"doc(Return repr(self).)doc");

  py::class_<MultiThreadScheduler,
             PyMultiThreadScheduler,
             gxf::GXFScheduler,
             Component,
             gxf::GXFComponent,
             std::shared_ptr<MultiThreadScheduler>>(
      m, "MultiThreadScheduler", doc::MultiThreadScheduler::doc_MultiThreadScheduler)
      .def(py::init<Fragment*,
                    std::shared_ptr<Clock>,
                    int64_t,
                    bool,
                    double,
                    int64_t,
                    int64_t,
                    const std::string&>(),
           "fragment"_a,
           py::kw_only(),
           "clock"_a = py::none(),
           "worker_thread_number"_a = 1LL,
           "stop_on_deadlock"_a = true,
           "check_recession_period_ms"_a = 5.0,
           "max_duration_ms"_a = -1LL,
           "stop_on_deadlock_timeout"_a = 0LL,
           "name"_a = "multithread_scheduler"s,
           doc::MultiThreadScheduler::doc_MultiThreadScheduler_python)
      .def_property_readonly("clock", &MultiThreadScheduler::clock)
      .def_property_readonly("worker_thread_number", &MultiThreadScheduler::worker_thread_number)
      .def_property_readonly("max_duration_ms", &MultiThreadScheduler::max_duration_ms)
      .def_property_readonly("stop_on_deadlock", &MultiThreadScheduler::stop_on_deadlock)
      .def_property_readonly("check_recession_period_ms",
                             &MultiThreadScheduler::check_recession_period_ms)
      .def_property_readonly("stop_on_deadlock_timeout",
                             &MultiThreadScheduler::stop_on_deadlock_timeout)
      .def_property_readonly("gxf_typename",
                             &MultiThreadScheduler::gxf_typename,
                             doc::MultiThreadScheduler::doc_gxf_typename)
      .def(
          "__repr__",
          // had to remove const here to access count() property getter
          [](MultiThreadScheduler& scheduler) {
            try {
              auto clk = scheduler.clock()->name();
              auto workers = scheduler.worker_thread_number();
              auto maxdur = scheduler.max_duration_ms();
              auto crp = scheduler.check_recession_period_ms();
              auto stop = scheduler.stop_on_deadlock();
              auto deadlock_timeout = scheduler.stop_on_deadlock_timeout();
              return fmt::format(
                  "MultiThreadScheduler(self, clock={}, worker_thread_number={}, "
                  "stop_on_deadlock={},"
                  "check_recession_period_ms={}, max_duration_ms={},"
                  "stop_on_deadlock_timeout={}"
                  "name={})",
                  clk,
                  workers,
                  stop,
                  crp,
                  maxdur,
                  deadlock_timeout,
                  scheduler.name());
            } catch (const std::runtime_error& e) {
              // fallback for when initialize() has not yet been called
              return fmt::format(
                  "MultiThreadScheduler(self, clock=<uninitialized>, "
                  "worker_thread_number=<uninitialized>, stop_on_deadlock=<uninitialized>, "
                  "check_recession_period_ms=<uninitialized>, max_duration_ms=<uninitialized>, "
                  "name={})",
                  scheduler.name());
            }
          },
          R"doc(Return repr(self).)doc");
}  // PYBIND11_MODULE
}  // namespace holoscan
