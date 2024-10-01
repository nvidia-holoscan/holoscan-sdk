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
#include <pybind11/chrono.h>  // will include timedelta.h for us

#include <cstdint>
#include <memory>
#include <string>

#include "./clocks_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/resources/gxf/clock.hpp"
#include "holoscan/core/resources/gxf/manual_clock.hpp"
#include "holoscan/core/resources/gxf/realtime_clock.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan {

int64_t get_duration_ns(const py::object& duration) {
  if (py::isinstance<py::int_>(duration)) {
    return py::cast<int64_t>(duration);
  } else {
    // Must acquire GIL before calling C API functions like PyDelta_Check
    py::gil_scoped_acquire scope_guard;

    // Must initialize PyDateTime_IMPORT here in order to be able to use PyDelta_Check below
    // see: https://docs.python.org/3/c-api/datetime.html?highlight=pydelta_check#datetime-objects
    if (!PyDateTimeAPI) { PyDateTime_IMPORT; }

    if (PyDelta_Check(duration.ptr())) {
      // timedelta stores integer days, seconds, microseconds
      int64_t days, seconds, microseconds;
      days = PyDateTime_DELTA_GET_DAYS(duration.ptr());
      seconds = PyDateTime_DELTA_GET_SECONDS(duration.ptr());
      if (days) {
        int seconds_per_day = 24 * 3600;
        seconds += days * seconds_per_day;
      }
      microseconds = PyDateTime_DELTA_GET_MICROSECONDS(duration.ptr());
      if (seconds) { microseconds += 1000000 * seconds; }
      int64_t delta_ns = 1000 * microseconds;
      return delta_ns;
    } else {
      throw std::runtime_error("expected an integer or datetime.timedelta type");
    }
  }
}

class PyRealtimeClock : public RealtimeClock {
 public:
  /* Inherit the constructors */
  using RealtimeClock::RealtimeClock;

  // Define a constructor that fully initializes the object.
  explicit PyRealtimeClock(Fragment* fragment, double initial_time_offset = 0.0,
                           double initial_time_scale = 1.0, bool use_time_since_epoch = false,
                           const std::string& name = "realtime_clock")
      : RealtimeClock(ArgList{Arg{"initial_time_offset", initial_time_offset},
                              Arg{"initial_time_scale", initial_time_scale},
                              Arg{"use_time_since_epoch", use_time_since_epoch}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
  }

  /* Trampolines (need one for each virtual function) */
  double time() const override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(double, RealtimeClock, time);
  }
  int64_t timestamp() const override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(int64_t, RealtimeClock, timestamp);
  }
  void sleep_for(int64_t duration_ns) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, RealtimeClock, sleep_for, duration_ns);
  }
  void sleep_until(int64_t target_time_ns) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, RealtimeClock, sleep_until, target_time_ns);
  }
};

class PyManualClock : public ManualClock {
 public:
  /* Inherit the constructors */
  using ManualClock::ManualClock;

  // Define a constructor that fully initializes the object.
  explicit PyManualClock(Fragment* fragment, int64_t initial_timestamp = 0LL,
                         const std::string& name = "manual_clock")
      : ManualClock(Arg{"initial_timestamp", initial_timestamp}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_.get());
  }

  /* Trampolines (need one for each virtual function) */
  double time() const override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(double, ManualClock, time);
  }
  int64_t timestamp() const override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(int64_t, ManualClock, timestamp);
  }
  void sleep_for(int64_t duration_ns) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, ManualClock, sleep_for, duration_ns);
  }
  void sleep_until(int64_t target_time_ns) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, ManualClock, sleep_until, target_time_ns);
  }
};

void init_clocks(py::module_& m) {
  py::class_<Clock, gxf::GXFResource, std::shared_ptr<Clock>>(m, "Clock", doc::Clock::doc_Clock);

  py::class_<RealtimeClock, PyRealtimeClock, Clock, std::shared_ptr<RealtimeClock>>(
      m, "RealtimeClock", doc::RealtimeClock::doc_RealtimeClock)
      .def(py::init<Fragment*, double, double, bool, const std::string&>(),
           "fragment"_a,
           "initial_time_offset"_a = 0.0,
           "initial_time_scale"_a = 1.0,
           "use_time_since_epoch"_a = false,
           "name"_a = "realtime_clock"s,
           doc::RealtimeClock::doc_RealtimeClock)
      .def("time", &RealtimeClock::time, doc::Clock::doc_time)
      .def("timestamp", &RealtimeClock::timestamp, doc::Clock::doc_timestamp)
      // define a version of sleep_for that can take either int or datetime.timedelta
      .def(
          "sleep_for",
          [](RealtimeClock& clk, const py::object& duration) {
            clk.sleep_for(holoscan::get_duration_ns(duration));
          },
          py::call_guard<py::gil_scoped_release>(),
          doc::Clock::doc_sleep_for)
      .def("sleep_until",
           &RealtimeClock::sleep_until,
           "target_time_ns"_a,
           doc::Clock::doc_sleep_until)
      .def("set_time_scale",
           &RealtimeClock::set_time_scale,
           "time_scale"_a,
           doc::RealtimeClock::doc_set_time_scale);

  py::class_<ManualClock, PyManualClock, Clock, std::shared_ptr<ManualClock>>(
      m, "ManualClock", doc::ManualClock::doc_ManualClock)
      .def(py::init<Fragment*, int64_t, const std::string&>(),
           "fragment"_a,
           "initial_timestamp"_a = 0LL,
           "name"_a = "realtime_clock"s,
           doc::ManualClock::doc_ManualClock)
      .def("time", &ManualClock::time, doc::Clock::doc_time)
      .def("timestamp", &ManualClock::timestamp, doc::Clock::doc_timestamp)
      // define a version of sleep_for that can take either int or datetime.timedelta
      .def(
          "sleep_for",
          [](ManualClock& clk, const py::object& duration) {
            clk.sleep_for(holoscan::get_duration_ns(duration));
          },
          py::call_guard<py::gil_scoped_release>(),
          doc::Clock::doc_sleep_for)
      .def("sleep_until",
           &ManualClock::sleep_until,
           "target_time_ns"_a,
           doc::Clock::doc_sleep_until);
}

}  // namespace holoscan
