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

#include <pybind11/chrono.h>  // will include timedelta.h for us
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <variant>

#include "../core/component_util.hpp"
#include "./clocks_pydoc.hpp"
#include "holoscan/core/clock.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/resources/gxf/clock.hpp"
#include "holoscan/core/resources/gxf/manual_clock.hpp"
#include "holoscan/core/resources/gxf/realtime_clock.hpp"
#include "holoscan/core/resources/gxf/synthetic_clock.hpp"
#include "holoscan/core/subgraph.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

namespace {
std::once_flag datetime_init_flag;
}

int64_t get_duration_ns(const py::object& duration) {
  if (py::isinstance<py::int_>(duration)) {
    return py::cast<int64_t>(duration);
  }
  // Must acquire GIL before calling C API functions like PyDelta_Check
  py::gil_scoped_acquire scope_guard;

  // Thread-safe initialization using std::once_flag
  // Must initialize PyDateTime_IMPORT here in order to be able to use PyDelta_Check below
  std::call_once(datetime_init_flag, []() { PyDateTime_IMPORT; });

  if (PyDelta_Check(duration.ptr())) {
    // timedelta stores integer days, seconds, microseconds
    int64_t days = PyDateTime_DELTA_GET_DAYS(duration.ptr());
    int64_t seconds = PyDateTime_DELTA_GET_SECONDS(duration.ptr());
    if (days > 0) {
      int seconds_per_day = 24 * 3600;
      seconds += days * seconds_per_day;
    }
    int64_t microseconds = PyDateTime_DELTA_GET_MICROSECONDS(duration.ptr());
    if (seconds > 0) {
      microseconds += 1000000 * seconds;
    }
    int64_t delta_ns = 1000 * microseconds;
    return delta_ns;
  }
  throw std::runtime_error("expected an integer or datetime.timedelta type");
}

class PyRealtimeClock : public RealtimeClock {
 public:
  /* Inherit the constructors */
  using RealtimeClock::RealtimeClock;

  // Define a constructor that fully initializes the object.
  explicit PyRealtimeClock(const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
                           double initial_time_offset = 0.0, double initial_time_scale = 1.0,
                           bool use_time_since_epoch = false,
                           const std::string& name = "realtime_clock") {
    // Add arguments individually to handle virtual inheritance properly
    add_arg(Arg{"initial_time_offset", initial_time_offset});
    add_arg(Arg{"initial_time_scale", initial_time_scale});
    add_arg(Arg{"use_time_since_epoch", use_time_since_epoch});

    init_component_base(this, fragment_or_subgraph, name, "resource");
  }

  /* Trampolines (need one for each virtual function) */
  [[nodiscard]] double time() const override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(double, RealtimeClock, time);
  }
  [[nodiscard]] int64_t timestamp() const override {
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
  explicit PyManualClock(const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
                         int64_t initial_timestamp = 0LL,
                         const std::string& name = "manual_clock") {
    // Add arguments individually to handle virtual inheritance properly
    add_arg(Arg{"initial_timestamp", initial_timestamp});

    init_component_base(this, fragment_or_subgraph, name, "resource");
  }

  /* Trampolines (need one for each virtual function) */
  [[nodiscard]] double time() const override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(double, ManualClock, time);
  }
  [[nodiscard]] int64_t timestamp() const override {
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

class PySyntheticClock : public SyntheticClock {
 public:
  /* Inherit the constructors */
  using SyntheticClock::SyntheticClock;

  // Define a constructor that fully initializes the object.
  explicit PySyntheticClock(const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
                            int64_t initial_timestamp = 0LL,
                            const std::string& name = "synthetic_clock") {
    // Add arguments individually to handle virtual inheritance properly
    add_arg(Arg{"initial_timestamp", initial_timestamp});

    init_component_base(this, fragment_or_subgraph, name, "resource");
  }

  /* Trampolines (need one for each virtual function) */
  [[nodiscard]] double time() const override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(double, SyntheticClock, time);
  }
  [[nodiscard]] int64_t timestamp() const override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(int64_t, SyntheticClock, timestamp);
  }
  void sleep_for(int64_t duration_ns) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, SyntheticClock, sleep_for, duration_ns);
  }
  void sleep_until(int64_t target_time_ns) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, SyntheticClock, sleep_until, target_time_ns);
  }
};

void init_clocks(py::module_& m) {
  // GXF-specific clock implementation
  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<gxf::Clock, ClockInterface, gxf::GXFResource, std::shared_ptr<gxf::Clock>>(m,
                                                                                        "GXFClock");

  py::class_<RealtimeClock, PyRealtimeClock, gxf::Clock, std::shared_ptr<RealtimeClock>>(
      m, "RealtimeClock", doc::RealtimeClock::doc_RealtimeClock)
      .def(py::init<std::variant<Fragment*, Subgraph*>, double, double, bool, const std::string&>(),
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

  py::class_<ManualClock, PyManualClock, gxf::Clock, std::shared_ptr<ManualClock>>(
      m, "ManualClock", doc::ManualClock::doc_ManualClock)
      .def(py::init<std::variant<Fragment*, Subgraph*>, int64_t, const std::string&>(),
           "fragment"_a,
           "initial_timestamp"_a = 0LL,
           "name"_a = "manual_clock"s,
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
          "duration"_a,
          doc::Clock::doc_sleep_for)
      .def("sleep_until",
           &ManualClock::sleep_until,
           "target_time_ns"_a,
           doc::Clock::doc_sleep_until);

  py::class_<SyntheticClock, PySyntheticClock, gxf::Clock, std::shared_ptr<SyntheticClock>>(
      m, "SyntheticClock", doc::SyntheticClock::doc_SyntheticClock)
      .def(py::init<std::variant<Fragment*, Subgraph*>, int64_t, const std::string&>(),
           "fragment"_a,
           "initial_timestamp"_a = 0LL,
           "name"_a = "synthetic_clock"s,
           doc::SyntheticClock::doc_SyntheticClock)
      .def("time", &SyntheticClock::time, doc::Clock::doc_time)
      .def("timestamp", &SyntheticClock::timestamp, doc::Clock::doc_timestamp)
      // define a version of sleep_for that can take either int or datetime.timedelta
      .def(
          "sleep_for",
          [](SyntheticClock& clk, const py::object& duration) {
            clk.sleep_for(holoscan::get_duration_ns(duration));
          },
          py::call_guard<py::gil_scoped_release>(),
          "duration"_a,
          doc::Clock::doc_sleep_for)
      .def("sleep_until",
           &SyntheticClock::sleep_until,
           "target_time_ns"_a,
           doc::Clock::doc_sleep_until)
      .def("advance_to",
           &SyntheticClock::advance_to,
           "new_time_ns"_a,
           doc::SyntheticClock::doc_advance_to)
      // define a version of advance_by that can take either int or datetime.timedelta
      .def(
          "advance_by",
          [](SyntheticClock& clk, const py::object& duration) {
            clk.advance_by(holoscan::get_duration_ns(duration));
          },
          py::call_guard<py::gil_scoped_release>(),
          "time_delta_ns"_a,
          doc::SyntheticClock::doc_advance_by);
}

}  // namespace holoscan
