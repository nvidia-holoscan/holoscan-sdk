/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include "./clock.hpp"
#include "./clock_pydoc.hpp"
#include "holoscan/core/clock.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/resources/gxf/manual_clock.hpp"
#include "holoscan/core/resources/gxf/realtime_clock.hpp"
#include "holoscan/core/resources/gxf/synthetic_clock.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace {
std::once_flag datetime_init_flag;
}

namespace py = pybind11;

namespace holoscan {

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

void init_clock(py::module_& m) {
  // Pure clock interface
  py::class_<ClockInterface, std::shared_ptr<ClockInterface>>(
      m, "ClockInterface", "Pure clock interface")
      .def("time", &ClockInterface::time, doc::Clock::doc_time)
      .def("timestamp", &ClockInterface::timestamp, doc::Clock::doc_timestamp)
      // define a version of sleep_for that can take either int or datetime.timedelta
      .def(
          "sleep_for",
          [](ClockInterface& clk, const py::object& duration) {
            clk.sleep_for(holoscan::get_duration_ns(duration));
          },
          py::call_guard<py::gil_scoped_release>(),
          doc::Clock::doc_sleep_for)
      .def("sleep_until",
           &ClockInterface::sleep_until,
           "target_time_ns"_a,
           doc::Clock::doc_sleep_until);

  // Clock resource (Bridge pattern)
  py::class_<Clock, Resource, std::shared_ptr<Clock>>(m, "Clock", doc::Clock::doc_Clock_args_kwargs)
      .def(py::init<std::shared_ptr<ClockInterface>>(),
           "clock_impl"_a,
           doc::Clock::doc_Clock_args_kwargs)
      .def("set_clock_impl", &Clock::set_clock_impl, "clock_impl"_a, doc::Clock::doc_set_clock_impl)
      .def("clock_impl", &Clock::clock_impl, doc::Clock::doc_clock_impl)
      .def(
          "cast_to",
          [](Clock& clk, const py::object& type) -> py::object {
            auto impl = clk.clock_impl();
            if (!impl) {
              return py::none();
            }

            try {
              auto builtins = py::module_::import("builtins");
              auto issubclass = builtins.attr("issubclass");

              if (py::bool_(issubclass(type, py::type::of<RealtimeClock>()))) {
                auto realtime_clock = std::dynamic_pointer_cast<RealtimeClock>(impl);
                if (realtime_clock) {
                  return py::cast(realtime_clock);
                } else {
                  throw std::runtime_error("Clock could not be cast to RealtimeClock");
                }
              } else if (py::bool_(issubclass(type, py::type::of<ManualClock>()))) {
                auto manual_clock = std::dynamic_pointer_cast<ManualClock>(impl);
                if (manual_clock) {
                  return py::cast(manual_clock);
                } else {
                  throw std::runtime_error("Clock could not be cast to ManualClock");
                }
              } else if (py::bool_(issubclass(type, py::type::of<SyntheticClock>()))) {
                auto synthetic_clock = std::dynamic_pointer_cast<SyntheticClock>(impl);
                if (synthetic_clock) {
                  return py::cast(synthetic_clock);
                } else {
                  throw std::runtime_error("Clock could not be cast to SyntheticClock");
                }
              } else {
                // Type is a valid class but not a supported clock type
                std::string type_name = py::str(type);
                throw std::runtime_error("Unsupported clock type: " + type_name);
              }
            } catch (const py::error_already_set&) {
              // If issubclass fails (e.g., type is not a class), raise a more helpful error
              throw std::runtime_error(
                  "cast_to() requires a supported clock class type (RealtimeClock, "
                  "ManualClock)");
            }
          },
          "type"_a,
          doc::Clock::doc_cast_to)
      .def("time", &Clock::time, doc::Clock::doc_time)
      .def("timestamp", &Clock::timestamp, doc::Clock::doc_timestamp)
      // define a version of sleep_for that can take either int or datetime.timedelta
      .def(
          "sleep_for",
          [](Clock& clk, const py::object& duration) {
            clk.sleep_for(holoscan::get_duration_ns(duration));
          },
          py::call_guard<py::gil_scoped_release>(),
          doc::Clock::doc_sleep_for)
      .def("sleep_until", &Clock::sleep_until, "target_time_ns"_a, doc::Clock::doc_sleep_until);
}

}  // namespace holoscan
