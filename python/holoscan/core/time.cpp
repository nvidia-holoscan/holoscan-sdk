/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "time.hpp"

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <functional>
#include <limits>
#include <new>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/core/errors.hpp>
#include <holoscan/time/time_point.hpp>

#include "module_name.hpp"

namespace holoscan {
namespace {

[[noreturn]] void throw_duration_error(std::string_view token, std::string_view field = {}) {
  std::string message =
      field.empty() ? std::string{token} : fmt::format("{} field_name={}", token, field);
  throw RuntimeError{Error{ErrorCode::kInvalidArgument, std::move(message)}};
}

std::int64_t checked_duration(std::int64_t value, std::int64_t scale) {
  const __int128 product = static_cast<__int128>(value) * static_cast<__int128>(scale);
  if (product < std::numeric_limits<std::int64_t>::min() ||
      product > std::numeric_limits<std::int64_t>::max()) {
    throw_duration_error("PARAM_VALIDATION_FAILED category=out_of_range");
  }
  return static_cast<std::int64_t>(product);
}

}  // namespace

std::int64_t python_duration_nanoseconds(nb::handle value, std::string_view field,
                                         std::string_view type_error_token,
                                         std::string_view range_error_token) {
  if (nb::isinstance<PyDuration>(value)) {
    return nb::cast<const PyDuration&>(value).nanoseconds;
  }
  const nb::object timedelta_type = nb::module_::import_("datetime").attr("timedelta");
  if (!nb::isinstance(value, timedelta_type)) {
    throw_duration_error(type_error_token, field);
  }
  const std::int64_t days = nb::cast<std::int64_t>(value.attr("days"));
  const std::int64_t seconds = nb::cast<std::int64_t>(value.attr("seconds"));
  const std::int64_t microseconds = nb::cast<std::int64_t>(value.attr("microseconds"));
  const __int128 total = static_cast<__int128>(days) * 86'400'000'000'000LL +
                         static_cast<__int128>(seconds) * 1'000'000'000LL +
                         static_cast<__int128>(microseconds) * 1'000LL;
  if (total < std::numeric_limits<std::int64_t>::min() ||
      total > std::numeric_limits<std::int64_t>::max()) {
    throw_duration_error(range_error_token, field);
  }
  return static_cast<std::int64_t>(total);
}

void init_time(nb::module_& module) {
  // Exposes an absolute timestamp together with the clock domain that makes it meaningful.
  auto time_point_type =
      python::bind_class_in_module<TimePoint>(module, "holoscan.time", "TimePoint");
  time_point_type
      .def(
          "__init__",
          [](TimePoint* self, std::int64_t timestamp_ns, std::uint64_t clock_id) {
            new (self)
                TimePoint{.timestamp_ns = timestamp_ns, .clock_id = ClockId{.value = clock_id}};
          },
          nb::arg("timestamp_ns"),
          nb::arg("clock_id"))
      .def_prop_ro("timestamp_ns", [](TimePoint value) { return value.timestamp_ns; })
      .def_prop_ro("clock_id", [](TimePoint value) { return value.clock_id.value; })
      .def_prop_ro("valid", &TimePoint::valid)
      .def(
          "__eq__", [](TimePoint lhs, TimePoint rhs) { return lhs == rhs; }, nb::is_operator())
      .def("__hash__", [](TimePoint value) { return std::hash<TimePoint>{}(value); })
      .def("__repr__", [](TimePoint value) {
        return "TimePoint(timestamp_ns=" + std::to_string(value.timestamp_ns) +
               ", clock_id=" + std::to_string(value.clock_id.value) + ")";
      });
  // Exposes a deterministic logical timestamp and its same-time microstep ordering coordinate.
  auto tag_time_type = python::bind_class_in_module<TagTime>(module, "holoscan.time", "TagTime");
  tag_time_type
      .def(
          nb::init<std::int64_t, std::uint32_t>(), nb::arg("logical_ns"), nb::arg("microstep") = 0U)
      .def_ro("logical_ns", &TagTime::logical_ns)
      .def_ro("microstep", &TagTime::microstep)
      .def_prop_ro("valid", &TagTime::valid)
      .def(
          "__eq__", [](TagTime lhs, TagTime rhs) { return lhs == rhs; }, nb::is_operator())
      .def("__hash__", [](TagTime value) { return std::hash<TagTime>{}(value); })
      .def("__repr__", [](TagTime value) {
        return "TagTime(logical_ns=" + std::to_string(value.logical_ns) +
               ", microstep=" + std::to_string(value.microstep) + ")";
      });
  auto duration_type =
      python::bind_class_in_module<PyDuration>(module, "holoscan.time", "Duration");
  duration_type
      .def_prop_ro("nanoseconds", [](const PyDuration& value) { return value.nanoseconds; })
      .def("__repr__", [](const PyDuration& value) {
        return fmt::format("<Duration {} ns>", value.nanoseconds);
      });
  python::bind_functions_in_module(module, "holoscan.time", [&] {
    module.def("ns", [](std::int64_t value) { return PyDuration{value}; }, nb::arg("value"));
    module.def(
        "us",
        [](std::int64_t value) { return PyDuration{checked_duration(value, 1'000LL)}; },
        nb::arg("value"));
    module.def(
        "ms",
        [](std::int64_t value) { return PyDuration{checked_duration(value, 1'000'000LL)}; },
        nb::arg("value"));
  });
}

}  // namespace holoscan
