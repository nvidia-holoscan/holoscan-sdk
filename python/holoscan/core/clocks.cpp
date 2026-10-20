/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "clocks.hpp"

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/time/clock_registry.hpp>

#include "graph_pydoc.hpp"
#include "module_name.hpp"

namespace holoscan {
namespace {

void install_clock_tokens(nb::module_& module) {
  constexpr std::array<std::string_view, 3U> names{
      "RealtimeClock", "ManualClock", "SyntheticClock"};
  for (const std::string_view name : names) {
    auto row = authoring::authoring_registry().clock_type("holoscan." + std::string{name});
    if (!row) {
      throw RuntimeError{row.error()};
    }
    module.attr(std::string{name}.c_str()) = PyClockType{&row->get()};
  }
}

}  // namespace

std::string PyClockType::name() const {
  return std::string{row_->name};
}

std::string PyClockType::repr() const {
  return fmt::format("<ClockType {}>", row_->name);
}

std::string PyClockKey::name() const {
  return std::string{key_.name()};
}

bool PyClockKey::is_default() const noexcept {
  return key_.is_default();
}

std::string PyClockKey::repr() const {
  return key_.is_default() ? "<ClockKey default>" : fmt::format("<ClockKey '{}'>", name());
}

void init_clocks(nb::module_& module) {
  auto clock_type = python::bind_class_in_module<PyClockType>(
      module, "holoscan.clocks", "ClockType", doc::graph::kClockType);
  clock_type.def_prop_ro("name", &PyClockType::name)
      .def_prop_ro("_registry_name", &PyClockType::name)
      .def("__repr__", &PyClockType::repr);
  auto clock_key_type = python::bind_class_in_module<PyClockKey>(
      module, "holoscan.clocks", "ClockKey", doc::graph::kClockKey);
  clock_key_type.def_prop_ro("name", &PyClockKey::name)
      .def_prop_ro("is_default", &PyClockKey::is_default)
      .def("__repr__", &PyClockKey::repr);
  auto clock_capabilities_type = python::bind_class_in_module<ClockCapabilities>(
      module, "holoscan.clocks", "ClockCapabilities");
  clock_capabilities_type.def_prop_ro("can_read_now", &ClockCapabilities::can_read_now)
      .def_prop_ro("can_wait_until", &ClockCapabilities::can_wait_until)
      .def_prop_ro("steady", &ClockCapabilities::steady)
      .def_prop_ro("has_outgoing_conversions", &ClockCapabilities::has_outgoing_conversions)
      .def_prop_ro("has_incoming_conversions", &ClockCapabilities::has_incoming_conversions);
  auto clock_ref_type =
      python::bind_class_in_module<ClockRef>(module, "holoscan.clocks", "ClockRef");
  clock_ref_type.def_prop_ro("id", [](const ClockRef& reference) { return reference.id().value; })
      .def_prop_ro("capabilities", &ClockRef::capabilities)
      .def("now",
           [](const ClockRef& reference) {
             auto result = reference.now();
             if (!result) {
               throw RuntimeError{result.error()};
             }
             return *result;
           })
      .def("at_ns", &ClockRef::at_ns, nb::arg("timestamp_ns"));
  auto clock_rate_ratio_type =
      python::bind_class_in_module<ClockRateRatio>(module, "holoscan.clocks", "ClockRateRatio");
  clock_rate_ratio_type
      .def(
          "__init__",
          [](ClockRateRatio* self, std::int64_t numerator, std::uint64_t denominator) {
            new (self) ClockRateRatio{numerator, denominator};
          },
          nb::arg("numerator") = 1,
          nb::arg("denominator") = 1U)
      .def_rw("numerator", &ClockRateRatio::numerator)
      .def_rw("denominator", &ClockRateRatio::denominator);
  auto affine_clock_conversion_type = python::bind_class_in_module<AffineClockConversion>(
      module, "holoscan.clocks", "AffineClockConversion");
  affine_clock_conversion_type
      .def(
          "__init__",
          [](AffineClockConversion* self, ClockRateRatio scale, std::int64_t offset_ns) {
            new (self) AffineClockConversion{std::move(scale), offset_ns};
          },
          nb::arg("scale") = ClockRateRatio{},
          nb::arg("offset_ns") = 0)
      .def_rw("scale", &AffineClockConversion::scale)
      .def_rw("offset_ns", &AffineClockConversion::offset_ns);
  install_clock_tokens(module);
}

}  // namespace holoscan
