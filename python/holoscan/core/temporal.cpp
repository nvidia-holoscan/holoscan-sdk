/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "temporal.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <chrono>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "clocks.hpp"
#include "module_name.hpp"
#include "native_io.hpp"
#include "time.hpp"

namespace holoscan {
namespace {

[[noreturn]] void throw_temporal_error(std::string_view token, std::string_view field = {}) {
  std::string message{token};
  if (!field.empty()) {
    message.append(" field_name=");
    message.append(field);
  }
  throw RuntimeError{Error{ErrorCode::kInvalidArgument, std::move(message)}};
}

// Converts the accepted Python duration vocabulary to Core's exact nanosecond representation.
[[nodiscard]] std::chrono::nanoseconds python_duration(nb::handle value, std::string_view field) {
  return std::chrono::nanoseconds{python_duration_nanoseconds(value, field)};
}

// Converts an optional Python ClockKey into the owning Core trigger key.
[[nodiscard]] ClockKey clock_key_option(nb::handle value, std::string_view field) {
  if (value.is_none()) {
    return {};
  }
  if (!nb::isinstance<PyClockKey>(value)) {
    throw_temporal_error("AUTHORING_TEMPORAL_CLOCK_TYPE_MISMATCH", field);
  }
  return nb::cast<const PyClockKey&>(value).value();
}

// Validates one nonnegative bounded Python integer used by temporal policy values.
template <typename T>
T unsigned_option(nb::handle value, std::string_view field) {
  static_assert(std::is_unsigned_v<T>);
  if (!nb::isinstance<nb::int_>(value) || nb::isinstance<nb::bool_>(value)) {
    throw_temporal_error("AUTHORING_TEMPORAL_OPTION_INVALID", field);
  }
  const unsigned long long result = PyLong_AsUnsignedLongLong(value.ptr());
  if (PyErr_Occurred() || result > std::numeric_limits<T>::max()) {
    PyErr_Clear();
    throw_temporal_error("AUTHORING_TEMPORAL_OPTION_INVALID", field);
  }
  return static_cast<T>(result);
}

// Owns the input port referenced by a Python OnEach trigger until contract conversion.
class PyOnEach {
 public:
  explicit PyOnEach(PyNativePort port) : port_(std::move(port)) {}

  [[nodiscard]] const PyNativePort& port() const noexcept { return port_; }
  [[nodiscard]] OnEach value() const { return OnEach{.port = port_.contract_input_ref()}; }

 private:
  PyNativePort port_;
};

// Owns the input-port list and duration backing a Python OnAligned trigger.
class PyOnAligned {
 public:
  PyOnAligned(std::vector<PyNativePort> ports, std::chrono::nanoseconds window)
      : ports_(std::move(ports)), window_(window) {}

  [[nodiscard]] const std::vector<PyNativePort>& ports() const noexcept { return ports_; }
  [[nodiscard]] std::chrono::nanoseconds window() const noexcept { return window_; }
  [[nodiscard]] OnAligned value() const {
    std::vector<PortRef> references;
    references.reserve(ports_.size());
    for (const PyNativePort& port : ports_) {
      references.push_back(port.contract_input_ref());
    }
    return OnAligned{.ports = PortList{std::move(references)}, .window = window_};
  }

 private:
  std::vector<PyNativePort> ports_;
  std::chrono::nanoseconds window_{};
};

// Owns the anchor and clock key backing a Python OnWindow trigger.
class PyOnWindow {
 public:
  PyOnWindow(PyNativePort anchor, std::chrono::nanoseconds max_wait, ClockKey wait_clock,
             std::uint32_t max_pending_windows)
      : anchor_(std::move(anchor)),
        max_wait_(max_wait),
        wait_clock_(std::move(wait_clock)),
        max_pending_windows_(max_pending_windows) {}

  [[nodiscard]] const PyNativePort& anchor() const noexcept { return anchor_; }
  [[nodiscard]] std::chrono::nanoseconds max_wait() const noexcept { return max_wait_; }
  [[nodiscard]] const ClockKey& wait_clock() const noexcept { return wait_clock_; }
  [[nodiscard]] std::uint32_t max_pending_windows() const noexcept { return max_pending_windows_; }
  [[nodiscard]] OnWindow value() const {
    return OnWindow{.anchor = anchor_.contract_input_ref(),
                    .max_wait = max_wait_,
                    .wait_clock = wait_clock_,
                    .max_pending_windows = max_pending_windows_};
  }

 private:
  PyNativePort anchor_;
  std::chrono::nanoseconds max_wait_{};
  ClockKey wait_clock_{};
  std::uint32_t max_pending_windows_{1U};
};

}  // namespace

void init_temporal(nb::module_& module) {
  auto exact_type = python::bind_class_in_module<Exact>(module, "holoscan", "Exact");
  exact_type.def(nb::init<>());
  auto latest_within_type =
      python::bind_class_in_module<LatestWithin>(module, "holoscan", "LatestWithin");
  latest_within_type
      .def(
          "__init__",
          [](LatestWithin* self, nb::handle max_age) {
            new (self) LatestWithin{.max_age = python_duration(max_age, "max_age")};
          },
          nb::kw_only(),
          nb::arg("max_age"),
          nb::sig("def __init__(self, *, max_age: datetime.timedelta | "
                  "holoscan.time.Duration) -> None"))
      .def_prop_ro("max_age",
                   [](const LatestWithin& value) { return PyDuration{value.max_age.count()}; });
  auto drain_all_type = python::bind_class_in_module<DrainAll>(module, "holoscan", "DrainAll");
  drain_all_type.def(nb::init<>());
  auto bracketed_window_type =
      python::bind_class_in_module<BracketedWindow>(module, "holoscan", "BracketedWindow");
  bracketed_window_type
      .def(
          "__init__",
          [](BracketedWindow* self,
             nb::handle lookbehind,
             nb::handle lookahead,
             nb::handle max_samples) {
            new (self) BracketedWindow{
                .lookbehind = unsigned_option<std::uint32_t>(lookbehind, "lookbehind"),
                .lookahead = unsigned_option<std::uint32_t>(lookahead, "lookahead"),
                .max_samples = unsigned_option<std::uint32_t>(max_samples, "max_samples"),
            };
          },
          nb::kw_only(),
          nb::arg("lookbehind") = 1U,
          nb::arg("lookahead") = 1U,
          nb::arg("max_samples") = 0U,
          nb::sig("def __init__(self, *, lookbehind: int = 1, lookahead: int = 1, "
                  "max_samples: int = 0) -> None"))
      .def_ro("lookbehind", &BracketedWindow::lookbehind)
      .def_ro("lookahead", &BracketedWindow::lookahead)
      .def_ro("max_samples", &BracketedWindow::max_samples);
  auto skip_type = python::bind_class_in_module<Skip>(module, "holoscan", "Skip");
  skip_type.def(nb::init<>());
  auto safe_default_type =
      python::bind_class_in_module<SafeDefault>(module, "holoscan", "SafeDefault");
  safe_default_type.def(nb::init<>());
  auto reuse_last_type = python::bind_class_in_module<ReuseLast>(module, "holoscan", "ReuseLast");
  reuse_last_type
      .def(
          "__init__",
          [](ReuseLast* self, nb::handle max_age) {
            new (self) ReuseLast{.max_age = python_duration(max_age, "max_age")};
          },
          nb::kw_only(),
          nb::arg("max_age"),
          nb::sig("def __init__(self, *, max_age: datetime.timedelta | "
                  "holoscan.time.Duration) -> None"))
      .def_prop_ro("max_age",
                   [](const ReuseLast& value) { return PyDuration{value.max_age.count()}; });

  auto on_any_input_type =
      python::bind_class_in_module<OnAnyInput>(module, "holoscan", "OnAnyInput");
  on_any_input_type.def(nb::init<>());
  auto on_each_type = python::bind_class_in_module<PyOnEach>(module, "holoscan", "OnEach");
  on_each_type
      .def(
          "__init__",
          [](PyOnEach* self, const PyNativePort& port) { new (self) PyOnEach{port}; },
          nb::arg("port"),
          nb::sig("def __init__(self, port: holoscan.OperatorPort) -> None"))
      .def_prop_ro("port", &PyOnEach::port, nb::rv_policy::reference_internal);
  auto on_clock_type = python::bind_class_in_module<OnClock>(module, "holoscan", "OnClock");
  on_clock_type
      .def(
          "__init__",
          [](OnClock* self, nb::handle period, nb::handle clock, nb::handle start_at) {
            const std::optional<TimePoint> phase =
                start_at.is_none() ? std::nullopt : std::optional{nb::cast<TimePoint>(start_at)};
            new (self) OnClock{
                .period = python_duration(period, "period"),
                .clock = clock_key_option(clock, "clock"),
                .start_at = phase,
            };
          },
          nb::kw_only(),
          nb::arg("period"),
          nb::arg("clock").none() = nb::none(),
          nb::arg("start_at").none() = nb::none(),
          nb::sig("def __init__(self, *, period: datetime.timedelta | "
                  "holoscan.time.Duration, clock: holoscan.clocks.ClockKey | None = None, "
                  "start_at: holoscan.time.TimePoint | None = None) -> None"))
      .def_prop_ro("period", [](const OnClock& value) { return PyDuration{value.period.count()}; })
      .def_prop_ro("clock", [](const OnClock& value) { return PyClockKey{value.clock}; })
      .def_prop_ro("start_at", [](const OnClock& value) -> nb::object {
        return value.start_at ? nb::cast(*value.start_at) : nb::none();
      });
  auto on_aligned_type = python::bind_class_in_module<PyOnAligned>(module, "holoscan", "OnAligned");
  on_aligned_type
      .def(
          "__init__",
          [](PyOnAligned* self, nb::handle port_values, nb::handle window) {
            if (!nb::isinstance<nb::sequence>(port_values)) {
              throw_temporal_error("AUTHORING_TEMPORAL_PORT_LIST_REQUIRED", "ports");
            }
            const nb::sequence sequence = nb::cast<nb::sequence>(port_values);
            std::vector<PyNativePort> ports;
            ports.reserve(nb::len(sequence));
            for (const nb::handle port : sequence) {
              try {
                ports.push_back(nb::cast<const PyNativePort&>(port));
              } catch (const nb::cast_error&) {
                throw_temporal_error("AUTHORING_TEMPORAL_PORT_TYPE_MISMATCH", "ports");
              }
            }
            new (self) PyOnAligned{std::move(ports), python_duration(window, "window")};
          },
          nb::kw_only(),
          nb::arg("ports"),
          nb::arg("window"),
          nb::sig("def __init__(self, *, ports: typing.Sequence[holoscan.OperatorPort], "
                  "window: datetime.timedelta | holoscan.time.Duration) -> None"))
      .def_prop_ro("ports", &PyOnAligned::ports)
      .def_prop_ro("window",
                   [](const PyOnAligned& value) { return PyDuration{value.window().count()}; });
  auto on_window_type = python::bind_class_in_module<PyOnWindow>(module, "holoscan", "OnWindow");
  on_window_type
      .def(
          "__init__",
          [](PyOnWindow* self,
             const PyNativePort& anchor,
             nb::handle max_wait,
             nb::handle wait_clock,
             nb::handle max_pending_windows) {
            new (self) PyOnWindow{
                anchor,
                python_duration(max_wait, "max_wait"),
                clock_key_option(wait_clock, "wait_clock"),
                unsigned_option<std::uint32_t>(max_pending_windows, "max_pending_windows")};
          },
          nb::kw_only(),
          nb::arg("anchor"),
          nb::arg("max_wait"),
          nb::arg("wait_clock").none() = nb::none(),
          nb::arg("max_pending_windows") = 1U,
          nb::sig("def __init__(self, *, anchor: holoscan.OperatorPort, max_wait: "
                  "datetime.timedelta | holoscan.time.Duration, wait_clock: "
                  "holoscan.clocks.ClockKey | None = None, max_pending_windows: int = 1) -> None"))
      .def_prop_ro("anchor", &PyOnWindow::anchor, nb::rv_policy::reference_internal)
      .def_prop_ro("max_wait",
                   [](const PyOnWindow& value) { return PyDuration{value.max_wait().count()}; })
      .def_prop_ro("wait_clock",
                   [](const PyOnWindow& value) { return PyClockKey{value.wait_clock()}; })
      .def_prop_ro("max_pending_windows", &PyOnWindow::max_pending_windows);

  auto contract_type = python::bind_class_in_module<Contract>(module, "holoscan", "Contract");
  contract_type.def(nb::init<>())
      .def(
          "trigger",
          [](Contract& contract, OnAnyInput trigger) -> Contract& {
            return contract.trigger(std::move(trigger));
          },
          nb::arg("trigger"),
          nb::rv_policy::reference_internal,
          nb::sig("def trigger(self, trigger: holoscan.OnAnyInput) -> typing.Self"))
      .def(
          "trigger",
          [](Contract& contract, const PyOnEach& trigger) -> Contract& {
            return contract.trigger(trigger.value());
          },
          nb::arg("trigger"),
          nb::rv_policy::reference_internal,
          nb::sig("def trigger(self, trigger: holoscan.OnEach) -> typing.Self"))
      .def(
          "trigger",
          [](Contract& contract, OnClock trigger) -> Contract& {
            return contract.trigger(std::move(trigger));
          },
          nb::arg("trigger"),
          nb::rv_policy::reference_internal,
          nb::sig("def trigger(self, trigger: holoscan.OnClock) -> typing.Self"))
      .def(
          "trigger",
          [](Contract& contract, const PyOnAligned& trigger) -> Contract& {
            return contract.trigger(trigger.value());
          },
          nb::arg("trigger"),
          nb::rv_policy::reference_internal,
          nb::sig("def trigger(self, trigger: holoscan.OnAligned) -> typing.Self"))
      .def(
          "trigger",
          [](Contract& contract, const PyOnWindow& trigger) -> Contract& {
            return contract.trigger(trigger.value());
          },
          nb::arg("trigger"),
          nb::rv_policy::reference_internal,
          nb::sig("def trigger(self, trigger: holoscan.OnWindow) -> typing.Self"))
      .def(
          "input",
          [](Contract& contract, const PyNativePort& port, Exact selector) -> Contract& {
            return port.contract_input(contract, selector);
          },
          nb::arg("port"),
          nb::arg("selector"),
          nb::rv_policy::reference_internal,
          nb::sig("def input(self, port: holoscan.OperatorPort, selector: holoscan.Exact) -> "
                  "typing.Self"))
      .def(
          "input",
          [](Contract& contract, const PyNativePort& port, LatestWithin selector) -> Contract& {
            return port.contract_input(contract, selector);
          },
          nb::arg("port"),
          nb::arg("selector"),
          nb::rv_policy::reference_internal,
          nb::sig("def input(self, port: holoscan.OperatorPort, selector: "
                  "holoscan.LatestWithin) -> typing.Self"))
      .def(
          "input",
          [](Contract& contract, const PyNativePort& port, DrainAll selector) -> Contract& {
            return port.contract_input(contract, selector);
          },
          nb::arg("port"),
          nb::arg("selector"),
          nb::rv_policy::reference_internal,
          nb::sig("def input(self, port: holoscan.OperatorPort, selector: holoscan.DrainAll) -> "
                  "typing.Self"))
      .def(
          "input",
          [](Contract& contract, const PyNativePort& port, BracketedWindow selector) -> Contract& {
            return port.contract_input(contract, selector);
          },
          nb::arg("port"),
          nb::arg("selector"),
          nb::rv_policy::reference_internal,
          nb::sig("def input(self, port: holoscan.OperatorPort, selector: "
                  "holoscan.BracketedWindow) -> typing.Self"))
      .def(
          "on_missing",
          [](Contract& contract, const PyNativePort& port, Skip strategy) -> Contract& {
            return contract.on_missing(port.contract_input_ref(), strategy);
          },
          nb::arg("port"),
          nb::arg("strategy"),
          nb::rv_policy::reference_internal,
          nb::sig("def on_missing(self, port: holoscan.OperatorPort, strategy: holoscan.Skip) -> "
                  "typing.Self"))
      .def(
          "on_missing",
          [](Contract& contract, const PyNativePort& port, SafeDefault strategy) -> Contract& {
            return contract.on_missing(port.contract_input_ref(), strategy);
          },
          nb::arg("port"),
          nb::arg("strategy"),
          nb::rv_policy::reference_internal,
          nb::sig("def on_missing(self, port: holoscan.OperatorPort, strategy: "
                  "holoscan.SafeDefault) -> typing.Self"))
      .def(
          "on_missing",
          [](Contract& contract, const PyNativePort& port, ReuseLast strategy) -> Contract& {
            return contract.on_missing(port.contract_input_ref(), strategy);
          },
          nb::arg("port"),
          nb::arg("strategy"),
          nb::rv_policy::reference_internal,
          nb::sig("def on_missing(self, port: holoscan.OperatorPort, strategy: "
                  "holoscan.ReuseLast) -> typing.Self"))
      .def(
          "deadline",
          [](Contract& contract, nb::handle budget) -> Contract& {
            return contract.deadline(python_duration(budget, "budget"));
          },
          nb::arg("budget"),
          nb::rv_policy::reference_internal,
          nb::sig("def deadline(self, budget: datetime.timedelta | holoscan.time.Duration) -> "
                  "typing.Self"))
      .def(
          "complete_after",
          [](Contract& contract, nb::handle limit) -> Contract& {
            return contract.complete_after(
                ComputeCalls{.limit = unsigned_option<std::uint64_t>(limit, "limit")});
          },
          nb::arg("limit"),
          nb::rv_policy::reference_internal,
          nb::sig("def complete_after(self, limit: int) -> typing.Self"))
      .def("on_violation",
           &Contract::on_violation,
           nb::arg("policy"),
           nb::rv_policy::reference_internal,
           nb::sig("def on_violation(self, policy: holoscan.OnViolation) -> typing.Self"));
}

}  // namespace holoscan
