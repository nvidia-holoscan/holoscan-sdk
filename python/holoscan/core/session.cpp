/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "session.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/lifecycle_control.hpp>
#include <holoscan/core/parameter_control.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>

#include "../timeout.hpp"
#include "graph.hpp"
#include "lifecycle_authoring.hpp"
#include "module_name.hpp"
#include "session_pydoc.hpp"
#include "session_reaper.hpp"
#include "time.hpp"

namespace holoscan {
namespace {

using Reaper = python::detail::RunSessionReaper;
constexpr auto kSignalPollInterval = std::chrono::milliseconds{100};
constexpr auto kExitDrainTimeout = std::chrono::seconds{5};

std::atomic<bool>& retirement_drain_pending() {
  static auto* pending = new std::atomic<bool>{false};
  return *pending;
}

int destroy_retired_pending_call(void*) noexcept {
  // Clear first so a concurrent retirement can enqueue another pending call even if it arrives
  // after destroy_retired() swaps the current queue.
  retirement_drain_pending().store(false, std::memory_order_release);
  Reaper::destroy_retired();
  return 0;
}

void destroy_retired_fallback() noexcept {
  if (!nb::is_alive() || Py_IsInitialized() == 0) {
    retirement_drain_pending().store(false, std::memory_order_release);
    return;
  }
  const nb::gil_scoped_acquire gil;
  if (!gil.is_valid()) {
    retirement_drain_pending().store(false, std::memory_order_release);
    return;
  }
  // The scheduled main-thread callback remains the preferred path. If it ran while this thread
  // waited for the GIL, it already cleared the flag and drained the queue. Otherwise this fallback
  // owns the same coalesced drain, including when Py_AddPendingCall rejected the handoff.
  if (retirement_drain_pending().exchange(false, std::memory_order_acq_rel)) {
    Reaper::destroy_retired();
  }
}

void request_interpreter_retirement_drain() noexcept {
  if (Py_IsInitialized() == 0) {
    return;
  }
  bool expected{};
  if (!retirement_drain_pending().compare_exchange_strong(
          expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
    destroy_retired_fallback();
    return;
  }
  static_cast<void>(Py_AddPendingCall(&destroy_retired_pending_call, nullptr));
  destroy_retired_fallback();
}

void drain_session_reaper() noexcept {
  // Close notifier admission while the interpreter is still live. The worker accounts for every
  // callback admitted before this store, so the GIL-released wait below cannot return while a
  // notifier is still trying to acquire the interpreter during finalization.
  Reaper::set_retirement_notifier(nullptr);
  {
    const nb::gil_scoped_release release;
    static_cast<void>(Reaper::wait_for_existing_for(kExitDrainTimeout));
  }
  Reaper::destroy_retired();
}

// Translate Python seconds/None without changing Core's nanosecond timeout contract.
std::optional<std::chrono::nanoseconds> parse_timeout(nb::handle timeout) {
  if (timeout.is_none()) {
    return std::nullopt;
  }
  return python::detail::seconds_to_timeout(nb::cast<double>(timeout));
}

const char* termination_name(RunTermination termination) noexcept {
  switch (termination) {
    case RunTermination::kRunning:
      return "RUNNING";
    case RunTermination::kCompleted:
      return "COMPLETED";
    case RunTermination::kRequestedCompleted:
      return "REQUESTED_COMPLETED";
    case RunTermination::kStopped:
      return "STOPPED";
    case RunTermination::kFailed:
      return "FAILED";
  }
  return "UNKNOWN";
}

[[noreturn]] void throw_control_error(std::string_view token) {
  throw RuntimeError{Error{ErrorCode::kInvalidArgument, std::string{token}}};
}

[[noreturn]] void throw_control_error(ErrorCode code, std::string_view token) {
  throw RuntimeError{Error{code, std::string{token}}};
}

void require_control_owner(pid_t owner_pid) {
  if (owner_pid != ::getpid()) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "PYTHON_RUN_SESSION_FORKED_PROCESS"}};
  }
}

// Cached public Python value classes used to distinguish owning patch encodings.
struct PythonPatchTypes {
  nb::object enum_type;
  nb::object timedelta_type;
};

// This recursive conversion keeps Python type precedence and the public patch vocabulary in one
// place; decomposing it would obscure the depth guard and numeric fallback order.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
ParamPatchValue python_patch_value(nb::handle value, const PythonPatchTypes& types,
                                   std::uint32_t depth = 0U) {
  if (depth >= 32U) {
    throw_control_error("PYTHON_PARAM_PATCH_NESTING_EXCEEDED");
  }
  if (value.is_none()) {
    return ParamPatchValue::none();
  }
  if (nb::isinstance<nb::bool_>(value)) {
    return ParamValue::from(nb::cast<bool>(value));
  }
  if (nb::isinstance(value, types.enum_type)) {
    return python_patch_value(value.attr("name"), types, depth);
  }
  if (nb::isinstance<nb::int_>(value)) {
    const long long signed_value = PyLong_AsLongLong(value.ptr());
    if (PyErr_Occurred() == nullptr) {
      return detail::ParamPatchValueAccess::from_language_numeric(
          ParamValue::from(static_cast<std::int64_t>(signed_value)));
    }
    PyErr_Clear();
    const unsigned long long unsigned_value = PyLong_AsUnsignedLongLong(value.ptr());
    if (PyErr_Occurred() == nullptr) {
      return detail::ParamPatchValueAccess::from_language_numeric(
          ParamValue::from(static_cast<std::uint64_t>(unsigned_value)));
    }
    PyErr_Clear();
    throw_control_error("PYTHON_PARAM_PATCH_INTEGER_OUT_OF_RANGE");
  }
  if (nb::isinstance<nb::float_>(value)) {
    return detail::ParamPatchValueAccess::from_language_numeric(
        ParamValue::from(nb::cast<double>(value)));
  }
  if (nb::isinstance<PyDuration>(value) || nb::isinstance(value, types.timedelta_type)) {
    return ParamValue::from(std::chrono::nanoseconds{python_duration_nanoseconds(value, "value")});
  }
  if (nb::isinstance<nb::str>(value)) {
    auto converted = ParamPatchValue::from_string(nb::cast<std::string>(value));
    if (!converted) {
      throw RuntimeError{converted.error()};
    }
    return std::move(*converted);
  }
  if (nb::isinstance<nb::dict>(value)) {
    std::vector<ParamPatchAssignment> assignments;
    const nb::dict mapping = nb::cast<nb::dict>(value);
    assignments.reserve(mapping.size());
    for (const auto& [key, child] : mapping) {
      if (!nb::isinstance<nb::str>(key)) {
        throw_control_error("PYTHON_PARAM_PATCH_OBJECT_KEY_INVALID");
      }
      assignments.push_back(
          ParamPatchAssignment{.name = nb::cast<std::string>(key),
                               .value = python_patch_value(child, types, depth + 1U)});
    }
    auto converted = ParamPatchValue::from_object(std::move(assignments));
    if (!converted) {
      throw RuntimeError{converted.error()};
    }
    return std::move(*converted);
  }
  if (nb::isinstance<nb::list>(value) || nb::isinstance<nb::tuple>(value)) {
    std::vector<ParamPatchValue> elements;
    const nb::iterable sequence = nb::cast<nb::iterable>(value);
    for (const nb::handle child : sequence) {
      elements.push_back(python_patch_value(child, types, depth + 1U));
    }
    auto converted = ParamPatchValue::from_sequence(std::move(elements));
    if (!converted) {
      throw RuntimeError{converted.error()};
    }
    return std::move(*converted);
  }
  throw_control_error("PYTHON_PARAM_PATCH_VALUE_UNSUPPORTED");
}

nb::object python_scalar_value(const ParamValue& value) {
  switch (value.kind()) {
    case ParamScalarKind::kBool:
      return nb::cast(value.as<bool>().value());
    case ParamScalarKind::kSignedInteger:
      switch (value.bit_width()) {
        case 8U:
          return nb::cast(value.as<std::int8_t>().value());
        case 16U:
          return nb::cast(value.as<std::int16_t>().value());
        case 32U:
          return nb::cast(value.as<std::int32_t>().value());
        case 64U:
          return nb::cast(value.as<std::int64_t>().value());
        default:
          break;
      }
      break;
    case ParamScalarKind::kUnsignedInteger:
      switch (value.bit_width()) {
        case 8U:
          return nb::cast(value.as<std::uint8_t>().value());
        case 16U:
          return nb::cast(value.as<std::uint16_t>().value());
        case 32U:
          return nb::cast(value.as<std::uint32_t>().value());
        case 64U:
          return nb::cast(value.as<std::uint64_t>().value());
        default:
          break;
      }
      break;
    case ParamScalarKind::kFloatingPoint:
      if (value.bit_width() == 32U) {
        return nb::cast(value.as<float>().value());
      }
      if (value.bit_width() == 64U) {
        return nb::cast(value.as<double>().value());
      }
      break;
    case ParamScalarKind::kDurationNanoseconds:
      return nb::cast(PyDuration{value.as<std::chrono::nanoseconds>().value().count()});
    case ParamScalarKind::kNone:
      break;
  }
  throw_control_error(ErrorCode::kNotSupported, "PYTHON_PARAM_READ_VALUE_UNSUPPORTED");
}

// Couples one resolved runtime address with the schema field used for value conversion.
struct ResolvedParamPath {
  ParamAddress address{};
  const ParamFieldCatalogRecord* field{};
};

ResolvedParamPath resolve_param_path(const ParameterCatalogView& catalog, std::string_view path) {
  std::optional<ResolvedParamPath> resolved;
  for (std::size_t split = path.find('/'); split != std::string_view::npos;
       split = path.find('/', split + 1U)) {
    auto operator_path = OperatorPath::parse(path.substr(0U, split));
    auto param_path = ParamPath::parse(path.substr(split + 1U));
    if (!operator_path || !param_path) {
      continue;
    }
    auto address = catalog.resolve(*operator_path, *param_path);
    if (!address) {
      continue;
    }
    auto field = catalog.field(*address);
    if (!field) {
      continue;
    }
    if (resolved) {
      throw_control_error("PYTHON_PARAM_PATH_AMBIGUOUS");
    }
    resolved = ResolvedParamPath{.address = *address, .field = field->get()};
  }
  if (!resolved) {
    throw RuntimeError{Error{ErrorCode::kNotFound, "PYTHON_PARAM_PATH_NOT_FOUND"}};
  }
  return *resolved;
}

// Owns one path/value update request until it can be converted against an admitted schema.
class PyParamPatch {
 public:
  PyParamPatch(std::string path, nb::object value,
               std::optional<ParamGeneration> expected_generation)
      : path_(std::move(path)),
        value_(std::move(value)),
        expected_generation_(expected_generation) {}

  [[nodiscard]] const std::string& path() const noexcept { return path_; }
  [[nodiscard]] const nb::object& value() const noexcept { return value_; }
  [[nodiscard]] std::optional<ParamGeneration> expected_generation() const noexcept {
    return expected_generation_;
  }

 private:
  std::string path_;
  nb::object value_;
  std::optional<ParamGeneration> expected_generation_;
};

// Owns the scalar projection and generation metadata returned by one parameter read.
class PyParamReadResult {
 public:
  explicit PyParamReadResult(ParamReadResult result)
      : result_(result), value_(python_scalar_value(result.value)) {}

  [[nodiscard]] const nb::object& value() const noexcept { return value_; }
  [[nodiscard]] ParamGeneration generation() const noexcept { return result_.generation; }
  [[nodiscard]] ParamUpdateSequence update_sequence() const noexcept {
    return result_.update_sequence;
  }
  [[nodiscard]] ParamUpdateState state() const noexcept { return result_.state; }

 private:
  ParamReadResult result_;
  nb::object value_;
};

// Process-bound nonblocking cursor over attributed parameter-update events.
class PyParamEventCursor {
 public:
  PyParamEventCursor(ParamEventCursor cursor, pid_t owner_pid)
      : cursor_(std::make_unique<ParamEventCursor>(std::move(cursor))), owner_pid_(owner_pid) {}
  PyParamEventCursor(const PyParamEventCursor&) = delete;
  PyParamEventCursor& operator=(const PyParamEventCursor&) = delete;
  PyParamEventCursor(PyParamEventCursor&&) noexcept = default;
  PyParamEventCursor& operator=(PyParamEventCursor&&) noexcept = default;

  ~PyParamEventCursor() noexcept {
    if (owner_pid_ != ::getpid()) {
      // NOLINTNEXTLINE(bugprone-unused-return-value)
      static_cast<void>(cursor_.release());
    }
  }

  [[nodiscard]] std::vector<ParamEvent> drain(std::size_t maximum) {
    require_control_owner(owner_pid_);
    std::vector<ParamEvent> events(maximum);
    expected<std::size_t, Error> count = make_unexpected(Error{ErrorCode::kFailure});
    {
      const nb::gil_scoped_release release;
      count = cursor_->drain(events);
    }
    if (!count) {
      throw RuntimeError{count.error()};
    }
    events.resize(*count);
    return events;
  }

 private:
  std::unique_ptr<ParamEventCursor> cursor_;
  pid_t owner_pid_{};
};

// Process-bound facade over an admitted run-scoped parameter-control capability.
class PyParamControl {
 public:
  PyParamControl(ParamControlHandle handle, pid_t owner_pid)
      : handle_(std::move(handle)), owner_pid_(owner_pid) {}

  [[nodiscard]] PyParamReadResult read(std::string_view path, ParamReadView view) const {
    require_control_owner(owner_pid_);
    const ResolvedParamPath resolved = resolve_param_path(handle_.catalog(), path);
    expected<ParamReadResult, Error> result = make_unexpected(Error{ErrorCode::kFailure});
    {
      const nb::gil_scoped_release release;
      result = handle_.read(resolved.address, view);
    }
    if (!result) {
      throw RuntimeError{result.error()};
    }
    return PyParamReadResult{*result};
  }

  [[nodiscard]] ParamBatchUpdateResult submit(nb::handle patches) {
    require_control_owner(owner_pid_);
    std::vector<PyParamPatch> requested;
    if (nb::isinstance<PyParamPatch>(patches)) {
      requested.push_back(nb::cast<const PyParamPatch&>(patches));
    } else {
      if (nb::isinstance<nb::str>(patches) || nb::isinstance<nb::bytes>(patches)) {
        throw_control_error("PYTHON_PARAM_PATCH_ITERABLE_REQUIRED");
      }
      for (const nb::handle patch : nb::cast<nb::iterable>(patches)) {
        if (!nb::isinstance<PyParamPatch>(patch)) {
          throw_control_error("PYTHON_PARAM_PATCH_REQUIRED");
        }
        requested.push_back(nb::cast<const PyParamPatch&>(patch));
      }
    }
    std::vector<ParamPatch> converted;
    converted.reserve(requested.size());
    const PythonPatchTypes types{
        .enum_type = nb::module_::import_("enum").attr("Enum"),
        .timedelta_type = nb::module_::import_("datetime").attr("timedelta")};
    for (const PyParamPatch& patch : requested) {
      const ResolvedParamPath resolved = resolve_param_path(handle_.catalog(), patch.path());
      converted.push_back(ParamPatch{.target = resolved.address,
                                     .value = python_patch_value(patch.value(), types),
                                     .expected_generation = patch.expected_generation()});
    }
    const nb::gil_scoped_release release;
    return handle_.submit(converted);
  }

  [[nodiscard]] ParamUpdateResult await_applied(
      ParamUpdateSequence sequence, std::optional<std::chrono::nanoseconds> timeout) const {
    require_control_owner(owner_pid_);
    if (timeout && *timeout == std::chrono::nanoseconds::zero()) {
      expected<ParamUpdateResult, Error> result = make_unexpected(Error{ErrorCode::kTimeout});
      {
        const nb::gil_scoped_release release;
        result = handle_.await(sequence, *timeout);
      }
      if (!result) {
        throw RuntimeError{result.error()};
      }
      return *result;
    }
    const auto started = std::chrono::steady_clock::now();
    for (;;) {
      std::chrono::nanoseconds slice = kSignalPollInterval;
      if (timeout) {
        const auto elapsed = std::chrono::steady_clock::now() - started;
        if (elapsed >= *timeout) {
          throw RuntimeError{Error{ErrorCode::kTimeout}};
        }
        slice = std::min(slice, *timeout - elapsed);
      }
      expected<ParamUpdateResult, Error> result = make_unexpected(Error{ErrorCode::kTimeout});
      {
        const nb::gil_scoped_release release;
        result = handle_.await(sequence, slice);
      }
      if (result) {
        return *result;
      }
      if (result.error().code != ErrorCode::kTimeout) {
        throw RuntimeError{result.error()};
      }
      if (PyErr_CheckSignals() != 0) {
        throw nb::python_error();
      }
    }
  }

  [[nodiscard]] PyParamEventCursor events() const {
    require_control_owner(owner_pid_);
    expected<ParamEventCursor, Error> cursor = make_unexpected(Error{ErrorCode::kFailure});
    {
      const nb::gil_scoped_release release;
      cursor = handle_.open_event_cursor();
    }
    if (!cursor) {
      throw RuntimeError{cursor.error()};
    }
    return PyParamEventCursor{std::move(*cursor), owner_pid_};
  }

 private:
  ParamControlHandle handle_;
  pid_t owner_pid_{};
};

std::string lifecycle_operator_path(nb::handle value) {
  return python_operator_path(value, "LIFECYCLE_CONTROL_OPERATOR_REQUIRED");
}

// Owns Python-authored restart targets and source-stage restrictions until control minting.
class PyLifecycleAccessPolicy {
 public:
  PyLifecycleAccessPolicy(const nb::iterable& targets, const nb::iterable& from_stages) {
    for (const nb::handle target : targets) {
      targets_.push_back(lifecycle_operator_path(target));
    }
    for (const nb::handle stage : from_stages) {
      stages_.push_back(nb::cast<LifecycleStage>(stage));
    }
  }

  [[nodiscard]] const std::vector<std::string>& targets() const noexcept { return targets_; }
  [[nodiscard]] const std::vector<LifecycleStage>& stages() const noexcept { return stages_; }

 private:
  std::vector<std::string> targets_;
  std::vector<LifecycleStage> stages_;
};

// Process-bound facade over an admitted run-scoped lifecycle-control capability.
class PyLifecycleControl {
 public:
  struct Target {
    std::string path;
    OperatorRef run{};
  };

  PyLifecycleControl(LifecycleControlHandle handle, std::vector<Target> targets, pid_t owner_pid)
      : handle_(std::move(handle)), targets_(std::move(targets)), owner_pid_(owner_pid) {}

  [[nodiscard]] LifecycleRestartReceipt request_restart(nb::handle target,
                                                        LifecycleRestartOptions options) const {
    require_control_owner(owner_pid_);
    const bool run_ref = nb::isinstance<PyLifecycleOperatorRef>(target);
    const OperatorRef requested =
        run_ref ? nb::cast<const PyLifecycleOperatorRef&>(target).value() : OperatorRef{};
    const std::string path = run_ref ? std::string{} : lifecycle_operator_path(target);
    const auto found = std::find_if(targets_.begin(), targets_.end(), [&](const auto& item) {
      return run_ref ? item.run == requested : item.path == path;
    });
    if (found == targets_.end()) {
      throw RuntimeError{Error{ErrorCode::kLifecycleRestartUnauthorized}};
    }
    expected<LifecycleRestartReceipt, Error> result = make_unexpected(Error{ErrorCode::kFailure});
    {
      const nb::gil_scoped_release release;
      result = handle_.request_restart(found->run, options);
    }
    if (!result) {
      throw RuntimeError{result.error()};
    }
    return *result;
  }

  [[nodiscard]] LifecycleRestartReceiptView inspect(LifecycleRestartReceipt receipt) const {
    require_control_owner(owner_pid_);
    expected<LifecycleRestartReceiptView, Error> result =
        make_unexpected(Error{ErrorCode::kFailure});
    {
      const nb::gil_scoped_release release;
      result = handle_.inspect(receipt);
    }
    if (!result) {
      throw RuntimeError{result.error()};
    }
    return *result;
  }

  [[nodiscard]] LifecycleRestartReceiptView await_terminal(
      LifecycleRestartReceipt receipt, std::optional<std::chrono::nanoseconds> timeout) const {
    require_control_owner(owner_pid_);
    if (timeout && *timeout == std::chrono::nanoseconds::zero()) {
      expected<LifecycleRestartReceiptView, Error> result =
          make_unexpected(Error{ErrorCode::kTimeout});
      {
        const nb::gil_scoped_release release;
        result = handle_.await(receipt, *timeout);
      }
      if (!result) {
        throw RuntimeError{result.error()};
      }
      return *result;
    }
    const auto started = std::chrono::steady_clock::now();
    for (;;) {
      std::chrono::nanoseconds slice = kSignalPollInterval;
      if (timeout) {
        const auto elapsed = std::chrono::steady_clock::now() - started;
        if (elapsed >= *timeout) {
          throw RuntimeError{Error{ErrorCode::kTimeout}};
        }
        slice = std::min(slice, *timeout - elapsed);
      }
      expected<LifecycleRestartReceiptView, Error> result =
          make_unexpected(Error{ErrorCode::kTimeout});
      {
        const nb::gil_scoped_release release;
        result = handle_.await(receipt, slice);
      }
      if (result) {
        return *result;
      }
      if (result.error().code != ErrorCode::kTimeout) {
        throw RuntimeError{result.error()};
      }
      if (PyErr_CheckSignals() != 0) {
        throw nb::python_error();
      }
    }
  }

 private:
  LifecycleControlHandle handle_;
  std::vector<Target> targets_;
  pid_t owner_pid_{};
};

// Retains both the native session and its plan while exposing bounded waits and control handles;
// representative: test_failed_session_exposes_structured_diagnostics_and_bounded_summary.
class PyRunSession {
 public:
  explicit PyRunSession(RunSession&& session, ExecutionPlan plan)
      : node_{std::make_unique<Reaper::Node>(std::move(session))},
        plan_{std::move(plan)},
        owner_pid_{::getpid()} {}

  PyRunSession(PyRunSession&&) noexcept = default;
  PyRunSession(const PyRunSession&) = delete;
  PyRunSession& operator=(const PyRunSession&) = delete;
  PyRunSession& operator=(PyRunSession&&) = delete;

  ~PyRunSession() noexcept {
    if (!node_) {
      return;
    }
    if (!in_owner_process()) {
      // A fork child has no copy of the Runtime worker thread. Let the operating system reclaim
      // the inherited process-private state instead of attempting an impossible join.
      // NOLINTNEXTLINE(bugprone-unused-return-value)
      static_cast<void>(node_.release());
      return;
    }
    if (!node_->session.wait_admission()) {
      Reaper::instance().enqueue(std::move(node_));
      return;
    }

    node_->session.request_stop();
    try {
      const nb::gil_scoped_release release;
      node_->session.wait();
    } catch (...) {
      Reaper::instance().enqueue(std::move(node_));
      return;
    }
    node_.reset();
  }

  [[nodiscard]] bool is_running() const {
    require_owner_process();
    return session().is_running();
  }

  [[nodiscard]] RunTermination termination() const {
    require_owner_process();
    return session().termination();
  }

  void request_stop() {
    require_owner_process();
    session().request_stop();
  }

  [[nodiscard]] bool wait(std::optional<std::chrono::nanoseconds> timeout) {
    require_wait_admission();
    if (timeout && *timeout <= std::chrono::nanoseconds::zero()) {
      const nb::gil_scoped_release release;
      return session().wait_for(std::chrono::nanoseconds::zero());
    }

    const auto started = std::chrono::steady_clock::now();
    for (;;) {
      std::chrono::nanoseconds slice = kSignalPollInterval;
      if (timeout) {
        const auto elapsed = std::chrono::steady_clock::now() - started;
        if (elapsed >= *timeout) {
          return false;
        }
        slice = std::min(slice, *timeout - elapsed);
      }
      bool terminal{};
      {
        const nb::gil_scoped_release release;
        terminal = session().wait_for(slice);
      }
      if (terminal) {
        break;
      }
      if (PyErr_CheckSignals() != 0) {
        throw nb::python_error();
      }
    }

    if (!timeout) {
      const nb::gil_scoped_release release;
      session().wait();
    }
    return true;
  }

  void close() {
    require_wait_admission();
    session().request_stop();
    const nb::gil_scoped_release release;
    session().wait();
  }

  [[nodiscard]] std::vector<Diagnostic> diagnostics() const {
    require_owner_process();
    const auto values = session().diagnostics();
    return {values.begin(), values.end()};
  }

  [[nodiscard]] PyParamControl parameter_control(const ParamAccessPolicy& policy) {
    require_owner_process();
    expected<ParamControlHandle, Error> handle = make_unexpected(Error{ErrorCode::kFailure});
    {
      const nb::gil_scoped_release release;
      handle = session().parameter_control(policy);
    }
    if (!handle) {
      throw RuntimeError{handle.error()};
    }
    return PyParamControl{std::move(*handle), owner_pid_};
  }

  [[nodiscard]] PyLifecycleControl lifecycle_control(const PyLifecycleAccessPolicy& policy) {
    require_owner_process();
    std::vector<OperatorRef> run_targets;
    std::vector<PyLifecycleControl::Target> targets;
    run_targets.reserve(policy.targets().size());
    targets.reserve(policy.targets().size());
    for (const std::string& path : policy.targets()) {
      const auto records = plan_.operator_parameters();
      const auto record = std::find_if(records.begin(), records.end(), [&](const auto& item) {
        return item.operator_path == path;
      });
      if (record == records.end()) {
        throw RuntimeError{Error{ErrorCode::kNotFound, "LIFECYCLE_OPERATOR_PATH_NOT_FOUND"}};
      }
      expected<OperatorRef, Error> run = make_unexpected(Error{ErrorCode::kFailure});
      {
        const nb::gil_scoped_release release;
        run = session().operator_ref(record->owner);
      }
      if (!run) {
        throw RuntimeError{run.error()};
      }
      run_targets.push_back(*run);
      targets.push_back(PyLifecycleControl::Target{.path = path, .run = *run});
    }
    expected<LifecycleControlHandle, Error> handle = make_unexpected(Error{ErrorCode::kFailure});
    {
      const nb::gil_scoped_release release;
      handle = session().lifecycle_control(LifecycleAccessPolicy{
          .allowed_targets = run_targets, .allowed_from_stages = policy.stages()});
    }
    if (!handle) {
      throw RuntimeError{handle.error()};
    }
    return PyLifecycleControl{std::move(*handle), std::move(targets), owner_pid_};
  }

  [[nodiscard]] std::string summary() const {
    require_owner_process();
    return format_diagnostics(session().diagnostics(), plan_.graph());
  }

  [[nodiscard]] std::string repr() const {
    std::ostringstream output;
    output << "<holoscan.RunSession " << termination_name(termination()) << '>';
    return output.str();
  }

 private:
  [[nodiscard]] RunSession& session() noexcept { return node_->session; }
  [[nodiscard]] const RunSession& session() const noexcept { return node_->session; }

  [[nodiscard]] bool in_owner_process() const noexcept { return owner_pid_ == ::getpid(); }

  void require_owner_process() const {
    if (!in_owner_process()) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "PYTHON_RUN_SESSION_FORKED_PROCESS"}};
    }
  }

  void require_wait_admission() const {
    require_owner_process();
    auto admitted = session().wait_admission();
    if (!admitted) {
      throw RuntimeError{admitted.error()};
    }
  }

  std::unique_ptr<Reaper::Node> node_;
  ExecutionPlan plan_;
  pid_t owner_pid_{};
};

PyRunSession start_session(const ExecutionPlan& plan) {
  // Successful fallback joins are handed back to an interpreter-owned thread so hosted Python
  // state is reclaimed during normal execution rather than retained until the atexit drain.
  Reaper::destroy_retired();
  const nb::gil_scoped_release release;
  return PyRunSession{holoscan::run_async(plan), plan};
}

PyRunSession run_session(const ExecutionPlan& plan) {
  PyRunSession session = start_session(plan);
  (void)session.wait(std::nullopt);
  return session;
}

}  // namespace

void init_session(nb::module_& module) {
  static const bool exit_hook_registered = [] {
    Reaper::set_retirement_notifier(&request_interpreter_retirement_drain);
    static_cast<void>(nb::module_::import_("atexit").attr("register")(
        nb::cpp_function([] { drain_session_reaper(); })));
    return true;
  }();
  static_cast<void>(exit_hook_registered);

  python::bind_enum_in_module<RunTermination>(module, "holoscan", "RunTermination")
      .value("RUNNING", RunTermination::kRunning)
      .value("COMPLETED", RunTermination::kCompleted)
      .value("REQUESTED_COMPLETED", RunTermination::kRequestedCompleted)
      .value("STOPPED", RunTermination::kStopped)
      .value("FAILED", RunTermination::kFailed);

  nb::enum_<ParamPermission>(module, "ParamPermission", nb::is_arithmetic(), nb::is_flag())
      .value("NONE", ParamPermission::kNone)
      .value("READ", ParamPermission::kRead)
      .value("UPDATE", ParamPermission::kUpdate)
      .value("AWAIT", ParamPermission::kAwait)
      .value("CAPTURE", ParamPermission::kCapture)
      .value("OBSERVE", ParamPermission::kObserve);
  nb::enum_<ParamReadView>(module, "ParamReadView")
      .value("PUBLISHED", ParamReadView::kPublished)
      .value("LATEST_ACCEPTED", ParamReadView::kLatestAccepted)
      .value("PENDING", ParamReadView::kPending);
  nb::enum_<ParamUpdateState>(module, "ParamUpdateState")
      .value("SUBMITTED", ParamUpdateState::kSubmitted)
      .value("REJECTED", ParamUpdateState::kRejected)
      .value("ACCEPTED", ParamUpdateState::kAccepted)
      .value("QUIESCING", ParamUpdateState::kQuiescing)
      .value("RESTARTING", ParamUpdateState::kRestarting)
      .value("ACCEPTED_NO_CHANGE", ParamUpdateState::kAcceptedNoChange)
      .value("PUBLISHED", ParamUpdateState::kPublished)
      .value("OBSERVED", ParamUpdateState::kObserved)
      .value("SUPERSEDED", ParamUpdateState::kSuperseded)
      .value("CANCELED", ParamUpdateState::kCanceled)
      .value("ROLLED_BACK", ParamUpdateState::kRolledBack)
      .value("FAILED_STOPPED", ParamUpdateState::kFailedStopped);
  nb::enum_<ParamUpdateStatus>(module, "ParamUpdateStatus")
      .value("ACCEPTED", ParamUpdateStatus::kAccepted)
      .value("ACCEPTED_NO_CHANGE", ParamUpdateStatus::kAcceptedNoChange)
      .value("SUBMITTED", ParamUpdateStatus::kSubmitted)
      .value("REJECTED", ParamUpdateStatus::kRejected)
      .value("DEFERRED", ParamUpdateStatus::kDeferred)
      .value("UPDATES_DISABLED", ParamUpdateStatus::kUpdatesDisabled)
      .value("OUT_OF_BOUNDS", ParamUpdateStatus::kOutOfBounds)
      .value("VALIDATION_FAILED", ParamUpdateStatus::kValidationFailed)
      .value("TYPE_MISMATCH", ParamUpdateStatus::kTypeMismatch)
      .value("UNKNOWN_PARAMETER", ParamUpdateStatus::kUnknownParameter)
      .value("STALE_OPERATOR_REF", ParamUpdateStatus::kStaleOperatorRef)
      .value("EXPECTED_GENERATION_MISMATCH", ParamUpdateStatus::kExpectedGenerationMismatch)
      .value("REQUIRES_LIFECYCLE_BARRIER", ParamUpdateStatus::kRequiresLifecycleBarrier)
      .value("REQUIRES_RECOMPILE", ParamUpdateStatus::kRequiresRecompile)
      .value("UNSUPPORTED_ATOMIC_SCOPE", ParamUpdateStatus::kUnsupportedAtomicScope)
      .value("PAST_EFFECTIVE_TIME", ParamUpdateStatus::kPastEffectiveTime)
      .value("UNSUPPORTED_CLOCK", ParamUpdateStatus::kUnsupportedClock)
      .value("QUEUE_FULL", ParamUpdateStatus::kQueueFull)
      .value("INVALID_CALLING_CONTEXT", ParamUpdateStatus::kInvalidCallingContext)
      .value("UNSUPPORTED_IN_PROFILE", ParamUpdateStatus::kUnsupportedInProfile)
      .value("CLOSED", ParamUpdateStatus::kClosed);
  nb::enum_<ParamEventKind>(module, "ParamEventKind")
      .value("TRANSITION", ParamEventKind::kTransition)
      .value("GAP", ParamEventKind::kGap);
  nb::enum_<LifecycleTransactionState>(module, "LifecycleTransactionState")
      .value("REJECTED", LifecycleTransactionState::kRejected)
      .value("ACCEPTED_NO_CHANGE", LifecycleTransactionState::kAcceptedNoChange)
      .value("QUEUED", LifecycleTransactionState::kQueued)
      .value("QUIESCING", LifecycleTransactionState::kQuiescing)
      .value("RESTARTING", LifecycleTransactionState::kRestarting)
      .value("PUBLISHED", LifecycleTransactionState::kPublished)
      .value("SUPERSEDED", LifecycleTransactionState::kSuperseded)
      .value("CANCELED", LifecycleTransactionState::kCanceled)
      .value("ROLLED_BACK", LifecycleTransactionState::kRolledBack)
      .value("FAILED_STOPPED", LifecycleTransactionState::kFailedStopped);

  python::bind_class_in_module<ParamAccessPolicy>(module, "holoscan.params", "ParamAccessPolicy")
      .def(
          "__init__",
          [](ParamAccessPolicy* self,
             ParamPermission permissions,
             bool include_internal,
             bool include_sensitive,
             std::uint32_t max_event_cursors) {
            new (self) ParamAccessPolicy{.origin = ParamControlOrigin::kLocalApplication,
                                         .permissions = permissions,
                                         .allowed_scope = ParamScopeFilter::run(),
                                         .include_internal = include_internal,
                                         .include_sensitive = include_sensitive,
                                         .max_event_cursors = max_event_cursors};
          },
          nb::kw_only(),
          nb::arg("permissions") = ParamPermission::kNone,
          nb::arg("include_internal") = false,
          nb::arg("include_sensitive") = false,
          nb::arg("max_event_cursors") = 0U)
      .def_rw("permissions", &ParamAccessPolicy::permissions)
      .def_rw("include_internal", &ParamAccessPolicy::include_internal)
      .def_rw("include_sensitive", &ParamAccessPolicy::include_sensitive)
      .def_rw("max_event_cursors", &ParamAccessPolicy::max_event_cursors);
  python::bind_class_in_module<PyParamPatch>(module, "holoscan.params", "ParamPatch")
      .def(nb::init<std::string, nb::object, std::optional<ParamGeneration>>(),
           nb::arg("path"),
           nb::arg("value").none(),
           nb::kw_only(),
           nb::arg("expected_generation").none() = nb::none())
      .def_prop_ro("path", &PyParamPatch::path)
      .def_prop_ro("value", &PyParamPatch::value)
      .def_prop_ro("expected_generation", &PyParamPatch::expected_generation);
  python::bind_class_in_module<PyParamReadResult>(module, "holoscan.params", "ParamReadResult")
      .def_prop_ro("value", &PyParamReadResult::value)
      .def_prop_ro("generation", &PyParamReadResult::generation)
      .def_prop_ro("update_sequence", &PyParamReadResult::update_sequence)
      .def_prop_ro("state", &PyParamReadResult::state);
  python::bind_class_in_module<ParamBatchUpdateResult>(
      module, "holoscan.params", "ParamBatchUpdateResult")
      .def_ro("status", &ParamBatchUpdateResult::status)
      .def_ro("committed", &ParamBatchUpdateResult::committed)
      .def_ro("patch_count", &ParamBatchUpdateResult::patch_count)
      .def_prop_ro("failed_patch",
                   [](const ParamBatchUpdateResult& result) -> nb::object {
                     return result.failed_patch == ParamBatchUpdateResult::kNoFailedPatch
                                ? nb::none()
                                : nb::cast(result.failed_patch);
                   })
      .def_ro("update_sequence", &ParamBatchUpdateResult::update_sequence)
      .def_ro("target_generation", &ParamBatchUpdateResult::target_generation);
  python::bind_class_in_module<ParamUpdateResult>(module, "holoscan.params", "ParamUpdateResult")
      .def_ro("status", &ParamUpdateResult::status)
      .def_ro("state", &ParamUpdateResult::state)
      .def_ro("update_sequence", &ParamUpdateResult::update_sequence)
      .def_ro("target_generation", &ParamUpdateResult::target_generation)
      .def_ro("lifecycle_transaction_id", &ParamUpdateResult::lifecycle_transaction_id)
      .def_ro("coalesced_into_transaction", &ParamUpdateResult::coalesced_into_transaction);
  python::bind_class_in_module<ParamEvent>(module, "holoscan.params", "ParamEvent")
      .def_ro("kind", &ParamEvent::kind)
      .def_ro("event_sequence", &ParamEvent::event_sequence)
      .def_ro("first_missing_sequence", &ParamEvent::first_missing_sequence)
      .def_ro("last_missing_sequence", &ParamEvent::last_missing_sequence)
      .def_prop_ro(
          "operator_instance",
          [](const ParamEvent& event) { return PyLifecycleOperatorRef{event.operator_instance}; })
      .def_ro("field", &ParamEvent::field)
      .def_ro("update_sequence", &ParamEvent::update_sequence)
      .def_ro("prior_generation", &ParamEvent::prior_generation)
      .def_ro("target_generation", &ParamEvent::target_generation)
      .def_ro("state", &ParamEvent::state)
      .def_ro("status", &ParamEvent::status)
      .def_ro("actor_id", &ParamEvent::actor_id)
      .def_ro("reason_id", &ParamEvent::reason_id);
  python::bind_class_in_module<PyParamEventCursor>(module, "holoscan.params", "ParamEventCursor")
      .def("drain", &PyParamEventCursor::drain, nb::arg("maximum") = 64U);
  python::bind_class_in_module<PyParamControl>(module, "holoscan.params", "ParamControlHandle")
      .def("read",
           &PyParamControl::read,
           nb::arg("path"),
           nb::arg("view") = ParamReadView::kPublished)
      .def("submit", &PyParamControl::submit, nb::arg("patches"))
      .def(
          "await_applied",
          [](const PyParamControl& control, ParamUpdateSequence sequence, nb::handle timeout) {
            return control.await_applied(sequence, parse_timeout(timeout));
          },
          nb::arg("update_sequence"),
          nb::arg("timeout").none() = nb::none())
      .def("events", &PyParamControl::events);

  python::bind_class_in_module<PyLifecycleAccessPolicy>(
      module, "holoscan.lifecycle", "LifecycleAccessPolicy")
      .def(nb::init<nb::iterable, nb::iterable>(), nb::arg("targets"), nb::arg("from_stages"));
  python::bind_class_in_module<LifecycleRestartOptions>(
      module, "holoscan.lifecycle", "LifecycleRestartOptions")
      .def(
          "__init__",
          [](LifecycleRestartOptions* self, LifecycleStage from_stage, ReasonId reason_id) {
            new (self) LifecycleRestartOptions{.from_stage = from_stage,
                                               .reason = LifecycleRestartReason{reason_id}};
          },
          nb::kw_only(),
          nb::arg("from_stage") = LifecycleStage::kConfigure,
          nb::arg("reason_id") = 0U)
      .def_rw("from_stage", &LifecycleRestartOptions::from_stage)
      .def_prop_rw(
          "reason_id",
          [](const LifecycleRestartOptions& options) { return options.reason.stable_id; },
          [](LifecycleRestartOptions& options, ReasonId value) {
            options.reason.stable_id = value;
          });
  python::bind_class_in_module<LifecycleRestartReceipt>(
      module, "holoscan.lifecycle", "LifecycleRestartReceipt")
      .def_ro("receipt_id", &LifecycleRestartReceipt::receipt_id)
      .def_ro("acceptance_sequence", &LifecycleRestartReceipt::acceptance_sequence)
      .def_prop_ro("target",
                   [](const LifecycleRestartReceipt& receipt) {
                     return PyLifecycleOperatorRef{receipt.target};
                   })
      .def_ro("requested_stage", &LifecycleRestartReceipt::requested_stage);
  python::bind_class_in_module<LifecycleRestartReceiptView>(
      module, "holoscan.lifecycle", "LifecycleRestartReceiptView")
      .def_ro("receipt", &LifecycleRestartReceiptView::receipt)
      .def_ro("state", &LifecycleRestartReceiptView::state)
      .def_ro("transaction_id", &LifecycleRestartReceiptView::transaction_id)
      .def_ro("coalesced_into_transaction",
              &LifecycleRestartReceiptView::coalesced_into_transaction)
      .def_ro("related_transaction_id", &LifecycleRestartReceiptView::related_transaction_id)
      .def_ro("fence_id", &LifecycleRestartReceiptView::fence_id)
      .def_ro("effective_stage", &LifecycleRestartReceiptView::effective_stage)
      .def_ro("candidate_generation", &LifecycleRestartReceiptView::candidate_generation);
  python::bind_class_in_module<PyLifecycleControl>(
      module, "holoscan.lifecycle", "LifecycleControlHandle")
      .def("request_restart",
           &PyLifecycleControl::request_restart,
           nb::arg("target"),
           nb::arg("options") = LifecycleRestartOptions{})
      .def("inspect", &PyLifecycleControl::inspect, nb::arg("receipt"))
      .def(
          "await_terminal",
          [](const PyLifecycleControl& control,
             LifecycleRestartReceipt receipt,
             nb::handle timeout) {
            return control.await_terminal(receipt, parse_timeout(timeout));
          },
          nb::arg("receipt"),
          nb::arg("timeout").none() = nb::none());

  python::bind_class_in_module<PyRunSession>(
      module, "holoscan", "RunSession", doc::session::kRunSession)
      .def_prop_ro("is_running", &PyRunSession::is_running)
      .def_prop_ro("termination", &PyRunSession::termination)
      .def_prop_ro("diagnostics", &PyRunSession::diagnostics)
      .def("summary", &PyRunSession::summary, doc::session::kSummary)
      .def("request_stop", &PyRunSession::request_stop)
      .def("parameter_control", &PyRunSession::parameter_control, nb::arg("policy"))
      .def("lifecycle_control", &PyRunSession::lifecycle_control, nb::arg("policy"))
      .def(
          "wait",
          [](PyRunSession& session, nb::handle timeout) {
            return session.wait(parse_timeout(timeout));
          },
          nb::arg("timeout").none() = nb::none(),
          doc::session::kWait,
          nb::sig("def wait(self, timeout: float | None = None) -> bool"))
      .def(
          "__enter__",
          [](PyRunSession& session) -> PyRunSession& { return session; },
          nb::rv_policy::reference_internal,
          nb::sig("def __enter__(self) -> typing.Self"))
      .def(
          "__exit__",
          [](PyRunSession& session, const nb::object&, const nb::object&, const nb::object&) {
            session.close();
            return false;
          },
          nb::arg("exc_type").none(),
          nb::arg("exc_value").none(),
          nb::arg("traceback").none(),
          nb::sig(
              "def __exit__(self, exc_type: type[BaseException] | None, "
              "exc_value: BaseException | None, traceback: types.TracebackType | None) -> bool"))
      .def("__repr__", &PyRunSession::repr);

  python::bind_functions_in_module(module, "holoscan", [&] {
    module.def("run", &run_session, nb::arg("plan"), doc::session::kRun);
    module.def("run_async", &start_session, nb::arg("plan"), doc::session::kRunAsync);
  });
  module.attr("RunTermination").attr("__module__") = "holoscan";
  for (const char* name : {"ParamPermission",
                           "ParamReadView",
                           "ParamUpdateState",
                           "ParamUpdateStatus",
                           "ParamEventKind"}) {
    module.attr(name).attr("__module__") = "holoscan.params";
  }
  module.attr("LifecycleTransactionState").attr("__module__") = "holoscan.lifecycle";
}

}  // namespace holoscan
