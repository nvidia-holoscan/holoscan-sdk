// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <unistd.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "../core/module_name.hpp"
#include "adapters.hpp"
#include "binding_services_compatibility.hpp"
#include "registered_adapters.hpp"

namespace holoscan::testing::python {
namespace {

template <typename T>
void require_core_type(const nb::module_& core, const char* name) {
  const auto type = nb::type<T>();
  if (!type.is_valid() || !nb::hasattr(core, name) || type.ptr() != core.attr(name).ptr()) {
    throw nb::import_error("holoscan.testing requires matching canonical Core type registration");
  }
}

struct Capture {
  std::shared_ptr<const PayloadAdapter> adapter;
  CapturePolicy policy;
  std::size_t capacity;
  std::size_t maximum_bytes;
  bool explicit_policy{};
};
using Captures = std::map<std::string, Capture, std::less<>>;

class Result {
 public:
  Result(TestResult result, Captures captures, std::shared_ptr<const BindingOwner> owner)
      : owner_(std::move(owner)), captures_(std::move(captures)), result_(std::move(result)) {}

  const TestResult& native() const { return result_; }

  nb::list samples(const std::string& name) const {
    const auto& capture = selected(name);
    if (capture.policy == CapturePolicy::kSnapshot) {
      throw nb::value_error(
          ("output '" + name + "' uses snapshot capture; use snapshots()/snapshot()").c_str());
    }
    return capture.adapter->samples(result_, name);
  }

  nb::list values(const std::string& name) const {
    nb::list output;
    for (const nb::handle sample : samples(name)) {
      output.append(sample.attr("data"));
    }
    return output;
  }

  nb::list snapshots(const std::string& name) const {
    const auto& capture = selected(name);
    if (capture.policy != CapturePolicy::kSnapshot) {
      throw nb::value_error(
          ("output '" + name + "' uses value capture; use values()/value()/samples()").c_str());
    }
    return capture.adapter->snapshots(result_, name);
  }

  void raise_for_status() const {
    if (result_.ok()) {
      return;
    }
    std::optional<ErrorCode> code;
    for (const auto& diagnostic : result_.diagnostics()) {
      if (diagnostic.severity != DiagnosticSeverity::kError || !diagnostic.detail ||
          diagnostic.detail->error.code == ErrorCode::kSuccess) {
        continue;
      }
      // A teardown-only failure still has a native cause. Prefer an initiating failure when
      // one exists, so cleanup cannot replace the original exception's error code.
      if (!code || diagnostic.detail->role == DiagnosticFailureRole::kPrimary) {
        code = diagnostic.detail->error.code;
      }
      if (diagnostic.detail->role == DiagnosticFailureRole::kPrimary) {
        break;
      }
    }
    throw RuntimeError{Error{code.value_or(ErrorCode::kFailure), result_.describe_failure()}};
  }

 private:
  const Capture& selected(std::string_view name) const {
    const auto found = captures_.find(name);
    if (found == captures_.end()) {
      throw nb::value_error(("unknown or unselected output '" + std::string(name) + "'").c_str());
    }
    return found->second;
  }

  // Destroy native results/adapters before releasing the modules that implement them.
  std::shared_ptr<const BindingOwner> owner_;
  Captures captures_;
  TestResult result_;
};

nb::object exactly_one(const std::string& name, const nb::list& values, std::string_view plural) {
  if (nb::len(values) != 1) {
    throw nb::value_error(("output '" + name + "' has " + std::to_string(nb::len(values)) +
                           " captured records; singular access requires one; use " +
                           std::string(plural))
                              .c_str());
  }
  return nb::borrow<nb::object>(values[0]);
}

struct Construction {
  std::shared_ptr<const BindingOwner> owner;
  GilSafeObject type;
  GilSafeObject params;

  authoring::OpaqueOperatorRef operator()(Graph& graph) const {
    const nb::gil_scoped_acquire gil;
    return owner->services->add_operator(
        graph, type.borrow(), "operator", nb::cast<nb::dict>(params.borrow()));
  }
};

// One ticket owns the complete fixture (including watchdog, capture storage and builder lease).
// Its preallocation precedes run admission, so callback finalization never extracts a bare session.
class Running {
 public:
  struct State {
    std::shared_ptr<const BindingOwner> owner;
    Captures captures;
    std::optional<RunningFixture> fixture;
    std::optional<TestResult> result;
    std::mutex joining;
    void join() {
      const std::lock_guard lock{joining};
      if (fixture && !result)
        result.emplace(fixture->finish());
    }
  };

  Running(Captures captures, std::shared_ptr<const BindingOwner> owner)
      : owner_(std::move(owner)), owner_pid_(::getpid()) {
    auto state = std::make_unique<State>();
    state->owner = owner_;
    state->captures = std::move(captures);
    ticket_ = owner_->services->prepare_cleanup(
        {state.get(),
         [](void* value) noexcept {
           auto& state = *static_cast<State*>(value);
           if (state.fixture)
             state.fixture->request_stop();
         },
         [](void* value) { static_cast<State*>(value)->join(); },
         [](void* value) noexcept { delete static_cast<State*>(value); }},
        owner_->testing_module.borrow());
    state_ = state.release();
  }
  ~Running() noexcept {
    if (ticket_ == nullptr)
      return;
    // A child has no Runtime worker; preserve the inherited process-private state as Core does.
    if (owner_pid_ != ::getpid())
      return;
    if (state_->fixture) {
      if (!state_->fixture->wait_admission()) {
        owner_->services->enqueue_cleanup(ticket_);
        return;
      }
      state_->fixture->request_stop();
      try {
        const nb::gil_scoped_release release;
        state_->join();
      } catch (...) {
        owner_->services->enqueue_cleanup(ticket_);
        return;
      }
    }
    owner_->services->destroy_cleanup(ticket_);
  }
  void start(NativeOperatorTest& fixture, TestRunOptions options) {
    const nb::gil_scoped_release release;
    state_->fixture.emplace(fixture.run_async(options));
  }
  std::shared_ptr<Result> finish() {
    require_join();  // Before taking the join mutex: a callback must not wait on its own joiner.
    {
      const nb::gil_scoped_release release;
      state_->join();
    }
    const auto& result = state_->result;
    if (!result.has_value()) {
      throw std::logic_error("joined operator test attempt has no result");
    }
    if (!result_)
      result_ = std::make_shared<Result>(*result, state_->captures, owner_);
    return result_;
  }
  void close() {
    require_join();
    native().request_stop();
    const nb::gil_scoped_release release;
    state_->join();
  }
  void request_stop() { native().request_stop(); }
  bool is_running() const { return native().is_running(); }
  RunTermination termination() const { return native().termination(); }
  std::size_t output_count(const std::string& name) const { return native().output_count(name); }
  bool wait_for_output(const std::string& name, std::size_t minimum, std::int64_t timeout) const {
    require_join();
    const nb::gil_scoped_release release;
    return native().wait_for_output(name, minimum, std::chrono::milliseconds{timeout});
  }
  nb::object operator_ref(nb::handle plan_ref) const {
    const auto reference =
        plan_ref.is_none() ? native().operator_ref()
                           : native().operator_ref(owner_->services->read_operator_ref(plan_ref));
    if (!reference)
      throw RuntimeError{reference.error()};
    return owner_->services->operator_ref(*reference);
  }
  nb::object lifecycle_control(nb::handle policy) {
    const auto value = owner_->services->lifecycle_policy(policy);
    if (!value.paths.empty())
      throw nb::value_error("fixture lifecycle policy requires run-bound operator_ref() targets");
    auto handle = native().lifecycle_control(LifecycleAccessPolicy{
        .allowed_targets = value.targets, .allowed_from_stages = value.stages});
    if (!handle)
      throw RuntimeError{handle.error()};
    return owner_->services->lifecycle_control(std::move(*handle), value.targets);
  }
  void release_frame(std::int64_t timeout) const {
    require_owner();
    const auto result = [&] {
      const nb::gil_scoped_release release;
      return native().release_frame(std::chrono::milliseconds{timeout});
    }();
    if (!result)
      throw RuntimeError{result.error()};
  }

 private:
  void require_owner() const {
    if (owner_pid_ != ::getpid())
      throw RuntimeError{Error{ErrorCode::kInvalidArgument,
                               "PYTHON_TEST_FIXTURE_FORKED_PROCESS: use and join the running "
                               "fixture in its parent process; create a new harness in the child"}};
  }
  const RunningFixture& native() const {
    require_owner();
    auto& fixture = state_->fixture;
    if (!fixture.has_value()) {
      throw std::logic_error("operator test attempt has not started");
    }
    return *fixture;
  }
  RunningFixture& native() {
    require_owner();
    auto& fixture = state_->fixture;
    if (!fixture.has_value()) {
      throw std::logic_error("operator test attempt has not started");
    }
    return *fixture;
  }
  void require_join() const {
    auto admitted = native().wait_admission();
    if (!admitted)
      throw RuntimeError{admitted.error().code,
                         admitted.error().message +
                             " use request_stop() here and join from an admitted owning thread"};
  }
  std::shared_ptr<const BindingOwner> owner_;
  pid_t owner_pid_;
  void* ticket_{};
  State* state_{};
  std::shared_ptr<Result> result_;
};

class Harness {
 public:
  Harness(nb::handle type, const nb::dict& params, nb::handle outputs,
          std::shared_ptr<const BindingOwner> owner)
      : owner_(std::move(owner)), automatic_outputs_(outputs.is_none()) {
    auto factory = construction(type, params);
    Graph probe{"operator-test-declarations"};
    ports_ = (*factory)(probe).ports();
    fixture_ =
        std::make_unique<NativeOperatorTest>([factory](Graph& graph) { return (*factory)(graph); });
    construction_ = std::move(factory);
    if (outputs.is_none()) {
      for (const auto& port : ports_) {
        if (port.direction == PortDirection::kOutput) {
          auto adapter = payload_adapter(port, owner_);
          declare_capture(port.name, adapter->default_policy(), 1024, 64U * 1024U * 1024U);
        }
      }
    } else {
      for (const nb::handle output : nb::cast<nb::sequence>(outputs)) {
        const auto name = nb::cast<std::string>(output);
        auto adapter = payload_adapter(port(name, PortDirection::kOutput), owner_);
        declare_capture(name, adapter->default_policy(), 1024, 64U * 1024U * 1024U);
      }
    }
  }

  void stage(const std::string& name, const nb::sequence& values, const nb::sequence& times) {
    auto lock = enter();
    auto adapter = payload_adapter(port(name, PortDirection::kInput), owner_);
    auto staged = adapter->stage(values, times);
    fixture_->stage_input(name, std::move(staged));
  }

  void frame(const nb::sequence& entries, nb::handle timestamp) {
    auto lock = enter();
    TestFrame frame;
    for (const nb::handle item : entries) {
      const auto entry = nb::cast<nb::sequence>(item);
      if (nb::len(entry) != 2) {
        throw nb::value_error("frame entries must be (name, value) pairs");
      }
      const auto name = nb::cast<std::string>(entry[0]);
      auto adapter = payload_adapter(port(name, PortDirection::kInput), owner_);
      nb::list values;
      nb::list times;
      values.append(entry[1]);
      times.append(timestamp);
      frame.fields.push_back(
          {name,
           adapter->stage(nb::borrow<nb::sequence>(values), nb::borrow<nb::sequence>(times))});
    }
    fixture_->push_frame(frame);
  }

  void capture(const std::string& name, CapturePolicy policy, std::size_t capacity,
               std::size_t maximum_bytes) {
    auto lock = enter();
    declare_capture(name, policy, capacity, maximum_bytes, true);
  }

  std::shared_ptr<Running> run_async(TestRunOptions options) {
    auto lock = enter();
    auto running = std::make_shared<Running>(captures_, owner_);
    running->start(*fixture_, options);
    return running;
  }

  std::shared_ptr<Result> run(TestRunOptions options) {
    require_blocking_admission();
    auto lock = enter();
    auto result = [&] {
      const nb::gil_scoped_release release;
      return fixture_->run(options);
    }();
    return std::make_shared<Result>(std::move(result), captures_, owner_);
  }

  std::shared_ptr<Result> run_controlled_clock(ControlledClockRun options) {
    require_blocking_admission();
    auto lock = enter();
    auto result = [&] {
      const nb::gil_scoped_release release;
      return fixture_->run_controlled_clock(options);
    }();
    return std::make_shared<Result>(std::move(result), captures_, owner_);
  }

  void close() {
    const std::unique_lock lock{mutex_, std::try_to_lock};
    if (!lock.owns_lock()) {
      throw std::runtime_error("operator test harness has an active operation");
    }
    fixture_.reset();
    construction_.reset();
  }

  void input_queue_depth(const std::string& name, std::optional<std::uint32_t> depth) {
    auto lock = enter();
    fixture_->input_queue_depth(name, depth);
  }

  void set_compile_options(CompileOptions options) {
    auto lock = enter();
    fixture_->set_compile_options(std::move(options));
  }

  void set_lifecycle_options(bool self_restart, bool external_restart) {
    auto lock = enter();
    fixture_->configure_graph([self_restart, external_restart](NativeTestGraphView& view) {
      auto lifecycle = view.graph().lifecycle();
      if (self_restart) {
        authoring::OpaqueOperatorRef::lifecycle_allow_self_restart(lifecycle,
                                                                   view.tested_operator());
      }
      if (external_restart) {
        authoring::OpaqueOperatorRef::lifecycle_allow_external_restart(lifecycle,
                                                                       view.tested_operator());
      }
    });
  }

  void set_params(const nb::dict& params) {
    auto lock = enter();
    auto factory = construction(construction_->type.borrow(), params);
    Graph probe{"operator-test-parameters"};
    auto ports = (*factory)(probe).ports();
    Captures captures;
    std::map<std::string, CaptureRequest, std::less<>> requests;
    for (const auto& port : ports) {
      if (port.direction != PortDirection::kOutput)
        continue;
      auto existing = captures_.find(port.name);
      if (!automatic_outputs_ && existing == captures_.end())
        continue;
      auto adapter = payload_adapter(port, owner_);
      const auto policy = existing == captures_.end() || !existing->second.explicit_policy
                              ? adapter->default_policy()
                              : existing->second.policy;
      const auto capacity = existing == captures_.end() ? 1024U : existing->second.capacity;
      const auto bytes =
          existing == captures_.end() ? 64U * 1024U * 1024U : existing->second.maximum_bytes;
      requests.emplace(port.name, adapter->capture(policy, capacity, bytes));
      captures.emplace(port.name,
                       Capture{std::move(adapter),
                               policy,
                               capacity,
                               bytes,
                               existing != captures_.end() && existing->second.explicit_policy});
    }
    // Explicit selections remain selectors and must not silently disappear after reconfiguration.
    for (const auto& [name, capture] : captures_) {
      if ((!automatic_outputs_ || capture.explicit_policy) && !captures.contains(name)) {
        throw nb::value_error("selected output absent after parameter update");
      }
    }
    fixture_->set_operator_configuration([factory](Graph& graph) { return (*factory)(graph); },
                                         std::move(requests));
    ports_.swap(ports);
    captures_.swap(captures);
    construction_ = std::move(factory);
  }

  void set_override(const std::string& kind, const std::string& name, nb::handle value) {
    auto lock = enter();
    const auto* services = owner_->services;
    if (kind == "input_flow" || kind == "output_flow") {
      const auto options = value.is_none() ? std::optional<ConnectionOptions>{}
                                           : services->connection_options(value);
      if (kind == "input_flow")
        fixture_->input_flow(name, options);
      else
        fixture_->output_flow(name, options);
    } else if (kind == "input_tensor_spec") {
      fixture_->input_tensor_spec(
          name,
          value.is_none() ? std::optional<TensorInputSpec>{} : services->tensor_input_spec(value));
    } else if (kind == "output_tensor_spec" || kind == "input_source_tensor_spec") {
      const auto spec =
          value.is_none() ? std::optional<TensorOutputSpec>{} : services->tensor_output_spec(value);
      if (kind == "output_tensor_spec")
        fixture_->output_tensor_spec(name, spec);
      else
        fixture_->input_source_tensor_spec(name, spec);
    } else if (kind == "input_source_device" || kind == "output_device") {
      const auto device = value.is_none() ? std::optional<DeviceId>{} : nb::cast<DeviceId>(value);
      if (kind == "input_source_device")
        fixture_->input_source_device(name, device);
      else
        fixture_->output_device(name, device);
    } else {
      throw nb::value_error("unknown native fixture override");
    }
  }

 private:
  void require_blocking_admission() const {
    if (owner_->services->callback_context_restricted())
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument,
                "PYTHON_TEST_JOIN_FROM_RUNTIME_CALLBACK cannot join from a Runtime callback; "
                "use request_stop() and finish/close from the owning thread"}};
  }
  std::unique_lock<std::mutex> enter() {
    std::unique_lock lock{mutex_, std::try_to_lock};
    if (!lock.owns_lock()) {
      throw std::runtime_error("operator test harness has an active operation");
    }
    if (!fixture_) {
      throw std::runtime_error("operator test harness is closed");
    }
    return lock;
  }

  std::shared_ptr<Construction> construction(nb::handle type, const nb::dict& params) const {
    const nb::object copied = nb::module_::import_("copy").attr("deepcopy")(params);
    return std::make_shared<Construction>(
        Construction{owner_, GilSafeObject{type}, GilSafeObject{copied}});
  }

  const PortRecord& port(std::string_view name, PortDirection direction) const {
    for (const auto& value : ports_) {
      if (value.name == name && value.direction == direction) {
        return value;
      }
    }
    std::string available;
    for (const auto& value : ports_) {
      if (value.direction == direction) {
        if (!available.empty())
          available += ", ";
        available += value.name;
      }
    }
    throw nb::value_error(("unknown port or direction mismatch: '" + std::string(name) +
                           "'; available ports: " + available)
                              .c_str());
  }

  void declare_capture(const std::string& name, CapturePolicy policy, std::size_t capacity,
                       std::size_t maximum_bytes, bool explicit_policy = false) {
    auto adapter = payload_adapter(port(name, PortDirection::kOutput), owner_);
    if ((policy == CapturePolicy::kValue) != (adapter->default_policy() == CapturePolicy::kValue)) {
      throw nb::value_error(("output '" + name +
                             "': capture policy does not match the payload; use " +
                             (adapter->default_policy() == CapturePolicy::kValue
                                  ? "capture()"
                                  : "capture_snapshot() or capture_native()"))
                                .c_str());
    }
    auto request = adapter->capture(policy, capacity, maximum_bytes);
    fixture_->declare_capture(name, std::move(request));
    captures_.insert_or_assign(
        name, Capture{std::move(adapter), policy, capacity, maximum_bytes, explicit_policy});
  }

  std::shared_ptr<const BindingOwner> owner_;
  std::shared_ptr<Construction> construction_;
  std::vector<PortRecord> ports_;
  Captures captures_;
  std::unique_ptr<NativeOperatorTest> fixture_;
  std::mutex mutex_;
  bool automatic_outputs_;
};

}  // namespace

NB_MODULE(_testing, module) {
  using namespace nb::literals;
  const auto core = nb::module_::import_("holoscan._core");
  const auto incompatible_services = [] {
    return nb::import_error(
        "holoscan.testing requires matching Core binding services, SDK version, "
        "and interface/build compatibility");
  };
  if (!nb::hasattr(core, "_binding_services")) {
    throw incompatible_services();
  }
  const nb::object capsule = core.attr("_binding_services");
  if (PyCapsule_IsValid(capsule.ptr(), holoscan::python::kBindingServicesCapsule) == 0) {
    throw incompatible_services();
  }
  const auto* services = static_cast<const holoscan::python::BindingServices*>(
      PyCapsule_GetPointer(capsule.ptr(), holoscan::python::kBindingServicesCapsule));
  if (services == nullptr || services->abi_version != holoscan::python::kBindingServicesAbi ||
      services->size != sizeof(holoscan::python::BindingServices) ||
      services->sdk_version == nullptr ||
      std::string_view(services->sdk_version) != HOLOSCAN_PYTHON_SERVICES_VERSION ||
      services->compatibility_id == nullptr ||
      std::string_view(services->compatibility_id) !=
          holoscan::python::kBindingServicesCompatibility) {
    throw incompatible_services();
  }
  require_core_type<CompileOptions>(core, "CompileOptions");
  require_core_type<DeviceId>(core, "DeviceId");
  auto owner = std::make_shared<BindingOwner>(
      BindingOwner{GilSafeObject{core}, GilSafeObject{module}, services});
  init_schema_snapshot(module);

  nb::enum_<CapturePolicy>(module, "_CapturePolicy")
      .value("VALUE", CapturePolicy::kValue)
      .value("SNAPSHOT", CapturePolicy::kSnapshot)
      .value("NATIVE", CapturePolicy::kNative);
  holoscan::python::bind_enum_in_module<FrameRelease>(module, "holoscan.testing", "FrameRelease")
      .value("AUTOMATIC", FrameRelease::kAutomatic)
      .value("MANUAL", FrameRelease::kManual);
  holoscan::python::bind_enum_in_module<ClockTestOutcome>(
      module, "holoscan.testing", "ClockTestOutcome")
      .value("NOT_REQUESTED", ClockTestOutcome::kNotRequested)
      .value("RETIRED", ClockTestOutcome::kRetired)
      .value("SUPPRESSED", ClockTestOutcome::kSuppressed)
      .value("FAILED", ClockTestOutcome::kFailed);
  holoscan::python::bind_class_in_module<TestRunOptions>(
      module,
      "holoscan.testing",
      "TestRunOptions",
      "Configure one finite attempt. Values are copied for the run; invalid options reject when "
      "the attempt starts.")
      .def(
          "__init__",
          [](TestRunOptions* self, std::int64_t timeout, bool expect_stop, FrameRelease frames) {
            new (self) TestRunOptions{std::chrono::milliseconds{timeout}, expect_stop, frames};
          },
          nb::kw_only(),
          "timeout_ms"_a = 30000,
          "expect_stop"_a = false,
          "frames"_a = FrameRelease::kAutomatic)
      .def_prop_rw(
          "timeout_ms",
          [](const TestRunOptions& v) { return v.timeout.count(); },
          [](TestRunOptions& v, std::int64_t t) { v.timeout = std::chrono::milliseconds{t}; },
          "Positive steady-clock watchdog in integer milliseconds; defaults to 30000. Expiry "
          "requests cooperative stop and fails the test, but cannot interrupt a blocked callback.")
      .def_rw("expect_stop",
              &TestRunOptions::expect_stop,
              "Accept clean cooperative stop as expected. Errors, watchdog expiry and missing "
              "controlled-clock/retention completion proof still fail.")
      .def_rw("frames",
              &TestRunOptions::frames,
              "Select automatic or manual finite frame release. Manual release requires run_async "
              "and grouped input staging.");
  holoscan::python::bind_class_in_module<ControlledClockRun>(
      module,
      "holoscan.testing",
      "ControlledClockRun",
      "Observe one positive period-boundary tick with a shared clock initially frozen at zero. The "
      "startup tick is real. This restricted profile is not an operator stepping API.")
      .def(
          "__init__",
          [](ControlledClockRun* self,
             std::int64_t tick,
             std::size_t capacity,
             std::int64_t timeout) {
            new (self) ControlledClockRun{
                std::chrono::nanoseconds{tick}, capacity, std::chrono::milliseconds{timeout}};
          },
          nb::kw_only(),
          "tick_ns"_a,
          "observation_capacity"_a = 1024,
          "timeout_ms"_a = 30000)
      .def_prop_rw(
          "tick_ns",
          [](const ControlledClockRun& v) { return v.tick.count(); },
          [](ControlledClockRun& v, std::int64_t t) { v.tick = std::chrono::nanoseconds{t}; },
          "Positive tick coordinate in integer nanoseconds, exactly on a boundary of the operator "
          "clock period.")
      .def_rw("observation_capacity",
              &ControlledClockRun::observation_capacity,
              "Positive bound on native observation records and total publication attempts; "
              "defaults to 1024.")
      .def_prop_rw(
          "timeout_ms",
          [](const ControlledClockRun& v) { return v.timeout.count(); },
          [](ControlledClockRun& v, std::int64_t t) { v.timeout = std::chrono::milliseconds{t}; },
          "Positive steady-clock cooperative watchdog in integer milliseconds, independent of "
          "synthetic time; defaults to 30000.");
  holoscan::python::bind_class_in_module<Retain>(
      module,
      "holoscan.testing",
      "Retain",
      "Request native Tensor backing without a host copy. Admission requires a finite source with "
      "one output and proven capacity and completion; unsupported geometries reject.")
      .def(
          "__init__",
          [](Retain* self, std::size_t count) { new (self) Retain{count}; },
          nb::kw_only(),
          "max_records"_a = 1024)
      .def_rw("max_records",
              &Retain::max_records,
              "Positive finite retained-record limit; defaults to 1024. Overflow fails capture "
              "without truncating output.");
  holoscan::python::bind_class_in_module<CaptureFailure>(
      module,
      "holoscan.testing",
      "CaptureFailure",
      "Own a failed capture's sample metadata and native error without retaining the payload or a "
      "finite read claim.")
      .def_ro("metadata",
              &CaptureFailure::metadata,
              "Return native sample metadata associated with the failed capture.")
      .def_ro("error",
              &CaptureFailure::error,
              "Return the native error code and detail explaining why capture failed.");
  holoscan::python::bind_class_in_module<TensorSnapshot>(
      module,
      "holoscan.testing",
      "TensorSnapshot",
      "Own contiguous host content and original Tensor metadata independently of runtime pools. "
      "This inspection record is not a Tensor for publication.")
      .def_prop_ro(
          "bytes",
          [](const TensorSnapshot& value) {
            return nb::bytes(reinterpret_cast<const char*>(value.bytes().data()),
                             value.bytes().size());
          },
          "Return a Python bytes copy of contiguous logical Tensor content. Source strides and "
          "offset describe the original allocation, not this host buffer.")
      .def_prop_ro(
          "shape", &TensorSnapshot::shape, "Return the original logical dimension extents.")
      .def_prop_ro(
          "dtype", &TensorSnapshot::dtype, "Return the original DLPack element representation.")
      .def_prop_ro(
          "source_device",
          &TensorSnapshot::source_device,
          "Return the original DLPack device, which can differ from the snapshot host storage.")
      .def_prop_ro(
          "source_byte_strides",
          &TensorSnapshot::source_byte_strides,
          "Return the original layout strides in bytes, including explicit contiguous strides.")
      .def_prop_ro("source_byte_offset",
                   &TensorSnapshot::source_byte_offset,
                   "Return the logical Tensor byte offset within its original source allocation.")
      .def_prop_ro("sample_metadata",
                   &TensorSnapshot::sample_metadata,
                   "Return semantic sample metadata copied when the output was collected.");
  holoscan::python::bind_class_in_module<Result>(
      module,
      "holoscan.testing",
      "TestResult",
      "Own joined diagnostics and captures independently of the harness. Inspection never executes "
      "work. Python-object identity is retained; snapshots own host content. Retaining the plan "
      "can retain staged Tensor leases.")
      .def_prop_ro(
          "ok",
          [](const Result& r) { return r.native().ok(); },
          "Report whether admission, capture and expected completion succeeded. Watchdog expiry "
          "and capture failures fail the test; a clean stopped run needs expect_stop or the native "
          "controlled-clock/retention completion proof.")
      .def_prop_ro(
          "termination",
          [](const Result& r) { return r.native().termination(); },
          "Return actual native run termination, independently of whether the test passed.")
      .def_prop_ro(
          "diagnostics",
          [](const Result& r) {
            const auto values = r.native().diagnostics();
            return std::vector<Diagnostic>{values.begin(), values.end()};
          },
          "Return a copied list of compile and runtime diagnostics retained after join.")
      .def_prop_ro(
          "plan",
          [](const Result& r) { return r.native().plan(); },
          "Return the compiled plan, including rejected-admission diagnostics. Inspection does not "
          "execute it.")
      .def_prop_ro(
          "clock_outcome",
          [](const Result& r) { return r.native().clock_outcome(); },
          "Return the controlled tick retirement or suppression outcome; ordinary runs return "
          "NOT_REQUESTED.")
      .def(
          "clock_selection",
          [](const Result& r, const std::string& name) { return r.native().clock_selection(name); },
          "Return the named input selection at the completed controlled tick. Reject ordinary "
          "runs, incomplete observations and unknown inputs.")
      .def(
          "capture_failure",
          [](const Result& r, const std::string& name) { return r.native().capture_failure(name); },
          "Return the failed capture error and sample metadata, or None if collection did not "
          "fail. Reject unknown or uncaptured outputs. The failed payload is not retained.")
      .def(
          "describe_failure",
          [](const Result& r) { return r.native().describe_failure(); },
          "Describe rejected admission, termination, capture errors and watchdog expiry; return an "
          "empty string on success.")
      .def("samples",
           &Result::samples,
           "Return value or native-retention captures with native sample metadata, preserving "
           "Python-object identity. Raise ValueError for unknown, unselected or snapshot outputs.")
      .def("values",
           &Result::values,
           "Return a new list of value or native-retention captures in collection order, possibly "
           "empty. Python objects are not deep-copied. Raise ValueError for unknown, unselected or "
           "snapshot outputs.")
      .def(
          "value",
          [](const Result& r, const std::string& name) {
            return exactly_one(name, r.values(name), "values()");
          },
          "Return the sole value or native-retention payload. Raise ValueError unless exactly one "
          "record exists, or for unknown, unselected or snapshot outputs.")
      .def("snapshots",
           &Result::snapshots,
           "Return independent host Tensor or schema snapshots in collection order, possibly "
           "empty. Raise ValueError for unknown, unselected or value-capture outputs.")
      .def(
          "snapshot",
          [](const Result& r, const std::string& name) {
            return exactly_one(name, r.snapshots(name), "snapshots()");
          },
          "Return the sole Tensor or schema snapshot. Raise ValueError unless exactly one record "
          "exists, or for unknown, unselected or value-capture outputs.")
      .def("_raise_for_status", &Result::raise_for_status);
  nb::class_<Running>(module, "_RunningFixture")
      .def("finish", &Running::finish)
      .def("close", &Running::close)
      .def("request_stop", &Running::request_stop)
      .def_prop_ro("is_running", &Running::is_running)
      .def_prop_ro("termination", &Running::termination)
      .def("output_count", &Running::output_count)
      .def("wait_for_output", &Running::wait_for_output)
      .def("release_frame", &Running::release_frame)
      .def("operator_ref", &Running::operator_ref, "plan_ref"_a.none())
      .def("lifecycle_control", &Running::lifecycle_control);
  for (const char* name : {"TestRunOptions", "TestResult"}) {
    module.attr(name).attr("__test__") = false;
  }
  nb::class_<Harness>(module, "_Harness")
      .def(
          "__init__",
          [owner](Harness* self, nb::handle type, const nb::dict& params, nb::handle outputs) {
            new (self) Harness{type, params, outputs, owner};
          },
          "operator"_a,
          "params"_a,
          "outputs"_a.none())
      .def("stage", &Harness::stage)
      .def("push_frame", &Harness::frame, "entries"_a, "timestamp"_a.none())
      .def("capture", &Harness::capture)
      .def("run", &Harness::run)
      .def("run_async", &Harness::run_async)
      .def("run_controlled_clock", &Harness::run_controlled_clock)
      .def("close", &Harness::close)
      .def("input_queue_depth", &Harness::input_queue_depth)
      .def("set_compile_options", &Harness::set_compile_options)
      .def("set_lifecycle_options",
           &Harness::set_lifecycle_options,
           "self_restart"_a.noconvert(),
           "external_restart"_a.noconvert())
      .def("set_params", &Harness::set_params)
      .def("set_override", &Harness::set_override, "kind"_a, "name"_a, "value"_a.none());
}

}  // namespace holoscan::testing::python
