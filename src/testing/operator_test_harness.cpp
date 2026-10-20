/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <functional>
#include <holoscan/testing/operator_test_harness.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <map>
#include <mutex>
#include <set>
#include <typeinfo>

#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include "operator_test_run.hpp"

namespace holoscan::testing {
namespace detail {

const void* builtin_capture_token(PayloadIdentityKind kind, ScalarIdentityKind scalar) noexcept {
  static const std::array<char, 12> tokens{};
  if (kind == PayloadIdentityKind::kTensor)
    return &tokens.back();
  const auto index = static_cast<std::size_t>(scalar);
  return kind == PayloadIdentityKind::kScalar && index < tokens.size() - 1U ? &tokens[index]
                                                                            : nullptr;
}

struct OwnedConnectionOptions {
  explicit OwnedConnectionOptions(ConnectionOptions value) : value_(value) {
    if (value.topic_name)
      topic_ = std::string{*value.topic_name};
    value_.topic_name.reset();
  }
  [[nodiscard]] ConnectionOptions view() const {
    auto result = value_;
    if (topic_)
      result.topic_name = *topic_;
    return result;
  }
  ConnectionOptions value_;
  std::optional<std::string> topic_;
};

template <typename Spec>
struct OwnedTensorSpec {
  explicit OwnedTensorSpec(Spec value) : value_(value) {
    if (!value_.bounds)
      return;
    auto& bounds = *value_.bounds;
    if (bounds.max_shape)
      shape_.emplace(bounds.max_shape->begin(), bounds.max_shape->end());
    if (bounds.max_abs_byte_strides) {
      strides_.emplace(bounds.max_abs_byte_strides->begin(), bounds.max_abs_byte_strides->end());
    }
    axes_.assign(bounds.axes.begin(), bounds.axes.end());
    bounds.max_shape.reset();
    bounds.max_abs_byte_strides.reset();
    bounds.axes = {};
  }
  [[nodiscard]] Spec view() const {
    auto result = value_;
    if (result.bounds) {
      if (shape_)
        result.bounds->max_shape = TensorShapeView{*shape_};
      if (strides_)
        result.bounds->max_abs_byte_strides = TensorMaxAbsByteStridesView{*strides_};
      result.bounds->axes = axes_;
    }
    return result;
  }
  Spec value_;
  std::optional<std::vector<std::int64_t>> shape_;
  std::optional<std::vector<std::uint64_t>> strides_;
  std::vector<TensorAxisBounds> axes_;
};

struct GroupDescription {
  struct Port {
    std::string name;
    std::shared_ptr<const InputScript> values;
  };
  struct Publication {
    std::size_t port;
    std::size_t value;
  };
  std::vector<Port> ports;
  std::vector<std::vector<Publication>> frames;
};

class FrameSource final : public Operator<> {
 public:
  explicit FrameSource(std::shared_ptr<const GroupDescription> script,
                       std::shared_ptr<CaptureMailbox<FrameReleaseState>> release)
      : script_(std::move(script)),
        release_(std::move(release)),
        release_state_(release_ ? std::make_shared<FrameReleaseState>() : nullptr) {
    outputs_.reserve(script_->ports.size());
    for (const auto& port : script_->ports)
      outputs_.push_back(port.values->make_writer());
  }
  void setup(OperatorSpec& spec) override {
    for (std::size_t i = 0; i < outputs_.size(); ++i) {
      outputs_[i]->setup(spec, script_->ports[i].name);
    }
    if (release_) {
      spec.notification_source(notification_, "release_frame")
          .capacity(1U)
          .sender_reference_capacity(2U);
      spec.lifecycle()
          .stage(LifecycleStage::kArm, &FrameSource::arm)
          .stage(LifecycleStage::kStop, &FrameSource::stop_generation);
    }
  }
  LifecycleStatus stop_generation(LifecycleContext& context) noexcept {
    if (context.transition_phase() == LifecycleTransitionPhase::kPreRestartTeardown) {
      // Reset per-transaction bookkeeping even if failure occurs before candidate Arm.
      const std::lock_guard lock(release_state_->mutex);
      predecessor_sender_ = {};
      candidate_sender_ = false;
    }
    return LifecycleStatus::kOk;
  }
  LifecycleStatus arm(LifecycleContext& context) noexcept {
    if (!claimed_) {
      const std::lock_guard lock(release_->initial->mutex);
      if (release_->initial->driver_closed || release_->claimed.exchange(true)) {
        context.report_error(
            ErrorCode::kClosed,
            "manual frame plan cannot be replayed; stage a new fixture attempt with run_async()");
        return LifecycleStatus::kFatalFailure;
      }
      release_state_ = release_->initial;
      claimed_ = true;
    }
    const std::lock_guard lock(release_state_->mutex);
    if (context.transition_phase() == LifecycleTransitionPhase::kRestoreOld) {
      // Native restoration reactivates the predecessor; it forbids acquiring new authority.
      if (candidate_sender_) {
        release_state_->sender = std::move(predecessor_sender_);
        candidate_sender_ = false;
      }
      return LifecycleStatus::kOk;
    }
    // A prior successful transaction leaves an obsolete predecessor record. No caller can
    // pin it across re-arm because release_frame posts under this same mutex.
    predecessor_sender_ = {};
    candidate_sender_ = false;
    auto sender = context.notification_sender(notification_);
    if (!sender) {
      context.report_error(sender.error());
      return LifecycleStatus::kFatalFailure;
    }
    predecessor_sender_ = std::move(release_state_->sender);
    release_state_->sender = std::move(*sender);
    release_state_->armed = true;
    candidate_sender_ = context.transition_phase() == LifecycleTransitionPhase::kRestartForward;
    return LifecycleStatus::kOk;
  }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    if (release_)
      result.trigger(OnNotified{.event = notification_});
    else
      result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
    result.complete_after(ComputeCalls{.limit = script_->frames.size()});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (next_ >= script_->frames.size()) {
      return make_unexpected(
          Error{ErrorCode::kProtocolError, "operator test frame source exhausted"});
    }
    for (const auto& publication : script_->frames[next_]) {
      auto emitted = outputs_[publication.port]->emit(publication.value, context);
      if (!emitted)
        return make_unexpected(std::move(emitted).error());
    }
    ++next_;
    return {};
  }
  [[nodiscard]] const OutputBase& output(std::size_t index) const {
    return outputs_.at(index)->output();
  }

 private:
  std::shared_ptr<const GroupDescription> script_;
  std::vector<std::unique_ptr<InputPortWriter>> outputs_;
  std::size_t next_{};
  std::shared_ptr<CaptureMailbox<FrameReleaseState>> release_;
  std::shared_ptr<FrameReleaseState> release_state_;
  NotificationSource notification_;
  NotificationSender predecessor_sender_;
  bool candidate_sender_{};
  bool claimed_{};
};

struct OperatorTestState {
  explicit OperatorTestState(NativeOperatorTest::OperatorFactory create)
      : factory(std::move(create)) {}
  std::shared_ptr<AttemptGate> gate = std::make_shared<AttemptGate>();
  CompileOptions options;
  std::function<void(NativeTestGraphView&)> hook;
  std::map<std::string, std::uint32_t, std::less<>> input_depths;
  std::map<std::string, OwnedConnectionOptions, std::less<>> input_flows;
  std::map<std::string, OwnedConnectionOptions, std::less<>> output_flows;
  std::map<std::string, OwnedTensorSpec<TensorInputSpec>, std::less<>> input_specs;
  std::map<std::string, OwnedTensorSpec<TensorOutputSpec>, std::less<>> output_specs;
  std::map<std::string, OwnedTensorSpec<TensorOutputSpec>, std::less<>> source_specs;
  std::map<std::string, DeviceId, std::less<>> source_devices;
  std::map<std::string, DeviceId, std::less<>> output_devices;
  NativeOperatorTest::OperatorFactory factory;
  std::map<std::string, std::shared_ptr<const InputScript>, std::less<>> inputs;
  std::shared_ptr<const GroupDescription> grouped;
  std::map<std::string, std::shared_ptr<const CaptureDeclaration>, std::less<>> captures;

  void validate_flow_selectors() const {
    for (const auto& [name, options] : input_flows) {
      const bool grouped_port =
          grouped && std::any_of(grouped->ports.begin(),
                                 grouped->ports.end(),
                                 [&](const auto& port) { return port.name == name; });
      if (!inputs.contains(name) && !grouped_port) {
        throw std::invalid_argument{"operator test input flow '" + name +
                                    "' has no staged producer"};
      }
    }
    for (const auto& [name, options] : output_flows) {
      if (!captures.contains(name)) {
        throw std::invalid_argument{"operator test output flow '" + name + "' has no capture"};
      }
    }
  }

  void require_inactive() const {
    if (gate->active) {
      throw std::logic_error{"operator test already has an active attempt"};
    }
  }
};

template <typename Validate>
void validate_selector(OperatorTestState& state, std::string_view name, PortDirection direction,
                       Validate&& validate) {
  if (name.empty())
    throw std::invalid_argument{"operator test port name must not be empty"};
  Graph graph{"operator-test-setting-validation"};
  const auto tested = state.factory(graph);
  holoscan::detail::OperatorTestingAccess::require_graph(graph, tested);
  try {
    const auto record =
        holoscan::detail::OperatorTestingAccess::port_record(tested, name, direction);
    std::forward<Validate>(validate)(tested, record);
  } catch (const RuntimeError& error) {
    throw RuntimeError{error.code(),
                       "operator test port '" + std::string(name) + "': " + error.what()};
  }
}

void validate_selector(OperatorTestState& state, std::string_view name, PortDirection direction) {
  validate_selector(state, name, direction, [](const auto&, const auto&) {});
}

void require_tensor_port(const PortRecord& port) {
  if (port.payload_identity.kind != PayloadIdentityKind::kTensor &&
      !(port.payload_identity.kind == PayloadIdentityKind::kSchema &&
        port.payload_identity.payload_kind == PayloadKind::kDescriptorBacked)) {
    throw std::invalid_argument{"operator test port '" + port.name +
                                "' does not carry Tensor data"};
  }
}

std::string helper_name(std::string_view prefix, std::string_view port) {
  std::string name{prefix};
  constexpr std::string_view digits{"0123456789abcdef"};
  for (const unsigned char c : port) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) {
      name += static_cast<char>(c);
    } else {
      name += "_x";
      name += digits[c >> 4U];
      name += digits[c & 15U];
      name += '_';
    }
  }
  return name;
}

void connect_input(Graph& graph, const authoring::OpaqueOperatorRef& tested, std::string_view name,
                   const OutputBase& source, const PayloadIdentityRecord& identity,
                   const void* local_type, ConnectionOptions options) {
  connect_input(graph, tested, name, source, identity, local_type, options, nullptr);
}

void connect_input(Graph& graph, const authoring::OpaqueOperatorRef& tested, std::string_view name,
                   const OutputBase& source, const PayloadIdentityRecord& identity,
                   const void* local_type, ConnectionOptions options,
                   const std::type_info* local_type_info) {
  const auto* target =
      static_cast<const InputBase*>(holoscan::detail::OperatorTestingAccess::resolve(
          tested, name, PortDirection::kInput, identity, local_type, local_type_info));
  holoscan::detail::GraphPayloadRegistrationAccess::add_flow(
      graph, source, *target, identity, identity, options);
}

void connect_output(Graph& graph, const authoring::OpaqueOperatorRef& tested, std::string_view name,
                    const InputBase& collector, const PayloadIdentityRecord& identity,
                    const void* local_type, ConnectionOptions options) {
  connect_output(graph, tested, name, collector, identity, local_type, options, nullptr);
}

void connect_output(Graph& graph, const authoring::OpaqueOperatorRef& tested, std::string_view name,
                    const InputBase& collector, const PayloadIdentityRecord& identity,
                    const void* local_type, ConnectionOptions options,
                    const std::type_info* local_type_info) {
  const auto* source =
      static_cast<const OutputBase*>(holoscan::detail::OperatorTestingAccess::resolve(
          tested, name, PortDirection::kOutput, identity, local_type, local_type_info));
  holoscan::detail::GraphPayloadRegistrationAccess::add_flow(
      graph, *source, collector, identity, identity, options);
}

HelperPorts build_input_sources(Graph& graph, const authoring::OpaqueOperatorRef& tested,
                                const OperatorTestState& state,
                                const std::shared_ptr<FrameReleaseState>& frames) {
  HelperPorts sources;
  for (const auto& [name, script] : state.inputs) {
    const auto flow = state.input_flows.find(name);
    const auto options = flow == state.input_flows.end()
                             ? std::optional<ConnectionOptions>{}
                             : std::optional<ConnectionOptions>{flow->second.view()};
    const auto source_spec = state.source_specs.find(name);
    const auto spec = source_spec == state.source_specs.end()
                          ? std::optional<TensorOutputSpec>{}
                          : std::optional<TensorOutputSpec>{source_spec->second.view()};
    auto helper =
        script->build(graph, tested, name, helper_name("input_source_", name), options, spec);
    sources.emplace(name, detail::HelperPort{std::move(helper), "output"});
  }
  if (state.grouped) {
    const auto mailbox =
        frames ? std::make_shared<CaptureMailbox<FrameReleaseState>>(frames) : nullptr;
    const auto source = graph.op<detail::FrameSource>("frame_source", state.grouped, mailbox);
    for (std::size_t i = 0U; i < state.grouped->ports.size(); ++i) {
      const auto& port = state.grouped->ports[i];
      const auto flow = state.input_flows.find(port.name);
      const auto options =
          flow == state.input_flows.end()
              ? ConnectionOptions{.queue_depth = static_cast<std::uint32_t>(port.values->size())}
              : flow->second.view();
      const auto source_spec = state.source_specs.find(port.name);
      if (source_spec != state.source_specs.end()) {
        holoscan::detail::OperatorTestingAccess::output_tensor_spec(
            holoscan::detail::OperatorTestingAccess::erase(source),
            port.name,
            source_spec->second.view());
      }
      detail::connect_input(graph,
                            tested,
                            port.name,
                            source->output(i),
                            port.values->identity(),
                            port.values->type_token(),
                            options,
                            port.values->payload_type_info());
      sources.emplace(
          port.name,
          detail::HelperPort{holoscan::detail::OperatorTestingAccess::erase(source), port.name});
    }
  }
  for (const auto& [name, spec] : state.source_specs) {
    if (!sources.contains(name))
      throw std::invalid_argument{"operator test source spec '" + name + "' has no producer"};
  }
  return sources;
}

std::shared_ptr<FrameReleaseState> prepare_frame_release(const OperatorTestState& state,
                                                         TestRunOptions options) {
  if (options.frames == FrameRelease::kAutomatic)
    return {};
  if (options.frames != FrameRelease::kManual)
    throw std::invalid_argument{"operator test frame release policy is invalid"};
  if (!state.inputs.empty() || !state.grouped || state.grouped->frames.empty())
    throw std::invalid_argument{"manual operator test requires a nonempty grouped script only"};
  return std::make_shared<FrameReleaseState>();
}

void validate_controlled_staging(const OperatorTestState& state) {
  if (state.inputs.empty() && (!state.grouped || state.grouped->ports.empty()))
    throw std::invalid_argument{"controlled operator test requires staged scalar inputs"};
  const auto validate_script = [](const detail::InputScript& script) {
    if (script.size() != 1U || script.identity().kind != PayloadIdentityKind::kScalar)
      throw std::invalid_argument{
          "controlled operator test requires exactly one scalar sample per input"};
  };
  for (const auto& [name, script] : state.inputs)
    validate_script(*script);
  if (state.grouped) {
    if (state.grouped->frames.size() != 1U)
      throw std::invalid_argument{"controlled operator test currently requires one grouped frame"};
    for (const auto& port : state.grouped->ports)
      validate_script(*port.values);
  }
}

void apply_port_overrides(const OperatorTestState& state,
                          const authoring::OpaqueOperatorRef& tested) {
  for (const auto& [name, depth] : state.input_depths) {
    holoscan::detail::OperatorTestingAccess::input_queue_depth(tested, name, depth);
  }
  for (const auto& [name, spec] : state.input_specs) {
    holoscan::detail::OperatorTestingAccess::input_tensor_spec(tested, name, spec.view());
  }
  for (const auto& [name, spec] : state.output_specs) {
    holoscan::detail::OperatorTestingAccess::output_tensor_spec(tested, name, spec.view());
  }
}

HelperPorts build_collectors(Graph& graph, const authoring::OpaqueOperatorRef& tested,
                             const OperatorTestState& state, TestResultState& result,
                             CaptureHandoffs& capture_handoffs) {
  HelperPorts collectors;
  for (const auto& [name, capture] : state.captures) {
    auto storage = capture->make_storage();
    auto detached = storage->empty_storage();
    capture_handoffs.emplace_back(storage, detached);
    const auto flow = state.output_flows.find(name);
    const auto options = flow == state.output_flows.end()
                             ? std::optional<ConnectionOptions>{}
                             : std::optional<ConnectionOptions>{flow->second.view()};
    auto helper = capture->build(
        graph, tested, name, helper_name("output_collector_", name), storage, options);
    collectors.emplace(name, HelperPort{std::move(helper), "input"});
    result.captures.emplace(name, std::move(detached));
  }
  return collectors;
}

void prepare_retained_captures(const OperatorTestState& state, TestResultState& result,
                               CaptureHandoffs& capture_handoffs, std::string_view tested_name) {
  std::size_t capture_index{};
  for (const auto& [name, capture] : state.captures) {
    if (const auto request = capture->retention_request()) {
      result.retention_required = true;
      capture_handoffs[capture_index].first->retained.plan =
          detail::prepare_retention(result.plan, tested_name, name, *request);
    }
    ++capture_index;
  }
}

std::vector<ClockFixtureConnection> clock_connections(const HelperPorts& ports) {
  std::vector<ClockFixtureConnection> connections;
  connections.reserve(ports.size());
  for (const auto& [name, helper] : ports)
    connections.push_back({name, std::string{helper.op.name()}, helper.port});
  return connections;
}

}  // namespace detail

TestResult::TestResult(std::shared_ptr<const detail::TestResultState> state)
    : state_(std::move(state)) {}

bool TestResult::ok() const noexcept {
  if (!state_->plan.ok() || state_->watchdog_expired || state_->diagnostics_copy_failed)
    return false;
  for (const auto& [name, capture] : state_->captures) {
    if (capture->failure)
      return false;
  }
  if (std::any_of(state_->diagnostics.begin(), state_->diagnostics.end(), [](const auto& value) {
        return value.severity == DiagnosticSeverity::kError;
      }))
    return false;
  if (state_->clock_requested) {
    return state_->clock && state_->clock->outcome() != ClockTestOutcome::kFailed &&
           state_->termination == RunTermination::kStopped;
  }
  if (state_->retention_required)
    return state_->observed_completion;  // Includes the proved zero-record natural boundary.
  return state_->termination == RunTermination::kCompleted ||
         state_->termination == RunTermination::kRequestedCompleted ||
         (state_->expect_stop && state_->termination == RunTermination::kStopped);
}

ClockTestOutcome TestResult::clock_outcome() const noexcept {
  return state_->clock ? state_->clock->outcome()
                       : (state_->clock_requested ? ClockTestOutcome::kFailed
                                                  : ClockTestOutcome::kNotRequested);
}

InputSelection TestResult::clock_selection(std::string_view name) const {
  if (!state_->clock)
    throw std::logic_error{"this result has no controlled tick"};
  return state_->clock->selection(name);
}

RunTermination TestResult::termination() const noexcept {
  return state_->termination;
}
const ExecutionPlan& TestResult::plan() const noexcept {
  return state_->plan;
}
std::span<const Diagnostic> TestResult::diagnostics() const noexcept {
  return state_->diagnostics;
}

std::string TestResult::describe_failure() const {
  if (ok()) {
    return {};
  }
  std::string result = state_->watchdog_expired ? "operator test watchdog expired\n" : "";
  if (state_->clock && !state_->clock->failure.empty())
    result += state_->clock->failure + '\n';
  if (state_->retention_required && !state_->observed_completion)
    result += "operator test Retain completion boundary was not reached\n";
  if (state_->diagnostics_copy_failed)
    result += "operator test could not copy all Runtime diagnostics: allocation failed\n";
  const auto termination_description = [this]() -> std::string_view {
    switch (termination()) {
      case RunTermination::kRunning:
        return "kRunning: the attempt has not finished";
      case RunTermination::kCompleted:
        return "kCompleted: natural execution and cleanup completed, but test evidence failed";
      case RunTermination::kRequestedCompleted:
        return "kRequestedCompleted: operator requested success; cleanup completed, but test "
               "evidence failed";
      case RunTermination::kStopped:
        return "kStopped: cooperative stop; requires explicit expect_stop or a qualified observed "
               "boundary";
      case RunTermination::kFailed:
        return "kFailed: admission, startup, execution or cleanup failed";
    }
    return "unknown Runtime termination";
  };
  result += "operator test termination: ";
  result += termination_description();
  for (const auto& [name, capture] : state_->captures) {
    if (const auto& failure = capture->failure; failure.has_value()) {
      const auto& failed_capture = *failure;
      result += "\ncapture '" + name + "': " + runtime_error_message(failed_capture.error);
      result += "; frame=" + std::to_string(failed_capture.metadata.frame_id);
      result += "; timestamp_ns=" + std::to_string(failed_capture.metadata.capture_timestamp_ns);
    }
  }
  for (const auto& diagnostic : state_->diagnostics) {
    result += '\n';
    result += format_diagnostic(diagnostic, state_->plan.graph());
  }
  return result;
}

std::optional<CaptureFailure> TestResult::capture_failure(std::string_view name) const {
  return capture_storage(name).failure;
}

const detail::CaptureStorage& TestResult::capture_storage(std::string_view name) const {
  const auto found = state_->captures.find(name);
  if (found == state_->captures.end()) {
    throw std::invalid_argument{"operator test output '" + std::string(name) +
                                "' was not captured"};
  }
  return *found->second;
}

const detail::HelperPort& NativeTestGraphView::helper_port(const detail::HelperPorts& ports,
                                                           std::string_view name) {
  const auto found = ports.find(name);
  if (found == ports.end())
    throw std::invalid_argument{"operator test helper selector '" + std::string(name) +
                                "' not found"};
  return found->second;
}

namespace {
void add_helpers(PartitionHandle& partition, const detail::HelperPorts& ports) {
  std::set<std::string_view> names;
  std::vector<authoring::OpaqueOperatorRef> operators;
  for (const auto& [name, port] : ports) {
    if (names.insert(port.op.name()).second)
      operators.push_back(port.op);
  }
  holoscan::detail::OperatorTestingAccess::add_to(partition, operators);
}
}  // namespace

void NativeTestGraphView::add_sources_to(PartitionHandle& partition) const {
  add_helpers(partition, sources_);
}
void NativeTestGraphView::add_collectors_to(PartitionHandle& partition) const {
  add_helpers(partition, collectors_);
}
TensorOutputDevicePlacement NativeTestGraphView::input_source_device_placement(
    std::string_view name, DeviceId device) const {
  const auto& helper = helper_port(sources_, name);
  return {.operator_path = helper.op.name(), .output_port = helper.port, .device = device};
}
TensorOutputDevicePlacement NativeTestGraphView::output_device_placement(std::string_view name,
                                                                         DeviceId device) const {
  static_cast<void>(
      holoscan::detail::OperatorTestingAccess::port_record(op_, name, PortDirection::kOutput));
  const auto [owned_name, inserted] = placement_names_.emplace(name);
  return {.operator_path = op_.name(), .output_port = *owned_name, .device = device};
}

NativeOperatorTest::NativeOperatorTest(OperatorFactory factory)
    : state_(std::make_unique<detail::OperatorTestState>(std::move(factory))) {
  if (!state_->factory) {
    throw std::invalid_argument{"operator test requires an operator factory"};
  }
}
NativeOperatorTest::~NativeOperatorTest() = default;
NativeOperatorTest::NativeOperatorTest(NativeOperatorTest&&) noexcept = default;

detail::OperatorTestState& NativeOperatorTest::require_state() const {
  if (!state_)
    throw std::logic_error{"operator test fixture is moved-from"};
  return *state_;
}

void NativeOperatorTest::set_operator_factory(OperatorFactory factory) {
  static_cast<void>(require_state());
  if (!factory) {
    throw std::invalid_argument{"operator test requires an operator factory"};
  }
  const std::lock_guard lock(state_->gate->mutex);
  state_->require_inactive();
  state_->factory = std::move(factory);
}

void NativeOperatorTest::set_operator_configuration(
    OperatorFactory factory, std::map<std::string, CaptureRequest, std::less<>> captures) {
  static_cast<void>(require_state());
  if (!factory)
    throw std::invalid_argument{"operator test requires an operator factory"};
  const detail::ActiveAttempt active{state_->gate};
  Graph graph{"operator-test-configuration-validation"};
  const auto tested = factory(graph);
  holoscan::detail::OperatorTestingAccess::require_graph(graph, tested);
  auto validate_input = [&](const auto& name, const auto& input) {
    static_cast<void>(holoscan::detail::OperatorTestingAccess::resolve(tested,
                                                                       name,
                                                                       PortDirection::kInput,
                                                                       input->identity(),
                                                                       input->type_token(),
                                                                       input->payload_type_info()));
  };
  for (const auto& [name, input] : state_->inputs)
    validate_input(name, input);
  if (state_->grouped) {
    for (const auto& port : state_->grouped->ports)
      validate_input(port.name, port.values);
  }
  decltype(state_->captures) next;
  for (auto& [name, request] : captures) {
    if (name.empty() || !request.capture_)
      throw std::invalid_argument{"invalid capture request"};
    const auto& capture = request.capture_;
    static_cast<void>(
        holoscan::detail::OperatorTestingAccess::resolve(tested,
                                                         name,
                                                         PortDirection::kOutput,
                                                         capture->identity(),
                                                         capture->type_token(),
                                                         capture->payload_type_info()));
    next.emplace(name, std::move(request.capture_));
  }
  // Validate every retained selector against the candidate before committing any state.
  // Flow settings may precede staging, so attempt-readiness checks do not apply here.
  auto validate_selectors = [&](const auto& values, PortDirection direction, bool tensor) {
    for (const auto& [name, value] : values) {
      static_cast<void>(value);
      try {
        const auto record =
            holoscan::detail::OperatorTestingAccess::port_record(tested, name, direction);
        if (tensor)
          detail::require_tensor_port(record);
      } catch (const RuntimeError& error) {
        throw RuntimeError{error.code(), "operator test port '" + name + "': " + error.what()};
      }
    }
  };
  validate_selectors(state_->input_depths, PortDirection::kInput, false);
  validate_selectors(state_->input_specs, PortDirection::kInput, true);
  validate_selectors(state_->output_specs, PortDirection::kOutput, true);
  validate_selectors(state_->input_flows, PortDirection::kInput, false);
  validate_selectors(state_->output_flows, PortDirection::kOutput, false);
  validate_selectors(state_->source_specs, PortDirection::kInput, true);
  validate_selectors(state_->source_devices, PortDirection::kInput, true);
  validate_selectors(state_->output_devices, PortDirection::kOutput, true);
  detail::apply_port_overrides(*state_, tested);
  state_->captures.swap(next);
  state_->factory = std::move(factory);
}

std::string NativeOperatorTest::resolve_input_name(
    const std::function<std::string(const authoring::OpaqueOperatorRef&)>& select) const {
  static_cast<void>(require_state());
  const detail::ActiveAttempt active{state_->gate};
  Graph graph{"operator-test-port-resolution"};
  const auto op = state_->factory(graph);
  holoscan::detail::OperatorTestingAccess::require_graph(graph, op);
  return select(op);
}

void NativeOperatorTest::stage_input(std::string_view name, StagedInput input) {
  stage_input(name, std::move(input.script_));
}

void NativeOperatorTest::stage_input(std::string_view name,
                                     std::shared_ptr<const detail::InputScript> script) {
  static_cast<void>(require_state());
  if (name.empty() || !script) {
    throw std::invalid_argument{"operator test requires a named input and owning adapter"};
  }
  const detail::ActiveAttempt active{state_->gate};
  Graph validation_graph{"operator-test-input-validation"};
  const auto tested = state_->factory(validation_graph);
  holoscan::detail::OperatorTestingAccess::require_graph(validation_graph, tested);
  static_cast<void>(holoscan::detail::OperatorTestingAccess::resolve(tested,
                                                                     name,
                                                                     PortDirection::kInput,
                                                                     script->identity(),
                                                                     script->type_token(),
                                                                     script->payload_type_info()));
  if (state_->grouped) {
    for (const auto& port : state_->grouped->ports) {
      if (port.name == name) {
        throw std::invalid_argument{"operator test input '" + std::string(name) +
                                    "' cannot mix independent and grouped producers"};
      }
    }
  }
  const auto existing = state_->inputs.find(name);
  if (existing != state_->inputs.end()) {
    try {
      auto combined = existing->second->appended(*script);
      existing->second = std::move(combined);
    } catch (const std::invalid_argument& error) {
      throw std::invalid_argument{"operator test input '" + std::string(name) +
                                  "': " + error.what()};
    }
  } else {
    state_->inputs.emplace(std::string(name), std::move(script));
  }
}

void NativeOperatorTest::push_frame(const TestFrame& frame) {
  static_cast<void>(require_state());
  const detail::ActiveAttempt active{state_->gate};
  Graph validation_graph{"operator-test-input-validation"};
  const auto tested = state_->factory(validation_graph);
  holoscan::detail::OperatorTestingAccess::require_graph(validation_graph, tested);
  auto next = std::make_shared<detail::GroupDescription>(
      state_->grouped ? *state_->grouped : detail::GroupDescription{});
  std::set<std::string, std::less<>> names;
  std::vector<detail::GroupDescription::Publication> publications;
  publications.reserve(frame.fields.size());
  for (const auto& field : frame.fields) {
    const auto& script = field.value.script_;
    if (field.name.empty() || !script || script->size() != 1U || !names.insert(field.name).second) {
      throw std::invalid_argument{"operator test frame port '" + field.name +
                                  "' requires a unique named single value"};
    }
    static_cast<void>(
        holoscan::detail::OperatorTestingAccess::resolve(tested,
                                                         field.name,
                                                         PortDirection::kInput,
                                                         script->identity(),
                                                         script->type_token(),
                                                         script->payload_type_info()));
    if (state_->inputs.contains(field.name)) {
      throw std::invalid_argument{"operator test input '" + field.name +
                                  "' cannot mix independent and grouped producers"};
    }
    auto port = std::find_if(next->ports.begin(), next->ports.end(), [&](const auto& entry) {
      return entry.name == field.name;
    });
    if (port == next->ports.end()) {
      publications.push_back({next->ports.size(), 0U});
      next->ports.push_back({field.name, script});
    } else {
      const auto port_index = static_cast<std::size_t>(std::distance(next->ports.begin(), port));
      publications.push_back({port_index, port->values->size()});
      try {
        port->values = port->values->appended(*script);
      } catch (const std::invalid_argument& error) {
        throw std::invalid_argument{"operator test input '" + field.name + "': " + error.what()};
      }
    }
  }
  next->frames.push_back(std::move(publications));
  state_->grouped = std::move(next);
}

void NativeOperatorTest::declare_capture(std::string_view name, CaptureRequest capture) {
  declare_capture(name, std::move(capture.capture_));
}

void NativeOperatorTest::declare_capture(
    std::string_view name, std::shared_ptr<const detail::CaptureDeclaration> capture) {
  static_cast<void>(require_state());
  if (name.empty() || !capture) {
    throw std::invalid_argument{"operator test requires a named output and owning adapter"};
  }
  const detail::ActiveAttempt active{state_->gate};
  detail::validate_selector(
      *state_, name, PortDirection::kOutput, [&](const auto& op, const auto&) {
        static_cast<void>(
            holoscan::detail::OperatorTestingAccess::resolve(op,
                                                             name,
                                                             PortDirection::kOutput,
                                                             capture->identity(),
                                                             capture->type_token(),
                                                             capture->payload_type_info()));
      });
  state_->captures.insert_or_assign(std::string(name), std::move(capture));
}

void NativeOperatorTest::input_queue_depth(std::string_view name,
                                           std::optional<std::uint32_t> depth) {
  static_cast<void>(require_state());
  const detail::ActiveAttempt active{state_->gate};
  detail::validate_selector(*state_, name, PortDirection::kInput, [&](const auto& op, const auto&) {
    if (depth)
      holoscan::detail::OperatorTestingAccess::input_queue_depth(op, name, *depth);
  });
  if (depth)
    state_->input_depths.insert_or_assign(std::string(name), *depth);
  else
    state_->input_depths.erase(std::string(name));
}

void NativeOperatorTest::input_flow(std::string_view name,
                                    std::optional<ConnectionOptions> options) {
  static_cast<void>(require_state());
  const detail::ActiveAttempt active{state_->gate};
  detail::validate_selector(*state_, name, PortDirection::kInput);
  if (options && ((options->queue_depth && *options->queue_depth == 0U) ||
                  (options->max_inflight_samples && *options->max_inflight_samples == 0U)))
    throw std::invalid_argument{"operator test flow '" + std::string(name) +
                                "' bounds must be positive"};
  if (options) {
    auto owned = detail::OwnedConnectionOptions{*options};
    state_->input_flows.insert_or_assign(std::string(name), std::move(owned));
  } else {
    state_->input_flows.erase(std::string(name));
  }
}

void NativeOperatorTest::output_flow(std::string_view name,
                                     std::optional<ConnectionOptions> options) {
  static_cast<void>(require_state());
  const detail::ActiveAttempt active{state_->gate};
  detail::validate_selector(*state_, name, PortDirection::kOutput);
  if (options && ((options->queue_depth && *options->queue_depth == 0U) ||
                  (options->max_inflight_samples && *options->max_inflight_samples == 0U)))
    throw std::invalid_argument{"operator test flow '" + std::string(name) +
                                "' bounds must be positive"};
  if (options) {
    auto owned = detail::OwnedConnectionOptions{*options};
    state_->output_flows.insert_or_assign(std::string(name), std::move(owned));
  } else {
    state_->output_flows.erase(std::string(name));
  }
}

void NativeOperatorTest::input_tensor_spec(std::string_view name,
                                           std::optional<TensorInputSpec> value) {
  static_cast<void>(require_state());
  if (name.empty())
    throw std::invalid_argument{"operator test selector must not be empty"};
  const detail::ActiveAttempt active{state_->gate};
  detail::validate_selector(
      *state_, name, PortDirection::kInput, [&](const auto& op, const auto& port) {
        detail::require_tensor_port(port);
        if (value)
          holoscan::detail::OperatorTestingAccess::input_tensor_spec(op, name, *value);
      });
  if (value)
    state_->input_specs.insert_or_assign(std::string(name),
                                         detail::OwnedTensorSpec<TensorInputSpec>{*value});
  else
    state_->input_specs.erase(std::string(name));
}

void NativeOperatorTest::output_tensor_spec(std::string_view name,
                                            std::optional<TensorOutputSpec> value) {
  static_cast<void>(require_state());
  if (name.empty())
    throw std::invalid_argument{"operator test selector must not be empty"};
  const detail::ActiveAttempt active{state_->gate};
  detail::validate_selector(
      *state_, name, PortDirection::kOutput, [&](const auto& op, const auto& port) {
        detail::require_tensor_port(port);
        if (value)
          holoscan::detail::OperatorTestingAccess::output_tensor_spec(op, name, *value);
      });
  if (value)
    state_->output_specs.insert_or_assign(std::string(name),
                                          detail::OwnedTensorSpec<TensorOutputSpec>{*value});
  else
    state_->output_specs.erase(std::string(name));
}

void NativeOperatorTest::input_source_tensor_spec(std::string_view name,
                                                  std::optional<TensorOutputSpec> value) {
  static_cast<void>(require_state());
  if (name.empty())
    throw std::invalid_argument{"operator test selector must not be empty"};
  const detail::ActiveAttempt active{state_->gate};
  detail::validate_selector(
      *state_, name, PortDirection::kInput, [&](const auto&, const auto& port) {
        detail::require_tensor_port(port);
        if (value) {
          auto normalized = holoscan::detail::normalize_tensor_output_spec(*value);
          if (!normalized)
            throw RuntimeError{normalized.error()};
        }
      });
  if (value)
    state_->source_specs.insert_or_assign(std::string(name),
                                          detail::OwnedTensorSpec<TensorOutputSpec>{*value});
  else
    state_->source_specs.erase(std::string(name));
}

void NativeOperatorTest::input_source_device(std::string_view name, std::optional<DeviceId> value) {
  static_cast<void>(require_state());
  if (name.empty())
    throw std::invalid_argument{"operator test selector must not be empty"};
  const detail::ActiveAttempt active{state_->gate};
  detail::validate_selector(
      *state_, name, PortDirection::kInput, [](const auto&, const auto& port) {
        detail::require_tensor_port(port);
      });
  if (value && value->value < 0)
    throw std::invalid_argument{"operator test port '" + std::string(name) +
                                "' device must be nonnegative"};
  if (value)
    state_->source_devices.insert_or_assign(std::string(name), *value);
  else
    state_->source_devices.erase(std::string(name));
}

void NativeOperatorTest::output_device(std::string_view name, std::optional<DeviceId> value) {
  static_cast<void>(require_state());
  if (name.empty())
    throw std::invalid_argument{"operator test selector must not be empty"};
  const detail::ActiveAttempt active{state_->gate};
  detail::validate_selector(
      *state_, name, PortDirection::kOutput, [](const auto&, const auto& port) {
        detail::require_tensor_port(port);
      });
  if (value && value->value < 0)
    throw std::invalid_argument{"operator test port '" + std::string(name) +
                                "' device must be nonnegative"};
  if (value)
    state_->output_devices.insert_or_assign(std::string(name), *value);
  else
    state_->output_devices.erase(std::string(name));
}

void NativeOperatorTest::set_compile_options(CompileOptions options) {
  static_cast<void>(require_state());
  const std::lock_guard lock(state_->gate->mutex);
  state_->require_inactive();
  state_->options = std::move(options);
}

void NativeOperatorTest::configure_graph(std::function<void(NativeTestGraphView&)> hook) {
  static_cast<void>(require_state());
  const std::lock_guard lock(state_->gate->mutex);
  state_->require_inactive();
  state_->hook = std::move(hook);
}

TestResult NativeOperatorTest::run() {
  return run(TestRunOptions{});
}

TestResult NativeOperatorTest::run(TestRunOptions options) {
  return run_impl(std::nullopt, false, options).finish();
}

TestResult NativeOperatorTest::run_controlled_clock(ControlledClockRun control) {
  return run_impl(control).finish();
}

RunningFixture NativeOperatorTest::run_async(TestRunOptions options) {
  return run_impl(std::nullopt, true, options);
}

RunningFixture NativeOperatorTest::run_impl(std::optional<ControlledClockRun> control, bool observe,
                                            TestRunOptions run_options) {
  static_cast<void>(require_state());
  const auto timeout = control ? control->timeout : run_options.timeout;
  const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::time_point::max() - std::chrono::steady_clock::now());
  if (timeout.count() <= 0 || timeout >= remaining)
    throw std::invalid_argument{"operator test timeout must be positive and fit the steady clock"};
  if (!observe && run_options.frames == FrameRelease::kManual)
    throw std::invalid_argument{"manual operator test requires run_async()"};
  auto running = std::make_unique<detail::RunningFixtureState>(state_->gate);
  running->frames = detail::prepare_frame_release(*state_, run_options);
  if (control)
    detail::validate_controlled_staging(*state_);
  const auto& result = running->result;
  result->clock_requested = control.has_value();
  result->expect_stop = run_options.expect_stop;
  auto& capture_handoffs = running->captures;
  Graph graph{"operator-test"};
  const auto tested = state_->factory(graph);
  holoscan::detail::OperatorTestingAccess::require_graph(graph, tested);
  detail::apply_port_overrides(*state_, tested);
  detail::HelperPorts sources;
  detail::HelperPorts collectors;
  state_->validate_flow_selectors();
  sources = detail::build_input_sources(graph, tested, *state_, running->frames);
  collectors = detail::build_collectors(graph, tested, *state_, *result, capture_handoffs);
  if (observe) {
    std::size_t index{};
    for (const auto& [name, capture] : state_->captures) {
      auto progress = std::make_shared<detail::CaptureProgress>();
      capture_handoffs[index++].first->progress = progress;
      running->progress.emplace(name, std::move(progress));
    }
  }
  std::shared_ptr<SyntheticClock> controlled_clock;
  if (control) {
    controlled_clock = std::make_shared<SyntheticClock>(0);
    graph.set_default_clock(graph.bind_shared_clock("fixture_clock", controlled_clock));
  } else {
    graph.set_default_clock(graph.add_clock<RealtimeClock>("fixture_clock"));
  }
  auto options = state_->options;
  NativeTestGraphView view{graph, tested, options, sources, collectors};
  for (const auto& [name, device] : state_->source_devices) {
    options.deployment.bind_tensor_output_device(view.input_source_device_placement(name, device));
  }
  for (const auto& [name, device] : state_->output_devices) {
    options.deployment.bind_tensor_output_device(view.output_device_placement(name, device));
  }
  if (state_->hook)
    state_->hook(view);
  const auto graph_view = graph.view();
  if (graph_view.partitions().empty() && !graph_view.automatic_partition_remaining()) {
    graph.partition("fixture").include_all();
  }
  result->plan = compile(graph, std::move(options));
  result->diagnostics.assign(result->plan.diagnostics().begin(), result->plan.diagnostics().end());
  if (!result->plan.ok()) {
    running->complete_cold();
    return RunningFixture{std::move(running)};
  }

  for (const auto& row : result->plan.operator_parameters()) {
    const auto path = OperatorPath::parse(row.operator_path);
    if (path && path->segments().size() == 1U && path->segments().front() == tested.name()) {
      running->tested_operator = row.owner;
      break;
    }
  }

  if (control) {
    const auto input_connections = detail::clock_connections(sources);
    const auto output_connections = detail::clock_connections(collectors);
    result->clock = detail::prepare_controlled_clock(
        result->plan, tested.name(), input_connections, output_connections, *control);
  }

  detail::prepare_retained_captures(*state_, *result, capture_handoffs, tested.name());

  // After the attempt starts, never replay pending inputs, including on an exception.
  {
    const std::lock_guard consume(state_->gate->mutex);
    state_->inputs.clear();
    state_->grouped.reset();
  }

  auto& session =
      running->session.emplace(result->clock ? holoscan::detail::start_operator_test_run(
                                                   result->plan, result->clock->observation)
                                             : holoscan::run_async(result->plan));
  running->start_watchdog(timeout);
  if (control.has_value() && result->clock)
    result->clock->drive(session, *controlled_clock, *control);
  return RunningFixture{std::move(running)};
}

}  // namespace holoscan::testing
