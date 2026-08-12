/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

// Provides generated owning types and schema-trait specializations.
#include <holoscan/schema/flatbuffer_schema_package.hpp>  // NOLINT(misc-include-cleaner)

#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "core/predicate_input_view_internal.hpp"
#include "core/temporal_contract_internal.hpp"
#include "runtime/execution_lane.hpp"
#include "runtime/predicate_evaluator.hpp"
#include "runtime/runtime_callback_context.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_test_graph.hpp"

#include "run_session_channel_execution_test_support.hpp"

namespace holoscan::test::run_session_channel_execution {

using namespace std::chrono_literals;

namespace {

// Thread-safe probe shared between source and sink operators in round-trip
// tests. The source records expected data before emitting; the sink records
// observed data and signals the waiting test via a condition variable.
// Used by: CompiledGraphRoundTripsCustomPod* and CompiledGraphRoundTripsGeneratedSchema*
struct RuntimePayloadRoundTripProbeState {
  static void reset(RuntimePayloadRoundTripSourceBehavior configured_behavior =
                        RuntimePayloadRoundTripSourceBehavior::kEmitOnce,
                    RuntimePayloadRoundTripSinkBehavior configured_sink_behavior =
                        RuntimePayloadRoundTripSinkBehavior::kObserve) {
    const std::lock_guard lock(mutex);
    expected_pod.reset();
    observed_pod.reset();
    expected_schema.reset();
    observed_schema.reset();
    pod_observation_count = 0U;
    behavior = configured_behavior;
    sink_behavior = configured_sink_behavior;
    source_entered = false;
    callback_order = {};
    callback_identities = {};
    callback_count = 0U;
    active_callbacks = 0U;
    max_active_callbacks = 0U;
    callback_overlap = false;
    source_thread = {};
    sink_thread = {};
    second_emit_result.reset();
  }

  static void callback_enter(std::uint8_t stage) {
    {
      const std::lock_guard lock(mutex);
      if (callback_count < callback_order.size()) {
        callback_order[callback_count] = stage;
        callback_identities[callback_count] = detail::runtime_callback_identity();
      }
      ++callback_count;
      ++active_callbacks;
      max_active_callbacks = std::max(max_active_callbacks, active_callbacks);
      callback_overlap = callback_overlap || active_callbacks > 1U;
      if (stage == 1U) {
        source_entered = true;
        source_thread = std::this_thread::get_id();
      } else if (stage == 2U) {
        sink_thread = std::this_thread::get_id();
      }
    }
    cv.notify_all();
  }

  static void callback_exit() {
    const std::lock_guard lock(mutex);
    if (active_callbacks != 0U) {
      --active_callbacks;
    }
  }

  [[nodiscard]] static RuntimePayloadRoundTripSourceBehavior source_behavior() {
    const std::lock_guard lock(mutex);
    return behavior;
  }

  [[nodiscard]] static RuntimePayloadRoundTripSinkBehavior configured_sink_behavior() {
    const std::lock_guard lock(mutex);
    return sink_behavior;
  }

  [[nodiscard]] static bool wait_for_source() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return source_entered; });
  }

  [[nodiscard]] static RuntimePayloadRoundTripCallbackEvidence callback_evidence() {
    const std::lock_guard lock(mutex);
    return RuntimePayloadRoundTripCallbackEvidence{
        .order = callback_order,
        .identities = callback_identities,
        .callback_count = callback_count,
        .active_callbacks = active_callbacks,
        .max_active_callbacks = max_active_callbacks,
        .overlap = callback_overlap,
        .same_thread = source_thread != std::thread::id() && source_thread == sink_thread,
    };
  }

  static void record_second_emit(ErrorCode code) {
    const std::lock_guard lock(mutex);
    second_emit_result = code;
  }

  [[nodiscard]] static std::optional<ErrorCode> second_emit_snapshot() {
    const std::lock_guard lock(mutex);
    return second_emit_result;
  }

  static void record_expected_pod(const Sample<RuntimeChannelRoundTripPod>& sample) {
    const std::lock_guard lock(mutex);
    expected_pod = RuntimePayloadRoundTripObservation{
        .pod = sample.data,
        .metadata = sample.metadata,
    };
  }

  static void record_observed_pod(const Sample<RuntimeChannelRoundTripPod>& sample) {
    {
      const std::lock_guard lock(mutex);
      observed_pod = RuntimePayloadRoundTripObservation{
          .pod = sample.data,
          .metadata = sample.metadata,
      };
      ++pod_observation_count;
    }
    cv.notify_all();
  }

  static void record_expected_schema(const Sample<schema::StringT>& sample) {
    const std::lock_guard lock(mutex);
    expected_schema = RuntimePayloadRoundTripObservation{
        .schema_value = sample.data.data,
        .metadata = sample.metadata,
    };
  }

  static void record_observed_schema(const Sample<schema::StringT>& sample) {
    {
      const std::lock_guard lock(mutex);
      observed_schema = RuntimePayloadRoundTripObservation{
          .schema_value = sample.data.data,
          .metadata = sample.metadata,
      };
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_pod() { return wait_for_pod_count(1U); }

  [[nodiscard]] static bool wait_for_pod_count(std::uint32_t count) {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [count] { return pod_observation_count >= count; });
  }

  [[nodiscard]] static bool wait_for_schema() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return observed_schema.has_value(); });
  }

  [[nodiscard]] static std::optional<RuntimePayloadRoundTripObservation> expected_pod_snapshot() {
    const std::lock_guard lock(mutex);
    return expected_pod;
  }

  [[nodiscard]] static std::optional<RuntimePayloadRoundTripObservation> observed_pod_snapshot() {
    const std::lock_guard lock(mutex);
    return observed_pod;
  }

  [[nodiscard]] static std::optional<RuntimePayloadRoundTripObservation>
  expected_schema_snapshot() {
    const std::lock_guard lock(mutex);
    return expected_schema;
  }

  [[nodiscard]] static std::optional<RuntimePayloadRoundTripObservation>
  observed_schema_snapshot() {
    const std::lock_guard lock(mutex);
    return observed_schema;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::optional<RuntimePayloadRoundTripObservation> expected_pod;
  static inline std::optional<RuntimePayloadRoundTripObservation> observed_pod;
  static inline std::optional<RuntimePayloadRoundTripObservation> expected_schema;
  static inline std::optional<RuntimePayloadRoundTripObservation> observed_schema;
  static inline std::uint32_t pod_observation_count{};
  static inline RuntimePayloadRoundTripSourceBehavior behavior{
      RuntimePayloadRoundTripSourceBehavior::kEmitOnce};
  static inline RuntimePayloadRoundTripSinkBehavior sink_behavior{
      RuntimePayloadRoundTripSinkBehavior::kObserve};
  static inline bool source_entered{};
  static inline std::array<std::uint8_t, 2U> callback_order{};
  static inline std::array<detail::RuntimeCallbackIdentity, 2U> callback_identities{};
  static inline std::uint32_t callback_count{};
  static inline std::uint32_t active_callbacks{};
  static inline std::uint32_t max_active_callbacks{};
  static inline bool callback_overlap{};
  static inline std::thread::id source_thread;
  static inline std::thread::id sink_thread;
  static inline std::optional<ErrorCode> second_emit_result;
};

class RuntimePayloadRoundTripCallbackScope final {
 public:
  explicit RuntimePayloadRoundTripCallbackScope(std::uint8_t stage) {
    RuntimePayloadRoundTripProbeState::callback_enter(stage);
  }
  RuntimePayloadRoundTripCallbackScope(const RuntimePayloadRoundTripCallbackScope&) = delete;
  RuntimePayloadRoundTripCallbackScope& operator=(const RuntimePayloadRoundTripCallbackScope&) =
      delete;
  ~RuntimePayloadRoundTripCallbackScope() { RuntimePayloadRoundTripProbeState::callback_exit(); }
};

// Emits one Sample<RuntimeChannelRoundTripPod>, or two across separate clock
// activations for the steady-state allocation probe, then idles. Records each
// sample before emitting so the test can compare round-trip fidelity.
// Used by: CompiledGraphRoundTripsCustomPodWithCallerSampleMetadata
class RuntimePodRoundTripSourceOperator final : public Operator<> {
 public:
  explicit RuntimePodRoundTripSourceOperator(
      RuntimePayloadRoundTripRoot root = RuntimePayloadRoundTripRoot::kClock)
      : root_(root) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output");
    switch (root_) {
      case RuntimePayloadRoundTripRoot::kClock:
        break;
      case RuntimePayloadRoundTripRoot::kNotification:
        spec.notification_source(notification_, "activation")
            .capacity(1U)
            .sender_reference_capacity(1U);
        spec.lifecycle().stage(LifecycleStage::kArm, &RuntimePodRoundTripSourceOperator::arm);
        break;
      case RuntimePayloadRoundTripRoot::kReadable:
        spec.readable_source(readable_, "activation");
        break;
      case RuntimePayloadRoundTripRoot::kSpin:
        spec.spin_source(spin_, "activation").capacity(1U);
        break;
    }
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    switch (root_) {
      case RuntimePayloadRoundTripRoot::kClock:
        break;
      case RuntimePayloadRoundTripRoot::kNotification:
        result.trigger(OnNotified{notification_});
        return result;
      case RuntimePayloadRoundTripRoot::kReadable:
        result.trigger(OnReadable{readable_});
        return result;
      case RuntimePayloadRoundTripRoot::kSpin:
        result.trigger(OnSpin{spin_});
        return result;
    }
    const auto period = RuntimePayloadRoundTripProbeState::source_behavior() ==
                                RuntimePayloadRoundTripSourceBehavior::kEmitTwiceAcrossActivations
                            ? 1ms
                            : 30s;
    result.trigger(OnClock{.period = period});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const RuntimePayloadRoundTripCallbackScope callback{1U};
    const RuntimePayloadRoundTripSourceBehavior behavior =
        RuntimePayloadRoundTripProbeState::source_behavior();
    const std::uint32_t emission_limit =
        behavior == RuntimePayloadRoundTripSourceBehavior::kEmitTwiceAcrossActivations ? 2U : 1U;
    if (emitted_count_ >= emission_limit) {
      return {};
    }
    ++emitted_count_;
    switch (behavior) {
      case RuntimePayloadRoundTripSourceBehavior::kEmitOnce:
      case RuntimePayloadRoundTripSourceBehavior::kEmitTwiceAcrossActivations:
        break;
      case RuntimePayloadRoundTripSourceBehavior::kNoEmission:
        return {};
      case RuntimePayloadRoundTripSourceBehavior::kBackpressured:
        return make_unexpected(
            Error{ErrorCode::kBackpressured, "intentional closed-island backpressure"});
      case RuntimePayloadRoundTripSourceBehavior::kTypedFailure:
        return make_unexpected(
            Error{ErrorCode::kFailure, "intentional closed-island source failure"});
      case RuntimePayloadRoundTripSourceBehavior::kException:
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
        throw std::runtime_error{"intentional closed-island source exception"};
#else
        return make_unexpected(Error{ErrorCode::kFailure});
#endif
      case RuntimePayloadRoundTripSourceBehavior::kSecondEmitHandled:
      case RuntimePayloadRoundTripSourceBehavior::kSecondEmitReturned:
        return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const TimePoint activation_time = context.activation_time();
    if (!activation_time.valid()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    Sample<RuntimeChannelRoundTripPod> sample{
        RuntimeChannelRoundTripPod{.first = 0x01020304U, .second = 0xA1B2C3D4U},
        SampleMetadata{
            .capture_timestamp_ns = activation_time.timestamp_ns,
            .source_clock_id = activation_time.clock_id,
            .trace_id = 0x13579BDFU,
            .frame_id = 0x2468ACE0U,
            .flags = SampleFlags::kDegraded,
        },
    };
    RuntimePayloadRoundTripProbeState::record_expected_pod(sample);
    return output.emit(std::move(sample));
  }

  LifecycleStatus arm(LifecycleContext& context) noexcept {
    auto sender = context.notification_sender(notification_);
    if (!sender) {
      return LifecycleStatus::kFatalFailure;
    }
    test::RuntimeNotificationProbeState::publish_sender(
        std::move(*sender), ErrorCode::kSuccess, ErrorCode::kSuccess);
    return LifecycleStatus::kOk;
  }

  Output<RuntimeChannelRoundTripPod> output;

 private:
  RuntimePayloadRoundTripRoot root_{RuntimePayloadRoundTripRoot::kClock};
  NotificationSource notification_;
  ReadableSource readable_;
  SpinSource spin_;
  std::uint32_t emitted_count_{};
};

// Receives one RuntimeChannelRoundTripPod sample, records it in
// RuntimePayloadRoundTripProbeState, and signals the waiting test thread.
// Used by: CompiledGraphRoundTripsCustomPodWithCallerSampleMetadata
class RuntimePodRoundTripSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    const RuntimePayloadRoundTripCallbackScope callback{2U};
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    RuntimePayloadRoundTripProbeState::record_observed_pod(*sample);
    switch (RuntimePayloadRoundTripProbeState::configured_sink_behavior()) {
      case RuntimePayloadRoundTripSinkBehavior::kObserve:
        return {};
      case RuntimePayloadRoundTripSinkBehavior::kTypedFailure:
        return make_unexpected(
            Error{ErrorCode::kFailure, "intentional closed-island sink failure"});
      case RuntimePayloadRoundTripSinkBehavior::kException:
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
        throw std::runtime_error{"intentional closed-island sink exception"};
#else
        return make_unexpected(Error{ErrorCode::kFailure});
#endif
    }
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }

  Input<RuntimeChannelRoundTripPod> input;
};

// Independently clocked input consumption is a side-input policy, not an exact successor
// activation. It remains on the ordinary selector path.
class RuntimePodSideInputSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 30s})
        .input(input, LatestWithin{.max_age = 30s})
        .on_missing(input, Skip{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<RuntimeChannelRoundTripPod> input;
};

// Predicate evaluation carries deferred selection and firing state that the closed-island runner
// does not own.
class RuntimePodPredicateSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    auto evaluator = predicate(
        "closed-island-negative-predicate",
        [](const PredicateEvaluationContext&,
           const PredicateInputView<RuntimeChannelRoundTripPod>&) noexcept { return true; });
    Contract result;
    result
        .trigger(
            OnPredicate{.evaluate_on = OnInputUpdate{input}, .predicate = std::move(evaluator)})
        .input(input, Exact{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<RuntimeChannelRoundTripPod> input;
};

// Deadline gating is controller-owned state and therefore excludes the otherwise exact successor.
class RuntimePodDeadlineSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input}).deadline(1ms).on_violation(OnViolation::kFailRun);
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<RuntimeChannelRoundTripPod> input;
};

// Runtime-mutable Parameter publication/pinning remains a controller-owned activation phase.
class RuntimePodMutableParamSinkOperator final : public Operator<test::RuntimeProbeParams> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<RuntimeChannelRoundTripPod> input;
};

// Emits twice into one capacity-one blocking edge. The first publication must be retained and the
// second must report finite backpressure without an implicit retry.
class RuntimePodDoubleEmitSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(2U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const RuntimePayloadRoundTripCallbackScope callback{1U};
    if (emitted_) {
      return {};
    }
    emitted_ = true;
    const TimePoint activation_time = context.activation_time();
    if (!activation_time.valid()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    Sample<RuntimeChannelRoundTripPod> sample{
        RuntimeChannelRoundTripPod{.first = 0x01020304U, .second = 0xA1B2C3D4U},
        SampleMetadata{
            .capture_timestamp_ns = activation_time.timestamp_ns,
            .source_clock_id = activation_time.clock_id,
            .trace_id = 0x13579BDFU,
            .frame_id = 0x2468ACE0U,
            .flags = SampleFlags::kDegraded,
        },
    };
    RuntimePayloadRoundTripProbeState::record_expected_pod(sample);
    auto first = output.emit(sample);
    if (!first) {
      return first;
    }
    auto second = output.emit(std::move(sample));
    const ErrorCode second_code = second ? ErrorCode::kSuccess : second.error().code;
    RuntimePayloadRoundTripProbeState::record_second_emit(second_code);
    if (RuntimePayloadRoundTripProbeState::source_behavior() ==
            RuntimePayloadRoundTripSourceBehavior::kSecondEmitReturned &&
        !second) {
      return make_unexpected(std::move(second).error());
    }
    return {};
  }

  Output<RuntimeChannelRoundTripPod> output;

 private:
  bool emitted_{};
};

// Emits one Sample<schema::StringT> on the first clock activation then idles.
// The FlatBuffers value exercises the compiler's encode/decode pipeline end-to-end.
// Used by: CompiledGraphRoundTripsGeneratedSchemaWithCallerSampleMetadata
class RuntimeSchemaRoundTripSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const RuntimePayloadRoundTripCallbackScope callback{1U};
    if (emitted_) {
      return {};
    }
    emitted_ = true;
    const TimePoint activation_time = context.activation_time();
    if (!activation_time.valid()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    schema::StringT value;
    value.data = "generated schema through compiled RunSession";
    Sample<schema::StringT> sample{
        std::move(value),
        SampleMetadata{
            .capture_timestamp_ns = activation_time.timestamp_ns,
            .source_clock_id = activation_time.clock_id,
            .trace_id = 0x10203040U,
            .frame_id = 0x50607080U,
            .flags = SampleFlags::kDegraded,
        },
    };
    RuntimePayloadRoundTripProbeState::record_expected_schema(sample);
    return output.emit(std::move(sample));
  }

  Output<schema::StringT> output;

 private:
  bool emitted_{};
};

// Receives one schema::StringT sample, records it in RuntimePayloadRoundTripProbeState,
// and signals the waiting test thread.
// Used by: CompiledGraphRoundTripsGeneratedSchemaWithCallerSampleMetadata
class RuntimeSchemaRoundTripSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    const RuntimePayloadRoundTripCallbackScope callback{2U};
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    RuntimePayloadRoundTripProbeState::record_observed_schema(*sample);
    return {};
  }

  Input<schema::StringT> input;
};

// Thread-safe observation for the public RunSession blocking-pressure regression. The source
// remains inside its first callback after observing finite pressure so the test can request stop
// while the capacity-one fanout and callback are both still live.
struct RuntimeBlockingPressureProbeState {
  static void reset(const bool fail_after_pressure = false) {
    const std::lock_guard lock(mutex);
    observed = false;
    result = ErrorCode::kSuccess;
    accepted_count = 0U;
    return_fatal.store(fail_after_pressure, std::memory_order_release);
    callback_active.store(false, std::memory_order_release);
  }

  static void record(const ErrorCode code, const std::uint32_t accepted) {
    {
      const std::lock_guard lock(mutex);
      observed = true;
      result = code;
      accepted_count = accepted;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return observed; });
  }

  [[nodiscard]] static std::pair<ErrorCode, std::uint32_t> snapshot() {
    const std::lock_guard lock(mutex);
    return {result, accepted_count};
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline bool observed{};
  static inline ErrorCode result{ErrorCode::kSuccess};
  static inline std::uint32_t accepted_count{};
  static inline std::atomic<bool> return_fatal{};
  static inline std::atomic<bool> callback_active{};
};

class RuntimeBlockingPressureSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(16U);
    spec.output(lossy_sibling, "lossy-sibling");
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    RuntimeBlockingPressureProbeState::callback_active.store(true, std::memory_order_release);
    struct CallbackExitGuard {
      ~CallbackExitGuard() {
        RuntimeBlockingPressureProbeState::callback_active.store(false, std::memory_order_release);
      }
    } const callback_exit_guard;

    std::uint32_t accepted = 0U;
    ErrorCode result = ErrorCode::kFailure;
    for (std::int32_t value = 0; value < 16; ++value) {
      auto emitted = output.emit(value);
      if (!emitted) {
        result = emitted.error().code;
        break;
      }
      ++accepted;
    }
    RuntimeBlockingPressureProbeState::record(result, accepted);
    if (RuntimeBlockingPressureProbeState::return_fatal.load(std::memory_order_acquire)) {
      return make_unexpected(
          Error{ErrorCode::kFailure, "intentional failure after bounded publication pressure"});
    }
    while (!context.cancellation().stop_requested()) {
      std::this_thread::yield();
    }
    return {};
  }

  Output<std::int32_t> output;
  Output<std::int32_t> lossy_sibling;
};

class RuntimeBlockingPressureSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(1U);
    spec.input(lossy_sibling, "lossy-sibling").queue_depth(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input}).input(lossy_sibling, Exact{}).on_missing(lossy_sibling, Skip{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto received = input.receive();
    return received ? expected<void, Error>{} : make_unexpected(std::move(received).error());
  }

  Input<std::int32_t> input;
  Input<std::int32_t> lossy_sibling;
};

// Minimal Tensor source used only to prove closed-island eligibility rejects Tensor payloads.
class RuntimeClosedIslandTensorSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorPortLayout{
            .memory_kind = MemoryKind::kHost,
            .dtype = DLDataType{kDLUInt, 8U, 1U},
            .rank = 1U,
        })
        .tensor_allocation(TensorAllocationBounds{.max_rank = 1U, .max_byte_span = 8U});
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<Tensor> output;
};

// Matching compile-only sink for the closed-island Tensor eligibility case.
class RuntimeClosedIslandTensorSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorPortLayout{
            .memory_kind = MemoryKind::kHost,
            .dtype = DLDataType{kDLUInt, 8U, 1U},
            .rank = 1U,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;
};

struct PredicateRetirementProbeState {
  static void reset() {
    source_count.store(0U, std::memory_order_relaxed);
    evaluation_count.store(0U, std::memory_order_relaxed);
    compute_count.store(0U, std::memory_order_relaxed);
    computed_driver.store(0U, std::memory_order_relaxed);
    computed_peer.store(0U, std::memory_order_relaxed);
  }

  static void record_source() {
    source_count.fetch_add(1U, std::memory_order_release);
    cv.notify_all();
  }

  static void record_evaluation() {
    evaluation_count.fetch_add(1U, std::memory_order_release);
    cv.notify_all();
  }

  static void record_compute(std::uint32_t driver, std::uint32_t peer = 0U) {
    computed_driver.store(driver, std::memory_order_relaxed);
    computed_peer.store(peer, std::memory_order_relaxed);
    compute_count.fetch_add(1U, std::memory_order_release);
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_compute() {
    std::unique_lock lock(mutex);
    return cv.wait_for(
        lock, 5s, [] { return compute_count.load(std::memory_order_acquire) != 0U; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::atomic<std::size_t> source_count{};
  static inline std::atomic<std::size_t> evaluation_count{};
  static inline std::atomic<std::size_t> compute_count{};
  static inline std::atomic<std::uint32_t> computed_driver{};
  static inline std::atomic<std::uint32_t> computed_peer{};
};

[[nodiscard]] expected<void, Error> emit_predicate_retirement_sample(Output<std::uint32_t>& output,
                                                                     std::uint32_t value,
                                                                     TimePoint capture_time) {
  return output.emit(Sample<std::uint32_t>{
      value,
      SampleMetadata{
          .capture_timestamp_ns = capture_time.timestamp_ns,
          .source_clock_id = capture_time.clock_id,
      },
  });
}

class PredicateRetirementSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(driver, "driver"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (next_value_ > 5U) {
      return {};
    }
    auto emitted = emit_predicate_retirement_sample(driver, next_value_, context.activation_time());
    if (!emitted) {
      return emitted;
    }
    ++next_value_;
    PredicateRetirementProbeState::record_source();
    return {};
  }

  Output<std::uint32_t> driver;

 private:
  std::uint32_t next_value_{1U};
};

class PredicateRetirementSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(driver, "driver").queue_depth(2U); }

  [[nodiscard]] Contract contract() const override {
    auto evaluator = predicate("predicate-retirement",
                               [](const PredicateEvaluationContext&,
                                  const PredicateInputView<std::uint32_t>& driver_view) noexcept {
                                 PredicateRetirementProbeState::record_evaluation();
                                 return driver_view.sample().data >= 5U;
                               });
    Contract result;
    result
        .trigger(OnPredicate{
            .evaluate_on = OnInputUpdate{driver},
            .predicate = std::move(evaluator),
            .firing = PredicateFiring::kOnEveryTrueEvaluation,
        })
        .input(driver, Exact{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto received = driver.receive();
    if (!received) {
      return make_unexpected(std::move(received.error()));
    }
    PredicateRetirementProbeState::record_compute(received->data);
    return {};
  }

  Input<std::uint32_t> driver;
};

class PredicateComputeSkipSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(compute_only, "compute_only");
    spec.output(driver, "driver");
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (next_value_ > 2U) {
      return {};
    }
    if (next_value_ == 2U) {
      auto emitted =
          emit_predicate_retirement_sample(compute_only, 200U, context.activation_time());
      if (!emitted) {
        return emitted;
      }
    }
    auto emitted = emit_predicate_retirement_sample(driver, next_value_, context.activation_time());
    if (!emitted) {
      return emitted;
    }
    ++next_value_;
    PredicateRetirementProbeState::record_source();
    return {};
  }

  Output<std::uint32_t> compute_only;
  Output<std::uint32_t> driver;

 private:
  std::uint32_t next_value_{1U};
};

class PredicateComputeSkipSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(driver, "driver").queue_depth(1U);
    spec.input(compute_only, "compute_only").queue_depth(1U);
  }

  [[nodiscard]] Contract contract() const override {
    auto evaluator = predicate(
        "predicate-compute-skip",
        [](const PredicateEvaluationContext&, const PredicateInputView<std::uint32_t>&) noexcept {
          PredicateRetirementProbeState::record_evaluation();
          return true;
        });
    Contract result;
    result
        .trigger(OnPredicate{
            .evaluate_on = OnInputUpdate{driver},
            .predicate = std::move(evaluator),
            .firing = PredicateFiring::kOnEveryTrueEvaluation,
        })
        .input(driver, Exact{})
        .input(compute_only, Exact{})
        .on_missing(compute_only, Skip{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto selected_driver = driver.receive();
    if (!selected_driver) {
      return make_unexpected(std::move(selected_driver.error()));
    }
    auto selected_peer = compute_only.receive();
    if (!selected_peer) {
      return make_unexpected(std::move(selected_peer.error()));
    }
    PredicateRetirementProbeState::record_compute(selected_driver->data, selected_peer->data);
    return {};
  }

  Input<std::uint32_t> driver;
  Input<std::uint32_t> compute_only;
};

class PredicateErrorDiagnosticSink final : public detail::RuntimeStructuralDiagnosticSink {
 public:
  void publish(std::span<const Diagnostic>) override {}

  void publish_runtime_details(std::span<const detail::RuntimeDiagnosticDetail> details) override {
    const std::lock_guard lock(mutex_);
    detail_sequences_.emplace_back(details.begin(), details.end());
  }

  [[nodiscard]] std::vector<std::vector<detail::RuntimeDiagnosticDetail>> snapshot() const {
    const std::lock_guard lock(mutex_);
    return detail_sequences_;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<std::vector<detail::RuntimeDiagnosticDetail>> detail_sequences_;
};

}  // namespace

// Graph topology:
//   Tensor source --[queue_depth=1]--> Tensor sink
[[nodiscard]] ExecutionPlan make_runtime_closed_island_tensor_ineligible_plan() {
  Graph graph{"runtime-closed-island-tensor-ineligible"};
  auto source = graph.op<RuntimeClosedIslandTensorSourceOperator>("source");
  auto sink = graph.op<RuntimeClosedIslandTensorSinkOperator>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

// Graph topology:
//   source --[queue_depth=1]--> sink
[[nodiscard]] ExecutionPlan make_runtime_pod_round_trip_plan(bool enable_lifecycle,
                                                             RuntimePayloadRoundTripRoot root) {
  Graph graph{"runtime-pod-round-trip"};
  auto source = graph.op<RuntimePodRoundTripSourceOperator>("source", root);
  auto sink = graph.op<RuntimePodRoundTripSinkOperator>("sink");
  if (root == RuntimePayloadRoundTripRoot::kSpin) {
    graph.partition("runtime-round-trip-spin").add(source).add(sink);
  } else if (root == RuntimePayloadRoundTripRoot::kReadable ||
             root == RuntimePayloadRoundTripRoot::kNotification) {
    graph.partition("runtime-round-trip-physical").add(source).add(sink);
  }
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  if (enable_lifecycle) {
    graph.lifecycle().allow_self_restart(source);
  }
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  if (root == RuntimePayloadRoundTripRoot::kSpin) {
    options.deployment.bind_thread_placement("runtime-round-trip-spin",
                                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  }
  if (enable_lifecycle) {
    options.lifecycle_transactions = LifecycleTransactionOptions{
        .max_pending_requests = 4U,
        .max_pending_transactions = 4U,
        .max_receipts = 8U,
        .max_fenced_operators = 8U,
        .max_candidate_generations = 8U,
        .max_events = 64U,
        .max_audit_records = 16U,
        .max_quiesce_records = 16U,
        .max_rollback_records = 32U,
    };
  }
  return compile(graph, std::move(options));
}

// Graph topology:
//   source --[queue_depth=1]--> sink
// The explicit partition boundary keeps the otherwise eligible edge on the
// ordinary cross-partition path even when both partitions share one lane.
[[nodiscard]] ExecutionPlan make_runtime_pod_cross_partition_round_trip_plan() {
  Graph graph{"runtime-pod-cross-partition-round-trip"};
  auto source = graph.op<RuntimePodRoundTripSourceOperator>("source");
  auto sink = graph.op<RuntimePodRoundTripSinkOperator>("sink");
  graph.partition("runtime-pod-source").add(source);
  graph.partition("runtime-pod-sink").add(sink);
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

template <typename Sink>
[[nodiscard]] ExecutionPlan make_runtime_pod_special_sink_plan(std::string_view graph_name) {
  Graph graph{std::string{graph_name}};
  auto source = graph.op<RuntimePodRoundTripSourceOperator>("source");
  auto sink = graph.op<Sink>("sink");
  graph.partition("runtime-pod-source").add(source);
  graph.partition("runtime-pod-special-sink").add(sink);
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_runtime_pod_mutable_param_plan() {
  Graph graph{"runtime-pod-mutable-param"};
  auto source = graph.op<RuntimePodRoundTripSourceOperator>("source");
  auto sink = graph.op<RuntimePodMutableParamSinkOperator>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  return compile(graph, std::move(options));
}

// One source output has both the otherwise eligible retained edge and a lossy sibling. The
// compiler must treat that non-retained incident edge as ambiguity, not optimize a partial fanout.
[[nodiscard]] ExecutionPlan make_runtime_pod_nonretained_sibling_plan() {
  Graph graph{"runtime-pod-nonretained-sibling"};
  auto source = graph.op<RuntimePodRoundTripSourceOperator>("source");
  auto retained = graph.op<RuntimePodRoundTripSinkOperator>("retained");
  auto lossy = graph.op<RuntimePodRoundTripSinkOperator>("lossy");
  graph.add_flow(source->output, retained->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(source->output,
                 lossy->input,
                 ConnectionOptions{
                     .queue_policy = QueuePolicy::kDropOldest,
                     .queue_depth = 2U,
                 });
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

// A graph-to-external export is another recipient of the source publication. The current
// two-Operator island deliberately rejects that mixed local/external delivery ownership.
[[nodiscard]] ExecutionPlan make_runtime_pod_external_export_ambiguity_plan() {
  Graph graph{"runtime-pod-external-export-ambiguity"};
  auto source = graph.op<RuntimePodRoundTripSourceOperator>("source");
  auto sink = graph.op<RuntimePodRoundTripSinkOperator>("sink");
  auto exported = graph.export_topic<RuntimeChannelRoundTripPod>(
      "holomq://runtime-pod-external-export-ambiguity");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(source->output, exported.input(), ConnectionOptions{.queue_depth = 1U});
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

// Graph topology:
//   two-emission source --[blocking, queue_depth=1]--> sink
[[nodiscard]] ExecutionPlan make_runtime_pod_double_emit_plan() {
  Graph graph{"runtime-pod-double-emit"};
  auto source = graph.op<RuntimePodDoubleEmitSourceOperator>("source");
  auto sink = graph.op<RuntimePodRoundTripSinkOperator>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

// Graph topology:
//   source --[queue_depth=1]--> sink
[[nodiscard]] ExecutionPlan make_runtime_schema_round_trip_plan(bool enable_lifecycle) {
  Graph graph{"runtime-schema-round-trip"};
  auto source = graph.op<RuntimeSchemaRoundTripSourceOperator>("source");
  auto sink = graph.op<RuntimeSchemaRoundTripSinkOperator>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  if (enable_lifecycle) {
    graph.lifecycle().allow_self_restart(source);
  }
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  if (enable_lifecycle) {
    options.lifecycle_transactions = LifecycleTransactionOptions{
        .max_pending_requests = 4U,
        .max_pending_transactions = 4U,
        .max_receipts = 8U,
        .max_fenced_operators = 8U,
        .max_candidate_generations = 8U,
        .max_events = 64U,
        .max_audit_records = 16U,
        .max_quiesce_records = 16U,
        .max_rollback_records = 32U,
    };
  }
  return compile(graph, std::move(options));
}

[[nodiscard]] RuntimeDirectContinuationRoundTripEvidence run_direct_continuation_round_trip(
    RuntimeDirectContinuationPayloadKind payload_kind, bool disable_elision, bool enable_lifecycle,
    bool disable_closed_island,
    std::shared_ptr<detail::RuntimeClosedIslandTestControl> closed_island_control,
    RuntimePayloadRoundTripSinkBehavior sink_behavior) {
  RuntimePayloadRoundTripProbeState::reset(RuntimePayloadRoundTripSourceBehavior::kEmitOnce,
                                           sink_behavior);
  ExecutionPlan plan = payload_kind == RuntimeDirectContinuationPayloadKind::kPod
                           ? make_runtime_pod_round_trip_plan(enable_lifecycle)
                           : make_runtime_schema_round_trip_plan(enable_lifecycle);
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  RuntimeDirectContinuationRoundTripEvidence evidence;
  const auto plan_state = detail::ExecutionPlanAccess::retain(plan);
  evidence.direct_delivery = plan_state != nullptr &&
                             plan_state->process_local_channel_services.size() == 1U &&
                             plan_state->process_local_channel_services.front().delivery_kind ==
                                 detail::ProcessLocalChannelDeliveryKind::kDirectRetained;
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  if (disable_elision) {
    plan = detail::RuntimeTestPlanAccess::disable_direct_continuation_elision_for_test(plan);
  }
  if (disable_closed_island) {
    plan = detail::RuntimeTestPlanAccess::disable_closed_island_composite_for_test(plan);
  }
  if (closed_island_control) {
    plan = detail::RuntimeTestPlanAccess::with_closed_island_test_control(
        plan, std::move(closed_island_control));
  }

  RunSession session = run_async(plan);
  if (payload_kind == RuntimeDirectContinuationPayloadKind::kPod) {
    evidence.observed = RuntimePayloadRoundTripProbeState::wait_for_pod();
    evidence.expected = RuntimePayloadRoundTripProbeState::expected_pod_snapshot();
    evidence.received = RuntimePayloadRoundTripProbeState::observed_pod_snapshot();
  } else {
    evidence.observed = RuntimePayloadRoundTripProbeState::wait_for_schema();
    evidence.expected = RuntimePayloadRoundTripProbeState::expected_schema_snapshot();
    evidence.received = RuntimePayloadRoundTripProbeState::observed_schema_snapshot();
  }
  session.request_stop();
  session.wait();
  evidence.termination = session.termination();
  evidence.diagnostic_count = session.diagnostics().size();
  evidence.callbacks = RuntimePayloadRoundTripProbeState::callback_evidence();
  evidence.counters = counters->snapshot();
  return evidence;
}

// Graph topology:
//   source(output)         --[blocking, depth=1]--> sink(input)
//   source(lossy-sibling) --[drop-oldest, depth=2]  --> sink(lossy-sibling)
// The test forces the HoloMQ fallback while the blocking output retains exactly one recipient. The
// source never publishes lossy-sibling; that edge preserves mixed-policy service materialization.
[[nodiscard]] ExecutionPlan make_runtime_blocking_pressure_plan(const std::string_view graph_name) {
  Graph graph{std::string{graph_name}};
  auto source = graph.op<RuntimeBlockingPressureSourceOperator>("source");
  auto sink = graph.op<RuntimeBlockingPressureSinkOperator>("sink");
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(source->lossy_sibling,
                 sink->lossy_sibling,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 2U});
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return detail::compile_with_process_local_holomq(graph, CompileOptions{});
}

[[nodiscard]] ExecutionPlan make_predicate_retirement_plan() {
  Graph graph{"predicate-retirement"};
  auto source = graph.op<PredicateRetirementSourceOperator>("source");
  auto sink = graph.op<PredicateRetirementSinkOperator>("sink");
  graph.add_flow(source->driver, sink->driver, ConnectionOptions{.queue_depth = 2U});
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_predicate_compute_skip_plan() {
  Graph graph{"predicate-compute-skip"};
  auto source = graph.op<PredicateComputeSkipSourceOperator>("source");
  auto sink = graph.op<PredicateComputeSkipSinkOperator>("sink");
  graph.add_flow(source->driver, sink->driver, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(source->compute_only, sink->compute_only, ConnectionOptions{.queue_depth = 1U});
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

[[nodiscard]] expected<std::vector<DirectPredicateOutcome>, Error> run_direct_predicate_oracle(
    test::RuntimeInputPredicatePattern pattern, PredicateFiring firing, bool configured,
    test::RuntimeInputPredicateMissing missing) {
  std::uint32_t candidate_state_count = 0U;
  test::RuntimeInputContractPredicateOperator declarations;
  const auto make_binding = [&]() -> PredicateBinding {
    if (configured) {
      return predicate<test::RuntimeInputPredicateTestState>(
          "runtime-input-direct-predicate",
          predicate_inputs(declarations.additional),
          RuntimeInputPredicateTestConfig{.minimum_driver_value = 2U},
          [pattern, &candidate_state_count](
              const PredicateEvaluationContext&,
              const PredicateInputView<std::uint32_t>& driver,
              const PredicateInputView<std::uint32_t>&,
              test::RuntimeInputPredicateTestState& state,
              const RuntimeInputPredicateTestConfig& config) noexcept -> expected<bool, Error> {
            ++state.evaluation_count;
            candidate_state_count = state.evaluation_count;
            return test::runtime_input_predicate_result(
                pattern, state.evaluation_count, driver.sample().data, config.minimum_driver_value);
          });
    }
    return predicate<test::RuntimeInputPredicateTestState>(
        "runtime-input-direct-predicate",
        predicate_inputs(declarations.additional),
        [pattern, &candidate_state_count](
            const PredicateEvaluationContext&,
            const PredicateInputView<std::uint32_t>& driver,
            const PredicateInputView<std::uint32_t>&,
            test::RuntimeInputPredicateTestState& state) noexcept -> expected<bool, Error> {
          ++state.evaluation_count;
          candidate_state_count = state.evaluation_count;
          return test::runtime_input_predicate_result(
              pattern, state.evaluation_count, driver.sample().data, 0U);
        });
  };

  auto storage = detail::ContractTestAccess::clone_predicate_storage(make_binding());
  if (!storage) {
    return make_unexpected(std::move(storage.error()));
  }
  detail::PredicateEvaluatorEngine<2U, 1U> engine{std::move(*storage), true, 1U, firing, 1U};
  std::vector<DirectPredicateOutcome> outcomes;
  outcomes.reserve(3U);
  constexpr std::uint64_t generation = 1U;
  const Sample<std::uint32_t> retained_additional{
      10U,
      SampleMetadata{.sequence = 1U,
                     .capture_timestamp_ns = 10,
                     .source_clock_id = ClockId{1U},
                     .trace_id = 1010U,
                     .frame_id = 2010U},
  };
  for (std::uint32_t index = 1U; index <= 3U; ++index) {
    const Sample<std::uint32_t> driver{
        index,
        SampleMetadata{.sequence = index,
                       .capture_timestamp_ns = static_cast<std::int64_t>(index * 10U),
                       .source_clock_id = ClockId{1U},
                       .trace_id = 1000U + index,
                       .frame_id = 2000U + index},
    };
    const Sample<std::uint32_t> fresh_additional{
        index * 10U,
        SampleMetadata{.sequence = index,
                       .capture_timestamp_ns = static_cast<std::int64_t>(index * 10U),
                       .source_clock_id = ClockId{1U},
                       .trace_id = 1000U + index * 10U,
                       .frame_id = 2000U + index * 10U},
    };
    const InputSelection additional_selection = [&]() {
      switch (missing) {
        case test::RuntimeInputPredicateMissing::kFresh:
          return InputSelection{InputState::kSelected, SelectionReason::kNone};
        case test::RuntimeInputPredicateMissing::kReuseLast:
          return index == 1U ? InputSelection{InputState::kSelected, SelectionReason::kNone}
                             : InputSelection{InputState::kReused, SelectionReason::kNeverSeen};
        case test::RuntimeInputPredicateMissing::kSkip:
          return InputSelection{InputState::kAbsent, SelectionReason::kNeverSeen};
        case test::RuntimeInputPredicateMissing::kSafeDefault:
          return InputSelection{InputState::kSafeDefault, SelectionReason::kNeverSeen};
      }
      return InputSelection{InputState::kNotSelected, SelectionReason::kNotSelectedForActivation};
    }();
    auto driver_view = detail::PredicateBindingAccess::input<std::uint32_t>({
        .sample = &driver,
        .selection = {InputState::kSelected, SelectionReason::kNone},
        .active_generation = &generation,
        .expected_generation = generation,
    });
    if (!driver_view) {
      return make_unexpected(std::move(driver_view.error()));
    }
    const Sample<std::uint32_t>* additional_sample = nullptr;
    if (additional_selection.state == InputState::kSelected) {
      additional_sample = &fresh_additional;
    } else if (additional_selection.state == InputState::kReused) {
      additional_sample = &retained_additional;
    }
    std::optional<PredicateInputView<std::uint32_t>> additional_view;
    if (additional_sample != nullptr) {
      auto view = detail::PredicateBindingAccess::input<std::uint32_t>({
          .sample = additional_sample,
          .selection = additional_selection,
          .active_generation = &generation,
          .expected_generation = generation,
      });
      if (!view) {
        return make_unexpected(std::move(view.error()));
      }
      additional_view.emplace(std::move(*view));
    }
    const std::array inputs{
        detail::PredicateFrozenInput{
            .port = PortId{1U},
            .selection = {InputState::kSelected, SelectionReason::kNone},
            .typed_view = &*driver_view,
        },
        detail::PredicateFrozenInput{
            .port = PortId{2U},
            .selection = additional_selection,
            .typed_view = additional_view ? &*additional_view : nullptr,
        },
    };
    auto decision = engine.evaluate(
        detail::PredicateFrozenSource{
            .source_sequence = index,
            .driver_port = PortId{1U},
            .reference_time = driver.metadata.capture_time(),
            .tag = Tag{.sequence = index, .generation = 1U},
            .tag_time = TagTime{.logical_ns = static_cast<std::int64_t>(index)},
        },
        inputs);
    if (!decision) {
      return make_unexpected(std::move(decision.error()));
    }
    outcomes.push_back(DirectPredicateOutcome{
        .disposition = decision->disposition,
        .evaluator_called = decision->evaluator_called,
        .value = decision->value,
        .firing_accepted = decision->firing_accepted,
        .prior_truth = decision->prior_truth,
        .additional_selection = additional_selection,
        .candidate_state_count = candidate_state_count,
        .evaluator_error = decision->evaluator_error,
    });
  }
  return outcomes;
}

void reset_payload_round_trip_probe(RuntimePayloadRoundTripSourceBehavior source_behavior,
                                    RuntimePayloadRoundTripSinkBehavior sink_behavior) {
  RuntimePayloadRoundTripProbeState::reset(source_behavior, sink_behavior);
}

bool wait_for_payload_source() {
  return RuntimePayloadRoundTripProbeState::wait_for_source();
}

bool wait_for_pod() {
  return RuntimePayloadRoundTripProbeState::wait_for_pod();
}

bool wait_for_pod_count(std::uint32_t count) {
  return RuntimePayloadRoundTripProbeState::wait_for_pod_count(count);
}

bool wait_for_schema() {
  return RuntimePayloadRoundTripProbeState::wait_for_schema();
}

RuntimePayloadRoundTripCallbackEvidence payload_callback_evidence() {
  return RuntimePayloadRoundTripProbeState::callback_evidence();
}

std::optional<ErrorCode> second_emit_snapshot() {
  return RuntimePayloadRoundTripProbeState::second_emit_snapshot();
}

std::optional<RuntimePayloadRoundTripObservation> expected_pod_snapshot() {
  return RuntimePayloadRoundTripProbeState::expected_pod_snapshot();
}

std::optional<RuntimePayloadRoundTripObservation> observed_pod_snapshot() {
  return RuntimePayloadRoundTripProbeState::observed_pod_snapshot();
}

std::optional<RuntimePayloadRoundTripObservation> expected_schema_snapshot() {
  return RuntimePayloadRoundTripProbeState::expected_schema_snapshot();
}

std::optional<RuntimePayloadRoundTripObservation> observed_schema_snapshot() {
  return RuntimePayloadRoundTripProbeState::observed_schema_snapshot();
}

void reset_blocking_pressure_probe(bool fail_after_pressure) {
  RuntimeBlockingPressureProbeState::reset(fail_after_pressure);
}

bool wait_for_blocking_pressure() {
  return RuntimeBlockingPressureProbeState::wait();
}

std::pair<ErrorCode, std::uint32_t> blocking_pressure_snapshot() {
  return RuntimeBlockingPressureProbeState::snapshot();
}

bool blocking_pressure_callback_active() {
  return RuntimeBlockingPressureProbeState::callback_active.load(std::memory_order_acquire);
}

void reset_predicate_retirement_probe() {
  PredicateRetirementProbeState::reset();
}

bool wait_for_predicate_retirement_compute() {
  return PredicateRetirementProbeState::wait_for_compute();
}

PredicateRetirementSnapshot predicate_retirement_snapshot() {
  return PredicateRetirementSnapshot{
      .source_count = PredicateRetirementProbeState::source_count.load(),
      .evaluation_count = PredicateRetirementProbeState::evaluation_count.load(),
      .compute_count = PredicateRetirementProbeState::compute_count.load(),
      .computed_driver = PredicateRetirementProbeState::computed_driver.load(),
      .computed_peer = PredicateRetirementProbeState::computed_peer.load(),
  };
}

ExecutionPlan make_runtime_pod_side_input_plan() {
  return make_runtime_pod_special_sink_plan<RuntimePodSideInputSinkOperator>(
      "runtime-pod-side-input");
}

ExecutionPlan make_runtime_pod_predicate_plan() {
  return make_runtime_pod_special_sink_plan<RuntimePodPredicateSinkOperator>(
      "runtime-pod-predicate");
}

ExecutionPlan make_runtime_pod_deadline_plan() {
  return make_runtime_pod_special_sink_plan<RuntimePodDeadlineSinkOperator>("runtime-pod-deadline");
}

expected<PredicateErrorScenarioEvidence, Error> run_channel_input_predicate_error_scenario() {
  holoscan::test::RuntimeInputPredicateChannelState::reset();
  const auto sink = std::make_shared<PredicateErrorDiagnosticSink>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(
      holoscan::test::make_runtime_input_predicate_channel_plan(
          holoscan::test::RuntimeInputPredicatePattern::kTrueThenError,
          PredicateFiring::kOnEveryTrueEvaluation),
      sink);
  if (!plan.ok()) {
    return make_unexpected(Error{ErrorCode::kFailure, std::string{plan.json()}});
  }
  if (plan.predicate_triggers().empty()) {
    return make_unexpected(Error{ErrorCode::kFailure, "missing predicate trigger"});
  }
  const PredicateTriggerPlanRecord trigger = plan.predicate_triggers().front();

  RunSession session = run_async(plan);
  session.wait();
  const auto diagnostics = session.diagnostics();
  return PredicateErrorScenarioEvidence{
      .termination = session.termination(),
      .evaluator_count = holoscan::test::RuntimeInputPredicateChannelState::evaluator_count.load(),
      .compute_count = holoscan::test::RuntimeInputPredicateChannelState::compute_count.load(),
      .evaluations = holoscan::test::RuntimeInputPredicateChannelState::evaluation_snapshot(),
      .diagnostics = std::vector<Diagnostic>{diagnostics.begin(), diagnostics.end()},
      .published_details = sink->snapshot(),
      .trigger = trigger,
  };
}

}  // namespace holoscan::test::run_session_channel_execution
