/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "compiler/compile_internal.hpp"
#include "runtime/runtime_support_thread_control.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_test_graph.hpp"
#include "support/direct_port_binding.hpp"

namespace holoscan {
namespace {

constexpr const char* kPartitionEnvironmentName = "HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM";

[[nodiscard]] std::optional<std::string> read_partition_environment() {
#if defined(_WIN32)
  SetLastError(ERROR_SUCCESS);
  const DWORD required = GetEnvironmentVariableA(kPartitionEnvironmentName, nullptr, 0U);
  if (required == 0U) {
    return GetLastError() == ERROR_ENVVAR_NOT_FOUND ? std::nullopt
                                                    : std::optional<std::string>{std::string{}};
  }
  std::string value(required, '\0');
  const DWORD written = GetEnvironmentVariableA(kPartitionEnvironmentName, value.data(), required);
  value.resize(written);
  return value;
#else
  const char* const value = std::getenv(kPartitionEnvironmentName);
  return value != nullptr ? std::optional<std::string>{value} : std::nullopt;
#endif
}

[[nodiscard]] bool set_partition_environment(std::optional<std::string_view> value) noexcept {
#if defined(_WIN32)
  if (!value.has_value()) {
    return SetEnvironmentVariableA(kPartitionEnvironmentName, nullptr) != 0;
  }
  const std::string owned{*value};
  return SetEnvironmentVariableA(kPartitionEnvironmentName, owned.c_str()) != 0;
#else
  return value.has_value() ? setenv(kPartitionEnvironmentName, std::string(*value).c_str(), 1) == 0
                           : unsetenv(kPartitionEnvironmentName) == 0;
#endif
}

// RAII guard that saves and restores the HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM env
// variable. Used in partition-parity tests to compile plans under controlled algorithm
// selection without permanently modifying the process environment.
class ScopedPartitionEnvironment {
 public:
  explicit ScopedPartitionEnvironment(std::optional<std::string_view> value)
      : previous_(read_partition_environment()) {
    EXPECT_TRUE(set_partition_environment(value));
  }

  ScopedPartitionEnvironment(const ScopedPartitionEnvironment&) = delete;
  ScopedPartitionEnvironment& operator=(const ScopedPartitionEnvironment&) = delete;
  ScopedPartitionEnvironment(ScopedPartitionEnvironment&&) = delete;
  ScopedPartitionEnvironment& operator=(ScopedPartitionEnvironment&&) = delete;

  ~ScopedPartitionEnvironment() {
    const std::optional<std::string_view> previous =
        previous_.has_value() ? std::optional<std::string_view>{*previous_} : std::nullopt;
    static_cast<void>(set_partition_environment(previous));
  }

 private:
  std::optional<std::string> previous_;
};

class RecordingSupportThreadControl final : public detail::RuntimeSupportThreadControl {
 public:
  explicit RecordingSupportThreadControl(const std::atomic_bool* work_started) noexcept
      : work_started_(work_started) {}

  [[nodiscard]] expected<void, Error> enter(const detail::RuntimeSupportThreadRole role) override {
    const std::uint32_t role_bit = std::uint32_t{1U} << static_cast<std::uint32_t>(role);
    if (work_started_ != nullptr && !work_started_->load(std::memory_order_acquire)) {
      entered_before_work_.fetch_or(role_bit, std::memory_order_release);
    }
    entered_.fetch_or(role_bit, std::memory_order_release);
    return {};
  }

  [[nodiscard]] bool entered(const detail::RuntimeSupportThreadRole role) const noexcept {
    return (entered_.load(std::memory_order_acquire) &
            (std::uint32_t{1U} << static_cast<std::uint32_t>(role))) != 0U;
  }

  [[nodiscard]] bool entered_before_work(
      const detail::RuntimeSupportThreadRole role) const noexcept {
    return (entered_before_work_.load(std::memory_order_acquire) &
            (std::uint32_t{1U} << static_cast<std::uint32_t>(role))) != 0U;
  }

 private:
  const std::atomic_bool* work_started_{};
  std::atomic<std::uint32_t> entered_{};
  std::atomic<std::uint32_t> entered_before_work_{};
};

enum class ProcessLocalTestDeliveryMode : std::uint8_t {
  kDirectRetained,
  kHoloMQ,
};

class ChannelSupportThreadModeTest : public testing::TestWithParam<ProcessLocalTestDeliveryMode> {};

[[nodiscard]] const char* process_local_delivery_name(
    const testing::TestParamInfo<ProcessLocalTestDeliveryMode>& info) noexcept {
  return info.param == ProcessLocalTestDeliveryMode::kHoloMQ ? "HoloMQ" : "DirectRetained";
}

// Aggregates results of a run_and_observe() call for partition-parity assertions.
// Captures termination state, ordered event sequence, per-event thread IDs, diagnostic
// tokens, and the peak concurrent activation count observed during the run.
struct RunObservation {
  RunTermination termination{RunTermination::kFailed};
  std::vector<std::int32_t> events;
  std::vector<std::thread::id> event_threads;
  std::vector<std::string> diagnostic_tokens;
  std::uint32_t max_active{};
};

void expect_two_partition_local_callback_order(const RunObservation& observation) {
  auto sorted_events = observation.events;
  std::sort(sorted_events.begin(), sorted_events.end());
  EXPECT_EQ(sorted_events, (std::vector<std::int32_t>{11, 12, 13, 21, 22, 23}));

  for (const std::int32_t identity : {1, 2}) {
    const auto start =
        std::find(observation.events.begin(), observation.events.end(), identity * 10 + 1);
    const auto compute =
        std::find(observation.events.begin(), observation.events.end(), identity * 10 + 2);
    const auto stop =
        std::find(observation.events.begin(), observation.events.end(), identity * 10 + 3);
    ASSERT_NE(start, observation.events.end());
    ASSERT_NE(compute, observation.events.end());
    ASSERT_NE(stop, observation.events.end());
    EXPECT_LT(start, compute);
    EXPECT_LT(compute, stop);
  }
}

[[nodiscard]] RunObservation run_and_observe(
    const ExecutionPlan& plan, test::RuntimeProbeMode mode = test::RuntimeProbeMode::kComplete) {
  test::RuntimeProbeState::reset(mode);
  RunSession session = run_async(plan);
  if (mode == test::RuntimeProbeMode::kWaitForStop) {
    EXPECT_TRUE(test::RuntimeProbeState::wait_until_compute_enters());
    session.request_stop();
  }
  session.wait();
  std::vector<std::string> diagnostic_tokens;
  diagnostic_tokens.reserve(session.diagnostics().size());
  for (const Diagnostic& diagnostic : session.diagnostics()) {
    diagnostic_tokens.push_back(diagnostic.token);
  }
  return RunObservation{
      .termination = session.termination(),
      .events = test::RuntimeProbeState::snapshot(),
      .event_threads = test::RuntimeProbeState::snapshot_threads(),
      .diagnostic_tokens = std::move(diagnostic_tokens),
      .max_active = test::RuntimeProbeState::max_active.load(std::memory_order_relaxed),
  };
}

[[nodiscard]] std::optional<std::string_view> expected_partition_algorithm_id() {
  const auto selected = read_partition_environment();
  if (!selected.has_value() || *selected == "serial_baseline/v1") {
    return "holoscan.partition.serial_baseline";
  }
  if (*selected == "source_chain/v1") {
    return "holoscan.partition.source_chain";
  }
  return std::nullopt;
}

[[nodiscard]] std::string_view partition_environment_for_parity_test() {
  const auto selected = read_partition_environment();
  return selected.has_value() && *selected == "source_chain/v1" ? "source_chain/v1"
                                                                : "serial_baseline/v1";
}

void expect_channel_plan_partition_facts(const ExecutionPlan& plan,
                                         std::string_view expected_algorithm_id) {
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto& accepted_algorithm = plan.partitioning_plan().accepted_algorithm;
  ASSERT_TRUE(accepted_algorithm.has_value());
  if (!accepted_algorithm) {
    return;
  }
  EXPECT_EQ(accepted_algorithm->id, expected_algorithm_id);
  EXPECT_EQ(accepted_algorithm->version, 1U);
  EXPECT_EQ(plan.scheduler_plan().ready_policy.id, "holoscan.ready.deterministic_serial");
  EXPECT_EQ(plan.scheduler_plan().ready_policy.version, 1U);
  EXPECT_EQ(plan.scheduler_plan().max_concurrent_activations, 1U);
  EXPECT_FALSE(plan.scheduler_plan().activation_overlap);
}

void run_aligned_channel_plan(const ExecutionPlan& plan, test::RuntimeAlignedChannelScript script) {
  test::RuntimeAlignedChannelProbeState::reset(script);
  RunSession session = run_async(plan);
  EXPECT_TRUE(test::RuntimeAlignedChannelProbeState::wait_for_sink(1U));
  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
}

void run_window_channel_plan(const ExecutionPlan& plan, test::RuntimeWindowChannelScript script) {
  test::RuntimeWindowChannelProbeState::reset(script);
  RunSession session = run_async(plan);
  EXPECT_TRUE(test::RuntimeWindowChannelProbeState::wait_for_compute());
  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
}

enum class ChannelInputTriggerKind : std::uint8_t {
  kAny,
  kEach,
};

// Shared static-storage aggregator for OnAnyInput/OnEach channel trigger tests. Tracks
// source and sink compute counts, observed values, and the other-input selection state
// per activation. Representative test: InputAnyExecutesThroughChannelBinding.
struct ChannelInputTriggerProbe {
  static void reset() {
    source_compute_count.store(0U, std::memory_order_relaxed);
    sink_compute_count.store(0U, std::memory_order_relaxed);
    first_observed_value.store(0U, std::memory_order_relaxed);
    second_observed_value.store(0U, std::memory_order_relaxed);
    other_selection_state.store(InputState::kAbsent, std::memory_order_relaxed);
    other_selection_reason.store(SelectionReason::kNeverSeen, std::memory_order_relaxed);
    other_receive_error.store(ErrorCode::kSuccess, std::memory_order_relaxed);
    allow_each_successor.store(false, std::memory_order_relaxed);
    work_started.store(false, std::memory_order_relaxed);
  }

  static void record_work_started() noexcept {
    work_started.store(true, std::memory_order_release);
  }

  static void record_source_compute() {
    source_compute_count.fetch_add(1U, std::memory_order_release);
    cv.notify_all();
  }

  static void record_sink_compute(std::uint32_t value, InputSelection other_selection,
                                  ErrorCode other_error) {
    const std::size_t ordinal = sink_compute_count.fetch_add(1U, std::memory_order_acq_rel);
    (ordinal == 0U ? first_observed_value : second_observed_value)
        .store(value, std::memory_order_relaxed);
    other_selection_state.store(other_selection.state, std::memory_order_relaxed);
    other_selection_reason.store(other_selection.reason, std::memory_order_relaxed);
    other_receive_error.store(other_error, std::memory_order_relaxed);
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_sink(std::size_t expected) {
    std::unique_lock lock{mutex};
    return cv.wait_for(lock, std::chrono::seconds{5}, [expected] {
      return sink_compute_count.load(std::memory_order_acquire) >= expected;
    });
  }

  [[nodiscard]] static bool wait_for_source(std::size_t expected) {
    std::unique_lock lock{mutex};
    return cv.wait_for(lock, std::chrono::seconds{5}, [expected] {
      return source_compute_count.load(std::memory_order_acquire) >= expected;
    });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::atomic<std::size_t> source_compute_count{};
  static inline std::atomic<std::size_t> sink_compute_count{};
  static inline std::atomic<std::uint32_t> first_observed_value{};
  static inline std::atomic<std::uint32_t> second_observed_value{};
  static inline std::atomic<InputState> other_selection_state{InputState::kAbsent};
  static inline std::atomic<SelectionReason> other_selection_reason{SelectionReason::kNeverSeen};
  static inline std::atomic<ErrorCode> other_receive_error{ErrorCode::kSuccess};
  static inline std::atomic<bool> allow_each_successor{};
  static inline std::atomic_bool work_started{};
};

// Clock-triggered source (1 ms period) that emits to 'first' or 'second' depending on kind.
// For kAny: emits once to 'second'. For kEach: emits to 'first', then to 'first' again
// after allow_each_successor is set. Used by InputAnyExecutesThroughChannelBinding.
class ChannelInputTriggerSource final : public Operator<> {
 public:
  explicit ChannelInputTriggerSource(ChannelInputTriggerKind kind) : kind_(kind) {}

  void setup(OperatorSpec& spec) override {
    spec.output(first, "first").max_emits_per_compute(1U);
    spec.output(second, "second").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    ChannelInputTriggerProbe::record_work_started();
    if (kind_ == ChannelInputTriggerKind::kAny) {
      if (step_ != 0U) {
        return {};
      }
      step_ = 1U;
      auto emitted = second.emit(22U);
      if (emitted) {
        ChannelInputTriggerProbe::record_source_compute();
      }
      return emitted;
    }
    if (step_ == 0U) {
      step_ = 1U;
      auto emitted = first.emit(11U);
      if (emitted) {
        ChannelInputTriggerProbe::record_source_compute();
      }
      return emitted;
    }
    if (step_ != 1U ||
        !ChannelInputTriggerProbe::allow_each_successor.load(std::memory_order_acquire)) {
      return {};
    }
    step_ = 2U;
    auto emitted = first.emit(12U);
    if (emitted) {
      ChannelInputTriggerProbe::record_source_compute();
    }
    return emitted;
  }

  Output<std::uint32_t> first;
  Output<std::uint32_t> second;

 private:
  ChannelInputTriggerKind kind_;
  std::size_t step_{};
};

class ChannelInputTriggerSink final : public Operator<> {
 public:
  explicit ChannelInputTriggerSink(ChannelInputTriggerKind kind) : kind_(kind) {}

  void setup(OperatorSpec& spec) override {
    spec.input(first, "first").queue_depth(2U);
    spec.input(second, "second").queue_depth(2U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    if (kind_ == ChannelInputTriggerKind::kAny) {
      result.trigger(OnAnyInput{});
    } else {
      result.trigger(OnEach{first});
    }
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    ChannelInputTriggerProbe::record_work_started();
    Input<std::uint32_t>& selected = kind_ == ChannelInputTriggerKind::kAny ? second : first;
    Input<std::uint32_t>& other = kind_ == ChannelInputTriggerKind::kAny ? first : second;
    auto received = selected.receive_data();
    if (!received) {
      return make_unexpected(std::move(received.error()));
    }
    auto other_received = other.receive_data();
    if (other_received || other_received.error().code != ErrorCode::kInputNotSelected) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    ChannelInputTriggerProbe::record_sink_compute(
        *received, other.selection(), other_received.error().code);
    return {};
  }

  Input<std::uint32_t> first;
  Input<std::uint32_t> second;

 private:
  ChannelInputTriggerKind kind_;
};

//   source(first)  --[depth=2]--> sink(first)
//   source(second) --[depth=2]--> sink(second)
//   Trigger: OnAnyInput (kind=kAny) or OnEach(first) (kind=kEach).
[[nodiscard]] ExecutionPlan make_channel_input_trigger_plan(
    ChannelInputTriggerKind kind, const bool force_holomq_process_local_inline = false) {
  Graph graph{"channel-input-trigger"};
  const auto source = graph.op<ChannelInputTriggerSource>("source", kind);
  const auto sink = graph.op<ChannelInputTriggerSink>("sink", kind);
  graph.add_flow(source->first, sink->first, ConnectionOptions{.queue_depth = 2U});
  graph.add_flow(source->second, sink->second, ConnectionOptions{.queue_depth = 2U});
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("channel-input-trigger-clock");
  graph.set_default_clock(default_clock);
  return force_holomq_process_local_inline
             ? detail::compile_with_process_local_holomq(graph, CompileOptions{})
             : compile(graph);
}

void expect_channel_input_trigger_execution(ChannelInputTriggerKind kind) {
  ChannelInputTriggerProbe::reset();
  const ExecutionPlan plan = make_channel_input_trigger_plan(kind);
  // Trigger-only OnAnyInput and OnEach pin the arrival witness directly. They must not synthesize
  // an explicit selector transaction or TemporalInputCache for the trigger input.
  EXPECT_TRUE(plan.input_selections().empty()) << plan.json();
  const auto expected_algorithm = expected_partition_algorithm_id();
  ASSERT_TRUE(expected_algorithm.has_value());
  const std::string_view expected_algorithm_id = expected_algorithm.value_or(std::string_view{});
  expect_channel_plan_partition_facts(plan, expected_algorithm_id);

  const ScopedPartitionEnvironment changed_after_compile{
      std::string_view{"definitely-not-an-algorithm"}};
  RunSession session = run_async(plan);
  if (kind == ChannelInputTriggerKind::kEach) {
    ASSERT_TRUE(ChannelInputTriggerProbe::wait_for_sink(1U));
    EXPECT_EQ(ChannelInputTriggerProbe::source_compute_count.load(std::memory_order_relaxed), 1U);
    ChannelInputTriggerProbe::allow_each_successor.store(true, std::memory_order_release);
  }
  const std::size_t expected_callbacks = kind == ChannelInputTriggerKind::kAny ? 1U : 2U;
  const bool observed = ChannelInputTriggerProbe::wait_for_sink(expected_callbacks);
  session.request_stop();
  session.wait();

  std::string diagnostic_tokens;
  for (const Diagnostic& diagnostic : session.diagnostics()) {
    if (!diagnostic_tokens.empty()) {
      diagnostic_tokens += ',';
    }
    diagnostic_tokens += diagnostic.token;
    diagnostic_tokens += ":vertex=";
    diagnostic_tokens += std::to_string(diagnostic.source.vertex.value);
  }

  ASSERT_TRUE(observed)
      << "source publications="
      << ChannelInputTriggerProbe::source_compute_count.load(std::memory_order_relaxed)
      << ", sink callbacks="
      << ChannelInputTriggerProbe::sink_compute_count.load(std::memory_order_relaxed)
      << ", diagnostics=" << diagnostic_tokens;
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(ChannelInputTriggerProbe::source_compute_count.load(std::memory_order_relaxed),
            expected_callbacks);
  EXPECT_EQ(ChannelInputTriggerProbe::sink_compute_count.load(std::memory_order_relaxed),
            expected_callbacks);
  EXPECT_EQ(ChannelInputTriggerProbe::first_observed_value.load(std::memory_order_relaxed),
            kind == ChannelInputTriggerKind::kAny ? 22U : 11U);
  if (kind == ChannelInputTriggerKind::kEach) {
    EXPECT_EQ(ChannelInputTriggerProbe::second_observed_value.load(std::memory_order_relaxed), 12U);
  }
  EXPECT_EQ(ChannelInputTriggerProbe::other_selection_state.load(std::memory_order_relaxed),
            InputState::kNotSelected);
  EXPECT_EQ(ChannelInputTriggerProbe::other_selection_reason.load(std::memory_order_relaxed),
            SelectionReason::kNotSelectedForActivation);
  EXPECT_EQ(ChannelInputTriggerProbe::other_receive_error.load(std::memory_order_relaxed),
            ErrorCode::kInputNotSelected);
}

enum class ChannelScalarSelectorKind : std::uint8_t {
  kExact,
  kLatestWithin,
};

// Shared static-storage aggregator for scalar-selector parity tests. Records the driver
// and candidate sequences and capture timestamps to verify selector correctness across
// queue policies. Representative test: ScalarQueuePolicyDoesNotRedefineSelectionAndFreshRunResets.
struct ChannelScalarSelectorProbe {
  static void reset() {
    const std::lock_guard lock{mutex};
    observed = false;
    source_publication_count.store(0U, std::memory_order_relaxed);
    value = 0U;
    driver_sequence = 0U;
    selected_sequence = 0U;
    driver_capture_ns = 0;
    selected_capture_ns = 0;
  }

  static void record_source_publication() noexcept {
    source_publication_count.fetch_add(1U, std::memory_order_release);
  }

  static void record(const Sample<std::uint32_t>& driver, const Sample<std::uint32_t>& selected) {
    const std::lock_guard lock{mutex};
    observed = true;
    value = selected.data;
    driver_sequence = driver.metadata.sequence;
    selected_sequence = selected.metadata.sequence;
    driver_capture_ns = driver.metadata.capture_timestamp_ns;
    selected_capture_ns = selected.metadata.capture_timestamp_ns;
    cv.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock{mutex};
    return cv.wait_for(lock, std::chrono::seconds{5}, [] { return observed; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline bool observed{};
  static inline std::atomic<std::size_t> source_publication_count{};
  static inline std::uint32_t value{};
  static inline std::uint64_t driver_sequence{};
  static inline std::uint64_t selected_sequence{};
  static inline std::int64_t driver_capture_ns{};
  static inline std::int64_t selected_capture_ns{};
};

// Single-shot source (30 s period) that emits candidate (value=71) and driver (value=1)
// at the same capture timestamp to avoid a retrograde TagTime violation when OnPredicate
// evaluates both peers against the frozen clock witness.
// Used by ScalarQueuePolicyDoesNotRedefineSelectionAndFreshRunResets.
class ChannelScalarSelectorSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(candidate, "candidate").max_emits_per_compute(1U);
    spec.output(driver, "driver").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::seconds{30}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (emitted_) {
      return {};
    }
    emitted_ = true;
    const TimePoint activation = context.activation_time();
    if (!activation.valid()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    // Keep the selected peers at the source activation coordinate. An absolute
    // fixture-local value would move backwards behind the already-frozen
    // OnClock witness and correctly fail the run's monotonic TagTime contract
    // before either selector can be exercised.
    const std::int64_t equal_capture_time_ns = activation.timestamp_ns;
    auto result = candidate.emit(Sample<std::uint32_t>{
        71U,
        SampleMetadata{.capture_timestamp_ns = equal_capture_time_ns,
                       .source_clock_id = activation.clock_id},
    });
    if (!result) {
      return result;
    }
    auto driver_result = driver.emit(Sample<std::uint32_t>{
        1U,
        SampleMetadata{.capture_timestamp_ns = equal_capture_time_ns,
                       .source_clock_id = activation.clock_id},
    });
    if (driver_result) {
      ChannelScalarSelectorProbe::record_source_publication();
    }
    return driver_result;
  }

  Output<std::uint32_t> candidate;
  Output<std::uint32_t> driver;

 private:
  bool emitted_{};
};

// Predicate-triggered sink that selects 'candidate' via Exact or LatestWithin(1 s) based
// on the selector kind. Trigger fires on every true evaluation against the driver update.
// Used by ScalarQueuePolicyDoesNotRedefineSelectionAndFreshRunResets.
class ChannelScalarSelectorSink final : public Operator<> {
 public:
  explicit ChannelScalarSelectorSink(ChannelScalarSelectorKind selector) : selector_(selector) {}

  void setup(OperatorSpec& spec) override {
    spec.input(candidate, "candidate").queue_depth(2U);
    spec.input(driver, "driver").queue_depth(2U);
  }

  [[nodiscard]] Contract contract() const override {
    auto evaluator =
        predicate("channel-scalar-selector",
                  predicate_inputs(candidate),
                  [](const PredicateEvaluationContext&,
                     const PredicateInputView<std::uint32_t>&,
                     const PredicateInputView<std::uint32_t>&) noexcept { return true; });
    Contract result;
    result
        .trigger(OnPredicate{
            .evaluate_on = OnInputUpdate{driver},
            .predicate = std::move(evaluator),
            .firing = PredicateFiring::kOnEveryTrueEvaluation,
        })
        .input(driver, Exact{});
    if (selector_ == ChannelScalarSelectorKind::kExact) {
      result.input(candidate, Exact{});
    } else {
      result.input(candidate, LatestWithin{.max_age = std::chrono::seconds{1}});
    }
    result.on_missing(candidate, Skip{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto selected_driver = driver.receive();
    if (!selected_driver) {
      return make_unexpected(std::move(selected_driver.error()));
    }
    auto selected_candidate = candidate.receive();
    if (!selected_candidate) {
      return make_unexpected(std::move(selected_candidate.error()));
    }
    ChannelScalarSelectorProbe::record(*selected_driver, *selected_candidate);
    return {};
  }

  Input<std::uint32_t> candidate;
  Input<std::uint32_t> driver;

 private:
  ChannelScalarSelectorKind selector_;
};

//   source(candidate) --[queue_policy, depth=2]--> sink(candidate)  [Exact or LatestWithin]
//   source(driver)    --[depth=2]                --> sink(driver)    [Exact]
//   Trigger: OnPredicate(evaluate_on=OnInputUpdate{driver}, firing=kOnEveryTrueEvaluation).
[[nodiscard]] ExecutionPlan make_channel_scalar_selector_plan(QueuePolicy queue_policy,
                                                              ChannelScalarSelectorKind selector) {
  Graph graph{"channel-scalar-selector"};
  const auto source = graph.op<ChannelScalarSelectorSource>("source");
  const auto sink = graph.op<ChannelScalarSelectorSink>("sink", selector);
  graph.add_flow(source->candidate,
                 sink->candidate,
                 ConnectionOptions{.queue_policy = queue_policy, .queue_depth = 2U});
  graph.add_flow(source->driver, sink->driver, ConnectionOptions{.queue_depth = 2U});
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("channel-scalar-selector-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

// REQ: RUNTIME-002
TEST(DispatcherTest, EnvironmentGuardRestoresAbsentPresentAndEmptyStates) {
  // Key setup: "outer" is the starting state; nested scopes modify it to absent, empty, "inner".
  const ScopedPartitionEnvironment restore_process_state{std::string_view{"outer"}};
  ASSERT_EQ(read_partition_environment(), std::optional<std::string>{"outer"});

  {
    const ScopedPartitionEnvironment absent{std::nullopt};
    EXPECT_FALSE(read_partition_environment().has_value());
  }
  EXPECT_EQ(read_partition_environment(), std::optional<std::string>{"outer"});

  {
    const ScopedPartitionEnvironment empty{std::string_view{}};
    EXPECT_EQ(read_partition_environment(), std::optional<std::string>{std::string{}});
    {
      const ScopedPartitionEnvironment present{std::string_view{"inner"}};
      EXPECT_EQ(read_partition_environment(), std::optional<std::string>{"inner"});
    }
    EXPECT_EQ(read_partition_environment(), std::optional<std::string>{std::string{}});
  }
  // Key assertion: each scope restores its predecessor's exact state on destruction.
  EXPECT_EQ(read_partition_environment(), std::optional<std::string>{"outer"});
}

// REQ: RUNTIME-002
TEST(DispatcherTest, RuntimeDoesNotRereadPartitionEnvironment) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan(true));
  test::RuntimeProbeState::reset();

  {
    // Key setup: env changed to an invalid algorithm name after the plan is compiled.
    const ScopedPartitionEnvironment changed_after_compile{
        std::string_view{"definitely-not-an-algorithm"}};
    RunSession session = run_async(plan);
    session.wait();
    EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  }
  // Key assertion: runtime uses frozen plan facts; the env change has no effect on execution.
  const RunObservation observation{
      .termination = RunTermination::kCompleted,
      .events = test::RuntimeProbeState::snapshot(),
      .event_threads = test::RuntimeProbeState::snapshot_threads(),
      .max_active = test::RuntimeProbeState::max_active.load(std::memory_order_relaxed),
  };
  expect_two_partition_local_callback_order(observation);
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-006
// REQ: RUNTIME-011
// GoogleTest fatal assertions terminate these partition-parity tests before optional access, but
// clang-tidy does not model the macro's control flow.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
TEST(DispatcherTest, AcceptedBaselineAndSourceChainPlansRunThroughTheSameSerialLane) {
  struct Observation {
    std::uint32_t first{};
    std::uint32_t second{};
    std::size_t source_computes{};
    std::size_t sink_computes{};
  };
  std::array<Observation, 2U> observations{};
  constexpr std::array algorithms{std::string_view{"serial_baseline/v1"},
                                  std::string_view{"source_chain/v1"}};

  for (std::size_t index = 0U; index < algorithms.size(); ++index) {
    const ExecutionPlan plan = [&] {
      const ScopedPartitionEnvironment selected{algorithms[index]};
      return test::make_runtime_aligned_channel_plan();
    }();
    ASSERT_TRUE(plan.ok()) << plan.json();
    ASSERT_TRUE(plan.partitioning_plan().accepted_algorithm.has_value());
    EXPECT_EQ(
        plan.partitioning_plan().accepted_algorithm->id,
        index == 0U ? "holoscan.partition.serial_baseline" : "holoscan.partition.source_chain");
    ASSERT_EQ(plan.execution_partitions().size(), 1U);
    EXPECT_EQ(plan.scheduler_plan().max_concurrent_activations, 1U);
    EXPECT_FALSE(plan.scheduler_plan().activation_overlap);

    const ScopedPartitionEnvironment changed_after_compile{
        std::string_view{"definitely-not-an-algorithm"}};
    run_aligned_channel_plan(plan, test::RuntimeAlignedChannelScript::kExactBoundary);
    observations[index] = Observation{
        .first =
            test::RuntimeAlignedChannelProbeState::observed_first.load(std::memory_order_relaxed),
        .second =
            test::RuntimeAlignedChannelProbeState::observed_second.load(std::memory_order_relaxed),
        .source_computes = test::RuntimeAlignedChannelProbeState::source_compute_count.load(
            std::memory_order_relaxed),
        .sink_computes = test::RuntimeAlignedChannelProbeState::sink_compute_count.load(
            std::memory_order_relaxed),
    };
  }

  EXPECT_EQ(observations[0].first, observations[1].first);
  EXPECT_EQ(observations[0].second, observations[1].second);
  EXPECT_EQ(observations[0].source_computes, observations[1].source_computes);
  EXPECT_EQ(observations[0].sink_computes, observations[1].sink_computes);
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-041
TEST(DispatcherTest, AcceptedManualPartitionsRunThroughPartitionLocalSerialLanes) {
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan(true));
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_TRUE(plan.partitioning_plan().accepted_algorithm.has_value());
  EXPECT_EQ(plan.partitioning_plan().accepted_algorithm->id, "holoscan.partition.serial_baseline");
  ASSERT_EQ(plan.execution_partitions().size(), 2U);
  EXPECT_EQ(plan.scheduler_plan().max_concurrent_activations, 2U);
  EXPECT_TRUE(plan.scheduler_plan().activation_overlap);

  const ScopedPartitionEnvironment changed_after_compile{
      std::string_view{"definitely-not-an-algorithm"}};
  // Key setup: the exact manual cover is frozen before the process environment changes.
  const RunObservation observation = run_and_observe(plan);

  EXPECT_EQ(observation.termination, RunTermination::kCompleted);
  // Key assertion: each partition preserves callback order without imposing a cross-partition
  // total order. The scheduler admits overlap, but a short callback need not overlap in every run.
  expect_two_partition_local_callback_order(observation);
  EXPECT_GE(observation.max_active, 1U);
  EXPECT_LE(observation.max_active, 2U);
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-041
TEST(DispatcherTest, ManualPartitionLanesPermitInterPartitionOverlap) {
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_multi_lane_test_plan());

  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_TRUE(plan.partitioning_plan().accepted_algorithm.has_value());
  EXPECT_EQ(plan.partitioning_plan().accepted_algorithm->id, "holoscan.partition.serial_baseline");
  EXPECT_EQ(plan.scheduler_plan().max_concurrent_activations, 2U);
  EXPECT_TRUE(plan.scheduler_plan().activation_overlap);

  std::vector<std::uint32_t> operator_lanes;
  for (const auto& partition : plan.execution_partitions()) {
    operator_lanes.push_back(partition.execution_lane);
  }
  std::sort(operator_lanes.begin(), operator_lanes.end());
  operator_lanes.erase(std::unique(operator_lanes.begin(), operator_lanes.end()),
                       operator_lanes.end());
  EXPECT_GE(operator_lanes.size(), 2U);

  const ScopedPartitionEnvironment changed_after_compile{
      std::string_view{"definitely-not-an-algorithm"}};
  // Key setup: the manual cover assigns distinct lanes and admits one activation per partition.
  const RunObservation observation = run_and_observe(plan);
  EXPECT_EQ(observation.termination, RunTermination::kCompleted);
  expect_two_partition_local_callback_order(observation);
  EXPECT_GE(observation.max_active, 1U);
  EXPECT_LE(observation.max_active, 2U);
  ASSERT_EQ(observation.event_threads.size(), observation.events.size());
  // Key assertion: each partition retains one stable owner thread, and the owners are distinct.
  const auto thread_for = [&](std::int32_t event) {
    const auto found = std::find(observation.events.begin(), observation.events.end(), event);
    EXPECT_NE(found, observation.events.end());
    return found == observation.events.end()
               ? std::thread::id{}
               : observation
                     .event_threads[static_cast<std::size_t>(found - observation.events.begin())];
  };
  EXPECT_EQ(thread_for(11), thread_for(12));
  EXPECT_EQ(thread_for(12), thread_for(13));
  EXPECT_EQ(thread_for(21), thread_for(22));
  EXPECT_EQ(thread_for(22), thread_for(23));
  EXPECT_NE(thread_for(11), thread_for(21));
}

// REQ: RUNTIME-003
// REQ: RUNTIME-004
// REQ: RUNTIME-011
// NOLINTEND(bugprone-unchecked-optional-access)
TEST(DispatcherTest, ManualPartitionPreservesStopAndFailureSemantics) {
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_TRUE(plan.partitioning_plan().accepted_algorithm.has_value());
  // The assertion above establishes that the optional is engaged.
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  EXPECT_EQ(plan.partitioning_plan().accepted_algorithm->id, "holoscan.partition.serial_baseline");
  ASSERT_EQ(plan.execution_partitions().size(), 1U);

  const ScopedPartitionEnvironment changed_after_compile{
      std::string_view{"definitely-not-an-algorithm"}};
  // Key setup: exercise callback failure, exception conversion, and cooperative stop after the
  // exact manual cover has been frozen.
  for (const auto mode : {test::RuntimeProbeMode::kFail,
                          test::RuntimeProbeMode::kThrowCompute,
                          test::RuntimeProbeMode::kWaitForStop}) {
    const RunObservation observation = run_and_observe(plan, mode);
    const bool stopped = mode == test::RuntimeProbeMode::kWaitForStop;
    // Key assertion: manual partition execution preserves terminal callback and diagnostic
    // semantics independently of the post-compile environment.
    EXPECT_EQ(observation.termination,
              stopped ? RunTermination::kStopped : RunTermination::kFailed);
    EXPECT_EQ(observation.events, (std::vector<std::int32_t>{11, 12, 13}));
    if (stopped) {
      EXPECT_TRUE(observation.diagnostic_tokens.empty());
    } else {
      EXPECT_EQ(observation.diagnostic_tokens,
                (std::vector<std::string>{"RUNTIME_OPERATOR_CALLBACK_FAILED"}));
    }
    EXPECT_EQ(observation.max_active, 1U);
  }
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-007
// REQ: RUNTIME-015
TEST(ChannelPartitionPlanParityTest, AcceptedCoverFacts) {
  const ScopedPartitionEnvironment selected{partition_environment_for_parity_test()};
  const auto expected_algorithm = expected_partition_algorithm_id();
  ASSERT_TRUE(expected_algorithm.has_value())
      << "the explicit partition-parity test accepts only the two admitted EA1 covers";
  const ExecutionPlan plan = test::make_runtime_aligned_channel_plan();
  const std::string_view expected_algorithm_id = expected_algorithm.value_or(std::string_view{});
  expect_channel_plan_partition_facts(plan, expected_algorithm_id);
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-015
// REQ: RUNTIME-018
// REQ: RUNTIME-034
TEST(ChannelPartitionPlanParityTest, InputAnyExecutesThroughChannelBinding) {
  const ScopedPartitionEnvironment selected{partition_environment_for_parity_test()};
  expect_channel_input_trigger_execution(ChannelInputTriggerKind::kAny);
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-015
// REQ: RUNTIME-018
TEST(ChannelPartitionPlanParityTest, InputEachExecutesThroughChannelBinding) {
  const ScopedPartitionEnvironment selected{partition_environment_for_parity_test()};
  expect_channel_input_trigger_execution(ChannelInputTriggerKind::kEach);
}

TEST_P(ChannelSupportThreadModeTest, PrivateControlObservesRuntimeSupportThreadsBeforeWork) {
  const ScopedPartitionEnvironment selected{partition_environment_for_parity_test()};
  const bool force_holomq = GetParam() == ProcessLocalTestDeliveryMode::kHoloMQ;
  // Key setup: attach a source-tree-only control to direct and forced-HoloMQ OnAnyInput plans.
  ChannelInputTriggerProbe::reset();
  auto control =
      std::make_shared<RecordingSupportThreadControl>(&ChannelInputTriggerProbe::work_started);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_support_thread_control(
      make_channel_input_trigger_plan(ChannelInputTriggerKind::kAny, force_holomq), control);

  RunSession session = run_async(plan);
  const bool observed = ChannelInputTriggerProbe::wait_for_sink(1U);
  session.request_stop();
  session.wait();

  EXPECT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_TRUE(control->entered(detail::RuntimeSupportThreadRole::kController));
  EXPECT_TRUE(control->entered_before_work(detail::RuntimeSupportThreadRole::kController));
  // Key assertion: direct retention needs no pump; the HoloMQ fallback enters both support roles.
  EXPECT_EQ(control->entered(detail::RuntimeSupportThreadRole::kChannelPump), force_holomq);
  EXPECT_EQ(control->entered_before_work(detail::RuntimeSupportThreadRole::kChannelPump),
            force_holomq);
}

INSTANTIATE_TEST_SUITE_P(ChannelDelivery, ChannelSupportThreadModeTest,
                         testing::Values(ProcessLocalTestDeliveryMode::kDirectRetained,
                                         ProcessLocalTestDeliveryMode::kHoloMQ),
                         process_local_delivery_name);

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-018
// REQ: RUNTIME-034
// REQ: RUNTIME-019
// REQ: RUNTIME-037
// REQ: CHAN-100
// REQ: RUNTIME-043
TEST(ChannelPartitionPlanParityTest, ScalarQueuePolicyDoesNotRedefineSelectionAndFreshRunResets) {
  const ScopedPartitionEnvironment selected{partition_environment_for_parity_test()};
  struct Vector {
    QueuePolicy queue_policy;
    ChannelScalarSelectorKind selector;
  };
  constexpr std::array vectors{
      Vector{QueuePolicy::kBlocking, ChannelScalarSelectorKind::kExact},
      Vector{QueuePolicy::kBlocking, ChannelScalarSelectorKind::kLatestWithin},
      Vector{QueuePolicy::kDropOldest, ChannelScalarSelectorKind::kExact},
      Vector{QueuePolicy::kDropOldest, ChannelScalarSelectorKind::kLatestWithin},
  };

  for (const Vector& vector : vectors) {
    SCOPED_TRACE(static_cast<unsigned>(vector.queue_policy));
    SCOPED_TRACE(static_cast<unsigned>(vector.selector));
    const ExecutionPlan plan =
        make_channel_scalar_selector_plan(vector.queue_policy, vector.selector);
    const auto expected_algorithm = expected_partition_algorithm_id();
    ASSERT_TRUE(expected_algorithm.has_value());
    const std::string_view expected_algorithm_id = expected_algorithm.value_or(std::string_view{});
    expect_channel_plan_partition_facts(plan, expected_algorithm_id);

    const ScopedPartitionEnvironment changed_after_compile{
        std::string_view{"definitely-not-an-algorithm"}};
    for (std::size_t run = 0U; run < 2U; ++run) {
      SCOPED_TRACE(run);
      // Key setup: probe reset between runs proves no channel state leaks across sessions.
      ChannelScalarSelectorProbe::reset();
      RunSession session = run_async(plan);
      const bool observed = ChannelScalarSelectorProbe::wait();
      session.request_stop();
      session.wait();

      std::string diagnostic_tokens;
      for (const Diagnostic& diagnostic : session.diagnostics()) {
        if (!diagnostic_tokens.empty()) {
          diagnostic_tokens += ',';
        }
        diagnostic_tokens += diagnostic.token;
        if (!diagnostic.service.empty()) {
          diagnostic_tokens += ":service=";
          diagnostic_tokens += diagnostic.service;
        }
        diagnostic_tokens += ":vertex=";
        diagnostic_tokens += std::to_string(diagnostic.source.vertex.value);
      }
      ASSERT_TRUE(observed) << "termination=" << static_cast<unsigned>(session.termination())
                            << ", source publications="
                            << ChannelScalarSelectorProbe::source_publication_count.load(
                                   std::memory_order_acquire)
                            << ", diagnostics=" << diagnostic_tokens;
      EXPECT_EQ(session.termination(), RunTermination::kStopped);
      EXPECT_TRUE(session.diagnostics().empty())
          << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
      // Key assertion: queue policy does not change selection; candidate value 71 seen both runs.
      EXPECT_EQ(
          ChannelScalarSelectorProbe::source_publication_count.load(std::memory_order_acquire), 1U);
      EXPECT_EQ(ChannelScalarSelectorProbe::value, 71U);
      EXPECT_EQ(ChannelScalarSelectorProbe::driver_sequence, 1U);
      EXPECT_EQ(ChannelScalarSelectorProbe::selected_sequence, 1U);
      EXPECT_EQ(ChannelScalarSelectorProbe::driver_capture_ns,
                ChannelScalarSelectorProbe::selected_capture_ns);
    }
  }
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-007
// REQ: RUNTIME-018
// REQ: RUNTIME-043
TEST(ChannelPartitionPlanParityTest, AllTriggersAndSelectors) {
  const ScopedPartitionEnvironment selected{partition_environment_for_parity_test()};
  const auto expected_algorithm = expected_partition_algorithm_id();
  ASSERT_TRUE(expected_algorithm.has_value())
      << "the explicit partition-parity test accepts only the two admitted EA1 covers";

  const ExecutionPlan aligned = test::make_runtime_aligned_channel_plan();
  const ExecutionPlan window = test::make_runtime_window_channel_plan();
  const ExecutionPlan predicate = test::make_runtime_input_predicate_channel_plan(
      test::RuntimeInputPredicatePattern::kFalseTrueTrue, PredicateFiring::kOnEveryTrueEvaluation);
  const ExecutionPlan any =
      test::make_runtime_input_trigger_replay_plan(test::RuntimeInputTriggerReplayKind::kAny);
  const ExecutionPlan each =
      test::make_runtime_input_trigger_replay_plan(test::RuntimeInputTriggerReplayKind::kEach);
  const std::string_view expected_algorithm_id = expected_algorithm.value_or(std::string_view{});
  expect_channel_plan_partition_facts(aligned, expected_algorithm_id);
  expect_channel_plan_partition_facts(window, expected_algorithm_id);
  expect_channel_plan_partition_facts(predicate, expected_algorithm_id);
  expect_channel_plan_partition_facts(any, expected_algorithm_id);
  expect_channel_plan_partition_facts(each, expected_algorithm_id);

  // Runtime consumes only accepted immutable plan facts. Changing the process
  // variable after compilation must not reselect a cover or change Channel behavior.
  const ScopedPartitionEnvironment changed_after_compile{
      std::string_view{"definitely-not-an-algorithm"}};

  // Key setup: four separate plan types compiled; env changed to invalid after compile.
  run_aligned_channel_plan(aligned, test::RuntimeAlignedChannelScript::kExactBoundary);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_first.load(), 1U);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_second.load(), 2U);

  run_window_channel_plan(window, test::RuntimeWindowChannelScript::kSuccessorClose);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::selection_state.load(), InputState::kSelected);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::batch_size.load(), 5U);

  test::RuntimeInputPredicateChannelState::reset();
  RunSession predicate_session = run_async(predicate);
  EXPECT_TRUE(test::RuntimeInputPredicateChannelState::wait_for_evaluator(3U));
  EXPECT_TRUE(test::RuntimeInputPredicateChannelState::wait_for_compute(2U));
  predicate_session.request_stop();
  predicate_session.wait();
  EXPECT_EQ(predicate_session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(predicate_session.diagnostics().empty());
  EXPECT_EQ(test::RuntimeInputPredicateChannelState::evaluator_count.load(), 3U);
  EXPECT_EQ(test::RuntimeInputPredicateChannelState::compute_count.load(), 2U);
  EXPECT_EQ(test::RuntimeInputPredicateChannelState::compute_driver_value.load(), 3U);
  // Key assertion: predicate pins the exact driver and additional values seen at evaluation time.
  EXPECT_EQ(test::RuntimeInputPredicateChannelState::compute_additional_value.load(), 30U);
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-007
// REQ: RUNTIME-018
TEST(ChannelPartitionPlanParityTest, CacheReadinessMetadataAndEpochs) {
  const ScopedPartitionEnvironment selected{partition_environment_for_parity_test()};
  const auto expected_algorithm = expected_partition_algorithm_id();
  ASSERT_TRUE(expected_algorithm.has_value())
      << "the explicit partition-parity test accepts only the two admitted EA1 covers";

  const ExecutionPlan aligned = test::make_runtime_aligned_channel_plan();
  const ExecutionPlan predicate = test::make_runtime_input_predicate_channel_plan(
      test::RuntimeInputPredicatePattern::kAlwaysTrue,
      PredicateFiring::kOnEveryTrueEvaluation,
      true);
  const std::string_view expected_algorithm_id = expected_algorithm.value_or(std::string_view{});
  expect_channel_plan_partition_facts(aligned, expected_algorithm_id);
  expect_channel_plan_partition_facts(predicate, expected_algorithm_id);
  ASSERT_EQ(predicate.predicate_triggers().size(), 1U);
  EXPECT_TRUE(predicate.predicate_triggers().front().has_config);
  EXPECT_EQ(predicate.predicate_triggers().front().config_schema,
            "test.runtime.input-predicate/v1");
  EXPECT_TRUE(predicate.predicate_triggers().front().config_digest.has_value());

  const ScopedPartitionEnvironment changed_after_compile{
      std::string_view{"definitely-not-an-algorithm"}};

  // Key setup: a stopped run with incomplete aligned state must not leak into a fresh session.
  // A stopped run with an incomplete aligned set must not leak pending cache
  // state into a fresh RunSession created from the same immutable plan.
  test::RuntimeAlignedChannelProbeState::reset(test::RuntimeAlignedChannelScript::kMissingHead);
  RunSession pending = run_async(aligned);
  EXPECT_TRUE(test::RuntimeAlignedChannelProbeState::wait_for_source(1U));
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::sink_compute_count.load(), 0U);
  pending.request_stop();
  pending.wait();
  EXPECT_EQ(pending.termination(), RunTermination::kStopped);
  EXPECT_TRUE(pending.diagnostics().empty());

  run_aligned_channel_plan(aligned, test::RuntimeAlignedChannelScript::kExactBoundary);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_first.load(), 1U);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_second.load(), 2U);

  // Input-predicate evaluation pins the exact driver and additional selector
  // result even though all newer candidates were already present in the queues.
  test::RuntimeInputPredicateChannelState::reset();
  RunSession predicate_session = run_async(predicate);
  EXPECT_TRUE(test::RuntimeInputPredicateChannelState::wait_for_evaluator(3U));
  EXPECT_TRUE(test::RuntimeInputPredicateChannelState::wait_for_compute(2U));
  predicate_session.request_stop();
  predicate_session.wait();
  EXPECT_EQ(predicate_session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(predicate_session.diagnostics().empty());
  EXPECT_EQ(test::RuntimeInputPredicateChannelState::evaluator_driver_sequence.load(), 3U);
  EXPECT_EQ(test::RuntimeInputPredicateChannelState::evaluator_additional_sequence.load(), 3U);
  EXPECT_EQ(test::RuntimeInputPredicateChannelState::compute_driver_sequence.load(), 3U);
  EXPECT_EQ(test::RuntimeInputPredicateChannelState::compute_additional_sequence.load(), 3U);
  EXPECT_EQ(test::RuntimeInputPredicateChannelState::compute_driver_value.load(), 3U);
  // Key assertion: fresh session sees clean state; predicate pins exact sequences from evaluation.
  EXPECT_EQ(test::RuntimeInputPredicateChannelState::compute_additional_value.load(), 30U);
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-006
// REQ: RUNTIME-011
// REQ: RUNTIME-040
// REQ: RUNTIME-041
TEST(DispatcherTest, RejectsMalformedFrozenPartitionAndSchedulerFactsBeforeCallbacks) {
  using PartitionMutation = detail::RuntimeTestPlanAccess::PartitionMutation;
  constexpr std::array mutations{
      PartitionMutation::kAcceptedAlgorithm,
      PartitionMutation::kAcceptedAlgorithmVersion,
      PartitionMutation::kRequestedAlgorithm,
      PartitionMutation::kAlgorithmMaturity,
      PartitionMutation::kInterfaceVersion,
      PartitionMutation::kDisposition,
      PartitionMutation::kSelectionSource,
      PartitionMutation::kReadyPolicy,
      PartitionMutation::kPartitionId,
      PartitionMutation::kPartitionLane,
      PartitionMutation::kPartitionMemberMissing,
      PartitionMutation::kPartitionMemberDuplicate,
      PartitionMutation::kDecisionMissing,
      PartitionMutation::kDecisionPartitionMismatch,
      PartitionMutation::kLanePartitionMismatch,
      PartitionMutation::kPlacementOrphan,
      PartitionMutation::kPlacementDuplicate,
      PartitionMutation::kPlacementPartitionMismatch,
      PartitionMutation::kProbeOrphan,
      PartitionMutation::kProbeDuplicate,
      PartitionMutation::kProbePartitionMismatch,
      PartitionMutation::kPartitionRunnerOrdinal,
  };

  // Key setup: every frozen partition, lane, placement, and probe relation is corrupted
  // independently on an otherwise valid plan.
  for (const PartitionMutation mutation : mutations) {
    SCOPED_TRACE(static_cast<unsigned>(mutation));
    test::RuntimeProbeState::reset();
    const ExecutionPlan baseline = test::enable_direct_runtime(test::make_runtime_test_plan(true));
    const ExecutionPlan plan = detail::RuntimeTestPlanAccess::mutate_partition(baseline, mutation);
    test::RuntimeProbeState::reset();

    RunSession session = run_async(plan);
    session.wait();

    // Key assertion: each mutation causes kFailed before any compute; no operator callbacks fired.
    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    const std::string_view expected_token = mutation == PartitionMutation::kPartitionRunnerOrdinal
                                                ? "RUNTIME_OPERATOR_REPLAY_MISMATCH"
                                                : "RUNTIME_PLACEMENT_BINDING_UNAVAILABLE";
    EXPECT_TRUE(test::has_runtime_diagnostic(session, expected_token));
    EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
    EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
  }
}

// REQ: RUNTIME-002
TEST(DispatcherTest, ConcurrentSessionsOwnIndependentOperators) {
  test::RuntimeProbeState::reset();
  const ExecutionPlan plan = test::enable_direct_runtime(test::make_runtime_test_plan());
  test::RuntimeProbeState::reset();

  // Key setup: two concurrent RunSessions created from the same immutable ExecutionPlan.
  RunSession first = run_async(plan);
  RunSession second = run_async(plan);
  first.wait();
  second.wait();

  EXPECT_EQ(first.termination(), RunTermination::kCompleted);
  EXPECT_EQ(second.termination(), RunTermination::kCompleted);
  // Key assertion: each session creates its own operator instances (constructed count = 2).
  EXPECT_EQ(test::RuntimeProbeState::constructed.load(std::memory_order_relaxed), 2U);
  EXPECT_EQ(test::RuntimeProbeState::clocks_constructed.load(std::memory_order_relaxed), 2U);
}

}  // namespace
}  // namespace holoscan
