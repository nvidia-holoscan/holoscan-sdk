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
#include <cstdlib>
#include <memory>
#include <mutex>
#include <new>
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
#elif defined(__linux__)
#include <sched.h>
#endif

#include <gtest/gtest.h>

#include "compiler/compile_internal.hpp"
#include "runtime/runtime_partition_test_plan_access.hpp"
#include "runtime/runtime_support_thread_control.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_test_graph.hpp"
#include "support/direct_port_binding.hpp"
#include "support/host_unique_test_name.hpp"
#include "support/optional_assertions.hpp"
#include "threading/linux_thread_controller.hpp"

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

// Records which Runtime support-thread roles enter before application work starts.
// Used by PrivateControlObservesRuntimeSupportThreadsBeforeWork.
class RecordingSupportThreadControl final : public detail::RuntimeSupportThreadControl {
 public:
  explicit RecordingSupportThreadControl(const std::atomic_bool* work_started) noexcept
      : work_started_(work_started) {}

  [[nodiscard]] expected<void, Error> enter(const SupportThreadRole role) override {
    const std::uint32_t role_bit = std::uint32_t{1U} << static_cast<std::uint32_t>(role);
    if (work_started_ != nullptr && !work_started_->load(std::memory_order_acquire)) {
      entered_before_work_.fetch_or(role_bit, std::memory_order_release);
    }
    entered_.fetch_or(role_bit, std::memory_order_release);
#if defined(__linux__)
    auto snapshot = detail::LinuxThreadController::snapshot();
    if (!snapshot) {
      return make_unexpected(std::move(snapshot).error());
    }
    {
      const std::lock_guard lock(affinity_mutex_);
      observed_affinities_[static_cast<std::size_t>(role)] = std::move(snapshot->affinity.cores);
    }
#endif
    return {};
  }

  [[nodiscard]] bool entered(const SupportThreadRole role) const noexcept {
    return (entered_.load(std::memory_order_acquire) &
            (std::uint32_t{1U} << static_cast<std::uint32_t>(role))) != 0U;
  }

  [[nodiscard]] bool entered_before_work(const SupportThreadRole role) const noexcept {
    return (entered_before_work_.load(std::memory_order_acquire) &
            (std::uint32_t{1U} << static_cast<std::uint32_t>(role))) != 0U;
  }

  [[nodiscard]] std::vector<std::uint32_t> observed_affinity(const SupportThreadRole role) const {
    const std::lock_guard lock(affinity_mutex_);
    return observed_affinities_[static_cast<std::size_t>(role)];
  }

 private:
  const std::atomic_bool* work_started_{};
  std::atomic<std::uint32_t> entered_{};
  std::atomic<std::uint32_t> entered_before_work_{};
  mutable std::mutex affinity_mutex_;
  std::array<std::vector<std::uint32_t>, 2U> observed_affinities_;
};

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
class PumpAllocationFailureSupportThreadControl final : public detail::RuntimeSupportThreadControl {
 public:
  [[nodiscard]] expected<void, Error> enter(const SupportThreadRole role) override {
    if (role == SupportThreadRole::kChannelPump) {
      throw std::bad_alloc{};
    }
    return {};
  }
};
#endif

enum class ProcessLocalTestDeliveryMode : std::uint8_t {
  kDirectRetained,
  kHoloMQEventless,
  kHoloMQForcedPump,
};

// Exercises support-thread admission across direct, eventless, and forced-pump Channel modes.
// Used by PrivateControlObservesRuntimeSupportThreadsBeforeWork.
class ChannelSupportThreadModeTest : public testing::TestWithParam<ProcessLocalTestDeliveryMode> {};

[[nodiscard]] const char* process_local_delivery_name(
    const testing::TestParamInfo<ProcessLocalTestDeliveryMode>& info) noexcept {
  switch (info.param) {
    case ProcessLocalTestDeliveryMode::kDirectRetained:
      return "DirectRetained";
    case ProcessLocalTestDeliveryMode::kHoloMQEventless:
      return "HoloMQEventless";
    case ProcessLocalTestDeliveryMode::kHoloMQForcedPump:
      return "HoloMQForcedPump";
  }
  return "Unknown";
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

constexpr std::string_view kAutomaticPartitionAlgorithmId = "holoscan.partition.source_chain";

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

// Captures the owner thread for two independent automatically partitioned operators.
// Used by AutomaticSourceChainRunsIndependentPartitionsOnDistinctOwners.
struct AutomaticOwnerProbeState {
  static void reset() {
    const std::lock_guard lock{mutex};
    observed = {};
  }

  static void record(std::size_t identity) {
    {
      const std::lock_guard lock{mutex};
      if (identity < observed.size()) {
        observed[identity] = std::this_thread::get_id();
      }
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_both() {
    std::unique_lock lock{mutex};
    return cv.wait_for(lock, std::chrono::seconds{5}, [] {
      return observed[0] != std::thread::id{} && observed[1] != std::thread::id{};
    });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::array<std::thread::id, 2U> observed{};
};

// Records one clock-triggered callback owner without publishing payload data.
// Used by AutomaticSourceChainRunsIndependentPartitionsOnDistinctOwners.
class AutomaticOwnerProbeOperator final : public Operator<> {
 public:
  explicit AutomaticOwnerProbeOperator(std::size_t identity) : identity_(identity) {}

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::seconds{30}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    AutomaticOwnerProbeState::record(identity_);
    return {};
  }

 private:
  std::size_t identity_{};
};

// Graph topology:
//   first (independent)    second (independent)
[[nodiscard]] ExecutionPlan make_automatic_multi_owner_plan() {
  Graph graph{"runtime-automatic-multi-owner"};
  (void)graph.op<AutomaticOwnerProbeOperator>("first", 0U);
  (void)graph.op<AutomaticOwnerProbeOperator>("second", 1U);
  const ClockKey clock = graph.add_clock<RealtimeClock>("runtime-automatic-clock");
  graph.set_default_clock(clock);
  return compile(graph);
}

// A finite automatic fan-in fixture. The two independent source partitions use the same clock,
// then feed a third linear join->drain partition. The state records both operator ordering and
// stable partition-owner threads without relying on a manual cover. Used by
// AutomaticSourceChainFanInDrainsFinalDataBeforeNaturalTerminal.
struct AutomaticFanInProbeState {
  static void reset() {
    const std::lock_guard lock{mutex};
    source_threads = {};
    join_thread = {};
    drain_thread = {};
    source_thread_changed = {};
    join_thread_changed = false;
    drain_thread_changed = false;
    source_compute_counts = {};
    release_final_drain = false;
    joined_values.clear();
    drained_values.clear();
  }

  static void record_source(std::size_t source) {
    const auto current = std::this_thread::get_id();
    {
      const std::lock_guard lock{mutex};
      if (source < source_threads.size()) {
        if (source_threads[source] == std::thread::id{}) {
          source_threads[source] = current;
        } else {
          source_thread_changed[source] =
              source_thread_changed[source] || source_threads[source] != current;
        }
        ++source_compute_counts[source];
      }
    }
    cv.notify_all();
  }

  static void record_join(std::uint32_t value) {
    {
      const std::lock_guard lock{mutex};
      if (join_thread == std::thread::id{}) {
        join_thread = std::this_thread::get_id();
      } else {
        join_thread_changed = join_thread_changed || join_thread != std::this_thread::get_id();
      }
      joined_values.push_back(value);
    }
    cv.notify_all();
  }

  static void record_drain(std::uint32_t value) {
    std::unique_lock lock{mutex};
    if (drain_thread == std::thread::id{}) {
      drain_thread = std::this_thread::get_id();
    } else {
      drain_thread_changed = drain_thread_changed || drain_thread != std::this_thread::get_id();
    }
    drained_values.push_back(value);
    cv.notify_all();
    if (drained_values.size() >= 2U) {
      cv.wait(lock, [] { return release_final_drain; });
    }
  }

  [[nodiscard]] static bool wait_for_final_drain() {
    std::unique_lock lock{mutex};
    return cv.wait_for(lock, std::chrono::seconds{5}, [] { return drained_values.size() >= 2U; });
  }

  static void release_final_drain_callback() {
    {
      const std::lock_guard lock{mutex};
      release_final_drain = true;
    }
    cv.notify_all();
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::array<std::thread::id, 2U> source_threads{};
  static inline std::array<bool, 2U> source_thread_changed{};
  static inline std::array<std::size_t, 2U> source_compute_counts{};
  static inline std::thread::id join_thread{};
  static inline std::thread::id drain_thread{};
  static inline bool join_thread_changed{};
  static inline bool drain_thread_changed{};
  static inline bool release_final_drain{};
  static inline std::vector<std::uint32_t> joined_values{};
  static inline std::vector<std::uint32_t> drained_values{};
};

// Emits a finite source stream while recording its stable partition-owner thread.
// Used by AutomaticSourceChainFanInDrainsFinalDataBeforeNaturalTerminal.
class AutomaticFanInSourceOperator final : public Operator<> {
 public:
  AutomaticFanInSourceOperator(std::size_t source, std::uint32_t base)
      : source_(source), base_(base) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}})
        .complete_after(ComputeCalls{.limit = 2U});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const TimePoint activation = context.activation_time();
    if (!activation.valid()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    AutomaticFanInProbeState::record_source(source_);
    const auto value = base_ + emitted_++;
    return output.emit(Sample<std::uint32_t>{
        value,
        SampleMetadata{.capture_timestamp_ns = activation.timestamp_ns,
                       .source_clock_id = activation.clock_id},
    });
  }

  Output<std::uint32_t> output;

 private:
  std::size_t source_{};
  std::uint32_t base_{};
  std::uint32_t emitted_{};
};

// Aligns the two finite source streams and publishes their combined values.
// Used by AutomaticSourceChainFanInDrainsFinalDataBeforeNaturalTerminal.
class AutomaticFanInJoinOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(left, "left").queue_depth(4U);
    spec.input(right, "right").queue_depth(4U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnAligned{.ports = ports(left, right), .window = std::chrono::seconds{1}})
        .input(left, Exact{})
        .input(right, Exact{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto left_sample = left.receive();
    if (!left_sample) {
      return make_unexpected(std::move(left_sample).error());
    }
    auto right_sample = right.receive();
    if (!right_sample) {
      return make_unexpected(std::move(right_sample).error());
    }
    const std::uint32_t joined = left_sample->data * 100U + right_sample->data;
    AutomaticFanInProbeState::record_join(joined);
    return output.emit(joined);
  }

  Input<std::uint32_t> left;
  Input<std::uint32_t> right;
  Output<std::uint32_t> output;
};

// Retains the terminal join callback so natural-completion drain ordering is observable.
// Used by AutomaticSourceChainFanInDrainsFinalDataBeforeNaturalTerminal.
class AutomaticFanInDrainOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(4U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    AutomaticFanInProbeState::record_drain(sample->data);
    return {};
  }

  Input<std::uint32_t> input;
};

// Counts callbacks from operators whose embedded paths require lossless percent encoding.
// Used by RuntimeAdmitsLosslesslyEncodedPartitionOperatorPaths.
struct EncodedPartitionPathProbeState {
  static void reset() noexcept {
    callbacks.store(0U, std::memory_order_relaxed);
    first_path.store(0U, std::memory_order_relaxed);
    second_path.store(0U, std::memory_order_relaxed);
    unexpected_path.store(false, std::memory_order_relaxed);
  }
  static inline std::atomic<std::size_t> callbacks{};
  static inline std::atomic<std::size_t> first_path{};
  static inline std::atomic<std::size_t> second_path{};
  static inline std::atomic_bool unexpected_path{};
};

// Supplies a single callback at an authored path containing slash-like segments.
// Used by RuntimeAdmitsLosslesslyEncodedPartitionOperatorPaths.
class EncodedPartitionPathOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}})
        .complete_after(ComputeCalls{.limit = 1U});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    EncodedPartitionPathProbeState::callbacks.fetch_add(1U, std::memory_order_relaxed);
    if (context.operator_path() == "a/b%2Fc") {
      EncodedPartitionPathProbeState::first_path.fetch_add(1U, std::memory_order_relaxed);
    } else if (context.operator_path() == "a%2Fb/c") {
      EncodedPartitionPathProbeState::second_path.fetch_add(1U, std::memory_order_relaxed);
    } else {
      EncodedPartitionPathProbeState::unexpected_path.store(true, std::memory_order_relaxed);
    }
    return {};
  }

  Output<std::uint32_t> output;
};

// Embeds one named operator behind an imported clock for encoded-path replay coverage.
// Used by RuntimeAdmitsLosslesslyEncodedPartitionOperatorPaths.
class EncodedPartitionPathGraph final : public Graph {
 public:
  explicit EncodedPartitionPathGraph(std::string operator_name)
      : operator_name_(std::move(operator_name)) {}

  [[nodiscard]] ClockKey clock_import() const noexcept { return clock_import_; }

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    set_default_clock(clock_import_);
    (void)op<EncodedPartitionPathOperator>(operator_name_);
  }

 private:
  std::string operator_name_;
  ClockKey clock_import_;
};

// Exposes an embedded source through a Graph interface port for partition-relation mutations.
// Used by RejectsNoncanonicalPartitionAuthorityBeforeCallbacks.
class PartitionInterfaceProbeGraph final : public Graph {
 public:
  Output<std::uint32_t> output{"output"};

  [[nodiscard]] ClockKey clock_import() const noexcept { return clock_import_; }

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    set_default_clock(clock_import_);
    const auto source = op<EncodedPartitionPathOperator>("source");
    export_output(output, source->output);
  }

 private:
  ClockKey clock_import_;
};

// Graph topology:
//   root
//   +-- a/b%2Fc  (embedded graph "a", operator "b/c")
//   +-- a%2Fb/c  (embedded graph "a/b", operator "c")
[[nodiscard]] ExecutionPlan make_encoded_partition_path_plan() {
  Graph graph{"runtime-encoded-partition-paths"};
  const auto first = graph.embed<EncodedPartitionPathGraph>("a", "b/c");
  const auto second = graph.embed<EncodedPartitionPathGraph>("a/b", "c");
  const ClockKey clock = graph.add_clock<RealtimeClock>("encoded-partition-path-clock");
  graph.set_default_clock(clock);
  graph.bind_clock(first, first->clock_import(), clock);
  graph.bind_clock(second, second->clock_import(), clock);
  return compile(graph);
}

// Graph topology:
//   child/source --[exported interface output]--> sink
[[nodiscard]] ExecutionPlan make_partition_interface_probe_plan() {
  Graph graph{"runtime-partition-interface-probe"};
  const auto child = graph.embed<PartitionInterfaceProbeGraph>("child");
  const auto sink = graph.op<AutomaticFanInDrainOperator>("sink");
  graph.add_flow(child->output, sink->input);
  const ClockKey clock = graph.add_clock<RealtimeClock>("runtime-partition-interface-clock");
  graph.set_default_clock(clock);
  graph.bind_clock(child, child->clock_import(), clock);
  return compile(graph);
}

// Graph topology:
//   operator  (explicitly assigned to runtime-constrained-worker)
[[nodiscard]] ExecutionPlan make_complete_constrained_partition_probe_plan() {
  Graph graph{"runtime-complete-constrained-partition-probe"};
  const auto probe = graph.op<test::RuntimeProbeOperator>("operator", 1);
  graph.partition("runtime-constrained-worker").add(probe);
  graph.auto_partition_remaining();
  const ClockKey clock = graph.add_clock<test::RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(clock);
  CompileOptions options;
  options.deployment.bind_thread_placement("runtime-constrained-worker",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return compile(graph, std::move(options));
}

// Graph topology:
//   left  --\
//            join --> drain
//   right --/
[[nodiscard]] ExecutionPlan make_automatic_fanin_drain_plan() {
  Graph graph{"runtime-automatic-fanin-drain"};
  const auto left = graph.op<AutomaticFanInSourceOperator>("left", 0U, 11U);
  const auto right = graph.op<AutomaticFanInSourceOperator>("right", 1U, 21U);
  const auto join = graph.op<AutomaticFanInJoinOperator>("join");
  const auto drain = graph.op<AutomaticFanInDrainOperator>("drain");
  graph.add_flow(left->output, join->left, ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(right->output, join->right, ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(join->output, drain->input, ConnectionOptions{.queue_depth = 4U});
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-automatic-fanin-clock"));
  return compile(graph);
}

// A finite hybrid Channel fixture whose authored source anchor is placed exactly while the sink
// remains automatic. Each partition executes twice so thread stability is observable. Used by
// HybridExactThreadAnchorAndAutomaticRemainderCarryChannelData.
struct HybridChannelOwnerProbeState {
  static void reset() {
    const std::lock_guard lock{mutex};
    source_thread = {};
    sink_thread = {};
    source_thread_changed = false;
    sink_thread_changed = false;
    source_compute_count = 0U;
    sink_compute_count = 0U;
    observed_first = 0U;
    observed_second = 0U;
  }

  static void record_source() {
    const std::lock_guard lock{mutex};
    const auto current = std::this_thread::get_id();
    if (source_thread == std::thread::id{}) {
      source_thread = current;
    } else {
      source_thread_changed = source_thread_changed || source_thread != current;
    }
    ++source_compute_count;
    cv.notify_all();
  }

  static void record_sink(std::uint32_t first, std::uint32_t second) {
    const std::lock_guard lock{mutex};
    const auto current = std::this_thread::get_id();
    if (sink_thread == std::thread::id{}) {
      sink_thread = current;
    } else {
      sink_thread_changed = sink_thread_changed || sink_thread != current;
    }
    ++sink_compute_count;
    observed_first = first;
    observed_second = second;
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_two_sink_callbacks() {
    std::unique_lock lock{mutex};
    return cv.wait_for(lock, std::chrono::seconds{5}, [] { return sink_compute_count >= 2U; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::thread::id source_thread{};
  static inline std::thread::id sink_thread{};
  static inline bool source_thread_changed{};
  static inline bool sink_thread_changed{};
  static inline std::size_t source_compute_count{};
  static inline std::size_t sink_compute_count{};
  static inline std::uint32_t observed_first{};
  static inline std::uint32_t observed_second{};
};

// Emits two aligned scalar streams from the explicitly placed partition anchor.
// Used by HybridExactThreadAnchorAndAutomaticRemainderCarryChannelData.
class HybridChannelSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(first, "first").max_emits_per_compute(1U);
    spec.output(second, "second").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}})
        .complete_after(ComputeCalls{.limit = 2U});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const TimePoint activation = context.activation_time();
    if (!activation.valid()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const std::uint32_t ordinal = emitted_++;
    const SampleMetadata metadata{.capture_timestamp_ns = activation.timestamp_ns,
                                  .source_clock_id = activation.clock_id};
    auto first_emitted = first.emit(Sample<std::uint32_t>{ordinal + 1U, metadata});
    if (!first_emitted) {
      return first_emitted;
    }
    auto second_emitted = second.emit(Sample<std::uint32_t>{ordinal + 101U, metadata});
    if (second_emitted) {
      HybridChannelOwnerProbeState::record_source();
    }
    return second_emitted;
  }

  Output<std::uint32_t> first;
  Output<std::uint32_t> second;

 private:
  std::uint32_t emitted_{};
};

// Consumes aligned pairs on the automatically assigned downstream partition.
// Used by HybridExactThreadAnchorAndAutomaticRemainderCarryChannelData.
class HybridChannelSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(first, "first").queue_depth(4U);
    spec.input(second, "second").queue_depth(4U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnAligned{.ports = ports(first, second), .window = std::chrono::milliseconds{1}})
        .input(first, Exact{})
        .input(second, Exact{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto first_sample = first.receive();
    if (!first_sample) {
      return make_unexpected(std::move(first_sample).error());
    }
    auto second_sample = second.receive();
    if (!second_sample) {
      return make_unexpected(std::move(second_sample).error());
    }
    HybridChannelOwnerProbeState::record_sink(first_sample->data, second_sample->data);
    return {};
  }

  Input<std::uint32_t> first;
  Input<std::uint32_t> second;
};

// Graph topology:
//   source(first)  --> sink(first)
//   source(second) --> sink(second)
[[nodiscard]] ExecutionPlan make_runtime_hybrid_multi_callback_channel_plan() {
  Graph graph{"runtime-hybrid-multi-callback-channel"};
  const auto source = graph.op<HybridChannelSourceOperator>("source");
  const auto sink = graph.op<HybridChannelSinkOperator>("sink");
  graph.add_flow(source->first, sink->first, ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(source->second, sink->second, ConnectionOptions{.queue_depth = 4U});
  graph.partition("runtime-aligned-anchor").add(source);
  graph.auto_partition_remaining();
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-hybrid-channel-clock"));
  CompileOptions options;
  options.deployment.bind_thread_placement("runtime-aligned-anchor",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return compile(graph, std::move(options));
}

// Shared static-storage aggregator for OnAnyInput/OnEach channel trigger tests. Tracks
// source and sink compute counts, observed values, and the other-input selection state
// per activation. Representative test: InputAnyExecutesThroughChannelBinding.
struct ChannelInputTriggerProbe {
  static void reset() {
    lifecycle_start_count.store(0U, std::memory_order_relaxed);
    source_compute_entry_count.store(0U, std::memory_order_relaxed);
    source_compute_count.store(0U, std::memory_order_relaxed);
    sink_compute_count.store(0U, std::memory_order_relaxed);
    first_observed_value.store(0U, std::memory_order_relaxed);
    second_observed_value.store(0U, std::memory_order_relaxed);
    other_selection_state.store(InputState::kAbsent, std::memory_order_relaxed);
    other_selection_reason.store(SelectionReason::kNeverSeen, std::memory_order_relaxed);
    other_receive_error.store(ErrorCode::kSuccess, std::memory_order_relaxed);
    allow_each_successor.store(false, std::memory_order_relaxed);
    work_started.store(false, std::memory_order_relaxed);
    first_work_cpu.store(-1, std::memory_order_relaxed);
    const std::lock_guard lock{mutex};
    first_work_affinity.clear();
  }

  static void record_work_started() {
#if defined(__linux__)
    int absent = -1;
    static_cast<void>(first_work_cpu.compare_exchange_strong(
        absent, sched_getcpu(), std::memory_order_release, std::memory_order_relaxed));
    auto snapshot = detail::LinuxThreadController::snapshot();
    if (snapshot) {
      const std::lock_guard lock{mutex};
      if (first_work_affinity.empty()) {
        first_work_affinity = std::move(snapshot->affinity.cores);
      }
    }
#endif
    work_started.store(true, std::memory_order_release);
  }

  static void record_lifecycle_start() {
    lifecycle_start_count.fetch_add(1U, std::memory_order_release);
  }

  static void record_source_compute_entry() {
    source_compute_entry_count.fetch_add(1U, std::memory_order_release);
    cv.notify_all();
  }

  [[nodiscard]] static std::vector<std::uint32_t> observed_first_work_affinity() {
    const std::lock_guard lock{mutex};
    return first_work_affinity;
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
  static inline std::atomic<std::size_t> lifecycle_start_count{};
  static inline std::atomic<std::size_t> source_compute_entry_count{};
  static inline std::atomic<std::size_t> source_compute_count{};
  static inline std::atomic<std::size_t> sink_compute_count{};
  static inline std::atomic<std::uint32_t> first_observed_value{};
  static inline std::atomic<std::uint32_t> second_observed_value{};
  static inline std::atomic<InputState> other_selection_state{InputState::kAbsent};
  static inline std::atomic<SelectionReason> other_selection_reason{SelectionReason::kNeverSeen};
  static inline std::atomic<ErrorCode> other_receive_error{ErrorCode::kSuccess};
  static inline std::atomic<bool> allow_each_successor{};
  static inline std::atomic_bool work_started{};
  static inline std::atomic<int> first_work_cpu{-1};
  static inline std::vector<std::uint32_t> first_work_affinity;
};

// Clock-triggered source (1 ms period) that emits to 'first' or 'second' depending on kind.
// For kAny: emits once to 'second'. For kEach: emits to 'first', then to 'first' again
// after allow_each_successor is set. Used by InputAnyExecutesThroughChannelBinding.
class ChannelInputTriggerSource final : public Operator<> {
 public:
  explicit ChannelInputTriggerSource(ChannelInputTriggerKind kind,
                                     bool restart_after_first_compute = false)
      : kind_(kind), restart_after_first_compute_(restart_after_first_compute) {}

  void setup(OperatorSpec& spec) override {
    spec.output(first, "first").max_emits_per_compute(1U);
    spec.output(second, "second").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }

  void start() override { ChannelInputTriggerProbe::record_lifecycle_start(); }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    ChannelInputTriggerProbe::record_work_started();
    ChannelInputTriggerProbe::record_source_compute_entry();
    if (restart_after_first_compute_ && !restart_requested_) {
      restart_requested_ = true;
      auto requested =
          context.request_restart(LifecycleRestartOptions{.from_stage = LifecycleStage::kStart});
      if (!requested) {
        return make_unexpected(std::move(requested.error()));
      }
      return {};
    }
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
  bool restart_after_first_compute_{};
  bool restart_requested_{};
  std::size_t step_{};
};

// Observes the selected input and verifies the sibling remains unselected for the activation.
// Used by InputAnyExecutesThroughChannelBinding and InputEachExecutesThroughChannelBinding.
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

  void start() override { ChannelInputTriggerProbe::record_lifecycle_start(); }

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
    ChannelInputTriggerKind kind, const bool force_holomq_process_local_inline = false,
    CompileOptions options = {}, const bool split_execution_partitions = false,
    const bool restart_after_first_compute = false) {
  Graph graph{test::make_host_unique_test_name("channel-input-trigger")};
  const auto source =
      graph.op<ChannelInputTriggerSource>("source", kind, restart_after_first_compute);
  const auto sink = graph.op<ChannelInputTriggerSink>("sink", kind);
  graph.add_flow(source->first, sink->first, ConnectionOptions{.queue_depth = 2U});
  graph.add_flow(source->second, sink->second, ConnectionOptions{.queue_depth = 2U});
  if (split_execution_partitions) {
    graph.partition("channel-input-source").add(source);
    graph.partition("channel-input-sink").add(sink);
    graph.auto_partition_remaining();
  }
  if (restart_after_first_compute) {
    graph.lifecycle().allow_self_restart(source);
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
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("channel-input-trigger-clock");
  graph.set_default_clock(default_clock);
  return force_holomq_process_local_inline
             ? detail::compile_with_process_local_holomq(graph, std::move(options))
             : compile(graph, std::move(options));
}

void expect_channel_input_trigger_execution(ChannelInputTriggerKind kind) {
  const ScopedPartitionEnvironment absent{std::nullopt};
  ChannelInputTriggerProbe::reset();
  const ExecutionPlan plan = make_channel_input_trigger_plan(kind);
  // Trigger-only OnAnyInput and OnEach pin the arrival witness directly. They must not synthesize
  // an explicit selector transaction or TemporalInputCache for the trigger input.
  EXPECT_TRUE(plan.input_selections().empty()) << plan.json();
  expect_channel_plan_partition_facts(plan, kAutomaticPartitionAlgorithmId);

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
                                                              ChannelScalarSelectorKind selector,
                                                              bool split_partitions = false) {
  Graph graph{"channel-scalar-selector"};
  const auto source = graph.op<ChannelScalarSelectorSource>("source");
  const auto sink = graph.op<ChannelScalarSelectorSink>("sink", selector);
  graph.add_flow(source->candidate,
                 sink->candidate,
                 ConnectionOptions{.queue_policy = queue_policy, .queue_depth = 2U});
  graph.add_flow(source->driver, sink->driver, ConnectionOptions{.queue_depth = 2U});
  if (split_partitions) {
    graph.partition("channel-scalar-source").add(source);
    graph.partition("channel-scalar-sink").add(sink);
  }
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

// REQ: COMPILE-068
// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-041
TEST(DispatcherTest, AutomaticSourceChainInterfaceV3RunsThroughPartitionLocalSerialOwner) {
  const ScopedPartitionEnvironment absent{std::nullopt};
  // Key setup: compile the aligned Channel graph through the automatic source-chain planner.
  const ExecutionPlan compiled = test::make_runtime_aligned_channel_plan();
  ASSERT_TRUE(compiled.ok()) << (compiled.diagnostics().empty()
                                     ? "no diagnostic"
                                     : compiled.diagnostics().front().token);
  ASSERT_TRUE(compiled.partitioning_plan().accepted_algorithm.has_value());
  EXPECT_EQ(compiled.partitioning_plan().interface_version, 3U);
  EXPECT_EQ(compiled.partitioning_plan().maturity, AlgorithmMaturity::kProduction);
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  EXPECT_EQ(compiled.partitioning_plan().accepted_algorithm->id, "holoscan.partition.source_chain");
  ASSERT_EQ(compiled.execution_partitions().size(), 1U);

  test::RuntimeAlignedChannelProbeState::reset(test::RuntimeAlignedChannelScript::kExactBoundary);
  RunSession session = run_async(compiled);
  const bool observed = test::RuntimeAlignedChannelProbeState::wait_for_sink(1U);
  session.request_stop();
  session.wait();

  // Key assertion: the accepted interface-v3 partition executes through one serial owner.
  ASSERT_TRUE(observed) << (session.diagnostics().empty() ? "sink callback not observed"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
}

// REQ: COMPILE-068
// REQ: RUNTIME-041
// REQ: OPERATOR-050
TEST(DispatcherTest, RuntimeAdmitsLosslesslyEncodedPartitionOperatorPaths) {
  const ScopedPartitionEnvironment absent{std::nullopt};
  EncodedPartitionPathProbeState::reset();
  // Key setup: compile two distinct embedded/operator segment pairs with the same decoded path.
  const ExecutionPlan plan = make_encoded_partition_path_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto first = plan.partitions().lookup_for_path("a/b%2Fc");
  const auto second = plan.partitions().lookup_for_path("a%2Fb/c");
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  ASSERT_EQ(first->operator_paths().size(), 1U);
  ASSERT_EQ(second->operator_paths().size(), 1U);
  EXPECT_EQ(first->operator_paths().front(), "a/b%2Fc");
  EXPECT_EQ(second->operator_paths().front(), "a%2Fb/c");
  EXPECT_NE(first->id(), second->id());

  RunSession session = run_async(plan);
  session.wait();

  // Key assertion: encoded paths remain distinct and both reconstructed callbacks complete.
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(EncodedPartitionPathProbeState::callbacks.load(std::memory_order_relaxed), 2U);
  EXPECT_EQ(EncodedPartitionPathProbeState::first_path.load(std::memory_order_relaxed), 1U);
  EXPECT_EQ(EncodedPartitionPathProbeState::second_path.load(std::memory_order_relaxed), 1U);
  EXPECT_FALSE(EncodedPartitionPathProbeState::unexpected_path.load(std::memory_order_relaxed));
}

// REQ: COMPILE-068
// REQ: RUNTIME-004
// REQ: RUNTIME-041
TEST(DispatcherTest, AutomaticSourceChainRunsIndependentPartitionsOnDistinctOwners) {
  const ScopedPartitionEnvironment absent{std::nullopt};
  // Key setup: compile two disconnected clock-triggered operators without an authored cover.
  const ExecutionPlan plan = make_automatic_multi_owner_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  EXPECT_EQ(plan.partitioning_plan().mode, PartitioningMode::kAutomatic);
  EXPECT_EQ(plan.partitioning_plan().authored_partition_count, 0U);
  EXPECT_EQ(plan.partitioning_plan().automatic_partition_count, 2U);
  ASSERT_EQ(plan.execution_partitions().size(), 2U);
  ASSERT_EQ(plan.execution_lanes().size(), 2U);
  EXPECT_EQ(plan.scheduler_plan().max_concurrent_activations, 2U);
  EXPECT_TRUE(plan.scheduler_plan().activation_overlap);
  for (const auto& partition : plan.execution_partitions()) {
    EXPECT_FALSE(partition.authored_name.has_value());
    EXPECT_EQ(partition.operator_paths.size(), 1U);
  }

  AutomaticOwnerProbeState::reset();
  RunSession session = run_async(plan);
  const bool observed = AutomaticOwnerProbeState::wait_for_both();
  session.request_stop();
  session.wait();

  // Key assertion: the automatic partitions execute on distinct stable owner threads.
  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_NE(AutomaticOwnerProbeState::observed[0], AutomaticOwnerProbeState::observed[1]);
}

// REQ: COMPILE-068
// REQ: RUNTIME-004
// REQ: RUNTIME-006
// REQ: RUNTIME-011
// REQ: RUNTIME-050
TEST(DispatcherTest, AutomaticSourceChainFanInDrainsFinalDataBeforeNaturalTerminal) {
  const ScopedPartitionEnvironment absent{std::nullopt};
  // Key setup: compile two source partitions feeding one join-and-drain partition.
  const ExecutionPlan plan = make_automatic_fanin_drain_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  EXPECT_EQ(plan.partitioning_plan().mode, PartitioningMode::kAutomatic);
  EXPECT_EQ(plan.partitioning_plan().authored_partition_count, 0U);
  EXPECT_EQ(plan.partitioning_plan().automatic_partition_count, 3U);
  ASSERT_EQ(plan.execution_partitions().size(), 3U);
  ASSERT_EQ(plan.execution_lanes().size(), 3U);
  EXPECT_EQ(plan.scheduler_plan().worker_count, 3U);
  EXPECT_EQ(plan.scheduler_plan().max_concurrent_activations, 3U);
  EXPECT_TRUE(plan.scheduler_plan().activation_overlap);
  EXPECT_EQ(plan.schedule_edges().size(), 2U);
  const auto left_partition =
      std::ranges::find_if(plan.execution_partitions(), [](const auto& partition) {
        return partition.operator_paths == std::vector<std::string>{"left"};
      });
  const auto right_partition =
      std::ranges::find_if(plan.execution_partitions(), [](const auto& partition) {
        return partition.operator_paths == std::vector<std::string>{"right"};
      });
  const auto join_partition =
      std::ranges::find_if(plan.execution_partitions(), [](const auto& partition) {
        return partition.operator_paths.size() == 2U &&
               std::ranges::find(partition.operator_paths, "join") !=
                   partition.operator_paths.end() &&
               std::ranges::find(partition.operator_paths, "drain") !=
                   partition.operator_paths.end();
      });
  ASSERT_NE(left_partition, plan.execution_partitions().end());
  ASSERT_NE(right_partition, plan.execution_partitions().end());
  ASSERT_NE(join_partition, plan.execution_partitions().end());
  EXPECT_NE(left_partition->id, right_partition->id);
  EXPECT_NE(left_partition->execution_lane, right_partition->execution_lane);
  EXPECT_NE(left_partition->execution_lane, join_partition->execution_lane);
  EXPECT_NE(right_partition->execution_lane, join_partition->execution_lane);

  AutomaticFanInProbeState::reset();
  RunSession session = run_async(plan);
  const bool final_data_drained = AutomaticFanInProbeState::wait_for_final_drain();
  const RunTermination termination_while_final_drain_blocked = session.termination();
  if (!final_data_drained) {
    session.request_stop();
  }
  AutomaticFanInProbeState::release_final_drain_callback();
  session.wait();

  // Key assertion: natural completion waits for the final drain callback and preserves owners.
  ASSERT_TRUE(final_data_drained);
  EXPECT_EQ(termination_while_final_drain_blocked, RunTermination::kRunning);
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  ASSERT_EQ(AutomaticFanInProbeState::joined_values, (std::vector<std::uint32_t>{1121U, 1222U}));
  EXPECT_EQ(AutomaticFanInProbeState::drained_values, AutomaticFanInProbeState::joined_values);
  EXPECT_NE(AutomaticFanInProbeState::source_threads[0], std::thread::id{});
  EXPECT_NE(AutomaticFanInProbeState::source_threads[1], std::thread::id{});
  EXPECT_NE(AutomaticFanInProbeState::join_thread, std::thread::id{});
  EXPECT_EQ(AutomaticFanInProbeState::join_thread, AutomaticFanInProbeState::drain_thread);
  EXPECT_NE(AutomaticFanInProbeState::source_threads[0],
            AutomaticFanInProbeState::source_threads[1]);
  EXPECT_NE(AutomaticFanInProbeState::source_threads[0], AutomaticFanInProbeState::join_thread);
  EXPECT_NE(AutomaticFanInProbeState::source_threads[1], AutomaticFanInProbeState::join_thread);
  EXPECT_FALSE(AutomaticFanInProbeState::source_thread_changed[0]);
  EXPECT_FALSE(AutomaticFanInProbeState::source_thread_changed[1]);
  EXPECT_EQ(AutomaticFanInProbeState::source_compute_counts[0], 2U);
  EXPECT_EQ(AutomaticFanInProbeState::source_compute_counts[1], 2U);
  EXPECT_FALSE(AutomaticFanInProbeState::join_thread_changed);
  EXPECT_FALSE(AutomaticFanInProbeState::drain_thread_changed);
}

// REQ: COMPILE-068
// REQ: RUNTIME-004
// REQ: RUNTIME-015
// REQ: RUNTIME-041
TEST(DispatcherTest, HybridExactThreadAnchorAndAutomaticRemainderCarryChannelData) {
  const ScopedPartitionEnvironment absent{std::nullopt};
  // Key setup: constrain the source and let the planner assign the aligned sink automatically.
  const ExecutionPlan plan = make_runtime_hybrid_multi_callback_channel_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_TRUE(plan.partitioning_plan().accepted_algorithm.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  EXPECT_EQ(plan.partitioning_plan().accepted_algorithm->id, "holoscan.partition.source_chain");
  EXPECT_EQ(plan.partitioning_plan().mode, PartitioningMode::kConstrainedAutomatic);
  EXPECT_EQ(plan.partitioning_plan().authored_partition_count, 1U);
  EXPECT_EQ(plan.partitioning_plan().automatic_partition_count, 1U);
  ASSERT_EQ(plan.execution_partitions().size(), 2U);
  const auto anchor = std::find_if(
      plan.execution_partitions().begin(),
      plan.execution_partitions().end(),
      [](const auto& partition) { return partition.authored_name == "runtime-aligned-anchor"; });
  ASSERT_NE(anchor, plan.execution_partitions().end());
  EXPECT_EQ(anchor->operator_paths, (std::vector<std::string>{"source"}));
  const auto automatic =
      std::find_if(plan.execution_partitions().begin(),
                   plan.execution_partitions().end(),
                   [](const auto& partition) { return !partition.authored_name.has_value(); });
  ASSERT_NE(automatic, plan.execution_partitions().end());
  EXPECT_EQ(automatic->operator_paths, (std::vector<std::string>{"sink"}));
  ASSERT_EQ(plan.partition_placements().size(), 1U);
  EXPECT_EQ(plan.partition_placements().front().partition, anchor->id);
  EXPECT_EQ(plan.partition_placements().front().execution_lane, anchor->execution_lane);
  ASSERT_TRUE(plan.partition_placements().front().accepted.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access): guarded by ASSERT_TRUE above.
  EXPECT_EQ(plan.partition_placements().front().accepted->wake_mode, WakeMode::kBusyPoll);

  const ScopedPartitionEnvironment poison_after_compile{
      std::string_view{"retired-selector-must-not-affect-runtime"}};
  HybridChannelOwnerProbeState::reset();
  RunSession session = run_async(plan);
  const bool observed = HybridChannelOwnerProbeState::wait_for_two_sink_callbacks();
  if (!observed) {
    session.request_stop();
  }
  session.wait();

  // Key assertion: both partitions retain stable, distinct owners while carrying aligned data.
  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(HybridChannelOwnerProbeState::source_compute_count, 2U);
  EXPECT_EQ(HybridChannelOwnerProbeState::sink_compute_count, 2U);
  EXPECT_EQ(HybridChannelOwnerProbeState::observed_first, 2U);
  EXPECT_EQ(HybridChannelOwnerProbeState::observed_second, 102U);
  EXPECT_NE(HybridChannelOwnerProbeState::source_thread, std::thread::id{});
  EXPECT_NE(HybridChannelOwnerProbeState::sink_thread, std::thread::id{});
  EXPECT_NE(HybridChannelOwnerProbeState::source_thread, HybridChannelOwnerProbeState::sink_thread);
  EXPECT_FALSE(HybridChannelOwnerProbeState::source_thread_changed);
  EXPECT_FALSE(HybridChannelOwnerProbeState::sink_thread_changed);
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
TEST(DispatcherTest, AutomaticAndCompleteManualPlansRunThroughTheSameSerialLane) {
  const ScopedPartitionEnvironment absent{std::nullopt};
  struct Observation {
    std::uint32_t first{};
    std::uint32_t second{};
    std::size_t source_computes{};
    std::size_t sink_computes{};
  };
  std::array<Observation, 2U> observations{};
  // Key setup: run equivalent automatic and complete-manual covers through one serial lane.
  for (std::size_t index = 0U; index < observations.size(); ++index) {
    const ExecutionPlan plan = test::make_runtime_aligned_channel_plan(index == 1U);
    ASSERT_TRUE(plan.ok()) << plan.json();
    ASSERT_TRUE(plan.partitioning_plan().accepted_algorithm.has_value());
    EXPECT_EQ(
        plan.partitioning_plan().accepted_algorithm->id,
        index == 0U ? "holoscan.partition.source_chain" : "holoscan.partition.serial_baseline");
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

  // Key assertion: both cover modes produce the same values and callback counts.
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
  const ScopedPartitionEnvironment absent{std::nullopt};
  // Key setup: compile the representative aligned Channel topology with automatic partitioning.
  const ExecutionPlan plan = test::make_runtime_aligned_channel_plan();
  // Key assertion: the helper verifies the complete accepted cover and scheduler authority.
  expect_channel_plan_partition_facts(plan, kAutomaticPartitionAlgorithmId);
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-015
// REQ: RUNTIME-018
// REQ: RUNTIME-034
TEST(ChannelPartitionPlanParityTest, InputAnyExecutesThroughChannelBinding) {
  // Key setup: run the two-input Channel graph with OnAnyInput readiness.
  // Key assertion: the shared verifier checks selection, callback, and terminal behavior.
  expect_channel_input_trigger_execution(ChannelInputTriggerKind::kAny);
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-015
// REQ: RUNTIME-018
TEST(ChannelPartitionPlanParityTest, InputEachExecutesThroughChannelBinding) {
  // Key setup: run the two-input Channel graph with OnEach(first) readiness.
  // Key assertion: the shared verifier checks selection, callback, and terminal behavior.
  expect_channel_input_trigger_execution(ChannelInputTriggerKind::kEach);
}

// REQ: RUNTIME-061
TEST_P(ChannelSupportThreadModeTest, AcceptedAffinityAppliesBeforePrivateObservationAndWork) {
  const ScopedPartitionEnvironment absent{std::nullopt};
  const bool force_holomq = GetParam() != ProcessLocalTestDeliveryMode::kDirectRetained;
  const bool force_channel_pump = GetParam() == ProcessLocalTestDeliveryMode::kHoloMQForcedPump;
  // Key setup: attach a source-tree-only observer to direct, pump-free eventless HoloMQ, and the
  // exact eventless forced-pump OnAnyInput control.
  ChannelInputTriggerProbe::reset();
  auto control =
      std::make_shared<RecordingSupportThreadControl>(&ChannelInputTriggerProbe::work_started);
  CompileOptions options;
#if defined(__linux__)
  const auto compiling_thread = detail::LinuxThreadController::snapshot();
  ASSERT_TRUE(compiling_thread.has_value());
  ASSERT_FALSE(compiling_thread->affinity.cores.empty());
  const std::uint32_t selected_cpu = compiling_thread->affinity.cores.front();
  options.deployment
      .bind_support_thread_affinity(SupportThreadRole::kController, CPUAffinity{selected_cpu})
      .bind_support_thread_affinity(SupportThreadRole::kChannelPump, CPUAffinity{selected_cpu});
#endif
  ExecutionPlan plan = make_channel_input_trigger_plan(
      ChannelInputTriggerKind::kAny, force_holomq, std::move(options));
  if (force_channel_pump) {
    plan = detail::RuntimeTestPlanAccess::force_process_local_channel_pump_for_test(plan);
  }
  plan = detail::RuntimeTestPlanAccess::with_support_thread_control(plan, control);

  RunSession session = run_async(plan);
  const bool observed = ChannelInputTriggerProbe::wait_for_sink(1U);
  session.request_stop();
  session.wait();

  EXPECT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_TRUE(control->entered(SupportThreadRole::kController));
  EXPECT_TRUE(control->entered_before_work(SupportThreadRole::kController));
  // Key assertion: only the exact cold control enters a pump; direct retention and the ordinary
  // all-eventless HoloMQ process begin work with the controller as their sole support thread.
  EXPECT_EQ(control->entered(SupportThreadRole::kChannelPump), force_channel_pump);
  EXPECT_EQ(control->entered_before_work(SupportThreadRole::kChannelPump), force_channel_pump);
#if defined(__linux__)
  EXPECT_EQ(control->observed_affinity(SupportThreadRole::kController),
            (std::vector<std::uint32_t>{selected_cpu}));
  EXPECT_EQ(ChannelInputTriggerProbe::first_work_cpu.load(std::memory_order_acquire),
            static_cast<int>(selected_cpu));
  if (force_channel_pump) {
    EXPECT_EQ(control->observed_affinity(SupportThreadRole::kChannelPump),
              (std::vector<std::uint32_t>{selected_cpu}));
  }
#endif
}

// REQ: RUNTIME-061
TEST(ChannelSupportThreadPlacement, MultiCpuAllowedSetAppliesWholeToControllerAndPump) {
#if defined(__linux__)
  const ScopedPartitionEnvironment absent{std::nullopt};
  const auto compiling_thread = detail::LinuxThreadController::snapshot();
  ASSERT_TRUE(compiling_thread.has_value());
  if (compiling_thread->affinity.cores.size() < 2U) {
    GTEST_SKIP() << "The effective compiling-thread affinity contains fewer than two CPUs";
  }
  const std::vector<std::uint32_t> allowed_cpus{compiling_thread->affinity.cores[0],
                                                compiling_thread->affinity.cores[1]};
  CompileOptions options;
  options.deployment
      .bind_support_thread_affinity(SupportThreadRole::kController, CPUAffinity{allowed_cpus})
      .bind_support_thread_affinity(SupportThreadRole::kChannelPump, CPUAffinity{allowed_cpus});
  ChannelInputTriggerProbe::reset();
  auto control =
      std::make_shared<RecordingSupportThreadControl>(&ChannelInputTriggerProbe::work_started);
  ExecutionPlan plan =
      make_channel_input_trigger_plan(ChannelInputTriggerKind::kAny, true, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  plan = detail::RuntimeTestPlanAccess::force_process_local_channel_pump_for_test(plan);
  plan = detail::RuntimeTestPlanAccess::with_support_thread_control(plan, control);

  RunSession session = run_async(plan);
  const bool observed = ChannelInputTriggerProbe::wait_for_sink(1U);
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(control->observed_affinity(SupportThreadRole::kController), allowed_cpus);
  EXPECT_EQ(control->observed_affinity(SupportThreadRole::kChannelPump), allowed_cpus);
  const int first_work_cpu =
      ChannelInputTriggerProbe::first_work_cpu.load(std::memory_order_acquire);
  ASSERT_GE(first_work_cpu, 0);
  EXPECT_NE(std::ranges::find(allowed_cpus, static_cast<std::uint32_t>(first_work_cpu)),
            allowed_cpus.end());
#else
  GTEST_SKIP() << "CPU-affinity application is supported only on Linux";
#endif
}

// REQ: RUNTIME-061
TEST(ChannelSupportThreadPlacement, ControllerOnlyAffinityPreservesUnboundWorkerMask) {
#if defined(__linux__)
  const ScopedPartitionEnvironment absent{std::nullopt};
  const auto compiling_thread = detail::LinuxThreadController::snapshot();
  ASSERT_TRUE(compiling_thread.has_value());
  if (compiling_thread->affinity.cores.size() < 2U) {
    GTEST_SKIP() << "The effective compiling-thread affinity contains fewer than two CPUs";
  }
  const std::vector<std::uint32_t> original_affinity = compiling_thread->affinity.cores;
  const std::uint32_t controller_cpu = original_affinity.front();
  CompileOptions options;
  options.deployment.bind_support_thread_affinity(SupportThreadRole::kController,
                                                  CPUAffinity{controller_cpu});
  ChannelInputTriggerProbe::reset();
  auto control =
      std::make_shared<RecordingSupportThreadControl>(&ChannelInputTriggerProbe::work_started);
  ExecutionPlan plan = make_channel_input_trigger_plan(
      ChannelInputTriggerKind::kAny, false, std::move(options), true);
  ASSERT_TRUE(plan.ok()) << plan.json();
  plan = detail::RuntimeTestPlanAccess::with_support_thread_control(plan, control);

  RunSession session = run_async(plan);
  const bool observed = ChannelInputTriggerProbe::wait_for_sink(1U);
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_EQ(control->observed_affinity(SupportThreadRole::kController),
            (std::vector<std::uint32_t>{controller_cpu}));
  EXPECT_EQ(ChannelInputTriggerProbe::observed_first_work_affinity(), original_affinity);
#else
  GTEST_SKIP() << "CPU-affinity application is supported only on Linux";
#endif
}

// REQ: RUNTIME-061
TEST(ChannelSupportThreadPlacement,
     ProductionPlacementFailureStopsBeforeObserverAndOrdinaryCompute) {
#if defined(__linux__)
  const auto compiling_thread = detail::LinuxThreadController::snapshot();
  ASSERT_TRUE(compiling_thread.has_value());
  ASSERT_FALSE(compiling_thread->affinity.cores.empty());
  for (const SupportThreadRole failing_role :
       {SupportThreadRole::kController, SupportThreadRole::kChannelPump}) {
    SCOPED_TRACE(static_cast<std::uint32_t>(failing_role));
    CompileOptions options;
    options.deployment
        .bind_support_thread_affinity(SupportThreadRole::kController,
                                      CPUAffinity{compiling_thread->affinity.cores.front()})
        .bind_support_thread_affinity(SupportThreadRole::kChannelPump,
                                      CPUAffinity{compiling_thread->affinity.cores.front()});
    ChannelInputTriggerProbe::reset();
    ExecutionPlan plan =
        make_channel_input_trigger_plan(ChannelInputTriggerKind::kAny, true, std::move(options));
    ASSERT_TRUE(plan.ok()) << plan.json();
    plan = detail::RuntimeTestPlanAccess::force_process_local_channel_pump_for_test(plan);
    plan = detail::RuntimeTestPlanAccess::with_support_thread_affinity_core_for_test(
        plan, failing_role, CPU_SETSIZE);
    ASSERT_TRUE(plan.ok());
    auto control =
        std::make_shared<RecordingSupportThreadControl>(&ChannelInputTriggerProbe::work_started);
    plan = detail::RuntimeTestPlanAccess::with_support_thread_control(plan, control);

    RunSession session = run_async(plan);
    const bool terminal = session.wait_for(std::chrono::seconds{5});
    if (!terminal) {
      session.request_stop();
    }
    session.wait();

    ASSERT_TRUE(terminal) << "support-thread placement failure did not terminate the session";
    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_FALSE(ChannelInputTriggerProbe::work_started.load(std::memory_order_acquire));
    EXPECT_EQ(ChannelInputTriggerProbe::lifecycle_start_count.load(std::memory_order_acquire),
              failing_role == SupportThreadRole::kController ? 0U : 2U);
    EXPECT_FALSE(control->entered(failing_role));
    const auto diagnostic = std::ranges::find(
        session.diagnostics(), std::string{"RUNTIME_STARTUP_FAILED"}, &Diagnostic::token);
    ASSERT_NE(diagnostic, session.diagnostics().end());
    const auto& detail = diagnostic->detail;
    HOLOSCAN_ASSERT_HAS_VALUE(detail);
    EXPECT_EQ(detail->operation, DiagnosticOperation::kRuntime);
    const std::string role_name =
        failing_role == SupportThreadRole::kController ? "controller" : "channel-pump";
    EXPECT_EQ(detail->operator_path, "support-thread/" + role_name);
    EXPECT_NE(diagnostic->required.find("requested=[" + std::to_string(CPU_SETSIZE) + "]"),
              std::string::npos);
    EXPECT_NE(diagnostic->provided.find("phase=runtime_apply"), std::string::npos);
    EXPECT_NE(detail->error.message.find("affinity"), std::string::npos);
    const std::string formatted = format_diagnostic(*diagnostic, plan.graph());
    EXPECT_NE(formatted.find(role_name), std::string::npos);
    EXPECT_NE(formatted.find("runtime_apply"), std::string::npos);
  }
#else
  GTEST_SKIP() << "CPU-affinity application is supported only on Linux";
#endif
}

// REQ: RUNTIME-061
TEST(ChannelSupportThreadPlacement, ReplacementPumpFailureRetainsPlacementContext) {
#if defined(__linux__)
  const auto compiling_thread = detail::LinuxThreadController::snapshot();
  ASSERT_TRUE(compiling_thread.has_value());
  ASSERT_FALSE(compiling_thread->affinity.cores.empty());
  CompileOptions options;
  options.deployment.bind_support_thread_affinity(
      SupportThreadRole::kChannelPump, CPUAffinity{compiling_thread->affinity.cores.front()});
  ChannelInputTriggerProbe::reset();
  ExecutionPlan plan = make_channel_input_trigger_plan(
      ChannelInputTriggerKind::kAny, true, std::move(options), false, true);
  ASSERT_TRUE(plan.ok()) << plan.json();
  plan = detail::RuntimeTestPlanAccess::force_process_local_channel_pump_for_test(plan);
  plan =
      detail::RuntimeTestPlanAccess::with_support_thread_affinity_core_after_first_entry_for_test(
          plan, SupportThreadRole::kChannelPump, CPU_SETSIZE);
  ASSERT_TRUE(plan.ok());

  RunSession session = run_async(plan);
  const bool terminal = session.wait_for(std::chrono::seconds{5});
  if (!terminal) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(terminal) << "replacement pump placement failure did not terminate the session";
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(ChannelInputTriggerProbe::work_started.load(std::memory_order_acquire));
  EXPECT_EQ(ChannelInputTriggerProbe::source_compute_entry_count.load(std::memory_order_acquire),
            1U);
  EXPECT_EQ(ChannelInputTriggerProbe::source_compute_count.load(std::memory_order_acquire), 0U);
  const auto diagnostic = std::ranges::find(
      session.diagnostics(), std::string{"RUNTIME_STARTUP_FAILED"}, &Diagnostic::token);
  ASSERT_NE(diagnostic, session.diagnostics().end());
  const auto& detail = diagnostic->detail;
  HOLOSCAN_ASSERT_HAS_VALUE(detail);
  EXPECT_EQ(detail->operation, DiagnosticOperation::kRuntime);
  EXPECT_EQ(detail->operator_path, "support-thread/channel-pump");
  EXPECT_NE(diagnostic->required.find("requested=[" + std::to_string(CPU_SETSIZE) + "]"),
            std::string::npos);
  EXPECT_NE(diagnostic->provided.find("phase=runtime_apply"), std::string::npos);
  EXPECT_NE(detail->error.message.find("affinity"), std::string::npos);
#else
  GTEST_SKIP() << "CPU-affinity application is supported only on Linux";
#endif
}

// REQ: RUNTIME-061
TEST(ChannelSupportThreadPlacement, PumpSetupAllocationFailureIsContainedBeforeCompute) {
#if (defined(__cpp_exceptions) || defined(_CPPUNWIND)) && defined(__linux__)
  const auto compiling_thread = detail::LinuxThreadController::snapshot();
  ASSERT_TRUE(compiling_thread.has_value());
  ASSERT_FALSE(compiling_thread->affinity.cores.empty());
  CompileOptions options;
  options.deployment.bind_support_thread_affinity(
      SupportThreadRole::kChannelPump, CPUAffinity{compiling_thread->affinity.cores.front()});
  ChannelInputTriggerProbe::reset();
  ExecutionPlan plan =
      make_channel_input_trigger_plan(ChannelInputTriggerKind::kAny, true, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  plan = detail::RuntimeTestPlanAccess::force_process_local_channel_pump_for_test(plan);
  plan = detail::RuntimeTestPlanAccess::with_support_thread_control(
      plan, std::make_shared<PumpAllocationFailureSupportThreadControl>());

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_FALSE(ChannelInputTriggerProbe::work_started.load(std::memory_order_acquire));
  EXPECT_EQ(ChannelInputTriggerProbe::lifecycle_start_count.load(std::memory_order_acquire), 2U);
  const auto diagnostic = std::ranges::find(
      session.diagnostics(), std::string{"RUNTIME_STARTUP_FAILED"}, &Diagnostic::token);
  ASSERT_NE(diagnostic, session.diagnostics().end());
  const auto& detail = diagnostic->detail;
  HOLOSCAN_ASSERT_HAS_VALUE(detail);
  EXPECT_EQ(detail->error.code, ErrorCode::kResourceExhausted);
#else
  GTEST_SKIP() << "Pump allocation-failure containment requires Linux and language exceptions";
#endif
}

// REQ: RUNTIME-061
TEST(ChannelSupportThreadPlacement, RejectsMalformedFrozenRecordsBeforeCallbacks) {
#if defined(__linux__)
  const auto compiling_thread = detail::LinuxThreadController::snapshot();
  ASSERT_TRUE(compiling_thread.has_value());
  ASSERT_FALSE(compiling_thread->affinity.cores.empty());
  CompileOptions options;
  options.deployment.bind_support_thread_affinity(
      SupportThreadRole::kController, CPUAffinity{compiling_thread->affinity.cores.front()});
  const ExecutionPlan baseline =
      make_channel_input_trigger_plan(ChannelInputTriggerKind::kAny, false, std::move(options));
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  using PartitionMutation = detail::RuntimePartitionTestPlanAccess::PartitionMutation;
  for (const PartitionMutation mutation : {PartitionMutation::kSupportAffinityDuplicate,
                                           PartitionMutation::kSupportAffinityUnknownRole,
                                           PartitionMutation::kSupportAffinityMalformedRequest}) {
    SCOPED_TRACE(static_cast<std::uint32_t>(mutation));
    ChannelInputTriggerProbe::reset();
    const ExecutionPlan malformed =
        detail::RuntimePartitionTestPlanAccess::mutate_partition(baseline, mutation);
    ASSERT_TRUE(malformed.ok());

    RunSession session = run_async(malformed);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_FALSE(ChannelInputTriggerProbe::work_started.load(std::memory_order_acquire));
    EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_PLACEMENT_BINDING_UNAVAILABLE"));
  }
#else
  GTEST_SKIP() << "CPU-affinity application is supported only on Linux";
#endif
}

// REQ: RUNTIME-061
TEST(ChannelSupportThreadPlacement, RejectsFrozenInlineControllerAffinityConflictBeforeCallbacks) {
#if defined(__linux__)
  const auto compiling_thread = detail::LinuxThreadController::snapshot();
  ASSERT_TRUE(compiling_thread.has_value());
  ASSERT_FALSE(compiling_thread->affinity.cores.empty());
  const std::uint32_t selected_cpu = compiling_thread->affinity.cores.front();

  Graph graph{test::make_host_unique_test_name("inline-controller-affinity-conflict")};
  const auto source = graph.op<ChannelInputTriggerSource>("source", ChannelInputTriggerKind::kAny);
  const auto sink = graph.op<ChannelInputTriggerSink>("sink", ChannelInputTriggerKind::kAny);
  graph.add_flow(source->first, sink->first, ConnectionOptions{.queue_depth = 2U});
  graph.add_flow(source->second, sink->second, ConnectionOptions{.queue_depth = 2U});
  graph.partition("worker").add(source).add(sink);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-controller-affinity-clock"));
  CompileOptions options;
  options.deployment
      .bind_support_thread_affinity(SupportThreadRole::kController, CPUAffinity{selected_cpu})
      .bind_thread_placement("worker", ThreadPlacement{.affinity = CPUAffinity{selected_cpu}});
  const ExecutionPlan baseline = compile(graph, std::move(options));
  ASSERT_TRUE(baseline.ok()) << baseline.json();

  ChannelInputTriggerProbe::reset();
  const ExecutionPlan malformed = detail::RuntimePartitionTestPlanAccess::mutate_partition(
      baseline,
      detail::RuntimePartitionTestPlanAccess::PartitionMutation::kInlineControllerAffinityConflict);
  ASSERT_TRUE(malformed.ok());

  RunSession session = run_async(malformed);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_FALSE(ChannelInputTriggerProbe::work_started.load(std::memory_order_acquire));
  EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_PLACEMENT_BINDING_UNAVAILABLE"));
#else
  GTEST_SKIP() << "CPU-affinity application is supported only on Linux";
#endif
}

INSTANTIATE_TEST_SUITE_P(ChannelDelivery, ChannelSupportThreadModeTest,
                         testing::Values(ProcessLocalTestDeliveryMode::kDirectRetained,
                                         ProcessLocalTestDeliveryMode::kHoloMQEventless,
                                         ProcessLocalTestDeliveryMode::kHoloMQForcedPump),
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
  const ScopedPartitionEnvironment absent{std::nullopt};
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
    expect_channel_plan_partition_facts(plan, kAutomaticPartitionAlgorithmId);

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

// REQ: RUNTIME-004
// REQ: RUNTIME-041
TEST(ChannelPartitionPlanParityTest, LatestWithinStartupValidatesOnlyItsOwningPartitionRunner) {
  const ScopedPartitionEnvironment absent{std::nullopt};
  // source(candidate) --[depth=2]--> sink(candidate) [LatestWithin]
  // source(driver)    --[depth=2]--> sink(driver)    [Exact]
  //        partition 1                   partition 2
  // Key setup: place the producer and LatestWithin consumer in distinct same-process runners.
  const ExecutionPlan plan = make_channel_scalar_selector_plan(
      QueuePolicy::kBlocking, ChannelScalarSelectorKind::kLatestWithin, true);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ChannelScalarSelectorProbe::reset();

  RunSession session = run_async(plan);
  const bool observed = ChannelScalarSelectorProbe::wait();
  session.request_stop();
  session.wait();

  // Key assertion: the non-owning runner skips the consumer-local derived-term validation.
  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// REQ: RUNTIME-002
// REQ: RUNTIME-004
// REQ: RUNTIME-007
// REQ: RUNTIME-018
// REQ: RUNTIME-043
TEST(ChannelPartitionPlanParityTest, AllTriggersAndSelectors) {
  const ScopedPartitionEnvironment absent{std::nullopt};
  const ExecutionPlan aligned = test::make_runtime_aligned_channel_plan();
  const ExecutionPlan window = test::make_runtime_window_channel_plan();
  const ExecutionPlan predicate = test::make_runtime_input_predicate_channel_plan(
      test::RuntimeInputPredicatePattern::kFalseTrueTrue, PredicateFiring::kOnEveryTrueEvaluation);
  const ExecutionPlan any =
      test::make_runtime_input_trigger_replay_plan(test::RuntimeInputTriggerReplayKind::kAny);
  const ExecutionPlan each =
      test::make_runtime_input_trigger_replay_plan(test::RuntimeInputTriggerReplayKind::kEach);
  expect_channel_plan_partition_facts(aligned, kAutomaticPartitionAlgorithmId);
  expect_channel_plan_partition_facts(window, kAutomaticPartitionAlgorithmId);
  expect_channel_plan_partition_facts(predicate, kAutomaticPartitionAlgorithmId);
  expect_channel_plan_partition_facts(any, kAutomaticPartitionAlgorithmId);
  expect_channel_plan_partition_facts(each, kAutomaticPartitionAlgorithmId);

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
  const ScopedPartitionEnvironment absent{std::nullopt};
  const ExecutionPlan aligned = test::make_runtime_aligned_channel_plan();
  const ExecutionPlan predicate = test::make_runtime_input_predicate_channel_plan(
      test::RuntimeInputPredicatePattern::kAlwaysTrue,
      PredicateFiring::kOnEveryTrueEvaluation,
      true);
  expect_channel_plan_partition_facts(aligned, kAutomaticPartitionAlgorithmId);
  expect_channel_plan_partition_facts(predicate, kAutomaticPartitionAlgorithmId);
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
  const ScopedPartitionEnvironment absent{std::nullopt};
  using PartitionMutation = detail::RuntimePartitionTestPlanAccess::PartitionMutation;
  constexpr std::array mutations{
      PartitionMutation::kAcceptedAlgorithm,
      PartitionMutation::kAcceptedAlgorithmVersion,
      PartitionMutation::kRequestedAlgorithm,
      PartitionMutation::kAlgorithmMaturity,
      PartitionMutation::kInterfaceVersion,
      PartitionMutation::kDisposition,
      PartitionMutation::kSelectionSource,
      PartitionMutation::kPartitioningMode,
      PartitionMutation::kAuthoredPartitionCount,
      PartitionMutation::kAutomaticPartitionCount,
      PartitionMutation::kReadyPolicy,
      PartitionMutation::kPartitionId,
      PartitionMutation::kPartitionLane,
      PartitionMutation::kPartitionAuthoredName,
      PartitionMutation::kPartitionOperatorPath,
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
    ASSERT_TRUE(baseline.ok()) << baseline.json();
    const ExecutionPlan plan =
        detail::RuntimePartitionTestPlanAccess::mutate_partition(baseline, mutation);
    ASSERT_TRUE(plan.ok()) << plan.json();
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
    EXPECT_FALSE(test::RuntimeProbeState::setup_entered.load(std::memory_order_acquire));
    EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
  }
}

// REQ: COMPILE-068
// REQ: RUNTIME-041
TEST(DispatcherTest, RejectsNoncanonicalPartitionAuthorityBeforeCallbacks) {
  const ScopedPartitionEnvironment absent{std::nullopt};
  using PartitionMutation = detail::RuntimePartitionTestPlanAccess::PartitionMutation;
  // Key setup: mutate each canonical ordering or ownership relation independently.
  const auto expect_rejected = [](const ExecutionPlan& baseline, PartitionMutation mutation) {
    ASSERT_TRUE(baseline.ok()) << baseline.json();
    test::RuntimeProbeState::reset();
    test::RuntimeAlignedChannelProbeState::reset(test::RuntimeAlignedChannelScript::kExactBoundary);
    EncodedPartitionPathProbeState::reset();
    AutomaticFanInProbeState::reset();
    const ExecutionPlan malformed =
        detail::RuntimePartitionTestPlanAccess::mutate_partition(baseline, mutation);
    ASSERT_TRUE(malformed.ok());

    RunSession session = run_async(malformed);
    session.wait();

    // Key assertion: every malformed relation fails before any callback becomes observable.
    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_TRUE(test::has_runtime_diagnostic(session, "RUNTIME_PLACEMENT_BINDING_UNAVAILABLE"));
    EXPECT_FALSE(test::RuntimeProbeState::compute_entered.load(std::memory_order_acquire));
    EXPECT_TRUE(test::RuntimeProbeState::snapshot().empty());
    EXPECT_EQ(
        test::RuntimeAlignedChannelProbeState::source_compute_count.load(std::memory_order_acquire),
        0U);
    EXPECT_EQ(
        test::RuntimeAlignedChannelProbeState::sink_compute_count.load(std::memory_order_acquire),
        0U);
    EXPECT_EQ(EncodedPartitionPathProbeState::callbacks.load(std::memory_order_acquire), 0U);
    {
      const std::lock_guard lock{AutomaticFanInProbeState::mutex};
      EXPECT_EQ(AutomaticFanInProbeState::source_compute_counts[0], 0U);
      EXPECT_EQ(AutomaticFanInProbeState::source_compute_counts[1], 0U);
      EXPECT_TRUE(AutomaticFanInProbeState::joined_values.empty());
      EXPECT_TRUE(AutomaticFanInProbeState::drained_values.empty());
    }
  };

  expect_rejected(test::make_runtime_aligned_channel_plan(true),
                  PartitionMutation::kPartitionMemberOrder);
  expect_rejected(test::make_runtime_aligned_channel_plan(true),
                  PartitionMutation::kPartitionOperatorPathOrder);
  expect_rejected(test::make_runtime_test_plan(true), PartitionMutation::kDecisionOrder);
  expect_rejected(make_partition_interface_probe_plan(),
                  PartitionMutation::kGraphInterfaceMemberAndDecision);
  expect_rejected(test::make_runtime_test_plan(), PartitionMutation::kPartitionOperatorPathIndex);
  expect_rejected(make_automatic_fanin_drain_plan(), PartitionMutation::kScheduleEdgeOrder);
  expect_rejected(make_complete_constrained_partition_probe_plan(),
                  PartitionMutation::kConstrainedModeManual);
}

// REQ: RUNTIME-041
TEST(DispatcherTest, PartitionMutationFixturesFailClosedWhenTheRequestedFieldIsAbsent) {
  const ScopedPartitionEnvironment absent{std::nullopt};
  using PartitionMutation = detail::RuntimePartitionTestPlanAccess::PartitionMutation;
  // Key setup: request mutations whose target fields are absent from otherwise accepted plans.
  const ExecutionPlan automatic = make_automatic_multi_owner_plan();
  ASSERT_TRUE(automatic.ok()) << automatic.json();
  // Key assertion: the mutation seam fails closed instead of returning an unchanged plan.
  EXPECT_FALSE(detail::RuntimePartitionTestPlanAccess::mutate_partition(
                   automatic, PartitionMutation::kPartitioningMode)
                   .ok());
  EXPECT_FALSE(detail::RuntimePartitionTestPlanAccess::mutate_partition(
                   automatic, PartitionMutation::kPartitionAuthoredName)
                   .ok());

  // Graph topology:
  //   (empty)
  Graph graph{"runtime-zero-work-partition-mutation"};
  const ExecutionPlan zero_work = compile(graph);
  ASSERT_TRUE(zero_work.ok()) << zero_work.json();
  ASSERT_TRUE(zero_work.execution_partitions().empty());
  EXPECT_FALSE(detail::RuntimePartitionTestPlanAccess::mutate_partition(
                   zero_work, PartitionMutation::kPartitionOperatorPath)
                   .ok());
  EXPECT_FALSE(detail::RuntimePartitionTestPlanAccess::mutate_partition(
                   zero_work, PartitionMutation::kDecisionMissing)
                   .ok());
  EXPECT_FALSE(detail::RuntimePartitionTestPlanAccess::mutate_partition(
                   zero_work, PartitionMutation::kDecisionPartitionMismatch)
                   .ok());
}

}  // namespace
}  // namespace holoscan
