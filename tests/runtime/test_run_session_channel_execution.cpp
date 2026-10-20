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
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/schema/sdk_schemas.hpp>

#include "common/allocation_counter.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "partition_parity_test_support.hpp"
#include "runtime/execution_lane.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/physical_readiness_binding.hpp"
#include "runtime/predicate_evaluator.hpp"
#include "runtime/runtime_callback_context.hpp"
#include "runtime/runtime_fanout_test_control.hpp"
#include "runtime/runtime_fixture_observation.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_tensor_resources.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_test_graph.hpp"

#include "../support/optional_assertions.hpp"
#include "run_session_channel_execution_test_support.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;
using namespace test::run_session_channel_execution;

enum class RuntimeLocalEnum : std::uint16_t {
  kExpected = 0x1234U,
};

// Retain fanout delivery evidence for
// RunSessionTest.CompiledGraphRoundTripsProcessLocalEnumThroughHoloMQFanout.
struct RuntimeLocalEnumProbe {
  static void reset() {
    const std::lock_guard lock(mutex);
    observed.reset();
    deliveries = 0U;
  }

  static void record(const Sample<RuntimeLocalEnum>& sample) {
    {
      const std::lock_guard lock(mutex);
      observed = sample;
      ++deliveries;
    }
    cv.notify_all();
  }

  [[nodiscard]] static std::optional<Sample<RuntimeLocalEnum>> wait(unsigned count = 1U) {
    std::unique_lock lock(mutex);
    if (!cv.wait_for(lock, 5s, [count] { return deliveries >= count; })) {
      return std::nullopt;
    }
    return observed;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::optional<Sample<RuntimeLocalEnum>> observed;
  static inline unsigned deliveries{};
};

enum class RuntimeOtherLocalEnum : std::uint16_t { kExpected = 0x1234U };

// Emit one local enum or substitute its declaration during replay in
// RunSessionTest.LocalEnumReplayRejectsEqualWidthTypeSubstitution.
class RuntimeLocalEnumSourceOperator final : public Operator<> {
 public:
  static inline std::atomic_bool alternate_declaration{};
  void setup(OperatorSpec& spec) override {
    if (alternate_declaration.load()) {
      spec.output(alternate, "output").max_emits_per_compute(1U);
    } else {
      spec.output(output, "output").max_emits_per_compute(1U);
    }
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 10ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    if (emitted_) {
      return {};
    }
    auto result = output.emit(RuntimeLocalEnum::kExpected);
    if (result) {
      emitted_ = true;
    }
    return result;
  }

  Output<RuntimeLocalEnum> output;
  Output<RuntimeOtherLocalEnum> alternate;

 private:
  bool emitted_{};
};

// Capture the typed enum sample for RunSessionTest.CompiledGraphRoundTripsProcessLocalEnum.
class RuntimeLocalEnumSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

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
    RuntimeLocalEnumProbe::record(*sample);
    return {};
  }

  Input<RuntimeLocalEnum> input;
};

// Build direct or two-reader enum delivery for
// RunSessionTest.CompiledGraphRoundTripsProcessLocalEnumThroughHoloMQFanout.
// Graph: source.output --> sink.input; fanout also connects source.output --> second-sink.input.
[[nodiscard]] ExecutionPlan make_runtime_local_enum_plan(bool holomq_fanout = false) {
  Graph graph{"runtime-local-enum"};
  auto source = graph.op<RuntimeLocalEnumSourceOperator>("source");
  auto sink = graph.op<RuntimeLocalEnumSinkOperator>("sink");
  graph.add_flow(source->output, sink->input);
  if (holomq_fanout) {
    auto second_sink = graph.op<RuntimeLocalEnumSinkOperator>("second-sink");
    graph.add_flow(source->output, second_sink->input);
  }
  graph.partition("runtime-local-enum").include_all();
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return holomq_fanout ? detail::compile_with_process_local_holomq(graph, {}) : compile(graph);
}

enum class RuntimeMixedPolicyFanoutBranch : std::uint8_t {
  kBlocking,
  kDropOldest,
};

struct RuntimeMixedPolicyFanoutObservation {
  std::int32_t value{};
  std::uint64_t sequence{};
};

// Records the two policy-specific subscriber streams produced by one public Output port.
struct RuntimeMixedPolicyFanoutProbe {
  static void reset() {
    const std::lock_guard lock(mutex);
    blocking.clear();
    drop_oldest.clear();
  }

  static void record(RuntimeMixedPolicyFanoutBranch branch, const Sample<std::int32_t>& sample) {
    {
      const std::lock_guard lock(mutex);
      auto& observations =
          branch == RuntimeMixedPolicyFanoutBranch::kBlocking ? blocking : drop_oldest;
      observations.push_back(RuntimeMixedPolicyFanoutObservation{
          .value = sample.data,
          .sequence = sample.metadata.sequence,
      });
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_both(std::size_t expected) {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [expected] {
      return blocking.size() >= expected && drop_oldest.size() >= expected;
    });
  }

  [[nodiscard]] static std::array<std::vector<RuntimeMixedPolicyFanoutObservation>, 2U> snapshot() {
    const std::lock_guard lock(mutex);
    return {blocking, drop_oldest};
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::vector<RuntimeMixedPolicyFanoutObservation> blocking;
  static inline std::vector<RuntimeMixedPolicyFanoutObservation> drop_oldest;
};

// Emits two publications far enough apart for both capacity-one branches to retire each sample.
class RuntimeMixedPolicyFanoutSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 10ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    if (next_value_ > 2) {
      return {};
    }
    auto emitted = output.emit(next_value_);
    if (emitted) {
      ++next_value_;
    }
    return emitted;
  }

  Output<std::int32_t> output;

 private:
  std::int32_t next_value_{1};
};

// Records one branch of a mixed-policy fanout without changing its temporal contract.
class RuntimeMixedPolicyFanoutSinkOperator final : public Operator<> {
 public:
  explicit RuntimeMixedPolicyFanoutSinkOperator(RuntimeMixedPolicyFanoutBranch branch)
      : branch_(branch) {}

  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

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
    RuntimeMixedPolicyFanoutProbe::record(branch_, *sample);
    return {};
  }

  Input<std::int32_t> input;

 private:
  RuntimeMixedPolicyFanoutBranch branch_;
};

[[nodiscard]] ExecutionPlan make_runtime_mixed_policy_fanout_plan() {
  Graph graph{"runtime-mixed-policy-fanout"};
  auto source = graph.op<RuntimeMixedPolicyFanoutSourceOperator>("source");
  auto blocking = graph.op<RuntimeMixedPolicyFanoutSinkOperator>(
      "blocking", RuntimeMixedPolicyFanoutBranch::kBlocking);
  auto drop_oldest = graph.op<RuntimeMixedPolicyFanoutSinkOperator>(
      "drop-oldest", RuntimeMixedPolicyFanoutBranch::kDropOldest);
  graph.add_flow(source->output, blocking->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(source->output,
                 drop_oldest->input,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});
  graph.partition("runtime-mixed-policy-fanout").include_all();
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  // The fanout control keeps this blocking fanout on HoloMQ, whose behavior these tests cover;
  // compile() would deliver it directly (CHAN-167).
  return detail::compile_with_process_local_holomq_fanout(graph, CompileOptions{});
}

enum class RuntimeCompletionPhysicalRoot : std::uint8_t {
  kReadable,
  kDoorbell,
};

struct RuntimeCompletionPhysicalProbe {
  struct Snapshot {
    std::size_t source_calls{};
    std::vector<std::uint32_t> sink_values;
  };

  static void reset() {
    const std::lock_guard lock(mutex);
    source_calls = 0U;
    sink_values.clear();
  }

  [[nodiscard]] static std::uint32_t enter_source() {
    std::uint32_t call = 0U;
    {
      const std::lock_guard lock(mutex);
      ++source_calls;
      call = static_cast<std::uint32_t>(source_calls);
    }
    cv.notify_all();
    return call;
  }

  static void record_sink(std::uint32_t value) {
    {
      const std::lock_guard lock(mutex);
      sink_values.push_back(value);
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_source_calls(std::size_t expected) {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [expected] { return source_calls >= expected; });
  }

  [[nodiscard]] static Snapshot snapshot() {
    const std::lock_guard lock(mutex);
    return Snapshot{.source_calls = source_calls, .sink_values = sink_values};
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::size_t source_calls{};
  static inline std::vector<std::uint32_t> sink_values;
};

class RuntimeCompletionPhysicalSourceOperator final : public Operator<> {
 public:
  explicit RuntimeCompletionPhysicalSourceOperator(RuntimeCompletionPhysicalRoot root) noexcept
      : root_(root) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
    if (root_ == RuntimeCompletionPhysicalRoot::kReadable) {
      spec.readable_source(readable_, "completion-readable");
    } else {
      spec.doorbell_source(doorbell_, "completion-doorbell").capacity(3U);
    }
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    if (root_ == RuntimeCompletionPhysicalRoot::kReadable) {
      result.trigger(OnReadable{readable_});
    } else {
      result.trigger(OnDoorbell{doorbell_});
    }
    result.complete_after(ComputeCalls{.limit = 2U});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    return output.emit(RuntimeCompletionPhysicalProbe::enter_source());
  }

  Output<std::uint32_t> output;

 private:
  RuntimeCompletionPhysicalRoot root_;
  ReadableSource readable_;
  DoorbellSource doorbell_;
};

class RuntimeCompletionPhysicalSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(3U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto received = input.receive();
    if (!received) {
      return make_unexpected(std::move(received).error());
    }
    RuntimeCompletionPhysicalProbe::record_sink(received->data);
    return {};
  }

  Input<std::uint32_t> input;
};

[[nodiscard]] ExecutionPlan make_runtime_completion_physical_plan(
    RuntimeCompletionPhysicalRoot root) {
  Graph graph{root == RuntimeCompletionPhysicalRoot::kReadable ? "runtime-completion-readable"
                                                               : "runtime-completion-doorbell"};
  const auto source = graph.op<RuntimeCompletionPhysicalSourceOperator>("source", root);
  const auto sink = graph.op<RuntimeCompletionPhysicalSinkOperator>("sink");
  graph.partition("runtime-completion-physical-source").add(source);
  graph.partition("runtime-completion-physical-sink").add(sink);
  graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 3U});
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

// Materializes the real bounded physical backend from Runtime's exact cold binding request, then
// exposes only its transport-neutral control so tests can supply readable or doorbell evidence.
class RuntimeCompletionPhysicalBindingFactory final : public detail::RuntimePortBindingFactory {
 public:
  struct BoundControl {
    detail::PhysicalReadinessControl control;
    ClockId reference_clock{};
  };

  explicit RuntimeCompletionPhysicalBindingFactory(RuntimeCompletionPhysicalRoot root) noexcept
      : root_(root) {}

  [[nodiscard]] expected<std::unique_ptr<detail::PortBinding>, Error> bind(
      const detail::RuntimePortBindingRequest& request,
      TimePoint start_coordinate) noexcept override {
    const detail::RuntimeEventKind expected_kind = root_ == RuntimeCompletionPhysicalRoot::kReadable
                                                       ? detail::RuntimeEventKind::kReadable
                                                       : detail::RuntimeEventKind::kDoorbell;
    if (request.kind != expected_kind) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const detail::PhysicalReadinessBindingDescriptor descriptor{
        .request = request,
        .wake_requirement = ReadinessWakeRequirement::kDeploymentSelected,
        .wake_mode = WakeMode::kSleepingWaitset,
        .execution_lane = request.execution_lane,
    };
    auto physical = detail::PhysicalReadinessBindingFactory::create(
        std::span<const detail::PhysicalReadinessBindingDescriptor>{&descriptor, 1U});
    if (!physical) {
      return make_unexpected(std::move(physical.error()));
    }
    auto binding = (*physical)->bind(request, start_coordinate);
    if (!binding) {
      return make_unexpected(std::move(binding.error()));
    }
    auto control = (*physical)->control(
        request.source, request.plan_generation, request.run_generation, request.source_generation);
    if (!control) {
      return make_unexpected(std::move(control.error()));
    }
    {
      const std::lock_guard lock(mutex_);
      physical_ = std::move(*physical);
      bound_ =
          BoundControl{.control = std::move(*control), .reference_clock = request.reference_clock};
    }
    cv_.notify_all();
    return std::move(*binding);
  }

  [[nodiscard]] std::optional<BoundControl> wait_for_control() const {
    std::unique_lock lock(mutex_);
    if (!cv_.wait_for(lock, 5s, [&] { return bound_.has_value(); })) {
      return std::nullopt;
    }
    return bound_;
  }

 private:
  RuntimeCompletionPhysicalRoot root_;
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
  std::shared_ptr<detail::PhysicalReadinessBindingFactory> physical_;
  std::optional<BoundControl> bound_;
};

[[nodiscard]] bool wait_for_readable_rearm(const detail::PhysicalReadinessControl& control,
                                           std::uint64_t wake_epoch_before_publication) {
  constexpr std::uint64_t kPublicationAndRetirementWakes = 2U;
  if (wake_epoch_before_publication >
      std::numeric_limits<std::uint64_t>::max() - kPublicationAndRetirementWakes) {
    return false;
  }
  const std::uint64_t rearmed_epoch =
      wake_epoch_before_publication + kPublicationAndRetirementWakes;
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (control.wake_epoch() < rearmed_epoch && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  return control.wake_epoch() >= rearmed_epoch;
}

void expect_runtime_completion_physical_outcome(const RunSession& session,
                                                const ExecutionPlan& plan) {
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const RuntimeCompletionPhysicalProbe::Snapshot observed =
      RuntimeCompletionPhysicalProbe::snapshot();
  EXPECT_EQ(observed.source_calls, 2U);
  EXPECT_EQ(observed.sink_values, (std::vector<std::uint32_t>{1U, 2U}));

  ASSERT_EQ(plan.operator_completion_rules().size(), 1U);
  ASSERT_EQ(session.operator_completion_progress().size(), 1U);
  const auto& rule = plan.operator_completion_rules().front();
  const auto& progress = session.operator_completion_progress().front();
  EXPECT_EQ(progress.owner.id, rule.owner.id);
  EXPECT_EQ(progress.kind, OperatorCompletionRuleKind::kComputeCalls);
  EXPECT_EQ(progress.configured_limit, 2U);
  EXPECT_EQ(progress.successful_compute_calls, 2U);
  EXPECT_TRUE(progress.limit_reached);
}

struct RuntimeClosedIslandAllocationProbe {
  static void begin(void* opaque) noexcept {
    auto& self = *static_cast<RuntimeClosedIslandAllocationProbe*>(opaque);
    const std::uint32_t transaction =
        self.transaction_count.fetch_add(1U, std::memory_order_acq_rel) + 1U;
    measuring = transaction == 2U;
    if (measuring) {
      self.observed.store(false, std::memory_order_release);
      active.emplace();
    }
  }

  static void end(void* opaque) noexcept {
    if (!measuring) {
      return;
    }
    auto& self = *static_cast<RuntimeClosedIslandAllocationProbe*>(opaque);
    const std::size_t count = active ? active->count() : std::numeric_limits<std::size_t>::max();
    active.reset();
    self.allocations.store(count, std::memory_order_release);
    self.observed.store(true, std::memory_order_release);
    measuring = false;
  }

  static inline thread_local std::optional<test::ScopedAllocationCounter> active;
  static inline thread_local bool measuring{};
  std::atomic<std::uint32_t> transaction_count{};
  std::atomic<std::size_t> allocations{std::numeric_limits<std::size_t>::max()};
  std::atomic<bool> observed{};
};

struct RuntimeClosedIslandCompletionProbe {
  struct Snapshot {
    std::array<detail::RuntimeCallbackIdentity, 2U> identities{};
    std::array<ErrorCode, 2U> errors{};
    std::array<std::uint64_t, detail::kRuntimeDirectCompositePhaseDeltaCount> phase_deltas{};
    ErrorCode first_error{ErrorCode::kSuccess};
    std::uint8_t callbacks_attempted{};
    std::uint8_t callbacks_completed{};
  };

  static void capture(void* opaque,
                      const detail::RuntimeDirectCompositeCompletionReport& report) noexcept {
    auto& self = *static_cast<RuntimeClosedIslandCompletionProbe*>(opaque);
    const std::lock_guard lock(self.mutex);
    if (!self.snapshot) {
      self.snapshot = Snapshot{
          .identities = report.callback_identities,
          .errors = report.callback_errors,
          .phase_deltas = report.phase_deltas,
          .first_error = report.first_error.code,
          .callbacks_attempted = report.callbacks_attempted,
          .callbacks_completed = report.callbacks_completed,
      };
    }
  }

  [[nodiscard]] std::optional<Snapshot> read() const {
    const std::lock_guard lock(mutex);
    return snapshot;
  }

  mutable std::mutex mutex;
  std::optional<Snapshot> snapshot;
};

void expect_sample_metadata_equal(const SampleMetadata& expected, const SampleMetadata& actual) {
  EXPECT_EQ(actual.version.major, expected.version.major);
  EXPECT_EQ(actual.version.minor, expected.version.minor);
  EXPECT_EQ(actual.sequence, expected.sequence);
  EXPECT_EQ(actual.capture_timestamp_ns, expected.capture_timestamp_ns);
  EXPECT_EQ(actual.source_clock_id, expected.source_clock_id);
  EXPECT_EQ(actual.publish_timestamp_ns, expected.publish_timestamp_ns);
  EXPECT_EQ(actual.publish_clock_id, expected.publish_clock_id);
  EXPECT_EQ(actual.trace_id, expected.trace_id);
  EXPECT_EQ(actual.frame_id, expected.frame_id);
  EXPECT_EQ(actual.byte_size, expected.byte_size);
  EXPECT_EQ(actual.payload_kind, expected.payload_kind);
  EXPECT_EQ(actual.flags, expected.flags);
}

void expect_caller_sample_metadata_preserved(const SampleMetadata& expected,
                                             const SampleMetadata& actual) {
  EXPECT_EQ(actual.version.major, expected.version.major);
  EXPECT_EQ(actual.version.minor, expected.version.minor);
  EXPECT_EQ(actual.capture_timestamp_ns, expected.capture_timestamp_ns);
  EXPECT_EQ(actual.source_clock_id, expected.source_clock_id);
  EXPECT_EQ(actual.trace_id, expected.trace_id);
  EXPECT_EQ(actual.frame_id, expected.frame_id);
  EXPECT_EQ(actual.flags, expected.flags);
}

// Runtime stamps publication time from the Graph default clock at the submission boundary, so a
// consumer must observe a coordinate in that domain rather than canonical absence. Comparing the
// clock identity against the plan catalog is what makes the coordinate usable: it proves the
// consumer can place the sample on the same timeline the runtime published it against.
void expect_runtime_publication_coordinate(const ExecutionPlan& plan,
                                           const SampleMetadata& actual) {
  const ClockId publish_clock = plan.clock_catalog().root_default_clock;
  ASSERT_TRUE(publish_clock.valid());
  EXPECT_EQ(actual.publish_clock_id, publish_clock);
  EXPECT_TRUE(actual.publish_time().valid());
  EXPECT_GT(actual.publish_timestamp_ns, 0);
}

void expect_input_selection_equal(InputSelection expected, InputSelection actual) {
  EXPECT_EQ(actual.state, expected.state);
  EXPECT_EQ(actual.reason, expected.reason);
}

// Replay must compare exact process-local C++ type identity, even for equal-width enums.
// REQ: GRAPH-042
TEST(RunSessionTest, LocalEnumReplayRejectsEqualWidthTypeSubstitution) {
  // Key setup: replay a compiled enum output using a distinct enum with the same native width.
  // Key assertion: exact replay succeeds; the substituted type fails with a port mismatch.
  // Graph:
  //   source.output --> sink.input [RuntimeLocalEnum]
  RuntimeLocalEnumSourceOperator::alternate_declaration.store(false);
  const auto plan = make_runtime_local_enum_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto exact = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                              RunGenerationId{81U});
  ASSERT_TRUE(exact) << exact.error().message;
  RuntimeLocalEnumSourceOperator::alternate_declaration.store(true);
  detail::RuntimeReplayMismatchDetail mismatch;
  auto changed = detail::GraphExec::reconstruct(
      detail::ExecutionPlanAccess::retain(plan), RunGenerationId{82U}, {}, &mismatch);
  RuntimeLocalEnumSourceOperator::alternate_declaration.store(false);
  ASSERT_FALSE(changed);
  EXPECT_EQ(changed.error().message, "RUNTIME_PORT_BINDING_UNAVAILABLE");
  EXPECT_EQ(mismatch.category, detail::RuntimeReplayMismatchCategory::kPorts);
}

// A local enum retains its exact C++ type for replay while the physical plan freezes only the
// native width required by the private process-local channel.
// REQ: GRAPH-042
// REQ: CHAN-096
// REQ: CHAN-147
// REQ: RUNTIME-019
TEST(RunSessionTest, CompiledGraphRoundTripsProcessLocalEnum) {
  // Key setup: compile and run one local enum flow using direct retained delivery.
  // Key assertion: the enum value, native byte size, inline kind, and stamped sequence survive.
  // Graph:
  //   source.output --> sink.input [RuntimeLocalEnum]
  RuntimeLocalEnumProbe::reset();
  const ExecutionPlan plan = make_runtime_local_enum_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->process_local_channel_services.size(), 1U);
  EXPECT_EQ(state->process_local_channel_services.front().payload_size_bytes,
            sizeof(RuntimeLocalEnum));
  EXPECT_EQ(state->process_local_channel_services.front().delivery_kind,
            detail::ProcessLocalChannelDeliveryKind::kDirectRetained);

  RunSession session = run_async(plan);
  const auto observed = RuntimeLocalEnumProbe::wait();
  session.request_stop();
  session.wait();

  if (!observed.has_value()) {
    ADD_FAILURE() << "local enum round trip omitted the expected sample";
    return;
  }
  EXPECT_EQ(observed->data, RuntimeLocalEnum::kExpected);
  EXPECT_EQ(observed->metadata.byte_size, sizeof(RuntimeLocalEnum));
  EXPECT_EQ(observed->metadata.payload_kind, PayloadKind::kInlineTyped);
  EXPECT_GT(observed->metadata.sequence, 0U);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// The conservative HoloMQ path must preserve the same local enum value for every fanout reader.
// REQ: GRAPH-042
// REQ: CHAN-147
TEST(RunSessionTest, CompiledGraphRoundTripsProcessLocalEnumThroughHoloMQFanout) {
  // Key setup: force one local enum output through HoloMQ to two readers.
  // Key assertion: both deliveries arrive with the expected enum value and metadata.
  // Graph:
  //   source.output --+--> sink.input
  //                   +--> second-sink.input
  RuntimeLocalEnumProbe::reset();
  const ExecutionPlan plan = make_runtime_local_enum_plan(true);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->process_local_channel_services.size(), 1U);
  ASSERT_EQ(state->process_local_channel_subscribers.size(), 2U);
  EXPECT_EQ(state->process_local_channel_services.front().delivery_kind,
            detail::ProcessLocalChannelDeliveryKind::kHoloMQ);
  EXPECT_EQ(state->process_local_channel_services.front().payload_size_bytes,
            sizeof(RuntimeLocalEnum));

  RunSession session = run_async(plan);
  const auto observed = RuntimeLocalEnumProbe::wait(2U);
  session.request_stop();
  session.wait();

  if (!observed.has_value()) {
    ADD_FAILURE() << "local enum fanout omitted the expected samples";
    return;
  }
  EXPECT_EQ(observed->data, RuntimeLocalEnum::kExpected);
  EXPECT_EQ(observed->metadata.byte_size, sizeof(RuntimeLocalEnum));
  EXPECT_EQ(observed->metadata.payload_kind, PayloadKind::kInlineTyped);
  EXPECT_GT(observed->metadata.sequence, 0U);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
}

// Process-local service identities derive from the plan, while HoloMQ service names are visible to
// every process on the host. Concurrent sessions of one plan, here in one process, must still own
// disjoint transport endpoints instead of exhausting or reading each other's services.
// REQ: CHAN-147
// REQ: CHAN-151
TEST(RunSessionTest, ConcurrentSessionsOfOnePlanKeepProcessLocalServicesDisjoint) {
  RuntimeLocalEnumProbe::reset();
  const ExecutionPlan plan = make_runtime_local_enum_plan(true);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->process_local_channel_services.size(), 1U);
  EXPECT_EQ(state->process_local_channel_services.front().delivery_kind,
            detail::ProcessLocalChannelDeliveryKind::kHoloMQ);

  RunSession first = run_async(plan);
  RunSession second = run_async(plan);
  const auto observed = RuntimeLocalEnumProbe::wait(4U);
  first.request_stop();
  second.request_stop();
  first.wait();
  second.wait();

  EXPECT_TRUE(observed.has_value()) << "each session must deliver to both of its own readers";
  for (const RunSession* session : {&first, &second}) {
    EXPECT_EQ(session->termination(), RunTermination::kStopped);
    EXPECT_TRUE(session->diagnostics().empty())
        << (session->diagnostics().empty() ? "" : session->diagnostics().front().token);
  }
  const std::lock_guard lock(RuntimeLocalEnumProbe::mutex);
  EXPECT_EQ(RuntimeLocalEnumProbe::deliveries, 4U);
}

// This is the production path rather than a manually assembled RuntimeChannelSession: compile()
// freezes the POD identity/layout, run_async() reconstructs both typed ports, and the public
// Output<T>::emit(Sample<T>) overload retains the caller-authored metadata while Runtime stamps
// publication-owned sequence, publication time, and representation fields.
// REQ: CHAN-074
// REQ: CHAN-096
// REQ: RUNTIME-019
TEST(RunSessionTest, CompiledGraphRoundTripsCustomPodWithCallerSampleMetadata) {
  // Key setup: compile() registers the POD identity/layout; run_async() reconstructs
  // both typed ports and stamps sequence/byte_size while preserving caller metadata.
  reset_payload_round_trip_probe();
  const ExecutionPlan plan = make_runtime_pod_round_trip_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool observed = wait_for_pod();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const auto expected = expected_pod_snapshot();
  const auto received = observed_pod_snapshot();
  if (!expected.has_value() || !received.has_value()) {
    ADD_FAILURE() << "custom-POD round trip omitted expected or observed evidence";
    return;
  }
  const auto& expected_sample = *expected;
  const auto& received_sample = *received;
  // Key assertion: field values and caller-authored metadata fields are identical;
  // sequence is > 0 (runtime-stamped) and payload_kind is kInlineTyped.
  EXPECT_EQ(received_sample.pod.first, expected_sample.pod.first);
  EXPECT_EQ(received_sample.pod.second, expected_sample.pod.second);
  expect_caller_sample_metadata_preserved(expected_sample.metadata, received_sample.metadata);
  EXPECT_GT(received_sample.metadata.sequence, 0U);
  EXPECT_EQ(received_sample.metadata.byte_size, sizeof(RuntimeChannelRoundTripPod));
  EXPECT_EQ(received_sample.metadata.payload_kind, PayloadKind::kInlineTyped);
  expect_runtime_publication_coordinate(plan, received_sample.metadata);
}

// Generated roots use the same concise Output<T>::emit(Sample<T>) call, but the compiled runtime
// routes the owning value through the package encoder/verifier/decoder rather than transporting its
// nontrivial C++ object representation.
// REQ: CHAN-075
// REQ: CHAN-120
// REQ: CHAN-097
// REQ: RUNTIME-019
TEST(RunSessionTest, CompiledGraphRoundTripsGeneratedSchemaWithCallerSampleMetadata) {
  reset_payload_round_trip_probe();
  const ExecutionPlan plan = make_runtime_schema_round_trip_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool observed = wait_for_schema();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const auto expected = expected_schema_snapshot();
  const auto received = observed_schema_snapshot();
  if (!expected.has_value() || !received.has_value()) {
    ADD_FAILURE() << "schema round trip omitted expected or observed evidence";
    return;
  }
  const auto& expected_sample = *expected;
  const auto& received_sample = *received;
  EXPECT_EQ(received_sample.schema_value, expected_sample.schema_value);
  expect_caller_sample_metadata_preserved(expected_sample.metadata, received_sample.metadata);
  EXPECT_GT(received_sample.metadata.sequence, 0U);
  EXPECT_GT(received_sample.metadata.byte_size, received_sample.schema_value.size());
  EXPECT_EQ(received_sample.metadata.payload_kind, PayloadKind::kSerializedBytes);
  expect_runtime_publication_coordinate(plan, received_sample.metadata);
}

// A direct-retained publication may carry one allocation-free continuation receipt back through
// the completed source callback. The stable partition owner performs the existing destination
// readiness poll inline, then validates and ingests its receipt; disabling only that elision keeps
// delivery direct and provides an otherwise-identical control arm.
// REQ: RUNTIME-011
// REQ: RUNTIME-019
TEST(RunSessionTest, DirectContinuationElidesOnePollForPodAndGeneratedSchema) {
  struct Case {
    RuntimeDirectContinuationPayloadKind payload_kind;
    std::string_view label;
  };
  constexpr std::array cases{
      Case{RuntimeDirectContinuationPayloadKind::kPod, "custom POD"},
      Case{RuntimeDirectContinuationPayloadKind::kGeneratedSchema, "generated schema"},
  };
  const auto count = [](const RuntimeDirectContinuationRoundTripEvidence& evidence,
                        detail::RuntimePhaseCounter counter) {
    return evidence.counters[detail::RuntimePhaseCounters::index(counter)];
  };

  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.label);
    const RuntimeDirectContinuationRoundTripEvidence optimized =
        run_direct_continuation_round_trip(test_case.payload_kind, false);
    const RuntimeDirectContinuationRoundTripEvidence control =
        run_direct_continuation_round_trip(test_case.payload_kind, true);

    ASSERT_TRUE(optimized.direct_delivery);
    ASSERT_TRUE(control.direct_delivery);
    ASSERT_TRUE(optimized.observed);
    ASSERT_TRUE(control.observed);
    EXPECT_EQ(optimized.termination, RunTermination::kStopped);
    EXPECT_EQ(control.termination, RunTermination::kStopped);
    EXPECT_EQ(optimized.diagnostic_count, 0U);
    EXPECT_EQ(control.diagnostic_count, 0U);
    ASSERT_TRUE(optimized.expected.has_value());
    ASSERT_TRUE(optimized.received.has_value());
    ASSERT_TRUE(control.expected.has_value());
    ASSERT_TRUE(control.received.has_value());

    if (test_case.payload_kind == RuntimeDirectContinuationPayloadKind::kPod) {
      EXPECT_EQ(optimized.received->pod.first, optimized.expected->pod.first);
      EXPECT_EQ(optimized.received->pod.second, optimized.expected->pod.second);
      EXPECT_EQ(control.received->pod.first, control.expected->pod.first);
      EXPECT_EQ(control.received->pod.second, control.expected->pod.second);
      EXPECT_EQ(optimized.received->pod.first, control.received->pod.first);
      EXPECT_EQ(optimized.received->pod.second, control.received->pod.second);
      EXPECT_EQ(optimized.received->metadata.byte_size, sizeof(RuntimeChannelRoundTripPod));
      EXPECT_EQ(control.received->metadata.byte_size, sizeof(RuntimeChannelRoundTripPod));
      EXPECT_EQ(optimized.received->metadata.payload_kind, PayloadKind::kInlineTyped);
      EXPECT_EQ(control.received->metadata.payload_kind, PayloadKind::kInlineTyped);
    } else {
      EXPECT_EQ(optimized.received->schema_value, optimized.expected->schema_value);
      EXPECT_EQ(control.received->schema_value, control.expected->schema_value);
      EXPECT_EQ(optimized.received->schema_value, control.received->schema_value);
      EXPECT_GT(optimized.received->metadata.byte_size, optimized.received->schema_value.size());
      EXPECT_GT(control.received->metadata.byte_size, control.received->schema_value.size());
      EXPECT_EQ(optimized.received->metadata.payload_kind, PayloadKind::kSerializedBytes);
      EXPECT_EQ(control.received->metadata.payload_kind, PayloadKind::kSerializedBytes);
    }
    expect_caller_sample_metadata_preserved(optimized.expected->metadata,
                                            optimized.received->metadata);
    expect_caller_sample_metadata_preserved(control.expected->metadata, control.received->metadata);
    EXPECT_EQ(optimized.received->metadata.trace_id, control.received->metadata.trace_id);
    EXPECT_EQ(optimized.received->metadata.frame_id, control.received->metadata.frame_id);
    EXPECT_EQ(optimized.received->metadata.flags, control.received->metadata.flags);
    EXPECT_GT(optimized.received->metadata.sequence, 0U);
    EXPECT_GT(control.received->metadata.sequence, 0U);

    EXPECT_EQ(count(optimized, detail::RuntimePhaseCounter::kDirectContinuationReceiptsAccepted),
              1U);
    EXPECT_EQ(count(optimized, detail::RuntimePhaseCounter::kDirectContinuationPollElisions), 1U);
    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kDirectContinuationReceiptsAccepted), 0U);
    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kDirectContinuationPollElisions), 0U);
    EXPECT_EQ(count(optimized, detail::RuntimePhaseCounter::kReadinessPollDispatches), 0U);
    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kReadinessPollDispatches), 0U);
    EXPECT_EQ(count(optimized, detail::RuntimePhaseCounter::kLaneDispatcherDispatches), 0U);
    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kLaneDispatcherDispatches), 0U);
    EXPECT_EQ(count(optimized, detail::RuntimePhaseCounter::kLaneDispatches), 0U);
    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kLaneDispatches), 0U);
    EXPECT_EQ(count(optimized, detail::RuntimePhaseCounter::kLaneComputeDispatches), 0U);
    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kLaneComputeDispatches), 0U);
    EXPECT_EQ(count(optimized, detail::RuntimePhaseCounter::kComputeDispatches),
              count(control, detail::RuntimePhaseCounter::kComputeDispatches));
    EXPECT_EQ(count(optimized, detail::RuntimePhaseCounter::kActivationsCompleted),
              count(control, detail::RuntimePhaseCounter::kActivationsCompleted));
  }
}

// Runtime reconstructs the compiler-proven closed-island authority before creating activation
// backends or entering user callbacks. Any private-record drift fails closed.
// REQ: RUNTIME-004
// REQ: RUNTIME-011
// REQ: RUNTIME-039
TEST(RunSessionTest, RejectsClosedIslandReplayMismatchBeforeCallbacks) {
  const ExecutionPlan baseline = make_runtime_pod_round_trip_plan();
  const auto baseline_state = detail::ExecutionPlanAccess::retain(baseline);
  ASSERT_NE(baseline_state, nullptr);
  ASSERT_EQ(baseline_state->closed_island_activations.size(), 1U);

  constexpr std::array mutations{
      detail::RuntimeTestPlanAccess::ClosedIslandMutation::kExecutionLane,
      detail::RuntimeTestPlanAccess::ClosedIslandMutation::kReadinessRoot,
      detail::RuntimeTestPlanAccess::ClosedIslandMutation::kReadinessSource,
      detail::RuntimeTestPlanAccess::ClosedIslandMutation::kReadinessClock,
      detail::RuntimeTestPlanAccess::ClosedIslandMutation::kEdge,
  };
  for (const auto mutation : mutations) {
    SCOPED_TRACE(static_cast<unsigned>(mutation));
    reset_payload_round_trip_probe();
    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    ExecutionPlan malformed =
        detail::RuntimeTestPlanAccess::corrupt_closed_island_activation_for_test(baseline,
                                                                                 mutation);
    malformed = detail::RuntimeTestPlanAccess::with_phase_counters(malformed, counters);
    ASSERT_TRUE(malformed.ok()) << malformed.json();

    RunSession session = run_async(malformed);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    ASSERT_EQ(session.diagnostics().size(), 1U);
    EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_REPLAY_MISMATCH");
    EXPECT_EQ(payload_callback_evidence().callback_count, 0U);
    EXPECT_FALSE(expected_pod_snapshot());
    EXPECT_FALSE(observed_pod_snapshot());
    const auto snapshot = counters->snapshot();
    EXPECT_EQ(
        snapshot[detail::RuntimePhaseCounters::index(detail::RuntimePhaseCounter::kLaneDispatches)],
        0U);
  }
}

// The compiler-proven closed island executes the same two public callbacks and exact payload
// transaction as the Checkpoint-1 control. Only the lane-task geometry changes.
// REQ: RUNTIME-011
// REQ: RUNTIME-019
// REQ: RUNTIME-039
TEST(RunSessionTest, ClosedIslandCompletionAcceptsFreshLogicalSuccessorCoordinate) {
  detail::RuntimeDirectCompositeCompletionReport report;
  report.callback_identities[0] = detail::RuntimeCallbackIdentity{
      .context = detail::RuntimeCallbackContext::kCompute,
      .operator_id = 1U,
      .tag = Tag{.sequence = 1U, .generation = 1U},
      .tag_time = TagTime{.logical_ns = 100, .microstep = 7U},
  };
  report.callback_identities[1] = detail::RuntimeCallbackIdentity{
      .context = detail::RuntimeCallbackContext::kCompute,
      .operator_id = 2U,
      .tag = Tag{.sequence = 1U, .generation = 1U},
      .tag_time = TagTime{.logical_ns = 101, .microstep = 0U},
  };
  report.phase_deltas = {2U, 2U, 2U, 2U, 1U};
  report.continuation = detail::RuntimeDirectCompositeContinuationDisposition::kAccepted;
  report.resume = detail::RuntimeDirectCompositeResumeDisposition::kCompositeCompleted;
  report.callbacks_attempted = 2U;
  report.callbacks_completed = 2U;
  report.source_retired = true;
  report.successor_retired = true;
  report.activation_cut_completed = true;

  EXPECT_TRUE(report.complete());
  report.callback_identities[1].tag_time.microstep = 1U;
  EXPECT_FALSE(report.complete());
}

// REQ: RUNTIME-039
TEST(RunSessionTest, ClosedIslandCompositeMatchesOrdinaryPodAndGeneratedSchema) {
  struct Case {
    RuntimeDirectContinuationPayloadKind payload_kind;
    std::string_view label;
  };
  constexpr std::array cases{
      Case{RuntimeDirectContinuationPayloadKind::kPod, "custom POD"},
      Case{RuntimeDirectContinuationPayloadKind::kGeneratedSchema, "generated schema"},
  };
  const auto count = [](const RuntimeDirectContinuationRoundTripEvidence& evidence,
                        detail::RuntimePhaseCounter counter) {
    return evidence.counters[detail::RuntimePhaseCounters::index(counter)];
  };

  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.label);
    RuntimeClosedIslandCompletionProbe completion_probe;
    const auto completion_control = std::make_shared<detail::RuntimeClosedIslandTestControl>();
    completion_control->set_completion_hook(&completion_probe,
                                            &RuntimeClosedIslandCompletionProbe::capture);
    const RuntimeDirectContinuationRoundTripEvidence candidate = run_direct_continuation_round_trip(
        test_case.payload_kind, false, false, false, completion_control);
    const RuntimeDirectContinuationRoundTripEvidence control = run_direct_continuation_round_trip(
        test_case.payload_kind, false, false, /*disable_closed_island=*/true);

    ASSERT_TRUE(candidate.direct_delivery);
    ASSERT_TRUE(control.direct_delivery);
    ASSERT_TRUE(candidate.observed);
    ASSERT_TRUE(control.observed);
    EXPECT_EQ(candidate.termination, RunTermination::kStopped);
    EXPECT_EQ(control.termination, RunTermination::kStopped);
    EXPECT_EQ(candidate.diagnostic_count, 0U);
    EXPECT_EQ(control.diagnostic_count, 0U);
    ASSERT_TRUE(candidate.expected);
    ASSERT_TRUE(candidate.received);
    ASSERT_TRUE(control.expected);
    ASSERT_TRUE(control.received);
    EXPECT_EQ(candidate.received->metadata.trace_id, control.received->metadata.trace_id);
    EXPECT_EQ(candidate.received->metadata.frame_id, control.received->metadata.frame_id);
    EXPECT_EQ(candidate.received->metadata.flags, control.received->metadata.flags);
    if (test_case.payload_kind == RuntimeDirectContinuationPayloadKind::kPod) {
      EXPECT_EQ(candidate.received->pod.first, control.received->pod.first);
      EXPECT_EQ(candidate.received->pod.second, control.received->pod.second);
    } else {
      EXPECT_EQ(candidate.received->schema_value, control.received->schema_value);
    }

    EXPECT_EQ(candidate.callbacks.order, (std::array<std::uint8_t, 2U>{1U, 2U}));
    EXPECT_EQ(candidate.callbacks.callback_count, 2U);
    EXPECT_EQ(candidate.callbacks.active_callbacks, 0U);
    EXPECT_EQ(candidate.callbacks.max_active_callbacks, 1U);
    EXPECT_FALSE(candidate.callbacks.overlap);
    EXPECT_TRUE(candidate.callbacks.same_thread);
    EXPECT_EQ(candidate.callbacks.identities[0].context, detail::RuntimeCallbackContext::kCompute);
    EXPECT_EQ(candidate.callbacks.identities[1].context, detail::RuntimeCallbackContext::kCompute);
    EXPECT_NE(candidate.callbacks.identities[0].operator_id, 0U);
    EXPECT_NE(candidate.callbacks.identities[1].operator_id, 0U);
    EXPECT_NE(candidate.callbacks.identities[0].operator_id,
              candidate.callbacks.identities[1].operator_id);
    EXPECT_EQ(candidate.callbacks.identities[0].tag_time.logical_ns,
              candidate.callbacks.identities[1].tag_time.logical_ns);
    ASSERT_NE(candidate.callbacks.identities[0].tag_time.microstep,
              std::numeric_limits<std::uint32_t>::max());
    EXPECT_EQ(candidate.callbacks.identities[1].tag_time.microstep,
              candidate.callbacks.identities[0].tag_time.microstep + 1U);
    const auto completion = completion_probe.read();
    ASSERT_TRUE(completion.has_value());
    EXPECT_EQ(completion->identities[0].operator_id, candidate.callbacks.identities[0].operator_id);
    EXPECT_EQ(completion->identities[0].tag_time, candidate.callbacks.identities[0].tag_time);
    EXPECT_EQ(completion->identities[1].operator_id, candidate.callbacks.identities[1].operator_id);
    EXPECT_EQ(completion->identities[1].tag_time, candidate.callbacks.identities[1].tag_time);
    EXPECT_EQ(completion->errors, (std::array{ErrorCode::kSuccess, ErrorCode::kSuccess}));
    EXPECT_EQ(completion->first_error, ErrorCode::kSuccess);
    EXPECT_EQ(completion->callbacks_attempted, 2U);
    EXPECT_EQ(completion->callbacks_completed, 2U);
    EXPECT_EQ(completion->phase_deltas,
              (std::array<std::uint64_t, detail::kRuntimeDirectCompositePhaseDeltaCount>{
                  2U, 2U, 2U, 2U, 1U}));
    EXPECT_EQ(control.callbacks.order, candidate.callbacks.order);
    EXPECT_EQ(control.callbacks.callback_count, 2U);
    EXPECT_EQ(control.callbacks.max_active_callbacks, 1U);
    EXPECT_FALSE(control.callbacks.overlap);

    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kDirectCompositeAttempts), 1U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kDirectCompositeCompletions), 1U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kDirectCompositeFallbacks), 0U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kDirectCompositeFailures), 0U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kDirectCompositeCallbacksAttempted),
              2U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kDirectCompositeCallbacksCompleted),
              2U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kLaneDispatcherDispatches), 0U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kLaneComputeDispatches), 0U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kLaneDispatches), 0U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kLaneInlineOwnerInvocations), 1U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kLaneCallbacksEntered), 1U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kLaneCallbacksExited), 1U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kComputeDispatches), 2U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kActivationsCompleted), 2U);

    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kDirectCompositeCompletions), 0U);
    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kLaneComputeDispatches), 0U);
    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kLaneDispatches), 0U);
    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kLaneInlineOwnerInvocations), 2U);
    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kComputeDispatches), 2U);
    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kActivationsCompleted), 2U);
  }
}

// Physical Spin and Readable roots both derive the same compiler-frozen island. The source-tree
// finite binding then supplies one Spin witness to prove the one-lane Runtime transaction.
// REQ: RUNTIME-011
// REQ: RUNTIME-039
TEST(RunSessionTest, ClosedIslandEligibilityAcceptsPhysicalSpinAndReadableRoots) {
  for (const auto root :
       {RuntimePayloadRoundTripRoot::kSpin, RuntimePayloadRoundTripRoot::kReadable}) {
    const ExecutionPlan compiled_plan = make_runtime_pod_round_trip_plan(false, root);
    ASSERT_TRUE(compiled_plan.ok()) << compiled_plan.json();
    const auto compiled = detail::ExecutionPlanAccess::retain(compiled_plan);
    ASSERT_NE(compiled, nullptr);
    ASSERT_EQ(compiled->closed_island_activations.size(), 1U) << compiled_plan.json();
    EXPECT_EQ(compiled->closed_island_activations.front().readiness_root,
              detail::ClosedIslandReadinessRootKind::kPhysical);
  }

  reset_payload_round_trip_probe();
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  ExecutionPlan plan = make_runtime_pod_round_trip_plan(false, RuntimePayloadRoundTripRoot::kSpin);
  plan = detail::RuntimeTestPlanAccess::enable_direct_once(plan);
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool observed = wait_for_pod();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  const auto snapshot = counters->snapshot();
  const auto count = [&](detail::RuntimePhaseCounter counter) {
    return snapshot[detail::RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeAttempts), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeCompletions), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeCallbacksAttempted), 2U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kPartitionCausalFrontierFallbacks), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneComputeDispatches), 0U);
}

// The compiler admits Readable as explicit producer authority. Cold reconstruction must bind the
// fresh declaration before normalizing its completion-bearing Contract, then close and drain the
// output after exactly two successful calls.
// REQ: CONTRACT-020
// REQ: CONTRACT-024
// REQ: RUNTIME-050
// REQ: CHAN-136
TEST(RunSessionTest, ComputeCallCompletionExecutesReadableColdReplayAndNaturalDrain) {
  RuntimeCompletionPhysicalProbe::reset();
  ExecutionPlan plan =
      make_runtime_completion_physical_plan(RuntimeCompletionPhysicalRoot::kReadable);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.operator_completion_rules().size(), 1U);
  const auto compiled = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(compiled, nullptr);
  ASSERT_EQ(compiled->readiness.size(), 1U);
  EXPECT_EQ(compiled->readiness.front().kind, ReadinessSourceKind::kReadable);
  EXPECT_TRUE(compiled->readiness.front().referenced);
  EXPECT_TRUE(compiled->readiness.front().materialized);

  const auto factory = std::make_shared<RuntimeCompletionPhysicalBindingFactory>(
      RuntimeCompletionPhysicalRoot::kReadable);
  plan = detail::RuntimeTestPlanAccess::with_port_binding_factory(plan, factory);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const auto bound = factory->wait_for_control();
  ASSERT_TRUE(bound.has_value());
  if (!bound) {
    return;
  }
  // The first readable publication and its asserted-level retirement each advance this
  // source-local epoch. The retirement wake occurs only after the cell and readable edge gate are
  // re-armed; raw zero credit alone is an earlier boundary and is not safe for injecting replay.
  const std::uint64_t wake_epoch_before_publication = bound->control.wake_epoch();
  ASSERT_TRUE(
      bound->control
          .observe_readable(true, TimePoint{.timestamp_ns = 1, .clock_id = bound->reference_clock})
          .has_value());
  ASSERT_TRUE(RuntimeCompletionPhysicalProbe::wait_for_source_calls(1U));
  ASSERT_TRUE(wait_for_readable_rearm(bound->control, wake_epoch_before_publication));
  EXPECT_EQ(bound->control.outstanding(), 0U);
  ASSERT_TRUE(
      bound->control
          .observe_readable(false, TimePoint{.timestamp_ns = 2, .clock_id = bound->reference_clock})
          .has_value());
  ASSERT_TRUE(
      bound->control
          .observe_readable(true, TimePoint{.timestamp_ns = 3, .clock_id = bound->reference_clock})
          .has_value());
  ASSERT_TRUE(RuntimeCompletionPhysicalProbe::wait_for_source_calls(2U));
  session.wait();

  expect_runtime_completion_physical_outcome(session, plan);
}

// Doorbell replay follows the same completion contract but retains its counted monotonic physical
// source and capacity. Both accepted witnesses must become successful calls before natural EOS.
// REQ: CONTRACT-020
// REQ: CONTRACT-024
// REQ: RUNTIME-050
// REQ: CHAN-136
TEST(RunSessionTest, ComputeCallCompletionExecutesDoorbellColdReplayAndNaturalDrain) {
  RuntimeCompletionPhysicalProbe::reset();
  ExecutionPlan plan =
      make_runtime_completion_physical_plan(RuntimeCompletionPhysicalRoot::kDoorbell);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.operator_completion_rules().size(), 1U);
  const auto compiled = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(compiled, nullptr);
  ASSERT_EQ(compiled->readiness.size(), 1U);
  EXPECT_EQ(compiled->readiness.front().kind, ReadinessSourceKind::kDoorbell);
  EXPECT_EQ(compiled->readiness.front().accepted_capacity, 3U);
  EXPECT_TRUE(compiled->readiness.front().referenced);
  EXPECT_TRUE(compiled->readiness.front().materialized);

  const auto factory = std::make_shared<RuntimeCompletionPhysicalBindingFactory>(
      RuntimeCompletionPhysicalRoot::kDoorbell);
  plan = detail::RuntimeTestPlanAccess::with_port_binding_factory(plan, factory);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const auto bound = factory->wait_for_control();
  ASSERT_TRUE(bound.has_value());
  if (!bound) {
    return;
  }
  const auto first = bound->control.ring_doorbell(
      1U, TimePoint{.timestamp_ns = 10, .clock_id = bound->reference_clock});
  ASSERT_TRUE(first.has_value());
  EXPECT_TRUE(first->witness_accepted);
  const auto second = bound->control.ring_doorbell(
      2U, TimePoint{.timestamp_ns = 11, .clock_id = bound->reference_clock});
  ASSERT_TRUE(second.has_value());
  EXPECT_TRUE(second->witness_accepted);
  session.wait();

  expect_runtime_completion_physical_outcome(session, plan);
}

// A lifecycle-acquired NotificationSender exercises the distinct startup materialization and
// replay branch, then proves the notified root retains the same one-task/two-callback geometry.
// REQ: RUNTIME-011
// REQ: RUNTIME-039
TEST(RunSessionTest, ClosedIslandEligibilityExecutesNotificationRoot) {
  test::RuntimeNotificationProbeState::reset();
  reset_payload_round_trip_probe();
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  ExecutionPlan plan =
      make_runtime_pod_round_trip_plan(false, RuntimePayloadRoundTripRoot::kNotification);
  const auto compiled = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(compiled, nullptr);
  ASSERT_EQ(compiled->closed_island_activations.size(), 1U) << plan.json();
  EXPECT_EQ(compiled->closed_island_activations.front().readiness_root,
            detail::ClosedIslandReadinessRootKind::kNotification);
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  ASSERT_TRUE(test::RuntimeNotificationProbeState::wait_until_sender());
  const NotificationSender sender = test::RuntimeNotificationProbeState::sender_copy();
  auto posted = sender.post();
  const auto start_deadline = std::chrono::steady_clock::now() + 5s;
  while (!posted && posted.error().code == ErrorCode::kNotReady &&
         std::chrono::steady_clock::now() < start_deadline) {
    std::this_thread::yield();
    posted = sender.post();
  }
  ASSERT_TRUE(posted);
  const bool observed = wait_for_pod();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  const RuntimePayloadRoundTripCallbackEvidence callbacks = payload_callback_evidence();
  EXPECT_EQ(callbacks.callback_count, 2U);
  EXPECT_TRUE(callbacks.same_thread);
  const auto snapshot = counters->snapshot();
  const auto count = [&](detail::RuntimePhaseCounter counter) {
    return snapshot[detail::RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeAttempts), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeCompletions), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeCallbacksAttempted), 2U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kPartitionCausalFrontierFallbacks), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneComputeDispatches), 0U);
}

// Tensor transport, ambiguous incident delivery, side-input/advanced temporal policy, mutable
// Parameters, lifecycle transactions, and multi-emission sources remain on the ordinary path.
// Each authored plan proves that the compiler omits rather than partially admits a composite when
// one bounded eligibility fact is absent.
// REQ: RUNTIME-011
// REQ: RUNTIME-039
TEST(RunSessionTest, ClosedIslandEligibilityRejectsUnsupportedStatefulShapes) {
  const auto expect_omitted = [](const ExecutionPlan& plan, std::string_view label) {
    SCOPED_TRACE(label);
    ASSERT_TRUE(plan.ok()) << plan.json();
    const auto state = detail::ExecutionPlanAccess::retain(plan);
    ASSERT_NE(state, nullptr);
    EXPECT_TRUE(state->closed_island_activations.empty());
  };

  expect_omitted(make_runtime_closed_island_tensor_ineligible_plan(), "Tensor payload");
  expect_omitted(make_runtime_pod_nonretained_sibling_plan(), "non-retained sibling incident edge");
  expect_omitted(make_runtime_pod_external_export_ambiguity_plan(),
                 "mixed local and external publication");
  expect_omitted(make_runtime_pod_side_input_plan(), "independently clocked side input");
  expect_omitted(make_runtime_pod_predicate_plan(), "predicate trigger");
  expect_omitted(make_runtime_pod_deadline_plan(), "deadline policy");
  expect_omitted(make_runtime_pod_mutable_param_plan(), "runtime-mutable Parameters");
  expect_omitted(make_runtime_pod_round_trip_plan(true), "lifecycle transaction");
  expect_omitted(make_runtime_pod_double_emit_plan(), "multi-emission source");
}

// A closed-island composite is owned by one partition runner. Distinct explicit
// partitions may share a physical lane, but their edge must remain on the
// ordinary cross-partition path instead of creating cross-runner authority.
// REQ: RUNTIME-039
// REQ: RUNTIME-041
TEST(RunSessionTest, ClosedIslandEligibilityDoesNotCrossExplicitPartitionBoundary) {
  reset_payload_round_trip_probe();
  const ExecutionPlan plan = make_runtime_pod_cross_partition_round_trip_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_TRUE(state->closed_island_activations.empty());

  RunSession session = run_async(plan);
  const bool observed = wait_for_pod();
  session.request_stop();
  session.wait();

  EXPECT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
}

// The first complete transaction warms every run-owned cache. A test-only allocation counter then
// spans the second source/successor transaction, including the gap between callbacks, without
// instrumenting production plans.
// REQ: RUNTIME-011
// REQ: RUNTIME-039
TEST(RunSessionTest, ClosedIslandCompositeWarmPodTransactionAllocatesNothing) {
  reset_payload_round_trip_probe(
      RuntimePayloadRoundTripSourceBehavior::kEmitTwiceAcrossActivations);
  RuntimeClosedIslandAllocationProbe allocation_probe;
  const auto control = std::make_shared<detail::RuntimeClosedIslandTestControl>();
  control->set_transaction_hooks(&allocation_probe,
                                 &RuntimeClosedIslandAllocationProbe::begin,
                                 &RuntimeClosedIslandAllocationProbe::end);

  ExecutionPlan plan = make_runtime_pod_round_trip_plan();
  plan = detail::RuntimeTestPlanAccess::with_closed_island_test_control(plan, control);
  ASSERT_TRUE(plan.ok()) << plan.json();
  RunSession session = run_async(plan);
  const bool observed = wait_for_pod_count(2U);
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_GE(allocation_probe.transaction_count.load(std::memory_order_acquire), 2U);
  EXPECT_TRUE(allocation_probe.observed.load(std::memory_order_acquire));
  EXPECT_EQ(allocation_probe.allocations.load(std::memory_order_acquire), 0U);
}

// No publication, a typed Error, and an exception must never synthesize a successor callback.
// Typed and exceptional failures retain the ordinary terminal diagnostic.
// REQ: RUNTIME-011
// REQ: RUNTIME-024
// REQ: RUNTIME-039
TEST(RunSessionTest, ClosedIslandCompositePreservesNoEmissionAndSourceFailureSemantics) {
  struct Case {
    RuntimePayloadRoundTripSourceBehavior behavior;
    RunTermination termination;
    bool expect_diagnostic;
    ErrorCode expected_error;
  };
  constexpr std::array cases{
      Case{RuntimePayloadRoundTripSourceBehavior::kNoEmission,
           RunTermination::kStopped,
           false,
           ErrorCode::kSuccess},
      Case{RuntimePayloadRoundTripSourceBehavior::kBackpressured,
           RunTermination::kFailed,
           true,
           ErrorCode::kBackpressured},
      Case{RuntimePayloadRoundTripSourceBehavior::kTypedFailure,
           RunTermination::kFailed,
           true,
           ErrorCode::kFailure},
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      Case{RuntimePayloadRoundTripSourceBehavior::kException,
           RunTermination::kFailed,
           true,
           ErrorCode::kFailure},
#endif
  };

  for (const Case& test_case : cases) {
    SCOPED_TRACE(static_cast<unsigned>(test_case.behavior));
    reset_payload_round_trip_probe(test_case.behavior);
    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    RuntimeClosedIslandCompletionProbe completion_probe;
    const auto completion_control = std::make_shared<detail::RuntimeClosedIslandTestControl>();
    completion_control->set_completion_hook(&completion_probe,
                                            &RuntimeClosedIslandCompletionProbe::capture);
    ExecutionPlan plan = make_runtime_pod_round_trip_plan();
    plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
    plan = detail::RuntimeTestPlanAccess::with_closed_island_test_control(plan, completion_control);
    ASSERT_TRUE(plan.ok()) << plan.json();

    RunSession session = run_async(plan);
    ASSERT_TRUE(wait_for_payload_source());
    if (test_case.termination == RunTermination::kStopped) {
      session.request_stop();
    }
    session.wait();

    EXPECT_EQ(session.termination(), test_case.termination);
    EXPECT_FALSE(observed_pod_snapshot());
    const auto callbacks = payload_callback_evidence();
    EXPECT_EQ(callbacks.callback_count, 1U);
    EXPECT_EQ(callbacks.max_active_callbacks, 1U);
    EXPECT_FALSE(callbacks.overlap);
    if (test_case.expect_diagnostic) {
      ASSERT_EQ(session.diagnostics().size(), 1U);
      EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
      EXPECT_EQ(session.diagnostics().front().service, "operator_compute");
    } else {
      EXPECT_TRUE(session.diagnostics().empty());
    }
    const auto snapshot = counters->snapshot();
    const auto count = [&](detail::RuntimePhaseCounter counter) {
      return snapshot[detail::RuntimePhaseCounters::index(counter)];
    };
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeCompletions), 0U);
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeCallbacksAttempted), 1U);
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeCallbacksCompleted),
              test_case.expect_diagnostic ? 0U : 1U);
    const auto completion = completion_probe.read();
    ASSERT_TRUE(completion.has_value());
    EXPECT_EQ(completion->callbacks_attempted, 1U);
    EXPECT_EQ(completion->callbacks_completed, test_case.expect_diagnostic ? 0U : 1U);
    EXPECT_EQ(completion->errors[0], test_case.expected_error);
    EXPECT_EQ(completion->first_error, test_case.expected_error);
    EXPECT_EQ(completion->phase_deltas[static_cast<std::size_t>(
                  detail::RuntimeDirectCompositePhaseDelta::kCallbacksAttempted)],
              1U);
    EXPECT_EQ(completion->phase_deltas[static_cast<std::size_t>(
                  detail::RuntimeDirectCompositePhaseDelta::kCallbacksCompleted)],
              test_case.expect_diagnostic ? 0U : 1U);
    EXPECT_EQ(completion->phase_deltas[static_cast<std::size_t>(
                  detail::RuntimeDirectCompositePhaseDelta::kActivationsRetired)],
              1U);
    EXPECT_EQ(completion->phase_deltas[static_cast<std::size_t>(
                  detail::RuntimeDirectCompositePhaseDelta::kActivationCuts)],
              test_case.expect_diagnostic ? 0U : 1U);
  }
}

// A successor typed Error or exception retires the same selected payload, activation, and source
// credit as ordinary dispatch while preserving the successor's callback identity and diagnostic.
// REQ: RUNTIME-011
// REQ: RUNTIME-024
// REQ: RUNTIME-039
TEST(RunSessionTest, ClosedIslandCompositePreservesSuccessorFailureSemantics) {
  constexpr std::array sink_behaviors{
      RuntimePayloadRoundTripSinkBehavior::kTypedFailure,
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      RuntimePayloadRoundTripSinkBehavior::kException,
#endif
  };

  for (const RuntimePayloadRoundTripSinkBehavior sink_behavior : sink_behaviors) {
    SCOPED_TRACE(static_cast<unsigned>(sink_behavior));
    const auto candidate = run_direct_continuation_round_trip(
        RuntimeDirectContinuationPayloadKind::kPod, false, false, false, {}, sink_behavior);
    const auto control = run_direct_continuation_round_trip(
        RuntimeDirectContinuationPayloadKind::kPod, false, false, true, {}, sink_behavior);
    for (const auto* evidence : {&candidate, &control}) {
      EXPECT_TRUE(evidence->observed);
      EXPECT_EQ(evidence->termination, RunTermination::kFailed);
      EXPECT_EQ(evidence->diagnostic_count, 1U);
      EXPECT_EQ(evidence->callbacks.callback_count, 2U);
      EXPECT_EQ(evidence->callbacks.max_active_callbacks, 1U);
      EXPECT_FALSE(evidence->callbacks.overlap);
    }
    const auto candidate_count = [&](detail::RuntimePhaseCounter counter) {
      return candidate.counters[detail::RuntimePhaseCounters::index(counter)];
    };
    EXPECT_EQ(candidate_count(detail::RuntimePhaseCounter::kDirectCompositeAttempts), 1U);
    EXPECT_EQ(candidate_count(detail::RuntimePhaseCounter::kDirectCompositeCompletions), 0U);
    EXPECT_EQ(candidate_count(detail::RuntimePhaseCounter::kDirectCompositeCallbacksAttempted), 2U);
    EXPECT_EQ(candidate_count(detail::RuntimePhaseCounter::kDirectCompositeCallbacksCompleted), 1U);
  }
}

// A source that may publish twice is conservatively omitted from the compiler-proven island.
// Ordinary dispatch still preserves finite backpressure: handling the second result preserves the
// first publication, while returning it remains a callback failure and suppresses the successor.
// REQ: CHAN-101
// REQ: RUNTIME-024
// REQ: RUNTIME-039
TEST(RunSessionTest, ConservativeClosedIslandEligibilityFallsBackForDoubleEmission) {
  for (const auto behavior : {RuntimePayloadRoundTripSourceBehavior::kSecondEmitHandled,
                              RuntimePayloadRoundTripSourceBehavior::kSecondEmitReturned}) {
    SCOPED_TRACE(static_cast<unsigned>(behavior));
    reset_payload_round_trip_probe(behavior);
    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    ExecutionPlan plan = make_runtime_pod_double_emit_plan();
    plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
    ASSERT_TRUE(plan.ok()) << plan.json();

    RunSession session = run_async(plan);
    if (behavior == RuntimePayloadRoundTripSourceBehavior::kSecondEmitHandled) {
      EXPECT_TRUE(wait_for_pod());
      session.request_stop();
    }
    session.wait();

    const auto second_emit = second_emit_snapshot();
    ASSERT_TRUE(second_emit);
    // The fatal assertion above proves the optional is engaged.
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_EQ(*second_emit, ErrorCode::kBackpressured);
    const auto snapshot = counters->snapshot();
    const auto count = [&](detail::RuntimePhaseCounter counter) {
      return snapshot[detail::RuntimePhaseCounters::index(counter)];
    };
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelEmitAttempts), 2U);
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelEmitAccepted), 1U);
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelEmitBackpressured), 1U);
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeAttempts), 0U);
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeCompletions), 0U);
    if (behavior == RuntimePayloadRoundTripSourceBehavior::kSecondEmitHandled) {
      EXPECT_EQ(session.termination(), RunTermination::kStopped);
      EXPECT_TRUE(observed_pod_snapshot());
      EXPECT_TRUE(session.diagnostics().empty());
    } else {
      EXPECT_EQ(session.termination(), RunTermination::kFailed);
      EXPECT_FALSE(observed_pod_snapshot());
      ASSERT_EQ(session.diagnostics().size(), 1U);
      EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
    }
  }
}

// Stop and generation checks occur at the callback boundary inside one lane task. Stop wins
// without entering the successor; stale evidence falls back to the ordinary readiness scan and
// still delivers exactly once.
// REQ: RUNTIME-011
// REQ: RUNTIME-024
// REQ: RUNTIME-039
TEST(RunSessionTest, ClosedIslandCompositeHonorsStopAndGenerationMismatchAtSuccessorBoundary) {
  {
    reset_payload_round_trip_probe();
    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    const auto control = std::make_shared<detail::RuntimeClosedIslandTestControl>();
    control->set_pause_before_successor(true);
    ExecutionPlan plan = make_runtime_pod_round_trip_plan();
    plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
    plan = detail::RuntimeTestPlanAccess::with_closed_island_test_control(plan, control);

    RunSession session = run_async(plan);
    const bool reached_successor_boundary = control->wait_until_before_successor();
    session.request_stop();
    control->release();
    session.wait();

    ASSERT_TRUE(reached_successor_boundary);
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(expected_pod_snapshot());
    EXPECT_FALSE(observed_pod_snapshot());
    EXPECT_EQ(payload_callback_evidence().callback_count, 1U);
    const auto snapshot = counters->snapshot();
    EXPECT_EQ(snapshot[detail::RuntimePhaseCounters::index(
                  detail::RuntimePhaseCounter::kDirectCompositeCompletions)],
              0U);
  }
  {
    reset_payload_round_trip_probe();
    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    RuntimeClosedIslandCompletionProbe completion_probe;
    const auto control = std::make_shared<detail::RuntimeClosedIslandTestControl>();
    control->set_pause_at_successor_boundary(true);
    control->set_completion_hook(&completion_probe, &RuntimeClosedIslandCompletionProbe::capture);
    ExecutionPlan plan = make_runtime_pod_round_trip_plan();
    plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
    plan = detail::RuntimeTestPlanAccess::with_closed_island_test_control(plan, control);

    RunSession session = run_async(plan);
    const bool reached_successor_boundary = control->wait_until_successor_boundary();
    session.request_stop();
    control->release();
    session.wait();

    ASSERT_TRUE(reached_successor_boundary);
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(expected_pod_snapshot());
    EXPECT_FALSE(observed_pod_snapshot());
    EXPECT_EQ(payload_callback_evidence().callback_count, 1U);
    const auto snapshot = counters->snapshot();
    EXPECT_EQ(snapshot[detail::RuntimePhaseCounters::index(
                  detail::RuntimePhaseCounter::kDirectCompositeCompletions)],
              0U);
    const auto completion = completion_probe.read();
    ASSERT_TRUE(completion.has_value());
    // The fatal assertion above proves the optional is engaged.
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_EQ(completion->phase_deltas[static_cast<std::size_t>(
                  detail::RuntimeDirectCompositePhaseDelta::kActivationsRetired)],
              2U);
  }
  {
    const auto control = std::make_shared<detail::RuntimeClosedIslandTestControl>();
    control->set_corrupt_generation(true);
    const RuntimeDirectContinuationRoundTripEvidence evidence = run_direct_continuation_round_trip(
        RuntimeDirectContinuationPayloadKind::kPod, false, false, false, control);
    const auto count = [&](detail::RuntimePhaseCounter counter) {
      return evidence.counters[detail::RuntimePhaseCounters::index(counter)];
    };

    ASSERT_TRUE(evidence.observed);
    EXPECT_EQ(evidence.termination, RunTermination::kStopped);
    EXPECT_EQ(evidence.diagnostic_count, 0U);
    EXPECT_EQ(evidence.callbacks.callback_count, 2U);
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeCompletions), 0U);
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeFallbacks), 1U);
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneDispatcherDispatches), 0U);
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneComputeDispatches), 0U);
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneDispatches), 0U);
    EXPECT_GE(count(detail::RuntimePhaseCounter::kLaneInlineOwnerInvocations), 1U);
  }
}

// A callback-produced receipt is only accepted when the next readiness scan consumes it and
// removes an ordinary poll. Stopping after the lane stages the receipt but before that scan must
// retire the session without reporting an elision that never happened.
// REQ: RUNTIME-011
// REQ: RUNTIME-019
TEST(RunSessionTest, StopAfterDirectContinuationStagingRecordsNoUnelidedAcceptance) {
  reset_payload_round_trip_probe();
  ExecutionPlan plan = make_runtime_pod_round_trip_plan();
  const auto plan_state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(plan_state, nullptr);
  ASSERT_EQ(plan_state->process_local_channel_services.size(), 1U);
  ASSERT_EQ(plan_state->process_local_channel_services.front().delivery_kind,
            detail::ProcessLocalChannelDeliveryKind::kDirectRetained);

  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const auto control = std::make_shared<detail::RuntimeDirectContinuationTestControl>();
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  plan = detail::RuntimeTestPlanAccess::disable_closed_island_composite_for_test(plan);
  plan = detail::RuntimeTestPlanAccess::with_direct_continuation_test_control(plan, control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool receipt_staged = control->wait_until_receipt_staged();
  session.request_stop();
  control->release();
  session.wait();

  ASSERT_TRUE(receipt_staged);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_TRUE(expected_pod_snapshot().has_value());
  EXPECT_FALSE(observed_pod_snapshot().has_value());

  const auto snapshot = counters->snapshot();
  const auto count = [&](const detail::RuntimePhaseCounter counter) {
    return snapshot[detail::RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelEmitAccepted), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelSamplesReceived), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelReadinessPublished), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kActivationsCompleted), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectContinuationReceiptsAccepted), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectContinuationPollElisions), 0U);
}

// Restart-capable plans remain on the ordinary controller-owned path until Checkpoint 4C proves
// the complete lifecycle transaction. This conservative arm also demonstrates that merely
// compiling the private optimization cannot alter successful direct-retained delivery.
// REQ: RUNTIME-011
// REQ: RUNTIME-019
TEST(RunSessionTest, DirectContinuationRemainsDisabledForLifecyclePlans) {
  constexpr std::array payload_kinds{
      RuntimeDirectContinuationPayloadKind::kPod,
      RuntimeDirectContinuationPayloadKind::kGeneratedSchema,
  };
  const auto count = [](const RuntimeDirectContinuationRoundTripEvidence& evidence,
                        detail::RuntimePhaseCounter counter) {
    return evidence.counters[detail::RuntimePhaseCounters::index(counter)];
  };

  for (const auto payload_kind : payload_kinds) {
    const RuntimeDirectContinuationRoundTripEvidence evidence =
        run_direct_continuation_round_trip(payload_kind, false, true);
    ASSERT_TRUE(evidence.direct_delivery);
    ASSERT_TRUE(evidence.observed);
    EXPECT_EQ(evidence.termination, RunTermination::kStopped);
    EXPECT_EQ(evidence.diagnostic_count, 0U);
    ASSERT_TRUE(evidence.expected.has_value());
    ASSERT_TRUE(evidence.received.has_value());
    // The fatal assertions above prove both optionals are engaged.
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    expect_caller_sample_metadata_preserved(evidence.expected->metadata,
                                            evidence.received->metadata);
    // NOLINTEND(bugprone-unchecked-optional-access)
    EXPECT_EQ(count(evidence, detail::RuntimePhaseCounter::kDirectContinuationReceiptsAccepted),
              0U);
    EXPECT_EQ(count(evidence, detail::RuntimePhaseCounter::kDirectContinuationPollElisions), 0U);
    EXPECT_EQ(count(evidence, detail::RuntimePhaseCounter::kDirectCompositeAttempts), 0U);
    EXPECT_EQ(count(evidence, detail::RuntimePhaseCounter::kDirectCompositeCompletions), 0U);
    EXPECT_EQ(count(evidence, detail::RuntimePhaseCounter::kReadinessPollDispatches), 0U);
    EXPECT_EQ(count(evidence, detail::RuntimePhaseCounter::kLaneDispatches), 0U);
  }
}

// One public output with blocking and drop-oldest consumers must remain one logical publication
// even though the compiler materializes one HoloMQ service per queue policy.
// REQ: CHAN-101
// REQ: CHAN-115
// REQ: RUNTIME-019
TEST(RunSessionTest, PublishesSameOutputAcrossSeparateMixedPolicyServices) {
  // Key setup: connect one scalar Output to capacity-one blocking and drop-oldest inputs on the
  // same execution partition, then compile and run through the ordinary public Graph API.
  RuntimeMixedPolicyFanoutProbe::reset();
  const ExecutionPlan plan = make_runtime_mixed_policy_fanout_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  EXPECT_EQ(plan.partitioning_plan().mode, PartitioningMode::kManual);
  EXPECT_EQ(plan.partitioning_plan().authored_partition_count, 1U);
  EXPECT_EQ(plan.partitioning_plan().automatic_partition_count, 0U);
  const auto& authored_name = plan.execution_partitions().front().authored_name;
  HOLOSCAN_ASSERT_HAS_VALUE(authored_name);
  EXPECT_EQ(*authored_name, "runtime-mixed-policy-fanout");
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->process_local_channel_services.size(), 2U);
  ASSERT_EQ(state->process_local_channel_publishers.size(), 2U);
  ASSERT_EQ(state->process_local_channel_subscribers.size(), 2U);
  EXPECT_EQ(state->process_local_channel_services[0].source,
            state->process_local_channel_services[1].source);
  EXPECT_TRUE(std::ranges::all_of(state->process_local_channel_services, [](const auto& service) {
    return service.delivery_kind == detail::ProcessLocalChannelDeliveryKind::kHoloMQ &&
           service.max_subscribers == 1U && !service.event_wake &&
           service.event_wake_reason == detail::ProcessLocalHoloMQEventWakeReason::kSuppressed;
  }));
  EXPECT_TRUE(std::ranges::none_of(state->process_local_channel_subscribers,
                                   [](const auto& subscriber) { return subscriber.event_wake; }));
  EXPECT_EQ(std::ranges::count_if(
                state->process_local_channel_services,
                [](const auto& service) { return service.queue_policy == QueuePolicy::kBlocking; }),
            1);
  EXPECT_EQ(std::ranges::count_if(state->process_local_channel_services,
                                  [](const auto& service) {
                                    return service.queue_policy == QueuePolicy::kDropOldest;
                                  }),
            1);

  RunSession session = run_async(plan);
  const bool observed = RuntimeMixedPolicyFanoutProbe::wait_for_both(2U);
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const auto observations = RuntimeMixedPolicyFanoutProbe::snapshot();
  ASSERT_EQ(observations[0].size(), 2U);
  ASSERT_EQ(observations[1].size(), 2U);
  // Key assertion: both policy-specific services expose the same logical values and publication
  // sequence in order; the lowering boundary does not duplicate or omit either publication.
  for (std::size_t index = 0U; index < observations[0].size(); ++index) {
    EXPECT_EQ(observations[0][index].value, static_cast<std::int32_t>(index + 1U));
    EXPECT_EQ(observations[1][index].value, observations[0][index].value);
    EXPECT_EQ(observations[1][index].sequence, observations[0][index].sequence);
  }
}

// The public compile/run path must not let a capacity-one blocking queue monopolize the serial
// execution lane. The callback observes finite pressure, remains live under the cooperative
// cancellation contract, and then lets request_stop() complete the ordinary RunSession teardown.
// REQ: CHAN-101
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(RunSessionTest, StopsWhileGraphOwnedBlockingQueueIsSaturated) {
  reset_blocking_pressure_probe();
  const ExecutionPlan plan = make_runtime_blocking_pressure_plan("runtime-blocking-pressure-stop");
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_FALSE(state->process_local_channel_services.empty());
  EXPECT_TRUE(std::ranges::all_of(state->process_local_channel_services, [](const auto& service) {
    return service.delivery_kind == detail::ProcessLocalChannelDeliveryKind::kHoloMQ;
  }));

  RunSession session = run_async(plan);
  const bool pressure_observed = wait_for_blocking_pressure();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(pressure_observed);
  const auto [result, accepted_count] = blocking_pressure_snapshot();
  EXPECT_EQ(result, ErrorCode::kBackpressured);
  EXPECT_GT(accepted_count, 0U);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_FALSE(blocking_pressure_callback_active());
}

// Returning a fatal Error after finite graph-owned pressure must preserve ordinary structural
// failure reporting and teardown. In particular, pressure handling cannot strand the callback or
// convert a user callback failure into a blocked stop path.
// REQ: CHAN-101
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(RunSessionTest, FailsStructurallyAfterGraphOwnedBlockingQueuePressure) {
  reset_blocking_pressure_probe(/*fail_after_pressure=*/true);
  const ExecutionPlan plan =
      make_runtime_blocking_pressure_plan("runtime-blocking-pressure-failure");
  ASSERT_TRUE(plan.ok()) << plan.json();

  const auto wait_started = std::chrono::steady_clock::now();
  RunSession session = run_async(plan);
  session.wait();
  const auto wait_duration = std::chrono::steady_clock::now() - wait_started;

  const auto [result, accepted_count] = blocking_pressure_snapshot();
  EXPECT_EQ(result, ErrorCode::kBackpressured);
  EXPECT_GT(accepted_count, 0U);
  EXPECT_LT(wait_duration, 5s);
  ASSERT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  const Diagnostic& diagnostic = session.diagnostics().front();
  EXPECT_EQ(diagnostic.token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
  EXPECT_EQ(diagnostic.service, "operator_compute");
  EXPECT_TRUE(static_cast<bool>(diagnostic.source.vertex));
  EXPECT_TRUE(diagnostic.endpoint.empty());
  EXPECT_TRUE(diagnostic.json_pointer.empty());
  EXPECT_FALSE(blocking_pressure_callback_active());
}

// Orders three single-sample producers around one blocked sink activation. The waits observe
// destination-side Channel counters, so each ordering point is a published arrival rather than a
// returned callback.
struct ReadinessRingPressureProbe {
  static inline std::shared_ptr<detail::RuntimePhaseCounters> counters;
  static inline std::atomic<bool> sink_blocked{};
  static inline std::atomic<std::uint32_t> activations{};
  static inline std::atomic<std::uint32_t> first_activation_inputs{};
  static inline std::atomic<std::uint32_t> received_inputs{};

  static void reset(std::shared_ptr<detail::RuntimePhaseCounters> observed) noexcept {
    counters = std::move(observed);
    sink_blocked.store(false);
    activations.store(0U);
    first_activation_inputs.store(0U);
    received_inputs.store(0U);
  }

  [[nodiscard]] static bool wait(const std::atomic<bool>& flag) noexcept {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!flag.load(std::memory_order_acquire)) {
      if (std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      std::this_thread::sleep_for(1ms);
    }
    return true;
  }

  [[nodiscard]] static bool wait(detail::RuntimePhaseCounter counter,
                                 std::uint64_t minimum) noexcept {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (counters->snapshot()[detail::RuntimePhaseCounters::index(counter)] < minimum) {
      if (std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      std::this_thread::sleep_for(1ms);
    }
    return true;
  }
};

enum class ReadinessRingPressureRole : std::uint8_t { kBlocker, kEarly, kLate };

// Order single-sample publications and EOS around a blocked sink in
// RunSessionTest.FullReadinessRingDefersLaterInputFinalSample.
class ReadinessRingPressureSource final : public Operator<> {
 public:
  explicit ReadinessRingPressureSource(ReadinessRingPressureRole role) noexcept : role_(role) {}

  void setup(OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms}).complete_after(ComputeCalls{.limit = 1U});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    using Probe = ReadinessRingPressureProbe;
    const auto timeout = [] { return make_unexpected(Error{ErrorCode::kTimeout}); };
    switch (role_) {
      case ReadinessRingPressureRole::kBlocker: {
        auto emitted = output.emit(0U);
        if (!emitted) {
          return emitted;
        }
        // Hold this producer's EOS until the sink is inside the activation for this sample, so
        // that terminal witness is still unpublished when the sink next polls.
        return Probe::wait(Probe::sink_blocked) ? expected<void, Error>{} : timeout();
      }
      case ReadinessRingPressureRole::kEarly:
        if (!Probe::wait(Probe::sink_blocked)) {
          return timeout();
        }
        return output.emit(1U);
      case ReadinessRingPressureRole::kLate:
        // Both other producers have closed and their EOS records hold earlier arrival orders, so
        // this final sample is the newest of four pending readiness candidates.
        if (!Probe::wait(detail::RuntimePhaseCounter::kChannelEndOfStreamRetained, 2U)) {
          return timeout();
        }
        return output.emit(2U);
    }
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  Output<std::uint32_t> output;

 private:
  ReadinessRingPressureRole role_;
};

// Hold the first activation until data and EOS exceed the readiness ring capacity in
// RunSessionTest.FullReadinessRingDefersLaterInputFinalSample.
class ReadinessRingPressureSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    for (std::size_t index = 0U; index < inputs.size(); ++index) {
      spec.input(inputs[index], "input-" + std::to_string(index)).queue_depth(1U);
    }
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnAnyInput{});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    using Probe = ReadinessRingPressureProbe;
    std::uint32_t received_inputs = 0U;
    for (std::uint32_t index = 0U; index < inputs.size(); ++index) {
      if (inputs[index].is_missing()) {
        continue;
      }
      auto received = inputs[index].receive();
      if (!received) {
        return make_unexpected(std::move(received).error());
      }
      if (received->data != index) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      received_inputs |= 1U << index;
    }
    Probe::received_inputs.fetch_or(received_inputs);
    if (Probe::activations.fetch_add(1U) == 0U) {
      Probe::first_activation_inputs.store(received_inputs);
      // Keep this owner out of its readiness scan until the final sample has been retained.
      Probe::sink_blocked.store(true, std::memory_order_release);
      if (!Probe::wait(detail::RuntimePhaseCounter::kChannelSamplesReceived, 3U)) {
        return make_unexpected(Error{ErrorCode::kTimeout});
      }
    }
    return {};
  }

  std::array<Input<std::uint32_t>, 3U> inputs;
};

// Graph topology, one partition per Operator:
//   blocker --[depth 1]--> sink.input-0
//   early   --[depth 1]--> sink.input-1
//   late    --[depth 1]--> sink.input-2
[[nodiscard]] ExecutionPlan make_readiness_ring_pressure_plan() {
  Graph graph{"runtime-readiness-ring-pressure"};
  const auto sink = graph.op<ReadinessRingPressureSink>("sink");
  graph.partition("runtime-readiness-ring-pressure-sink").add(sink);
  constexpr std::array roles{ReadinessRingPressureRole::kBlocker,
                             ReadinessRingPressureRole::kEarly,
                             ReadinessRingPressureRole::kLate};
  for (std::size_t index = 0U; index < roles.size(); ++index) {
    const std::string name = "source-" + std::to_string(index);
    const auto source = graph.op<ReadinessRingPressureSource>(name, roles[index]);
    graph.partition("runtime-readiness-ring-pressure-" + name).add(source);
    graph.add_flow(source->output,
                   sink->inputs[index],
                   ConnectionOptions{.queue_policy = QueuePolicy::kBlocking, .queue_depth = 1U});
  }
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

// A process-local readiness ring holds one witness per retained data slot, so a terminal witness
// can fill it ahead of another input's final sample. That full ring is retryable: the published
// prefix is dispatched, and the retained suffix is republished after a witness retires. The sink
// blocks in its first activation while the first producer closes, the second publishes data and
// then EOS, and the third publishes only after both EOS records are retained. The sink's next poll
// therefore finds three witnesses for a three-entry ring ahead of that final data sample.
// REQ: RUNTIME-016
// REQ: CHAN-136
TEST(RunSessionTest, FullReadinessRingDefersLaterInputFinalSample) {
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  ReadinessRingPressureProbe::reset(counters);
  ExecutionPlan plan = make_readiness_ring_pressure_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->process_local_channel_services.size(), 3U);
  EXPECT_TRUE(std::ranges::all_of(state->process_local_channel_services, [](const auto& service) {
    return service.delivery_kind == detail::ProcessLocalChannelDeliveryKind::kDirectRetained;
  }));
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  // A lost retry leaves the final sample unpublished and the session running.
  const bool finished = session.wait_for(10s);
  if (!finished) {
    session.request_stop();
    session.wait();
  }
  ReadinessRingPressureProbe::counters.reset();

  ASSERT_TRUE(finished) << "the deferred readiness suffix was never republished";
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(ReadinessRingPressureProbe::first_activation_inputs.load(), 0b001U);
  EXPECT_EQ(ReadinessRingPressureProbe::received_inputs.load(), 0b111U);
  const auto snapshot = counters->snapshot();
  const auto count = [&](detail::RuntimePhaseCounter counter) {
    return snapshot[detail::RuntimePhaseCounters::index(counter)];
  };
  EXPECT_GE(count(detail::RuntimePhaseCounter::kChannelPollReadinessRingFullData), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelEndOfStreamRetained), 3U);
}

// REQ: RUNTIME-004
// REQ: RUNTIME-015
// REQ: RUNTIME-018
// REQ: RUNTIME-034
TEST(RunSessionTest, ChannelAlignedTriggerAcceptsTheInclusiveExactBoundary) {
  // Key setup: align two captures exactly 10 ns apart with two unwired optional sides.
  // Key assertion: the tuple is accepted and optional reads preserve empty/cardinality errors.
  // Graph:
  //   source.first  --> sink.first
  //   source.second --> sink.second
  //   [unwired] sink.optional, sink.additional_optional
  test::RuntimeAlignedChannelProbeState::reset(test::RuntimeAlignedChannelScript::kExactBoundary);
  const ExecutionPlan plan = test::make_runtime_aligned_channel_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool observed = test::RuntimeAlignedChannelProbeState::wait_for_sink(1U);
  session.request_stop();
  session.wait();

  EXPECT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token) << " vertex="
      << (session.diagnostics().empty() ? 0U : session.diagnostics().front().source.vertex.value);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::source_compute_count.load(), 1U);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::sink_compute_count.load(), 1U);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_first.load(), 1U);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_second.load(), 2U);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_second_capture.load() -
                test::RuntimeAlignedChannelProbeState::observed_first_capture.load(),
            10);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_optional_state.load(),
            InputState::kSafeDefault);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_optional_reason.load(),
            SelectionReason::kNotConnected);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_optional_error.load(),
            ErrorCode::kInputEmpty);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_optional_batch_error.load(),
            ErrorCode::kInputSelectionCardinalityMismatch);
}

// REQ: RUNTIME-004
// REQ: RUNTIME-018
TEST(RunSessionTest, ChannelAlignedTriggerRetiresEveryMinimumAndMakesProgress) {
  test::RuntimeAlignedChannelProbeState::reset(
      test::RuntimeAlignedChannelScript::kMisalignmentProgress);
  const ExecutionPlan plan = test::make_runtime_aligned_channel_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool observed = test::RuntimeAlignedChannelProbeState::wait_for_sink(1U);
  session.request_stop();
  session.wait();

  EXPECT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token) << " vertex="
      << (session.diagnostics().empty() ? 0U : session.diagnostics().front().source.vertex.value);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::sink_compute_count.load(), 1U);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_first.load(), 3U);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_second.load(), 2U);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_second_capture.load() -
                test::RuntimeAlignedChannelProbeState::observed_first_capture.load(),
            10);
}

// REQ: RUNTIME-004
// REQ: RUNTIME-018
TEST(RunSessionTest, ChannelAlignedTriggerKeepsMissingHeadPendingAndFreshRunResetsIt) {
  test::RuntimeAlignedChannelProbeState::reset(test::RuntimeAlignedChannelScript::kMissingHead);
  const ExecutionPlan plan = test::make_runtime_aligned_channel_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession pending = run_async(plan);
  const bool source_progressed = test::RuntimeAlignedChannelProbeState::wait_for_source(1U);
  const std::size_t pending_compute_count =
      test::RuntimeAlignedChannelProbeState::sink_compute_count.load(std::memory_order_acquire);
  pending.request_stop();
  pending.wait();

  ASSERT_TRUE(source_progressed);
  EXPECT_EQ(pending_compute_count, 0U);
  EXPECT_EQ(pending.termination(), RunTermination::kStopped);
  EXPECT_TRUE(pending.diagnostics().empty());

  test::RuntimeAlignedChannelProbeState::reset(test::RuntimeAlignedChannelScript::kExactBoundary);
  RunSession fresh = run_async(plan);
  const bool observed = test::RuntimeAlignedChannelProbeState::wait_for_sink(1U);
  fresh.request_stop();
  fresh.wait();

  EXPECT_TRUE(observed);
  EXPECT_EQ(fresh.termination(), RunTermination::kStopped);
  EXPECT_TRUE(fresh.diagnostics().empty())
      << (fresh.diagnostics().empty() ? "" : fresh.diagnostics().front().token);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::sink_compute_count.load(), 1U);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_first.load(), 1U);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_second.load(), 2U);
}

// An optional side selector cannot turn a non-aligned member pair into activation authority.
// REQ: CONTRACT-006
// REQ: RUNTIME-055
TEST(RunSessionTest, ChannelAlignedOptionalSideInputDoesNotActivateMisalignedMembers) {
  // Key setup: place the two aligned members 11 ns apart, outside the 10 ns tolerance.
  // Key assertion: unwired optional sides cannot cause the sink to compute.
  // Graph:
  //   source.first  --> sink.first
  //   source.second --> sink.second
  //   [unwired] sink.optional, sink.additional_optional
  test::RuntimeAlignedChannelProbeState::reset(
      test::RuntimeAlignedChannelScript::kMisalignedMembers);
  const ExecutionPlan plan = test::make_runtime_aligned_channel_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  ASSERT_TRUE(test::RuntimeAlignedChannelProbeState::wait_for_source(1U));
  std::this_thread::sleep_for(10ms);
  session.request_stop();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::sink_compute_count.load(), 0U);
}

// Topology absence and producer silence are distinct even when the activation reference comes
// from an aligned member set.
// REQ: CONTRACT-022
// REQ: RUNTIME-055
TEST(RunSessionTest, ChannelAlignedConnectedSilentSideInputRemainsNeverSeen) {
  // Key setup: connect a silent optional producer alongside an aligned member pair.
  // Key assertion: the sink computes with kSafeDefault/kNeverSeen and an empty single-sample read.
  // Graph:
  //   source.first    --> sink.first
  //   source.second   --> sink.second
  //   source.optional --> sink.optional
  //   [unwired] sink.additional_optional
  test::RuntimeAlignedChannelProbeState::reset(test::RuntimeAlignedChannelScript::kExactBoundary);
  const ExecutionPlan plan = test::make_runtime_aligned_channel_plan(false, true);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool observed = test::RuntimeAlignedChannelProbeState::wait_for_sink(1U);
  session.request_stop();
  session.wait();

  EXPECT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_optional_state.load(),
            InputState::kSafeDefault);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_optional_reason.load(),
            SelectionReason::kNeverSeen);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_optional_error.load(),
            ErrorCode::kInputEmpty);
}

// The member that arrives last need not define the aligned semantic coordinate. Side selectors
// evaluate at the aligned high reference rather than that final witness coordinate.
// REQ: CONTRACT-006
// REQ: RUNTIME-055
TEST(RunSessionTest, ChannelAlignedSideInputUsesAlignedReferenceWhenOlderMemberArrivesLast) {
  // Key setup: publish the side at the aligned high reference before the older member arrives last.
  // Key assertion: the side sample is selected at that reference and retains value 3.
  // Graph:
  //   source.first    --> sink.first
  //   source.second   --> sink.second
  //   source.optional --> sink.optional
  //   [unwired] sink.additional_optional
  test::RuntimeAlignedChannelProbeState::reset(
      test::RuntimeAlignedChannelScript::kSideInputReference);
  const ExecutionPlan plan = test::make_runtime_aligned_channel_plan(false, true);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool observed = test::RuntimeAlignedChannelProbeState::wait_for_sink(1U);
  session.request_stop();
  session.wait();

  EXPECT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_first_capture.load() -
                test::RuntimeAlignedChannelProbeState::observed_second_capture.load(),
            5);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_optional_state.load(),
            InputState::kSelected);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_optional_reason.load(),
            SelectionReason::kNone);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_optional_error.load(),
            ErrorCode::kSuccess);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_optional.load(), 3U);
}

// An unwired side input owns no producer generation and therefore cannot block closure after every
// connected aligned member has delivered EOS.
// REQ: CONTRACT-022
// REQ: RUNTIME-055
TEST(RunSessionTest, ChannelAlignedUnwiredSideInputDoesNotBlockNaturalCompletion) {
  // Key setup: complete a finite producer after one aligned pair with both optional sides unwired.
  // Key assertion: connected EOS completes the run naturally despite the absent sides.
  // Graph:
  //   source.first  --> sink.first
  //   source.second --> sink.second
  //   [unwired] sink.optional, sink.additional_optional
  test::RuntimeAlignedChannelProbeState::reset(test::RuntimeAlignedChannelScript::kExactBoundary);
  const ExecutionPlan plan = test::make_runtime_aligned_channel_plan(false, false, true);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool observed = test::RuntimeAlignedChannelProbeState::wait_for_sink(1U);
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  const bool completed_naturally = !session.is_running();
  if (!completed_naturally) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(observed);
  ASSERT_TRUE(completed_naturally);
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_optional_state.load(),
            InputState::kSafeDefault);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_optional_reason.load(),
            SelectionReason::kNotConnected);
}

// Replayed declarations alone must not opt an independently scheduled Operator into Channel
// activation preparation when reconstruction did not materialize endpoints for those ports.
// REQ: RUNTIME-004
// REQ: RUNTIME-015
TEST(RunSessionTest, ChannelActivationOwnershipRequiresMaterializedEndpoints) {
  test::RuntimeAlignedChannelProbeState::reset(test::RuntimeAlignedChannelScript::kExactBoundary);
  test::RuntimeUnconnectedPortProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_mixed_channel_unconnected_port_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool channel_observed = test::RuntimeAlignedChannelProbeState::wait_for_sink(1U);
  const bool independent_observed = test::RuntimeUnconnectedPortProbeState::wait_for_compute();
  session.request_stop();
  session.wait();

  EXPECT_TRUE(channel_observed);
  EXPECT_TRUE(independent_observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// An explicitly degraded optional input has callback-visible typed state without acquiring any
// transport authority. Connected-but-silent selectors retain their separate kNeverSeen oracle.
// REQ: CONTRACT-022
// REQ: RUNTIME-055
TEST(RunSessionTest, UnwiredExactSafeDefaultRunsWithNotConnectedSelection) {
  // Key setup: run a clock-driven optional Exact/SafeDefault input and collect channel counters.
  // Key assertion: one callback sees kNotConnected without payload, subscriber, or transport work.
  // Graph:
  //   [unwired] optional.optional [clock-driven operator]
  auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  ExecutionPlan plan = test::make_runtime_optional_safe_default_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->input_selections.size(), 1U);
  EXPECT_TRUE(state->graph.edges().empty());
  EXPECT_TRUE(state->payload_edges.empty());
  EXPECT_TRUE(state->channel_subscribers.empty());
  EXPECT_TRUE(state->process_local_channel_subscribers.empty());
  EXPECT_TRUE(state->cross_process_channel_subscribers.empty());
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(test::RuntimeOptionalSafeDefaultOperator::compute_count.load(), 1U);
  EXPECT_EQ(test::RuntimeOptionalSafeDefaultOperator::observed_state.load(),
            InputState::kSafeDefault);
  EXPECT_EQ(test::RuntimeOptionalSafeDefaultOperator::observed_reason.load(),
            SelectionReason::kNotConnected);
  EXPECT_EQ(test::RuntimeOptionalSafeDefaultOperator::scalar_error.load(), ErrorCode::kInputEmpty);
  EXPECT_EQ(test::RuntimeOptionalSafeDefaultOperator::batch_error.load(),
            ErrorCode::kInputSelectionCardinalityMismatch);

  const auto snapshot = counters->snapshot();
  const auto count = [&](detail::RuntimePhaseCounter counter) {
    return snapshot[detail::RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelPrepareCalls), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelSamplesReceived), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelReadinessPublished), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelInputReads), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelWitnessRetired), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelRetireCalls), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelActivationCutSubscriberProbes), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelActivationCutTransportProbes), 0U);
}

// The transport scan order is deliberately t+10 then t+5. One serial compute must publish an
// indivisible process-local activation cut before the next freeze so global TagTime ordering still
// invokes the t+5 sink first, independent of the background pump's interleaving.
// REQ: RUNTIME-004
// REQ: RUNTIME-015
// REQ: RUNTIME-016
// REQ: RUNTIME-018
TEST(RunSessionTest, PostComputeActivationCutOrdersOpposingSubscriberScanByTagTime) {
  // Key setup: compile two source outputs with two HoloMQ recipients apiece, suppress all four
  // event wakes, and publish the later-tagged service before the earlier-tagged service.
  test::RuntimeActivationCutProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_activation_cut_order_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->process_local_channel_services.size(), 2U);
  ASSERT_EQ(state->process_local_channel_subscribers.size(), 4U);
  EXPECT_TRUE(std::ranges::all_of(state->process_local_channel_services, [](const auto& service) {
    return service.delivery_kind == detail::ProcessLocalChannelDeliveryKind::kHoloMQ &&
           !service.event_wake &&
           service.event_wake_reason == detail::ProcessLocalHoloMQEventWakeReason::kSuppressed;
  }));
  EXPECT_TRUE(std::ranges::none_of(state->process_local_channel_subscribers,
                                   [](const auto& subscriber) { return subscriber.event_wake; }));
  const auto source_name = [&](PortId source) -> std::string_view {
    const auto port = std::find_if(state->ports.begin(), state->ports.end(), [&](const auto& row) {
      return row.id == source;
    });
    return port == state->ports.end() ? std::string_view{} : std::string_view{port->local_name};
  };
  EXPECT_EQ(source_name(state->process_local_channel_subscribers[0].source), "a-later");
  EXPECT_EQ(source_name(state->process_local_channel_subscribers[1].source), "a-later");
  EXPECT_EQ(source_name(state->process_local_channel_subscribers[2].source), "z-earlier");
  EXPECT_EQ(source_name(state->process_local_channel_subscribers[3].source), "z-earlier");

  RunSession session = run_async(plan);
  const bool observed = test::RuntimeActivationCutProbeState::wait_for_sinks(4U);
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const auto observations = test::RuntimeActivationCutProbeState::snapshot();
  ASSERT_EQ(observations.size(), 4U);
  // Key assertion: the single post-compute cut materializes every eventless recipient before the
  // next freeze, so both earlier-tagged activations run before either later-tagged activation.
  EXPECT_EQ(observations[0].value, 5U);
  EXPECT_EQ(observations[1].value, 5U);
  EXPECT_EQ(observations[2].value, 10U);
  EXPECT_EQ(observations[3].value, 10U);
  EXPECT_LT(observations[1].capture_timestamp_ns, observations[2].capture_timestamp_ns);
  EXPECT_LT(observations[1].tag_time, observations[2].tag_time);
  for (const auto& observation : observations) {
    EXPECT_EQ(observation.capture_timestamp_ns, observation.tag_time.logical_ns);
  }
}

// REQ: RUNTIME-004
// REQ: RUNTIME-007
// REQ: RUNTIME-015
TEST(RunSessionTest, ChannelWindowSuccessorKeepsAnchorButUsesLaterActivationTime) {
  test::RuntimeWindowChannelProbeState::reset(test::RuntimeWindowChannelScript::kSuccessorClose);
  const ExecutionPlan plan = test::make_runtime_window_channel_plan();
  test::partition_parity::expect_accepted_cover(plan);

  const test::partition_parity::ScopedEnvironmentMutation changed_after_compile{
      "definitely-not-an-algorithm"};
  RunSession session = run_async(plan);
  const bool observed = test::RuntimeWindowChannelProbeState::wait_for_compute();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::selection_state.load(), InputState::kSelected);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::selection_reason.load(), SelectionReason::kNone);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::batch_size.load(), 5U);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::anchor_capture_ns.load() -
                test::RuntimeWindowChannelProbeState::batch_first_capture_ns.load(),
            15);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::batch_last_capture_ns.load() -
                test::RuntimeWindowChannelProbeState::anchor_capture_ns.load(),
            5);
  EXPECT_GT(test::RuntimeWindowChannelProbeState::activation_time_ns.load(),
            test::RuntimeWindowChannelProbeState::anchor_capture_ns.load());
}

// REQ: RUNTIME-004
// REQ: RUNTIME-007
// REQ: RUNTIME-015
TEST(RunSessionTest, ChannelWindowTimeoutInvokesWithMissingWindowSelection) {
  test::RuntimeWindowChannelProbeState::reset(test::RuntimeWindowChannelScript::kTimeoutClose);
  const ExecutionPlan plan = test::make_runtime_window_channel_plan();
  test::partition_parity::expect_accepted_cover(plan);

  const test::partition_parity::ScopedEnvironmentMutation changed_after_compile{
      "definitely-not-an-algorithm"};
  RunSession session = run_async(plan);
  const bool observed = test::RuntimeWindowChannelProbeState::wait_for_compute();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::selection_state.load(), InputState::kSafeDefault);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::selection_reason.load(),
            SelectionReason::kWindowTimeout);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::batch_size.load(), 0U);
  EXPECT_GT(test::RuntimeWindowChannelProbeState::activation_time_ns.load(),
            test::RuntimeWindowChannelProbeState::anchor_capture_ns.load());
}

// One physical producer EOS closes both pending bracketed windows. Each all-or-nothing terminal
// decision must invoke the SafeDefault branch once, publish downstream once, and retire before the
// sink forwards EOS and the finite graph completes naturally.
// REQ: RUNTIME-050
// REQ: CHAN-136
TEST(RunSessionTest, ChannelWindowEndOfStreamDrainsEveryPendingDecisionBeforeDownstreamEos) {
  test::RuntimeWindowChannelProbeState::reset(test::RuntimeWindowChannelScript::kEndOfStreamDrain);
  const ExecutionPlan compiled = test::make_runtime_window_channel_plan();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::disable_direct_continuation_elision_for_test(compiled);
  test::partition_parity::expect_accepted_cover(plan);

  RunSession session = run_async(plan);
  const bool drained = test::RuntimeWindowChannelProbeState::wait_for_compute_count(2U);
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  const bool completed_naturally = !session.is_running();
  if (!completed_naturally) {
    session.request_stop();
  }
  session.wait();

  ASSERT_TRUE(drained)
      << "window_compute=" << test::RuntimeWindowChannelProbeState::compute_count.load()
      << " eos_defaults="
      << test::RuntimeWindowChannelProbeState::safe_default_end_of_stream_count.load()
      << " downstream=" << test::RuntimeWindowChannelProbeState::downstream_count.load()
      << " termination=" << static_cast<std::uint32_t>(session.termination());
  ASSERT_TRUE(completed_naturally);
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::compute_count.load(), 2U);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::safe_default_end_of_stream_count.load(), 2U);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::downstream_count.load(), 2U);
  const auto& tags = test::RuntimeWindowChannelProbeState::terminal_tags;
  const auto& tag_times = test::RuntimeWindowChannelProbeState::terminal_tag_times;
  ASSERT_TRUE(tags[0].valid());
  ASSERT_TRUE(tags[1].valid());
  EXPECT_EQ(tags[1].generation, tags[0].generation);
  EXPECT_EQ(tags[1].sequence, tags[0].sequence + 1U);
  ASSERT_TRUE(tag_times[0].valid());
  ASSERT_TRUE(tag_times[1].valid());
  EXPECT_EQ(tag_times[1].logical_ns, tag_times[0].logical_ns);
  EXPECT_EQ(tag_times[1].microstep, tag_times[0].microstep + 1U);
}

// Each partition runner observes only the window-trigger state it owns. A
// same-lane source partition must not attempt to evaluate the sink partition's
// selector state while polling timeout deadlines.
// REQ: RUNTIME-004
// REQ: RUNTIME-007
// REQ: RUNTIME-041
TEST(RunSessionTest, CrossPartitionWindowTimeoutEvaluatesOnlyTheOwningRunner) {
  test::RuntimeWindowChannelProbeState::reset(test::RuntimeWindowChannelScript::kTimeoutClose);
  const ExecutionPlan baseline = test::make_runtime_window_channel_plan(true);
  test::partition_parity::expect_accepted_cover(baseline);
  const auto* const compiled = detail::ExecutionPlanAccess::state(baseline);
  ASSERT_NE(compiled, nullptr);
  ASSERT_EQ(compiled->window_triggers.size(), 1U);
  const auto sink_factory =
      std::find_if(compiled->executable_factories.operators.begin(),
                   compiled->executable_factories.operators.end(),
                   [&](const auto& factory) {
                     return factory.operator_id == compiled->window_triggers.front().owner.id;
                   });
  ASSERT_NE(sink_factory, compiled->executable_factories.operators.end());
  const auto sink_partition =
      std::find_if(compiled->execution_partitions.begin(),
                   compiled->execution_partitions.end(),
                   [&](const auto& partition) {
                     return std::find(partition.members.begin(),
                                      partition.members.end(),
                                      sink_factory->vertex) != partition.members.end();
                   });
  ASSERT_NE(sink_partition, compiled->execution_partitions.end());
  const auto timeout_control = std::make_shared<detail::RuntimeWindowTimeoutTestControl>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_window_timeout_test_control(baseline, timeout_control);

  RunSession session = run_async(plan);
  const bool observed = test::RuntimeWindowChannelProbeState::wait_for_compute();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::selection_state.load(), InputState::kSafeDefault);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::selection_reason.load(),
            SelectionReason::kWindowTimeout);
  const auto timeout_observations = timeout_control->snapshot();
  ASSERT_EQ(timeout_observations.size(), 1U);
  EXPECT_EQ(timeout_observations.front().owner, sink_factory->vertex);
  EXPECT_EQ(timeout_observations.front().partition, sink_partition->id);
}

// REQ: RUNTIME-004
// REQ: RUNTIME-015
TEST(RunSessionTest, RejectsInputPredicateSelectorFiringAndDeadlineReplayMismatches) {
  const ExecutionPlan plan = test::make_runtime_input_predicate_replay_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  constexpr std::array mismatches{
      test::RuntimeInputContractReplayMode::kSelectorMismatch,
      test::RuntimeInputContractReplayMode::kPredicateFiringMismatch,
      test::RuntimeInputContractReplayMode::kMaxStartDelayMismatch,
  };
  for (const auto mismatch : mismatches) {
    SCOPED_TRACE(static_cast<unsigned>(mismatch));
    test::RuntimeInputContractPredicateOperator::mode.store(mismatch, std::memory_order_relaxed);
    auto reconstructed = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                                        RunGenerationId{19U});
    ASSERT_FALSE(reconstructed);
    EXPECT_EQ(reconstructed.error().message, "RUNTIME_OPERATOR_REPLAY_MISMATCH");
  }
  test::RuntimeInputContractPredicateOperator::mode.store(
      test::RuntimeInputContractReplayMode::kMatch, std::memory_order_relaxed);
}

// REQ: RUNTIME-004
// REQ: RUNTIME-007
// REQ: RUNTIME-019
// REQ: CONTRACT-011
TEST(RunSessionTest, SuccessfulFalsePredicateRetiresInputsBeyondTheQueueCapacity) {
  reset_predicate_retirement_probe();
  const ExecutionPlan plan = make_predicate_retirement_plan();
  test::partition_parity::expect_accepted_cover(plan);

  RunSession session = run_async(plan);
  const bool computed = wait_for_predicate_retirement_compute();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(computed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  // Four successful false evaluations exceed the two-slot queue. Reaching the
  // fifth witness proves each no-fire transaction retired its predicate input.
  const PredicateRetirementSnapshot snapshot = predicate_retirement_snapshot();
  EXPECT_EQ(snapshot.source_count, 5U);
  EXPECT_EQ(snapshot.evaluation_count, 5U);
  EXPECT_EQ(snapshot.compute_count, 1U);
  EXPECT_EQ(snapshot.computed_driver, 5U);
}

// REQ: RUNTIME-062
TEST(RunSessionTest, FixtureObservationPreservesPredicateSuppressionAndComputeOnlySelection) {
  reset_predicate_retirement_probe();
  const auto plan = make_predicate_compute_skip_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  auto observation =
      std::make_shared<detail::RuntimeFixtureObservation>(1024U, plan.ports().size(), 1024U);
  auto session = detail::start_operator_test_run(plan, observation);
  const bool computed = wait_for_predicate_retirement_compute();
  session.request_stop();
  session.wait();
  ASSERT_TRUE(computed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  ASSERT_FALSE(observation->invalid())
      << "retained observation records: " << observation->records().size();
  const auto vertices = plan.graph().vertices();
  const auto sink = std::ranges::find(vertices, "sink", &VertexRecord::name);
  ASSERT_NE(sink, vertices.end());
  PortId driver{}, peer{};
  for (const auto& port : plan.ports()) {
    if (port.owner != sink->id)
      continue;
    if (port.local_name == "driver")
      driver = port.id;
    if (port.local_name == "compute_only")
      peer = port.id;
  }
  ASSERT_NE(driver, PortId{});
  ASSERT_NE(peer, PortId{});
  std::size_t matched{};
  for (const auto& record : observation->records()) {
    if (record.owner != sink->id ||
        record.kind != detail::FixtureObservationKind::kCandidateFinished)
      continue;
    const auto selected =
        std::ranges::find(record.inputs, driver, &detail::FixtureObservedInput::input);
    if (selected == record.inputs.end() || !selected->has_sample)
      continue;
    ++matched;
    const bool first = selected->sample.source_sequence == 1U;
    EXPECT_EQ(selected->sample.source_sequence, first ? 1U : 2U);
    EXPECT_EQ(record.admitted, !first);
    EXPECT_EQ(record.outcome,
              first ? detail::FixtureCandidateOutcome::kSuppressed
                    : detail::FixtureCandidateOutcome::kRetired);
    const auto compute_only =
        std::ranges::find(record.inputs, peer, &detail::FixtureObservedInput::input);
    ASSERT_NE(compute_only, record.inputs.end());
    EXPECT_EQ(compute_only->has_sample, !first);
    EXPECT_EQ(compute_only->selection.state, first ? InputState::kAbsent : InputState::kSelected);
    if (!first)
      EXPECT_EQ(compute_only->sample.source_sequence, 1U);
  }
  EXPECT_EQ(matched, 2U);
  const auto snapshot = predicate_retirement_snapshot();
  EXPECT_EQ(snapshot.computed_driver, 2U);
  EXPECT_EQ(snapshot.computed_peer, 200U);
}

// REQ: RUNTIME-004
// REQ: RUNTIME-007
// REQ: RUNTIME-019
// REQ: CONTRACT-011
TEST(RunSessionTest, ComputeOnlySkipDoesNotRollBackASuccessfulPredicateInput) {
  // Key setup: force both source outputs through separate HoloMQ services into one two-input
  // predicate target, where the first driver has no compute-only peer.
  reset_predicate_retirement_probe();
  const ExecutionPlan plan = make_predicate_compute_skip_plan();
  test::partition_parity::expect_accepted_cover(plan);
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->process_local_channel_services.size(), 2U);
  ASSERT_EQ(state->process_local_channel_subscribers.size(), 2U);
  EXPECT_TRUE(std::ranges::all_of(state->process_local_channel_services, [](const auto& service) {
    return service.delivery_kind == detail::ProcessLocalChannelDeliveryKind::kHoloMQ &&
           !service.event_wake &&
           service.event_wake_reason == detail::ProcessLocalHoloMQEventWakeReason::kSuppressed;
  }));
  EXPECT_TRUE(std::ranges::none_of(state->process_local_channel_subscribers,
                                   [](const auto& subscriber) { return subscriber.event_wake; }));

  RunSession session = run_async(plan);
  const bool computed = wait_for_predicate_retirement_compute();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(computed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  // Key assertion: driver 1 evaluates successfully but its missing compute-only peer suppresses
  // the callback. Both eventless services then cross the next source activation cut, so driver 2
  // advances and adopts peer 200 with the same capture coordinate instead of replaying driver 1.
  const PredicateRetirementSnapshot snapshot = predicate_retirement_snapshot();
  EXPECT_EQ(snapshot.source_count, 2U);
  EXPECT_EQ(snapshot.evaluation_count, 2U);
  EXPECT_EQ(snapshot.compute_count, 1U);
  EXPECT_EQ(snapshot.computed_driver, 2U);
  EXPECT_EQ(snapshot.computed_peer, 200U);
}

// REQ: RUNTIME-004
// REQ: RUNTIME-007
// REQ: RUNTIME-019
// Key setup: for each vector the direct oracle runs first to derive expected
// evaluator/compute counts; the Channel binding then runs and its observations
// are compared element-by-element against the oracle's per-candidate output.
TEST(RunSessionTest, ChannelInputPredicateMatchesDirectOracleAcrossTheBehaviorMatrix) {
  struct Vector {
    test::RuntimeInputPredicatePattern pattern;
    PredicateFiring firing;
    bool configured;
    test::RuntimeInputPredicateMissing missing;
  };
  constexpr std::array vectors{
      Vector{test::RuntimeInputPredicatePattern::kFalseTrueTrue,
             PredicateFiring::kOnFalseToTrue,
             false,
             test::RuntimeInputPredicateMissing::kFresh},
      Vector{test::RuntimeInputPredicatePattern::kFalseTrueTrue,
             PredicateFiring::kOnEveryTrueEvaluation,
             false,
             test::RuntimeInputPredicateMissing::kFresh},
      Vector{test::RuntimeInputPredicatePattern::kAlwaysTrue,
             PredicateFiring::kOnEveryTrueEvaluation,
             true,
             test::RuntimeInputPredicateMissing::kFresh},
      Vector{test::RuntimeInputPredicatePattern::kAlwaysTrue,
             PredicateFiring::kOnEveryTrueEvaluation,
             false,
             test::RuntimeInputPredicateMissing::kReuseLast},
      Vector{test::RuntimeInputPredicatePattern::kAlwaysTrue,
             PredicateFiring::kOnEveryTrueEvaluation,
             false,
             test::RuntimeInputPredicateMissing::kSkip},
      Vector{test::RuntimeInputPredicatePattern::kAlwaysTrue,
             PredicateFiring::kOnEveryTrueEvaluation,
             false,
             test::RuntimeInputPredicateMissing::kSafeDefault},
  };

  for (const Vector& vector : vectors) {
    SCOPED_TRACE(static_cast<unsigned>(vector.firing));
    SCOPED_TRACE(static_cast<unsigned>(vector.missing));
    auto direct = run_direct_predicate_oracle(
        vector.pattern, vector.firing, vector.configured, vector.missing);
    ASSERT_TRUE(direct) << direct.error().message;
    const auto expected_evaluator_count = static_cast<std::size_t>(std::count_if(
        direct->begin(), direct->end(), [](const auto& row) { return row.evaluator_called; }));
    const auto expected_compute_count =
        static_cast<std::size_t>(std::count_if(direct->begin(), direct->end(), [](const auto& row) {
          return row.disposition == detail::PredicateCandidateDisposition::kDegradedCompute ||
                 (row.disposition == detail::PredicateCandidateDisposition::kNominalCompute &&
                  row.firing_accepted);
        }));

    test::RuntimeInputPredicateChannelState::reset();
    const ExecutionPlan plan = test::make_runtime_input_predicate_channel_plan(
        vector.pattern, vector.firing, vector.configured, vector.missing);
    test::partition_parity::expect_accepted_cover(plan);
    ASSERT_EQ(plan.predicate_triggers().size(), 1U);
    ASSERT_EQ(plan.predicate_inputs().size(), 2U);
    EXPECT_TRUE(plan.predicate_inputs()[0].implicit_driver);
    EXPECT_EQ(plan.predicate_inputs()[0].argument_ordinal, 0U);
    EXPECT_FALSE(plan.predicate_inputs()[1].implicit_driver);
    EXPECT_EQ(plan.predicate_inputs()[1].argument_ordinal, 1U);
    EXPECT_EQ(plan.predicate_triggers().front().has_config, vector.configured);
    // Callable admission is compiler-owned and is exhaustively classified by
    // CompilePredicatePlan.AppliesTheClosedCallableProfileMatrix. The Channel
    // binding consumes this admitted record without transport-side reclassification.
    ASSERT_EQ(plan.predicate_callables().size(), 1U);
    EXPECT_EQ(plan.predicate_callables().front().effective_profile, Profile::kStandard);
    EXPECT_EQ(plan.predicate_callables().front().disposition,
              PredicateCallableDisposition::kAdmitted);
    EXPECT_FALSE(plan.predicate_callables().front().implementation_fingerprint.empty());
    if (vector.configured) {
      EXPECT_EQ(plan.predicate_triggers().front().config_schema, "test.runtime.input-predicate/v1");
      ASSERT_TRUE(plan.predicate_triggers().front().config_digest.has_value());
      EXPECT_EQ(plan.predicate_triggers().front().config_digest.value_or(std::string{}).size(),
                71U);
    } else {
      EXPECT_FALSE(plan.predicate_triggers().front().config_schema.has_value());
      EXPECT_FALSE(plan.predicate_triggers().front().config_digest.has_value());
    }

    const test::partition_parity::ScopedEnvironmentMutation changed_after_compile{
        "definitely-not-an-algorithm"};
    RunSession session = run_async(plan);
    const bool sourced = test::RuntimeInputPredicateChannelState::wait_for_source();
    const bool evaluated =
        expected_evaluator_count == 0U ||
        test::RuntimeInputPredicateChannelState::wait_for_evaluator(expected_evaluator_count);
    const bool computed =
        expected_compute_count == 0U ||
        test::RuntimeInputPredicateChannelState::wait_for_compute(expected_compute_count);
    if (expected_evaluator_count == 0U || expected_compute_count == 0U) {
      std::this_thread::sleep_for(std::chrono::milliseconds{25});
    }
    session.request_stop();
    session.wait();

    ASSERT_TRUE(sourced);
    ASSERT_TRUE(evaluated);
    ASSERT_TRUE(computed);
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty())
        << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
    // Key assertion: channel evaluator and compute counts exactly match the oracle.
    EXPECT_EQ(test::RuntimeInputPredicateChannelState::evaluator_count.load(),
              expected_evaluator_count);
    EXPECT_EQ(test::RuntimeInputPredicateChannelState::compute_count.load(),
              expected_compute_count);

    const auto evaluations = test::RuntimeInputPredicateChannelState::evaluation_snapshot();
    const auto computes = test::RuntimeInputPredicateChannelState::compute_snapshot();
    ASSERT_EQ(evaluations.size(), expected_evaluator_count);
    ASSERT_EQ(computes.size(), expected_compute_count);
    std::size_t evaluation_index = 0U;
    std::size_t compute_index = 0U;
    for (std::size_t candidate = 0U; candidate < direct->size(); ++candidate) {
      const auto& expected = (*direct)[candidate];
      if (expected.evaluator_called) {
        ASSERT_LT(evaluation_index, evaluations.size());
        const auto& actual = evaluations[evaluation_index++];
        EXPECT_EQ(actual.driver_value, candidate + 1U);
        EXPECT_EQ(actual.candidate_state_count, expected.candidate_state_count);
        EXPECT_EQ(actual.value, expected.value);
        EXPECT_EQ(actual.failed,
                  expected.disposition == detail::PredicateCandidateDisposition::kEvaluatorError);
        expect_input_selection_equal({InputState::kSelected, SelectionReason::kNone},
                                     actual.driver_selection);
        expect_input_selection_equal(expected.additional_selection, actual.additional_selection);
        EXPECT_EQ(actual.reference_time, actual.driver_metadata.capture_time());
        if (vector.missing == test::RuntimeInputPredicateMissing::kReuseLast && candidate > 0U) {
          EXPECT_EQ(actual.additional_value, 10U);
          EXPECT_EQ(actual.additional_metadata.sequence, 1U);
        }
      }
      const bool expects_compute =
          expected.disposition == detail::PredicateCandidateDisposition::kDegradedCompute ||
          (expected.disposition == detail::PredicateCandidateDisposition::kNominalCompute &&
           expected.firing_accepted);
      if (!expects_compute) {
        continue;
      }
      ASSERT_LT(compute_index, computes.size());
      const auto& actual_compute = computes[compute_index++];
      EXPECT_EQ(actual_compute.driver_value, candidate + 1U);
      expect_input_selection_equal({InputState::kSelected, SelectionReason::kNone},
                                   actual_compute.driver_selection);
      expect_input_selection_equal(expected.additional_selection,
                                   actual_compute.additional_selection);
      if (expected.disposition == detail::PredicateCandidateDisposition::kDegradedCompute) {
        EXPECT_FALSE(actual_compute.has_additional);
        expect_input_selection_equal(
            {InputState::kNotSelected, SelectionReason::kNotSelectedForActivation},
            actual_compute.compute_only_selection);
        continue;
      }
      const auto found = std::find_if(evaluations.begin(), evaluations.end(), [&](const auto& row) {
        return row.driver_value == actual_compute.driver_value;
      });
      ASSERT_NE(found, evaluations.end());
      EXPECT_EQ(actual_compute.tag, found->tag);
      EXPECT_EQ(actual_compute.tag_time, found->tag_time);
      EXPECT_EQ(actual_compute.activation_time, found->reference_time);
      expect_sample_metadata_equal(found->driver_metadata, actual_compute.driver_metadata);
      ASSERT_TRUE(actual_compute.has_additional);
      EXPECT_EQ(actual_compute.additional_value, found->additional_value);
      expect_sample_metadata_equal(found->additional_metadata, actual_compute.additional_metadata);
    }
    if (vector.missing == test::RuntimeInputPredicateMissing::kFresh) {
      ASSERT_EQ(evaluations.size(), 3U);
      // All three additional samples were queued before any driver. Exact
      // evaluation nevertheless pins the candidate-time sample rather than the
      // newest arrival, and compute adopts that identical snapshot.
      EXPECT_EQ(evaluations.front().additional_metadata.sequence, 1U);
      EXPECT_EQ(evaluations.back().additional_metadata.sequence, 3U);
    }
  }
}

// REQ: RUNTIME-004
// REQ: RUNTIME-007
// REQ: RUNTIME-019
TEST(RunSessionTest, ChannelInputPredicateErrorRollsBackWithoutCompute) {
  // Key setup: kTrueThenError fires on the first candidate (compute==1), then
  // returns an error on the second. The evaluator's private state count must stay
  // at 2 — proving the error rolled back without resetting prior truth.
  auto direct = run_direct_predicate_oracle(test::RuntimeInputPredicatePattern::kTrueThenError,
                                            PredicateFiring::kOnEveryTrueEvaluation,
                                            false,
                                            test::RuntimeInputPredicateMissing::kFresh);
  ASSERT_TRUE(direct) << direct.error().message;
  ASSERT_EQ(direct->size(), 3U);
  EXPECT_EQ((*direct)[0].disposition, detail::PredicateCandidateDisposition::kNominalCompute);
  EXPECT_EQ((*direct)[1].disposition, detail::PredicateCandidateDisposition::kEvaluatorError);
  EXPECT_EQ((*direct)[2].disposition, detail::PredicateCandidateDisposition::kEvaluatorError);
  EXPECT_TRUE((*direct)[1].prior_truth);
  EXPECT_TRUE((*direct)[2].prior_truth);
  EXPECT_EQ((*direct)[1].candidate_state_count, 2U);
  EXPECT_EQ((*direct)[2].candidate_state_count, 2U);

  auto scenario = run_channel_input_predicate_error_scenario();
  ASSERT_TRUE(scenario) << scenario.error().message;
  const PredicateErrorScenarioEvidence& evidence = *scenario;
  const PredicateTriggerPlanRecord& trigger = evidence.trigger;

  // Key assertion: the session fails after exactly 2 evaluations and 1 compute;
  // the second evaluation's candidate_state_count==2 confirms state was preserved.
  EXPECT_EQ(evidence.termination, RunTermination::kFailed);
  EXPECT_EQ(evidence.evaluator_count, 2U);
  EXPECT_EQ(evidence.compute_count, 1U);
  const auto evaluations = evidence.evaluations;
  ASSERT_EQ(evaluations.size(), 2U);
  EXPECT_EQ(evaluations[0].candidate_state_count, 1U);
  EXPECT_EQ(evaluations[1].candidate_state_count, 2U);
  EXPECT_FALSE(evaluations[0].failed);
  EXPECT_TRUE(evaluations[1].failed);
  ASSERT_EQ(evidence.diagnostics.size(), 1U);
  EXPECT_EQ(evidence.diagnostics.front().token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
  EXPECT_EQ(evidence.diagnostics.front().service, "predicate_evaluator");
  const auto published = evidence.published_details;
  ASSERT_EQ(published.size(), 1U);
  ASSERT_EQ(published.front().size(), 1U);
  const detail::RuntimeDiagnosticDetail& detail = published.front().front();
  EXPECT_TRUE(detail.present);
  EXPECT_EQ(detail.error.code, ErrorCode::kFailure);
  EXPECT_EQ(detail.error.message, "input predicate failure");
  EXPECT_EQ(detail.predicate_id, trigger.id);
  EXPECT_EQ(detail.operator_id, trigger.owner.id);
  EXPECT_TRUE(static_cast<bool>(detail.source));
  // The source emits three predicate inputs and three compute-only samples
  // before the driver stream, so the second driver is publication witness 8
  // even though its per-port Sample sequence is 2.
  EXPECT_EQ(detail.source_sequence, 8U);
  EXPECT_EQ(detail.tag, evaluations[1].tag);
  EXPECT_EQ(detail.tag_time, evaluations[1].tag_time);
  EXPECT_EQ(detail.reference_time, evaluations[1].reference_time);
  EXPECT_EQ(detail.state_generation, detail.tag.generation);
}

// Generated-schema fanout across execution partitions for the private fanout experiment. Each
// sample carries its publication order as text. Completion uses capacity for every sample plus
// EOS; a separate depth-one case gates the slow sink to observe pressure without relying on sleep.
struct RuntimeSchemaFanoutExperimentProbe {
  static constexpr std::uint32_t kComputeCalls = 64U;
  static inline std::atomic<std::uint32_t> emitted{};
  static constexpr std::uint32_t kMaximumPressureEmits = 12U;
  static constexpr int kNoOutcome = -1;
  static constexpr int kNoPressure = -2;
  static inline std::atomic<bool> pressure{};
  static inline std::atomic<bool> release_slow{};
  static inline std::atomic<int> outcome{kNoOutcome};
  static inline std::atomic<std::uint32_t> attempts{};
  static inline std::array<std::atomic<std::uint32_t>, 2U> received{};
  static inline std::array<std::atomic<std::uint64_t>, 2U> last_value{};
  static inline std::atomic<bool> order_violation{};

  static void reset(const bool under_pressure) {
    pressure.store(under_pressure);
    release_slow.store(false);
    outcome.store(kNoOutcome);
    attempts.store(0U);
    emitted.store(0U);
    for (auto& count : received) {
      count.store(0U);
    }
    for (auto& value : last_value) {
      value.store(0U);
    }
    order_violation.store(false);
  }

  // Publish the outcome before releasing the sink that deliberately holds its first sample.
  static void settle(const int code) {
    outcome.store(code);
    release_slow.store(true);
  }

  template <typename Predicate>
  [[nodiscard]] static bool wait_for(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    while (!predicate()) {
      if (std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      std::this_thread::sleep_for(100us);
    }
    return true;
  }
};

class RuntimeSchemaFanoutExperimentSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms})
        .complete_after(ComputeCalls{.limit = RuntimeSchemaFanoutExperimentProbe::kComputeCalls});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    using Probe = RuntimeSchemaFanoutExperimentProbe;
    const bool pressure = Probe::pressure.load();
    if (pressure) {
      if (Probe::outcome.load() != Probe::kNoOutcome) {
        return {};
      }
      // Keep the fast recipient drained so only the gated recipient can limit publication.
      if (!Probe::wait_for([] { return Probe::received[0U].load() == Probe::emitted.load(); })) {
        Probe::settle(static_cast<int>(ErrorCode::kTimeout));
        return make_unexpected(Error{ErrorCode::kTimeout});
      }
    }
    schema::StringT value;
    value.data = std::to_string(++value_);
    Probe::attempts.fetch_add(1U);
    auto published = output.emit(std::move(value));
    if (!published) {
      if (!pressure) {
        return published;
      }
      const ErrorCode code = published.error().code;
      Probe::settle(static_cast<int>(code));
      const bool post_submit =
          code == ErrorCode::kPublicationPartial || code == ErrorCode::kPublicationIndeterminate;
      if (code != ErrorCode::kBackpressured && !post_submit) {
        return published;
      }
      // Let both recipients observe exactly what reached them before propagating the error.
      // The failed sample is never retried or counted as a successful publication.
      const std::uint32_t successes = Probe::emitted.load();
      if (!Probe::wait_for([&] {
            return Probe::received[1U].load() == successes &&
                   Probe::received[0U].load() == successes + (post_submit ? 1U : 0U);
          })) {
        return make_unexpected(Error{ErrorCode::kTimeout});
      }
      if (post_submit) {
        return published;
      }
      return {};
    }
    if (Probe::emitted.fetch_add(1U) + 1U >= Probe::kMaximumPressureEmits && pressure) {
      Probe::settle(Probe::kNoPressure);
    }
    return {};
  }

  Output<schema::StringT> output;

 private:
  std::uint64_t value_{};
};

template <std::size_t kSink, bool kSlow>
class RuntimeSchemaFanoutExperimentSinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    using Probe = RuntimeSchemaFanoutExperimentProbe;
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    const std::uint64_t value = std::stoull(sample->data.data);
    if (value <= Probe::last_value[kSink].exchange(value)) {
      Probe::order_violation.store(true);
    }
    Probe::received[kSink].fetch_add(1U);
    if constexpr (kSlow) {
      if (Probe::pressure.load()) {
        if (!Probe::wait_for([] { return Probe::release_slow.load(); })) {
          return make_unexpected(Error{ErrorCode::kTimeout});
        }
      } else {
        std::this_thread::sleep_for(2ms);
      }
    }
    return {};
  }

  Input<schema::StringT> input;
};

enum class SchemaFanoutArm : std::uint8_t { kPump, kEmit, kEmitConcurrent };

class RuntimeSchemaFanoutExperiment : public ::testing::TestWithParam<SchemaFanoutArm> {
 protected:
  [[nodiscard]] static RunSession run(
      const SchemaFanoutArm arm, const std::uint32_t queue_depth,
      const std::shared_ptr<detail::RuntimePhaseCounters>& counters) {
    Graph graph{"runtime-schema-fanout-experiment"};
    auto source = graph.op<RuntimeSchemaFanoutExperimentSourceOperator>("source");
    auto fast = graph.op<RuntimeSchemaFanoutExperimentSinkOperator<0U, false>>("fast");
    auto slow = graph.op<RuntimeSchemaFanoutExperimentSinkOperator<1U, true>>("slow");
    graph.add_flow(source->output, fast->input, ConnectionOptions{.queue_depth = queue_depth});
    graph.add_flow(source->output, slow->input, ConnectionOptions{.queue_depth = queue_depth});
    graph.partition("source-partition").add(source);
    graph.partition("fast-partition").add(fast);
    graph.partition("slow-partition").add(slow);
    const ClockKey clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
    graph.set_default_clock(clock);
    ExecutionPlan plan =
        arm == SchemaFanoutArm::kPump
            ? detail::compile_with_process_local_pump_delivery(graph, CompileOptions{})
            : detail::compile_with_process_local_fanout_experiment(
                  graph, CompileOptions{}, detail::ProcessLocalFanoutExperiment::kEmitRecipients);
    EXPECT_TRUE(plan.ok()) << plan.json();
    plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
    if (arm == SchemaFanoutArm::kEmitConcurrent) {
      auto control = std::make_shared<detail::RuntimeFanoutTestControl>();
      control->concurrent_admission = true;
      plan = detail::RuntimeTestPlanAccess::with_fanout_test_control(plan, std::move(control));
    }

    RunSession session = run_async(plan);
    if (!session.wait_for(10s)) {
      using Probe = RuntimeSchemaFanoutExperimentProbe;
      Probe::release_slow.store(true);
      ADD_FAILURE() << "session did not complete: emitted=" << Probe::emitted.load()
                    << " fast=" << Probe::received[0U].load()
                    << " slow=" << Probe::received[1U].load();
      session.request_stop();
    }
    session.wait();

    return session;
  }
};

// Every connection can retain all 64 samples and EOS, regardless of lane scheduling. No pressure
// outcome is accepted in this completion test; the depth-one test below covers that boundary.
// REQ: CHAN-136
TEST_P(RuntimeSchemaFanoutExperiment, DeliversEveryPublishedSchemaSampleAcrossPartitions) {
  using Probe = RuntimeSchemaFanoutExperimentProbe;
  Probe::reset(false);
  const SchemaFanoutArm arm = GetParam();
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const RunSession session = run(arm, Probe::kComputeCalls + 1U, counters);

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(Probe::attempts.load(), Probe::kComputeCalls);
  EXPECT_EQ(Probe::emitted.load(), Probe::kComputeCalls);
  EXPECT_EQ(Probe::received[0U].load(), Probe::kComputeCalls);
  EXPECT_EQ(Probe::received[1U].load(), Probe::kComputeCalls);
  EXPECT_EQ(Probe::last_value[0U].load(), Probe::kComputeCalls);
  EXPECT_EQ(Probe::last_value[1U].load(), Probe::kComputeCalls);
  EXPECT_FALSE(Probe::order_violation.load());
  const auto pump_waits = counters->snapshot()[detail::RuntimePhaseCounters::index(
      detail::RuntimePhaseCounter::kChannelPumpWaits)];
  if (arm == SchemaFanoutArm::kPump) {
    EXPECT_GT(pump_waits, 0U);
  } else {
    EXPECT_EQ(pump_waits, 0U);
  }
}

// Hold the slow sink's first sample and keep the fast sink drained. The next blocked publication
// must report an explicit pre-submit or post-submit outcome. Opening the gate preserves exact
// recipient visibility and ordering; no later compute retries or replaces the failed sample.
// REQ: CHAN-108
// REQ: CHAN-136
TEST_P(RuntimeSchemaFanoutExperiment, DepthOnePressureReportsTheBlockedPublicationOutcome) {
  using Probe = RuntimeSchemaFanoutExperimentProbe;
  Probe::reset(true);
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const RunSession session = run(GetParam(), 1U, counters);

  const int outcome = Probe::outcome.load();
  const std::uint32_t emitted = Probe::emitted.load();
  RecordProperty("publication_outcome", outcome);
  RecordProperty("successful_publications", static_cast<int>(emitted));
  RecordProperty("publication_attempts", static_cast<int>(Probe::attempts.load()));
  RecordProperty("fast_received", static_cast<int>(Probe::received[0U].load()));
  RecordProperty("slow_received", static_cast<int>(Probe::received[1U].load()));
  ASSERT_TRUE(outcome == static_cast<int>(ErrorCode::kBackpressured) ||
              outcome == static_cast<int>(ErrorCode::kPublicationPartial) ||
              outcome == static_cast<int>(ErrorCode::kPublicationIndeterminate))
      << "outcome=" << outcome << " emitted=" << emitted;
  EXPECT_GE(emitted, 1U);
  EXPECT_LT(emitted, Probe::kMaximumPressureEmits);
  EXPECT_EQ(Probe::attempts.load(), emitted + 1U);
  const bool post_submit = outcome != static_cast<int>(ErrorCode::kBackpressured);
  const std::uint32_t fast_count = emitted + (post_submit ? 1U : 0U);
  EXPECT_EQ(Probe::received[0U].load(), fast_count);
  EXPECT_EQ(Probe::last_value[0U].load(), fast_count);
  EXPECT_EQ(Probe::received[1U].load(), emitted);
  EXPECT_EQ(Probe::last_value[1U].load(), emitted);
  EXPECT_FALSE(Probe::order_violation.load());
  if (post_submit) {
    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    ASSERT_FALSE(session.diagnostics().empty());
    EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
  } else {
    EXPECT_EQ(session.termination(), RunTermination::kCompleted);
    EXPECT_TRUE(session.diagnostics().empty())
        << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  }
}

// Arm names give each case a readable test identity.
INSTANTIATE_TEST_SUITE_P(Arms, RuntimeSchemaFanoutExperiment,
                         ::testing::Values(SchemaFanoutArm::kPump, SchemaFanoutArm::kEmit,
                                           SchemaFanoutArm::kEmitConcurrent),
                         [](const ::testing::TestParamInfo<SchemaFanoutArm>& info) {
                           switch (info.param) {
                             case SchemaFanoutArm::kPump:
                               return std::string{"Pump"};
                             case SchemaFanoutArm::kEmit:
                               return std::string{"Emit"};
                             case SchemaFanoutArm::kEmitConcurrent:
                               return std::string{"EmitConcurrent"};
                           }
                           return std::string{"Unknown"};
                         });

// Stop-time publication classification (S2) and stop-aware sends (S3). The source publishes one
// sample and waits until the stalled consumer holds it. It then keeps publishing until one
// publication reports pressure, which it drops: the stalled recipient is now full, however much
// the delivery path buffers. Finally it requests stop from its own callback and publishes into
// that full recipient. S3 must end the blocked publication at once, and S2 must end the run
// stopped with a warning when the source returns that publication's error.
enum class StopPublicationArm : std::uint8_t {
  kPumpSingleRecipient,
  kPumpFanout,
  // Publication credits (CHAN-166), on by default: the stall ends pre-submit, nothing delivered.
  kFanoutCredits,
  // A blocking fanout family by direct retention (CHAN-167): the capacity check before any store
  // waits for the stalled recipient, and the stop ends that wait with nothing delivered.
  kDirectBlockingFanout,
  // The source requests stop and returns kBackpressured without publishing: no evidence, so the
  // failure keeps its ordinary classification.
  kSyntheticWithoutPublication,
  // The stop-caused failure is genuine, but retiring that activation fails, which still fails the
  // run.
  kRetirementFailure,
  // The stop cancels a publication, which the source handles. It then returns a new
  // kBackpressured, which is not that publication's error.
  kSyntheticAfterHandledCancellation,
  // Reconstruct the cancelled error's exact code and message, without copying its provenance.
  kReconstructedCancelledError,
  // Before requesting stop, the same activation waits out the budget of a publication on the
  // same output. The stop then cancels a retry, and the source returns the expired error.
  kExpiredBeforeCancellationOnSameOutput,
  // As above, but the expired publication is on a second output with its own stalled recipient.
  kExpiredBeforeCancellationOnOtherOutput,
};

struct RuntimeStopPublicationProbe {
  static inline std::array<std::atomic<bool>, 2U> stalled{};
  static inline std::atomic<bool> release{};
  static inline std::atomic<bool> stop_emit_done{};
  static inline std::atomic<ErrorCode> stop_emit_error{ErrorCode::kSuccess};
  static inline std::atomic<bool> stop_emit_marked{};
  static inline std::atomic<std::uint64_t> stop_emit_provenance{};
  static inline std::atomic<std::int64_t> stop_emit_ns{};
  static inline std::atomic<StopPublicationArm> arm{StopPublicationArm::kPumpSingleRecipient};
  static inline std::atomic<std::uint64_t> successful_calls{};

  static void reset(const StopPublicationArm selected) {
    for (auto& lane : stalled) {
      lane.store(false);
    }
    release.store(false);
    stop_emit_done.store(false);
    stop_emit_error.store(ErrorCode::kSuccess);
    stop_emit_marked.store(false);
    stop_emit_provenance.store(0U);
    stop_emit_ns.store(0);
    arm.store(selected);
    successful_calls.store(0U);
  }
};

[[nodiscard]] bool publication_pressure(const ErrorCode code) noexcept {
  return code == ErrorCode::kBackpressured || code == ErrorCode::kPublicationPartial ||
         code == ErrorCode::kPublicationIndeterminate;
}

template <typename Payload>
class RuntimeStopPublicationSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output");
    spec.output(side, "side");
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms}).complete_after(ComputeCalls{.limit = 10'000U});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto result = step(context);
    if (result) {
      RuntimeStopPublicationProbe::successful_calls.fetch_add(1U);
    }
    return result;
  }

  Output<Payload> output;
  // Connected only for kExpiredBeforeCancellationOnOtherOutput.
  Output<Payload> side;

 private:
  using Arm = StopPublicationArm;

  [[nodiscard]] static expected<void, Error> publish(Output<Payload>& port,
                                                     const std::uint64_t value) {
    if constexpr (std::is_same_v<Payload, schema::TimeT>) {
      schema::TimeT payload;
      payload.time_ns = static_cast<std::int64_t>(value);
      return port.emit_ref(payload);
    } else {
      return port.emit(value);
    }
  }

  // Publish once into a recipient that may already be full. Returns whether it reported pressure.
  [[nodiscard]] static expected<bool, Error> fill(Output<Payload>& port,
                                                  const std::uint64_t value) {
    auto published = publish(port, value);
    if (published) {
      return false;
    }
    if (!publication_pressure(published.error().code)) {
      return make_unexpected(std::move(published).error());
    }
    // Pre-stop pressure: drop this sample, as a lossy source would. The recipient is full.
    return true;
  }

  [[nodiscard]] expected<void, Error> step(ExecutionContext& context) {
    using Probe = RuntimeStopPublicationProbe;
    const Arm arm = Probe::arm.load();
    const bool side_used = arm == Arm::kExpiredBeforeCancellationOnOtherOutput;
    switch (phase_) {
      case Phase::kFirst:
        phase_ = Phase::kWaitForStall;
        if (side_used) {
          HOLOSCAN_TRY(publish(side, 1U));
        }
        return publish(output, 1U);
      case Phase::kWaitForStall:
        if (!Probe::stalled[0].load() || (side_used && !Probe::stalled[1].load())) {
          return {};
        }
        phase_ = Phase::kFill;
        [[fallthrough]];
      case Phase::kFill: {
        if (!output_full_) {
          HOLOSCAN_TRY(output_full_, fill(output, ++value_));
        }
        if (side_used && !side_full_) {
          HOLOSCAN_TRY(side_full_, fill(side, ++value_));
        }
        if (output_full_ && (!side_used || side_full_)) {
          phase_ = Phase::kStop;
        }
        return {};
      }
      case Phase::kStop:
        phase_ = Phase::kDone;
        return stop_step(context, arm);
      case Phase::kDone:
        return {};
    }
    return {};
  }

  [[nodiscard]] expected<void, Error> stop_step(ExecutionContext& context, const Arm arm) {
    using Probe = RuntimeStopPublicationProbe;
    // A genuine failure from this activation before the stop: the full recipient outlasts the
    // whole budget.
    std::optional<Error> expired;
    if (arm == Arm::kExpiredBeforeCancellationOnSameOutput ||
        arm == Arm::kExpiredBeforeCancellationOnOtherOutput) {
      auto& port = arm == Arm::kExpiredBeforeCancellationOnSameOutput ? output : side;
      auto published = publish(port, ++value_);
      if (published) {
        return make_unexpected(Error{ErrorCode::kFailure, "full recipient accepted a sample"});
      }
      expired = std::move(published).error();
    }
    context.request_stop();
    if (arm == Arm::kSyntheticWithoutPublication) {
      Probe::stop_emit_done.store(true);
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    const auto started = std::chrono::steady_clock::now();
    auto published = publish(output, ++value_);
    Probe::stop_emit_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now() - started)
                                  .count());
    Probe::stop_emit_error.store(published ? ErrorCode::kSuccess : published.error().code);
    Probe::stop_emit_marked.store(
        !published && published.error().message.starts_with("RUNTIME_PUBLICATION_STOPPED"));
    Probe::stop_emit_provenance.store(published ? 0U : published.error().provenance);
    Probe::stop_emit_done.store(true);
    if (expired) {
      return make_unexpected(std::move(*expired));
    }
    if (arm == Arm::kSyntheticAfterHandledCancellation) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    if (arm == Arm::kReconstructedCancelledError && !published) {
      return make_unexpected(Error{published.error().code, published.error().message});
    }
    // Propagate the failure, as HOLOSCAN_TRY would.
    return published;
  }

  enum class Phase : std::uint8_t { kFirst, kWaitForStall, kFill, kStop, kDone };
  Phase phase_{Phase::kFirst};
  std::uint64_t value_{1U};
  bool output_full_{};
  bool side_full_{};
};

template <typename Payload, bool kStall, std::size_t kLane = 0U>
class RuntimeStopPublicationSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    using Probe = RuntimeStopPublicationProbe;
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample.error()));
    }
    if constexpr (kStall) {
      if (!Probe::stalled[kLane].exchange(true)) {
        // Hold sample 1 until the source's stop-time publication has returned.
        const auto deadline = std::chrono::steady_clock::now() + 10s;
        while (!Probe::release.load() && std::chrono::steady_clock::now() < deadline) {
          std::this_thread::sleep_for(100us);
        }
      }
    }
    return {};
  }

  Input<Payload> input;
};

class RuntimeStopPublication : public ::testing::TestWithParam<StopPublicationArm> {};
class SchemaEmitRefStopPublication : public ::testing::TestWithParam<StopPublicationArm> {};

// REQ: RUNTIME-076
// REQ: RUNTIME-077
// REQ: CHAN-158
// REQ: CHAN-166
// REQ: CHAN-167
template <typename Payload>
void check_stop_publication(const StopPublicationArm arm) {
  using Probe = RuntimeStopPublicationProbe;
  Probe::reset(arm);
  Graph graph{"runtime-stop-publication"};
  auto source = graph.op<RuntimeStopPublicationSource<Payload>>("source");
  auto stalled = graph.op<RuntimeStopPublicationSink<Payload, true>>("stalled");
  graph.add_flow(source->output, stalled->input, ConnectionOptions{.queue_depth = 1U});
  graph.partition("source-partition").add(source);
  graph.partition("stalled-partition").add(stalled);
  // A fast second recipient makes the output a fanout family, which the HoloMQ arms keep on HoloMQ
  // and the direct arm stores directly; a lone recipient would be direct.
  if (arm == StopPublicationArm::kPumpFanout || arm == StopPublicationArm::kFanoutCredits ||
      arm == StopPublicationArm::kDirectBlockingFanout ||
      arm == StopPublicationArm::kSyntheticWithoutPublication) {
    auto fast = graph.op<RuntimeStopPublicationSink<Payload, false>>("fast");
    graph.add_flow(source->output, fast->input, ConnectionOptions{.queue_depth = 1U});
    graph.partition("fast-partition").add(fast);
  }
  if (arm == StopPublicationArm::kExpiredBeforeCancellationOnOtherOutput) {
    auto side = graph.op<RuntimeStopPublicationSink<Payload, true, 1U>>("side");
    graph.add_flow(source->side, side->input, ConnectionOptions{.queue_depth = 1U});
    graph.partition("side-partition").add(side);
  }
  const ClockKey clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(clock);
  const bool credits = arm == StopPublicationArm::kFanoutCredits;
  const bool direct_fanout = arm == StopPublicationArm::kDirectBlockingFanout;
  // compile() delivers a blocking fanout family directly (CHAN-167); the HoloMQ arms keep HoloMQ.
  const bool holomq_fanout = arm == StopPublicationArm::kPumpFanout || credits ||
                             arm == StopPublicationArm::kSyntheticWithoutPublication;
  const ExecutionPlan compiled =
      holomq_fanout ? detail::compile_with_process_local_holomq_fanout(graph, CompileOptions{})
                    : compile(graph);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  // The source's stop-time activation is the run's only failed callback, so this fault fails
  // exactly its retirement.
  ExecutionPlan plan =
      arm == StopPublicationArm::kRetirementFailure
          ? detail::RuntimeTestPlanAccess::fail_next_channel_activation_retirement_for_test(
                compiled, true)
          : compiled;
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  if (arm == StopPublicationArm::kPumpFanout) {
    // Without credits, HoloMQ delivers to the fast recipient first, so the stopped publication is
    // indeterminate rather than rejected.
    auto control = std::make_shared<detail::RuntimeFanoutTestControl>();
    control->disable_publication_credits = true;
    plan = detail::RuntimeTestPlanAccess::with_fanout_test_control(plan, std::move(control));
  }

  RunSession session = run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (!Probe::stop_emit_done.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  EXPECT_TRUE(Probe::stop_emit_done.load()) << "the source never reached its stop publication";
  Probe::release.store(true);
  if (!session.wait_for(10s)) {
    ADD_FAILURE() << "session did not stop";
    session.request_stop();
  }
  session.wait();

  // The failed stop-time call never consumes a completion unit, whatever the outcome.
  ASSERT_EQ(session.operator_completion_progress().size(), 1U);
  EXPECT_EQ(session.operator_completion_progress().front().successful_compute_calls,
            Probe::successful_calls.load());

  const auto& diagnostics = session.diagnostics();
  const auto has_token = [&](const std::string_view token) {
    return std::ranges::any_of(
        diagnostics, [&](const Diagnostic& diagnostic) { return diagnostic.token == token; });
  };
  if (arm == StopPublicationArm::kSyntheticWithoutPublication) {
    // Key assertion: a pressure code without a stop-caused publication failure still fails.
    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    ASSERT_FALSE(diagnostics.empty());
    EXPECT_EQ(diagnostics.front().token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
    return;
  }
  if (arm == StopPublicationArm::kSyntheticAfterHandledCancellation ||
      arm == StopPublicationArm::kReconstructedCancelledError ||
      arm == StopPublicationArm::kExpiredBeforeCancellationOnSameOutput ||
      arm == StopPublicationArm::kExpiredBeforeCancellationOnOtherOutput) {
    // The stop did cancel a publication of the failed activation...
    EXPECT_EQ(Probe::stop_emit_error.load(), ErrorCode::kBackpressured);
    EXPECT_TRUE(Probe::stop_emit_marked.load());
    EXPECT_NE(Probe::stop_emit_provenance.load(), 0U);
    // Key assertion: ...but the callback returned another error, so the run still fails.
    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    ASSERT_FALSE(diagnostics.empty());
    EXPECT_EQ(diagnostics.front().token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
    EXPECT_FALSE(has_token("RUNTIME_OPERATOR_PUBLICATION_STOPPED"));
    return;
  }
  if (arm == StopPublicationArm::kRetirementFailure) {
    // Key assertion: the activation's failed retirement wins over its stop-caused callback error.
    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_TRUE(has_token("RUNTIME_LIFECYCLE_CALLBACK_FAILED"));
    EXPECT_TRUE(has_token("RUNTIME_OPERATOR_PUBLICATION_STOPPED"));
    EXPECT_FALSE(has_token("RUNTIME_OPERATOR_CALLBACK_FAILED"));
    EXPECT_EQ(Probe::stop_emit_error.load(), ErrorCode::kBackpressured);
    EXPECT_TRUE(Probe::stop_emit_marked.load());
    EXPECT_NE(Probe::stop_emit_provenance.load(), 0U);
    return;
  }
  // Key assertion (S2): the stop-caused failure ends the run stopped, with one warning.
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  ASSERT_EQ(diagnostics.size(), 1U) << (diagnostics.empty() ? "" : diagnostics.front().token);
  EXPECT_EQ(diagnostics.front().token, "RUNTIME_OPERATOR_PUBLICATION_STOPPED");
  EXPECT_EQ(diagnostics.front().severity, DiagnosticSeverity::kWarning);
  const ErrorCode expected_error = arm == StopPublicationArm::kPumpFanout
                                       ? ErrorCode::kPublicationIndeterminate
                                       : ErrorCode::kBackpressured;
  EXPECT_EQ(Probe::stop_emit_error.load(), expected_error);
  EXPECT_TRUE(Probe::stop_emit_marked.load());
  EXPECT_NE(Probe::stop_emit_provenance.load(), 0U);
  // Key assertion (S3): the blocked publication did not wait out the graph blocking budget.
  constexpr std::int64_t kGraphBlockingBudgetNs = 5'000'000;
  EXPECT_LT(Probe::stop_emit_ns.load(), kGraphBlockingBudgetNs);
  if (credits || direct_fanout) {
    EXPECT_GE(counters->snapshot()[detail::RuntimePhaseCounters::index(
                  detail::RuntimePhaseCounter::kChannelPublicationStopAborts)],
              1U);
  }
}

TEST_P(RuntimeStopPublication, StopCausedPublicationFailureEndsTheRunStopped) {
  check_stop_publication<std::uint64_t>(GetParam());
}

// REQ: SCHEMA-010
// REQ: RUNTIME-076
// REQ: RUNTIME-077
TEST_P(SchemaEmitRefStopPublication, StopCausedPublicationFailureEndsTheRunStopped) {
  check_stop_publication<schema::TimeT>(GetParam());
}

const auto kStopPublicationArms = ::testing::Values(
    StopPublicationArm::kPumpSingleRecipient, StopPublicationArm::kPumpFanout,
    StopPublicationArm::kFanoutCredits, StopPublicationArm::kDirectBlockingFanout,
    StopPublicationArm::kSyntheticWithoutPublication, StopPublicationArm::kRetirementFailure,
    StopPublicationArm::kSyntheticAfterHandledCancellation,
    StopPublicationArm::kReconstructedCancelledError,
    StopPublicationArm::kExpiredBeforeCancellationOnSameOutput,
    StopPublicationArm::kExpiredBeforeCancellationOnOtherOutput);

const auto kStopPublicationArmName = [](const ::testing::TestParamInfo<StopPublicationArm>& info) {
  switch (info.param) {
    case StopPublicationArm::kPumpSingleRecipient:
      return std::string{"PumpSingleRecipient"};
    case StopPublicationArm::kPumpFanout:
      return std::string{"PumpFanout"};
    case StopPublicationArm::kFanoutCredits:
      return std::string{"FanoutCredits"};
    case StopPublicationArm::kDirectBlockingFanout:
      return std::string{"DirectBlockingFanout"};
    case StopPublicationArm::kSyntheticWithoutPublication:
      return std::string{"SyntheticWithoutPublication"};
    case StopPublicationArm::kRetirementFailure:
      return std::string{"RetirementFailure"};
    case StopPublicationArm::kSyntheticAfterHandledCancellation:
      return std::string{"SyntheticAfterHandledCancellation"};
    case StopPublicationArm::kReconstructedCancelledError:
      return std::string{"ReconstructedCancelledError"};
    case StopPublicationArm::kExpiredBeforeCancellationOnSameOutput:
      return std::string{"ExpiredBeforeCancellationOnSameOutput"};
    case StopPublicationArm::kExpiredBeforeCancellationOnOtherOutput:
      return std::string{"ExpiredBeforeCancellationOnOtherOutput"};
  }
  return std::string{"Unknown"};
};

INSTANTIATE_TEST_SUITE_P(Arms, RuntimeStopPublication, kStopPublicationArms,
                         kStopPublicationArmName);
INSTANTIATE_TEST_SUITE_P(Arms, SchemaEmitRefStopPublication, kStopPublicationArms,
                         kStopPublicationArmName);

}  // namespace
}  // namespace holoscan
