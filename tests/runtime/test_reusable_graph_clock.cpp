/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "runtime/graph_exec.hpp"
#include "runtime/runtime_rate_state.hpp"
#include "runtime/runtime_test_access.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

struct ReusableClockProbe {
  void reset() {
    const std::lock_guard lock(mutex);
    values.clear();
  }

  void record(std::uint32_t value) {
    const std::lock_guard lock(mutex);
    values.push_back(value);
    cv.notify_all();
  }

  [[nodiscard]] bool wait_for_value() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [this] { return !values.empty(); });
  }

  [[nodiscard]] bool wait_for_value_until(std::chrono::steady_clock::time_point deadline) {
    std::unique_lock lock(mutex);
    return cv.wait_until(lock, deadline, [this] { return !values.empty(); });
  }

  [[nodiscard]] std::vector<std::uint32_t> snapshot() {
    const std::lock_guard lock(mutex);
    return values;
  }

  std::mutex mutex;
  std::condition_variable cv;
  std::vector<std::uint32_t> values;
};

class ReusableClockSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const TimePoint activation = context.activation_time();
    return output.emit(Sample<std::uint32_t>{
        next_++,
        SampleMetadata{
            .capture_timestamp_ns = activation.timestamp_ns,
            .source_clock_id = activation.clock_id,
        },
    });
  }

  Output<std::uint32_t> output;

 private:
  std::uint32_t next_{1U};
};

class ReusableClockRelay final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(2U);
    spec.output(output, "output");
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto received = input.receive();
    if (!received) {
      return make_unexpected(std::move(received.error()));
    }
    return output.emit(std::move(received).value());
  }

  Input<std::uint32_t> input;
  Output<std::uint32_t> output;
};

class ReusableClockSink final : public Operator<> {
 public:
  explicit ReusableClockSink(std::shared_ptr<ReusableClockProbe> probe)
      : probe_(std::move(probe)) {}

  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(2U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto received = input.receive();
    if (!received) {
      return make_unexpected(std::move(received.error()));
    }
    probe_->record(received->data);
    return {};
  }

  Input<std::uint32_t> input;

 private:
  std::shared_ptr<ReusableClockProbe> probe_;
};

class ReusableClockRelayGraph final : public Graph {
 public:
  [[nodiscard]] ClockKey clock_import() const noexcept { return clock_import_; }

  Input<std::uint32_t> input{"input"};
  Output<std::uint32_t> output{"output"};

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    set_default_clock(clock_import_);
    auto relay = op<ReusableClockRelay>("relay");
    export_input(input, relay->input);
    export_output(output, relay->output);
  }

 private:
  ClockKey clock_import_;
};

class ReusableClockWrapperGraph final : public Graph {
 public:
  [[nodiscard]] ClockKey clock_import() const noexcept { return clock_import_; }
  [[nodiscard]] ClockKey unused_clock_import() const noexcept { return unused_clock_import_; }

  Input<std::uint32_t> input{"input"};
  Output<std::uint32_t> output{"output"};

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    unused_clock_import_ = import_clock("unused_timeline");
    set_default_clock(clock_import_);
    auto child = embed<ReusableClockRelayGraph>("relay_graph");
    bind_clock(child, child->clock_import(), clock_import_);
    export_input(input, child->input);
    export_output(output, child->output);
  }

 private:
  ClockKey clock_import_;
  ClockKey unused_clock_import_;
};

class ReusableClockConversionGraph final : public Graph {
 public:
  [[nodiscard]] ClockKey source_import() const noexcept { return source_import_; }
  [[nodiscard]] ClockKey target_import() const noexcept { return target_import_; }

 protected:
  void compose() override {
    source_import_ = import_clock("source");
    target_import_ = import_clock("target");
    set_default_clock(target_import_);
    add_clock_conversion(source_import_, target_import_, AffineClockConversion{.offset_ns = 5});
  }

 private:
  ClockKey source_import_;
  ClockKey target_import_;
};

/// One embedded instance's cold clock-resolution outcome, recorded from its \c kWarmUp body.
struct ColdClockResolution {
  ClockId resolved_default{};  ///< The child's selected default, bound alike in both instances.
  ClockId resolved_scoped{};   ///< The shared authored name bound differently per instance.
  bool default_readable{};
  bool default_matches_named{};
  bool absent_name_rejected{};
};

struct ColdClockResolutionProbe {
  static void reset() {
    const std::lock_guard lock(mutex);
    resolutions.clear();
  }

  static void record(ColdClockResolution resolution) {
    const std::lock_guard lock(mutex);
    resolutions.push_back(resolution);
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_resolutions(std::size_t count) {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [count] { return resolutions.size() >= count; });
  }

  [[nodiscard]] static std::vector<ColdClockResolution> snapshot() {
    const std::lock_guard lock(mutex);
    return resolutions;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::vector<ColdClockResolution> resolutions;
};

/// Resolve this Operator's declared clocks from a cold lifecycle body.
///
/// This is the workflow \c ClockDiscipline prescribes for priming: obtain the graph clock in
/// \c kWarmUp, before any activation, so acquisition never begins with an unsettled timestamp
/// relationship. It requires the Runtime lifecycle clock resolver to be installed.
class ColdClockResolutionProbeOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(2U);
    spec.lifecycle().stage(LifecycleStage::kWarmUp, &ColdClockResolutionProbeOperator::warm_up);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  LifecycleStatus warm_up(LifecycleContext& context) noexcept {
    ColdClockResolution resolution;
    if (auto resolved = context.resolve_clock(ClockKey::default_clock())) {
      resolution.resolved_default = resolved->id();
      // A reference that cannot be read would not let an operator prime anything, so the useful
      // assertion is that the resolved clock is live rather than merely named.
      resolution.default_readable = resolved->now().has_value();
    }
    // The child selected this import as its default, so the omitted key and the authored name must
    // name one clock.
    if (auto resolved = context.resolve_clock(ClockKey::named("timeline"))) {
      resolution.default_matches_named = resolved->id() == resolution.resolved_default;
    }
    if (auto resolved = context.resolve_clock(ClockKey::named("reference"))) {
      resolution.resolved_scoped = resolved->id();
    }
    // A name this Graph scope never declared must fail rather than fall through to an ancestor.
    resolution.absent_name_rejected =
        !context.resolve_clock(ClockKey::named("absent_timeline")).has_value();
    ColdClockResolutionProbe::record(resolution);
    return LifecycleStatus::kOk;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto received = input.receive();
    if (!received) {
      return make_unexpected(std::move(received.error()));
    }
    return {};
  }

  Input<std::uint32_t> input;
};

/// A self-contained pipeline that imports its scheduling clock and one further reference clock.
///
/// Only \c timeline is selected as the default, so it alone reaches this instance's triggers. Every
/// trigger in a plan must share one reference clock, so binding two instances' defaults to
/// different clocks would be rejected with \c TEMPORAL_SCHEDULING_DOMAIN_UNSUPPORTED before the
/// resolution under test ever runs. The unscheduled \c reference import carries the per-instance
/// difference instead, which is what lets one plan hold two scopes that disagree about a name.
class ColdClockResolutionGraph final : public Graph {
 public:
  [[nodiscard]] ClockKey clock_import() const noexcept { return clock_import_; }
  [[nodiscard]] ClockKey scoped_import() const noexcept { return scoped_import_; }

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    scoped_import_ = import_clock("reference");
    set_default_clock(clock_import_);
    auto source = op<ReusableClockSource>("source");
    auto probe = op<ColdClockResolutionProbeOperator>("probe");
    add_flow(source->output, probe->input, ConnectionOptions{.queue_depth = 2U});
  }

 private:
  ClockKey clock_import_;
  ClockKey scoped_import_;
};

[[nodiscard]] ExecutionPlan reusable_clock_plan(const std::shared_ptr<ReusableClockProbe>& probe) {
  Graph graph{"reusable-clock-runtime"};
  auto source = graph.op<ReusableClockSource>("source");
  auto child = graph.embed<ReusableClockWrapperGraph>("child");
  auto sink = graph.op<ReusableClockSink>("sink", probe);
  graph.add_flow(source->output, child->input, ConnectionOptions{.queue_depth = 2U});
  graph.add_flow(child->output, sink->input, ConnectionOptions{.queue_depth = 2U});
  const ClockKey clock = graph.add_clock<RealtimeClock>("runtime_clock");
  const ClockKey alternate = graph.add_clock<RealtimeClock>("alternate_runtime_clock");
  graph.set_default_clock(clock);
  graph.bind_clock(child, child->clock_import(), clock);
  graph.bind_clock(child, child->unused_clock_import(), alternate);
  return compile(graph);
}

// REQ: RUNTIME-027
TEST(ReusableGraphClockTest, ReconstructsAndReplaysExplicitClockImportBindings) {
  const auto probe = std::make_shared<ReusableClockProbe>();
  const ExecutionPlan plan = reusable_clock_plan(probe);
  ASSERT_TRUE(plan.ok()) << plan.json();

  for (std::size_t run = 0U; run < 2U; ++run) {
    probe->reset();
    RunSession session = run_async(plan);
    const bool observed = probe->wait_for_value();
    session.request_stop();
    session.wait();

    ASSERT_TRUE(observed) << plan.json();
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty());
    const auto values = probe->snapshot();
    ASSERT_FALSE(values.empty());
    EXPECT_EQ(values.front(), 1U);
  }
}

// REQ: LIFECYCLE-003
// REQ: RUNTIME-002
// Two independently embedded copies intentionally retain the same local Operator name. Runtime
// lifecycle reconstruction must identify them by their frozen qualified paths, not by that leaf.
TEST(ReusableGraphClockTest, ReconstructsRepeatedReusableOperatorLocalNames) {
  Graph graph{"repeated-reusable-local-names"};
  auto source = graph.op<ReusableClockSource>("source");
  auto first = graph.embed<ReusableClockWrapperGraph>("first");
  auto second = graph.embed<ReusableClockWrapperGraph>("second");
  const auto first_probe = std::make_shared<ReusableClockProbe>();
  const auto second_probe = std::make_shared<ReusableClockProbe>();
  auto first_sink = graph.op<ReusableClockSink>("first_sink", first_probe);
  auto second_sink = graph.op<ReusableClockSink>("second_sink", second_probe);
  graph.add_flow(source->output, first->input, ConnectionOptions{.queue_depth = 2U});
  graph.add_flow(source->output, second->input, ConnectionOptions{.queue_depth = 2U});
  graph.add_flow(first->output, first_sink->input, ConnectionOptions{.queue_depth = 2U});
  graph.add_flow(second->output, second_sink->input, ConnectionOptions{.queue_depth = 2U});

  const ClockKey clock = graph.add_clock<RealtimeClock>("runtime_clock");
  const ClockKey alternate = graph.add_clock<RealtimeClock>("alternate_runtime_clock");
  graph.set_default_clock(clock);
  graph.bind_clock(first, first->clock_import(), clock);
  graph.bind_clock(first, first->unused_clock_import(), alternate);
  graph.bind_clock(second, second->clock_import(), clock);
  graph.bind_clock(second, second->unused_clock_import(), alternate);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_FALSE(plan.interface_port_bindings().empty());

  // Reusable-Graph interface vertices exist only in the immutable plan. Their presence makes the
  // canonical VertexId sequence sparse relative to reconstructed OperatorInstanceId values, so
  // Runtime rate-state construction must bind operator ports through the replayed instances.
  auto reconstructed = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                                      RunGenerationId{31U});
  ASSERT_TRUE(reconstructed) << reconstructed.error().message;
  auto rates = detail::RuntimeRateState::create(
      (*reconstructed)->plan(), (*reconstructed)->operators(), RunGenerationId{31U}, 1U);
  ASSERT_TRUE(rates) << rates.error().message;

  // A process-group worker replays only its local Operator subset against the same complete plan.
  // Remote Operator ports and reusable-Graph interface aliases remain plan-visible but have no
  // process-local declaration or owner binding in this RuntimeRateState.
  ASSERT_GT((*reconstructed)->operators().size(), 1U);
  auto local_rates = detail::RuntimeRateState::create(
      (*reconstructed)->plan(), (*reconstructed)->operators().first(1U), RunGenerationId{32U}, 1U);
  ASSERT_TRUE(local_rates) << local_rates.error().message;

  RunSession session = run_async(plan);
  // Independent branches need not alternate. Preserve one five-second budget while waiting
  // for each sink's first value, so two records from one branch cannot complete this test.
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  const bool first_observed = first_probe->wait_for_value_until(deadline);
  const bool second_observed = second_probe->wait_for_value_until(deadline);
  session.request_stop();
  session.wait();

  EXPECT_TRUE(first_observed) << "first_sink did not receive a value: " << plan.json();
  EXPECT_TRUE(second_observed) << "second_sink did not receive a value: " << plan.json();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  for (const auto& probe : std::array{first_probe, second_probe}) {
    SCOPED_TRACE(probe == first_probe ? "first_sink" : "second_sink");
    const auto values = probe->snapshot();
    EXPECT_FALSE(values.empty());
    if (!values.empty()) {
      EXPECT_EQ(values.front(), 1U);
    }
  }
}

// REQ: RUNTIME-027
TEST(ReusableGraphClockTest, ReplaysCanonicalEquivalentClockConversionsAtColdStart) {
  Graph graph{"canonical-equivalent-clock-conversions"};
  auto first = graph.embed<ReusableClockConversionGraph>("first");
  auto second = graph.embed<ReusableClockConversionGraph>("second");
  const ClockKey source = graph.add_clock<RealtimeClock>("source");
  const ClockKey target = graph.add_clock<RealtimeClock>("target");
  graph.set_default_clock(target);
  graph.bind_clock(first, first->source_import(), source);
  graph.bind_clock(first, first->target_import(), target);
  graph.bind_clock(second, second->source_import(), source);
  graph.bind_clock(second, second->target_import(), target);

  // Key setup: two authored conversions resolve to one canonical parent-clock pair.
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.clock_conversions().size(), 1U);

  // Key assertion: Runtime replays that canonical projection instead of requiring positional
  // parity with both authored declarations.
  RunSession session = run_async(plan);
  session.wait();
  EXPECT_NE(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(session.diagnostics().empty());
}

// REQ: RUNTIME-027
TEST(ReusableGraphClockTest, RejectsMissingChangedAndMalformedFrozenClockImportChains) {
  using Mutation = detail::RuntimeTestPlanAccess::ClockImportMutation;
  constexpr std::array mutations{
      Mutation::kMissingImport,
      Mutation::kMissingBinding,
      Mutation::kMissingImportAndBinding,
      Mutation::kChangedParentClock,
      Mutation::kNonImmediateParent,
      Mutation::kFrozenResolvedClock,
      Mutation::kDuplicateImportedDefault,
  };
  const ExecutionPlan baseline = reusable_clock_plan(std::make_shared<ReusableClockProbe>());
  ASSERT_TRUE(baseline.ok()) << baseline.json();

  for (const Mutation mutation : mutations) {
    SCOPED_TRACE(static_cast<unsigned>(mutation));
    const ExecutionPlan malformed =
        detail::RuntimeTestPlanAccess::mutate_clock_import(baseline, mutation);
    ASSERT_TRUE(malformed.ok()) << malformed.json();
    const auto accepted = detail::ExecutionPlanAccess::retain(malformed);
    ASSERT_TRUE(accepted);
    const auto reconstructed = detail::GraphExec::reconstruct(accepted, RunGenerationId{23U});
    ASSERT_FALSE(reconstructed);
    EXPECT_EQ(reconstructed.error().code, ErrorCode::kFailure);
    EXPECT_EQ(reconstructed.error().message, "RUNTIME_CLOCK_INVENTORY_MISMATCH");
  }
}

// A ClockKey is authored per Graph, so one name means different clocks in independently embedded
// instances. Both children below import "reference" and bind it to a different parent clock, which
// is the case that separates graph-scoped resolution from a flat lookup: a global by-name lookup
// would hand both instances the same clock, and passing the authored name to the live registry
// would find nothing at all, because runtime reconstruction registers private id-derived keys.
// REQ: LIFECYCLE-005
// REQ: RUNTIME-027
TEST(ReusableGraphClockTest, ResolvesGraphScopedClockKeysDuringColdLifecycle) {
  Graph graph{"cold-clock-resolution"};
  auto first = graph.embed<ColdClockResolutionGraph>("first");
  auto second = graph.embed<ColdClockResolutionGraph>("second");

  const ClockKey primary = graph.add_clock<RealtimeClock>("runtime_clock");
  const ClockKey alternate = graph.add_clock<RealtimeClock>("alternate_runtime_clock");
  graph.set_default_clock(primary);
  // Both scheduling imports bind alike, which keeps the plan in one scheduling domain.
  graph.bind_clock(first, first->clock_import(), primary);
  graph.bind_clock(second, second->clock_import(), primary);
  // Key setup: one shared authored name, two different bound parent clocks.
  graph.bind_clock(first, first->scoped_import(), primary);
  graph.bind_clock(second, second->scoped_import(), alternate);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();

  ColdClockResolutionProbe::reset();
  RunSession session = run_async(plan);
  const bool observed = ColdClockResolutionProbe::wait_for_resolutions(2U);
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed) << plan.json();
  EXPECT_NE(session.termination(), RunTermination::kFailed);
  EXPECT_TRUE(session.diagnostics().empty());

  const auto resolutions = ColdClockResolutionProbe::snapshot();
  ASSERT_EQ(resolutions.size(), 2U);
  for (const ColdClockResolution& resolution : resolutions) {
    // Key assertion: cold resolution succeeds at all, which is what the null resolver denied.
    EXPECT_TRUE(resolution.resolved_default.valid());
    EXPECT_TRUE(resolution.default_readable);
    EXPECT_TRUE(resolution.default_matches_named);
    EXPECT_TRUE(resolution.resolved_scoped.valid());
    EXPECT_TRUE(resolution.absent_name_rejected);
  }
  // Both children selected an import bound to the same parent clock, so their defaults agree.
  EXPECT_EQ(resolutions[0].resolved_default.value, resolutions[1].resolved_default.value);
  // Key assertion: the shared authored name resolved per Graph scope rather than globally.
  EXPECT_NE(resolutions[0].resolved_scoped.value, resolutions[1].resolved_scoped.value);
}

}  // namespace
}  // namespace holoscan
