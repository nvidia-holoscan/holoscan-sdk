/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "runtime/graph_exec.hpp"
#include "runtime/runtime_test_access.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

struct ReusableClockProbe {
  static void reset() {
    const std::lock_guard lock(mutex);
    values.clear();
  }

  static void record(std::uint32_t value) {
    const std::lock_guard lock(mutex);
    values.push_back(value);
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_value() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return !values.empty(); });
  }

  [[nodiscard]] static bool wait_for_values(std::size_t count) {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [count] { return values.size() >= count; });
  }

  [[nodiscard]] static std::vector<std::uint32_t> snapshot() {
    const std::lock_guard lock(mutex);
    return values;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::vector<std::uint32_t> values;
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
    ReusableClockProbe::record(received->data);
    return {};
  }

  Input<std::uint32_t> input;
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

[[nodiscard]] ExecutionPlan reusable_clock_plan() {
  Graph graph{"reusable-clock-runtime"};
  auto source = graph.op<ReusableClockSource>("source");
  auto child = graph.embed<ReusableClockWrapperGraph>("child");
  auto sink = graph.op<ReusableClockSink>("sink");
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
  const ExecutionPlan plan = reusable_clock_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  for (std::size_t run = 0U; run < 2U; ++run) {
    ReusableClockProbe::reset();
    RunSession session = run_async(plan);
    const bool observed = ReusableClockProbe::wait_for_value();
    session.request_stop();
    session.wait();

    ASSERT_TRUE(observed);
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty());
    const auto values = ReusableClockProbe::snapshot();
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
  auto first_sink = graph.op<ReusableClockSink>("first_sink");
  auto second_sink = graph.op<ReusableClockSink>("second_sink");
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

  ReusableClockProbe::reset();
  RunSession session = run_async(plan);
  const bool observed = ReusableClockProbe::wait_for_values(2U);
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  const auto values = ReusableClockProbe::snapshot();
  ASSERT_GE(values.size(), 2U);
  EXPECT_EQ(values[0], 1U);
  EXPECT_EQ(values[1], 1U);
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
  const ExecutionPlan baseline = reusable_clock_plan();
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

}  // namespace
}  // namespace holoscan
