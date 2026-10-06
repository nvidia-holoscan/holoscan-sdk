/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

struct ScaleParams {
  std::int32_t factor{1};
};

HOLOSCAN_PARAM_SCHEMA(ScaleParams,
                      holoscan::param_schema("com.nvidia.holoscan.test.ScaleParams", 1),
                      holoscan::field<&ScaleParams::factor>("factor"));

class ScaleOp final : public holoscan::Operator<ScaleParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in");
    spec.output(out, "out");
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto value = in.receive();
    if (!value) {
      return holoscan::make_unexpected(std::move(value).error());
    }
    value->data *= params().factor;
    return out.emit(std::move(*value));
  }

  holoscan::Input<std::int32_t> in;
  holoscan::Output<std::int32_t> out;
};

// REQ: OPERATOR-064
// REQ: OPERATOR-058
template <typename Fixture>
concept HasUnambiguousRunOptions = requires(Fixture& h) {
  h.run();
  h.run({});
  h.run({.timeout = std::chrono::seconds{5}});
  h.run_controlled_clock(holoscan::testing::ControlledClockRun{});
};
static_assert(HasUnambiguousRunOptions<holoscan::testing::NativeOperatorTest>);
static_assert(HasUnambiguousRunOptions<holoscan::testing::OperatorTestHarness<ScaleOp>>);

template <typename Value>
concept CanStageScalarValue =
    requires(holoscan::testing::OperatorTestHarness<ScaleOp>& h, Value value) {
      h.push(&ScaleOp::in, value);
      h.push("in", value);
    };
static_assert(CanStageScalarValue<std::int32_t>);
static_assert(CanStageScalarValue<std::int16_t>);
static_assert(!CanStageScalarValue<std::vector<std::int32_t>>);
static_assert(!holoscan::PublishedPayload<std::vector<std::int32_t>>);

class AlignedSumOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(left, "left");
    spec.input(right, "right");
    spec.output(sum, "sum");
  }
  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result
        .trigger(holoscan::OnAligned{.ports = holoscan::ports(left, right),
                                     .window = std::chrono::milliseconds{100}})
        .input(left, holoscan::Exact{})
        .input(right, holoscan::Exact{});
    return result;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto a = left.receive();
    auto b = right.receive();
    if (!a)
      return holoscan::make_unexpected(std::move(a).error());
    if (!b)
      return holoscan::make_unexpected(std::move(b).error());
    return sum.emit(a->data + b->data);
  }
  holoscan::Input<std::int32_t> left;
  holoscan::Input<std::int32_t> right;
  holoscan::Output<std::int32_t> sum;
};

class ControlledPairOp final : public holoscan::Operator<ScaleParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(control, "control").optional();
    spec.input(left, "left");
    spec.input(right, "right");
    spec.output(out, "out");
  }
  holoscan::Contract contract() const override {
    holoscan::Contract c;
    c.trigger(holoscan::OnAligned{.ports = holoscan::ports(left, right),
                                  .window = std::chrono::nanoseconds{0}})
        .input(left, holoscan::Exact{})
        .input(right, holoscan::Exact{})
        .input(control, holoscan::Exact{})
        .on_missing(control, holoscan::SafeDefault{});
    return c;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto a = left.receive();
    auto b = right.receive();
    if (!a)
      return holoscan::make_unexpected(std::move(a).error());
    if (!b)
      return holoscan::make_unexpected(std::move(b).error());
    auto factor = params().factor;
    if (control.selection().state != holoscan::InputState::kSafeDefault) {
      auto value = control.receive();
      if (!value)
        return holoscan::make_unexpected(std::move(value).error());
      factor = value->data;
    }
    return out.emit((a->data + b->data) * factor);
  }
  holoscan::Input<std::int32_t> control;
  holoscan::Input<std::int32_t> left;
  holoscan::Input<std::int32_t> right;
  holoscan::Output<std::int32_t> out;
};

class FreshPairOp final : public holoscan::Operator<> {
 public:
  explicit FreshPairOp(bool emit_output = true, bool loaned_output = false,
                       std::uint32_t output_count = 1U, bool finite_completion = false)
      : emit_output_(emit_output),
        loaned_output_(loaned_output),
        output_count_(output_count),
        finite_completion_(finite_completion) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(left, "left");
    spec.input(right, "right");
    spec.output(out, "out").max_emits_per_compute(output_count_);
  }
  holoscan::Contract contract() const override {
    holoscan::Contract c;
    c.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{10}})
        .input(left, holoscan::LatestWithin{.max_age = std::chrono::milliseconds{100}})
        .input(right, holoscan::LatestWithin{.max_age = std::chrono::milliseconds{100}})
        .on_missing(left, holoscan::Skip{})
        .on_missing(right, holoscan::Skip{});
    if (finite_completion_)
      c.complete_after(holoscan::ComputeCalls{.limit = 1U});
    return c;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto a = left.receive();
    auto b = right.receive();
    if (!a)
      return holoscan::make_unexpected(std::move(a).error());
    if (!b)
      return holoscan::make_unexpected(std::move(b).error());
    if (!emit_output_)
      return {};
    for (std::uint32_t i = 0; i < output_count_; ++i) {
      if (loaned_output_) {
        auto loan = out.allocate();
        if (!loan)
          return holoscan::make_unexpected(std::move(loan).error());
        loan->value() = a->data + b->data;
        auto emitted = out.emit(std::move(*loan));
        if (!emitted)
          return emitted;
      } else {
        auto emitted = out.emit(a->data + b->data);
        if (!emitted)
          return emitted;
      }
    }
    return {};
  }
  holoscan::Input<std::int32_t> left;
  holoscan::Input<std::int32_t> right;
  holoscan::Output<std::int32_t> out;

 private:
  bool emit_output_{};
  bool loaned_output_{};
  std::uint32_t output_count_{};
  bool finite_completion_{};
};

struct LifecycleProbe {
  std::uint32_t starts{};
  std::uint32_t computes{};
  std::uint32_t stops{};
};

class CountingOp final : public holoscan::Operator<> {
 public:
  explicit CountingOp(std::shared_ptr<LifecycleProbe> probe) : probe_(std::move(probe)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in");
    spec.output(out, "out");
  }

  void start() override { ++probe_->starts; }
  void stop() override { ++probe_->stops; }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    ++probe_->computes;
    ++count_;
    return out.emit(count_);
  }

  holoscan::Input<std::int32_t> in;
  holoscan::Output<std::uint32_t> out;

 private:
  std::shared_ptr<LifecycleProbe> probe_;
  std::uint32_t count_{};
};

}  // namespace

// REQ: OPERATOR-058
TEST(OperatorTestHarness, ControlledClockDistinguishesBoundaryStaleAndFuture) {
  using namespace std::chrono_literals;
  for (const auto capture : {899ms, 900ms, 1001ms}) {
    SCOPED_TRACE(capture.count());
    auto h = holoscan::testing::create_operator_test<FreshPairOp>();
    h.push_samples<std::int32_t>("left", {holoscan::testing::sample_at(std::int32_t{1}, capture)});
    h.push_samples<std::int32_t>("right", {holoscan::testing::sample_at(std::int32_t{2}, capture)});
    h.capture<std::int32_t>("out");
    const auto result =
        h.run_controlled_clock(holoscan::testing::ControlledClockRun{.tick = 1000ms});
    ASSERT_TRUE(result.ok()) << result.describe_failure();
    EXPECT_EQ(result.termination(), holoscan::RunTermination::kStopped);
    if (capture == 900ms) {
      EXPECT_EQ(result.clock_outcome(), holoscan::testing::ClockTestOutcome::kRetired);
      EXPECT_EQ(result.value<std::int32_t>("out"), 3);
      EXPECT_EQ(result.clock_selection("left").state, holoscan::InputState::kSelected);
    } else {
      EXPECT_EQ(result.clock_outcome(), holoscan::testing::ClockTestOutcome::kSuppressed);
      EXPECT_TRUE(result.values<std::int32_t>("out").empty());
      EXPECT_EQ(result.clock_selection("left").state, holoscan::InputState::kAbsent);
    }
  }
}

// REQ: OPERATOR-058
TEST(OperatorTestHarness, ControlledClockRetiredNoOutputIsNotSuppression) {
  using namespace std::chrono_literals;
  auto h = holoscan::testing::create_operator_test<FreshPairOp>(false);
  h.push_samples<std::int32_t>("left", {holoscan::testing::sample_at(std::int32_t{1}, 900ms)});
  h.push_samples<std::int32_t>("right", {holoscan::testing::sample_at(std::int32_t{2}, 900ms)});
  h.capture<std::int32_t>("out");
  const auto result = h.run_controlled_clock(holoscan::testing::ControlledClockRun{.tick = 1000ms});
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.clock_outcome(), holoscan::testing::ClockTestOutcome::kRetired);
  EXPECT_TRUE(result.values<std::int32_t>("out").empty());
}

// REQ: OPERATOR-058
TEST(OperatorTestHarness, ControlledClockDrainsEverySampleAndLoanPublication) {
  using namespace std::chrono_literals;
  for (const bool loaned : {false, true}) {
    SCOPED_TRACE(loaned);
    auto h = holoscan::testing::create_operator_test<FreshPairOp>(true, loaned, 2U);
    h.push_samples<std::int32_t>("left", {holoscan::testing::sample_at(std::int32_t{1}, 900ms)});
    h.push_samples<std::int32_t>("right", {holoscan::testing::sample_at(std::int32_t{2}, 900ms)});
    h.capture<std::int32_t>("out");
    const auto result =
        h.run_controlled_clock(holoscan::testing::ControlledClockRun{.tick = 1000ms});
    ASSERT_TRUE(result.ok()) << result.describe_failure();
    EXPECT_EQ(result.clock_outcome(), holoscan::testing::ClockTestOutcome::kRetired);
    EXPECT_EQ(result.values<std::int32_t>("out"), (std::vector<std::int32_t>{3, 3}));
  }
}

// REQ: OPERATOR-058
TEST(OperatorTestHarness, ControlledClockPreservesNativeCompletionPlacementRejection) {
  using namespace std::chrono_literals;
  auto h = holoscan::testing::create_operator_test<FreshPairOp>(false, false, 1U, true);
  h.push_samples<std::int32_t>("left", {holoscan::testing::sample_at(std::int32_t{1}, 900ms)});
  h.push_samples<std::int32_t>("right", {holoscan::testing::sample_at(std::int32_t{2}, 900ms)});
  h.capture<std::int32_t>("out");
  const auto result = h.run_controlled_clock(holoscan::testing::ControlledClockRun{.tick = 1000ms});
  EXPECT_FALSE(result.plan().ok());
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.clock_outcome(), holoscan::testing::ClockTestOutcome::kFailed);
  EXPECT_NE(result.describe_failure().find("OPERATOR_COMPLETION_PLACEMENT_UNSUPPORTED"),
            std::string::npos);
}

// REQ: OPERATOR-058
TEST(OperatorTestHarness, ControlledClockOverflowCannotProveAbsence) {
  using namespace std::chrono_literals;
  auto h = holoscan::testing::create_operator_test<FreshPairOp>();
  h.push_samples<std::int32_t>("left", {holoscan::testing::sample_at(std::int32_t{1}, 899ms)});
  h.push_samples<std::int32_t>("right", {holoscan::testing::sample_at(std::int32_t{2}, 899ms)});
  h.capture<std::int32_t>("out");
  const auto result = h.run_controlled_clock(
      holoscan::testing::ControlledClockRun{.tick = 1000ms, .observation_capacity = 1U});
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.clock_outcome(), holoscan::testing::ClockTestOutcome::kFailed);
  EXPECT_NE(result.describe_failure().find("observation"), std::string::npos);
}

namespace {
class UnrelatedClockOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec&) override {}
  holoscan::Contract contract() const override {
    holoscan::Contract c;
    c.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{10}});
    return c;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }
};
}  // namespace

// REQ: OPERATOR-058
TEST(OperatorTestHarness, ControlledClockRejectsUnrelatedOperatorsAndUnrepresentableTimeout) {
  using namespace std::chrono_literals;
  auto h = holoscan::testing::create_operator_test<FreshPairOp>();
  h.push_samples<std::int32_t>("left", {holoscan::testing::sample_at(std::int32_t{1}, 900ms)});
  h.push_samples<std::int32_t>("right", {holoscan::testing::sample_at(std::int32_t{2}, 900ms)});
  h.capture<std::int32_t>("out");
  h.configure_graph([](auto& view) { view.graph().template op<UnrelatedClockOp>("unrelated"); });
  EXPECT_THROW((void)h.run_controlled_clock(holoscan::testing::ControlledClockRun{.tick = 1000ms}),
               std::invalid_argument);
  h.configure_graph(nullptr);
  EXPECT_THROW((void)h.run_controlled_clock(holoscan::testing::ControlledClockRun{
                   .tick = 1000ms, .timeout = std::chrono::milliseconds::max()}),
               std::invalid_argument);
  const auto result = h.run_controlled_clock(holoscan::testing::ControlledClockRun{.tick = 1000ms});
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<std::int32_t>("out"), 3);
}

// REQ: OPERATOR-058
TEST(OperatorTestHarness, ControlledClockColdRejectionPreservesStaging) {
  using namespace std::chrono_literals;
  auto h = holoscan::testing::create_operator_test<FreshPairOp>();
  h.push_samples<std::int32_t>("left", {holoscan::testing::sample_at(std::int32_t{1}, 900ms)});
  h.push_samples<std::int32_t>("right", {holoscan::testing::sample_at(std::int32_t{2}, 900ms)});
  h.capture<std::int32_t>("out");
  EXPECT_THROW((void)h.run_controlled_clock(holoscan::testing::ControlledClockRun{.tick = 1001ms}),
               std::invalid_argument);
  h.configure_graph([](auto& view) {
    view.graph().set_default_clock(
        view.graph().template add_clock<holoscan::RealtimeClock>("other"));
  });
  EXPECT_THROW((void)h.run_controlled_clock(holoscan::testing::ControlledClockRun{.tick = 1000ms}),
               std::invalid_argument);
  h.configure_graph(nullptr);
  const auto result = h.run_controlled_clock(holoscan::testing::ControlledClockRun{.tick = 1000ms});
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<std::int32_t>("out"), 3);
  EXPECT_THROW((void)h.run_controlled_clock(holoscan::testing::ControlledClockRun{.tick = 1000ms}),
               std::invalid_argument);
}

// REQ: OPERATOR-058
TEST(OperatorTestHarness, ControlledClockCompileRejectionRemainsARequestedFailedObservation) {
  using namespace std::chrono_literals;
  auto h = holoscan::testing::create_operator_test<FreshPairOp>();
  h.push_samples<std::int32_t>("left", {holoscan::testing::sample_at(std::int32_t{1}, 900ms)});
  h.push_samples<std::int32_t>("right", {holoscan::testing::sample_at(std::int32_t{2}, 900ms)});
  h.capture<std::int32_t>("out");
  h.configure_graph(
      [](auto& view) { view.graph().partition("incomplete").add(view.tested_operator()); });
  const auto rejected =
      h.run_controlled_clock(holoscan::testing::ControlledClockRun{.tick = 1000ms});
  EXPECT_FALSE(rejected.ok());
  EXPECT_FALSE(rejected.plan().ok());
  EXPECT_EQ(rejected.clock_outcome(), holoscan::testing::ClockTestOutcome::kFailed);
  h.configure_graph(nullptr);
  const auto result = h.run_controlled_clock(holoscan::testing::ControlledClockRun{.tick = 1000ms});
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<std::int32_t>("out"), 3);
}

// REQ: OPERATOR-058
TEST(OperatorTestHarness, ControlledClockRejectsEmptyGroupedInputsBeforeGraphConfiguration) {
  using namespace std::chrono_literals;
  auto h = holoscan::testing::create_operator_test<UnrelatedClockOp>();
  h.push_frame(holoscan::testing::frame_at(900ms));
  bool hook_called = false;
  h.configure_graph([&](auto&) { hook_called = true; });
  EXPECT_THROW((void)h.run_controlled_clock(holoscan::testing::ControlledClockRun{.tick = 1000ms}),
               std::invalid_argument);
  EXPECT_FALSE(hook_called);
}

// REQ: OPERATOR-058
TEST(OperatorTestHarness, ControlledClockSupportsOneGroupedFrameAndRejectsMultiple) {
  using namespace std::chrono_literals;
  using holoscan::testing::frame_at;
  using holoscan::testing::input;
  auto h = holoscan::testing::create_operator_test<FreshPairOp>();
  h.push_frame(frame_at(900ms, input("left", std::int32_t{1}), input("right", std::int32_t{2})));
  h.capture<std::int32_t>("out");
  const auto result = h.run_controlled_clock(holoscan::testing::ControlledClockRun{.tick = 1000ms});
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<std::int32_t>("out"), 3);
  h.push_frame(frame_at(900ms, input("left", std::int32_t{1}), input("right", std::int32_t{2})));
  h.push_frame(frame_at(950ms, input("left", std::int32_t{3}), input("right", std::int32_t{4})));
  EXPECT_THROW((void)h.run_controlled_clock(holoscan::testing::ControlledClockRun{.tick = 1000ms}),
               std::invalid_argument);
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, TypedMemberPushUsesConfiguredParams) {
  auto h = holoscan::testing::create_operator_test<ScaleOp>();
  h.set_params(ScaleParams{.factor = 2});
  h.push(&ScaleOp::in, std::int32_t{21});
  h.capture<std::int32_t>("out");

  const auto result = h.run();

  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.termination(), holoscan::RunTermination::kCompleted);
  EXPECT_EQ(result.value<std::int32_t>("out"), 42);
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, NamedPushMatchesTypedMemberPush) {
  auto h = holoscan::testing::create_operator_test<ScaleOp>();
  h.set_params(ScaleParams{.factor = 2});
  h.push("in", std::int32_t{21});
  h.capture<std::int32_t>("out");

  const auto result = h.run();

  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<std::int32_t>("out"), 42);
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, FiniteSequenceDrainsAndKeepsOwningResultsAcrossRuns) {
  auto h = holoscan::testing::create_operator_test<ScaleOp>();
  h.set_params(ScaleParams{.factor = 3});
  h.push_many("in", std::vector<std::int32_t>{1, 2, 3});
  h.capture<std::int32_t>("out");
  const auto first = h.run();
  ASSERT_TRUE(first.ok()) << first.describe_failure();

  h.set_params(ScaleParams{.factor = 4});
  h.push("in", std::int32_t{5});
  const auto second = h.run();

  ASSERT_TRUE(second.ok()) << second.describe_failure();
  EXPECT_EQ(first.values<std::int32_t>("out"), (std::vector<std::int32_t>{3, 6, 9}));
  EXPECT_EQ(second.value<std::int32_t>("out"), 20);
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, ExplicitEmptyInputCompletesWithoutCallingOperator) {
  const auto probe = std::make_shared<LifecycleProbe>();
  auto h = holoscan::testing::create_operator_test<CountingOp>(probe);
  h.push_many("in", std::vector<std::int32_t>{});
  h.capture<std::uint32_t>("out");

  const auto result = h.run();

  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_TRUE(result.values<std::uint32_t>("out").empty());
  EXPECT_EQ(probe->starts, 1U);
  EXPECT_EQ(probe->computes, 0U);
  EXPECT_EQ(probe->stops, 1U);
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, RuntimeOwnsLifecycleAndFreshInstances) {
  const auto probe = std::make_shared<LifecycleProbe>();
  auto h = holoscan::testing::create_operator_test<CountingOp>(probe);
  h.capture<std::uint32_t>("out");
  h.push_many("in", std::vector<std::int32_t>{4, 5});
  const auto first = h.run();
  ASSERT_TRUE(first.ok()) << first.describe_failure();

  h.push("in", std::int32_t{6});
  const auto second = h.run();

  ASSERT_TRUE(second.ok()) << second.describe_failure();
  EXPECT_EQ(first.values<std::uint32_t>("out"), (std::vector<std::uint32_t>{1, 2}));
  EXPECT_EQ(second.value<std::uint32_t>("out"), 1U);
  EXPECT_EQ(probe->starts, 2U);
  EXPECT_EQ(probe->computes, 3U);
  EXPECT_EQ(probe->stops, 2U);
}

// REQ: OPERATOR-065
TEST(OperatorTestHarness, AlignedInputsUseInclusiveWindowAndDrainUnmatchedHeads) {
  using namespace std::chrono_literals;
  using holoscan::testing::sample_at;
  for (const auto offset : {99ms, 100ms, 101ms}) {
    auto h = holoscan::testing::create_operator_test<AlignedSumOp>();
    h.push_samples("left", std::vector{sample_at(std::int32_t{1}, 0ms)});
    h.push_samples("right", std::vector{sample_at(std::int32_t{10}, offset)});
    h.capture<std::int32_t>("sum");
    const auto result = h.run();
    ASSERT_TRUE(result.ok()) << offset.count() << ": " << result.describe_failure();
    EXPECT_EQ(result.values<std::int32_t>("sum"),
              offset <= 100ms ? std::vector<std::int32_t>{11} : std::vector<std::int32_t>{});
  }
}

// REQ: OPERATOR-065
TEST(OperatorTestHarness, AlignedSequenceUsesRuntimeFifoSelectionAndFinalEos) {
  using namespace std::chrono_literals;
  using holoscan::testing::sample_at;
  auto h = holoscan::testing::create_operator_test<AlignedSumOp>();
  h.push_samples("left",
                 std::vector{sample_at(std::int32_t{1}, 0ms),
                             sample_at(std::int32_t{2}, 200ms),
                             sample_at(std::int32_t{3}, 400ms)});
  h.push_samples("right",
                 std::vector{sample_at(std::int32_t{10}, 100ms),
                             sample_at(std::int32_t{20}, 301ms),
                             sample_at(std::int32_t{30}, 450ms)});
  h.capture<std::int32_t>("sum");
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.values<std::int32_t>("sum"), (std::vector<std::int32_t>{11, 23}));
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, ResultCaptureIsIndependentOfReexecutingItsPlan) {
  auto h = holoscan::testing::create_operator_test<ScaleOp>();
  h.push("in", std::int32_t{21});
  h.capture<std::int32_t>("out", 1U);
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  auto second = holoscan::run_async(result.plan());
  auto third = holoscan::run_async(result.plan());
  second.wait();
  third.wait();
  EXPECT_EQ(third.termination(), holoscan::RunTermination::kCompleted);
  ASSERT_EQ(second.termination(), holoscan::RunTermination::kCompleted);
  EXPECT_EQ(result.values<std::int32_t>("out"), (std::vector<std::int32_t>{21}));
}

// REQ: OPERATOR-072
TEST(OperatorTestHarness, ConfigurationOwnsStringsReplacesHookAndPreservesNativePartitionIntent) {
  auto h = holoscan::testing::create_operator_test<ScaleOp>();
  h.set_params(ScaleParams{.factor = 2});
  h.input_queue_depth("in", 16U);
  {
    std::string topic = "owned-input-topic";
    h.input_flow("in",
                 holoscan::ConnectionOptions{
                     .queue_depth = 16U, .topic_name = topic, .max_inflight_samples = 16U});
    topic.assign("changed-after-setter");
  }
  h.output_flow("out", holoscan::ConnectionOptions{.queue_depth = 8U});
  h.set_compile_options(holoscan::CompileOptions{});
  h.configure_graph([](auto&) { ADD_FAILURE() << "replaced hook must not run"; });
  bool hook_called = false;
  h.configure_graph([&](auto& f) {
    hook_called = true;
    EXPECT_EQ(f.tested_operator().name(), "operator");
    const auto view = f.graph().view();
    ASSERT_EQ(view.edges().size(), 2U);
    bool found_topic = false;
    for (const auto& edge : view.edges()) {
      if (edge.add_flow_options.topic_name == "owned-input-topic") {
        found_topic = true;
        EXPECT_EQ(edge.add_flow_options.queue_depth, 16U);
      }
    }
    EXPECT_TRUE(found_topic);
    f.graph().partition("operator").add(f.tested_operator());
    f.graph().auto_partition_remaining();
  });
  h.push_many("in", std::vector<std::int32_t>{1, 2, 3});
  h.capture<std::int32_t>("out");
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_TRUE(hook_called);
  EXPECT_EQ(result.values<std::int32_t>("out"), (std::vector<std::int32_t>{2, 4, 6}));
}

// REQ: OPERATOR-072
TEST(OperatorTestHarness, HelperPlacementRejectsExpiredPartitionWithEmptyAndNonemptyHelpers) {
  const auto expired = [] {
    holoscan::Graph graph{"expired"};
    return graph.partition("worker");
  }();
  for (const bool staged : {false, true}) {
    auto h = holoscan::testing::create_operator_test<ScaleOp>();
    if (staged) {
      h.push("in", std::int32_t{21});
      h.capture<std::int32_t>("out");
    }
    h.configure_graph([&](auto& view) {
      auto partition = expired;
      const auto expect_invalid = [](auto&& add) {
        try {
          add();
          ADD_FAILURE() << "expired partition accepted";
        } catch (const holoscan::RuntimeError& error) {
          EXPECT_NE(std::string{error.what()}.find("GRAPH_PARTITION_HANDLE_INVALID"),
                    std::string::npos);
        }
      };
      expect_invalid([&] { view.add_sources_to(partition); });
      expect_invalid([&] { view.add_collectors_to(partition); });
    });
    static_cast<void>(h.run());
  }
}

// REQ: OPERATOR-072
TEST(OperatorTestHarness, InvalidQueueSettingCanBeClearedWithoutConsumingInput) {
  auto h = holoscan::testing::create_operator_test<ScaleOp>();
  h.push("in", std::int32_t{21});
  h.capture<std::int32_t>("out");
  EXPECT_THROW(h.input_flow("in", holoscan::ConnectionOptions{.queue_depth = 0U}),
               std::invalid_argument);
  h.input_flow("in", std::nullopt);
  h.configure_graph(nullptr);
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<std::int32_t>("out"), 21);
}

// REQ: OPERATOR-071
TEST(OperatorTestHarness, GroupedFramesPreserveOptionalControlThenParameterFallback) {
  using namespace std::chrono_literals;
  using namespace holoscan::testing;
  auto h = create_operator_test<ControlledPairOp>();
  h.set_params(ScaleParams{.factor = 2});
  h.push_frame(frame_at(100ms,
                        input("control", std::int32_t{3}),
                        input("left", std::int32_t{1}),
                        input("right", std::int32_t{10})));
  h.push_frame(frame_at(200ms, input("left", std::int32_t{2}), input("right", std::int32_t{20})));
  h.capture<std::int32_t>("out");
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.values<std::int32_t>("out"), (std::vector<std::int32_t>{33, 44}));
}

// REQ: OPERATOR-071
TEST(OperatorTestHarness, GroupedFrameFailureIsAtomicAndCannotMixInputProducers) {
  using namespace std::chrono_literals;
  using namespace holoscan::testing;
  auto h = create_operator_test<ControlledPairOp>();
  h.push_frame(frame_at(100ms, input("left", std::int32_t{1}), input("right", std::int32_t{10})));
  EXPECT_THROW(
      h.push_frame(frame_at(200ms, input("left", std::int32_t{2}), input("right", float{20}))),
      holoscan::RuntimeError);
  EXPECT_THROW(h.push("left", std::int32_t{3}), std::invalid_argument);
  h.capture<std::int32_t>("out");
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.values<std::int32_t>("out"), (std::vector<std::int32_t>{11}));
}

// REQ: OPERATOR-072
TEST(OperatorTestHarness, UnknownFlowSelectorsFailWithoutConsumingPendingInputs) {
  auto h = holoscan::testing::create_operator_test<ScaleOp>();
  h.push("in", std::int32_t{9});
  h.capture<std::int32_t>("out");
  EXPECT_THROW(h.input_flow("typo", holoscan::ConnectionOptions{}), holoscan::RuntimeError);
  EXPECT_THROW(h.input_flow("typo", std::nullopt), holoscan::RuntimeError);
  EXPECT_THROW(h.output_flow("uncaptured", holoscan::ConnectionOptions{}), holoscan::RuntimeError);
  EXPECT_THROW(h.output_flow("uncaptured", std::nullopt), holoscan::RuntimeError);
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<std::int32_t>("out"), 9);
}

// REQ: OPERATOR-072
TEST(OperatorTestHarness, NativeViewPartitionsActualHelpersAndChecksTypedSelectors) {
  auto h = holoscan::testing::create_operator_test<ScaleOp>();
  h.push("in", std::int32_t{9});
  h.capture<std::int32_t>("out");
  h.configure_graph([](auto& f) {
    auto source = f.graph().partition("source");
    auto collector = f.graph().partition("collector");
    f.add_sources_to(source);
    f.add_collectors_to(collector);
    f.graph().partition("operator").add(f.tested_operator());
    EXPECT_EQ(f.template source_output<std::int32_t>("in").name(), "output");
    EXPECT_EQ(f.template capture_input<std::int32_t>("out").name(), "input");
    EXPECT_THROW(static_cast<void>(f.template source_output<float>("in")), holoscan::RuntimeError);
    const auto source_placement = f.input_source_device_placement("in", holoscan::DeviceId{1});
    EXPECT_EQ(source_placement.output_port, "output");
    const auto placement = f.output_device_placement("out", holoscan::DeviceId{1});
    EXPECT_EQ(placement.operator_path, "operator");
    EXPECT_EQ(placement.output_port, "out");
  });
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<std::int32_t>("out"), 9);
}

namespace {
class HostTensorProducer final : public holoscan::Operator<> {
 public:
  explicit HostTensorProducer(std::uint64_t calls = 2U, int fault = 0)
      : calls_(calls), fault_(fault) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(out, "out")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = DLDataType{kDLInt, 32, 1},
                               .rank = 1U},
            .bounds = holoscan::tensor_bounds(16U),
            .storage = holoscan::TensorOutputStorage::kRuntimePool});
  }
  holoscan::Contract contract() const override {
    holoscan::Contract c;
    c.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{1}})
        .complete_after(holoscan::ComputeCalls{.limit = calls_});
    return c;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    if (fault_ == 1)
      return {};
    if (fault_ == 2 && next_ == 20) {
      context.request_stop();
      return {};
    }
    const std::array<std::int64_t, 1> shape{4};
    auto loan = out.allocate_tensor(shape);
    if (!loan)
      return holoscan::make_unexpected(std::move(loan).error());
    auto filled = loan->write_host_and_commit([&](const auto& write) {
      for (int i = 0; i < 4; ++i)
        write.template data_as<std::int32_t>()[i] = next_ + i;
    });
    if (!filled)
      return filled;
    next_ += 10;
    auto emitted = out.emit(std::move(*loan));
    if (!emitted)
      return emitted;
    if (fault_ == 3 && next_ == 30)
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "failure after final emit"});
    return {};
  }
  holoscan::Output<holoscan::Tensor> out;

 private:
  std::uint64_t calls_;
  int fault_;
  std::int32_t next_{10};
};
}  // namespace

namespace {
class ControlledTensorOutputProbe final : public holoscan::Operator<> {
 public:
  explicit ControlledTensorOutputProbe(std::shared_ptr<LifecycleProbe> probe)
      : probe_(std::move(probe)) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in");
    spec.output(out, "out")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = DLDataType{kDLInt, 32, 1},
                               .rank = 1U},
            .bounds = holoscan::tensor_bounds(16U),
            .storage = holoscan::TensorOutputStorage::kRuntimePool});
  }
  holoscan::Contract contract() const override {
    holoscan::Contract value;
    value.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{10}})
        .input(in, holoscan::LatestWithin{.max_age = std::chrono::milliseconds{100}})
        .on_missing(in, holoscan::Skip{});
    return value;
  }
  void start() override { ++probe_->starts; }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    ++probe_->computes;
    return {};
  }
  holoscan::Input<std::int32_t> in;
  holoscan::Output<holoscan::Tensor> out;

 private:
  std::shared_ptr<LifecycleProbe> probe_;
};
}  // namespace

// REQ: OPERATOR-058
TEST(OperatorTestHarness, ControlledTensorRetainRejectsBeforeExecutionAndPreservesStaging) {
  using namespace std::chrono_literals;
  using namespace holoscan::testing;
  auto probe = std::make_shared<LifecycleProbe>();
  NativeOperatorTest h{[probe](holoscan::Graph& graph) {
    return holoscan::detail::OperatorTestingAccess::erase(
        graph.op<ControlledTensorOutputProbe>("operator", probe));
  }};
  h.stage_input("in", staged_samples<std::int32_t>({sample_at(std::int32_t{7}, 900ms)}));
  h.declare_capture("out", retain_capture<holoscan::Tensor>(8U));
  try {
    const auto result = h.run_controlled_clock({.tick = 1000ms, .timeout = 1s});
    FAIL() << "Expected cold scalar-profile rejection: " << result.describe_failure();
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string{error.what()}.find("scalar payloads only"), std::string::npos);
  }
  EXPECT_EQ(probe->starts, 0U);
  EXPECT_EQ(probe->computes, 0U);
  h.set_operator_factory([](holoscan::Graph& graph) {
    return holoscan::detail::OperatorTestingAccess::erase(graph.op<ScaleOp>("operator"));
  });
  h.declare_capture("out", value_capture<std::int32_t>(1U));
  const auto recovered = h.run({.timeout = 1s});
  ASSERT_TRUE(recovered.ok()) << recovered.describe_failure();
  EXPECT_EQ(recovered.value<std::int32_t>("out"), 7);
}

// REQ: OPERATOR-073
TEST(OperatorTestHarness, TensorSnapshotsOwnBytesAcrossPoolReuseAndLaterRuns) {
  auto h = holoscan::testing::create_operator_test<HostTensorProducer>();
  h.capture_snapshot<holoscan::Tensor>("out");
  auto first = h.run();
  ASSERT_TRUE(first.ok()) << first.describe_failure();
  const auto captures = first.snapshots<holoscan::Tensor>("out");
  ASSERT_EQ(captures.size(), 2U);
  ASSERT_EQ(captures[0].bytes().size(), 16U);
  std::array<std::int32_t, 4> values{};
  std::memcpy(values.data(), captures[0].bytes().data(), captures[0].bytes().size());
  EXPECT_EQ(values, (std::array<std::int32_t, 4>{10, 11, 12, 13}));
  std::memcpy(values.data(), captures[1].bytes().data(), captures[1].bytes().size());
  EXPECT_EQ(values, (std::array<std::int32_t, 4>{20, 21, 22, 23}));
  EXPECT_EQ(captures[0].source_device().device_type, kDLCPU);
  EXPECT_EQ(captures[0].shape(), (std::vector<std::int64_t>{4}));
  EXPECT_THROW(static_cast<void>(first.snapshot<holoscan::Tensor>("out")), std::logic_error);
  EXPECT_THROW(static_cast<void>(first.values<holoscan::Tensor>("out")), std::invalid_argument);
  auto second = h.run();
  ASSERT_TRUE(second.ok()) << second.describe_failure();
  EXPECT_EQ(first.snapshots<holoscan::Tensor>("out")[0].bytes(), captures[0].bytes());
}

// REQ: OPERATOR-073
namespace {
template <typename T>
concept HasValueCaptureFactory = requires { holoscan::testing::value_capture<T>(); };
template <typename T>
concept HasSnapshotCaptureFactory = requires { holoscan::testing::snapshot_capture<T>(); };
template <typename T>
concept HasRetainCaptureFactory = requires { holoscan::testing::retain_capture<T>(); };
template <typename T>
concept HasPublicSnapshot = requires { typename holoscan::testing::Snapshot<T>; };
template <typename T>
concept CanCaptureOrdinaryValue =
    requires(holoscan::testing::OperatorTestHarness<HostTensorProducer>& fixture) {
      fixture.template capture<T>("out");
    };
template <typename T>
concept CanReadSnapshots =
    requires(const holoscan::testing::TestResult& result) { result.template snapshots<T>("out"); };
template <typename T>
concept CanStagePrivateAdapter =
    requires(holoscan::testing::NativeOperatorTest& fixture, std::shared_ptr<const T> adapter) {
      fixture.stage_input("in", adapter);
    };
template <typename T>
concept CanDeclarePrivateAdapter =
    requires(holoscan::testing::NativeOperatorTest& fixture, std::shared_ptr<const T> adapter) {
      fixture.declare_capture("out", adapter);
    };
static_assert(HasValueCaptureFactory<std::int32_t>);
static_assert(HasValueCaptureFactory<holoscan::schema::StringT>);
static_assert(!HasValueCaptureFactory<holoscan::Tensor>);
static_assert(!HasValueCaptureFactory<holoscan::schema::TensorPacketT>);
static_assert(!CanCaptureOrdinaryValue<holoscan::Tensor>);
static_assert(!CanCaptureOrdinaryValue<holoscan::schema::TensorPacketT>);
static_assert(HasSnapshotCaptureFactory<holoscan::Tensor>);
static_assert(HasSnapshotCaptureFactory<holoscan::schema::TensorPacketT>);
static_assert(!HasSnapshotCaptureFactory<std::int32_t>);
static_assert(!HasSnapshotCaptureFactory<holoscan::schema::StringT>);
static_assert(HasRetainCaptureFactory<holoscan::Tensor>);
static_assert(HasRetainCaptureFactory<holoscan::schema::TensorPacketT>);
static_assert(!HasRetainCaptureFactory<std::int32_t>);
static_assert(HasPublicSnapshot<holoscan::Tensor>);
static_assert(HasPublicSnapshot<holoscan::schema::TensorPacketT>);
static_assert(!HasPublicSnapshot<std::int32_t>);
static_assert(!CanReadSnapshots<std::int32_t>);
static_assert(!CanReadSnapshots<holoscan::schema::StringT>);
static_assert(!CanStagePrivateAdapter<holoscan::testing::detail::InputScript>);
static_assert(!CanDeclarePrivateAdapter<holoscan::testing::detail::CaptureDeclaration>);
static_assert(
    !std::constructible_from<holoscan::testing::StagedInput,
                             std::shared_ptr<const holoscan::testing::detail::InputScript>>);
static_assert(
    !std::constructible_from<holoscan::testing::CaptureRequest,
                             std::shared_ptr<const holoscan::testing::detail::CaptureDeclaration>>);
}  // namespace

TEST(OperatorTestHarness, TensorValueCaptureRequiresExplicitOwnershipPolicy) {
  EXPECT_NO_THROW(static_cast<void>(holoscan::testing::snapshot_capture<holoscan::Tensor>()));
  EXPECT_NO_THROW(static_cast<void>(holoscan::testing::retain_capture<holoscan::Tensor>()));
}

// REQ: OPERATOR-072
TEST(OperatorTestHarness, TensorOverridesDeepOwnBoundsAndRejectScalarPorts) {
  auto scalar = holoscan::testing::create_operator_test<ScaleOp>();
  scalar.push("in", std::int32_t{7});
  scalar.capture<std::int32_t>("out");
  EXPECT_THROW(
      scalar.input_tensor_spec("in",
                               holoscan::TensorInputSpec{
                                   .representation = {.memory_kind = holoscan::MemoryKind::kHost}}),
      std::invalid_argument);
  EXPECT_THROW(scalar.input_tensor_spec("in", std::nullopt), std::invalid_argument);
  ASSERT_TRUE(scalar.run().ok());

  auto h = holoscan::testing::create_operator_test<HostTensorProducer>();
  {
    std::vector<std::int64_t> shape{4};
    h.output_tensor_spec("out",
                         holoscan::TensorOutputSpec{
                             .representation = {.dtype = DLDataType{kDLInt, 32, 1}},
                             .bounds = holoscan::tensor_bounds(holoscan::TensorShapeView{shape})});
    shape[0] = 999;
  }
  h.capture_snapshot<holoscan::Tensor>("out");
  h.configure_graph([](auto& f) {
    const auto view = f.graph().view();
    for (const auto& port : view.ports()) {
      if (port.name != "out")
        continue;
      ASSERT_TRUE(port.graph_override.tensor_contract.has_value());
      ASSERT_EQ(port.graph_override.tensor_contract->axes.size(), 1U);
      EXPECT_EQ(port.graph_override.tensor_contract->max_byte_span, 16U);
    }
  });
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
}

// REQ: OPERATOR-072
TEST(OperatorTestHarness, HostOutputDeviceBindingPreservesNativeRejectionAndCanBeCleared) {
  auto h = holoscan::testing::create_operator_test<HostTensorProducer>();
  h.output_device("out", holoscan::DeviceId{1});
  h.capture_snapshot<holoscan::Tensor>("out");
  auto rejected = h.run();
  ASSERT_FALSE(rejected.ok());
  EXPECT_NE(rejected.describe_failure().find("PHYSICAL_TENSOR_DEVICE_BINDING_UNEXPECTED"),
            std::string::npos);
  h.output_device("out", std::nullopt);
  auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
}

namespace {
class HostTensorPass final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in").expects_tensor(
        holoscan::TensorInputSpec{.representation = {.memory_kind = holoscan::MemoryKind::kHost,
                                                     .dtype = DLDataType{kDLInt, 32, 1},
                                                     .rank = 1U}});
    spec.output(out, "out")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = DLDataType{kDLInt, 32, 1},
                               .rank = 1U},
            .bounds = holoscan::tensor_bounds(16U)});
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = in.receive();
    if (!sample)
      return holoscan::make_unexpected(std::move(sample).error());
    return out.emit(std::move(*sample));
  }
  holoscan::Input<holoscan::Tensor> in;
  holoscan::Output<holoscan::Tensor> out;
};

holoscan::Tensor input_tensor(std::int32_t first) {
  struct Storage {
    DLManagedTensor managed{};
    std::array<std::int64_t, 1> shape{4};
    std::array<std::int32_t, 5> data{};
  };
  auto owner = std::make_unique<Storage>();
  owner->data = {-1, first, first + 1, first + 2, first + 3};
  owner->managed.dl_tensor = DLTensor{.data = owner->data.data(),
                                      .device = DLDevice{kDLCPU, 0},
                                      .ndim = 1,
                                      .dtype = DLDataType{kDLInt, 32, 1},
                                      .shape = owner->shape.data(),
                                      .strides = nullptr,
                                      .byte_offset = sizeof(std::int32_t)};
  owner->managed.manager_ctx = owner.get();
  owner->managed.deleter = [](DLManagedTensor* value) {
    delete static_cast<Storage*>(value->manager_ctx);
  };
  auto* managed = &owner.release()->managed;
  return holoscan::Tensor{managed};
}
}  // namespace

// REQ: OPERATOR-073
TEST(OperatorTestHarness, StagedTensorUsesRuntimeStorageAndPreservesCaptureMetadata) {
  using namespace std::chrono_literals;
  auto h = holoscan::testing::create_operator_test<HostTensorPass>();
  h.push_samples("in",
                 std::vector<holoscan::testing::TestSample<holoscan::Tensor>>{
                     holoscan::testing::sample_at(input_tensor(10), 100ms),
                     holoscan::testing::sample_at(input_tensor(20), 200ms)});
  h.capture_snapshot<holoscan::Tensor>("out");
  h.configure_graph([](auto& f) {
    const auto graph_view = f.graph().view();
    bool saw_source = false;
    for (const auto& port : graph_view.ports()) {
      if (port.name != "output")
        continue;
      saw_source = true;
      ASSERT_TRUE(port.setup.tensor_allocation.has_value());
      EXPECT_EQ(port.setup.tensor_allocation->max_byte_span, 16U);
    }
    EXPECT_TRUE(saw_source);
  });
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  auto records = result.snapshots<holoscan::Tensor>("out");
  ASSERT_EQ(records.size(), 2U);
  std::array<std::int32_t, 4> copied{};
  std::memcpy(copied.data(), records[0].bytes().data(), records[0].bytes().size());
  EXPECT_EQ(copied, (std::array<std::int32_t, 4>{10, 11, 12, 13}));
  EXPECT_EQ(records[0].sample_metadata().capture_timestamp_ns, 100000000);
  EXPECT_EQ(records[1].sample_metadata().capture_timestamp_ns, 200000000);
}

namespace {
class DeviceTensorProducer final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(out, "out")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kCudaDevice,
                               .dtype = DLDataType{kDLUInt, 8, 1},
                               .rank = 1U},
            .bounds = holoscan::tensor_bounds(16U),
            .storage = holoscan::TensorOutputStorage::kRuntimePool});
  }
  holoscan::Contract contract() const override {
    holoscan::Contract c;
    c.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{1}})
        .complete_after(holoscan::ComputeCalls{.limit = 2U});
    return c;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    const std::array<std::int64_t, 1> shape{16};
    auto loan = out.allocate_tensor(shape);
    if (!loan)
      return holoscan::make_unexpected(std::move(loan).error());
    const auto stream = context.cuda_stream();
    auto filled = loan->write_and_commit(
        stream, [&](const auto& write) -> holoscan::expected<void, holoscan::Error> {
          const auto status =
              cudaMemsetAsync(write.data(), ++next_, write.byte_size(), stream.get());
          if (status != cudaSuccess)
            return holoscan::make_unexpected(
                holoscan::Error{holoscan::ErrorCode::kFailure, cudaGetErrorString(status)});
          return {};
        });
    if (!filled)
      return filled;
    return out.emit(std::move(*loan));
  }
  holoscan::Output<holoscan::Tensor> out;

 private:
  int next_{};
};
}  // namespace

// REQ: OPERATOR-073
TEST(OperatorTestHarness, DeviceSnapshotWaitsForProducerAndUsesExplicitDevice) {
  int count{};
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
    GTEST_SKIP() << "CUDA device unavailable";
  const int device = count > 1 ? 1 : 0;
  auto h = holoscan::testing::create_operator_test<DeviceTensorProducer>();
  h.output_device("out", holoscan::DeviceId{device});
  h.capture_snapshot<holoscan::Tensor>("out");
  auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  auto values = result.snapshots<holoscan::Tensor>("out");
  ASSERT_EQ(values.size(), 2U);
  EXPECT_EQ(values[0].source_device().device_id, device);
  EXPECT_EQ(values[0].source_device().device_type, kDLCUDA);
  EXPECT_EQ(values[0].bytes(), std::vector<std::byte>(16U, std::byte{1}));
  EXPECT_EQ(values[1].bytes(), std::vector<std::byte>(16U, std::byte{2}));
}

// REQ: OPERATOR-073
TEST(OperatorTestHarness, SnapshotOverflowReturnsEarlierCompleteRecords) {
  auto h = holoscan::testing::create_operator_test<HostTensorProducer>();
  h.capture_snapshot<holoscan::Tensor>("out", 1U);
  const auto result = h.run();
  EXPECT_FALSE(result.ok());
  EXPECT_NE(result.describe_failure().find("capture 'out'"), std::string::npos);
  EXPECT_NE(result.describe_failure().find("kFailed"), std::string::npos);
  const auto snapshots = result.snapshots<holoscan::Tensor>("out");
  ASSERT_EQ(snapshots.size(), 1U);
  std::array<std::int32_t, 4> copied{};
  std::memcpy(copied.data(), snapshots[0].bytes().data(), snapshots[0].bytes().size());
  EXPECT_EQ(copied, (std::array<std::int32_t, 4>{10, 11, 12, 13}));
}

namespace {
struct ReleaseHook {
  std::promise<void>& signal;
  bool released{};
  void open() {
    if (!std::exchange(released, true))
      signal.set_value();
  }
  ~ReleaseHook() { open(); }
};
}  // namespace

// REQ: OPERATOR-064
TEST(OperatorTestHarness, ActiveAttemptRejectsMutatorsAndAnotherRun) {
  auto h = holoscan::testing::create_operator_test<ScaleOp>();
  h.push("in", std::int32_t{5});
  h.capture<std::int32_t>("out");
  std::promise<void> entered;
  std::promise<void> release;
  auto entry = entered.get_future();
  auto gate = release.get_future().share();
  h.configure_graph([&](auto&) {
    entered.set_value();
    gate.wait();
  });
  auto running = std::async(std::launch::async, [&] { return h.run(); });
  // Destroy this guard before the async future, including on an assertion return. A hook
  // arriving after the entry deadline must still be released before the future joins.
  ReleaseHook release_hook{release};
  ASSERT_EQ(entry.wait_for(std::chrono::seconds{5}), std::future_status::ready)
      << "operator-test graph hook did not arrive before the rendezvous deadline";
  EXPECT_THROW(h.push("in", std::int32_t{6}), std::logic_error);
  EXPECT_THROW(h.push(&ScaleOp::in, std::int32_t{6}), std::logic_error);
  EXPECT_THROW(h.set_params(ScaleParams{.factor = 3}), std::logic_error);
  EXPECT_THROW(h.capture<std::int32_t>("out"), std::logic_error);
  EXPECT_THROW(h.configure_graph(nullptr), std::logic_error);
  EXPECT_THROW(h.output_flow("out", std::nullopt), std::logic_error);
  EXPECT_THROW(static_cast<void>(h.run()), std::logic_error);
  release_hook.open();
  const auto result = running.get();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<std::int32_t>("out"), 5);
}

namespace {
class SinkOp final : public holoscan::Operator<> {
 public:
  explicit SinkOp(std::shared_ptr<std::vector<std::int32_t>> seen) : seen_(std::move(seen)) {}
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in"); }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto value = in.receive();
    if (!value)
      return holoscan::make_unexpected(std::move(value).error());
    seen_->push_back(value->data);
    return {};
  }
  holoscan::Input<std::int32_t> in;

 private:
  std::shared_ptr<std::vector<std::int32_t>> seen_;
};
class FailingOp final : public holoscan::Operator<> {
 public:
  explicit FailingOp(int phase, std::shared_ptr<LifecycleProbe> probe)
      : phase_(phase), probe_(std::move(probe)) {}
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in"); }
  void start() override {
    ++probe_->starts;
    if (phase_ == 0)
      throw std::runtime_error{"fixture-start-failure"};
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    ++probe_->computes;
    if (phase_ == 1)
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "fixture-compute-failure"});
    return {};
  }
  void stop() override {
    ++probe_->stops;
    if (phase_ == 2)
      throw std::runtime_error{"fixture-stop-failure"};
  }
  holoscan::Input<std::int32_t> in;

 private:
  int phase_;
  std::shared_ptr<LifecycleProbe> probe_;
};
}  // namespace

// REQ: OPERATOR-064
TEST(OperatorTestHarness, NoOutputSinkCompletesAndDoesNotReplayConsumedInput) {
  auto seen = std::make_shared<std::vector<std::int32_t>>();
  auto h = holoscan::testing::create_operator_test<SinkOp>(seen);
  h.push_many<std::int32_t>("in", {1, 2, 3});
  const auto first = h.run();
  ASSERT_TRUE(first.ok()) << first.describe_failure();
  EXPECT_EQ(*seen, (std::vector<std::int32_t>{1, 2, 3}));
  h.push_many<std::int32_t>("in", {});
  const auto second = h.run();
  ASSERT_TRUE(second.ok()) << second.describe_failure();
  EXPECT_EQ(*seen, (std::vector<std::int32_t>{1, 2, 3}));
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, RuntimeReportsStartupComputeAndCleanupFailure) {
  for (int phase = 0; phase != 3; ++phase) {
    SCOPED_TRACE(phase);
    auto probe = std::make_shared<LifecycleProbe>();
    auto h = holoscan::testing::create_operator_test<FailingOp>(phase, probe);
    h.push("in", std::int32_t{1});
    const auto result = h.run();
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.termination(), holoscan::RunTermination::kFailed);
    EXPECT_FALSE(result.diagnostics().empty());
    EXPECT_EQ(probe->starts, 1U);
    if (phase != 0)
      EXPECT_EQ(probe->stops, 1U);
  }
}

// REQ: OPERATOR-073
TEST(OperatorTestHarness, NativeValueAdapterCannotBypassTensorCapturePolicy) {
  EXPECT_THROW(
      static_cast<void>(holoscan::testing::detail::ValueCaptureDeclaration<holoscan::Tensor>{1U}),
      std::invalid_argument);
}

// REQ: OPERATOR-073
TEST(OperatorTestHarness, SnapshotStreamExceedsNativePoolCapacityWithoutRetainingSlots) {
  auto h = holoscan::testing::create_operator_test<HostTensorProducer>(32U);
  h.capture_snapshot<holoscan::Tensor>("out", 32U);
  h.output_flow("out", holoscan::ConnectionOptions{.queue_depth = 2U});
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(result.plan().tensor_allocations().size(), 1U);
  EXPECT_LT(result.plan().tensor_allocations()[0].slot_count, 32U);
  const auto snapshots = result.snapshots<holoscan::Tensor>("out");
  ASSERT_EQ(snapshots.size(), 32U);
  for (std::size_t i = 0; i < snapshots.size(); ++i) {
    std::array<std::int32_t, 4> values{};
    std::memcpy(values.data(), snapshots[i].bytes().data(), snapshots[i].bytes().size());
    EXPECT_EQ(values[0], static_cast<std::int32_t>((i + 1U) * 10U));
  }
}

// REQ: OPERATOR-072
TEST(OperatorTestHarness, CompileOptionsPreserveNativeSchedulingRejectionAndReplacement) {
  auto h = holoscan::testing::create_operator_test<ScaleOp>();
  h.push("in", std::int32_t{7});
  h.capture<std::int32_t>("out");
  h.configure_graph([](auto& f) { f.graph().partition("worker").include_all(); });
  holoscan::CompileOptions invalid;
  invalid.deployment.bind_thread_placement(
      "worker",
      holoscan::ThreadPlacement{
          .scheduling = {.policy = holoscan::SchedulingPolicy::kFIFO, .priority = 0}});
  h.set_compile_options(invalid);
  const auto rejected = h.run();
  EXPECT_FALSE(rejected.ok());
  EXPECT_FALSE(rejected.plan().ok());
  ASSERT_FALSE(rejected.plan().partition_placements().empty());
  EXPECT_EQ(rejected.plan().partition_placements()[0].requested.scheduling.priority, 0);
  h.set_compile_options(holoscan::CompileOptions{});
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<std::int32_t>("out"), 7);
}

// REQ: OPERATOR-072, OPERATOR-073
TEST(OperatorTestHarness, ExplicitGreenContextUsesNativeCapabilityAndDeployment) {
  int count{};
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
    GTEST_SKIP() << "CUDA unavailable";
  const holoscan::DeviceId device{count > 1 ? 1 : 0};
  const auto capability = holoscan::query_cuda_green_context_capability(device);
  if (!capability)
    GTEST_SKIP() << "Green Context unavailable: " << capability.error().message;
  auto h = holoscan::testing::create_operator_test<DeviceTensorProducer>();
  h.capture_snapshot<holoscan::Tensor>("out");
  h.output_device("out", device);
  h.configure_graph([](auto& f) { f.graph().partition("worker").include_all(); });
  holoscan::CompileOptions options;
  options.deployment.bind_cuda_green_context(
      "worker", {.device = device, .sm_count = capability->structured_sm_quantum});
  h.set_compile_options(options);
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(result.plan().cuda_green_context_requests().size(), 1U);
  const auto& accepted = result.plan().cuda_green_context_requests()[0].accepted;
  if (accepted) {
    EXPECT_EQ(accepted->device, device);
  } else {
    FAIL() << "Expected an accepted Green Context request";
  }
  ASSERT_EQ(result.snapshots<holoscan::Tensor>("out").size(), 2U);
  EXPECT_EQ(result.snapshots<holoscan::Tensor>("out")[1].bytes(),
            std::vector<std::byte>(16U, std::byte{2}));
}

// REQ: OPERATOR-072
TEST(OperatorTestHarness, PlacementRecordOwnsSelectorUntilHookReturns) {
  auto h = holoscan::testing::create_operator_test<ScaleOp>();
  h.push("in", std::int32_t{1});
  h.capture<std::int32_t>("out");
  h.configure_graph([](auto& f) {
    std::string selector = "out";
    const auto record = f.output_device_placement(selector, holoscan::DeviceId{1});
    selector[0] = 'x';
    EXPECT_EQ(record.output_port, "out");
  });
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
}

namespace {
class ReentrantAuthoringOp final : public holoscan::Operator<> {
 public:
  explicit ReentrantAuthoringOp(const std::shared_ptr<std::function<void()>>& callback) {
    if (*callback)
      (*callback)();
  }
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in"); }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }
  holoscan::Input<std::int32_t> in;
};
}  // namespace

// REQ: OPERATOR-064
TEST(OperatorTestHarness, TypedPortResolutionRejectsReentrantAuthoringWithoutDeadlock) {
#if defined(__linux__)
  EXPECT_EXIT(
      {
        alarm(3);
        auto callback = std::make_shared<std::function<void()>>();
        auto h = holoscan::testing::create_operator_test<ReentrantAuthoringOp>(callback);
        bool rejected = false;
        *callback = [&] {
          try {
            h.input_flow("in", std::nullopt);
          } catch (const std::logic_error&) {
            rejected = true;
          }
        };
        h.push(&ReentrantAuthoringOp::in, std::int32_t{1});
        _exit(rejected ? 0 : 1);
      },
      ::testing::ExitedWithCode(0),
      "");
#else
  GTEST_SKIP() << "Process-isolated deadlock guard requires POSIX alarm";
#endif
}

// REQ: OPERATOR-073
TEST(OperatorTestHarness, NativeSnapshotAdapterEnforcesTheSameColdBoundsAsTypedFacade) {
  using Capture = holoscan::testing::detail::SnapshotCaptureDeclaration<holoscan::Tensor>;
  EXPECT_THROW(static_cast<void>(Capture{0U, 1U}), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(Capture{1U, 0U}), std::invalid_argument);
  if constexpr (sizeof(std::size_t) > sizeof(std::uint32_t)) {
    EXPECT_THROW(static_cast<void>(Capture{static_cast<std::size_t>(UINT32_MAX) + 1U, 1U}),
                 std::invalid_argument);
  }
}

// A deliberately distinct native type; equal semantic metadata must not authorize a cast.
struct TestingSchemaB {};
using TestingSchemaA = holoscan::schema::StringT;

struct SnapshotWithoutCloneProbe {
  std::shared_ptr<holoscan::Tensor> data;
};
namespace holoscan {
template <>
consteval SchemaIdentity schema_identity<SnapshotWithoutCloneProbe>() {
  return schema_identity<schema::TensorPacketT>();
}
template <>
struct DescriptorTensorSchemaNativeTraits<SnapshotWithoutCloneProbe> {
  using root_type = schema::TensorPacket;
  static constexpr std::size_t metadata_capacity = 512U;
  static constexpr std::size_t tensor_count = 1U;
  static constexpr std::array<std::uint16_t, 1U> tensor_field_ids{3U};
  static const Tensor* tensor_at(const SnapshotWithoutCloneProbe& value, std::size_t) noexcept {
    return value.data.get();
  }
};
}  // namespace holoscan
static_assert(holoscan::testing::TensorCapturePayload<SnapshotWithoutCloneProbe>);
static_assert(!holoscan::testing::TensorSnapshotPayload<SnapshotWithoutCloneProbe>);
static_assert(!HasSnapshotCaptureFactory<SnapshotWithoutCloneProbe>);

struct TestingPodA {
  std::uint32_t value{};
};
struct TestingPodB {
  std::uint32_t value{};
};
namespace holoscan {
template <>
struct ChannelTypeIdentity<TestingPodA> {
  static constexpr std::string_view name = "com.nvidia.holoscan.test.TestingPod";
  static constexpr std::uint32_t version = 1U;
};
template <>
struct ChannelTypeIdentity<TestingPodB> : ChannelTypeIdentity<TestingPodA> {};
template <>
struct ChannelPodLayout<TestingPodA> {
  static constexpr auto fields =
      pod_fields(pod_field("value", &TestingPodA::value, offsetof(TestingPodA, value)));
};
template <>
struct ChannelPodLayout<TestingPodB> {
  static constexpr auto fields =
      pod_fields(pod_field("value", &TestingPodB::value, offsetof(TestingPodB, value)));
};
}  // namespace holoscan
namespace {
class PodEcho final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in");
    spec.output(out, "out");
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto value = in.receive();
    if (!value)
      return holoscan::make_unexpected(std::move(value).error());
    return out.emit(std::move(*value));
  }
  holoscan::Input<TestingPodA> in;
  holoscan::Output<TestingPodA> out;
};
}  // namespace

namespace {
enum class IdentityEnumA : std::uint32_t { kValue };
enum class IdentityEnumB : std::uint32_t { kValue };
class IdentityEnumOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in"); }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }
  holoscan::Input<IdentityEnumA> in;
};
}  // namespace

namespace {
// Named declarations are covered by PodEcho; this fixture covers constructor-named ports and
// prototype replacement without involving Runtime execution.
template <holoscan::PublishedPayload T>
class NativeWitnessOp final : public holoscan::Operator<ScaleParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in);
    spec.output(out);
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }
  holoscan::Input<T> in{"in"};
  holoscan::Output<T> out{"out"};
};

const void* resolve_native_payload(const holoscan::authoring::OpaqueOperatorRef& op,
                                   std::string_view name, holoscan::PortDirection direction,
                                   const holoscan::PayloadIdentityRecord& identity,
                                   const void* local_token, const std::type_info* native_type) {
  return holoscan::detail::OperatorTestingAccess::resolve(
      op, name, direction, identity, local_token, native_type);
}
}  // namespace

// REQ: OPERATOR-064
TEST(OperatorTestHarness, NativePayloadWitnessAcceptsMatchingPodWithForeignToken) {
  holoscan::Graph graph{"native-pod-identity"};
  const auto op = graph.op<PodEcho>("pod");
  const auto erased = holoscan::detail::OperatorTestingAccess::erase(op);
  const auto identity = holoscan::detail::operator_spec_payload_identity<TestingPodA>();
  const char foreign_token{};
  EXPECT_NE(static_cast<const void*>(&foreign_token),
            static_cast<const void*>(&holoscan::detail::kOperatorSpecLocalTypeToken<TestingPodA>));

  EXPECT_EQ(resolve_native_payload(erased,
                                   "in",
                                   holoscan::PortDirection::kInput,
                                   identity,
                                   &foreign_token,
                                   &typeid(TestingPodA)),
            &op->in);
  EXPECT_EQ(resolve_native_payload(erased,
                                   "out",
                                   holoscan::PortDirection::kOutput,
                                   identity,
                                   &foreign_token,
                                   &typeid(TestingPodA)),
            &op->out);
  EXPECT_EQ(
      resolve_native_payload(
          erased, "in", holoscan::PortDirection::kInput, identity, nullptr, &typeid(TestingPodA)),
      &op->in);
  EXPECT_THROW(
      resolve_native_payload(
          erased, "in", holoscan::PortDirection::kInput, identity, &foreign_token, nullptr),
      holoscan::RuntimeError);
  // The original token-only overload retains its rejection of a foreign custom token.
  EXPECT_THROW(static_cast<void>(holoscan::detail::OperatorTestingAccess::resolve(
                   erased, "in", holoscan::PortDirection::kInput, identity, &foreign_token)),
               holoscan::RuntimeError);
  EXPECT_EQ(holoscan::detail::OperatorTestingAccess::resolve(
                erased,
                "in",
                holoscan::PortDirection::kInput,
                identity,
                &holoscan::detail::kOperatorSpecLocalTypeToken<TestingPodA>),
            &op->in);
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, NativePayloadWitnessAcceptsMatchingSchemaWithForeignToken) {
  holoscan::Graph graph{"native-schema-identity"};
  const auto op = graph.op<NativeWitnessOp<TestingSchemaA>>("schema");
  const auto erased = holoscan::detail::OperatorTestingAccess::erase(op);
  const auto identity = holoscan::detail::operator_spec_payload_identity<TestingSchemaA>();
  const char foreign_token{};
  EXPECT_EQ(resolve_native_payload(erased,
                                   "in",
                                   holoscan::PortDirection::kInput,
                                   identity,
                                   &foreign_token,
                                   &typeid(TestingSchemaA)),
            &op->in);
  EXPECT_EQ(resolve_native_payload(erased,
                                   "out",
                                   holoscan::PortDirection::kOutput,
                                   identity,
                                   &foreign_token,
                                   &typeid(TestingSchemaA)),
            &op->out);
  // Same schema family/file/profile is not exact C++ object identity.
  EXPECT_THROW(resolve_native_payload(erased,
                                      "in",
                                      holoscan::PortDirection::kInput,
                                      identity,
                                      &foreign_token,
                                      &typeid(TestingSchemaB)),
               holoscan::RuntimeError);
  auto changed_profile = identity;
  changed_profile.transport_profile_tag ^= 1U;
  EXPECT_THROW(resolve_native_payload(erased,
                                      "in",
                                      holoscan::PortDirection::kInput,
                                      changed_profile,
                                      &foreign_token,
                                      &typeid(TestingSchemaA)),
               holoscan::RuntimeError);
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, NativePayloadWitnessRejectsSameIdentityDifferentNativeTypes) {
  holoscan::Graph graph{"native-mismatch"};
  const auto op = graph.op<PodEcho>("pod");
  const auto erased = holoscan::detail::OperatorTestingAccess::erase(op);
  const char foreign_token{};
  const auto identity = holoscan::detail::operator_spec_payload_identity<TestingPodB>();
  // These two PODs intentionally have identical names, versions, object layouts and fields.
  // Reject the distinct native type before any Input<TestingPodB> cast could be attempted.
  try {
    static_cast<void>(resolve_native_payload(erased,
                                             "in",
                                             holoscan::PortDirection::kInput,
                                             identity,
                                             &foreign_token,
                                             &typeid(TestingPodB)));
    FAIL() << "Distinct native POD types must not resolve through semantic identity";
  } catch (const holoscan::RuntimeError& error) {
    const std::string message{error.what()};
    EXPECT_NE(message.find("GRAPH_PORT_TYPE_MISMATCH"), std::string::npos);
    EXPECT_NE(message.find("operator='pod'"), std::string::npos);
    EXPECT_NE(message.find("port='in'"), std::string::npos);
    EXPECT_NE(message.find("native payload type differs"), std::string::npos);
    EXPECT_NE(message.find("requested=input com.nvidia.holoscan.test.TestingPod"),
              std::string::npos);
    EXPECT_NE(message.find("declared=input com.nvidia.holoscan.test.TestingPod"),
              std::string::npos);
  }
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, NativePayloadWitnessPreservesIdentityDirectionAndEnumGuards) {
  holoscan::Graph graph{"native-identity-guards"};
  const auto op = graph.op<PodEcho>("pod");
  const auto erased = holoscan::detail::OperatorTestingAccess::erase(op);
  const char foreign_token{};
  const auto identity = holoscan::detail::operator_spec_payload_identity<TestingPodA>();
  auto different_version = identity;
  if (auto& pod = different_version.pod; pod.has_value()) {
    ++pod->version;
  } else {
    FAIL() << "POD payload identity is missing";
  }
  EXPECT_THROW(resolve_native_payload(erased,
                                      "in",
                                      holoscan::PortDirection::kInput,
                                      different_version,
                                      &foreign_token,
                                      &typeid(TestingPodA)),
               holoscan::RuntimeError);
  EXPECT_THROW(resolve_native_payload(erased,
                                      "in",
                                      holoscan::PortDirection::kOutput,
                                      identity,
                                      &foreign_token,
                                      &typeid(TestingPodA)),
               holoscan::RuntimeError);
  EXPECT_THROW(resolve_native_payload(erased,
                                      "typo",
                                      holoscan::PortDirection::kInput,
                                      identity,
                                      &foreign_token,
                                      &typeid(TestingPodA)),
               holoscan::RuntimeError);
  const auto enum_op = graph.op<IdentityEnumOp>("enum");
  const auto enum_erased = holoscan::detail::OperatorTestingAccess::erase(enum_op);
  const auto enum_identity = holoscan::detail::operator_spec_payload_identity<IdentityEnumA>();
  EXPECT_THROW(resolve_native_payload(enum_erased,
                                      "in",
                                      holoscan::PortDirection::kInput,
                                      enum_identity,
                                      &foreign_token,
                                      &typeid(IdentityEnumA)),
               holoscan::RuntimeError);
  EXPECT_THROW(resolve_native_payload(enum_erased,
                                      "in",
                                      holoscan::PortDirection::kInput,
                                      enum_identity,
                                      nullptr,
                                      &typeid(IdentityEnumA)),
               holoscan::RuntimeError);
  EXPECT_EQ(resolve_native_payload(enum_erased,
                                   "in",
                                   holoscan::PortDirection::kInput,
                                   enum_identity,
                                   &holoscan::detail::kOperatorSpecLocalTypeToken<IdentityEnumA>,
                                   &typeid(IdentityEnumA)),
            &enum_op->in);
  const auto scale = graph.op<ScaleOp>("scale");
  const auto scalar_erased = holoscan::detail::OperatorTestingAccess::erase(scale);
  EXPECT_EQ(resolve_native_payload(scalar_erased,
                                   "in",
                                   holoscan::PortDirection::kInput,
                                   holoscan::detail::operator_spec_payload_identity<std::int32_t>(),
                                   &foreign_token,
                                   nullptr),
            &scale->in);
  EXPECT_THROW(
      resolve_native_payload(scalar_erased,
                             "in",
                             holoscan::PortDirection::kInput,
                             holoscan::detail::operator_spec_payload_identity<std::int64_t>(),
                             &foreign_token,
                             &typeid(std::int32_t)),
      holoscan::RuntimeError);
  const auto expired = [] {
    holoscan::Graph temporary{"expired-native-witness"};
    return holoscan::detail::OperatorTestingAccess::erase(temporary.op<PodEcho>("pod"));
  }();
  EXPECT_THROW(resolve_native_payload(expired,
                                      "in",
                                      holoscan::PortDirection::kInput,
                                      identity,
                                      &foreign_token,
                                      &typeid(TestingPodA)),
               holoscan::RuntimeError);
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, NativePayloadWitnessSurvivesParameterReplacement) {
  holoscan::Graph graph{"native-witness-params"};
  auto op = graph.op<NativeWitnessOp<TestingPodA>>("pod");
  const auto erased = holoscan::detail::OperatorTestingAccess::erase(op);
  const auto identity = holoscan::detail::operator_spec_payload_identity<TestingPodA>();
  const char foreign_token{};
  EXPECT_EQ(resolve_native_payload(erased,
                                   "in",
                                   holoscan::PortDirection::kInput,
                                   identity,
                                   &foreign_token,
                                   &typeid(TestingPodA)),
            &op->in);
  op.set_params(ScaleParams{.factor = 2});
  EXPECT_EQ(resolve_native_payload(erased,
                                   "in",
                                   holoscan::PortDirection::kInput,
                                   identity,
                                   &foreign_token,
                                   &typeid(TestingPodA)),
            &op->in);
  EXPECT_EQ(resolve_native_payload(erased,
                                   "out",
                                   holoscan::PortDirection::kOutput,
                                   identity,
                                   &foreign_token,
                                   &typeid(TestingPodA)),
            &op->out);
}

namespace {
// Honest same-payload metadata does not imply the concrete ValueInputScript implementation.
class AlternatePodScript final : public holoscan::testing::detail::InputScript {
 public:
  [[nodiscard]] std::size_t size() const noexcept override { return 1U; }
  [[nodiscard]] holoscan::PayloadIdentityRecord identity() const override {
    return holoscan::detail::operator_spec_payload_identity<TestingPodA>();
  }
  [[nodiscard]] const void* type_token() const noexcept override {
    return holoscan::testing::detail::capture_type_token<TestingPodA>();
  }
  [[nodiscard]] const std::type_info* payload_type_info() const noexcept override {
    return &typeid(TestingPodA);
  }
  [[nodiscard]] std::unique_ptr<holoscan::testing::detail::InputPortWriter> make_writer()
      const override {
    throw std::logic_error{"alternate script must be rejected before writer construction"};
  }
  [[nodiscard]] std::shared_ptr<const holoscan::testing::detail::InputScript> appended(
      const holoscan::testing::detail::InputScript&) const override {
    throw std::logic_error{"alternate script append is not exercised"};
  }
  holoscan::authoring::OpaqueOperatorRef build(
      holoscan::Graph&, const holoscan::authoring::OpaqueOperatorRef&, std::string_view,
      std::string_view, std::optional<holoscan::ConnectionOptions>,
      std::optional<holoscan::TensorOutputSpec>) const override {
    throw std::logic_error{"alternate script must be rejected before helper construction"};
  }
};
}  // namespace

// REQ: OPERATOR-064
TEST(OperatorTestHarness, AppendRejectsAlternateConcreteAdapterWithSamePayloadWitness) {
  const holoscan::testing::detail::ValueInputScript<TestingPodA> original{
      std::vector<holoscan::testing::TestSample<TestingPodA>>{{.data = {.value = 7U}}}};
  const AlternatePodScript alternate;
  ASSERT_EQ(original.type_token(), alternate.type_token());
  ASSERT_EQ(*original.payload_type_info(), *alternate.payload_type_info());
  EXPECT_THROW(static_cast<void>(original.appended(alternate)), std::invalid_argument);
  EXPECT_EQ(original.size(), 1U);
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, PublicOpaqueHandlesOwnReusableInputAndCaptureRequests) {
  using namespace holoscan::testing;
  static_assert(std::copy_constructible<StagedInput> && std::movable<StagedInput>);
  static_assert(std::copy_constructible<CaptureRequest> && std::movable<CaptureRequest>);
  NativeOperatorTest fixture{[](holoscan::Graph& graph) {
    return holoscan::detail::OperatorTestingAccess::erase(graph.op<PodEcho>("pod"));
  }};
  auto input = staged_values<TestingPodA>({TestingPodA{.value = 7U}});
  const auto request = value_capture<TestingPodA>(4U);
  fixture.stage_input("in", input);
  EXPECT_THROW(fixture.stage_input("in", StagedInput{}), std::invalid_argument);
  EXPECT_THROW(fixture.declare_capture("out", CaptureRequest{}), std::invalid_argument);
  EXPECT_THROW(fixture.stage_input("in", staged_values<TestingPodB>({TestingPodB{.value = 99U}})),
               holoscan::RuntimeError);
  fixture.stage_input("in", input);
  fixture.declare_capture("out", request);
  const auto first = fixture.run();
  ASSERT_TRUE(first.ok()) << first.describe_failure();
  const auto values = first.values<TestingPodA>("out");
  ASSERT_EQ(values.size(), 2U);
  EXPECT_EQ(values[0].value, 7U);
  EXPECT_EQ(values[1].value, 7U);
  try {
    static_cast<void>(first.samples<TestingPodB>("out"));
    FAIL() << "Wrong concrete capture type must reject";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string{error.what()}.find("output='out'"), std::string::npos);
  }
  EXPECT_THROW(static_cast<void>(first.snapshots<holoscan::Tensor>("out")), std::invalid_argument);
  fixture.stage_input("in", std::move(input));
  fixture.declare_capture("out", request);
  // Intentional contract probe: moved-from StagedInput must be rejected as an empty handle.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_THROW(fixture.stage_input("in", input), std::invalid_argument);
  const auto second = fixture.run();
  ASSERT_TRUE(second.ok()) << second.describe_failure();
  EXPECT_EQ(second.value<TestingPodA>("out").value, 7U);
  EXPECT_EQ(first.values<TestingPodA>("out").size(), 2U);
  EXPECT_NO_THROW(static_cast<void>(staged_values<TestingPodA>({})));
  EXPECT_THROW(static_cast<void>(value_capture<TestingPodA>(0U)), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(snapshot_capture<holoscan::Tensor>(1U, 0U)),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(retain_capture<holoscan::Tensor>(0U)), std::invalid_argument);
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, NamedStagingRejectsWrongNativeTypeBeforeChangingPendingBatch) {
  // Simulate distinct module-local address tokens: only closed SDK built-ins may match.
  holoscan::Graph identity_graph{"identity"};
  const auto scale = identity_graph.op<ScaleOp>("scale");
  const auto erased = holoscan::detail::OperatorTestingAccess::erase(scale);
  const char foreign_token{};
  const auto resolve = [&](auto identity, holoscan::PortDirection direction) {
    return holoscan::detail::OperatorTestingAccess::resolve(
        erased, "in", direction, identity, &foreign_token);
  };
  EXPECT_EQ(resolve(holoscan::detail::operator_spec_payload_identity<std::int32_t>(),
                    holoscan::PortDirection::kInput),
            &scale->in);
  EXPECT_THROW(resolve(holoscan::detail::operator_spec_payload_identity<std::int64_t>(),
                       holoscan::PortDirection::kInput),
               holoscan::RuntimeError);
  EXPECT_THROW(resolve(holoscan::detail::operator_spec_payload_identity<holoscan::Tensor>(),
                       holoscan::PortDirection::kInput),
               holoscan::RuntimeError);
  EXPECT_THROW(resolve(holoscan::detail::operator_spec_payload_identity<std::int32_t>(),
                       holoscan::PortDirection::kOutput),
               holoscan::RuntimeError);
  const auto enum_op =
      holoscan::detail::OperatorTestingAccess::erase(identity_graph.op<IdentityEnumOp>("enum"));
  EXPECT_THROW(static_cast<void>(holoscan::detail::OperatorTestingAccess::resolve(
                   enum_op,
                   "in",
                   holoscan::PortDirection::kInput,
                   holoscan::detail::operator_spec_payload_identity<IdentityEnumB>(),
                   &holoscan::detail::kOperatorSpecLocalTypeToken<IdentityEnumB>)),
               holoscan::RuntimeError);
  EXPECT_THROW(static_cast<void>(holoscan::detail::OperatorTestingAccess::resolve(
                   enum_op,
                   "in",
                   holoscan::PortDirection::kInput,
                   holoscan::detail::operator_spec_payload_identity<IdentityEnumA>(),
                   nullptr)),
               holoscan::RuntimeError);

  auto h = holoscan::testing::create_operator_test<PodEcho>();
  h.push("in", TestingPodA{.value = 1U});
  EXPECT_THROW(h.push("in", TestingPodB{.value = 2U}), holoscan::RuntimeError);
  EXPECT_THROW(h.push("typo", TestingPodA{.value = 3U}), holoscan::RuntimeError);
  h.capture<TestingPodA>("out");
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<TestingPodA>("out").value, 1U);
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, NullMemberSelectorIsRejectedWithoutChangingPendingInput) {
  auto h = holoscan::testing::create_operator_test<ScaleOp>();
  h.push(&ScaleOp::in, std::int32_t{21});
  holoscan::Input<std::int32_t> ScaleOp::* const missing = nullptr;
  EXPECT_THROW(h.push(missing, std::int32_t{99}), std::invalid_argument);
  h.capture<std::int32_t>("out");
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<std::int32_t>("out"), 21);
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, NativeFactoryRejectsForeignAndExpiredAuthoringHandles) {
  holoscan::Graph other{"other"};
  const auto handle = other.op<ScaleOp>("scale");
  const auto foreign = holoscan::detail::OperatorTestingAccess::erase(handle);
  holoscan::testing::NativeOperatorTest h{[foreign](auto&) { return foreign; }};
  EXPECT_THROW(static_cast<void>(h.run()), holoscan::RuntimeError);
  auto expired = [] {
    holoscan::Graph temporary{"temporary"};
    return holoscan::detail::OperatorTestingAccess::erase(temporary.op<ScaleOp>("scale"));
  }();
  h.set_operator_factory([expired](auto&) { return expired; });
  EXPECT_THROW(static_cast<void>(h.run()), holoscan::RuntimeError);
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, RolledBackTestingHandlesRejectSameIndexReplacements) {
  using holoscan::detail::OperatorTestingAccess;
  holoscan::Graph graph{"testing-handle-rollback"};
  struct Context {
    holoscan::Graph& graph;
    std::optional<holoscan::authoring::OpaqueOperatorRef> source;
    std::optional<holoscan::FlowHandle<std::int32_t>> flow;
    std::optional<holoscan::PartitionHandle> partition;
  } context{graph, {}, {}, {}};
  const auto failed = holoscan::detail::graph_apply_atomic_mutation(
      holoscan::detail::GraphPayloadRegistrationAccess::state(graph).lock(),
      &context,
      [](void* opaque) -> holoscan::expected<void, holoscan::Error> {
        auto& call = *static_cast<Context*>(opaque);
        const auto source = call.graph.op<ScaleOp>("source");
        const auto sink = call.graph.op<ScaleOp>("sink");
        call.source = OperatorTestingAccess::erase(source);
        call.graph.add_flow(source->out, sink->in);
        call.flow = OperatorTestingAccess::flow<std::int32_t>(
            call.graph, *call.source, "out", OperatorTestingAccess::erase(sink), "in");
        call.partition = call.graph.partition("partition");
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, "injected authoring failure"});
      });
  ASSERT_FALSE(failed);
  if (!context.source || !context.flow || !context.partition) {
    ADD_FAILURE() << "Failed construction did not retain all testing handles";
    return;
  }
  const auto stale_source = context.source.value();
  const auto stale_flow = context.flow.value();
  auto stale_partition = context.partition.value();
  const auto source = graph.op<ScaleOp>("replacement-source");
  const auto sink = graph.op<ScaleOp>("replacement-sink");
  graph.add_flow(source->out, sink->in);
  auto partition = graph.partition("replacement-partition");
  const std::array members{OperatorTestingAccess::erase(source)};
  const auto before = graph.view();
  EXPECT_THROW(OperatorTestingAccess::input_queue_depth(stale_source, "in", 9U),
               holoscan::RuntimeError);
  EXPECT_THROW(static_cast<void>(OperatorTestingAccess::resolve(
                   stale_source,
                   "out",
                   holoscan::PortDirection::kOutput,
                   holoscan::detail::operator_spec_payload_identity<std::int32_t>(),
                   &holoscan::detail::kOperatorSpecLocalTypeToken<std::int32_t>,
                   &typeid(std::int32_t))),
               holoscan::RuntimeError);
  EXPECT_THROW(OperatorTestingAccess::add_to(stale_partition, members), holoscan::RuntimeError);
  EXPECT_FALSE(stale_flow.graph_valid());
  EXPECT_THROW(graph.configure(stale_flow).queue_depth(9U), holoscan::RuntimeError);
  EXPECT_EQ(graph.view().identity(), before.identity());
  EXPECT_EQ(graph.view().generation(), before.generation());
  const auto live = OperatorTestingAccess::flow<std::int32_t>(
      graph, OperatorTestingAccess::erase(source), "out", OperatorTestingAccess::erase(sink), "in");
  EXPECT_TRUE(live.graph_valid());
  EXPECT_NO_THROW(OperatorTestingAccess::add_to(partition, members));
  EXPECT_NO_THROW(graph.configure(live).queue_depth(3U));
}

// REQ: OPERATOR-073
TEST(OperatorTestHarness, StagedTensorPromisesItsObservedShapeToBoundedConsumer) {
  auto h = holoscan::testing::create_operator_test<HostTensorPass>();
  const std::array<std::int64_t, 1> shape{4};
  h.input_tensor_spec(
      "in",
      holoscan::TensorInputSpec{.representation = {.dtype = DLDataType{kDLInt, 32, 1}, .rank = 1U},
                                .bounds = holoscan::tensor_bounds(shape)});
  h.push("in", input_tensor(10));
  h.capture_snapshot<holoscan::Tensor>("out");
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.snapshot<holoscan::Tensor>("out").shape(), (std::vector<std::int64_t>{4}));
}

#if defined(__linux__)
namespace {
struct CpuObservation {
  int cpu{-1};
  int policy{-1};
  int priority{-1};
};
class CpuObservedOp final : public holoscan::Operator<> {
 public:
  explicit CpuObservedOp(std::shared_ptr<CpuObservation> observation)
      : observation_(std::move(observation)) {}
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in"); }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    observation_->cpu = sched_getcpu();
    observation_->policy = sched_getscheduler(0);
    sched_param parameter{};
    if (sched_getparam(0, &parameter) != 0)
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
    observation_->priority = parameter.sched_priority;
    return {};
  }
  holoscan::Input<std::int32_t> in;

 private:
  std::shared_ptr<CpuObservation> observation_;
};
}  // namespace
#endif

// REQ: OPERATOR-072
TEST(OperatorTestHarness, NativeThreadPlacementIsObservedInsideTheActualComputeCallback) {
#if defined(__linux__)
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  ASSERT_EQ(sched_getaffinity(0, sizeof(allowed), &allowed), 0);
  std::uint32_t cpu{};
  for (; cpu < CPU_SETSIZE; ++cpu)
    if (CPU_ISSET(cpu, &allowed))
      break;
  ASSERT_LT(cpu, CPU_SETSIZE);
  auto observation = std::make_shared<CpuObservation>();
  auto h = holoscan::testing::create_operator_test<CpuObservedOp>(observation);
  h.push("in", std::int32_t{1});
  h.configure_graph([](auto& f) { f.graph().partition("worker").include_all(); });
  holoscan::CompileOptions options;
  options.deployment.bind_thread_placement(
      "worker", holoscan::ThreadPlacement{.affinity = holoscan::CPUAffinity{cpu}});
  h.set_compile_options(options);
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(observation->cpu, static_cast<int>(cpu));
  EXPECT_EQ(observation->policy, SCHED_OTHER);
  EXPECT_EQ(observation->priority, 0);
  ASSERT_EQ(result.plan().partition_placements().size(), 1U);
  const auto& accepted = result.plan().partition_placements()[0].accepted;
  if (accepted && accepted->affinity) {
    EXPECT_EQ(accepted->affinity->cores, std::vector<std::uint32_t>{cpu});
  } else {
    FAIL() << "Expected an accepted CPU affinity request";
  }
  cpu_set_t after;
  CPU_ZERO(&after);
  ASSERT_EQ(sched_getaffinity(0, sizeof(after), &after), 0);
  EXPECT_TRUE(CPU_EQUAL(&allowed, &after));
#else
  GTEST_SKIP() << "Linux affinity readback is unavailable on this platform";
#endif
}

// REQ: OPERATOR-072
TEST(OperatorTestHarness, NativeHookResolvesTypedFlowHandlesAndKeepsGraphLifetime) {
  auto h = holoscan::testing::create_operator_test<ScaleOp>();
  h.push_many<std::int32_t>("in", {1, 2});
  h.capture<std::int32_t>("out");
  std::optional<holoscan::FlowHandle<std::int32_t>> saved;
  h.configure_graph([&](auto& f) {
    const auto incoming = f.template input_flow<std::int32_t>("in");
    const auto outgoing = f.template output_flow<std::int32_t>("out");
    EXPECT_TRUE(incoming.graph_valid());
    EXPECT_EQ(incoming.target().name(), "in");
    EXPECT_EQ(outgoing.source().name(), "out");
    EXPECT_THROW(static_cast<void>(f.template input_flow<float>("in")), holoscan::RuntimeError);
    f.graph().configure(incoming).queue_depth(4U);
    f.graph().configure(outgoing).queue_depth(4U);
    saved = incoming;
  });
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.values<std::int32_t>("out"), (std::vector<std::int32_t>{1, 2}));
  if (saved) {
    EXPECT_FALSE(saved->graph_valid());
  } else {
    FAIL() << "Expected the configured flow handle";
  }
}

namespace {
class RestartRequestOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in");
    spec.output(out, "out");
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    auto value = in.receive();
    if (!value)
      return holoscan::make_unexpected(std::move(value).error());
    const auto stage =
        value->data == 0 ? holoscan::LifecycleStage::kStart : holoscan::LifecycleStage::kStop;
    auto receipt = context.request_restart({.from_stage = stage});
    if (receipt)
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "unexpected restart admission"});
    return out.emit(static_cast<std::int32_t>(receipt.error().code));
  }
  holoscan::Input<std::int32_t> in;
  holoscan::Output<std::int32_t> out;
};

class UnboundedClockOp final : public holoscan::Operator<> {
 public:
  explicit UnboundedClockOp(std::shared_ptr<LifecycleProbe> probe, bool hang = false)
      : probe_(std::move(probe)), hang_(hang) {}
  void setup(holoscan::OperatorSpec&) override {}
  holoscan::Contract contract() const override {
    holoscan::Contract c;
    c.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{100}});
    return c;
  }
  void start() override { ++probe_->starts; }
  void stop() override { ++probe_->stops; }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    ++probe_->computes;
    if (hang_) {
      std::promise<void> never;
      never.get_future().wait();
    }
    return {};
  }

 private:
  std::shared_ptr<LifecycleProbe> probe_;
  bool hang_{};
};
}  // namespace

// REQ: OPERATOR-064
TEST(OperatorTestHarness, DefaultPolicyPreservesDisabledRestartAndInvalidStagePrecedence) {
  auto h = holoscan::testing::create_operator_test<RestartRequestOp>();
  h.push_many("in", std::vector<std::int32_t>{0, 1});
  h.capture<std::int32_t>("out");
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.values<std::int32_t>("out"),
            (std::vector<std::int32_t>{
                static_cast<std::int32_t>(holoscan::ErrorCode::kLifecycleTransactionsDisabled),
                static_cast<std::int32_t>(holoscan::ErrorCode::kLifecycleInvalidRestartStage)}));
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, CooperativeWatchdogReportsFailureAfterJoinedCleanup) {
  auto probe = std::make_shared<LifecycleProbe>();
  auto h = holoscan::testing::create_operator_test<UnboundedClockOp>(probe);
  const auto result =
      h.run(holoscan::testing::TestRunOptions{.timeout = std::chrono::milliseconds{500}});
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.termination(), holoscan::RunTermination::kStopped);
  EXPECT_NE(result.describe_failure().find("watchdog expired"), std::string::npos);
  // Startup may not run before the deadline on a heavily loaded host. Cleanup still joins.
  EXPECT_EQ(probe->stops, probe->starts);
}

#if defined(__linux__)
namespace {
// Both gates have cleanup owners constructed after the async future. An early return
// therefore opens both gates before joining, including when the hook has not entered yet.
bool hook_cleanup_after_failed_rendezvous(bool late_entry) {
  using namespace std::chrono_literals;
  using namespace holoscan::testing;
  NativeOperatorTest h{[](holoscan::Graph& graph) {
    return holoscan::detail::OperatorTestingAccess::erase(graph.op<ScaleOp>("operator"));
  }};
  h.stage_input("in", staged_values<std::int32_t>({7}));
  h.declare_capture("out", value_capture<std::int32_t>(1U));
  std::promise<void> entered, release, allow_entry;
  auto entry = entered.get_future();
  auto gate = release.get_future().share();
  auto factory_gate = allow_entry.get_future().share();
  h.configure_graph([&](auto&) {
    entered.set_value();
    gate.wait();
  });
  h.set_operator_factory([&](holoscan::Graph& graph) {
    if (!late_entry)
      throw std::runtime_error{"before graph hook"};
    factory_gate.wait();
    return holoscan::detail::OperatorTestingAccess::erase(graph.op<ScaleOp>("operator"));
  });
  auto running = std::async(std::launch::async, [&] { return h.run({.timeout = 1s}); });
  ReleaseHook allow_factory{allow_entry};
  {
    const ReleaseHook release_hook{release};
    if (entry.wait_for(0ms) != std::future_status::timeout)
      return false;
  }
  // The scope-exit path must open the release gate even before a late hook arrives.
  if (gate.wait_for(0ms) != std::future_status::ready)
    return false;
  allow_factory.open();
  if (running.wait_for(5s) != std::future_status::ready)
    return false;
  try {
    const auto result = running.get();
    return late_entry && entry.wait_for(0ms) == std::future_status::ready && result.ok() &&
           result.value<std::int32_t>("out") == 7;
  } catch (const std::runtime_error& error) {
    return !late_entry && std::string(error.what()) == "before graph hook" &&
           entry.wait_for(0ms) == std::future_status::timeout;
  }
}
}  // namespace

// REQ: OPERATOR-064
TEST(OperatorTestHarness, FailedHookRendezvousReleasesEarlyFailureAndLateEntry) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        std::signal(SIGALRM, SIG_DFL);
        alarm(15);
        _exit(hook_cleanup_after_failed_rendezvous(false) &&
                      hook_cleanup_after_failed_rendezvous(true)
                  ? 0
                  : 1);
      },
      ::testing::ExitedWithCode(0),
      "");
}
#endif

#if defined(__linux__)
// REQ: OPERATOR-064
TEST(OperatorTestHarness, HardHungCallbackRemainsConfinedToAnExternallyTerminatedProcess) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        std::signal(SIGALRM, SIG_DFL);
        alarm(2);
        auto probe = std::make_shared<LifecycleProbe>();
        auto h = holoscan::testing::create_operator_test<UnboundedClockOp>(probe, true);
        static_cast<void>(h.run());
        _exit(1);
      },
      ::testing::KilledBySignal(SIGALRM),
      "");
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, BlockingFacadePreservesCallerSignalMaskAndHandlers) {
  struct sigaction before_int{}, before_term{}, after_int{}, after_term{};
  sigset_t before_mask{}, after_mask{};
  ASSERT_EQ(sigaction(SIGINT, nullptr, &before_int), 0);
  ASSERT_EQ(sigaction(SIGTERM, nullptr, &before_term), 0);
  ASSERT_EQ(pthread_sigmask(SIG_SETMASK, nullptr, &before_mask), 0);
  auto h = holoscan::testing::create_operator_test<ScaleOp>();
  h.push("in", std::int32_t{21});
  h.capture<std::int32_t>("out");
  ASSERT_TRUE(h.run().ok());
  ASSERT_EQ(sigaction(SIGINT, nullptr, &after_int), 0);
  ASSERT_EQ(sigaction(SIGTERM, nullptr, &after_term), 0);
  ASSERT_EQ(pthread_sigmask(SIG_SETMASK, nullptr, &after_mask), 0);
  EXPECT_EQ(before_int.sa_handler, after_int.sa_handler);
  EXPECT_EQ(before_term.sa_handler, after_term.sa_handler);
  EXPECT_EQ(before_int.sa_flags, after_int.sa_flags);
  EXPECT_EQ(before_term.sa_flags, after_term.sa_flags);
  for (int signal = 1; signal < NSIG; ++signal) {
    EXPECT_EQ(sigismember(&before_mask, signal), sigismember(&after_mask, signal));
  }
}
#endif

namespace {
class MatrixPass final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in");
    spec.output(out, "out")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 8, 1},
                               .rank = 2U},
            .bounds = holoscan::tensor_bounds(16U)});
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = in.receive();
    if (!sample)
      return holoscan::make_unexpected(std::move(sample).error());
    return out.emit(std::move(*sample));
  }
  holoscan::Input<holoscan::Tensor> in;
  holoscan::Output<holoscan::Tensor> out;
};
holoscan::Tensor matrix_tensor(std::array<std::int64_t, 2> shape) {
  struct Storage {
    DLManagedTensor managed{};
    std::array<std::int64_t, 2> shape;
    std::array<std::uint8_t, 4> bytes{1, 2, 3, 4};
  };
  auto storage = std::make_unique<Storage>();
  storage->shape = shape;
  storage->managed.dl_tensor = {.data = storage->bytes.data(),
                                .device = {kDLCPU, 0},
                                .ndim = 2,
                                .dtype = {kDLUInt, 8, 1},
                                .shape = storage->shape.data(),
                                .strides = nullptr,
                                .byte_offset = 0};
  storage->managed.manager_ctx = storage.get();
  storage->managed.deleter = [](DLManagedTensor* value) {
    delete static_cast<Storage*>(value->manager_ctx);
  };
  return holoscan::Tensor{&storage.release()->managed};
}
}  // namespace

// REQ: OPERATOR-073
TEST(OperatorTestHarness, CrossedDynamicExtentsProduceAConsistentBoundingShapeAndByteSpan) {
  auto h = holoscan::testing::create_operator_test<MatrixPass>();
  h.push_many("in", std::vector<holoscan::Tensor>{matrix_tensor({1, 4}), matrix_tensor({4, 1})});
  h.capture_snapshot<holoscan::Tensor>("out");
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto snapshots = result.snapshots<holoscan::Tensor>("out");
  ASSERT_EQ(snapshots.size(), 2U);
  EXPECT_EQ(snapshots[0].shape(), (std::vector<std::int64_t>{1, 4}));
  EXPECT_EQ(snapshots[1].shape(), (std::vector<std::int64_t>{4, 1}));
  EXPECT_EQ(snapshots[0].bytes(),
            (std::vector<std::byte>{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}}));
  EXPECT_EQ(snapshots[0].bytes(), snapshots[1].bytes());
}

namespace {
holoscan::Tensor writable_input_tensor() {
  const std::array<std::int64_t, 1> shape{4};
  auto tensor = holoscan::make_tensor(shape,
                                      DLDataType{kDLInt, 32, 1},
                                      std::nullopt,
                                      holoscan::MemoryStorageType::kSystem,
                                      std::make_shared<holoscan::UnboundedAllocator>());
  if (!tensor)
    throw holoscan::RuntimeError{std::move(tensor).error()};
  auto pointer = tensor->mutable_data();
  if (!pointer)
    throw holoscan::RuntimeError{std::move(pointer).error()};
  std::memset(*pointer, 0, 16);
  return std::move(*tensor);
}
}  // namespace

// REQ: OPERATOR-073
TEST(OperatorTestHarness, StagingReadLeaseProtectsTensorThroughRetainedPlanReplay) {
  auto tensor = writable_input_tensor();
  ASSERT_TRUE(tensor.mutable_data());
  std::optional<holoscan::testing::TestResult> result;
  {
    auto h = holoscan::testing::create_operator_test<HostTensorPass>();
    h.push("in", tensor);
    ASSERT_FALSE(tensor.mutable_data());
    h.capture_snapshot<holoscan::Tensor>("out");
    result = h.run();
    ASSERT_TRUE(result->ok()) << result->describe_failure();
  }
  EXPECT_FALSE(tensor.mutable_data());
  {
    auto replay = holoscan::run_async(result->plan());
    replay.wait();
    EXPECT_EQ(replay.termination(), holoscan::RunTermination::kCompleted);
    result.reset();
    // A separate RunSession still owns the replayable plan until destruction.
    EXPECT_FALSE(tensor.mutable_data());
  }
  EXPECT_TRUE(tensor.mutable_data());
}

// REQ: OPERATOR-073
TEST(OperatorTestHarness, FailedLastTensorLeaseLeavesEarlierPendingBatchAndNoPartialClaims) {
  auto first = writable_input_tensor();
  auto second = writable_input_tensor();
  auto writer = second.to_dlpack();
  ASSERT_TRUE(writer);
  auto h = holoscan::testing::create_operator_test<HostTensorPass>();
  h.push("in", input_tensor(1));
  EXPECT_THROW(h.push_many("in", std::vector<holoscan::Tensor>{first, second}),
               holoscan::RuntimeError);
  EXPECT_TRUE(first.mutable_data());
  (*writer)->deleter(*writer);
  EXPECT_TRUE(second.mutable_data());
  h.capture_snapshot<holoscan::Tensor>("out");
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(result.snapshots<holoscan::Tensor>("out").size(), 1U);
}

namespace {
// A storage-adapter probe, not a Runtime-qualified generated schema.
struct HandoffCopyProbe {
  std::uint32_t value{};
  bool throw_on_copy{};
  HandoffCopyProbe() = default;
  HandoffCopyProbe(std::uint32_t value, bool fail) : value(value), throw_on_copy(fail) {}
  HandoffCopyProbe(const HandoffCopyProbe& other)
      : value(other.value), throw_on_copy(other.throw_on_copy) {
    if (throw_on_copy)
      throw std::runtime_error{"handoff copied a payload"};
  }
  HandoffCopyProbe& operator=(const HandoffCopyProbe&) = default;
  HandoffCopyProbe(HandoffCopyProbe&&) noexcept = default;
  HandoffCopyProbe& operator=(HandoffCopyProbe&&) noexcept = default;
};
}  // namespace
namespace holoscan {
template <>
consteval SchemaIdentity schema_identity<HandoffCopyProbe>() {
  return make_schema_identity(make_schema_family_identity("holoscan.test.HandoffCopyProbe"),
                              make_flatbuffer_file_identifier("HOCP"));
}
}  // namespace holoscan

// REQ: OPERATOR-064
TEST(OperatorTestHarness, CaptureHandoffPreservesPrefixAndFailureWithoutCopyingPayloads) {
  using Storage = holoscan::testing::detail::ValueCaptureStorage<HandoffCopyProbe>;
  Storage source;
  source.samples.emplace_back(HandoffCopyProbe{42, true}, holoscan::SampleMetadata{.sequence = 9});
  source.failure.emplace(holoscan::testing::CaptureFailure{
      .metadata = {.sequence = 10},
      .error = holoscan::Error{holoscan::ErrorCode::kResourceExhausted, "full"}});
  EXPECT_THROW(static_cast<void>(Storage{source}), std::runtime_error);
  auto destination = source.empty_storage();
  static_assert(noexcept(source.transfer_to(*destination)));
  source.transfer_to(*destination);
  const auto& captured = static_cast<const Storage&>(*destination);
  ASSERT_EQ(captured.samples.size(), 1U);
  EXPECT_EQ(captured.samples[0].data.value, 42U);
  EXPECT_EQ(captured.samples[0].metadata.sequence, 9U);
  if (captured.failure) {
    EXPECT_EQ(captured.failure->metadata.sequence, 10U);
    EXPECT_EQ(captured.failure->error.code, holoscan::ErrorCode::kResourceExhausted);
  } else {
    FAIL() << "Expected the transferred capture failure";
  }
  EXPECT_TRUE(source.samples.empty());
  EXPECT_FALSE(source.failure);
}

namespace {
class FirstStartFailsOp final : public holoscan::Operator<> {
 public:
  explicit FirstStartFailsOp(std::shared_ptr<std::atomic<bool>> first) : first_(std::move(first)) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in");
    spec.output(out, "out");
  }
  void start() override {
    if (first_->exchange(false))
      throw std::runtime_error{"first startup fails"};
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto value = in.receive();
    if (!value)
      return holoscan::make_unexpected(std::move(value).error());
    return out.emit(std::move(*value));
  }
  holoscan::Input<std::int32_t> in;
  holoscan::Output<std::int32_t> out;

 private:
  std::shared_ptr<std::atomic<bool>> first_;
};
}  // namespace

// REQ: OPERATOR-064
TEST(OperatorTestHarness, StartupFailureResultCannotBeMutatedByTheFirstSuccessfulPlanReplay) {
  auto first = std::make_shared<std::atomic<bool>>(true);
  auto h = holoscan::testing::create_operator_test<FirstStartFailsOp>(first);
  h.push("in", std::int32_t{42});
  h.capture<std::int32_t>("out");
  const auto result = h.run();
  ASSERT_FALSE(result.ok());
  ASSERT_TRUE(result.values<std::int32_t>("out").empty());
  auto replay = holoscan::run_async(result.plan());
  replay.wait();
  EXPECT_EQ(replay.termination(), holoscan::RunTermination::kCompleted);
  EXPECT_TRUE(result.values<std::int32_t>("out").empty());
}

namespace {
class ConstructorNamedScale final : public holoscan::Operator<ScaleParams> {
 public:
  explicit ConstructorNamedScale(std::string name) : name_(std::move(name)) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, params().factor == 7 ? "reshaped" : name_);
    spec.output(out, "out");
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = in.receive();
    if (!sample)
      return holoscan::make_unexpected(std::move(sample).error());
    return out.emit(sample->data * params().factor);
  }
  holoscan::Input<std::int32_t> in;
  holoscan::Output<std::int32_t> out;

 private:
  std::string name_;
};
class MultipleOutputsOp final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in");
    spec.output(burst, "burst").max_emits_per_compute(2U);
    spec.output(positive, "positive");
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = in.receive();
    if (!sample)
      return holoscan::make_unexpected(std::move(sample).error());
    auto first = burst.emit(sample->data);
    if (!first)
      return first;
    auto second = burst.emit(sample->data + 10);
    if (!second)
      return second;
    if (sample->data > 0)
      return positive.emit(sample->data);
    return {};
  }
  holoscan::Input<std::int32_t> in;
  holoscan::Output<std::int32_t> burst;
  holoscan::Output<std::int32_t> positive;
};
}  // namespace

// REQ: OPERATOR-064
TEST(OperatorTestHarness, ConstructorSelectedPortAndParamsShapeRejectionPreservePendingInput) {
  auto h = holoscan::testing::create_operator_test<ConstructorNamedScale>(std::string{"selected"});
  h.set_params(ScaleParams{.factor = 2});
  h.push(&ConstructorNamedScale::in, std::int32_t{21});
  h.capture<std::int32_t>("out");
  h.set_params(ScaleParams{.factor = 7});
  try {
    static_cast<void>(h.run());
    FAIL() << "Params-driven port reshaping must reject";
  } catch (const holoscan::RuntimeError& error) {
    EXPECT_NE(std::string{error.what()}.find("GRAPH_PARAMETER_PORT_SHAPE_CHANGED"),
              std::string::npos);
  }
  h.set_params(ScaleParams{.factor = 3});
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<std::int32_t>("out"), 63);
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, OutputPortsHaveIndependentZeroOneAndBurstCardinality) {
  auto h = holoscan::testing::create_operator_test<MultipleOutputsOp>();
  h.push_many("in", std::vector<std::int32_t>{1, 0, 2});
  h.capture<std::int32_t>("burst");
  h.capture<std::int32_t>("positive");
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.values<std::int32_t>("burst"), (std::vector<std::int32_t>{1, 11, 0, 10, 2, 12}));
  EXPECT_EQ(result.values<std::int32_t>("positive"), (std::vector<std::int32_t>{1, 2}));
  EXPECT_THROW(static_cast<void>(result.value<std::int32_t>("burst")), std::logic_error);
}

// REQ: OPERATOR-072
TEST(OperatorTestHarness, DeferredStrideAndAxisViewsSurviveCallerDestructionAndFreshRuns) {
  for (const bool paired_axes : {false, true}) {
    auto h = holoscan::testing::create_operator_test<HostTensorProducer>();
    {
      std::vector<std::int64_t> shape{4};
      std::vector<std::uint64_t> strides{4};
      std::vector<holoscan::TensorAxisBounds> axes{{4, 4}};
      holoscan::TensorBounds bounds{.max_byte_span = 16U};
      if (paired_axes) {
        bounds.axes = axes;
      } else {
        bounds.max_shape = holoscan::TensorShapeView{shape};
        bounds.max_abs_byte_strides = holoscan::TensorMaxAbsByteStridesView{strides};
      }
      h.output_tensor_spec(
          "out",
          holoscan::TensorOutputSpec{
              .representation = {.dtype = DLDataType{kDLInt, 32, 1},
                                 .rank = 1U,
                                 .strides = holoscan::TensorStridePolicy::kBoundedStrided},
              .bounds = bounds});
      shape[0] = 999;
      strides[0] = 999;
      axes[0] = {999, 999};
    }
    h.capture_snapshot<holoscan::Tensor>("out");
    std::size_t hook_calls{};
    h.configure_graph([&](auto& f) {
      ++hook_calls;
      const auto graph = f.graph().view();
      for (const auto& port : graph.ports()) {
        if (port.name != "out")
          continue;
        const auto& contract = port.graph_override.tensor_contract;
        if (contract) {
          EXPECT_EQ(contract->axes, (std::vector<holoscan::TensorAxisBounds>{{4, 4}}));
          EXPECT_EQ(contract->max_byte_span, 16U);
        } else {
          ADD_FAILURE() << "Expected the owned output override";
        }
      }
    });
    for (int attempt = 0; attempt < 2; ++attempt) {
      const auto result = h.run();
      ASSERT_TRUE(result.ok()) << result.describe_failure();
      EXPECT_EQ(result.snapshots<holoscan::Tensor>("out").size(), 2U);
    }
    EXPECT_EQ(hook_calls, 2U);
  }
}

namespace {
class UnusedDeviceOutput final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(out, "out")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kCudaDevice,
                               .dtype = DLDataType{kDLUInt, 8, 1},
                               .rank = 1U},
            .bounds = holoscan::tensor_bounds(16U),
            .storage = holoscan::TensorOutputStorage::kRuntimePool});
  }
  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{1}})
        .complete_after(holoscan::ComputeCalls{.limit = 0U});
    return result;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    ADD_FAILURE() << "A zero-compute fixture must not execute";
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
  }
  holoscan::Output<holoscan::Tensor> out;
};
}  // namespace

// REQ: OPERATOR-072
TEST(OperatorTestHarness, UncapturedDeviceOutputStillRequiresExplicitPlacement) {
  int count{};
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
    GTEST_SKIP() << "CUDA device unavailable";
  auto h = holoscan::testing::create_operator_test<UnusedDeviceOutput>();
  const auto rejected = h.run();
  ASSERT_FALSE(rejected.plan().ok());
  EXPECT_NE(rejected.describe_failure().find("IMPLICIT_DEVICE_ZERO_REJECTED"), std::string::npos);
  h.output_device("out", holoscan::DeviceId{count > 1 ? 1 : 0});
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
}

// REQ: OPERATOR-072
TEST(OperatorTestHarness, ExplicitAndDeferredDeviceBindingsPreserveNativeDuplicateRejection) {
  int count{};
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
    GTEST_SKIP() << "CUDA device unavailable";
  const holoscan::DeviceId device{count > 1 ? 1 : 0};
  auto h = holoscan::testing::create_operator_test<DeviceTensorProducer>();
  h.output_device("out", device);
  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device({"operator", "out", device});
  h.set_compile_options(options);
  h.capture_snapshot<holoscan::Tensor>("out");
  const auto rejected = h.run();
  ASSERT_FALSE(rejected.plan().ok());
  EXPECT_TRUE(std::any_of(
      rejected.diagnostics().begin(), rejected.diagnostics().end(), [](const auto& diagnostic) {
        return diagnostic.provided.find("reason=duplicate_output") != std::string::npos;
      }));
  h.set_compile_options(holoscan::CompileOptions{});
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.snapshots<holoscan::Tensor>("out").size(), 2U);
}

// REQ: OPERATOR-072
TEST(OperatorTestHarness, NativeDropAndInflightFlowOptionsReachTheAuthoredEdges) {
  auto h = holoscan::testing::create_operator_test<ScaleOp>();
  const holoscan::ConnectionOptions options{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                            .queue_depth = 2U,
                                            .max_inflight_samples = 4U};
  h.input_flow("in", options);
  h.output_flow("out", options);
  h.push("in", std::int32_t{7});
  h.capture<std::int32_t>("out");
  h.configure_graph([](auto& f) {
    const auto graph = f.graph().view();
    ASSERT_EQ(graph.edges().size(), 2U);
    for (const auto& edge : graph.edges()) {
      EXPECT_EQ(edge.add_flow_options.queue_policy, holoscan::QueuePolicy::kDropOldest);
      EXPECT_EQ(edge.add_flow_options.queue_depth, 2U);
      EXPECT_EQ(edge.add_flow_options.max_inflight_samples, 4U);
    }
  });
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<std::int32_t>("out"), 7);
}

namespace {
class TimestampSequenceSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(out, "out").max_emits_per_compute(3U);
  }
  holoscan::Contract contract() const override {
    holoscan::Contract contract;
    contract.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{1}})
        .complete_after(holoscan::ComputeCalls{.limit = 1U});
    return contract;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    using namespace std::chrono_literals;
    const std::array<std::int32_t, 3> values{7, 11, 13};
    const std::array<std::chrono::nanoseconds, 3> times{-5ms, 1000ms, 0ms};
    for (std::size_t i = 0; i < values.size(); ++i) {
      holoscan::SampleMetadata metadata;
      metadata.capture_timestamp_ns = times[i].count();
      metadata.source_clock_id = context.clock().id();
      metadata.frame_id = 41U + i;
      metadata.trace_id = 17U + i;
      auto emitted = out.emit(holoscan::Sample<std::int32_t>{values[i], metadata});
      if (!emitted)
        return emitted;
    }
    return {};
  }
  holoscan::Output<std::int32_t> out;
};
class TimestampSequenceCollector final : public holoscan::Operator<> {
 public:
  explicit TimestampSequenceCollector(
      std::shared_ptr<std::vector<holoscan::Sample<std::int32_t>>> samples)
      : samples_(std::move(samples)) {}
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in"); }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = in.receive();
    if (!sample)
      return holoscan::make_unexpected(std::move(sample).error());
    samples_->emplace_back(sample->data, sample->metadata);
    return {};
  }
  holoscan::Input<std::int32_t> in;

 private:
  std::shared_ptr<std::vector<holoscan::Sample<std::int32_t>>> samples_;
};
}  // namespace

// REQ: OPERATOR-065
TEST(OperatorTestHarness, TimestampedInputsPreserveNativeSelectionAndSemanticMetadata) {
  using namespace std::chrono_literals;
  using holoscan::testing::sample_at;
  auto h = holoscan::testing::create_operator_test<ScaleOp>();
  auto first = sample_at(std::int32_t{7}, -5ms);
  first.metadata.frame_id = 41U;
  first.metadata.trace_id = 17U;
  auto second = sample_at(std::int32_t{11}, 1000ms);
  second.metadata.frame_id = 42U;
  second.metadata.trace_id = 18U;
  auto third = sample_at(std::int32_t{13}, 0ms);
  third.metadata.frame_id = 43U;
  third.metadata.trace_id = 19U;
  h.push_samples("in", std::vector{first, second, third});
  h.capture<std::int32_t>("out");
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  // The Runtime may order candidates by capture coordinate even though the source publishes
  // in caller order. Compare an independent explicit Graph instead of inventing fixture FIFO.
  auto native_samples = std::make_shared<std::vector<holoscan::Sample<std::int32_t>>>();
  holoscan::Graph graph{"timestamp-native-parity"};
  auto source = graph.op<TimestampSequenceSource>("source");
  auto op = graph.op<ScaleOp>("operator");
  auto collector = graph.op<TimestampSequenceCollector>("collector", native_samples);
  graph.add_flow(source->out, op->in, holoscan::ConnectionOptions{.queue_depth = 3U});
  graph.add_flow(op->out, collector->in, holoscan::ConnectionOptions{.queue_depth = 1024U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  graph.partition("fixture").include_all();
  auto native_plan = holoscan::compile(graph);
  ASSERT_TRUE(native_plan.ok());
  auto native = holoscan::run_async(native_plan);
  native.wait();
  ASSERT_EQ(native.termination(), holoscan::RunTermination::kCompleted);
  const auto samples = result.samples<std::int32_t>("out");
  ASSERT_EQ(samples.size(), 3U);
  const std::array<std::chrono::nanoseconds, 3> coordinates{-5ms, 1000ms, 0ms};
  ASSERT_TRUE(samples.front().metadata.source_clock_id.valid());
  for (std::size_t i = 0; i < samples.size(); ++i) {
    ASSERT_EQ(native_samples->size(), samples.size());
    EXPECT_EQ(samples[i].data, (*native_samples)[i].data);
    EXPECT_EQ(samples[i].metadata.capture_timestamp_ns,
              (*native_samples)[i].metadata.capture_timestamp_ns);
    EXPECT_EQ(samples[i].metadata.frame_id, (*native_samples)[i].metadata.frame_id);
    EXPECT_EQ(samples[i].metadata.trace_id, (*native_samples)[i].metadata.trace_id);
    EXPECT_EQ(samples[i].metadata.source_clock_id, samples.front().metadata.source_clock_id);
    const auto publication_index = samples[i].metadata.frame_id - 41U;
    ASSERT_LT(publication_index, coordinates.size());
    EXPECT_EQ(samples[i].metadata.capture_timestamp_ns, coordinates[publication_index].count());
    EXPECT_EQ(samples[i].metadata.trace_id, 17U + publication_index);
  }
}

// REQ: OPERATOR-073
TEST(OperatorTestHarness, RetainPreservesNativeTensorBackingAfterRunAndLaterAttempt) {
  auto h = holoscan::testing::create_operator_test<HostTensorProducer>();
  h.capture<holoscan::Tensor>("out", holoscan::testing::Retain{.max_records = 8U});
  const auto first = h.run();
  ASSERT_TRUE(first.ok()) << first.describe_failure();
  const auto values = first.values<holoscan::Tensor>("out");
  ASSERT_EQ(values.size(), 2U);
  ASSERT_NE(values[0].data_as<std::int32_t>(), nullptr);
  EXPECT_EQ(values[0].data_as<std::int32_t>()[0], 10);
  EXPECT_EQ(values[1].data_as<std::int32_t>()[0], 20);
  const auto second = h.run();
  ASSERT_TRUE(second.ok()) << second.describe_failure();
  EXPECT_EQ(values[0].data_as<std::int32_t>()[0], 10);
}

// REQ: OPERATOR-073
TEST(OperatorTestHarness, UnprovedRetainRejectsBeforeConsumingInputAndSnapshotCanReplaceIt) {
  auto h = holoscan::testing::create_operator_test<HostTensorPass>();
  h.push("in", input_tensor(10));
  h.capture<holoscan::Tensor>("out", holoscan::testing::Retain{.max_records = 8U});
  EXPECT_THROW(static_cast<void>(h.run()), std::invalid_argument);
  h.capture_snapshot<holoscan::Tensor>("out");
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.snapshots<holoscan::Tensor>("out").size(), 1U);
}

// REQ: OPERATOR-073
TEST(OperatorTestHarness, RetainRejectsRecordLimitSmallerThanProvedWholeRunOutput) {
  auto h = holoscan::testing::create_operator_test<HostTensorProducer>();
  h.capture<holoscan::Tensor>("out", holoscan::testing::Retain{.max_records = 1U});
  EXPECT_THROW(static_cast<void>(h.run()), std::invalid_argument);
}

// REQ: OPERATOR-073
TEST(OperatorTestHarness, RetainRejectsIncompleteBoundaryEvenWhenRuntimeCompletes) {
  for (const int fault : {1, 2, 3}) {
    auto h = holoscan::testing::create_operator_test<HostTensorProducer>(2U, fault);
    h.capture<holoscan::Tensor>("out", holoscan::testing::Retain{.max_records = 8U});
    const auto result = h.run();
    EXPECT_FALSE(result.ok()) << "fault=" << fault;
    EXPECT_EQ(result.termination(),
              fault == 1   ? holoscan::RunTermination::kCompleted
              : fault == 2 ? holoscan::RunTermination::kStopped
                           : holoscan::RunTermination::kFailed);
  }
}

// REQ: OPERATOR-073
TEST(OperatorTestHarness, RetainZeroLimitCompletesNaturallyAndSmallPoolRejectsCold) {
  auto empty = holoscan::testing::create_operator_test<HostTensorProducer>(0U);
  empty.capture<holoscan::Tensor>("out", holoscan::testing::Retain{.max_records = 8U});
  const auto result = empty.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.termination(), holoscan::RunTermination::kCompleted);
  EXPECT_TRUE(result.values<holoscan::Tensor>("out").empty());
  auto small = holoscan::testing::create_operator_test<HostTensorProducer>(20U);
  small.capture<holoscan::Tensor>("out", holoscan::testing::Retain{.max_records = 20U});
  small.output_flow("out", holoscan::ConnectionOptions{.queue_depth = 1U});
  EXPECT_THROW(static_cast<void>(small.run()), std::invalid_argument);
}

// REQ: OPERATOR-073
TEST(OperatorTestHarness, RetainPlanReplayUsesIndependentCollectorProgressAndBacking) {
  auto h = holoscan::testing::create_operator_test<HostTensorProducer>();
  h.capture<holoscan::Tensor>("out", holoscan::testing::Retain{.max_records = 8U});
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto run = [plan = result.plan()] {
    auto session = holoscan::run_async(plan);
    session.wait();
    return session.termination();
  };
  auto first = std::async(std::launch::async, run);
  auto second = std::async(std::launch::async, run);
  EXPECT_EQ(first.get(), holoscan::RunTermination::kStopped);
  EXPECT_EQ(second.get(), holoscan::RunTermination::kStopped);
  EXPECT_EQ(result.values<holoscan::Tensor>("out").front().data_as<std::int32_t>()[0], 10);
}

// REQ: OPERATOR-073
TEST(OperatorTestHarness, NativeRetainPreservesEncodedOperatorNames) {
  holoscan::testing::NativeOperatorTest h{[](holoscan::Graph& graph) {
    return holoscan::detail::OperatorTestingAccess::erase(
        graph.op<HostTensorProducer>("camera/source"));
  }};
  h.declare_capture("out", holoscan::testing::retain_capture<holoscan::Tensor>(8U));
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.values<holoscan::Tensor>("out").size(), 2U);
}

// REQ: OPERATOR-073
TEST(OperatorTestHarness, RetainDeviceReadinessOutlivesTheRuntimeProducerStream) {
  int count{};
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
    GTEST_SKIP();
  int previous{};
  ASSERT_EQ(cudaGetDevice(&previous), cudaSuccess);
  struct Restore {
    int device;
    ~Restore() { cudaSetDevice(device); }
  };
  const Restore restore{previous};
  const auto device = count > 1 ? 1 : 0;
  auto h = holoscan::testing::create_operator_test<DeviceTensorProducer>();
  h.output_device("out", holoscan::DeviceId{device});
  h.capture<holoscan::Tensor>("out", holoscan::testing::Retain{.max_records = 8U});
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
  cudaStream_t stream{};
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  struct Destroy {
    cudaStream_t stream;
    ~Destroy() { cudaStreamDestroy(stream); }
  };
  const Destroy destroy{stream};
  const auto values = result.values<holoscan::Tensor>("out");
  ASSERT_EQ(values.size(), 2U);
  for (std::size_t index = 0; index < values.size(); ++index) {
    std::array<std::uint8_t, 16> bytes{};
    auto copied = values[index].copy_to_host(bytes.data(), bytes.size(), stream);
    ASSERT_TRUE(copied) << copied.error().message;
    EXPECT_EQ(values[index].device().device_id, device);
    EXPECT_TRUE(
        std::all_of(bytes.begin(), bytes.end(), [index](auto byte) { return byte == index + 1U; }));
  }
}

namespace {
struct RetainProvenanceProbe {
  std::shared_ptr<const holoscan::testing::detail::RetainPlan> plan;
  holoscan::Tensor predecessor;
  holoscan::Tensor foreign;
  std::array<bool, 7> checked{};
};
class RetainProvenanceCollector final : public holoscan::Operator<> {
 public:
  explicit RetainProvenanceCollector(std::shared_ptr<RetainProvenanceProbe> probe)
      : probe_(std::move(probe)) {}
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in"); }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    auto sample = in.receive();
    if (!sample)
      return holoscan::make_unexpected(sample.error());
    using holoscan::testing::detail::validate_retained_tensor;
    const auto& proof = *probe_->plan;
    const auto self = context.self();
    probe_->checked[0] = bool(validate_retained_tensor(sample->data, proof, 0U, self));
    probe_->checked[1] = !validate_retained_tensor(probe_->foreign, proof, 0U, self);
    probe_->checked[2] = !validate_retained_tensor(probe_->predecessor, proof, 0U, self);
    auto changed = proof;
    ++changed.owner.plan_generation;
    probe_->checked[3] = !validate_retained_tensor(sample->data, changed, 0U, self);
    auto wrong_run = self;
    ++wrong_run.run_generation;
    probe_->checked[4] = !validate_retained_tensor(sample->data, proof, 0U, wrong_run);
    changed = proof;
    changed.pool_ids[0] = "unrelated-field-pool";
    probe_->checked[5] = !validate_retained_tensor(sample->data, changed, 0U, self);
    probe_->checked[6] = !validate_retained_tensor(sample->data, proof, 1U, self);
    context.request_stop();
    return {};
  }
  holoscan::Input<holoscan::Tensor> in;

 private:
  std::shared_ptr<RetainProvenanceProbe> probe_;
};
}  // namespace

// REQ: OPERATOR-073
TEST(OperatorTestHarness, RetainProvenanceRejectsForeignPriorRunAndWrongFieldBacking) {
  auto first = holoscan::testing::create_operator_test<HostTensorProducer>();
  first.capture<holoscan::Tensor>("out", holoscan::testing::Retain{.max_records = 8U});
  const auto predecessor = first.run();
  ASSERT_TRUE(predecessor.ok()) << predecessor.describe_failure();
  auto probe = std::make_shared<RetainProvenanceProbe>();
  probe->predecessor = predecessor.values<holoscan::Tensor>("out").front();
  probe->foreign = input_tensor(10);
  holoscan::Graph graph{"retention-provenance"};
  const auto source = graph.op<HostTensorProducer>("source", 1U);
  const auto collector = graph.op<RetainProvenanceCollector>("collector", probe);
  graph.add_flow(source->out, collector->in, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  graph.partition("fixture").include_all();
  const auto plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());
  probe->plan =
      holoscan::testing::detail::prepare_retention(plan, "source", "out", {.capacity = 8U});
  auto session = holoscan::run_async(plan);
  session.wait();
  EXPECT_EQ(session.termination(), holoscan::RunTermination::kStopped);
  for (std::size_t index = 0; index < probe->checked.size(); ++index)
    EXPECT_TRUE(probe->checked[index]) << "provenance case " << index;
}

namespace {
class PaddedTensorSource final : public holoscan::Operator<> {
 public:
  explicit PaddedTensorSource(std::shared_ptr<std::optional<holoscan::TensorOutputLoan>> consumed)
      : consumed_(std::move(consumed)) {}
  void setup(holoscan::OperatorSpec& spec) override {
    constexpr std::array axes{holoscan::TensorAxisBounds{2U, 4U},
                              holoscan::TensorAxisBounds{2U, 1U}};
    spec.output(out, "out")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 8, 1},
                               .rank = 2U,
                               .strides = holoscan::TensorStridePolicy::kBoundedStrided},
            .bounds = holoscan::TensorBounds{.max_rank = 2U, .max_byte_span = 8U, .axes = axes},
            .storage = holoscan::TensorOutputStorage::kRuntimePool});
  }
  holoscan::Contract contract() const override {
    holoscan::Contract c;
    c.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{1}})
        .complete_after(holoscan::ComputeCalls{.limit = 1U});
    return c;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    const std::array<std::int64_t, 2> shape{2, 2};
    const std::array<std::int64_t, 2> strides{4, 1};
    auto loan = out.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = shape,
        .dtype = DLDataType{kDLUInt, 8, 1},
        .byte_strides = holoscan::TensorByteStridesView{holoscan::TensorByteStridesSpan{strides}}});
    if (!loan)
      return holoscan::make_unexpected(loan.error());
    auto filled = loan->write_host_and_commit(
        [](const auto& write) -> holoscan::expected<void, holoscan::Error> {
          if (write.byte_size() < 6U)
            return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
          auto* bytes = write.template data_as<std::uint8_t>();
          bytes[0] = 1U;
          bytes[1] = 2U;
          bytes[4] = 3U;
          bytes[5] = 4U;
          return {};
        });
    if (!filled)
      return filled;
    auto& staged_loan = consumed_->emplace(std::move(*loan));
    return out.emit(std::move(staged_loan));
  }
  holoscan::Output<holoscan::Tensor> out;

 private:
  std::shared_ptr<std::optional<holoscan::TensorOutputLoan>> consumed_;
};
}  // namespace

// REQ: OPERATOR-073
TEST(OperatorTestHarness, RetainPreservesPaddedNativeLayoutWithoutGrantingWriteAuthority) {
  auto consumed = std::make_shared<std::optional<holoscan::TensorOutputLoan>>();
  auto h = holoscan::testing::create_operator_test<PaddedTensorSource>(consumed);
  h.capture_snapshot<holoscan::Tensor>("out");
  const auto snapshot = h.run();
  EXPECT_FALSE(snapshot.ok());
  EXPECT_TRUE(snapshot.snapshots<holoscan::Tensor>("out").empty());
  h.capture<holoscan::Tensor>("out", holoscan::testing::Retain{.max_records = 4U});
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  auto value = result.value<holoscan::Tensor>("out");
  EXPECT_EQ(value.shape(), (std::vector<std::int64_t>{2, 2}));
  EXPECT_EQ(value.strides(), (std::vector<std::int64_t>{4, 1}));
  EXPECT_FALSE(value.is_contiguous());
  EXPECT_EQ(value.data_as<std::uint8_t>()[5], 4U);
  auto& consumed_loan = *consumed;
  if (!consumed_loan.has_value()) {
    ADD_FAILURE() << "producer did not preserve its consumed loan";
    return;
  }
  const auto stale_write = consumed_loan->write_host();
  ASSERT_FALSE(stale_write);
  EXPECT_EQ(stale_write.error().code, holoscan::ErrorCode::kInvalidArgument);
}

namespace {
class RequestedCompletionProbe final : public holoscan::Operator<> {
 public:
  explicit RequestedCompletionProbe(bool stop = false, bool cleanup_failure = false)
      : stop_(stop), cleanup_failure_(cleanup_failure) {}
  void setup(holoscan::OperatorSpec&) override {}
  holoscan::Contract contract() const override {
    holoscan::Contract value;
    value.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{1}});
    return value;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    if (stop_)
      context.request_stop();
    else
      context.request_completion();
    return {};
  }
  void stop() override {
    if (cleanup_failure_)
      throw std::runtime_error{"completion probe cleanup failure"};
  }

 private:
  bool stop_{};
  bool cleanup_failure_{};
};

holoscan::Tensor empty_host_tensor() {
  struct Storage {
    DLManagedTensor managed{};
    std::array<std::int64_t, 1> shape{0};
  };
  auto owner = std::make_unique<Storage>();
  owner->managed.dl_tensor = DLTensor{.data = nullptr,
                                      .device = DLDevice{kDLCPU, 0},
                                      .ndim = 1,
                                      .dtype = DLDataType{kDLInt, 32, 1},
                                      .shape = owner->shape.data(),
                                      .strides = nullptr,
                                      .byte_offset = 0};
  owner->managed.manager_ctx = owner.get();
  owner->managed.deleter = [](DLManagedTensor* value) {
    delete static_cast<Storage*>(value->manager_ctx);
  };
  return holoscan::Tensor{&owner.release()->managed};
}
}  // namespace

// REQ: OPERATOR-064
TEST(OperatorTestHarness, GregRequestedCompletionIsSuccessfulWithoutInventingDrain) {
  auto h = holoscan::testing::create_operator_test<RequestedCompletionProbe>();
  const auto result = h.run();
  ASSERT_EQ(result.termination(), holoscan::RunTermination::kRequestedCompleted);
  EXPECT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_TRUE(result.describe_failure().empty());
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, GregCaptureSelectorAndTypeRejectBeforeRunning) {
  auto h = holoscan::testing::create_operator_test<ScaleOp>();
  h.push("in", std::int32_t{3});
  EXPECT_THROW(h.capture<std::int32_t>("missing_output"), holoscan::RuntimeError);
  EXPECT_THROW(h.capture<float>("out"), holoscan::RuntimeError);
  h.capture<std::int32_t>("out");
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<std::int32_t>("out"), 3);
}

// REQ: OPERATOR-073
TEST(OperatorTestHarness, RadarEmptyTensorIsOneRecordWithPreservedMetadata) {
  auto h = holoscan::testing::create_operator_test<HostTensorPass>();
  h.push("in", empty_host_tensor());
  h.capture_snapshot<holoscan::Tensor>("out");
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto snapshots = result.snapshots<holoscan::Tensor>("out");
  ASSERT_EQ(snapshots.size(), 1U);
  EXPECT_TRUE(snapshots.front().bytes().empty());
  EXPECT_EQ(snapshots.front().shape(), (std::vector<std::int64_t>{0}));
  EXPECT_EQ(snapshots.front().dtype().code, kDLInt);
  EXPECT_EQ(snapshots.front().dtype().bits, 32);
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, ExpectedStopDoesNotHideCleanupFailureOrWatchdogExpiry) {
  auto stopped = holoscan::testing::create_operator_test<RequestedCompletionProbe>(true);
  const auto unexpected = stopped.run();
  EXPECT_EQ(unexpected.termination(), holoscan::RunTermination::kStopped);
  EXPECT_FALSE(unexpected.ok());
  EXPECT_NE(unexpected.describe_failure().find("kStopped"), std::string::npos);
  EXPECT_TRUE(stopped.run(holoscan::testing::TestRunOptions{.expect_stop = true}).ok());
  auto failure = holoscan::testing::create_operator_test<RequestedCompletionProbe>(true, true);
  const auto failed = failure.run(holoscan::testing::TestRunOptions{.expect_stop = true});
  EXPECT_FALSE(failed.ok());
  EXPECT_EQ(failed.termination(), holoscan::RunTermination::kFailed);
  auto probe = std::make_shared<LifecycleProbe>();
  auto idle = holoscan::testing::create_operator_test<UnboundedClockOp>(probe);
  const auto timed_out = idle.run(holoscan::testing::TestRunOptions{
      .timeout = std::chrono::milliseconds{20}, .expect_stop = true});
  EXPECT_FALSE(timed_out.ok());
  EXPECT_NE(timed_out.describe_failure().find("watchdog expired"), std::string::npos);
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, InvalidBlockingTimeoutAndMovedFromUsePreserveStaging) {
  auto original = holoscan::testing::create_operator_test<ScaleOp>();
  original.push("in", std::int32_t{9});
  original.capture<std::int32_t>("out");
  EXPECT_THROW(static_cast<void>(original.run(
                   holoscan::testing::TestRunOptions{.timeout = std::chrono::milliseconds{0}})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(original.run(
                   holoscan::testing::TestRunOptions{.timeout = std::chrono::milliseconds::max()})),
               std::invalid_argument);
  auto moved = std::move(original);
  // Intentional contract probes: public operations must reject the moved-from harness.
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_THROW(original.push("in", std::int32_t{1}), std::logic_error);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_THROW(original.input_queue_depth("in", 3U), std::logic_error);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_THROW(static_cast<void>(original.run()), std::logic_error);
  const auto result = moved.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<std::int32_t>("out"), 9);
}

// REQ: OPERATOR-072
TEST(OperatorTestHarness, EveryPortSetterValidatesBeforeChangingPendingInputs) {
  auto h = holoscan::testing::create_operator_test<ScaleOp>();
  h.push("in", std::int32_t{4});
  h.capture<std::int32_t>("out");
  EXPECT_THROW(h.input_queue_depth("typo", 2U), holoscan::RuntimeError);
  EXPECT_THROW(h.input_tensor_spec("typo", std::nullopt), holoscan::RuntimeError);
  EXPECT_THROW(h.output_tensor_spec("in", std::nullopt), holoscan::RuntimeError);
  EXPECT_THROW(h.input_source_tensor_spec("typo", std::nullopt), holoscan::RuntimeError);
  EXPECT_THROW(h.input_source_device("typo", std::nullopt), holoscan::RuntimeError);
  EXPECT_THROW(h.output_device("typo", std::nullopt), holoscan::RuntimeError);
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<std::int32_t>("out"), 4);
}

namespace {
template <typename Value>
concept CanStageTimedScalarMember =
    requires(holoscan::testing::OperatorTestHarness<ScaleOp>& fixture, Value value) {
      fixture.push_at(&ScaleOp::in, value, std::chrono::nanoseconds{1});
    };
template <typename Op>
concept CanCaptureValueMember =
    requires(holoscan::testing::OperatorTestHarness<Op>& fixture) { fixture.capture(&Op::out); };
template <typename Op>
concept CanCaptureSnapshotMember = requires(holoscan::testing::OperatorTestHarness<Op>& fixture) {
  fixture.capture_snapshot(&Op::out);
};
template <typename Op>
concept CanCaptureRetainedMember = requires(holoscan::testing::OperatorTestHarness<Op>& fixture) {
  fixture.capture(&Op::out, holoscan::testing::Retain{});
};
static_assert(CanStageTimedScalarMember<std::int16_t>);
static_assert(!CanStageTimedScalarMember<std::vector<std::int32_t>>);
static_assert(CanCaptureValueMember<ScaleOp>);
static_assert(!CanCaptureSnapshotMember<ScaleOp>);
static_assert(!CanCaptureRetainedMember<ScaleOp>);
static_assert(!CanCaptureValueMember<HostTensorProducer>);
static_assert(CanCaptureSnapshotMember<HostTensorProducer>);
static_assert(CanCaptureRetainedMember<HostTensorProducer>);
static_assert(std::is_move_constructible_v<holoscan::testing::OperatorTestHarness<ScaleOp>>);
static_assert(!std::is_move_assignable_v<holoscan::testing::OperatorTestHarness<ScaleOp>>);

class NamedConversionOp final : public holoscan::Operator<ScaleParams> {
 public:
  NamedConversionOp(std::string input_name, std::string output_name)
      : input_name_(std::move(input_name)), output_name_(std::move(output_name)) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, input_name_);
    spec.output(out, output_name_);
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = in.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    sample->data *= params().factor;
    return out.emit(std::move(*sample));
  }
  holoscan::Input<std::int32_t> in;
  holoscan::Output<std::int32_t> out;

 private:
  std::string input_name_;
  std::string output_name_;
};
}  // namespace

// REQ: OPERATOR-064
TEST(OperatorTestHarness, MemberSelectorsConvertInputAndUseConfiguredPortNames) {
  auto h = holoscan::testing::create_operator_test<NamedConversionOp>(
      std::string{"selected_input"}, std::string{"selected_output"});
  h.set_params(ScaleParams{.factor = 2});
  h.push(&NamedConversionOp::in, std::int16_t{21});
  h.capture(&NamedConversionOp::out, 2U);
  const auto first = h.run();
  ASSERT_TRUE(first.ok()) << first.describe_failure();
  EXPECT_EQ(first.value<std::int32_t>("selected_output"), 42);

  h.set_params(ScaleParams{.factor = 3});
  h.push(&NamedConversionOp::in, std::int16_t{7});
  const auto second = h.run();
  ASSERT_TRUE(second.ok()) << second.describe_failure();
  EXPECT_EQ(second.value<std::int32_t>("selected_output"), 21);
  EXPECT_EQ(first.value<std::int32_t>("selected_output"), 42);
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, InitializerListDeducesValueBatchAndAllowsExplicitEmptyBatch) {
  auto h = holoscan::testing::create_operator_test<ScaleOp>();
  h.set_params(ScaleParams{.factor = 2});
  h.push_many("in", {std::int32_t{1}, std::int32_t{2}, std::int32_t{3}});
  h.capture(&ScaleOp::out);
  const auto first = h.run();
  ASSERT_TRUE(first.ok()) << first.describe_failure();
  EXPECT_EQ(first.values<std::int32_t>("out"), (std::vector<std::int32_t>{2, 4, 6}));

  h.push_many<std::int32_t>("in", {});
  const auto empty = h.run();
  ASSERT_TRUE(empty.ok()) << empty.describe_failure();
  EXPECT_TRUE(empty.values<std::int32_t>("out").empty());
}

// REQ: OPERATOR-065
TEST(OperatorTestHarness, TimestampConveniencesDeduceEnvelopesAndPreserveCaptureCoordinates) {
  using namespace std::chrono_literals;
  using holoscan::testing::sample_at;
  auto h = holoscan::testing::create_operator_test<ScaleOp>();
  auto first = sample_at(std::int32_t{11}, 10ms);
  first.metadata.frame_id = 17U;
  first.metadata.trace_id = 23U;
  h.push_samples("in", {first, sample_at(std::int32_t{13}, 20ms)});
  h.push_at("in", std::int32_t{21}, 30ms);
  h.push_at(&ScaleOp::in, std::int16_t{34}, 40ms);
  h.capture(&ScaleOp::out);
  const auto result = h.run(holoscan::testing::TestRunOptions{.timeout = 5s});
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto samples = result.samples<std::int32_t>("out");
  ASSERT_EQ(samples.size(), 4U);
  const std::array<std::int32_t, 4> values{11, 13, 21, 34};
  for (std::size_t i = 0; i < samples.size(); ++i) {
    EXPECT_EQ(samples[i].data, values[i]);
    EXPECT_EQ(samples[i].metadata.capture_timestamp_ns,
              std::chrono::duration_cast<std::chrono::nanoseconds>(10ms * (i + 1U)).count());
    EXPECT_NE(samples[i].metadata.source_clock_id.value, 0U);
  }
  EXPECT_EQ(samples.front().metadata.frame_id, 17U);
  EXPECT_EQ(samples.front().metadata.trace_id, 23U);

  h.push_samples<std::int32_t>("in", {});
  const auto empty = h.run();
  ASSERT_TRUE(empty.ok()) << empty.describe_failure();
  EXPECT_TRUE(empty.samples<std::int32_t>("out").empty());
}

// REQ: OPERATOR-071
TEST(OperatorTestHarness, CoordinateFreeFramePreservesAbsentCaptureTime) {
  using holoscan::testing::frame;
  using holoscan::testing::input;
  auto h = holoscan::testing::create_operator_test<ScaleOp>();
  h.push_frame(frame(input("in", std::int32_t{21})));
  h.push_frame(frame());
  h.capture(&ScaleOp::out);
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto samples = result.samples<std::int32_t>("out");
  ASSERT_EQ(samples.size(), 1U);
  const auto& sample = samples.front();
  EXPECT_EQ(sample.data, 21);
  EXPECT_EQ(sample.metadata.capture_timestamp_ns, 0);
  EXPECT_EQ(sample.metadata.source_clock_id.value, 0U);
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, NullValueMemberSelectorsPreservePendingInputAndCapture) {
  auto h = holoscan::testing::create_operator_test<ScaleOp>();
  h.push(&ScaleOp::in, std::int16_t{21});
  h.capture(&ScaleOp::out);
  holoscan::Input<std::int32_t> ScaleOp::* const missing_input = nullptr;
  holoscan::Output<std::int32_t> ScaleOp::* const missing_output = nullptr;
  EXPECT_THROW(h.push(missing_input, std::int16_t{99}), std::invalid_argument);
  EXPECT_THROW(h.push_at(missing_input, std::int16_t{99}, std::chrono::nanoseconds{1}),
               std::invalid_argument);
  EXPECT_THROW(h.capture(missing_output), std::invalid_argument);
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<std::int32_t>("out"), 21);
}

// REQ: OPERATOR-073
TEST(OperatorTestHarness, TensorOutputMemberSelectsSnapshotOrRetainPolicy) {
  auto snapshot_fixture = holoscan::testing::create_operator_test<HostTensorProducer>();
  snapshot_fixture.capture_snapshot(&HostTensorProducer::out, 2U, 16U);
  holoscan::Output<holoscan::Tensor> HostTensorProducer::* const missing = nullptr;
  EXPECT_THROW(snapshot_fixture.capture_snapshot(missing), std::invalid_argument);
  EXPECT_THROW(snapshot_fixture.capture(missing, holoscan::testing::Retain{2U}),
               std::invalid_argument);
  const auto snapshots = snapshot_fixture.run();
  ASSERT_TRUE(snapshots.ok()) << snapshots.describe_failure();
  const auto records = snapshots.snapshots<holoscan::Tensor>("out");
  ASSERT_EQ(records.size(), 2U);
  ASSERT_EQ(records.front().bytes().size(), 16U);
  std::array<std::int32_t, 4> values{};
  std::memcpy(values.data(), records.front().bytes().data(), records.front().bytes().size());
  EXPECT_EQ(values, (std::array<std::int32_t, 4>{10, 11, 12, 13}));

  auto retain_fixture = holoscan::testing::create_operator_test<HostTensorProducer>();
  retain_fixture.capture(&HostTensorProducer::out, holoscan::testing::Retain{2U});
  const auto retained = retain_fixture.run();
  ASSERT_TRUE(retained.ok()) << retained.describe_failure();
  const auto tensors = retained.values<holoscan::Tensor>("out");
  ASSERT_EQ(tensors.size(), 2U);
  EXPECT_EQ(tensors.front().data_as<std::int32_t>()[0], 10);
}

// REQ: OPERATOR-064
TEST(OperatorTestHarness, InputAndValueOutputMemberSelectorsGuardMovedFromFixture) {
  auto source = holoscan::testing::create_operator_test<ScaleOp>();
  source.push(&ScaleOp::in, std::int16_t{21});
  source.capture(&ScaleOp::out);
  auto destination = std::move(source);
  // Intentional contract probes: member selectors must reject the moved-from harness.
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_THROW(source.push(&ScaleOp::in, std::int16_t{99}), std::logic_error);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_THROW(source.push_at(&ScaleOp::in, std::int16_t{99}, std::chrono::nanoseconds{1}),
               std::logic_error);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_THROW(source.capture(&ScaleOp::out), std::logic_error);
  holoscan::Input<std::int32_t> ScaleOp::* const missing_input = nullptr;
  holoscan::Output<std::int32_t> ScaleOp::* const missing_output = nullptr;
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_THROW(source.push(missing_input, std::int16_t{99}), std::logic_error);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_THROW(source.capture(missing_output), std::logic_error);
  const auto result = destination.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<std::int32_t>("out"), 21);
}

// REQ: OPERATOR-073
TEST(OperatorTestHarness, TensorOutputMemberSelectorsGuardMovedFromFixture) {
  auto source = holoscan::testing::create_operator_test<HostTensorProducer>();
  source.capture_snapshot(&HostTensorProducer::out, 2U, 16U);
  auto destination = std::move(source);
  // Intentional contract probes: snapshot and Retain selectors reject the moved-from harness.
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_THROW(source.capture_snapshot(&HostTensorProducer::out), std::logic_error);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_THROW(source.capture(&HostTensorProducer::out, holoscan::testing::Retain{2U}),
               std::logic_error);
  const auto result = destination.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.snapshots<holoscan::Tensor>("out").size(), 2U);
}

// REQ: OPERATOR-058
TEST(OperatorTestHarness, ControlledClockUnstampedInputPreservesAbsentCaptureTime) {
  using namespace std::chrono_literals;
  auto h = holoscan::testing::create_operator_test<FreshPairOp>();
  h.push("left", std::int32_t{1});
  h.push("right", std::int32_t{2});
  h.capture<std::int32_t>("out");
  const auto result = h.run_controlled_clock(holoscan::testing::ControlledClockRun{.tick = 100ms});
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.clock_outcome(), holoscan::testing::ClockTestOutcome::kSuppressed);
  EXPECT_TRUE(result.samples<std::int32_t>("out").empty());
  // Missing capture time is not an implicit zero in the fixture clock domain.
  EXPECT_NE(result.clock_selection("left").state, holoscan::InputState::kSelected);
  EXPECT_NE(result.clock_selection("right").state, holoscan::InputState::kSelected);
}

// REQ: RUNTIME-059
TEST(OperatorTestHarness, ControlledClockPreservesStartupOrRequestedOutput) {
  using namespace std::chrono_literals;
  auto h = holoscan::testing::create_operator_test<FreshPairOp>();
  h.push_at("left", std::int32_t{1}, 0ns);
  h.push_at("right", std::int32_t{2}, 0ns);
  h.capture<std::int32_t>("out");
  const auto result = h.run_controlled_clock(holoscan::testing::ControlledClockRun{.tick = 100ms});
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto samples = result.samples<std::int32_t>("out");
  // LatestWithin consumes this finite pair once. An admitted startup compute is still returned
  // when the requested tick subsequently has no remaining input and is suppressed.
  ASSERT_EQ(samples.size(), 1U);
  EXPECT_EQ(samples.front().data, 3);
  const auto stamp = samples.front().metadata.publish_timestamp_ns;
  ASSERT_TRUE(stamp == 0 || stamp == 100000000);
  EXPECT_EQ(result.clock_outcome(),
            stamp == 0 ? holoscan::testing::ClockTestOutcome::kSuppressed
                       : holoscan::testing::ClockTestOutcome::kRetired);
}
