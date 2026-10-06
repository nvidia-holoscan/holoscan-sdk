/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <holoscan/testing/operator_test_harness.hpp>
#include <memory>
#include <utility>
#include <vector>

namespace {
using namespace holoscan;
using namespace holoscan::testing;
using namespace std::chrono_literals;

struct RateProbe {
  std::atomic<bool> fail_start{};
  std::atomic<bool> fail_warm_up{};
};
class RateDriver final : public Operator<> {
 public:
  explicit RateDriver(std::shared_ptr<RateProbe> probe) : probe_(std::move(probe)) {}
  void setup(OperatorSpec& spec) override {
    spec.input(in, "in");
    spec.output(out, "out")
        .lifecycle_nominal_rate(LifecycleStage::kArm)
        .max_nominal_rate(Rate::hz(1000U));
    spec.lifecycle()
        .stage(LifecycleStage::kArm, &RateDriver::arm)
        .stage(LifecycleStage::kStart, &RateDriver::begin)
        .stage(LifecycleStage::kWarmUp, &RateDriver::warm_up);
  }
  LifecycleStatus arm(LifecycleContext& c) noexcept {
    if (c.transition_phase() == LifecycleTransitionPhase::kRestoreOld)
      return LifecycleStatus::kOk;
    const auto bound = c.bind_nominal_rate(out, Rate::hz(1000U));
    if (!bound) {
      c.report_error(bound.error());
      return LifecycleStatus::kFatalFailure;
    }
    return LifecycleStatus::kOk;
  }
  LifecycleStatus warm_up(LifecycleContext& c) noexcept {
    if (c.transition_phase() == LifecycleTransitionPhase::kRestartForward &&
        probe_->fail_warm_up.exchange(false)) {
      c.report_error(ErrorCode::kFailure, "candidate WarmUp failed before helper Arm");
      return LifecycleStatus::kFatalFailure;
    }
    return LifecycleStatus::kOk;
  }
  LifecycleStatus begin(LifecycleContext& c) noexcept {
    if (c.transition_phase() == LifecycleTransitionPhase::kRestartForward &&
        probe_->fail_start.exchange(false)) {
      c.report_error(ErrorCode::kFailure, "candidate Start failed after helper Arm");
      return LifecycleStatus::kFatalFailure;
    }
    return LifecycleStatus::kOk;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    auto value = in.receive();
    if (!value)
      return make_unexpected(value.error());
    return out.emit(value->data);
  }
  Input<std::int32_t> in;
  Output<std::int32_t> out;

 private:
  std::shared_ptr<RateProbe> probe_;
};
class RateFollower final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(driver, "driver");
    spec.input(side, "side");
  }
  Contract contract() const override {
    Contract c;
    c.trigger(OnEach{driver})
        .input(side, LatestWithin{.max_age = 1s})
        .on_missing(side, SafeDefault{});
    return c;
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Input<std::int32_t> driver;
  Input<std::int32_t> side;
};
expected<void, Error> release(RunningFixture& running) {
  return running.release_frame(5s);
}
}  // namespace

// REQ: OPERATOR-068
// REQ: OPERATOR-069
TEST(OperatorTestingHelperRestart, NativeClosureRearmsRollsBackAndReusesSenderRecords) {
  auto probe = std::make_shared<RateProbe>();
  auto h = create_operator_test<RateDriver>(probe);
  for (std::int32_t value = 1; value <= 6; ++value)
    h.push_frame(frame_at(std::chrono::milliseconds{value}, input("in", value)));
  h.capture<std::int32_t>("out");
  h.configure_graph([](auto& view) {
    auto follower = view.graph().template op<RateFollower>("rate_follower");
    view.graph().add_flow(
        view.tested_operator()->out, follower->driver, ConnectionOptions{.queue_depth = 8U});
    view.graph().add_flow(view.template source_output<std::int32_t>("in"),
                          follower->side,
                          ConnectionOptions{.queue_depth = 8U});
    view.graph().lifecycle().allow_external_restart(view.tested_operator());
    view.compile_options().lifecycle_transactions =
        LifecycleTransactionOptions{.max_pending_requests = 8U,
                                    .max_pending_transactions = 8U,
                                    .max_receipts = 16U,
                                    .max_fenced_operators = 16U,
                                    .max_candidate_generations = 16U,
                                    .max_events = 128U,
                                    .max_audit_records = 32U,
                                    .max_quiesce_records = 64U,
                                    .max_rollback_records = 64U};
  });
  auto running = h.run_async({.frames = FrameRelease::kManual});
  const auto first_post = release(running);
  ASSERT_TRUE(first_post) << running.finish().describe_failure();
  ASSERT_TRUE(running.wait_for_output("out", 1U, 5s));
  const auto target = running.operator_ref();
  ASSERT_TRUE(target);
  const std::array targets{*target};
  constexpr std::array stages{LifecycleStage::kArm, LifecycleStage::kAllocate};
  auto control =
      running.lifecycle_control({.allowed_targets = targets, .allowed_from_stages = stages});
  ASSERT_TRUE(control);
  for (std::size_t attempt = 0; attempt < 5U; ++attempt) {
    probe->fail_start = attempt == 1U;
    probe->fail_warm_up = attempt == 3U;
    const auto request = control->request_restart(
        *target, {.from_stage = attempt == 3U ? LifecycleStage::kAllocate : LifecycleStage::kArm});
    ASSERT_TRUE(request);
    const auto terminal = control->await(*request, 5s);
    ASSERT_TRUE(terminal);
    ASSERT_EQ(terminal->state,
              (attempt == 1U || attempt == 3U) ? LifecycleTransactionState::kRolledBack
                                               : LifecycleTransactionState::kPublished);
    ASSERT_TRUE(release(running));
    ASSERT_TRUE(running.wait_for_output("out", attempt + 2U, 5s));
  }
  const auto result = running.finish();
  ASSERT_FALSE(result.plan().lifecycle_restart_closures().empty());
  // Tested operator, follower and frame source belong to the compiler closure.
  const auto closures = result.plan().lifecycle_restart_closures();
  const auto closure = std::find_if(closures.begin(), closures.end(), [&](const auto& row) {
    return row.target.id == target->id;
  });
  ASSERT_NE(closure, closures.end());
  const auto operators = result.plan().operator_parameters();
  const auto helper = std::find_if(operators.begin(), operators.end(), [](const auto& row) {
    const auto path = OperatorPath::parse(row.operator_path);
    return path && path->segments().size() == 1U && path->segments().front() == "frame_source";
  });
  ASSERT_NE(helper, operators.end());
  EXPECT_NE(std::find(closure->closure.begin(), closure->closure.end(), helper->owner),
            closure->closure.end());
  EXPECT_EQ(result.termination(), RunTermination::kCompleted) << result.describe_failure();
  EXPECT_EQ(result.values<std::int32_t>("out"), (std::vector<std::int32_t>{1, 2, 3, 4, 5, 6}));
}
