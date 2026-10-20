/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/run.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "testing/operator_test_run.hpp"

namespace {
using namespace std::chrono_literals;

class ManagedScale final : public holoscan::Operator<> {
 public:
  explicit ManagedScale(bool fail_start = false) : fail_start_(fail_start) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in");
    spec.output(out, "out");
  }
  void start() override {
    if (fail_start_)
      throw std::runtime_error{"intentional startup failure"};
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = in.receive();
    if (!sample)
      return holoscan::make_unexpected(std::move(sample).error());
    return out.emit(sample->data * 2);
  }
  holoscan::Input<std::int32_t> in;
  holoscan::Output<std::int32_t> out;

 private:
  bool fail_start_;
};

auto scale_fixture(bool fail_start = false) {
  auto h = holoscan::testing::create_operator_test<ManagedScale>(fail_start);
  h.push_many<std::int32_t>("in", {3, 5});
  h.capture<std::int32_t>("out");
  return h;
}

struct StopProbe {
  std::atomic<int> starts{};
  std::atomic<int> stops{};
  std::atomic<int> computes{};
  std::mutex mutex;
  std::condition_variable changed;
  bool wait_for_start(int count) {
    std::unique_lock lock(mutex);
    return changed.wait_for(lock, 5s, [&] { return starts.load() >= count; });
  }
  bool wait_for_compute() {
    std::unique_lock lock(mutex);
    return changed.wait_for(lock, 5s, [&] { return computes.load() != 0; });
  }
};

class IdleManagedSource final : public holoscan::Operator<> {
 public:
  explicit IdleManagedSource(std::shared_ptr<StopProbe> probe) : probe_(std::move(probe)) {}
  void setup(holoscan::OperatorSpec&) override {}
  holoscan::Contract contract() const override {
    holoscan::Contract c;
    c.trigger(holoscan::OnClock{.period = 1h});
    return c;
  }
  void start() override {
    const std::lock_guard lock(probe_->mutex);
    ++probe_->starts;
    probe_->changed.notify_all();
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    const std::lock_guard lock(probe_->mutex);
    ++probe_->computes;
    probe_->changed.notify_all();
    return {};
  }
  void stop() override { ++probe_->stops; }

 private:
  std::shared_ptr<StopProbe> probe_;
};
}  // namespace

// REQ: OPERATOR-066
TEST(OperatorTestingManaged, AsyncFinishMatchesBlockingAndIsIdempotent) {
  auto blocking = scale_fixture().run();
  auto h = scale_fixture();
  auto running = h.run_async();
  const auto first = running.finish();
  const auto again = running.finish();
  ASSERT_TRUE(first.ok()) << first.describe_failure();
  EXPECT_EQ(first.values<std::int32_t>("out"), blocking.values<std::int32_t>("out"));
  EXPECT_EQ(first.values<std::int32_t>("out"), (std::vector<std::int32_t>{6, 10}));
  EXPECT_EQ(&first.plan(), &again.plan());
  EXPECT_FALSE(running.is_running());
  EXPECT_EQ(running.termination(), first.termination());
  running.request_stop();
}

// REQ: OPERATOR-066
TEST(OperatorTestingManaged, RunningOwnerSurvivesBuilderMoveAndDestruction) {
  auto running = [] {
    auto h = scale_fixture();
    auto r = h.run_async();
    auto moved_builder = std::move(h);
    EXPECT_THROW(moved_builder.push("in", std::int32_t{9}), std::logic_error);
    return r;
  }();
  auto moved = std::move(running);
  const auto result = moved.finish();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.values<std::int32_t>("out"), (std::vector<std::int32_t>{6, 10}));
}

// REQ: OPERATOR-066
TEST(OperatorTestingManaged, AdmissionLeaseLastsUntilJoinAndThenAllowsFreshAttempt) {
  auto h = scale_fixture();
  auto running = h.run_async();
  EXPECT_THROW(h.push("in", std::int32_t{7}), std::logic_error);
  EXPECT_THROW((void)h.run_async(), std::logic_error);
  const auto previous = running.finish();
  ASSERT_TRUE(previous.ok()) << previous.describe_failure();
  h.push("in", std::int32_t{7});
  const auto fresh = h.run_async().finish();
  ASSERT_TRUE(fresh.ok()) << fresh.describe_failure();
  EXPECT_EQ(fresh.value<std::int32_t>("out"), 14);
  EXPECT_EQ(previous.values<std::int32_t>("out"), (std::vector<std::int32_t>{6, 10}));
}

// REQ: OPERATOR-066
TEST(OperatorTestingManaged, ColdHookExceptionPreservesInputsAndReleasesLease) {
  auto h = scale_fixture();
  h.configure_graph([](auto&) { throw std::runtime_error{"cold hook"}; });
  EXPECT_THROW((void)h.run_async(), std::runtime_error);
  h.configure_graph({});
  const auto result = h.run_async().finish();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.values<std::int32_t>("out"), (std::vector<std::int32_t>{6, 10}));
}

// REQ: OPERATOR-066
TEST(OperatorTestingManaged, CompileFailureIsTerminalAndPreservesInputs) {
  auto h = scale_fixture();
  h.configure_graph([](auto& view) { view.graph().partition("empty"); });
  auto rejected = h.run_async({.expect_stop = true});
  EXPECT_FALSE(rejected.is_running());
  const auto released = rejected.release_frame(5s);
  ASSERT_FALSE(released);
  EXPECT_EQ(released.error().code, holoscan::ErrorCode::kClosed);
  EXPECT_NE(released.error().message.find("rejected before start"), std::string::npos);
  EXPECT_NE(released.error().message.find("finish()"), std::string::npos);
  const auto identity = rejected.operator_ref();
  ASSERT_FALSE(identity);
  EXPECT_EQ(identity.error().code, holoscan::ErrorCode::kStaleOperatorRef);
  EXPECT_NE(identity.error().message.find("finish()"), std::string::npos);
  const auto control = rejected.lifecycle_control({});
  ASSERT_FALSE(control);
  EXPECT_EQ(control.error().code, holoscan::ErrorCode::kClosed);
  EXPECT_NE(control.error().message.find("rejected before start"), std::string::npos);
  const auto failure = rejected.finish();
  EXPECT_FALSE(failure.ok());
  EXPECT_FALSE(failure.diagnostics().empty());
  h.configure_graph({});
  const auto accepted = h.run_async().finish();
  ASSERT_TRUE(accepted.ok()) << accepted.describe_failure();
  EXPECT_EQ(accepted.values<std::int32_t>("out"), (std::vector<std::int32_t>{6, 10}));
}

// REQ: OPERATOR-066
TEST(OperatorTestingManaged, FailedStartupConsumesStaging) {
  auto h = scale_fixture(true);
  const auto failed = h.run_async({.expect_stop = true}).finish();
  EXPECT_FALSE(failed.ok());
  EXPECT_EQ(failed.termination(), holoscan::RunTermination::kFailed);
  // The second plan has no finite producer: attempted startup consumed the script.
  const auto next = h.run_async().finish();
  EXPECT_EQ(next.plan().graph().vertices().size() + 1U, failed.plan().graph().vertices().size());
  EXPECT_TRUE(next.values<std::int32_t>("out").empty());
}

// REQ: OPERATOR-066
TEST(OperatorTestingManaged, ExplicitStopAndDestructionJoinBeforeReleasingAttempt) {
  auto probe = std::make_shared<StopProbe>();
  auto h = holoscan::testing::create_operator_test<IdleManagedSource>(probe);
  {
    auto running = h.run_async();
    ASSERT_TRUE(probe->wait_for_start(1));
    running.request_stop();
    const auto result = running.finish();
    EXPECT_EQ(result.termination(), holoscan::RunTermination::kStopped);
    EXPECT_FALSE(result.ok());
  }
  const auto stopped = probe->stops.load();
  {
    auto running = h.run_async();
    ASSERT_TRUE(probe->wait_for_start(2));
  }
  EXPECT_EQ(probe->starts.load(), probe->stops.load());
  EXPECT_GE(probe->stops.load(), stopped);
  EXPECT_NO_THROW(h.set_compile_options({}));
}

// REQ: OPERATOR-067
TEST(OperatorTestingManaged, CaptureObservationIsBoundedAndPayloadsWaitForFinish) {
  auto h = scale_fixture();
  auto running = h.run_async();
  ASSERT_TRUE(running.wait_for_output("out", 2U, 5s));
  EXPECT_EQ(running.output_count("out"), 2U);
  EXPECT_FALSE(running.wait_for_output("out", 3U, 5s));
  EXPECT_THROW((void)running.output_count("missing"), std::invalid_argument);
  EXPECT_THROW((void)running.wait_for_output("out", 1U, -1ms), std::invalid_argument);
  const auto result = running.finish();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(running.output_count("out"), 2U);
}

namespace {
holoscan::LifecycleTransactionOptions managed_capacity() {
  return {.max_pending_requests = 8U,
          .max_pending_transactions = 8U,
          .max_receipts = 16U,
          .max_fenced_operators = 16U,
          .max_candidate_generations = 16U,
          .max_events = 128U,
          .max_audit_records = 32U,
          .max_quiesce_records = 64U,
          .max_rollback_records = 64U};
}
}  // namespace

// REQ: OPERATOR-068
TEST(OperatorTestingManaged, RestrictedNativeControlRestartsWithinTheSameAttempt) {
  auto probe = std::make_shared<StopProbe>();
  auto h = holoscan::testing::create_operator_test<IdleManagedSource>(probe);
  h.configure_graph([](auto& view) {
    view.graph().lifecycle().allow_external_restart(view.tested_operator());
    view.compile_options().lifecycle_transactions = managed_capacity();
  });
  auto running = h.run_async();
  ASSERT_TRUE(probe->wait_for_start(1));
  ASSERT_TRUE(probe->wait_for_compute());
  const auto target = running.operator_ref();
  ASSERT_TRUE(target);
  const std::array targets{*target};
  const std::array stages{holoscan::LifecycleStage::kStart};
  auto control =
      running.lifecycle_control({.allowed_targets = targets, .allowed_from_stages = stages});
  ASSERT_TRUE(control);
  auto denied = running.lifecycle_control({});
  ASSERT_TRUE(denied);
  const auto request =
      control->request_restart(*target, {.from_stage = holoscan::LifecycleStage::kStart});
  ASSERT_TRUE(request);
  const auto receipt = control->await(*request, 5s);
  ASSERT_TRUE(receipt);
  EXPECT_EQ(receipt->state, holoscan::LifecycleTransactionState::kPublished);
  EXPECT_EQ(denied->request_restart(*target).error().code,
            holoscan::ErrorCode::kLifecycleRestartUnauthorized);
  EXPECT_EQ(probe->starts.load(), 2);
  EXPECT_EQ(running.operator_ref()->run_generation, target->run_generation);
  running.request_stop();
  const auto result = running.finish();
  EXPECT_EQ(result.termination(), holoscan::RunTermination::kStopped);
  EXPECT_EQ(running.operator_ref().error().code, holoscan::ErrorCode::kStaleOperatorRef);
  EXPECT_EQ(control->inspect(*request)->state, holoscan::LifecycleTransactionState::kPublished);
}

namespace {
holoscan::expected<void, holoscan::Error> release_next(holoscan::testing::RunningFixture& running) {
  return running.release_frame(5s);
}
}  // namespace

// REQ: OPERATOR-069
TEST(OperatorTestingManaged, ManualFramesResumeAfterNativeRestartWithoutReplayingThePrefix) {
  auto h = holoscan::testing::create_operator_test<ManagedScale>();
  h.push_frame(holoscan::testing::frame_at(10ms, holoscan::testing::input("in", std::int32_t{3})));
  h.push_frame(holoscan::testing::frame_at(20ms, holoscan::testing::input("in", std::int32_t{5})));
  h.capture<std::int32_t>("out");
  h.configure_graph([](auto& view) {
    view.graph().lifecycle().allow_external_restart(view.tested_operator());
    view.compile_options().lifecycle_transactions = managed_capacity();
  });
  auto running = h.run_async({.frames = holoscan::testing::FrameRelease::kManual});
  EXPECT_EQ(running.output_count("out"), 0U);
  ASSERT_TRUE(release_next(running));
  ASSERT_TRUE(running.wait_for_output("out", 1U, 5s));
  const auto target = running.operator_ref();
  ASSERT_TRUE(target);
  const std::array targets{*target};
  const std::array stages{holoscan::LifecycleStage::kStart};
  auto control =
      running.lifecycle_control({.allowed_targets = targets, .allowed_from_stages = stages});
  ASSERT_TRUE(control);
  const auto request =
      control->request_restart(*target, {.from_stage = holoscan::LifecycleStage::kStart});
  ASSERT_TRUE(request);
  const auto receipt = control->await(*request, 5s);
  ASSERT_TRUE(receipt);
  ASSERT_EQ(receipt->state, holoscan::LifecycleTransactionState::kPublished);
  ASSERT_TRUE(release_next(running));
  ASSERT_TRUE(running.wait_for_output("out", 2U, 5s));
  const auto result = running.finish();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.values<std::int32_t>("out"), (std::vector<std::int32_t>{6, 10}));
  EXPECT_EQ(running.release_frame().error().code, holoscan::ErrorCode::kClosed);
}

// REQ: OPERATOR-069
TEST(OperatorTestingManaged, ManualAdmissionRejectsIndependentStagingWithoutConsumingIt) {
  auto h = scale_fixture();
  EXPECT_THROW((void)h.run_async({.frames = holoscan::testing::FrameRelease::kManual}),
               std::invalid_argument);
  auto running = h.run_async();
  EXPECT_EQ(running.release_frame().error().code, holoscan::ErrorCode::kInvalidArgument);
  const auto result = running.finish();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.values<std::int32_t>("out"), (std::vector<std::int32_t>{6, 10}));
}

// REQ: OPERATOR-066
TEST(OperatorTestingManaged, WatchdogIsCooperativeAndInvalidTimeoutPreservesInputs) {
  auto h = scale_fixture();
  EXPECT_THROW((void)h.run_async({.timeout = 0ms}), std::invalid_argument);
  const auto result = h.run_async().finish();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  auto probe = std::make_shared<StopProbe>();
  auto idle = holoscan::testing::create_operator_test<IdleManagedSource>(probe);
  auto running = idle.run_async({.timeout = 50ms, .expect_stop = true});
  const auto timed_out = running.finish();
  EXPECT_FALSE(timed_out.ok());
  EXPECT_EQ(timed_out.termination(), holoscan::RunTermination::kStopped);
  EXPECT_NE(timed_out.describe_failure().find("watchdog expired"), std::string::npos);
  // Startup can lose the short-deadline race. ExplicitStopAndDestructionJoinBeforeReleasingAttempt
  // separately observes Start before asserting joined lifecycle cleanup.
}

// REQ: OPERATOR-067
TEST(OperatorTestingManaged, FailedCaptureDoesNotIncrementTheObservedPrefix) {
  auto h = scale_fixture();
  h.capture<std::int32_t>("out", 1U);
  h.output_flow("out", holoscan::ConnectionOptions{.queue_depth = 2U});
  auto run = h.run_async({.expect_stop = true});
  ASSERT_TRUE(run.wait_for_output("out", 1U, 5s));
  EXPECT_FALSE(run.wait_for_output("out", 2U, 5s));
  const auto result = run.finish();
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(run.output_count("out"), 1U);
  EXPECT_EQ(result.value<std::int32_t>("out"), 6);
  EXPECT_TRUE(result.capture_failure("out"));
}

// REQ: OPERATOR-068
TEST(OperatorTestingManaged, TerminalReceiptsStillConsumeTheNativeFiniteReceiptBudget) {
  auto probe = std::make_shared<StopProbe>();
  auto h = holoscan::testing::create_operator_test<IdleManagedSource>(probe);
  h.configure_graph([](auto& view) {
    view.graph().lifecycle().allow_external_restart(view.tested_operator());
    auto options = managed_capacity();
    options.max_receipts = 4U;
    options.max_pending_requests = 2U;
    options.max_pending_transactions = 2U;
    view.compile_options().lifecycle_transactions = options;
  });
  auto run = h.run_async();
  ASSERT_TRUE(probe->wait_for_start(1));
  ASSERT_TRUE(probe->wait_for_compute());
  const auto target = run.operator_ref();
  ASSERT_TRUE(target);
  const std::array targets{*target};
  constexpr std::array stages{holoscan::LifecycleStage::kStart};
  auto control = run.lifecycle_control({.allowed_targets = targets, .allowed_from_stages = stages});
  ASSERT_TRUE(control);
  std::array<holoscan::LifecycleRestartReceipt, 4> receipts;
  for (auto& receipt : receipts) {
    const auto request =
        control->request_restart(*target, {.from_stage = holoscan::LifecycleStage::kStart});
    ASSERT_TRUE(request);
    receipt = *request;
    const auto terminal = control->await(receipt, 5s);
    ASSERT_TRUE(terminal);
    ASSERT_EQ(terminal->state, holoscan::LifecycleTransactionState::kPublished);
  }
  const auto exhausted =
      control->request_restart(*target, {.from_stage = holoscan::LifecycleStage::kStart});
  ASSERT_FALSE(exhausted);
  EXPECT_EQ(exhausted.error().code, holoscan::ErrorCode::kLifecycleRestartQueueFull);
  for (const auto& receipt : receipts)
    EXPECT_EQ(control->inspect(receipt)->state, holoscan::LifecycleTransactionState::kPublished);
  run.request_stop();
  EXPECT_EQ(run.finish().termination(), holoscan::RunTermination::kStopped);
}

namespace {
// Bound a regression that would otherwise wait forever for an unreachable manual driver.
// The caller still uses an outer test-process deadline for a broken native stop/join path.
bool join_promptly(holoscan::RunSession& session) {
  auto joined = std::async(std::launch::async, [&] { session.wait(); });
  const bool completed = joined.wait_for(5s) == std::future_status::ready;
  if (!completed)
    session.request_stop();
  joined.get();
  return completed;
}

std::string replay_diagnostics(const holoscan::ExecutionPlan& plan,
                               const holoscan::RunSession& session) {
  std::string result;
  for (const auto& diagnostic : session.diagnostics()) {
    result += holoscan::format_diagnostic(diagnostic, plan.graph());
    result += '\n';
  }
  return result;
}

class FailFirstWarmUp final : public holoscan::Operator<> {
 public:
  explicit FailFirstWarmUp(std::shared_ptr<std::atomic<bool>> first) : first_(std::move(first)) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in");
    spec.output(out, "out");
    spec.lifecycle().stage(holoscan::LifecycleStage::kWarmUp, &FailFirstWarmUp::warm_up);
  }
  holoscan::LifecycleStatus warm_up(holoscan::LifecycleContext& context) noexcept {
    if (first_->exchange(false)) {
      context.report_error(holoscan::ErrorCode::kFailure,
                           "initial WarmUp failed before helper Arm");
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    return holoscan::LifecycleStatus::kOk;
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

struct WarmUpGate {
  std::mutex mutex;
  std::condition_variable changed;
  bool entered{};
  bool released{};
  bool wait_until_entered() {
    std::unique_lock lock(mutex);
    return changed.wait_for(lock, 5s, [&] { return entered; });
  }
  void release() {
    const std::lock_guard lock(mutex);
    released = true;
    changed.notify_all();
  }
};
struct ReleaseWarmUp {
  std::shared_ptr<WarmUpGate> gate;
  ~ReleaseWarmUp() { gate->release(); }
};
class GatedWarmUp final : public holoscan::Operator<> {
 public:
  explicit GatedWarmUp(std::shared_ptr<WarmUpGate> gate) : gate_(std::move(gate)) {}
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in");
    spec.output(out, "out");
    spec.lifecycle().stage(holoscan::LifecycleStage::kWarmUp, &GatedWarmUp::warm_up);
  }
  holoscan::LifecycleStatus warm_up(holoscan::LifecycleContext& context) noexcept {
    std::unique_lock lock(gate_->mutex);
    gate_->entered = true;
    gate_->changed.notify_all();
    if (!gate_->changed.wait_for(lock, 5s, [&] { return gate_->released; })) {
      context.report_error(holoscan::ErrorCode::kTimeout, "test WarmUp gate was not released");
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    return holoscan::LifecycleStatus::kOk;
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
  std::shared_ptr<WarmUpGate> gate_;
};
}  // namespace

// REQ: OPERATOR-066
// REQ: OPERATOR-069
TEST(OperatorTestingManaged, CompletedManualPlanRejectsUndrivenReplayPromptly) {
  using namespace holoscan::testing;
  auto h = create_operator_test<ManagedScale>();
  h.push_frame(frame_at(0ms, input("in", std::int32_t{3})));
  h.capture<std::int32_t>("out");
  auto running = h.run_async({.frames = FrameRelease::kManual});
  ASSERT_TRUE(running.release_frame(5s));
  const auto original = running.finish();
  ASSERT_TRUE(original.ok()) << original.describe_failure();
  EXPECT_EQ(original.value<std::int32_t>("out"), 6);

  auto replay = holoscan::run_async(original.plan());
  EXPECT_TRUE(join_promptly(replay));
  EXPECT_EQ(replay.termination(), holoscan::RunTermination::kFailed);
  const auto diagnostics = replay_diagnostics(original.plan(), replay);
  EXPECT_NE(diagnostics.find("manual frame plan cannot be replayed"), std::string::npos);
  EXPECT_NE(diagnostics.find("new fixture attempt"), std::string::npos);
  EXPECT_EQ(original.value<std::int32_t>("out"), 6);
}

// REQ: OPERATOR-066
// REQ: OPERATOR-069
TEST(OperatorTestingManaged, ManualReplayRejectsAfterOriginalFailureBeforeArm) {
  using namespace holoscan::testing;
  auto first = std::make_shared<std::atomic<bool>>(true);
  auto h = create_operator_test<FailFirstWarmUp>(first);
  h.push_frame(frame_at(0ms, input("in", std::int32_t{7})));
  h.capture<std::int32_t>("out");
  const auto failed = h.run_async({.frames = FrameRelease::kManual}).finish();
  ASSERT_FALSE(failed.ok());
  ASSERT_EQ(failed.termination(), holoscan::RunTermination::kFailed);
  ASSERT_FALSE(first->load());
  EXPECT_NE(failed.describe_failure().find("before helper Arm"), std::string::npos);

  auto replay = holoscan::run_async(failed.plan());
  EXPECT_TRUE(join_promptly(replay));
  EXPECT_EQ(replay.termination(), holoscan::RunTermination::kFailed);
  EXPECT_NE(replay_diagnostics(failed.plan(), replay).find("manual frame plan cannot be replayed"),
            std::string::npos);
  EXPECT_TRUE(failed.samples<std::int32_t>("out").empty());

  h.push_frame(frame_at(1ms, input("in", std::int32_t{9})));
  auto fresh = h.run_async({.frames = FrameRelease::kManual});
  ASSERT_TRUE(fresh.release_frame(5s));
  const auto accepted = fresh.finish();
  ASSERT_TRUE(accepted.ok()) << accepted.describe_failure();
  EXPECT_EQ(accepted.value<std::int32_t>("out"), 9);
}

// REQ: OPERATOR-066
TEST(OperatorTestingManaged, AutomaticGroupedPlanReplaysConcurrentlyWithIndependentCaptures) {
  using namespace holoscan::testing;
  auto h = create_operator_test<ManagedScale>();
  h.push_frame(frame_at(0ms, input("in", std::int32_t{3})));
  h.capture<std::int32_t>("out");
  const auto original = h.run_async().finish();
  ASSERT_TRUE(original.ok()) << original.describe_failure();
  auto first = holoscan::run_async(original.plan());
  auto second = holoscan::run_async(original.plan());
  EXPECT_TRUE(join_promptly(first));
  EXPECT_TRUE(join_promptly(second));
  EXPECT_EQ(first.termination(), holoscan::RunTermination::kCompleted);
  EXPECT_EQ(second.termination(), holoscan::RunTermination::kCompleted);
  EXPECT_EQ(original.values<std::int32_t>("out"), (std::vector<std::int32_t>{6}));
}

// REQ: OPERATOR-069
TEST(OperatorTestingManaged, BoundedReleaseDoesNotPostBeforeArmOrAfterInvalidDuration) {
  using namespace holoscan::testing;
  auto gate = std::make_shared<WarmUpGate>();
  auto h = create_operator_test<GatedWarmUp>(gate);
  h.push_frame(frame_at(0ms, input("in", std::int32_t{7})));
  h.capture<std::int32_t>("out");
  auto running = h.run_async({.frames = FrameRelease::kManual});
  const ReleaseWarmUp release{gate};
  ASSERT_TRUE(gate->wait_until_entered());
  const auto negative = running.release_frame(-1ms);
  ASSERT_FALSE(negative);
  EXPECT_EQ(negative.error().code, holoscan::ErrorCode::kInvalidArgument);
  const auto huge = running.release_frame(std::chrono::milliseconds::max());
  ASSERT_FALSE(huge);
  EXPECT_EQ(huge.error().code, holoscan::ErrorCode::kInvalidArgument);
  const auto immediate = running.release_frame(0ms);
  ASSERT_FALSE(immediate);
  EXPECT_EQ(immediate.error().code, holoscan::ErrorCode::kNotReady);
  EXPECT_EQ(immediate.error().message.find("rejected before start"), std::string::npos);
  const auto timed = running.release_frame(5ms);
  ASSERT_FALSE(timed);
  EXPECT_EQ(timed.error().code, holoscan::ErrorCode::kTimeout);
  EXPECT_EQ(running.output_count("out"), 0U);
  gate->release();
  ASSERT_TRUE(running.release_frame(5s));
  const auto result = running.finish();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.values<std::int32_t>("out"), (std::vector<std::int32_t>{7}));
  const auto closed = running.release_frame(5s);
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, holoscan::ErrorCode::kClosed);
  EXPECT_EQ(closed.error().message.find("rejected before start"), std::string::npos);
}

// REQ: OPERATOR-066
TEST(OperatorTestingManaged, ExplicitExpectedStopSucceedsOnlyAfterJoinedCleanStop) {
  auto probe = std::make_shared<StopProbe>();
  auto h = holoscan::testing::create_operator_test<IdleManagedSource>(probe);
  auto running = h.run_async({.expect_stop = true});
  ASSERT_TRUE(probe->wait_for_start(1));
  running.request_stop();
  const auto result = running.finish();
  EXPECT_EQ(result.termination(), holoscan::RunTermination::kStopped);
  EXPECT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(probe->starts.load(), probe->stops.load());
}

// REQ: OPERATOR-066
TEST(OperatorTestingManaged, MovedOwnerAndBuilderEntryPointsKeepTheirDefinedFailures) {
  auto h = scale_fixture();
  auto running = h.run_async();
  auto owner = std::move(running);
  auto builder = std::move(h);
  // Intentional contract probes: public operations retain their moved-from behavior.
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_THROW(static_cast<void>(h.run_async()), std::logic_error);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_FALSE(running.is_running());
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(running.termination(), holoscan::RunTermination::kFailed);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_NO_THROW(running.request_stop());
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_THROW(static_cast<void>(running.finish()), std::logic_error);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_THROW(static_cast<void>(running.output_count("out")), std::logic_error);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_THROW(static_cast<void>(running.wait_for_output("out", 1U, 0ms)), std::logic_error);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  const auto released = running.release_frame(0ms);
  ASSERT_FALSE(released);
  EXPECT_EQ(released.error().code, holoscan::ErrorCode::kInvalidArgument);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  const auto identity = running.operator_ref();
  ASSERT_FALSE(identity);
  EXPECT_EQ(identity.error().code, holoscan::ErrorCode::kStaleOperatorRef);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  const auto control = running.lifecycle_control({});
  ASSERT_FALSE(control);
  EXPECT_EQ(control.error().code, holoscan::ErrorCode::kClosed);
  const auto result = owner.finish();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.values<std::int32_t>("out"), (std::vector<std::int32_t>{6, 10}));
  EXPECT_NO_THROW(builder.set_compile_options({}));
}

// REQ: OPERATOR-066
TEST(OperatorTestingManaged, OptionalOwnerMustResetBeforeEvaluatingReplacementAttempt) {
  auto h = scale_fixture();
  std::optional<holoscan::testing::RunningFixture> running;
  running.emplace(h.run_async());
  // Argument evaluation reaches the builder while the previous owner still holds its lease.
  EXPECT_THROW(running.emplace(h.run_async()), std::logic_error);
  EXPECT_TRUE(running.has_value());
  running.reset();
  h.push("in", std::int32_t{7});
  auto& replacement = running.emplace(h.run_async());
  const auto result = replacement.finish();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<std::int32_t>("out"), 14);
}

// REQ: OPERATOR-066
// REQ: OPERATOR-069
TEST(OperatorTestingManaged, BlockingManualOptionsRejectBeforeConsumingGroupedFrames) {
  using namespace holoscan::testing;
  auto h = create_operator_test<ManagedScale>();
  h.push_frame(frame_at(0ms, input("in", std::int32_t{3})));
  h.capture<std::int32_t>("out");
  EXPECT_THROW(
      static_cast<void>(h.run(TestRunOptions{.timeout = 50ms, .frames = FrameRelease::kManual})),
      std::invalid_argument);
  auto running = h.run_async({.frames = FrameRelease::kManual});
  ASSERT_TRUE(running.release_frame(5s));
  const auto result = running.finish();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<std::int32_t>("out"), 6);
}

namespace {
struct HeldFinalInputProbe {
  unsigned computes{};
  unsigned stops{};
  bool held_at_stop{};
  bool released_in_stop{};
  std::int32_t value_at_stop{};
};

class HoldFinalInputUntilStop final : public holoscan::Operator<> {
 public:
  explicit HoldFinalInputUntilStop(std::shared_ptr<HeldFinalInputProbe> probe)
      : probe_(std::move(probe)) {}
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in"); }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    auto received = in.receive();
    if (!received)
      return holoscan::make_unexpected(std::move(received).error());
    // Move the complete Runtime sample, including its finite read claim. A copied scalar
    // alone would not exercise the borrowed-input lifetime or retirement boundary.
    held_.emplace(std::move(*received));
    ++probe_->computes;
    // Request while the final input is still retained. This removes timing dependence on
    // an external thread racing natural EOS completion after compute returns.
    context.request_stop();
    return {};
  }
  void stop() override {
    ++probe_->stops;
    probe_->held_at_stop = held_.has_value();
    if (held_)
      probe_->value_at_stop = held_->data;
    held_.reset();
    probe_->released_in_stop = !held_.has_value();
  }
  holoscan::Input<std::int32_t> in;

 private:
  std::shared_ptr<HeldFinalInputProbe> probe_;
  std::optional<holoscan::Sample<std::int32_t>> held_;
};
}  // namespace

// REQ: OPERATOR-066
TEST(OperatorTestingManaged, ExpectedStopJoinsAfterFinalBorrowedInputIsReleasedByStop) {
  using namespace std::chrono_literals;
  for (const bool expect_stop : {false, true}) {
    SCOPED_TRACE(expect_stop);
    auto probe = std::make_shared<HeldFinalInputProbe>();
    auto h = holoscan::testing::create_operator_test<HoldFinalInputUntilStop>(probe);
    h.push("in", std::int32_t{73});
    auto running = h.run_async({.timeout = 5s, .expect_stop = expect_stop});
    const auto result = running.finish();
    EXPECT_EQ(result.termination(), holoscan::RunTermination::kStopped)
        << result.describe_failure();
    EXPECT_EQ(result.ok(), expect_stop) << result.describe_failure();
    EXPECT_TRUE(result.diagnostics().empty()) << result.describe_failure();
    EXPECT_FALSE(running.is_running());
    EXPECT_EQ(probe->computes, 1U);
    EXPECT_EQ(probe->stops, 1U);
    EXPECT_TRUE(probe->held_at_stop);
    EXPECT_EQ(probe->value_at_stop, 73);
    EXPECT_TRUE(probe->released_in_stop);
    // No second cleanup or collection, and the fixture lease is released only after join.
    EXPECT_EQ(running.finish().ok(), expect_stop);
    EXPECT_EQ(probe->stops, 1U);
    EXPECT_NO_THROW(h.set_compile_options({}));
    if (expect_stop)
      EXPECT_TRUE(result.describe_failure().empty());
  }
}

TEST(OperatorTestingManaged, FrameReleaseRechecksDeadlineAfterMutexContention) {
  holoscan::testing::detail::FrameReleaseState frames;
  std::unique_lock held(frames.mutex);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{100};
  std::promise<void> entered;
  auto entered_future = entered.get_future();
  auto pending = std::async(std::launch::async, [&] {
    entered.set_value();
    return frames.post(deadline);
  });
  // Always unlock before getting/destroying the asynchronous owner, including failed assertions.
  entered_future.wait();
  EXPECT_EQ(pending.wait_until(deadline), std::future_status::timeout);
  held.unlock();
  const auto expired = pending.get();
  ASSERT_FALSE(expired);
  EXPECT_EQ(expired.error().code, holoscan::ErrorCode::kTimeout);
  const auto immediate = frames.post();
  ASSERT_FALSE(immediate);
  EXPECT_EQ(immediate.error().code, holoscan::ErrorCode::kNotReady);
  frames.stop_requested = true;
  const auto stopped = frames.post(deadline);
  ASSERT_FALSE(stopped);
  EXPECT_EQ(stopped.error().code, holoscan::ErrorCode::kClosed);
  frames.stop_requested = false;
  frames.driver_closed = true;
  const auto closed = frames.post(deadline);
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, holoscan::ErrorCode::kClosed);
}

// REQ: OPERATOR-066
// REQ: OPERATOR-069
TEST(OperatorTestingManaged, ManualStopBeforeArmPreservesCleanExpectedStop) {
  using namespace holoscan::testing;
  auto gate = std::make_shared<WarmUpGate>();
  auto h = create_operator_test<GatedWarmUp>(gate);
  h.push_frame(frame_at(0ms, input("in", std::int32_t{7})));
  h.capture<std::int32_t>("out");
  auto running = h.run_async({.expect_stop = true, .frames = FrameRelease::kManual});
  // WarmUp must be released before the running owner is joined on any exit path.
  const ReleaseWarmUp release{gate};
  ASSERT_TRUE(gate->wait_until_entered());

  running.request_stop();
  ASSERT_TRUE(running.is_running());
  const auto immediate = running.release_frame();
  EXPECT_FALSE(immediate);
  if (!immediate)
    EXPECT_EQ(immediate.error().code, holoscan::ErrorCode::kClosed);
  const auto bounded = running.release_frame(5ms);
  EXPECT_FALSE(bounded);
  if (!bounded)
    EXPECT_EQ(bounded.error().code, holoscan::ErrorCode::kClosed);

  gate->release();
  const auto result = running.finish();
  EXPECT_EQ(result.termination(), holoscan::RunTermination::kStopped);
  EXPECT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_TRUE(result.values<std::int32_t>("out").empty());
}
