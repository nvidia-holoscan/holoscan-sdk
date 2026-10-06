/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

namespace {
using namespace std::chrono_literals;
using namespace holoscan;
using namespace holoscan::testing;

enum class FailureMode { kNone, kForward, kRollback, kRestore, kComputeAfterRequest };
struct StageEvent {
  LifecycleStage stage;
  LifecycleTransitionPhase phase;
  std::uintptr_t object;
  bool compute_returned;
};
struct RestartProbe {
  std::mutex mutex;
  std::condition_variable changed;
  bool blocked{};
  bool released{};
  bool self{};
  bool hold_loan{};
  bool loan_live{};
  bool emitted_after_request{};
  FailureMode failure{};
  std::optional<LifecycleRestartReceipt> self_receipt;
  std::array<StageEvent, 128> events{};
  std::size_t size{};
  std::vector<int> ingested;

  bool wait_blocked() {
    std::unique_lock lock(mutex);
    return changed.wait_for(lock, 5s, [&] { return blocked; });
  }
  void release() {
    const std::lock_guard lock(mutex);
    released = true;
    changed.notify_all();
  }
  void record(StageEvent event) {
    const std::lock_guard lock(mutex);
    if (size < events.size())
      events[size++] = event;
    changed.notify_all();
  }
  std::vector<StageEvent> snapshot() {
    const std::lock_guard lock(mutex);
    return {events.begin(), events.begin() + size};
  }
};
struct ReleaseCompute {
  std::shared_ptr<RestartProbe> probe;
  ~ReleaseCompute() { probe->release(); }
};

class RestartOperator final : public Operator<> {
 public:
  explicit RestartOperator(std::shared_ptr<RestartProbe> probe) : probe_(std::move(probe)) {}
  void setup(OperatorSpec& spec) override {
    spec.input(in, "in");
    spec.output(out, "out");
    spec.lifecycle()
        .stage(LifecycleStage::kConfigure, &RestartOperator::configure)
        .stage(LifecycleStage::kAllocate, &RestartOperator::allocate)
        .stage(LifecycleStage::kWarmUp, &RestartOperator::warm_up)
        .stage(LifecycleStage::kStart, &RestartOperator::begin)
        .stage(LifecycleStage::kStop, &RestartOperator::end)
        .stage(LifecycleStage::kRelease, &RestartOperator::release);
  }
  LifecycleStatus configure(LifecycleContext& c) noexcept {
    return stage(c, LifecycleStage::kConfigure);
  }
  LifecycleStatus allocate(LifecycleContext& c) noexcept {
    return stage(c, LifecycleStage::kAllocate);
  }
  LifecycleStatus warm_up(LifecycleContext& c) noexcept {
    return stage(c, LifecycleStage::kWarmUp);
  }
  LifecycleStatus begin(LifecycleContext& c) noexcept { return stage(c, LifecycleStage::kStart); }
  LifecycleStatus end(LifecycleContext& c) noexcept { return stage(c, LifecycleStage::kStop); }
  LifecycleStatus release(LifecycleContext& c) noexcept {
    return stage(c, LifecycleStage::kRelease);
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = in.receive();
    if (!sample)
      return make_unexpected(std::move(sample).error());
    if (sample->data != 1)
      return out.emit(sample->data * 10);
    std::optional<LoanedSample<std::int32_t>> loan;
    if (probe_->hold_loan) {
      auto allocated = out.allocate();
      if (!allocated)
        return make_unexpected(allocated.error());
      loan.emplace(std::move(*allocated));
      const std::lock_guard lock(probe_->mutex);
      probe_->loan_live = true;
    }
    if (probe_->self) {
      const auto request = context.request_restart();
      if (!request)
        return make_unexpected(request.error());
      const std::lock_guard lock(probe_->mutex);
      probe_->self_receipt = *request;
    }
    {
      std::unique_lock lock(probe_->mutex);
      probe_->blocked = true;
      probe_->changed.notify_all();
      if (!probe_->changed.wait_for(lock, 5s, [&] { return probe_->released; }))
        return make_unexpected(Error{ErrorCode::kTimeout, "test callback was not released"});
    }
    if (loan)
      loan->cancel();
    probe_->record({LifecycleStage::kStart,
                    LifecycleTransitionPhase::kNone,
                    reinterpret_cast<std::uintptr_t>(this),
                    true});
    if (probe_->failure == FailureMode::kComputeAfterRequest) {
      auto emitted = out.emit(std::int32_t{99});
      if (!emitted)
        return emitted;
      probe_->emitted_after_request = true;
      return make_unexpected(Error{ErrorCode::kFailure, "failure after accepted request and emit"});
    }
    return {};
  }
  Input<std::int32_t> in;
  Output<std::int32_t> out;

 private:
  LifecycleStatus stage(LifecycleContext& c, LifecycleStage stage) noexcept {
    const auto phase = c.transition_phase();
    probe_->record({stage, phase, reinterpret_cast<std::uintptr_t>(this), false});
    const bool primary = probe_->failure != FailureMode::kNone &&
                         probe_->failure != FailureMode::kComputeAfterRequest &&
                         phase == LifecycleTransitionPhase::kRestartForward &&
                         stage == LifecycleStage::kWarmUp;
    const bool rollback = probe_->failure == FailureMode::kRollback &&
                          phase == LifecycleTransitionPhase::kForwardRollback &&
                          stage == LifecycleStage::kRelease;
    const bool restore = probe_->failure == FailureMode::kRestore &&
                         phase == LifecycleTransitionPhase::kRestoreOld &&
                         stage == LifecycleStage::kConfigure;
    if (primary || rollback || restore) {
      c.report_error(
          ErrorCode::kInvalidArgument,
          primary ? "candidate failed" : (rollback ? "rollback failed" : "restoration failed"));
      return LifecycleStatus::kFatalFailure;
    }
    return LifecycleStatus::kOk;
  }
  std::shared_ptr<RestartProbe> probe_;
};

LifecycleTransactionOptions capacity(std::uint32_t requests = 8U) {
  return {.max_pending_requests = requests,
          .max_pending_transactions = 8U,
          .max_receipts = 16U,
          .max_fenced_operators = 16U,
          .max_candidate_generations = 16U,
          .max_events = 128U,
          .max_audit_records = 32U,
          .max_quiesce_records = 64U,
          .max_rollback_records = 64U};
}

OperatorTestHarness<RestartOperator> make_fixture(std::shared_ptr<RestartProbe> probe,
                                                  const std::vector<int>& values,
                                                  std::uint32_t requests = 8U) {
  auto h = create_operator_test<RestartOperator>(std::move(probe));
  for (const auto value : values)
    h.push_frame(frame_at(std::chrono::milliseconds{value}, input("in", std::int32_t{value})));
  h.capture<std::int32_t>("out");
  h.configure_graph([requests](auto& view) {
    view.graph()
        .lifecycle()
        .allow_self_restart(view.tested_operator())
        .allow_external_restart(view.tested_operator());
    view.compile_options().lifecycle_transactions = capacity(requests);
  });
  return h;
}

expected<void, Error> post(RunningFixture& run) {
  return run.release_frame(5s);
}

expected<LifecycleControlHandle, Error> control_for(RunningFixture& run) {
  auto target = run.operator_ref();
  if (!target)
    return make_unexpected(target.error());
  const std::array targets{*target};
  constexpr std::array stages{LifecycleStage::kConfigure,
                              LifecycleStage::kAllocate,
                              LifecycleStage::kWarmUp,
                              LifecycleStage::kStart};
  return run.lifecycle_control({.allowed_targets = targets, .allowed_from_stages = stages});
}

bool has_phase(const TestResult& result, LifecycleTransitionPhase phase) {
  return std::any_of(result.diagnostics().begin(), result.diagnostics().end(), [&](const auto& d) {
    return d.detail && d.detail->phase == phase;
  });
}
}  // namespace

// REQ: OPERATOR-068
// REQ: OPERATOR-069
TEST(OperatorTestingRestart, SelfRequestWaitsForRealComputeRetirementAndResumes) {
  auto probe = std::make_shared<RestartProbe>();
  probe->self = true;
  probe->hold_loan = true;
  auto h = make_fixture(probe, {1, 2});
  auto run = h.run_async({.frames = FrameRelease::kManual});
  const ReleaseCompute release{probe};
  ASSERT_TRUE(post(run));
  ASSERT_TRUE(probe->wait_blocked());
  auto control = control_for(run);
  ASSERT_TRUE(control);
  const auto request = probe->self_receipt;
  if (!request.has_value()) {
    ADD_FAILURE() << "operator did not publish its restart receipt";
    return;
  }
  EXPECT_TRUE(probe->loan_live);
  EXPECT_EQ(control->inspect(request.value())->state, LifecycleTransactionState::kQueued);
  EXPECT_EQ(control->await(request.value(), 0ns).error().code, ErrorCode::kTimeout);
  const auto before = probe->snapshot();
  EXPECT_TRUE(std::none_of(before.begin(), before.end(), [](const auto& event) {
    return event.phase == LifecycleTransitionPhase::kRestartForward;
  }));
  probe->release();
  const auto terminal = control->await(request.value(), 5s);
  ASSERT_TRUE(terminal);
  ASSERT_EQ(terminal->state, LifecycleTransactionState::kPublished);
  EXPECT_EQ(terminal->effective_stage, LifecycleStage::kConfigure);
  ASSERT_TRUE(post(run));
  ASSERT_TRUE(run.wait_for_output("out", 1U, 5s));
  const auto result = run.finish();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<std::int32_t>("out"), 20);
  const auto events = probe->snapshot();
  const auto returned =
      std::find_if(events.begin(), events.end(), [](const auto& e) { return e.compute_returned; });
  const auto restarted = std::find_if(events.begin(), events.end(), [](const auto& e) {
    return e.phase == LifecycleTransitionPhase::kRestartForward;
  });
  ASSERT_NE(returned, events.end());
  ASSERT_NE(restarted, events.end());
  EXPECT_LT(returned, restarted);
  EXPECT_EQ(returned->object, restarted->object);
}

// REQ: OPERATOR-068
TEST(OperatorTestingRestart, BoundedRequestsPreserveAcceptedReceiptsAndForeignChecks) {
  auto probe = std::make_shared<RestartProbe>();
  auto h = make_fixture(probe, {1, 2}, 2U);
  auto run = h.run_async({.frames = FrameRelease::kManual});
  const ReleaseCompute release{probe};
  ASSERT_TRUE(post(run));
  ASSERT_TRUE(probe->wait_blocked());
  auto control = control_for(run);
  ASSERT_TRUE(control);
  const auto target = run.operator_ref();
  ASSERT_TRUE(target);
  auto foreign = *target;
  ++foreign.run_generation;
  EXPECT_EQ(control->request_restart(foreign).error().code, ErrorCode::kCrossSessionOperatorRef);
  EXPECT_EQ(control->request_restart(*target, {.from_stage = LifecycleStage::kStop}).error().code,
            ErrorCode::kLifecycleInvalidRestartStage);
  const auto first = control->request_restart(*target);
  const auto second = control->request_restart(*target, {.from_stage = LifecycleStage::kAllocate});
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  const auto before = control->inspect(*first);
  ASSERT_TRUE(before);
  EXPECT_EQ(control->request_restart(*target, {.from_stage = LifecycleStage::kWarmUp}).error().code,
            ErrorCode::kLifecycleRestartQueueFull);
  const auto after = control->inspect(*first);
  ASSERT_TRUE(after);
  EXPECT_EQ(after->state, before->state);
  EXPECT_EQ(after->fence_id, before->fence_id);
  probe->release();
  ASSERT_TRUE(control->await(*first, 5s));
  ASSERT_TRUE(control->await(*second, 5s));
  ASSERT_TRUE(post(run));
  const auto result = run.finish();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<std::int32_t>("out"), 20);
  auto unknown = *first;
  unknown.receipt_id += 1000U;
  EXPECT_EQ(control->inspect(unknown).error().code, ErrorCode::kCrossSessionOperatorRef);
  EXPECT_EQ(control->inspect(*first)->state, LifecycleTransactionState::kPublished);
}

// REQ: OPERATOR-068
TEST(OperatorTestingRestart, StopCancelsPendingSelfRequestAndRetainsEarlierCapture) {
  for (const bool fail_compute : {false, true}) {
    SCOPED_TRACE(fail_compute);
    auto probe = std::make_shared<RestartProbe>();
    probe->self = true;
    if (fail_compute)
      probe->failure = FailureMode::kComputeAfterRequest;
    auto h = make_fixture(probe, {0, 1, 2});
    auto run = h.run_async({.frames = FrameRelease::kManual});
    const ReleaseCompute release{probe};
    ASSERT_TRUE(post(run));
    ASSERT_TRUE(run.wait_for_output("out", 1U, 5s));
    ASSERT_TRUE(post(run));
    ASSERT_TRUE(probe->wait_blocked());
    auto control = control_for(run);
    ASSERT_TRUE(control);
    const auto request = probe->self_receipt;
    if (!request.has_value()) {
      ADD_FAILURE() << "operator did not publish its restart receipt";
      return;
    }
    if (!fail_compute)
      run.request_stop();
    probe->release();
    const auto result = run.finish();
    EXPECT_EQ(result.termination(),
              fail_compute ? RunTermination::kFailed : RunTermination::kStopped);
    // Native failed-callback cleanup discards this activation's queued publication. The
    // previously collected prefix remains independent of that publication and its receipt.
    EXPECT_EQ(result.value<std::int32_t>("out"), 0);
    EXPECT_EQ(probe->emitted_after_request, fail_compute);
    EXPECT_EQ(control->inspect(request.value())->state, LifecycleTransactionState::kCanceled);
    const auto events = probe->snapshot();
    EXPECT_TRUE(std::none_of(events.begin(), events.end(), [](const auto& event) {
      return event.phase == LifecycleTransitionPhase::kRestartForward;
    }));
  }
}

// REQ: OPERATOR-068
TEST(OperatorTestingRestart, ForwardRollbackAndRestorationErrorsRemainDistinct) {
  for (const auto failure :
       {FailureMode::kForward, FailureMode::kRollback, FailureMode::kRestore}) {
    SCOPED_TRACE(static_cast<int>(failure));
    auto probe = std::make_shared<RestartProbe>();
    probe->failure = failure;
    auto h = make_fixture(probe, {0, 2});
    auto run = h.run_async({.frames = FrameRelease::kManual});
    ASSERT_TRUE(post(run));
    ASSERT_TRUE(run.wait_for_output("out", 1U, 5s));
    auto control = control_for(run);
    ASSERT_TRUE(control);
    const auto target = run.operator_ref();
    ASSERT_TRUE(target);
    const auto request = control->request_restart(*target);
    ASSERT_TRUE(request);
    const auto terminal = control->await(*request, 5s);
    ASSERT_TRUE(terminal);
    if (failure == FailureMode::kForward) {
      EXPECT_EQ(terminal->state, LifecycleTransactionState::kRolledBack);
      ASSERT_TRUE(post(run));
    } else {
      EXPECT_EQ(terminal->state, LifecycleTransactionState::kFailedStopped);
    }
    const auto result = run.finish();
    EXPECT_TRUE(has_phase(result, LifecycleTransitionPhase::kRestartForward));
    if (failure == FailureMode::kForward) {
      EXPECT_TRUE(result.ok()) << result.describe_failure();
      EXPECT_EQ(result.values<std::int32_t>("out"), (std::vector<std::int32_t>{0, 20}));
    } else {
      EXPECT_EQ(result.termination(), RunTermination::kFailed);
      EXPECT_EQ(result.value<std::int32_t>("out"), 0);
      EXPECT_TRUE(has_phase(result,
                            failure == FailureMode::kRollback
                                ? LifecycleTransitionPhase::kForwardRollback
                                : LifecycleTransitionPhase::kRestoreOld));
    }
  }
}

namespace {
class IngressProbe final : public Operator<> {
 public:
  explicit IngressProbe(std::shared_ptr<RestartProbe> probe) : probe_(std::move(probe)) {}
  void setup(OperatorSpec& spec) override { spec.input(in, "in"); }
  expected<void, Error> compute(ExecutionContext&) override {
    auto sample = in.receive();
    if (!sample)
      return make_unexpected(std::move(sample).error());
    const std::lock_guard lock(probe_->mutex);
    probe_->ingested.push_back(sample->data);
    probe_->changed.notify_all();
    return {};
  }
  Input<std::int32_t> in;

 private:
  std::shared_ptr<RestartProbe> probe_;
};
}  // namespace

// REQ: OPERATOR-068
// REQ: OPERATOR-069
TEST(OperatorTestingRestart, QueuedIngressUsesNativeDiscardPolicyAcrossPartitions) {
  auto probe = std::make_shared<RestartProbe>();
  auto h = make_fixture(probe, {0, 1, 2, 3});
  h.input_queue_depth("in", 4U);
  h.input_flow("in", ConnectionOptions{.queue_depth = 4U});
  h.configure_graph([probe](auto& view) {
    auto ingestion = view.graph().template op<IngressProbe>("ingestion", probe);
    view.graph().add_flow(view.template source_output<std::int32_t>("in"), ingestion->in);
    auto producer = view.graph().partition("producer");
    view.add_sources_to(producer);
    producer.add(ingestion);
    auto consumer = view.graph().partition("operator").add(view.tested_operator());
    view.add_collectors_to(consumer);
    view.graph().lifecycle().allow_external_restart(view.tested_operator());
    view.compile_options().lifecycle_transactions = capacity();
  });
  auto run = h.run_async({.frames = FrameRelease::kManual});
  const ReleaseCompute release{probe};
  ASSERT_TRUE(post(run));
  ASSERT_TRUE(run.wait_for_output("out", 1U, 5s));
  ASSERT_TRUE(post(run));
  ASSERT_TRUE(probe->wait_blocked());
  ASSERT_TRUE(post(run));
  {
    std::unique_lock lock(probe->mutex);
    // Runtime materializes all fan-out recipients at one activation cut before freezing
    // a recipient. Probe delivery therefore proves the operator's queued copy was ingested.
    ASSERT_TRUE(probe->changed.wait_for(lock, 5s, [&] { return probe->ingested.size() == 3U; }));
    EXPECT_EQ(probe->ingested, (std::vector<int>{0, 1, 2}));
  }
  auto control = control_for(run);
  ASSERT_TRUE(control);
  auto target = run.operator_ref();
  ASSERT_TRUE(target);
  auto request = control->request_restart(*target, {.from_stage = LifecycleStage::kStart});
  ASSERT_TRUE(request);
  EXPECT_EQ(control->await(*request, 0ns).error().code, ErrorCode::kTimeout);
  probe->release();
  auto terminal = control->await(*request, 5s);
  ASSERT_TRUE(terminal);
  ASSERT_EQ(terminal->state, LifecycleTransactionState::kPublished);
  ASSERT_TRUE(post(run));
  ASSERT_TRUE(run.wait_for_output("out", 2U, 5s));
  const auto result = run.finish();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.values<std::int32_t>("out"), (std::vector<std::int32_t>{0, 30}));
}

// REQ: OPERATOR-068
TEST(OperatorTestingRestart, CompatibleRequestsCoalesceAndReceiptsOutliveTheRunningOwner) {
  LifecycleControlHandle retained_control;
  LifecycleRestartReceipt retained_receipt;
  {
    auto probe = std::make_shared<RestartProbe>();
    auto h = make_fixture(probe, {1, 2});
    auto run = h.run_async({.frames = FrameRelease::kManual});
    const ReleaseCompute release{probe};
    ASSERT_TRUE(post(run));
    ASSERT_TRUE(probe->wait_blocked());
    auto control = control_for(run);
    ASSERT_TRUE(control);
    auto target = run.operator_ref();
    ASSERT_TRUE(target);
    const auto first = control->request_restart(*target);
    const auto second = control->request_restart(*target);
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    const auto first_view = control->inspect(*first);
    const auto second_view = control->inspect(*second);
    ASSERT_TRUE(first_view);
    ASSERT_TRUE(second_view);
    EXPECT_EQ(second_view->coalesced_into_transaction, first_view->transaction_id);
    EXPECT_EQ(second_view->fence_id, first_view->fence_id);
    probe->release();
    ASSERT_TRUE(control->await(*first, 5s));
    ASSERT_TRUE(control->await(*second, 5s));
    ASSERT_TRUE(post(run));
    EXPECT_TRUE(run.finish().ok());
    retained_control = *control;
    retained_receipt = *second;
  }
  const auto receipt = retained_control.inspect(retained_receipt);
  ASSERT_TRUE(receipt);
  EXPECT_EQ(receipt->state, LifecycleTransactionState::kPublished);
  EXPECT_EQ(retained_control.request_restart(retained_receipt.target).error().code,
            ErrorCode::kClosed);
}

// REQ: OPERATOR-067
// REQ: OPERATOR-069
TEST(OperatorTestingRestart, ManualNotificationBackpressureAndStopDoNotReplayTokens) {
  auto probe = std::make_shared<RestartProbe>();
  auto h = make_fixture(probe, {1, 2, 3});
  auto run = h.run_async({.frames = FrameRelease::kManual});
  const ReleaseCompute release{probe};
  ASSERT_TRUE(post(run));
  ASSERT_TRUE(probe->wait_blocked());
  ASSERT_TRUE(run.release_frame());
  const auto full = run.release_frame(0ms);
  ASSERT_FALSE(full);
  EXPECT_EQ(full.error().code, ErrorCode::kBackpressured);
  const auto timed = run.release_frame(5ms);
  ASSERT_FALSE(timed);
  EXPECT_EQ(timed.error().code, ErrorCode::kTimeout);
  run.request_stop();
  probe->release();
  const auto stopped = run.finish();
  EXPECT_TRUE(stopped.values<std::int32_t>("out").empty());
  EXPECT_EQ(run.release_frame().error().code, ErrorCode::kClosed);
  h.push_frame(frame_at(4ms, input("in", std::int32_t{4})));
  const auto fresh = h.run();
  ASSERT_TRUE(fresh.ok()) << fresh.describe_failure();
  EXPECT_EQ(fresh.values<std::int32_t>("out"), (std::vector<std::int32_t>{40}));
}

// REQ: OPERATOR-066
// REQ: OPERATOR-069
TEST(OperatorTestingRestart, ManualStopClosesAdmissionBeforeCallbackRetires) {
  auto probe = std::make_shared<RestartProbe>();
  auto h = make_fixture(probe, {1, 2});
  auto run = h.run_async({.expect_stop = true, .frames = FrameRelease::kManual});
  // Release the callback before run's destructor joins, including assertion failure paths.
  const ReleaseCompute release{probe};
  ASSERT_TRUE(post(run));
  ASSERT_TRUE(probe->wait_blocked());

  run.request_stop();
  ASSERT_TRUE(run.is_running());
  const auto immediate = run.release_frame();
  EXPECT_FALSE(immediate);
  if (!immediate)
    EXPECT_EQ(immediate.error().code, ErrorCode::kClosed);
  const auto bounded = run.release_frame(5ms);
  EXPECT_FALSE(bounded);
  if (!bounded)
    EXPECT_EQ(bounded.error().code, ErrorCode::kClosed);
  EXPECT_EQ(run.output_count("out"), 0U);

  probe->release();
  const auto result = run.finish();
  EXPECT_EQ(result.termination(), RunTermination::kStopped);
  EXPECT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_TRUE(result.values<std::int32_t>("out").empty());
}
