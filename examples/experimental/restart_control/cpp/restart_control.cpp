/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// restart_control (experimental): a worker operator wedges after three
// samples and restarts itself with ctx.request_restart(); a supervisor
// thread then mints a LifecycleControlHandle from the RunSession and drives
// one external restart with await(), plus one unauthorized-target attempt
// that is rejected as kLifecycleRestartUnauthorized. Sparse lifecycle
// callbacks record ctx.transition_phase() so both restart-forward passes
// are visible in the printed stage arc, and the sink's single kStart event
// proves the restart closure never touched the rest of the graph.

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/lifecycle_control.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

[[nodiscard]] constexpr std::string_view stage_name(holoscan::LifecycleStage stage) noexcept {
  switch (stage) {
    case holoscan::LifecycleStage::kConfigure:
      return "configure";
    case holoscan::LifecycleStage::kDiscover:
      return "discover";
    case holoscan::LifecycleStage::kAllocate:
      return "allocate";
    case holoscan::LifecycleStage::kWarmUp:
      return "warm-up";
    case holoscan::LifecycleStage::kArm:
      return "arm";
    case holoscan::LifecycleStage::kStart:
      return "start";
    case holoscan::LifecycleStage::kStop:
      return "stop";
    case holoscan::LifecycleStage::kRelease:
      return "release";
  }
  return "unknown";
}

// The restart machinery is exactly what makes the non-steady phases
// reachable: examples/lifecycle_callbacks only ever sees the first and
// last of these.
[[nodiscard]] constexpr std::string_view phase_name(
    holoscan::LifecycleTransitionPhase phase) noexcept {
  switch (phase) {
    case holoscan::LifecycleTransitionPhase::kNone:
      return "none";
    case holoscan::LifecycleTransitionPhase::kStartupForward:
      return "startup-forward";
    case holoscan::LifecycleTransitionPhase::kPreRestartTeardown:
      return "pre-restart-teardown";
    case holoscan::LifecycleTransitionPhase::kRestartForward:
      return "restart-forward";
    case holoscan::LifecycleTransitionPhase::kForwardRollback:
      return "forward-rollback";
    case holoscan::LifecycleTransitionPhase::kRestoreOld:
      return "restore-old";
    case holoscan::LifecycleTransitionPhase::kShutdownTeardown:
      return "shutdown-teardown";
  }
  return "unknown";
}

[[nodiscard]] constexpr std::string_view transaction_state_name(
    holoscan::LifecycleTransactionState state) noexcept {
  switch (state) {
    case holoscan::LifecycleTransactionState::kRejected:
      return "rejected";
    case holoscan::LifecycleTransactionState::kAcceptedNoChange:
      return "accepted-no-change";
    case holoscan::LifecycleTransactionState::kQueued:
      return "queued";
    case holoscan::LifecycleTransactionState::kQuiescing:
      return "quiescing";
    case holoscan::LifecycleTransactionState::kRestarting:
      return "restarting";
    case holoscan::LifecycleTransactionState::kPublished:
      return "published";
    case holoscan::LifecycleTransactionState::kSuperseded:
      return "superseded";
    case holoscan::LifecycleTransactionState::kCanceled:
      return "canceled";
    case holoscan::LifecycleTransactionState::kRolledBack:
      return "rolled-back";
    case holoscan::LifecycleTransactionState::kFailedStopped:
      return "failed-stopped";
  }
  return "unknown";
}

// One sequence-stamped lifecycle checkpoint observation. Lifecycle
// callbacks are noexcept, so recording uses an atomic counter into a fixed
// array; main() reads the trace only after session.wait().
struct StageEvent {
  std::string_view op{};
  holoscan::LifecycleStage stage{holoscan::LifecycleStage::kConfigure};
  holoscan::LifecycleTransitionPhase phase{holoscan::LifecycleTransitionPhase::kNone};
};

struct LifecycleTrace {
  static constexpr std::size_t kCapacity = 32U;

  static holoscan::LifecycleStatus record(std::string_view op, holoscan::LifecycleStage stage,
                                          const holoscan::LifecycleContext& context) noexcept {
    const std::size_t index = count.fetch_add(1U, std::memory_order_relaxed);
    if (index < kCapacity) {
      events[index] = StageEvent{.op = op, .stage = stage, .phase = context.transition_phase()};
    }
    return holoscan::LifecycleStatus::kOk;
  }

  [[nodiscard]] static std::vector<StageEvent> for_op(std::string_view op) {
    const std::size_t total = count.load(std::memory_order_relaxed);
    const std::size_t recorded = total < kCapacity ? total : kCapacity;
    std::vector<StageEvent> result;
    for (std::size_t index = 0U; index < recorded; ++index) {
      if (events[index].op == op) {
        result.push_back(events[index]);
      }
    }
    return result;
  }

  static inline std::array<StageEvent, kCapacity> events{};
  static inline std::atomic<std::size_t> count{};
};

// Cross-restart worker bookkeeping lives at namespace scope, exactly like
// the runtime tests keep it: a lifecycle restart rewinds the operator's
// *lifecycle epoch*, and only state the operator re-derives in its stage
// callbacks is per-epoch. The wedge latch and the emitted-value counter
// must survive both restarts, so they are not per-epoch members.
struct WorkerState {
  static inline std::atomic<std::uint32_t> total_emitted{};
  static inline std::atomic_bool self_restart_requested{};
};

// The accepted self-restart receipt, handed from compute() to main().
struct RestartProbe {
  static void record(const holoscan::LifecycleRestartReceipt& accepted) {
    {
      const std::lock_guard lock(mutex);
      receipt = accepted;
      valid = true;
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_receipt() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [] { return valid; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline bool valid{};
  static inline holoscan::LifecycleRestartReceipt receipt{};
};

// Steady-state observation: every value the sink receives, in order.
struct SinkObservation {
  static void record(std::uint32_t value) {
    {
      const std::lock_guard lock(mutex);
      values.push_back(value);
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for(std::size_t expected) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [expected] { return values.size() >= expected; });
  }

  [[nodiscard]] static std::vector<std::uint32_t> snapshot() {
    const std::lock_guard lock(mutex);
    return values;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::vector<std::uint32_t> values;
};

// What the supervisor thread observed through its LifecycleControlHandle.
struct SupervisorReport {
  bool ok{};
  std::string_view failure{};
  holoscan::LifecycleTransactionState self_state{};
  std::uint64_t external_receipt_id{};
  holoscan::LifecycleTransactionState external_state{};
  holoscan::LifecycleStage external_effective_stage{};
  holoscan::ErrorCode denied_code{holoscan::ErrorCode::kSuccess};
};

// Clock-driven worker. Each lifecycle epoch it emits three samples and then
// wedges (activations keep firing, nothing comes out). The first wedge is
// resolved from the inside: compute() calls ctx.request_restart(). The
// second wedge is resolved from the outside by the supervisor thread.
class Worker final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(out, "out");  // max_emits_per_compute defaults to 1
    // Sparse registration: four checkpoints so every transition phase of a
    // kConfigure-rewind restart lands in the trace. This trace uses required
    // .stage() callbacks because all four checkpoints must execute; restart
    // also preserves .optional() callbacks.
    spec.lifecycle()
        .stage(holoscan::LifecycleStage::kConfigure, &Worker::configure)
        .stage(holoscan::LifecycleStage::kStart, &Worker::on_start)
        .stage(holoscan::LifecycleStage::kStop, &Worker::on_stop)
        .stage(holoscan::LifecycleStage::kRelease, &Worker::release);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 20ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (emitted_this_epoch_ < kSamplesPerEpoch) {
      ++emitted_this_epoch_;
      return out.emit(WorkerState::total_emitted.fetch_add(1U, std::memory_order_relaxed) + 1U);
    }
    // Wedged: the per-epoch budget is spent. Request a self-restart exactly
    // once; the second wedge waits for external control (or the final stop).
    if (!WorkerState::self_restart_requested.exchange(true, std::memory_order_relaxed)) {
      // Default LifecycleRestartOptions: from_stage = kConfigure. The call
      // is nonblocking -- acceptance queues the transaction and installs an
      // activation fence; the restart begins only after this compute()
      // callback retires, never mid-compute.
      auto receipt = context.request_restart();
      if (!receipt) {
        return holoscan::make_unexpected(std::move(receipt).error());
      }
      RestartProbe::record(*receipt);
    }
    return {};
  }

  holoscan::Output<std::uint32_t> out;

 private:
  holoscan::LifecycleStatus configure(holoscan::LifecycleContext& context) noexcept {
    return LifecycleTrace::record("worker", holoscan::LifecycleStage::kConfigure, context);
  }

  // kStart opens a fresh lifecycle epoch: the per-epoch emission budget is
  // re-derived here, which is what un-wedges the worker after each restart.
  holoscan::LifecycleStatus on_start(holoscan::LifecycleContext& context) noexcept {
    emitted_this_epoch_ = 0U;
    return LifecycleTrace::record("worker", holoscan::LifecycleStage::kStart, context);
  }

  holoscan::LifecycleStatus on_stop(holoscan::LifecycleContext& context) noexcept {
    return LifecycleTrace::record("worker", holoscan::LifecycleStage::kStop, context);
  }

  holoscan::LifecycleStatus release(holoscan::LifecycleContext& context) noexcept {
    return LifecycleTrace::record("worker", holoscan::LifecycleStage::kRelease, context);
  }

  static constexpr std::uint32_t kSamplesPerEpoch = 3U;
  std::uint32_t emitted_this_epoch_{};
};

// Data-driven sink. It has no restart grants -- it is this example's
// unauthorized external target -- and its single kStart event proves the
// worker's restarts never propagated to it: a restart executes over the
// *lifecycle-dependency closure*, not over data-flow neighbors.
class StreamSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in").queue_depth(8U);
    spec.lifecycle().stage(holoscan::LifecycleStage::kStart, &StreamSink::on_start);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto value = in.receive_data();
    if (!value) {
      return holoscan::make_unexpected(std::move(value).error());
    }
    SinkObservation::record(*value);
    return {};
  }

  holoscan::Input<std::uint32_t> in;

 private:
  holoscan::LifecycleStatus on_start(holoscan::LifecycleContext& context) noexcept {
    return LifecycleTrace::record("sink", holoscan::LifecycleStage::kStart, context);
  }
};

[[nodiscard]] bool matches(const StageEvent& event, holoscan::LifecycleStage stage,
                           holoscan::LifecycleTransitionPhase phase) noexcept {
  return event.stage == stage && event.phase == phase;
}

struct RestartTargets {
  [[nodiscard]] bool complete() const noexcept { return worker.plan_bound() && sink.plan_bound(); }

  holoscan::OperatorRef worker{};
  holoscan::OperatorRef sink{};
};

[[nodiscard]] RestartTargets find_restart_targets(const holoscan::ExecutionPlan& plan) {
  RestartTargets targets;
  for (const holoscan::LifecycleRestartTargetPlanRecord& record :
       plan.lifecycle_restart_targets()) {
    if (record.self_allowed && record.external_control_allowed) {
      targets.worker = record.target;
    } else if (!record.self_allowed && !record.external_control_allowed) {
      targets.sink = record.target;
    }
  }
  return targets;
}

[[nodiscard]] SupervisorReport run_supervisor(
    holoscan::RunSession& session, const RestartTargets& targets,
    const holoscan::LifecycleRestartReceipt& self_receipt) {
  SupervisorReport report;
  const auto worker_run = session.operator_ref(targets.worker);
  const auto sink_run = session.operator_ref(targets.sink);  // identity only, no authority
  if (!worker_run || !sink_run) {
    report.failure = "operator_ref binding failed";
    return report;
  }
  const std::array allowed_targets{*worker_run};
  constexpr std::array stages{holoscan::LifecycleStage::kConfigure};
  auto control = session.lifecycle_control(holoscan::LifecycleAccessPolicy{
      .allowed_targets = allowed_targets,
      .allowed_from_stages = stages,
  });
  if (!control) {
    report.failure = "lifecycle_control mint failed";
    return report;
  }

  // Receipts are run-scoped identities: the handle can inspect the
  // worker's own self-restart receipt without having submitted it.
  const auto self_view = control->inspect(self_receipt);
  if (!self_view) {
    report.failure = "inspect(self receipt) failed";
    return report;
  }
  report.self_state = self_view->state;

  const auto receipt = control->request_restart(*worker_run);
  if (!receipt) {
    report.failure = "external request_restart failed";
    return report;
  }
  report.external_receipt_id = receipt->receipt_id;
  const auto completed = control->await(*receipt, 5s);  // the one blocking handle operation
  if (!completed) {
    report.failure = "await(external receipt) failed";
    return report;
  }
  report.external_state = completed->state;
  report.external_effective_stage = completed->effective_stage;

  // The sink was never granted external restart authority, so the same
  // call against it is rejected as a typed error -- no state changes.
  const auto denied = control->request_restart(*sink_run);
  if (denied) {
    report.failure = "unauthorized restart unexpectedly accepted";
    return report;
  }
  report.denied_code = denied.error().code;
  report.ok = true;
  return report;
}

[[nodiscard]] bool values_are_sequential(const std::vector<std::uint32_t>& values) {
  if (values.size() != 9U) {
    return false;
  }
  for (std::size_t index = 0U; index < values.size(); ++index) {
    if (values[index] != index + 1U) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool worker_arc_matches(const std::vector<StageEvent>& arc) {
  constexpr auto kStartup = holoscan::LifecycleTransitionPhase::kStartupForward;
  constexpr auto kTeardown = holoscan::LifecycleTransitionPhase::kPreRestartTeardown;
  constexpr auto kForward = holoscan::LifecycleTransitionPhase::kRestartForward;
  constexpr auto kShutdown = holoscan::LifecycleTransitionPhase::kShutdownTeardown;
  return arc.size() == 12U && matches(arc[0], holoscan::LifecycleStage::kConfigure, kStartup) &&
         matches(arc[1], holoscan::LifecycleStage::kStart, kStartup) &&
         matches(arc[2], holoscan::LifecycleStage::kStop, kTeardown) &&
         matches(arc[3], holoscan::LifecycleStage::kRelease, kTeardown) &&
         matches(arc[4], holoscan::LifecycleStage::kConfigure, kForward) &&
         matches(arc[5], holoscan::LifecycleStage::kStart, kForward) &&
         matches(arc[6], holoscan::LifecycleStage::kStop, kTeardown) &&
         matches(arc[7], holoscan::LifecycleStage::kRelease, kTeardown) &&
         matches(arc[8], holoscan::LifecycleStage::kConfigure, kForward) &&
         matches(arc[9], holoscan::LifecycleStage::kStart, kForward) &&
         matches(arc[10], holoscan::LifecycleStage::kStop, kShutdown) &&
         matches(arc[11], holoscan::LifecycleStage::kRelease, kShutdown);
}

void print_lifecycle_arc(std::string_view operator_name, const std::vector<StageEvent>& arc) {
  std::cout << operator_name << " observed arc (" << arc.size()
            << (arc.size() == 1U ? " event):" : " events):");
  for (const StageEvent& event : arc) {
    std::cout << '\n' << "  " << stage_name(event.stage) << " @" << phase_name(event.phase);
  }
  std::cout << '\n';
}

[[nodiscard]] bool verify_runtime_story(const SupervisorReport& report) {
  const std::vector<std::uint32_t> values = SinkObservation::snapshot();
  const std::vector<StageEvent> worker_arc = LifecycleTrace::for_op("worker");
  const std::vector<StageEvent> sink_arc = LifecycleTrace::for_op("sink");
  constexpr auto kStartup = holoscan::LifecycleTransitionPhase::kStartupForward;

  const bool values_in_order = values_are_sequential(values);
  const bool worker_arc_ok = worker_arc_matches(worker_arc);
  const bool sink_arc_ok =
      sink_arc.size() == 1U && matches(sink_arc[0], holoscan::LifecycleStage::kStart, kStartup);
  print_lifecycle_arc("worker", worker_arc);
  print_lifecycle_arc("sink", sink_arc);

  const bool published = report.self_state == holoscan::LifecycleTransactionState::kPublished &&
                         report.external_state == holoscan::LifecycleTransactionState::kPublished;
  const bool denied_as_expected =
      report.denied_code == holoscan::ErrorCode::kLifecycleRestartUnauthorized;
  if (values_in_order && worker_arc_ok && sink_arc_ok && published && denied_as_expected) {
    return true;
  }
  std::cerr << "restart_control: semantic verification failed (values=" << values.size()
            << ", worker events=" << worker_arc.size() << ", sink events=" << sink_arc.size()
            << ")\n";
  return false;
}

}  // namespace

int main() {
  holoscan::Graph graph{"restart-control"};
  const auto worker = graph.op<Worker>("worker");
  const auto sink = graph.op<StreamSink>("sink");
  graph.add_flow(worker->out, sink->in, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // Deny-by-default: nothing may restart an operator unless the graph says
  // so. The worker gets both grants; the sink deliberately gets none.
  graph.lifecycle().allow_self_restart(worker).allow_external_restart(worker);

  // Restart transactions require a compile-time opt-in with explicit bounds. Without
  // CompileOptions::lifecycle_transactions, every dedicated restart API path returns
  // kLifecycleTransactionsDisabled. Enabling the opt-in for a plan containing a
  // Profile::kRealTime or Profile::kRegulated region fails compilation;
  // Profile::kStandard, Profile::kRobotics, and Profile::kResearch admit dedicated restarts.
  holoscan::CompileOptions options;
  options.profile = holoscan::Profile::kStandard;
  options.lifecycle_transactions = holoscan::LifecycleTransactionOptions{
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

  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  // The accepted plan records the complete restart-authority baseline: one
  // record per operator. Pick the two plan-bound targets by their admitted
  // authority instead of by name.
  const RestartTargets restart_targets = find_restart_targets(plan);
  if (!restart_targets.complete()) {
    std::cerr << "restart_control: restart-target records missing from the plan\n";
    return 1;
  }
  std::cout << "restart_control: plan accepted; restart authority: worker=self+external, "
               "sink=none\n";

  holoscan::RunSession session = holoscan::run_async(plan);

  // Act 1 -- self-restart. The worker emits 1..3, wedges, and requests its
  // own restart from compute(); the sink sees 4..6 once kStart re-runs.
  if (!RestartProbe::wait_for_receipt() || !SinkObservation::wait_for(6U)) {
    std::cerr << "restart_control: timed out waiting for the self-restart epoch\n";
    session.request_stop();
    session.wait();
    return 2;
  }
  const holoscan::LifecycleRestartReceipt self_receipt = RestartProbe::receipt;
  std::cout << "restart_control: worker wedged after 3 samples; self restart accepted "
            << "(receipt_id=" << self_receipt.receipt_id
            << ", requested_stage=" << stage_name(self_receipt.requested_stage) << ")\n";

  // Act 2 -- external control from a supervisor thread. The handle is a
  // restriction-only capability minted from the RunSession: its policy can
  // narrow, never broaden, what the plan admitted at compile time.
  SupervisorReport report{};
  std::thread supervisor{[&session, &report, &restart_targets, self_receipt] {
    report = run_supervisor(session, restart_targets, self_receipt);
  }};
  supervisor.join();

  if (!report.ok) {
    std::cerr << "restart_control: supervisor failed: " << report.failure << '\n';
    session.request_stop();
    session.wait();
    return 2;
  }
  std::cout << "restart_control: self restart transaction state="
            << transaction_state_name(report.self_state) << " (inspected via control handle)\n";
  std::cout << "restart_control: external restart (receipt_id=" << report.external_receipt_id
            << ") await -> state=" << transaction_state_name(report.external_state)
            << ", effective_stage=" << stage_name(report.external_effective_stage) << '\n';
  std::cout << "restart_control: unauthorized target rejected: "
            << holoscan::error_code_name(report.denied_code) << '\n';

  // Act 3 -- the externally restarted worker emits 7..9, then stop.
  const bool third_epoch = SinkObservation::wait_for(9U);
  session.request_stop();
  session.wait();
  if (!third_epoch) {
    std::cerr << "restart_control: timed out waiting for the external-restart epoch\n";
    return 2;
  }
  if (session.termination() != holoscan::RunTermination::kStopped) {
    std::cerr << "restart_control: unexpected termination state\n";
    return 2;
  }

  // All runtime threads are joined: verify the semantic story end to end.
  if (!verify_runtime_story(report)) {
    return 3;
  }

  std::cout << "restart_control: restarts published (1 self, 1 external); unauthorized target "
               "rejected (kLifecycleRestartUnauthorized); sink received 9 samples across 3 "
               "epochs termination=stopped\n";
  return 0;
}
