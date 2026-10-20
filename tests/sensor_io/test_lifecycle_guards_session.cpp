/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Drives StageGuards from an actual RunSession rather than by calling the guard directly.
//
// test_lifecycle_guards.cpp exercises StageGuards::run as a helper and pins its flag algebra. That
// is necessary but not sufficient for the failure this file exists to catch, which is a
// disagreement between the guard and the runtime that invokes it. Core admits kSkipped as a
// successful logical checkpoint (\c callback_succeeded, public/src/core/lifecycle_kernel.cpp),
// records the completion, and goes on to publish the operator as started. A guard that reads
// kSkipped as "did not happen" therefore withholds the next stage's body while the run reports
// success, and no diagnostic anywhere says so. A helper-level test cannot show that second half: it
// can prove which bodies the guard withheld, but not that the run around them still terminated
// cleanly, which is what makes the bug silent in production.
//
// Each test below asserts the exact ordered sequence of guard decisions the run produced, which
// pins the interleaving of Core's stage schedule with the guard's own gating in a single
// comparison. A decision is recorded for every registered stage the runtime invoked, tagged with
// whether the guard let the body run, so a regression names both the stage and the direction of the
// mistake.
//
// Two scenarios that the unit tests do cover are deliberately absent here, because Core makes them
// unobservable at this level:
//
//   * A *failed* kArm aborts the forward transition inside the kernel, so kStart and kStop are
//     never invoked at all and the guard is never asked. What the guard would have done is
//     therefore only observable from the helper.
//   * A restart from kConfigure is what the existing runtime restart coverage already requests by
//     default, and its suffix subsumes the guard's full reset.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/lifecycle_guards.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace holoscan {
namespace {

using namespace std::chrono_literals;

/// @brief Select what the guarded operator's stage bodies report, and where it restarts.
enum class GuardScenario : std::uint8_t {
  /// kArm reports kSkipped and kStart reports kOk. The Start body must still run.
  kSkippedArm,
  /// kArm and kStart both report kSkipped, so the operator owns nothing for kStop to quiesce.
  kSkippedArmAndStart,
  /// kArm reports kOk and the operator restarts from kStart, whose suffix never revisits kArm.
  kStartOnlyRestart,
  /// kArm reports kOk, then kSkipped when the restart suffix re-enters it.
  kArmRestartWithSkippedRearm,
  /// As above, and the restarted kStart reports kSkipped too, so no body in the suffix takes
  /// anything. The device the first kArm acquired is still owed a quiesce.
  kArmRestartWithSkippedRearmAndStart,
  /// Restart from kAllocate, the only requestable stage whose teardown suffix reaches kRelease.
  kAllocateRestart,
};

/// @brief Report the stage this scenario requests a restart from, if it requests one.
[[nodiscard]] constexpr std::optional<LifecycleStage> restart_from(
    GuardScenario scenario) noexcept {
  switch (scenario) {
    case GuardScenario::kStartOnlyRestart:
      return LifecycleStage::kStart;
    case GuardScenario::kArmRestartWithSkippedRearm:
    case GuardScenario::kArmRestartWithSkippedRearmAndStart:
      return LifecycleStage::kArm;
    case GuardScenario::kAllocateRestart:
      return LifecycleStage::kAllocate;
    case GuardScenario::kSkippedArm:
    case GuardScenario::kSkippedArmAndStart:
      break;
  }
  return std::nullopt;
}

/// @brief Report whether this scenario re-enters kArm through a restart suffix.
[[nodiscard]] constexpr bool rearms_through_restart(GuardScenario scenario) noexcept {
  return scenario == GuardScenario::kArmRestartWithSkippedRearm ||
         scenario == GuardScenario::kArmRestartWithSkippedRearmAndStart;
}

/// @brief One guard decision, recorded in the order the runtime made it.
struct StageEvent {
  LifecycleStage stage{};
  bool body_ran{};
};

/// @brief Render one decision so a failed comparison names the stage and which way the guard went.
[[nodiscard]] std::string render(StageEvent event) {
  std::string name{"?"};
  switch (event.stage) {
    case LifecycleStage::kConfigure:
      name = "Configure";
      break;
    case LifecycleStage::kDiscover:
      name = "Discover";
      break;
    case LifecycleStage::kAllocate:
      name = "Allocate";
      break;
    case LifecycleStage::kWarmUp:
      name = "WarmUp";
      break;
    case LifecycleStage::kArm:
      name = "Arm";
      break;
    case LifecycleStage::kStart:
      name = "Start";
      break;
    case LifecycleStage::kStop:
      name = "Stop";
      break;
    case LifecycleStage::kRelease:
      name = "Release";
      break;
  }
  return event.body_ran ? name : "withheld:" + name;
}

/// @brief Cross-thread record of the guard decisions one scenario produced.
///
/// The trace is written from lifecycle callbacks, which are noexcept, so the write path takes no
/// lock and allocates nothing: a bounded array, a release store of the size, and a notify. Core
/// quiesces an operator's callbacks against each other, so the element writes need no atomicity
/// beyond publishing the index that orders them.
template <GuardScenario kScenario>
struct GuardProbe {
  static constexpr std::size_t kTraceCapacity = 16U;

  static void reset() {
    const std::lock_guard lock{mutex};
    trace.fill(StageEvent{});
    trace_size.store(0U, std::memory_order_release);
    restart_error.reset();
  }

  static void record(LifecycleStage stage, bool body_ran) noexcept {
    const std::size_t index = trace_size.load(std::memory_order_relaxed);
    if (index < kTraceCapacity) {
      trace[index] = StageEvent{.stage = stage, .body_ran = body_ran};
    }
    trace_size.store(index + 1U, std::memory_order_release);
    cv.notify_all();
  }

  static void fail(ErrorCode code) {
    {
      const std::lock_guard lock{mutex};
      restart_error = code;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_until(std::size_t decisions) {
    std::unique_lock lock{mutex};
    return cv.wait_for(lock, 5s, [decisions] {
      return trace_size.load(std::memory_order_acquire) >= decisions || restart_error.has_value();
    });
  }

  /// @brief Render every decision recorded so far, in the order the runtime made them.
  [[nodiscard]] static std::vector<std::string> rendered() {
    const std::size_t size = std::min(trace_size.load(std::memory_order_acquire), kTraceCapacity);
    std::vector<std::string> result;
    result.reserve(size);
    for (std::size_t index = 0U; index < size; ++index) {
      result.emplace_back(render(trace[index]));
    }
    return result;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::array<StageEvent, kTraceCapacity> trace{};
  static inline std::atomic<std::size_t> trace_size{};
  static inline std::optional<ErrorCode> restart_error;
};

/// @brief A sensor-shaped operator that runs every stage body behind its own StageGuards.
///
/// The operator carries no ports on purpose. What is under test is the interaction between Core's
/// stage schedule and the guard, and a port would add a readiness path that could mask a withheld
/// Start body by keeping the operator inactive for an unrelated reason.
template <GuardScenario kScenario>
class GuardedSensorOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.lifecycle()
        .stage(LifecycleStage::kConfigure, &GuardedSensorOperator::configure_stage)
        .stage(LifecycleStage::kAllocate, &GuardedSensorOperator::allocate_stage)
        .stage(LifecycleStage::kArm, &GuardedSensorOperator::arm_stage)
        .stage(LifecycleStage::kStart, &GuardedSensorOperator::start_stage)
        .stage(LifecycleStage::kStop, &GuardedSensorOperator::stop_stage)
        .stage(LifecycleStage::kRelease, &GuardedSensorOperator::release_stage);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    constexpr std::optional<LifecycleStage> kRestartFrom = restart_from(kScenario);
    if constexpr (kRestartFrom.has_value()) {
      if (!restart_requested_) {
        restart_requested_ = true;
        // The constexpr branch excludes empty restart stages in every instantiation.
        auto receipt =
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            context.request_restart(LifecycleRestartOptions{.from_stage = *kRestartFrom});
        if (!receipt) {
          Probe::fail(receipt.error().code);
          return make_unexpected(std::move(receipt).error());
        }
      }
    }
    return {};
  }

  LifecycleStatus configure_stage(LifecycleContext&) noexcept {
    return guarded<LifecycleStage::kConfigure>(LifecycleStatus::kOk);
  }

  LifecycleStatus allocate_stage(LifecycleContext&) noexcept {
    return guarded<LifecycleStage::kAllocate>(LifecycleStatus::kOk);
  }

  LifecycleStatus arm_stage(LifecycleContext&) noexcept {
    return guarded<LifecycleStage::kArm>(arm_body_status());
  }

  LifecycleStatus start_stage(LifecycleContext&) noexcept {
    return guarded<LifecycleStage::kStart>(start_body_status());
  }

  LifecycleStatus stop_stage(LifecycleContext&) noexcept {
    return guarded<LifecycleStage::kStop>(LifecycleStatus::kOk);
  }

  LifecycleStatus release_stage(LifecycleContext&) noexcept {
    return guarded<LifecycleStage::kRelease>(LifecycleStatus::kOk);
  }

 private:
  using Probe = GuardProbe<kScenario>;

  /// @brief Report what this scenario's Arm body returns on this invocation.
  [[nodiscard]] LifecycleStatus arm_body_status() noexcept {
    [[maybe_unused]] const std::size_t invocation = arm_invocations_++;
    if constexpr (kScenario == GuardScenario::kSkippedArm ||
                  kScenario == GuardScenario::kSkippedArmAndStart) {
      return LifecycleStatus::kSkipped;
    } else if constexpr (rearms_through_restart(kScenario)) {
      // The re-entered Arm body has nothing left to queue and says so. Core records the checkpoint
      // as complete either way, so the kStart that follows in the same suffix must still run.
      return invocation == 0U ? LifecycleStatus::kOk : LifecycleStatus::kSkipped;
    } else {
      return LifecycleStatus::kOk;
    }
  }

  /// @brief Report what this scenario's Start body returns on this invocation.
  [[nodiscard]] LifecycleStatus start_body_status() noexcept {
    [[maybe_unused]] const std::size_t invocation = start_invocations_++;
    if constexpr (kScenario == GuardScenario::kSkippedArmAndStart) {
      return LifecycleStatus::kSkipped;
    } else if constexpr (kScenario == GuardScenario::kArmRestartWithSkippedRearmAndStart) {
      // The restarted Start has nothing to republish either, so the whole restart suffix reports
      // that it created no work while the first epoch's acquisition is still outstanding.
      return invocation == 0U ? LifecycleStatus::kOk : LifecycleStatus::kSkipped;
    } else {
      return LifecycleStatus::kOk;
    }
  }

  /// @brief Run one stage body behind the guard and record whether the guard let it run.
  template <LifecycleStage kStage>
  LifecycleStatus guarded(LifecycleStatus body_status) noexcept {
    bool body_ran = false;
    const LifecycleStatus status = guards_.run<kStage>([&body_ran, body_status]() noexcept {
      body_ran = true;
      return body_status;
    });
    Probe::record(kStage, body_ran);
    return status;
  }

  sensor_io::StageGuards guards_;
  std::size_t arm_invocations_{};
  std::size_t start_invocations_{};
  bool restart_requested_{};
};

// One guarded operator, no flows, driven by a 1 ms clock. A restarting scenario additionally opts
// into active lifecycle transactions and names itself as a self-restart target.
template <GuardScenario kScenario>
[[nodiscard]] ExecutionPlan make_guarded_sensor_plan() {
  Graph graph{"sensor-io-lifecycle-guards-session"};
  [[maybe_unused]] const auto sensor = graph.op<GuardedSensorOperator<kScenario>>("guarded-sensor");
  graph.set_default_clock(graph.add_clock<RealtimeClock>("sensor-io-guards-clock"));

  CompileOptions options;
  if constexpr (restart_from(kScenario).has_value()) {
    graph.lifecycle().allow_self_restart(sensor);
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
  return compile(graph, std::move(options));
}

/// @brief The guard decisions a scenario produced, before and after shutdown.
struct ScenarioTrace {
  /// Decisions recorded once the run reached the steady state the scenario was waiting for.
  std::vector<std::string> at_steady_state;
  /// Every decision, including the teardown suffix that request_stop drove.
  std::vector<std::string> complete;
};

/// @brief Run one scenario to its steady state, then shut it down and report both traces.
/// @param decisions Number of guard decisions that marks the steady state for this scenario.
template <GuardScenario kScenario>
[[nodiscard]] ScenarioTrace run_scenario(std::size_t decisions) {
  using Probe = GuardProbe<kScenario>;
  Probe::reset();
  const ExecutionPlan plan = make_guarded_sensor_plan<kScenario>();
  EXPECT_TRUE(plan.ok()) << plan.json();
  if (!plan.ok()) {
    return ScenarioTrace{};
  }

  RunSession session = run_async(plan);
  const bool reached = Probe::wait_until(decisions);
  // Stable to read without the lock: the scenario's next decision is a teardown one, and teardown
  // is driven by the request_stop below. A restarting scenario requests its restart exactly once.
  const std::vector<std::string> at_steady_state = Probe::rendered();
  session.request_stop();
  session.wait();

  EXPECT_TRUE(reached) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                         : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  {
    const std::lock_guard lock{Probe::mutex};
    EXPECT_FALSE(Probe::restart_error.has_value());
  }
  return ScenarioTrace{.at_steady_state = at_steady_state, .complete = Probe::rendered()};
}

// The failure this file exists for, end to end. A sensor with nothing to queue returns kSkipped
// from kArm; Core records the checkpoint and publishes the operator as started. Gating kStart on
// arm *ownership* would withhold the Start body here while the run terminated cleanly with no
// diagnostic, so the sensor would never stream a frame and nothing would report a problem.
// REQ: SENSOR-IO-001
TEST(SensorIoLifecycleGuardsSession, SkippedArmStillRunsTheStartBodyInARealRun) {
  const ScenarioTrace trace = run_scenario<GuardScenario::kSkippedArm>(4U);

  EXPECT_EQ(trace.at_steady_state,
            (std::vector<std::string>{"Configure", "Allocate", "Arm", "Start"}));
  EXPECT_EQ(trace.complete,
            (std::vector<std::string>{"Configure", "Allocate", "Arm", "Start", "Stop", "Release"}));
}

// The other half of the same distinction, which is why it takes two flags rather than one. kArm and
// kStart both report kSkipped, so the operator owns no device state at all, and the kStop that Core
// still schedules against its recorded kArm checkpoint must be withheld rather than quiesce a
// device that was never armed. kRelease still runs, because kAllocate did own resources.
// REQ: SENSOR-IO-001
TEST(SensorIoLifecycleGuardsSession, SkippedArmAndStartWithholdTheStopBodyInARealRun) {
  const ScenarioTrace trace = run_scenario<GuardScenario::kSkippedArmAndStart>(4U);

  EXPECT_EQ(trace.at_steady_state,
            (std::vector<std::string>{"Configure", "Allocate", "Arm", "Start"}));
  EXPECT_EQ(trace.complete,
            (std::vector<std::string>{
                "Configure", "Allocate", "Arm", "Start", "withheld:Stop", "Release"}));
}

// A restart requested at kStart runs the suffix kStop then kStart and never revisits kArm, because
// Core retains the completed kArm checkpoint. A successful kStop must therefore leave the guard's
// arm record standing: clearing it would make the restart's kStart skip, and Core would publish a
// restart whose sensor never started.
// REQ: SENSOR-IO-001
TEST(SensorIoLifecycleGuardsSession, StartOnlyRestartRunsTheStartBodyAgain) {
  const ScenarioTrace trace = run_scenario<GuardScenario::kStartOnlyRestart>(6U);

  EXPECT_EQ(trace.at_steady_state,
            (std::vector<std::string>{"Configure", "Allocate", "Arm", "Start", "Stop", "Start"}));
  EXPECT_EQ(trace.complete,
            (std::vector<std::string>{
                "Configure", "Allocate", "Arm", "Start", "Stop", "Start", "Stop", "Release"}));
}

// Both halves in one sequence, which is the case a single collapsed flag cannot express: the
// restart suffix re-enters the Arm body, which this time has nothing to queue and reports kSkipped,
// and the kStart that follows in the same suffix must still run.
// REQ: SENSOR-IO-001
TEST(SensorIoLifecycleGuardsSession, ArmRestartWithASkippedRearmStillRunsTheStartBody) {
  const ScenarioTrace trace = run_scenario<GuardScenario::kArmRestartWithSkippedRearm>(7U);

  EXPECT_EQ(
      trace.at_steady_state,
      (std::vector<std::string>{"Configure", "Allocate", "Arm", "Start", "Stop", "Arm", "Start"}));
  EXPECT_EQ(
      trace.complete,
      (std::vector<std::string>{
          "Configure", "Allocate", "Arm", "Start", "Stop", "Arm", "Start", "Stop", "Release"}));
}

// The silent-teardown regression, end to end. The restart suffix re-enters kArm and kStart and both
// report kSkipped, so nothing in the suffix acquires anything -- but nothing in it releases
// anything either, because a kArm restart runs no kRelease and no kConfigure. Inferring
// relinquishment from those skips cleared both records, and the shutdown kStop then found nothing
// to quiesce and withheld its body while the run terminated cleanly with no diagnostic. The device
// the first kArm acquired would be left running past the end of the run. REQ: SENSOR-IO-001
TEST(SensorIoLifecycleGuardsSession, SkippedRestartSuffixStillRunsTheShutdownStopBody) {
  const ScenarioTrace trace = run_scenario<GuardScenario::kArmRestartWithSkippedRearmAndStart>(7U);

  EXPECT_EQ(
      trace.at_steady_state,
      (std::vector<std::string>{"Configure", "Allocate", "Arm", "Start", "Stop", "Arm", "Start"}));
  // The final Stop runs rather than reporting withheld:Stop, which is the whole point.
  EXPECT_EQ(
      trace.complete,
      (std::vector<std::string>{
          "Configure", "Allocate", "Arm", "Start", "Stop", "Arm", "Start", "Stop", "Release"}));
}

// kAllocate is the only requestable restart stage whose teardown suffix reaches kRelease, so the
// guard's whole ownership and progress record is torn down and rebuilt while the kConfigure record
// survives. Every forward body must run again: a guard that failed to clear its arm or start record
// at kRelease would withhold one of them, and one that cleared kConfigure would withhold kAllocate.
// REQ: SENSOR-IO-001
TEST(SensorIoLifecycleGuardsSession, AllocateRestartRebuildsEveryForwardBody) {
  const ScenarioTrace trace = run_scenario<GuardScenario::kAllocateRestart>(9U);

  EXPECT_EQ(
      trace.at_steady_state,
      (std::vector<std::string>{
          "Configure", "Allocate", "Arm", "Start", "Stop", "Release", "Allocate", "Arm", "Start"}));
  EXPECT_EQ(trace.complete,
            (std::vector<std::string>{"Configure",
                                      "Allocate",
                                      "Arm",
                                      "Start",
                                      "Stop",
                                      "Release",
                                      "Allocate",
                                      "Arm",
                                      "Start",
                                      "Stop",
                                      "Release"}));
}

}  // namespace
}  // namespace holoscan
