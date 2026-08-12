/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// lifecycle_callbacks: two operators register sparse lifecycle-stage
// callbacks and order their checkpoints across operators in two ways — a
// fact-based same-stage dependency (produces / requires_dependency on a
// shared LifecycleDependencyKey) at kAllocate, and a graph-scope edge
// (graph.lifecycle().before) at kStart. Every callback records one
// sequence-stamped event; main() prints and verifies the full stage arc
// only after the run has stopped.

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string_view>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

// Same-stage ordering fact. The producer declares that its kAllocate
// checkpoint provides the frame table; the consumer requires that fact
// before its own kAllocate checkpoint may run. The key lives at namespace
// scope so both operators name the same fact without naming — or even
// knowing about — each other.
constexpr holoscan::LifecycleDependencyKey kFrameTableReady{
    .stable_name = "lifecycle_callbacks.frame_table.ready"};

// The eight global lifecycle barriers, in execution order. kConfigure
// through kStart run forward at startup; kStop and kRelease are teardown.
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

// All four bounded callback dispositions. kOk and kSkipped are success
// dispositions; kRetryableFailure and kFatalFailure fail the transition —
// even from a callback registered as optional().
[[nodiscard]] constexpr std::string_view status_name(holoscan::LifecycleStatus status) noexcept {
  switch (status) {
    case holoscan::LifecycleStatus::kOk:
      return "ok";
    case holoscan::LifecycleStatus::kSkipped:
      return "skipped";
    case holoscan::LifecycleStatus::kRetryableFailure:
      return "retryable-failure";
    case holoscan::LifecycleStatus::kFatalFailure:
      return "fatal-failure";
  }
  return "unknown";
}

// This example only ever observes the two steady phases; the restart and
// rollback phases belong to experimental/restart_control.
[[nodiscard]] constexpr std::string_view phase_name(
    holoscan::LifecycleTransitionPhase phase) noexcept {
  switch (phase) {
    case holoscan::LifecycleTransitionPhase::kStartupForward:
      return "startup-forward";
    case holoscan::LifecycleTransitionPhase::kShutdownTeardown:
      return "shutdown-teardown";
    case holoscan::LifecycleTransitionPhase::kNone:
    case holoscan::LifecycleTransitionPhase::kPreRestartTeardown:
    case holoscan::LifecycleTransitionPhase::kRestartForward:
    case holoscan::LifecycleTransitionPhase::kForwardRollback:
    case holoscan::LifecycleTransitionPhase::kRestoreOld:
      return "other";
  }
  return "unknown";
}

// One sequence-stamped lifecycle checkpoint observation.
struct StageEvent {
  std::string_view op{};
  holoscan::LifecycleStage stage{holoscan::LifecycleStage::kConfigure};
  holoscan::LifecycleTransitionPhase phase{holoscan::LifecycleTransitionPhase::kNone};
  holoscan::LifecycleStatus status{holoscan::LifecycleStatus::kOk};
};

// Lifecycle callbacks are noexcept, so recording must not lock, allocate,
// or throw: an atomic counter claims a slot in a fixed array. main() reads
// the trace only after session.wait() has joined every runtime thread,
// which orders all callback writes before the read.
struct LifecycleTrace {
  static constexpr std::size_t kCapacity = 16U;

  static void reset() noexcept { count.store(0U, std::memory_order_relaxed); }

  // Returns the status unchanged so a callback body can end with
  // `return LifecycleTrace::record(...)`.
  static holoscan::LifecycleStatus record(std::string_view op, holoscan::LifecycleStage stage,
                                          const holoscan::LifecycleContext& context,
                                          holoscan::LifecycleStatus status) noexcept {
    const std::size_t index = count.fetch_add(1U, std::memory_order_relaxed);
    if (index < kCapacity) {
      events[index] = StageEvent{
          .op = op, .stage = stage, .phase = context.transition_phase(), .status = status};
    }
    return status;
  }

  static inline std::array<StageEvent, kCapacity> events{};
  static inline std::atomic<std::size_t> count{};
};

// Steady-state observation: the consumer records each received sample so
// main() knows when enough compute activations have happened to stop.
struct RunObservation {
  static void reset() noexcept { count.store(0U, std::memory_order_relaxed); }

  static void record() {
    {
      const std::lock_guard lock(mutex);
      count.fetch_add(1U, std::memory_order_release);
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for(std::size_t expected) {
    std::unique_lock lock(mutex);
    return ready.wait_for(
        lock, 10s, [expected] { return count.load(std::memory_order_acquire) >= expected; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::size_t> count{};
};

// Clock-driven source. Its compute()-time resource — a table of frame
// values — is reserved at the kAllocate checkpoint, guaranteed to complete
// before the first compute() activation.
class FrameProducer final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(out, "out");  // max_emits_per_compute defaults to 1
    // Sparse registration: only the checkpoints with application work need a callback.
    // Core records every unregistered stage as a successful skipped logical checkpoint.
    // Start and Stop are optional because either callback may be absent; Allocate is
    // required because compute depends on the populated table.
    spec.lifecycle()
        .stage(holoscan::LifecycleStage::kAllocate, &FrameProducer::allocate)
        .optional(holoscan::LifecycleStage::kStart, &FrameProducer::on_start)
        .optional(holoscan::LifecycleStage::kStop, &FrameProducer::on_stop)
        .optional(holoscan::LifecycleStage::kRelease, &FrameProducer::release)
        .produces(holoscan::LifecycleStage::kAllocate, kFrameTableReady);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 5ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    // The table was filled at kAllocate; steady state stays allocation-free.
    const std::uint32_t value = frames_[next_ % frames_.size()];
    ++next_;
    return out.emit(value);
  }

  holoscan::Output<std::uint32_t> out;

 private:
  holoscan::LifecycleStatus allocate(holoscan::LifecycleContext& context) noexcept {
    for (std::size_t index = 0U; index < frames_.size(); ++index) {
      frames_[index] = static_cast<std::uint32_t>((index + 1U) * 10U);
    }
    return LifecycleTrace::record(
        "producer", holoscan::LifecycleStage::kAllocate, context, holoscan::LifecycleStatus::kOk);
  }

  holoscan::LifecycleStatus on_start(holoscan::LifecycleContext& context) noexcept {
    return LifecycleTrace::record(
        "producer", holoscan::LifecycleStage::kStart, context, holoscan::LifecycleStatus::kOk);
  }

  holoscan::LifecycleStatus on_stop(holoscan::LifecycleContext& context) noexcept {
    // The frame table is immutable after start, so there is nothing to
    // quiesce: kSkipped reports "no work" and is a success disposition.
    return LifecycleTrace::record(
        "producer", holoscan::LifecycleStage::kStop, context, holoscan::LifecycleStatus::kSkipped);
  }

  holoscan::LifecycleStatus release(holoscan::LifecycleContext& context) noexcept {
    frames_.fill(0U);  // return the epoch's resources
    return LifecycleTrace::record(
        "producer", holoscan::LifecycleStage::kRelease, context, holoscan::LifecycleStatus::kOk);
  }

  std::array<std::uint32_t, 32U> frames_{};
  std::uint32_t next_{};
};

// Data-driven sink. Its kAllocate checkpoint requires the producer's
// frame-table fact, so within the kAllocate stage the producer's callback
// is guaranteed to complete first.
class FrameConsumer final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in").queue_depth(8U);
    // The consumer needs Allocate, while Start, Stop, and Release demonstrate
    // optional checkpoints whose absence would leave a valid operator.
    spec.lifecycle()
        .stage(holoscan::LifecycleStage::kAllocate, &FrameConsumer::allocate)
        .optional(holoscan::LifecycleStage::kStart, &FrameConsumer::on_start)
        .optional(holoscan::LifecycleStage::kStop, &FrameConsumer::on_stop)
        .optional(holoscan::LifecycleStage::kRelease, &FrameConsumer::release)
        .requires_dependency(holoscan::LifecycleStage::kAllocate, kFrameTableReady);
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
    total_ += *value;
    RunObservation::record();
    return {};
  }

  holoscan::Input<std::uint32_t> in;

 private:
  holoscan::LifecycleStatus allocate(holoscan::LifecycleContext& context) noexcept {
    total_ = 0U;
    return LifecycleTrace::record(
        "consumer", holoscan::LifecycleStage::kAllocate, context, holoscan::LifecycleStatus::kOk);
  }

  holoscan::LifecycleStatus on_start(holoscan::LifecycleContext& context) noexcept {
    return LifecycleTrace::record(
        "consumer", holoscan::LifecycleStage::kStart, context, holoscan::LifecycleStatus::kOk);
  }

  holoscan::LifecycleStatus on_stop(holoscan::LifecycleContext& context) noexcept {
    // Activation acceptance is closed; the accumulated total is now final.
    return LifecycleTrace::record(
        "consumer", holoscan::LifecycleStage::kStop, context, holoscan::LifecycleStatus::kOk);
  }

  holoscan::LifecycleStatus release(holoscan::LifecycleContext& context) noexcept {
    total_ = 0U;
    return LifecycleTrace::record(
        "consumer", holoscan::LifecycleStage::kRelease, context, holoscan::LifecycleStatus::kOk);
  }

  std::uint64_t total_{};
};

[[nodiscard]] bool matches(const StageEvent& event, std::string_view op,
                           holoscan::LifecycleStage stage, holoscan::LifecycleTransitionPhase phase,
                           holoscan::LifecycleStatus status) noexcept {
  return event.op == op && event.stage == stage && event.phase == phase && event.status == status;
}

[[nodiscard]] bool teardown_pair(const StageEvent& first, const StageEvent& second,
                                 holoscan::LifecycleStage stage,
                                 holoscan::LifecycleStatus producer_status,
                                 holoscan::LifecycleStatus consumer_status) noexcept {
  // Teardown stages are global barriers. Runtime orders each pair deterministically (kStop by
  // reverse successful-start order and kRelease by stable compiled order here), but this helper
  // checks membership and grouping instead of duplicating scheduler policy.
  constexpr auto kShutdown = holoscan::LifecycleTransitionPhase::kShutdownTeardown;
  return (matches(first, "producer", stage, kShutdown, producer_status) &&
          matches(second, "consumer", stage, kShutdown, consumer_status)) ||
         (matches(first, "consumer", stage, kShutdown, consumer_status) &&
          matches(second, "producer", stage, kShutdown, producer_status));
}

}  // namespace

int main() {
  RunObservation::reset();
  LifecycleTrace::reset();

  holoscan::Graph graph{"lifecycle-callbacks"};
  const auto producer = graph.op<FrameProducer>("producer");
  const auto consumer = graph.op<FrameConsumer>("consumer");
  graph.add_flow(producer->out, consumer->in, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  // Graph-scope ordering: composition decides that the producer publishes
  // readiness (kStart) before the consumer does. Same stage, two operators,
  // declared where the topology is known rather than inside either operator.
  graph.lifecycle().before(producer, consumer, holoscan::LifecycleStage::kStart);

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = RunObservation::wait_for(5U);
  session.request_stop();
  session.wait();

  if (!complete) {
    std::cerr << "lifecycle_callbacks: timed out waiting for samples termination="
              << static_cast<int>(session.termination()) << " diagnostics=";
    for (const auto& diagnostic : session.diagnostics()) {
      std::cerr << diagnostic.token << "@vertex=" << diagnostic.source.vertex.value << ',';
    }
    std::cerr << '\n';
    return 2;
  }
  if (session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "lifecycle_callbacks: unexpected termination state\n";
    return 2;
  }

  // All callbacks have completed: print the sequence-stamped stage arc.
  const std::size_t total = LifecycleTrace::count.load(std::memory_order_relaxed);
  const std::size_t recorded =
      total < LifecycleTrace::kCapacity ? total : LifecycleTrace::kCapacity;
  for (std::size_t index = 0U; index < recorded; ++index) {
    const StageEvent& event = LifecycleTrace::events[index];
    std::cout << "lifecycle[" << index << "] " << event.op << ' ' << stage_name(event.stage)
              << " phase=" << phase_name(event.phase) << " -> " << status_name(event.status)
              << '\n';
  }

  // Assert the arc: both declared cross-operator orderings held at startup,
  // and teardown ran stage-by-stage after the stop request.
  const auto& events = LifecycleTrace::events;
  constexpr auto kStartup = holoscan::LifecycleTransitionPhase::kStartupForward;
  constexpr auto kOk = holoscan::LifecycleStatus::kOk;
  const bool startup_ordered =
      matches(events[0], "producer", holoscan::LifecycleStage::kAllocate, kStartup, kOk) &&
      matches(events[1], "consumer", holoscan::LifecycleStage::kAllocate, kStartup, kOk) &&
      matches(events[2], "producer", holoscan::LifecycleStage::kStart, kStartup, kOk) &&
      matches(events[3], "consumer", holoscan::LifecycleStage::kStart, kStartup, kOk);
  const bool teardown_ordered =
      teardown_pair(events[4],
                    events[5],
                    holoscan::LifecycleStage::kStop,
                    holoscan::LifecycleStatus::kSkipped,
                    kOk) &&
      teardown_pair(events[6], events[7], holoscan::LifecycleStage::kRelease, kOk, kOk);
  if (total != 8U || !startup_ordered || !teardown_ordered) {
    std::cerr << "lifecycle_callbacks: stage arc mismatch (" << total << " events)\n";
    return 3;
  }

  std::cout << "lifecycle_callbacks: stage arc verified (8 events; producer precedes consumer "
               "at allocate and start) termination=stopped\n";
  return 0;
}
