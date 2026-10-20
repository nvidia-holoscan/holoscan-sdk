/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>

#include <holoscan/core/activation.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

constexpr std::chrono::nanoseconds kSourcePeriod = 5ms;
constexpr std::chrono::nanoseconds kMaxStartDelayBudget = 2ms;
constexpr std::chrono::nanoseconds kStallDuration = 25ms;
// The overrun stage stalls on every 6th tick: 25 ms is far beyond the 2 ms budget, yet
// six tick periods (30 ms) still cover one stall, so the pipeline drains between overruns
// and the consumers alternate between on-time and late activations deterministically.
constexpr std::uint32_t kStallEvery = 6U;
constexpr std::size_t kMinActivations = 12U;
constexpr std::size_t kTimelineCapacity = 12U;
constexpr std::chrono::seconds kObservationTimeout = 10s;
constexpr std::chrono::milliseconds kFailRunPollPeriod = 10ms;

// Per-consumer observation collected during the run and printed from main() after
// session.wait(); operators never write to stdout (interleaved output is not line-atomic).
struct MaxStartDelayObservation {
  struct TimelineEntry {
    std::uint32_t value{};
    bool late{};
  };

  struct Slot {
    std::size_t activations{};
    std::size_t late{};
    std::uint32_t first_late_value{};
    holoscan::ActivationStatus first_late_status{};
    std::array<TimelineEntry, kTimelineCapacity> timeline{};
    std::size_t timeline_size{};
  };

  static void reset() {
    const std::lock_guard lock(mutex);
    slots = {};
  }

  static void record(holoscan::OnViolation policy, std::uint32_t value,
                     holoscan::ActivationStatus status) {
    {
      const std::lock_guard lock(mutex);
      Slot& slot = slots[static_cast<std::size_t>(policy)];
      ++slot.activations;
      const bool late = status.reason == holoscan::ActivationReason::kMaxStartDelayExceeded;
      if (late) {
        if (slot.late == 0U) {
          slot.first_late_value = value;
          slot.first_late_status = status;
        }
        ++slot.late;
      }
      if (slot.timeline_size < slot.timeline.size()) {
        slot.timeline[slot.timeline_size] = TimelineEntry{value, late};
        ++slot.timeline_size;
      }
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_default_run() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, kObservationTimeout, [] {
      const Slot& safe = slots[static_cast<std::size_t>(holoscan::OnViolation::kSafeDefault)];
      const Slot& witness = slots[static_cast<std::size_t>(holoscan::OnViolation::kEmitWitness)];
      return safe.activations >= kMinActivations && safe.late >= 1U &&
             witness.activations >= kMinActivations && witness.late >= 1U;
    });
  }

  [[nodiscard]] static Slot snapshot(holoscan::OnViolation policy) {
    const std::lock_guard lock(mutex);
    return slots[static_cast<std::size_t>(policy)];
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static std::array<Slot, 3> slots;
};

// Defined outside the class: clang cannot instantiate Slot's default member
// initializers inside the enclosing class definition.
inline std::array<MaxStartDelayObservation::Slot, 3> MaxStartDelayObservation::slots{};

// Clock-driven source: one monotonically increasing tick every kSourcePeriod. Each tick
// is stamped with the activation's frozen time as its event-time capture coordinate, the
// same stamping idiom the selector examples use. The max_start_delay gate measures from the
// original frozen source decision, so any later wait before compute starts can consume
// the budget; the stamp remains event time rather than a substitute max_start_delay clock.
class TickSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kSourcePeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    return output.emit(holoscan::Sample<std::uint32_t>{
        ++next_,
        holoscan::SampleMetadata{.capture_timestamp_ns = context.activation_time().timestamp_ns,
                                 .source_clock_id = context.activation_time().clock_id},
    });
  }

  holoscan::Output<std::uint32_t> output;

 private:
  std::uint32_t next_{};
};

// The deliberate troublemaker: a plain consumer (no max_start_delay of its own) that overruns
// its compute budget on every kStallEvery-th tick. Its "a-" name prefix places it first
// in the serial dispatch order, so while it sleeps, the max_start_delay-carrying consumers'
// already-frozen activations for the same tick cannot start until the shared worker
// lane frees -- *their* latest-start budgets are exceeded, *their* policies fire. The
// overrunning activation itself is not a violation because it already started on time.
// This graph uses cross-operator interference as a deterministic witness; same-operator
// backlog can also exceed a max_start_delay when its original source decision is already frozen.
class OverrunStage final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(32U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto value = input.receive_data();
    if (!value) {
      return holoscan::make_unexpected(std::move(value).error());
    }
    if (*value % kStallEvery == 0U) {
      std::this_thread::sleep_for(kStallDuration);  // simulated compute overrun
    }
    return {};
  }

  holoscan::Input<std::uint32_t> input;
};

// Consumer with a maximum start delay and no overrun of its own: every violation it
// reports was inflicted by OverrunStage occupying the shared lane. A late activation
// still enters compute() exactly once (kSafeDefault / kEmitWitness) with the frozen
// ActivationStatus; the operator authors its own degraded path.
class MaxStartDelayConsumer final : public holoscan::Operator<> {
 public:
  explicit MaxStartDelayConsumer(holoscan::OnViolation policy) noexcept : policy_(policy) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(32U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    // A max_start_delay and its violation policy must be authored together.
    result.trigger(holoscan::OnEach{input})
        .max_start_delay(kMaxStartDelayBudget)
        .on_violation(policy_);
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const holoscan::ActivationStatus status = context.activation();
    if (status.reason == holoscan::ActivationReason::kMaxStartDelayExceeded) {
      // Degraded path: consume the sample if present, record the frozen status, and
      // return promptly. The Runtime fabricates nothing; the fallback is operator-authored.
      auto late_value = input.receive_data();
      MaxStartDelayObservation::record(policy_, late_value ? *late_value : 0U, status);
      return {};
    }
    auto value = input.receive_data();
    if (!value) {
      return holoscan::make_unexpected(std::move(value).error());
    }
    MaxStartDelayObservation::record(policy_, *value, status);
    return {};
  }

  holoscan::Input<std::uint32_t> input;

 private:
  holoscan::OnViolation policy_{holoscan::OnViolation::kEmitWitness};
};

[[nodiscard]] std::string_view reason_name(holoscan::ActivationReason reason) noexcept {
  switch (reason) {
    case holoscan::ActivationReason::kNominal:
      return "kNominal";
    case holoscan::ActivationReason::kMaxStartDelayExceeded:
      return "kMaxStartDelayExceeded";
    case holoscan::ActivationReason::kInvariantViolated:
      return "kInvariantViolated";
  }
  return "unknown";
}

[[nodiscard]] std::string_view policy_name(holoscan::OnViolation policy) noexcept {
  switch (policy) {
    case holoscan::OnViolation::kSafeDefault:
      return "kSafeDefault";
    case holoscan::OnViolation::kEmitWitness:
      return "kEmitWitness";
    case holoscan::OnViolation::kFailRun:
      return "kFailRun";
  }
  return "unknown";
}

[[nodiscard]] std::string_view termination_name(holoscan::RunTermination termination) noexcept {
  switch (termination) {
    case holoscan::RunTermination::kRunning:
      return "kRunning";
    case holoscan::RunTermination::kCompleted:
      return "kCompleted";
    case holoscan::RunTermination::kRequestedCompleted:
      return "kRequestedCompleted";
    case holoscan::RunTermination::kStopped:
      return "kStopped";
    case holoscan::RunTermination::kFailed:
      return "kFailed";
  }
  return "unknown";
}

void print_timeline(std::string_view label, const MaxStartDelayObservation::Slot& slot) {
  std::cout << "[max_start_delay] " << label << " timeline:";
  for (std::size_t index = 0; index < slot.timeline_size; ++index) {
    const MaxStartDelayObservation::TimelineEntry& entry = slot.timeline[index];
    std::cout << ' ' << entry.value << ':' << (entry.late ? "LATE" : "on-time");
  }
  std::cout << '\n';
}

// Default run: the overrun stage delays a kSafeDefault consumer and a kEmitWitness
// consumer on every 6th tick; both keep the run alive and label on-time vs late.
int run_default() {
  MaxStartDelayObservation::reset();

  holoscan::Graph graph{"contract-max-start-delay"};
  const auto ticks = graph.op<TickSource>("ticks");
  // "a-overrun-stage" sorts first: under serial alphabetic dispatch it runs before both
  // max_start_delay consumers for the same tick, so its stall delays their frozen activations.
  const auto overrun = graph.op<OverrunStage>("a-overrun-stage");
  const auto safe =
      graph.op<MaxStartDelayConsumer>("safe-default", holoscan::OnViolation::kSafeDefault);
  const auto witness =
      graph.op<MaxStartDelayConsumer>("emit-witness", holoscan::OnViolation::kEmitWitness);
  graph.add_flow(ticks->output, overrun->input, holoscan::ConnectionOptions{.queue_depth = 32U});
  graph.add_flow(ticks->output, safe->input, holoscan::ConnectionOptions{.queue_depth = 32U});
  graph.add_flow(ticks->output, witness->input, holoscan::ConnectionOptions{.queue_depth = 32U});
  // This example intentionally demonstrates max_start_delay pressure from deterministic serial
  // dispatch. Declare that one-partition execution contract instead of relying on the
  // production automatic partitioner, which correctly separates fan-out legs.
  graph.partition("max-start-delay-serial-order").include_all();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = MaxStartDelayObservation::wait_for_default_run();
  session.request_stop();
  session.wait();
  if (!complete || session.termination() != holoscan::RunTermination::kStopped) {
    const auto safe_seen = MaxStartDelayObservation::snapshot(holoscan::OnViolation::kSafeDefault);
    const auto witness_seen =
        MaxStartDelayObservation::snapshot(holoscan::OnViolation::kEmitWitness);
    std::cerr << "max_start_delay observation incomplete: termination="
              << termination_name(session.termination())
              << " safe_default activations=" << safe_seen.activations << " late=" << safe_seen.late
              << " emit_witness activations=" << witness_seen.activations
              << " late=" << witness_seen.late << '\n';
    return 2;
  }

  const auto safe_slot = MaxStartDelayObservation::snapshot(holoscan::OnViolation::kSafeDefault);
  const auto witness_slot = MaxStartDelayObservation::snapshot(holoscan::OnViolation::kEmitWitness);
  std::cout << "[max_start_delay] config: source_period=5ms max_start_delay=2ms overrun_stall=25ms"
            << " stall_every=" << kStallEvery << '\n';
  std::cout << "[max_start_delay] safe_default: activations=" << safe_slot.activations
            << " late=" << safe_slot.late << " policy=kSafeDefault\n";
  print_timeline("safe_default", safe_slot);
  std::cout << "[max_start_delay] safe_default: first_late value=" << safe_slot.first_late_value
            << " reason=" << reason_name(safe_slot.first_late_status.reason)
            << " policy=" << policy_name(safe_slot.first_late_status.policy)
            << " action=degraded-return\n";
  std::cout << "[max_start_delay] emit_witness: activations=" << witness_slot.activations
            << " late=" << witness_slot.late << " policy=kEmitWitness\n";
  print_timeline("emit_witness", witness_slot);
  std::cout << "[max_start_delay] emit_witness: witness value=" << witness_slot.first_late_value
            << " reason=" << reason_name(witness_slot.first_late_status.reason)
            << " policy=" << policy_name(witness_slot.first_late_status.policy) << '\n';
  std::cout << "contract max_start_delay complete\n";
  return 0;
}

// --fail-run: the same overrun stage, but the delayed consumer authors kFailRun. Its
// first late candidate never enters compute; the Runtime fails the run and publishes
// diagnostics.
int run_fail_run() {
  MaxStartDelayObservation::reset();

  holoscan::Graph graph{"contract-max-start-delay-fail-run"};
  const auto ticks = graph.op<TickSource>("ticks");
  const auto overrun = graph.op<OverrunStage>("a-overrun-stage");
  const auto fail = graph.op<MaxStartDelayConsumer>("fail-run", holoscan::OnViolation::kFailRun);
  graph.add_flow(ticks->output, overrun->input, holoscan::ConnectionOptions{.queue_depth = 32U});
  graph.add_flow(ticks->output, fail->input, holoscan::ConnectionOptions{.queue_depth = 32U});
  graph.partition("max_start_delay-fail-run-serial-order").include_all();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const auto give_up = std::chrono::steady_clock::now() + kObservationTimeout;
  while (session.is_running() && std::chrono::steady_clock::now() < give_up) {
    std::this_thread::sleep_for(kFailRunPollPeriod);
  }
  session.request_stop();  // safety net; idempotent no-op once the run has already failed
  session.wait();

  const auto fail_slot = MaxStartDelayObservation::snapshot(holoscan::OnViolation::kFailRun);
  std::cout << "[max_start_delay] fail-run: max_start_delay=2ms on_violation=kFailRun "
               "overrun_stall=25ms\n";
  std::cout << "[max_start_delay] fail-run: on_time_activations=" << fail_slot.activations
            << " late_compute_callbacks=" << fail_slot.late << '\n';
  std::cout << "[max_start_delay] fail-run: termination=" << termination_name(session.termination())
            << '\n';
  std::size_t index = 0;
  std::size_t max_start_delay_miss_warnings = 0;
  std::size_t fail_run_errors = 0;
  const auto diagnostics = session.diagnostics();
  for (const holoscan::Diagnostic& diagnostic : diagnostics) {
    std::cout << "[max_start_delay] fail-run diagnostic[" << index
              << "]: token=" << diagnostic.token << " severity="
              << (diagnostic.severity == holoscan::DiagnosticSeverity::kError ? "error" : "warning")
              << '\n';
    if (diagnostic.token == "MAX_START_DELAY_MISS" &&
        diagnostic.severity == holoscan::DiagnosticSeverity::kWarning) {
      ++max_start_delay_miss_warnings;
    }
    if (diagnostic.token == "RUNTIME_MAX_START_DELAY_FAIL_RUN" &&
        diagnostic.severity == holoscan::DiagnosticSeverity::kError) {
      ++fail_run_errors;
    }
    ++index;
  }
  if (session.termination() != holoscan::RunTermination::kFailed || diagnostics.size() != 2U ||
      max_start_delay_miss_warnings != 1U || fail_run_errors != 1U || fail_slot.late != 0U) {
    std::cerr << "expected kFailRun termination, one warning MAX_START_DELAY_MISS, one error "
                 "RUNTIME_MAX_START_DELAY_FAIL_RUN, and zero late callbacks were not observed\n";
    return 4;
  }
  std::cout << "contract max_start_delay fail-run complete\n";
  return 3;  // nonzero by design: the run terminated through OnViolation::kFailRun
}

}  // namespace

// Run the nonfatal policy demonstration or the fail-run diagnostic check.
int main(int argc, char** argv) {
  bool fail_run = false;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--fail-run") {
      fail_run = true;
    } else {
      std::cerr << "usage: contract_max_start_delay [--fail-run]\n";
      return 2;
    }
  }
  return fail_run ? run_fail_run() : run_default();
}
