/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/predicate_input_view.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

// Every matching driver tick activates the input-driven consumer.
constexpr std::uint32_t kGateDivisor = 3U;
// The clocked consumer's predicate is true while the level signal is at or above this.
constexpr std::uint32_t kLevelThreshold = 50U;
constexpr std::uint32_t kHighLevel = 100U;
constexpr std::uint32_t kLowLevel = 0U;
// The level source holds each square-wave phase for this many emissions.
constexpr std::uint32_t kPhaseTicks = 8U;

constexpr auto kDriverPeriod = 5ms;
constexpr auto kLevelPeriod = 5ms;
constexpr auto kEdgeEvaluationPeriod = 10ms;
constexpr std::uint32_t kInputQueueDepth = 4U;

// Stop thresholds: keep running until predicate gating is unambiguously visible.
constexpr std::size_t kMinGateActivations = 4U;
constexpr std::size_t kMinGateFalseEvaluations = 4U;
constexpr std::size_t kMinClockedActivations = 4U;
constexpr std::size_t kMinClockedRejectedEvaluations = 3U;

// Predicate evaluators run on the scheduler's decision path and compute() runs on
// worker threads, so nothing here prints. Counters are recorded under one mutex and
// main() prints the summary only after the session has fully stopped.
struct PredicateObservation {
  [[nodiscard]] static constexpr bool gate_truth(std::uint32_t driver_value) noexcept {
    return driver_value % kGateDivisor == 0U;
  }

  static void reset() noexcept {
    gate_evaluations.store(0U, std::memory_order_relaxed);
    gate_true_evaluations.store(0U, std::memory_order_relaxed);
    gate_activations.store(0U, std::memory_order_relaxed);
    gate_misfires.store(0U, std::memory_order_relaxed);
    edge_evaluations.store(0U, std::memory_order_relaxed);
    edge_true_evaluations.store(0U, std::memory_order_relaxed);
    edge_activations.store(0U, std::memory_order_relaxed);
    edge_low_activations.store(0U, std::memory_order_relaxed);
  }

  static void record_gate_evaluation(bool truth) {
    {
      const std::lock_guard lock(mutex);
      gate_evaluations.fetch_add(1U, std::memory_order_relaxed);
      if (truth) {
        gate_true_evaluations.fetch_add(1U, std::memory_order_relaxed);
      }
    }
    ready.notify_all();
  }

  static void record_gate_activation(std::uint32_t driver_value) {
    {
      const std::lock_guard lock(mutex);
      gate_activations.fetch_add(1U, std::memory_order_relaxed);
      if (!gate_truth(driver_value)) {
        gate_misfires.fetch_add(1U, std::memory_order_relaxed);
      }
    }
    ready.notify_all();
  }

  static void record_edge_evaluation(bool truth) {
    {
      const std::lock_guard lock(mutex);
      edge_evaluations.fetch_add(1U, std::memory_order_relaxed);
      if (truth) {
        edge_true_evaluations.fetch_add(1U, std::memory_order_relaxed);
      }
    }
    ready.notify_all();
  }

  static void record_edge_activation(std::uint32_t level_value) {
    {
      const std::lock_guard lock(mutex);
      edge_activations.fetch_add(1U, std::memory_order_relaxed);
      if (level_value < kLevelThreshold) {
        edge_low_activations.fetch_add(1U, std::memory_order_relaxed);
      }
    }
    ready.notify_all();
  }

  // Bounded wait: settles once both consumers show more evaluations than
  // activations, i.e. once the gating effect itself is observable.
  [[nodiscard]] static bool wait_until_settled() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [] {
      const std::size_t gate_evals = gate_evaluations.load(std::memory_order_relaxed);
      const std::size_t gate_trues = gate_true_evaluations.load(std::memory_order_relaxed);
      const std::size_t gate_acts = gate_activations.load(std::memory_order_relaxed);
      const std::size_t edge_evals = edge_evaluations.load(std::memory_order_relaxed);
      const std::size_t edge_acts = edge_activations.load(std::memory_order_relaxed);
      return gate_acts >= kMinGateActivations && gate_trues == gate_acts &&
             gate_evals >= gate_trues + kMinGateFalseEvaluations &&
             edge_acts >= kMinClockedActivations &&
             edge_evals >= edge_acts + kMinClockedRejectedEvaluations;
    });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::size_t> gate_evaluations{};
  static inline std::atomic<std::size_t> gate_true_evaluations{};
  static inline std::atomic<std::size_t> gate_activations{};
  static inline std::atomic<std::size_t> gate_misfires{};
  static inline std::atomic<std::size_t> edge_evaluations{};
  static inline std::atomic<std::size_t> edge_true_evaluations{};
  static inline std::atomic<std::size_t> edge_activations{};
  static inline std::atomic<std::size_t> edge_low_activations{};
};

// Emits 1, 2, ... — the driver stream whose every accepted sample triggers one
// predicate evaluation on the gate consumer.
class TickSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kDriverPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const std::uint32_t value = next_ + 1U;
    auto emitted = output.emit(holoscan::Sample<std::uint32_t>{
        value,
        holoscan::SampleMetadata{.capture_timestamp_ns = context.activation_time().timestamp_ns,
                                 .source_clock_id = context.activation_time().clock_id},
    });
    if (!emitted && emitted.error().code == holoscan::ErrorCode::kBackpressured) {
      return {};
    }
    if (!emitted) {
      return emitted;
    }
    ++next_;
    return {};
  }

  holoscan::Output<std::uint32_t> output;

 private:
  std::uint32_t next_{};
};

// Emits a square wave: kHighLevel for kPhaseTicks emissions, then kLowLevel for
// kPhaseTicks, and so on. The clocked consumer samples the latest recent level
// independently of this publication cadence.
class LevelSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kLevelPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const std::uint32_t value = ((tick_ / kPhaseTicks) % 2U == 0U) ? kHighLevel : kLowLevel;
    auto emitted = output.emit(holoscan::Sample<std::uint32_t>{
        value,
        holoscan::SampleMetadata{.capture_timestamp_ns = context.activation_time().timestamp_ns,
                                 .source_clock_id = context.activation_time().clock_id},
    });
    if (!emitted && emitted.error().code == holoscan::ErrorCode::kBackpressured) {
      return {};
    }
    if (!emitted) {
      return emitted;
    }
    ++tick_;
    return {};
  }

  holoscan::Output<std::uint32_t> output;

 private:
  std::uint32_t tick_{};
};

// Variant 1: input-driven evaluation with level-triggered firing. Every driver
// sample divisible by three activates compute().
class GatedConsumer final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(driver, "driver").queue_depth(kInputQueueDepth);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    auto evaluator = holoscan::predicate(
        "tick-divisible-edge",
        [](const holoscan::PredicateEvaluationContext&,
           const holoscan::PredicateInputView<std::uint32_t>& driver_view) noexcept {
          const bool truth = PredicateObservation::gate_truth(driver_view.sample().data);
          PredicateObservation::record_gate_evaluation(truth);
          return truth;
        });
    holoscan::Contract result;
    result
        .trigger(holoscan::OnPredicate{
            .evaluate_on = holoscan::OnInputUpdate{driver},
            .predicate = std::move(evaluator),
            .firing = holoscan::PredicateFiring::kOnEveryTrueEvaluation,
        })
        .input(driver, holoscan::Exact{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto driver_sample = driver.receive();
    if (!driver_sample) {
      return holoscan::make_unexpected(std::move(driver_sample).error());
    }
    PredicateObservation::record_gate_activation(driver_sample->data);
    return {};
  }

  holoscan::Input<std::uint32_t> driver;
};

// Variant 2: clock-driven evaluation with one typed input and edge-triggered
// firing. The predicate samples the latest recent level on each 10 ms clock
// tick and activates compute() only when that level crosses from low to high.
class ClockedConsumer final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(level, "level").queue_depth(kInputQueueDepth);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    auto evaluator = holoscan::predicate(
        "level-threshold-edge",
        holoscan::predicate_inputs(level),
        [](const holoscan::PredicateEvaluationContext&,
           const holoscan::PredicateInputView<std::uint32_t>& level_view) noexcept {
          const bool truth = level_view.sample().data >= kLevelThreshold;
          PredicateObservation::record_edge_evaluation(truth);
          return truth;
        });
    holoscan::Contract result;
    result
        .trigger(holoscan::OnPredicate{
            .evaluate_on = holoscan::OnClock{.period = kEdgeEvaluationPeriod},
            .predicate = std::move(evaluator),
            .firing = holoscan::PredicateFiring::kOnFalseToTrue,
        })
        .input(level, holoscan::LatestWithin{.max_age = 100ms})
        .on_missing(level, holoscan::Skip{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto level_sample = level.receive();
    if (!level_sample) {
      std::cerr << "clocked input receive failed: "
                << holoscan::error_code_name(level_sample.error().code) << '\n';
      return holoscan::make_unexpected(std::move(level_sample).error());
    }
    PredicateObservation::record_edge_activation(level_sample->data);
    return {};
  }

  holoscan::Input<std::uint32_t> level;
};

}  // namespace

int main() {
  PredicateObservation::reset();

  holoscan::Graph graph{"holoscan-contract-on-predicate"};
  const auto ticks = graph.op<TickSource>("ticks");
  const auto level = graph.op<LevelSource>("level");
  const auto gated = graph.op<GatedConsumer>("gated");
  const auto clocked = graph.op<ClockedConsumer>("clocked");
  graph.add_flow(
      ticks->output, gated->driver, holoscan::ConnectionOptions{.queue_depth = kInputQueueDepth});
  graph.add_flow(level->output,
                 clocked->level,
                 holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = kInputQueueDepth});
  graph.partition("input-predicate").add(ticks).add(gated);
  graph.partition("level-source").add(level);
  graph.partition("clocked-predicate").add(clocked);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool settled = PredicateObservation::wait_until_settled();
  session.request_stop();
  session.wait();

  const std::size_t gate_evals =
      PredicateObservation::gate_evaluations.load(std::memory_order_relaxed);
  const std::size_t gate_trues =
      PredicateObservation::gate_true_evaluations.load(std::memory_order_relaxed);
  const std::size_t gate_acts =
      PredicateObservation::gate_activations.load(std::memory_order_relaxed);
  const std::size_t gate_misfires =
      PredicateObservation::gate_misfires.load(std::memory_order_relaxed);
  const std::size_t edge_evals =
      PredicateObservation::edge_evaluations.load(std::memory_order_relaxed);
  const std::size_t edge_trues =
      PredicateObservation::edge_true_evaluations.load(std::memory_order_relaxed);
  const std::size_t edge_acts =
      PredicateObservation::edge_activations.load(std::memory_order_relaxed);
  const std::size_t edge_lows =
      PredicateObservation::edge_low_activations.load(std::memory_order_relaxed);

  std::cout << "[on_predicate] plan temporal trigger records=" << plan.temporal_triggers().size()
            << '\n';
  std::cout << "[on_predicate] gate consumer (evaluate_on=OnInputUpdate{driver},"
               " firing=kOnEveryTrueEvaluation)\n";
  std::cout << "[on_predicate]   evaluations=" << gate_evals << " true_evaluations=" << gate_trues
            << " activations=" << gate_acts << " misfires=" << gate_misfires << '\n';
  std::cout << "[on_predicate] clocked consumer (evaluate_on=OnClock{10ms},"
               " firing=kOnFalseToTrue)\n";
  std::cout << "[on_predicate]   evaluations=" << edge_evals << " true_evaluations=" << edge_trues
            << " activations=" << edge_acts << " low_activations=" << edge_lows << '\n';

  if (!settled || session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "run did not settle and stop cleanly: settled=" << (settled ? 1 : 0)
              << " termination=" << static_cast<int>(session.termination())
              << " diagnostics=" << session.diagnostics().size() << '\n';
    for (const holoscan::Diagnostic& diagnostic : session.diagnostics()) {
      std::cerr << "  diagnostic token=" << diagnostic.token << " service=" << diagnostic.service
                << " severity="
                << (diagnostic.severity == holoscan::DiagnosticSeverity::kError ? "error"
                                                                                : "warning")
                << " required=" << diagnostic.required << " provided=" << diagnostic.provided
                << " vertex=" << diagnostic.source.vertex.value << '\n';
    }
    return 2;
  }

  // A true evaluation may commit immediately before request_stop() closes callback admission.
  // That terminal candidate is retired without entering compute().
  const bool gate_cutoff_ok = gate_acts <= gate_trues && gate_trues - gate_acts <= 1U;
  const bool gating_ok = gate_misfires == 0U && gate_acts >= kMinGateActivations &&
                         gate_cutoff_ok && gate_evals > gate_trues;
  const bool clocked_ok = edge_lows == 0U && edge_acts >= kMinClockedActivations &&
                          edge_trues > edge_acts && edge_evals > edge_trues;
  std::cout << "on_predicate verdict: gating=" << (gating_ok ? "ok" : "violated")
            << " clocked_gating=" << (clocked_ok ? "ok" : "violated") << '\n';
  if (!gating_ok || !clocked_ok) {
    return 3;
  }
  std::cout << "on_predicate example complete\n";
  return 0;
}
