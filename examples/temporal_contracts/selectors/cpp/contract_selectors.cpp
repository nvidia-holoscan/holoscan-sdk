/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Data-input selectors. One 7 ms stream fans out to three consumers that
// share one 20 ms OnClock cadence and differ in their data-input selector:
// Exact{}, LatestWithin{.max_age}, or DrainAll{}.
// Every consumer uses the direct OnClock-plus-input contract. Queue policy
// bounds survivors independently of selector semantics; DrainAll remains the
// deliberate batching choice for the fast shared stream.

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string_view>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

constexpr std::chrono::nanoseconds kTickPeriod = 20ms;   // shared consumer cadence
constexpr std::chrono::nanoseconds kStreamPeriod = 7ms;  // faster, deliberately incommensurate
constexpr std::chrono::nanoseconds kMaxSampleAge = 2ms;  // some clock candidates are stale
constexpr std::uint32_t kQueueDepth = 32U;
constexpr std::size_t kObservedDrainActivations = 8U;

struct SelectorStats {
  std::size_t activations{};
  std::size_t samples{};
  std::size_t max_batch{};
};

// Operators only record; main() prints after wait() so interleaved operator
// output can never split a summary line the CTest regex anchors to.
struct SelectorObservation {
  static void reset() {
    const std::lock_guard lock(mutex);
    exact = {};
    latest = {};
    drain = {};
  }

  static void record(SelectorStats& stats, std::size_t batch_size) {
    {
      const std::lock_guard lock(mutex);
      stats.activations += 1U;
      stats.samples += batch_size;
      stats.max_batch = std::max(stats.max_batch, batch_size);
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_drain(std::size_t expected) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [expected] { return drain.activations >= expected; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline SelectorStats exact{};
  static inline SelectorStats latest{};
  static inline SelectorStats drain{};
};

// The single stream shared by all three selector paths. Each fanout leg sees
// the same sequence value and capture coordinate.
class PeriodicSource final : public holoscan::Operator<> {
 public:
  explicit PeriodicSource(std::chrono::nanoseconds period) : period_(period) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = period_});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const std::uint32_t value = ++next_;
    return output.emit(holoscan::Sample<std::uint32_t>{
        value,
        holoscan::SampleMetadata{.capture_timestamp_ns = context.activation_time().timestamp_ns,
                                 .source_clock_id = context.activation_time().clock_id},
    });
  }

  holoscan::Output<std::uint32_t> output;

 private:
  std::chrono::nanoseconds period_;
  std::uint32_t next_{};
};

// Exact{} demands the shared-stream sample whose event time equals the frozen
// scheduled clock coordinate. The 7 ms and 20 ms periods rarely coincide, so
// Skip{} suppresses most clock candidates and compute() rarely runs.
class ExactConsumer final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(data, "data").queue_depth(kQueueDepth);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTickPeriod})
        .input(data, holoscan::Exact{})
        .on_missing(data, holoscan::Skip{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = data.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    SelectorObservation::record(SelectorObservation::exact, 1U);
    return {};
  }

  holoscan::Input<std::uint32_t> data;
};

// LatestWithin{} selects at most one newest causal shared-stream sample. The
// 2 ms age bound is shorter than the 7 ms producer period, so some clock
// candidates select one fresh value and the others are suppressed by Skip{}.
class LatestWithinConsumer final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(data, "data").queue_depth(kQueueDepth);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTickPeriod})
        .input(data, holoscan::LatestWithin{.max_age = kMaxSampleAge})
        .on_missing(data, holoscan::Skip{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = data.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    SelectorObservation::record(SelectorObservation::latest, 1U);
    return {};
  }

  holoscan::Input<std::uint32_t> data;
};

// DrainAll{} consumes every FIFO survivor since the previous activation as one
// batch. It is fieldless and needs no on_missing: an empty batch is a valid
// activation, and compilation derives its finite bound from queue geometry.
// Batch admission lets this consumer preserve every queued survivor from the
// same 7 ms stream that the two scalar consumers only sample.
class DrainAllConsumer final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(data, "data").queue_depth(kQueueDepth);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTickPeriod}).input(data, holoscan::DrainAll{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto batch = data.selected_batch();
    if (!batch) {
      return holoscan::make_unexpected(std::move(batch).error());
    }
    std::size_t batch_size = 0U;
    for ([[maybe_unused]] const holoscan::Sample<std::uint32_t>& sample : *batch) {
      // Each FIFO survivor since the previous tick is visible here exactly once.
      ++batch_size;
    }
    SelectorObservation::record(SelectorObservation::drain, batch_size);
    return {};
  }

  holoscan::Input<std::uint32_t> data;
};

void print_stats(std::string_view label, const SelectorStats& stats) {
  std::cout << "[selectors] " << label << " activations=" << stats.activations
            << " samples=" << stats.samples << " max_batch=" << stats.max_batch << '\n';
}

}  // namespace

int main() {
  SelectorObservation::reset();

  holoscan::Graph graph{"ea1-contract-selectors"};
  const auto stream = graph.op<PeriodicSource>("stream", kStreamPeriod);
  const auto exact = graph.op<ExactConsumer>("exact");
  const auto latest = graph.op<LatestWithinConsumer>("latest_within");
  const auto drain = graph.op<DrainAllConsumer>("drain_all");
  // Sampling edges use bounded kDropOldest: a Skip-suppressed consumer drains its queue
  // rarely (Exact) or at most once per activation (LatestWithin), and under
  // the default kBlocking policy an undrained queue would return finite
  // kBackpressured and require exact-value retry by the shared source. The
  // drain edge stays kBlocking because DrainAll must preserve every survivor;
  // it empties that queue on every tick.
  graph.add_flow(stream->output,
                 exact->data,
                 holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = kQueueDepth});
  graph.add_flow(stream->output,
                 latest->data,
                 holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = kQueueDepth});
  graph.add_flow(
      stream->output, drain->data, holoscan::ConnectionOptions{.queue_depth = kQueueDepth});
  // The producer and three clock-selected consumers are independent execution roots.
  graph.partition("stream-source").add(stream);
  graph.partition("exact-selector").add(exact);
  graph.partition("latest-within-selector").add(latest);
  graph.partition("drain-all-selector").add(drain);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = SelectorObservation::wait_for_drain(kObservedDrainActivations);
  session.request_stop();
  session.wait();
  if (!complete || session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "contract selectors run failed (complete=" << complete
              << " termination=" << static_cast<int>(session.termination()) << ")\n";
    for (const holoscan::Diagnostic& diagnostic : session.diagnostics()) {
      std::cerr << "  diagnostic: token=" << diagnostic.token << '\n';
    }
    return 2;
  }

  // All operator callbacks have retired after wait(); take one coherent
  // snapshot so both the executable assertions and stdout describe the same
  // completed run.
  SelectorStats exact_stats;
  SelectorStats latest_stats;
  SelectorStats drain_stats;
  {
    const std::lock_guard lock(SelectorObservation::mutex);
    exact_stats = SelectorObservation::exact;
    latest_stats = SelectorObservation::latest;
    drain_stats = SelectorObservation::drain;
  }

  std::cout << "[selectors] cadence=OnClock{20ms} stream=7ms max_age=2ms\n";
  print_stats("exact", exact_stats);
  print_stats("latest_within", latest_stats);
  print_stats("drain_all", drain_stats);

  const bool exact_is_scalar =
      exact_stats.samples == exact_stats.activations && exact_stats.max_batch <= 1U;
  const bool latest_selects_and_skips =
      latest_stats.activations > 0U && latest_stats.activations < drain_stats.activations &&
      latest_stats.samples == latest_stats.activations && latest_stats.max_batch == 1U;
  const bool drain_batches = drain_stats.activations >= kObservedDrainActivations &&
                             drain_stats.samples > drain_stats.activations &&
                             drain_stats.max_batch > 1U;
  if (!exact_is_scalar || !latest_selects_and_skips || !drain_batches) {
    std::cerr << "selector semantic assertions failed (exact_scalar=" << exact_is_scalar
              << " latest_selects_and_skips=" << latest_selects_and_skips
              << " drain_batches=" << drain_batches << ")\n";
    return 3;
  }

  std::cout << "[selectors] assertions=pass\n";
  std::cout << "contract selectors complete\n";
  return 0;
}
