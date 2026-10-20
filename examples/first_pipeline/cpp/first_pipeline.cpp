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
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

// The guide page uses a leisurely 100ms period; the shipped example ticks at
// 10ms so the smoke test finishes in well under a second. Same lesson, faster.
constexpr std::chrono::nanoseconds kTickPeriod = 10ms;
constexpr std::uint32_t kTickLimit = 10U;

// Operators run on runtime threads; main() must not poll or sleep. The sink
// records every result here, and main() blocks on the condition variable until
// the expected count arrives (or a 10s safety cap expires).
struct PipelineObservation {
  static void reset() noexcept {
    count.store(0U, std::memory_order_relaxed);
    last_value.store(0U, std::memory_order_relaxed);
  }

  static void record(std::uint32_t value) {
    {
      const std::lock_guard lock(mutex);
      last_value.store(value, std::memory_order_relaxed);
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
  static inline std::atomic<std::uint32_t> last_value{};
};

// A source has no inputs, so it cannot react to data; it must opt into time.
// OnClock activates compute() on every period of the graph's default clock.
class CounterSource final : public holoscan::Operator<> {
 public:
  explicit CounterSource(std::uint32_t limit) noexcept : limit_(limit) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(out, "out");  // max_emits_per_compute defaults to 1
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTickPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    if (next_ >= limit_) {
      return {};  // exhausted: keep ticking idly; main() observes the count and stops the run
    }
    return out.emit(++next_);
  }

  holoscan::Output<std::uint32_t> out;

 private:
  std::uint32_t limit_;
  std::uint32_t next_{};
};

// A transform reacts to data, not time: OnEach activates compute() once per
// accepted sample and implicitly pins that driving sample for this activation.
class DoubleIt final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(in, "in").queue_depth(4U);
    spec.output(out, "out");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    // OnEach pins the driving sample itself; declaring a selector for the
    // driver port is rejected at compile ("must not reselect its driver").
    result.trigger(holoscan::OnEach{in});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    HOLOSCAN_TRY(const std::uint32_t value, in.receive_data());
    return out.emit(value * 2U);
  }

  holoscan::Input<std::uint32_t> in;
  holoscan::Output<std::uint32_t> out;
};

class PrintSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(4U); }

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
    std::cout << "tick " << *value << '\n';  // sole printer while the graph runs
    PipelineObservation::record(*value);
    return {};
  }

  holoscan::Input<std::uint32_t> in;
};

}  // namespace

int main() {
  PipelineObservation::reset();

  // Phase 1: compose. Structural mistakes (bad ports, type mismatches) fail here.
  holoscan::Graph graph{"first-pipeline"};
  const auto source = graph.op<CounterSource>("source", kTickLimit);
  const auto doubler = graph.op<DoubleIt>("doubler");
  const auto sink = graph.op<PrintSink>("sink");
  graph.add_flow(source->out, doubler->in, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(doubler->out, sink->in, holoscan::ConnectionOptions{.queue_depth = 4U});
  // Two-step clock opt-in: add a clock resource, then make it the default that
  // OnClock schedules against. There is no implicit ambient clock.
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // Phase 2: compile. The plan is either accepted or rejected with diagnostics.
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  // Phase 3: run. main() owns termination: observe, then cooperatively stop.
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = PipelineObservation::wait_for(kTickLimit);
  session.request_stop();
  session.wait();

  if (!complete) {
    std::cerr << "first_pipeline: timed out waiting for " << kTickLimit << " ticks\n";
    return 2;
  }
  if (session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "first_pipeline: unexpected termination state\n";
    return 2;
  }

  std::cout << "first_pipeline: received "
            << PipelineObservation::count.load(std::memory_order_relaxed) << " doubled ticks (last="
            << PipelineObservation::last_value.load(std::memory_order_relaxed)
            << ") termination=stopped\n";
  return 0;
}
