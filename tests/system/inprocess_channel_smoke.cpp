/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Smoke: a three-operator in-process pipeline (source → filter → sink) runs
// end-to-end with a realtime clock, verifying that the compiled plan uses only
// in-process channels (no "process_local_channel_" tokens), that all three
// source values flow through the filter transformation (x→2x+1), and that the
// session terminates cleanly with no diagnostics.

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
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

constexpr std::array<std::int32_t, 3> kSourceValues{1, 2, 3};
constexpr std::array<std::int32_t, 3> kExpectedValues{3, 5, 7};

// Thread-safe collector for values received by ScalarSink.  Capped at the
// expected count so any overflow is detectable as a failure.
struct SinkObservations {
  static void reset() {
    const std::lock_guard lock(mutex);
    values.fill(0);
    count = 0U;
    overflow = false;
  }

  static void record(std::int32_t value) {
    {
      const std::lock_guard lock(mutex);
      if (count < values.size()) {
        values[count++] = value;
      } else {
        overflow = true;
      }
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_all() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return count == values.size() || overflow; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::array<std::int32_t, kExpectedValues.size()> values{};
  static inline std::size_t count{};
  static inline bool overflow{};
};

// Clock-driven source that emits values from kSourceValues one per compute call.
// Attaches a capture timestamp to each Sample so the filter can forward it.
class ScalarSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const std::size_t index = next.fetch_add(1U, std::memory_order_relaxed);
    if (index >= kSourceValues.size()) {
      return {};
    }
    const holoscan::TimePoint activation = context.activation_time();
    if (!activation.valid()) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kProtocolError});
    }
    return output.emit(holoscan::Sample<std::int32_t>{
        kSourceValues[index],
        holoscan::SampleMetadata{.capture_timestamp_ns = activation.timestamp_ns,
                                 .source_clock_id = activation.clock_id},
    });
  }

  static void reset() noexcept { next.store(0U, std::memory_order_relaxed); }

  holoscan::Output<std::int32_t> output;

 private:
  static inline std::atomic<std::size_t> next{};
};

// Transforms each received scalar by the formula x → (2x + 1), preserving
// the Sample metadata (timestamp, clock) from the source.
class ScalarFilter final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(4U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnAnyInput{})
        .input(input, holoscan::Exact{})
        .on_missing(input, holoscan::Skip{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    sample->data = (sample->data * 2) + 1;
    return output.emit(std::move(*sample));
  }

  holoscan::Input<std::int32_t> input;
  holoscan::Output<std::int32_t> output;
};

// Receives filtered values and appends them to SinkObservations for post-run
// comparison against kExpectedValues.
class ScalarSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(4U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnAnyInput{})
        .input(input, holoscan::Exact{})
        .on_missing(input, holoscan::Skip{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto value = input.receive_data();
    if (!value) {
      return holoscan::make_unexpected(std::move(value).error());
    }
    SinkObservations::record(*value);
    return {};
  }

  holoscan::Input<std::int32_t> input;
};

void print_diagnostics(const holoscan::RunSession& session) {
  for (const auto& diagnostic : session.diagnostics()) {
    std::cerr << diagnostic.token << " vertex=" << diagnostic.source.vertex.value << " port="
              << (diagnostic.source.port ? std::to_string(diagnostic.source.port.value())
                                         : "<none>")
              << " service=" << diagnostic.service << " required=" << diagnostic.required
              << " provided=" << diagnostic.provided << '\n';
  }
}

}  // namespace

// REQ: RUNTIME-015
// REQ: RUNTIME-019
// REQ: RUNTIME-018
// REQ: RUNTIME-019
// REQ: CHAN-065
// REQ: CHAN-066
int main() {
  // source(ScalarSource) ──► filter(ScalarFilter) ──► sink(ScalarSink)
  // values: [1, 2, 3]  →  transform (x→2x+1)  →  expected: [3, 5, 7]
  ScalarSource::reset();
  SinkObservations::reset();

  holoscan::Graph graph{"holoscan-inprocess-channel-smoke"};
  const auto source = graph.op<ScalarSource>("source");
  const auto filter = graph.op<ScalarFilter>("filter");
  const auto sink = graph.op<ScalarSink>("sink");
  graph.add_flow(source->output, filter->input, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(filter->output, sink->input, holoscan::ConnectionOptions{.queue_depth = 4U});
  const holoscan::ClockKey clock = graph.add_clock<holoscan::RealtimeClock>("runtime-clock");
  graph.set_default_clock(clock);

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok() || plan.json().find("process_local_channel_") != std::string::npos) {
    std::cerr << plan.json() << '\n';
    return 1;  // exit 1: plan must compile and must not reference any process_local_channel_
               //         (that token would indicate a fallback to an unintended channel type)
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool observed_all = SinkObservations::wait_for_all();
  session.request_stop();
  session.wait();

  if (session.termination() != holoscan::RunTermination::kStopped) {
    print_diagnostics(session);
    std::cerr << plan.json() << '\n';
    return 2;  // exit 2: session must terminate cleanly via stop request (kStopped)
  }
  if (!session.diagnostics().empty()) {
    print_diagnostics(session);
    return 3;  // exit 3: no runtime service diagnostics may be reported
  }
  if (!observed_all) {
    return 4;  // exit 4: all three expected values must arrive at the sink within 5s
  }
  const std::lock_guard lock(SinkObservations::mutex);
  if (SinkObservations::overflow || SinkObservations::count != kExpectedValues.size() ||
      SinkObservations::values != kExpectedValues) {
    return 5;  // exit 5: sink must receive exactly {3, 5, 7} with no overflow
  }
  return 0;
}
