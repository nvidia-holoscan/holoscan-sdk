/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <utility>
#include <vector>

#include <holoscan/core/activation.hpp>
#include <holoscan/core/compile.hpp>
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

constexpr std::size_t kMinLeftActivations = 4U;
constexpr std::size_t kMinRightActivations = 2U;
constexpr std::size_t kPrintedActivations = 6U;

struct PortEvent {
  bool from_left{};
  std::uint32_t value{};
};

// Collects per-activation attribution off the operator threads; main() prints only after the
// session has stopped, so stdout stays line-atomic and the CTest regex anchors to post-stop lines.
struct AnyInputObservation {
  static void reset() {
    const std::lock_guard lock(mutex);
    events.clear();
    left_count = 0U;
    right_count = 0U;
    anomaly_count = 0U;
  }

  static void record(bool from_left, std::uint32_t value, bool other_port_missing) {
    {
      const std::lock_guard lock(mutex);
      events.push_back(PortEvent{from_left, value});
      (from_left ? left_count : right_count) += 1U;
      if (!other_port_missing) {
        anomaly_count += 1U;
      }
    }
    ready.notify_all();
  }

  static void record_anomaly() {
    {
      const std::lock_guard lock(mutex);
      anomaly_count += 1U;
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_both_ports() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [] {
      return left_count >= kMinLeftActivations && right_count >= kMinRightActivations;
    });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::vector<PortEvent> events;  // Guarded by mutex.
  static inline std::size_t left_count{};       // Guarded by mutex.
  static inline std::size_t right_count{};      // Guarded by mutex.
  static inline std::size_t anomaly_count{};    // Guarded by mutex.
};

// Zero-input source. It MUST author a trigger: an empty contract() on a zero-input operator is
// rejected when the graph is compiled ("NO_TRIGGER: a zero-input or partially authored contract
// needs one trigger", surfaced as a TEMPORAL_CONTRACT_INVALID diagnostic in plan.json()).
class PeriodicSource final : public holoscan::Operator<> {
 public:
  PeriodicSource(std::chrono::nanoseconds period, std::uint32_t scale) noexcept
      : period_(period), scale_(scale) {}

  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = period_});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const std::uint32_t value = (next_++ + 1U) * scale_;
    return output.emit(holoscan::Sample<std::uint32_t>{
        value,
        holoscan::SampleMetadata{.capture_timestamp_ns = context.activation_time().timestamp_ns,
                                 .source_clock_id = context.activation_time().clock_id},
    });
  }

  holoscan::Output<std::uint32_t> output;

 private:
  std::chrono::nanoseconds period_;
  std::uint32_t scale_{};
  std::uint32_t next_{};
};

// The lesson: no contract() override. Because this operator has connected inputs, compilation
// synthesizes the OnAnyInput{} default, and every accepted sample on either port activates
// compute() exactly once. selection()/is_missing() attribute each activation to its port.
class AnyInputProbe final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(left, "left").queue_depth(8U);
    spec.input(right, "right").queue_depth(8U);
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    const bool left_selected = left.selection().state == holoscan::InputState::kSelected;
    const bool right_selected = right.selection().state == holoscan::InputState::kSelected;
    if (left_selected == right_selected) {
      // OnAnyInput promises exactly one selected port per activation; anything else is recorded
      // as an anomaly and fails the final summary check in main().
      AnyInputObservation::record_anomaly();
      return {};
    }
    auto sample = left_selected ? left.receive() : right.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    const bool other_port_missing = left_selected ? right.is_missing() : left.is_missing();
    AnyInputObservation::record(left_selected, sample->data, other_port_missing);
    return {};
  }

  holoscan::Input<std::uint32_t> left;
  holoscan::Input<std::uint32_t> right;
};

}  // namespace

int main() {
  AnyInputObservation::reset();

  holoscan::Graph graph{"ea1-contract-on-any-input"};
  const auto left_clock = graph.op<PeriodicSource>("left_clock", 15ms, 1U);
  const auto right_clock = graph.op<PeriodicSource>("right_clock", 40ms, 100U);
  const auto probe = graph.op<AnyInputProbe>("probe");
  graph.add_flow(left_clock->output, probe->left, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(right_clock->output, probe->right, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.partition("left-input-path").add(left_clock).add(probe);
  graph.partition("right-input-source").add(right_clock);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = AnyInputObservation::wait_for_both_ports();
  session.request_stop();
  session.wait();
  if (!complete || session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "run did not stop cleanly with both ports observed\n";
    return 2;
  }

  std::vector<PortEvent> events;
  std::size_t left_count = 0U;
  std::size_t right_count = 0U;
  std::size_t anomaly_count = 0U;
  {
    const std::lock_guard lock(AnyInputObservation::mutex);
    events = AnyInputObservation::events;
    left_count = AnyInputObservation::left_count;
    right_count = AnyInputObservation::right_count;
    anomaly_count = AnyInputObservation::anomaly_count;
  }

  const std::size_t printed = std::min(events.size(), kPrintedActivations);
  for (std::size_t index = 0; index < printed; ++index) {
    std::cout << "[on_any_input] activation " << (index + 1U)
              << ": port=" << (events[index].from_left ? "left" : "right")
              << " value=" << events[index].value << '\n';
  }
  std::cout << "[on_any_input] observed: left_activations=" << left_count
            << " right_activations=" << right_count << " anomalies=" << anomaly_count << '\n';
  if (left_count < kMinLeftActivations || right_count < kMinRightActivations ||
      anomaly_count != 0U) {
    std::cerr << "port attribution check failed\n";
    return 3;
  }
  std::cout << "[on_any_input] every activation was attributed to exactly one input port\n";
  std::cout << "on_any_input example complete\n";
  return 0;
}
