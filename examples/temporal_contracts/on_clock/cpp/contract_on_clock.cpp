/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Fixed-rate activation from OnClock. A 50 ms sensor source and
// a 20 ms clock-triggered controller run at independent rates; the controller
// directly samples the sensor with LatestWithin{40ms} + on_missing
// ReuseLast{100ms}. The
// controller therefore activates MORE OFTEN than the sensor produces — the
// cadence comes from the 20 ms clock, never from sensor arrivals — and the
// post-run summary proves it (activations > samples_produced).
// The controller intentionally uses the direct public shape: OnClock plus a
// selected data input. Runtime preserves that shape across compile, run,
// restart, and replay boundaries.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
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

// The control loop activates two and a half times faster than the sensor
// produces: the fixed 20 ms cadence must come from the clock, not arrivals.
constexpr std::chrono::nanoseconds kSensorPeriod = 50ms;
constexpr std::chrono::nanoseconds kControlPeriod = 20ms;
constexpr std::chrono::nanoseconds kFreshnessBound = 40ms;
constexpr std::chrono::nanoseconds kReuseBound = 100ms;
constexpr std::size_t kCommandTarget = 8U;

struct ControlObservation {
  struct ActivationRecord {
    holoscan::InputState state{};
    std::uint32_t value{};
  };

  static void reset() {
    const std::lock_guard lock(mutex);
    produced.store(0U, std::memory_order_relaxed);
    commands.store(0U, std::memory_order_relaxed);
    activations.clear();
  }

  static void record_produced() noexcept { produced.fetch_add(1U, std::memory_order_relaxed); }

  static void record_activation(holoscan::InputState state, std::uint32_t value) {
    const std::lock_guard lock(mutex);
    activations.push_back(ActivationRecord{.state = state, .value = value});
  }

  static void record_command() {
    {
      const std::lock_guard lock(mutex);
      commands.fetch_add(1U, std::memory_order_release);
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_commands(std::size_t expected) {
    std::unique_lock lock(mutex);
    return ready.wait_for(
        lock, 10s, [expected] { return commands.load(std::memory_order_acquire) >= expected; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::size_t> produced{};
  static inline std::atomic<std::size_t> commands{};
  static inline std::vector<ActivationRecord> activations;
};

[[nodiscard]] const char* input_state_name(holoscan::InputState state) noexcept {
  switch (state) {
    case holoscan::InputState::kSelected:
      return "fresh";
    case holoscan::InputState::kReused:
      return "reused";
    case holoscan::InputState::kSafeDefault:
      return "safe-default";
    case holoscan::InputState::kAbsent:
      return "absent";
    case holoscan::InputState::kNotSelected:
      return "not-selected";
  }
  return "unknown";
}

// A slow (50 ms) sensor. Every reading carries a capture timestamp taken from
// the activation time: LatestWithin and ReuseLast measure age in event time,
// against exactly this stamp.
class SensorSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.output(sample, "sample"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kSensorPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const std::uint32_t value = ++next_;
    ControlObservation::record_produced();
    return sample.emit(holoscan::Sample<std::uint32_t>{
        value,
        holoscan::SampleMetadata{.capture_timestamp_ns = context.activation_time().timestamp_ns,
                                 .source_clock_id = context.activation_time().clock_id},
    });
  }

  holoscan::Output<std::uint32_t> sample;

 private:
  std::uint32_t next_{};
};

// The control loop: OnClock directly owns the 20 ms cadence. The sensor is a
// sampled input: LatestWithin
// admits the newest reading at most 40 ms old exactly once, and on the ticks
// between arrivals ReuseLast re-serves the last admitted reading (state
// kReused) for up to 100 ms. When neither policy can serve a value — only
// before the very first reading here — the whole activation is suppressed:
// compute() never runs with a missing sensor, so it never fabricates a
// command.
class RateController final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(sensor, "sensor").queue_depth(4U);
    spec.output(command, "command");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kControlPeriod})
        .input(sensor, holoscan::LatestWithin{.max_age = kFreshnessBound})
        .on_missing(sensor, holoscan::ReuseLast{.max_age = kReuseBound});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto reading = sensor.receive();
    if (!reading) {
      return holoscan::make_unexpected(std::move(reading).error());
    }
    ControlObservation::record_activation(sensor.selection().state, reading->data);
    // This is a newly issued actuation command, so "command issued now" is its
    // physically meaningful event time. Forwarding the sensor observation
    // would preserve the original Sample<T> capture coordinate instead.
    const auto issued_at = context.clock().now();
    if (!issued_at) {
      return holoscan::make_unexpected(std::move(issued_at).error());
    }
    return command.emit(holoscan::Sample<std::uint32_t>{
        reading->data,
        holoscan::SampleMetadata{.capture_timestamp_ns = issued_at->timestamp_ns,
                                 .source_clock_id = issued_at->clock_id},
    });
  }

  holoscan::Input<std::uint32_t> sensor;
  holoscan::Output<std::uint32_t> command;
};

class ActuatorSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(8U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    ControlObservation::record_command();
    return {};
  }

  holoscan::Input<std::uint32_t> input;
};

}  // namespace

// REQ: SYSTEM-013
int main() {
  ControlObservation::reset();

  holoscan::Graph graph{"contract-on-clock"};
  const auto sensor = graph.op<SensorSource>("sensor");
  const auto controller = graph.op<RateController>("controller");
  const auto actuator = graph.op<ActuatorSink>("actuator");
  // The 50 ms sensor is slower than the 20 ms consumer, so every reading is
  // admitted at the next tick and a small queue is plenty.
  graph.add_flow(
      sensor->sample, controller->sensor, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(
      controller->command, actuator->input, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.partition("sensor-source").add(sensor);
  graph.partition("control-loop").add(controller).add(actuator);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = ControlObservation::wait_for_commands(kCommandTarget);
  session.request_stop();
  session.wait();
  if (!complete || session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "run did not produce " << kCommandTarget << " commands and stop cleanly\n"
              << "  complete=" << complete
              << " termination=" << static_cast<int>(session.termination())
              << " produced=" << ControlObservation::produced.load(std::memory_order_relaxed)
              << " commands=" << ControlObservation::commands.load(std::memory_order_acquire)
              << '\n';
    for (const auto& diagnostic : session.diagnostics()) {
      std::cerr << "  diagnostic token=" << diagnostic.token << '\n';
    }
    return 2;
  }

  // Print only after wait(): operator threads are quiescent, so stdout lines
  // are ordered and the CTest regular expression can anchor on them.
  std::vector<ControlObservation::ActivationRecord> records;
  {
    const std::lock_guard lock(ControlObservation::mutex);
    records = ControlObservation::activations;
  }
  std::size_t fresh = 0U;
  std::size_t reused = 0U;
  for (const auto& record : records) {
    if (record.state == holoscan::InputState::kSelected) {
      ++fresh;
    } else if (record.state == holoscan::InputState::kReused) {
      ++reused;
    }
  }
  const std::size_t produced = ControlObservation::produced.load(std::memory_order_relaxed);
  const std::size_t command_count = ControlObservation::commands.load(std::memory_order_relaxed);

  for (std::size_t index = 0U; index < records.size() && index < kCommandTarget; ++index) {
    std::cout << "[on_clock] activation=" << index + 1U
              << " input=" << input_state_name(records[index].state)
              << " sensor_value=" << records[index].value << '\n';
  }
  std::cout << "[on_clock] sensor_period_ms=50 control_period_ms=20\n";
  std::cout << "[on_clock] samples_produced=" << produced << " activations=" << records.size()
            << " commands=" << command_count << '\n';
  std::cout << "[on_clock] fresh=" << fresh << " reused=" << reused << '\n';

  // The semantic claim of the example: the 20 ms clock activated the
  // controller more often than the 50 ms sensor produced. Arrivals cannot
  // explain the cadence; only the clock can.
  if (records.empty() || fresh == 0U || produced == 0U || produced >= records.size()) {
    std::cerr << "expected the 20 ms control loop to outpace the 50 ms sensor\n";
    return 3;
  }
  std::cout << "on_clock example complete: fixed-rate activation verified\n";
  return 0;
}
