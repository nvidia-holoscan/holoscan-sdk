/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "custom_control_schema_traits.hpp"

namespace {

using namespace std::chrono_literals;

constexpr std::size_t kControlCount = 5U;
constexpr std::chrono::nanoseconds kTickPeriod = 20ms;

// This finite example stops after the sink receives every control.
struct ControlObservation {
  static void reset() {
    const std::lock_guard lock(mutex);
    received = 0U;
  }

  static void record() {
    {
      const std::lock_guard lock(mutex);
      ++received;
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for(std::size_t expected) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [expected] { return received >= expected; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::size_t received{};
};

class ControlSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTickPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    if (emitted_ >= kControlCount) {
      return {};
    }

    ++emitted_;
    app::CustomControlT control;
    control.sequence = emitted_;
    control.gain = 0.25F * static_cast<float>(emitted_);
    control.label = "ctrl-" + std::to_string(emitted_);
    return output.emit(std::move(control));
  }

  holoscan::Output<app::CustomControlT> output;

 private:
  std::uint64_t emitted_{};
};

class ControlSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(4U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto control = input.receive_data();
    if (!control) {
      return holoscan::make_unexpected(std::move(control).error());
    }
    std::cout << "control: sequence=" << control->sequence << " gain=" << control->gain
              << " label=" << control->label << '\n';
    ControlObservation::record();
    return {};
  }

  holoscan::Input<app::CustomControlT> input;
};

}  // namespace

int main() {
  ControlObservation::reset();

  holoscan::Graph graph{"custom-schema"};
  const auto source = graph.op<ControlSource>("control-source");
  const auto sink = graph.op<ControlSink>("control-sink");
  graph.add_flow(source->output, sink->input, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const bool complete = ControlObservation::wait_for(kControlCount);
  session.request_stop();
  session.wait();

  if (!complete) {
    std::cerr << "custom_schema: timed out waiting for " << kControlCount << " controls\n";
    return 2;
  }
  if (session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "custom_schema: unexpected termination state\n";
    return 2;
  }

  std::cout << "custom_schema: received " << kControlCount << " controls\n";
  return 0;
}
