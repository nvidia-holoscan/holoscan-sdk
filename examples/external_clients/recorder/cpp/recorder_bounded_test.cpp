/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>

#include <holoscan/holoscan.hpp>

#include "result_recorder.hpp"

namespace {

using namespace std::chrono_literals;

// Emits an unbounded clock-driven sequence so the recorder cannot rely on producer EOS.
// The output tolerates the ordinary optional-export attachment window.
// Representative test: EXAMPLE_RESULT_RECORDER_BOUNDED_TEST.
class UnboundedRecorderSource final : public holoscan::Operator<> {
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
      holoscan::ExecutionContext&) override {
    auto emitted = output.emit(++next_);
    if (!emitted && emitted.error().code != holoscan::ErrorCode::kNotReady &&
        emitted.error().code != holoscan::ErrorCode::kBackpressured) {
      return holoscan::make_unexpected(std::move(emitted).error());
    }
    return {};
  }

  holoscan::Output<std::uint32_t> output;

 private:
  std::uint32_t next_{};
};

}  // namespace

int main() {
  const std::string endpoint_name =
      "result_recorder_bounded_" +
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  holoscan::Graph graph{"result-recorder-bounded"};
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));
  const auto source = graph.op<UnboundedRecorderSource>("source");
  holoscan::TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.bounds.subscriber =
      holoscan::SubscriberEndpointBounds{.queue_depth = 2U, .max_borrowed_samples = 1U};
  const auto exported =
      graph.export_topic<std::uint32_t>("holomq://" + endpoint_name, std::move(options));
  graph.add_flow(source->output, exported.input(), holoscan::ConnectionOptions{.queue_depth = 2U});

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    return 1;
  }
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    return 2;
  }
  auto subscriber =
      runtime->create_subscriber<std::uint32_t>(plan.endpoint_catalog(), endpoint_name);
  if (!subscriber) {
    return 3;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  auto recorded = holoscan::examples::record_results(
      *subscriber, plan.endpoint_catalog().digest(), std::cout, 1U, 10s);
  auto closed = subscriber->close();
  auto after_close = subscriber->receive_sample();
  session.request_stop();
  session.wait();

  if (!recorded || recorded->received_count != 1U || recorded->remote_end_of_stream || !closed ||
      after_close || after_close.error().code != holoscan::ErrorCode::kClosed ||
      session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    return 4;
  }
  std::cout << "bounded recorder complete\n";
  return 0;
}
