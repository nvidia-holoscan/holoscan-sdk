/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <iostream>
#include <stdexcept>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

class StartupFailure final : public holoscan::Operator<> {
 public:
  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }
  void start() override { throw std::runtime_error{"example startup failure before compute"}; }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }
};

}  // namespace
int main() {
  // Present compile and terminal results explicitly; fatal invariant reports remain enabled.
  holoscan::set_automatic_diagnostic_logging_enabled(false);
  if (holoscan::automatic_diagnostic_logging_enabled()) {
    std::cerr << "Environment enables automatic reports; the explicit report may repeat them.\n";
  }
  holoscan::Graph graph{"diagnostic-reporting"};
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  graph.op<StartupFailure>("device");
  const auto plan = holoscan::compile(graph);
  holoscan::log_diagnostics(plan);  // Includes accepted-plan warnings; clean plans are silent.
  if (!plan.ok()) {
    return 1;
  }
  auto session = holoscan::run_async(plan);
  session.wait();  // Async return alone is not a complete startup-diagnostics barrier.
  holoscan::log_diagnostics(session.diagnostics(), plan.graph());
  if (session.termination() != holoscan::RunTermination::kFailed || session.diagnostics().empty()) {
    return 1;
  }
  std::cout << "diagnostic_reporting example complete: caller-owned startup failure report\n";
  return 0;  // The deliberate failed session was observed and reported successfully.
}
