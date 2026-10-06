/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstdint>
#include <iostream>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "pipeline_test_operators.hpp"

namespace {

using holoscan::python::test::CounterSource;
using namespace std::chrono_literals;

// Reproduces the first_operator teaching contract for Python plan-parity coverage.
// Representative test: test_teaching_contract_matches_independent_cpp_temporal_records.
class SensorFusionTwin final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    (void)spec.input(camera, "camera").queue_depth(8U);
    (void)spec.input(imu, "imu").queue_depth(8U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{camera})
        .input(imu, holoscan::LatestWithin{.max_age = 5ms})
        .on_missing(imu, holoscan::ReuseLast{.max_age = 20ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<std::uint32_t> camera;
  holoscan::Input<std::uint32_t> imu;
};

}  // namespace

int main() {
  //   camera --uint32--\
  //                      fusion
  //   imu ----uint32--/
  // Key setup: Compose the C++ twin of the Python-hosted teaching contract.
  holoscan::Graph graph{"native-temporal-contract"};
  const auto camera = graph.op<CounterSource>("camera");
  const auto imu = graph.op<CounterSource>("imu");
  const auto fusion = graph.op<SensorFusionTwin>("fusion");
  graph.add_flow(camera->out, fusion->camera, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(imu->out, fusion->imu, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // Key assertion: Emit only an admitted canonical C++ plan for the Python record comparison.
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json();
    return 1;
  }
  std::cout << plan.json();
  return 0;
}
