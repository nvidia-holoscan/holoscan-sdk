/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <filesystem>
#include <iostream>

#include <holoscan/config/io.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/params/config.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "pipeline_test_operators.hpp"

namespace holoscan::python::test {
namespace {

struct ConfigPlanDetectorParams {
  std::uint32_t threshold{4U};
  float gain{1.0F};
  std::uint32_t window{16U};
};

HOLOSCAN_PARAM_SCHEMA(ConfigPlanDetectorParams,
                      param_schema("holoscan.tests.ConfigDetectorParams", 1),
                      field<&ConfigPlanDetectorParams::threshold>("threshold")
                          .range(1U, 64U)
                          .alias("legacy_threshold"),
                      field<&ConfigPlanDetectorParams::gain>("gain").range(0.0F, 10.0F),
                      field<&ConfigPlanDetectorParams::window>("window").range(1U, 64U));

// Reconstructs the typed detector side of test_config_file_plan_matches_independent_cpp_authoring.
class ConfigPlanDetector final : public Operator<ConfigPlanDetectorParams> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "in"); }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
};

}  // namespace
}  // namespace holoscan::python::test

int main(int argc, char** argv) {
  using holoscan::python::test::ConfigPlanDetector;
  using holoscan::python::test::ConfigPlanDetectorParams;
  using holoscan::python::test::CounterSource;

  if (argc != 2) {
    return 2;
  }
  const auto document = holoscan::config_io::try_load(std::filesystem::path{argv[1]},
                                                      holoscan::config_io::Format::kJson);
  if (!document) {
    return 3;
  }

  // Key setup: Independently author and configure the graph compared with Python.
  //
  //   source --uint32--> detector
  holoscan::Graph graph{"config-file"};
  const auto source = graph.op<CounterSource>("source");
  auto detector = graph.op<ConfigPlanDetector>("detector");
  detector.set_params(ConfigPlanDetectorParams{.threshold = 6U, .gain = 1.0F, .window = 8U});
  graph.add_flow(source->out, detector->input, holoscan::ConnectionOptions{.queue_depth = 8U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  auto configured = holoscan::params::set_params(
      graph, *document, {holoscan::params::bind(detector, "/detector")});
  if (!configured) {
    return 4;
  }
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    return 5;
  }

  // Key assertion: Emit canonical plan bytes for the Python parity assertion.
  std::cout << plan.json();
  return 0;
}
