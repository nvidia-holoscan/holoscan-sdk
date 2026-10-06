/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/external_topic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "pipeline_test_operators.hpp"

namespace {

using holoscan::python::test::TopicCameraSource;
using holoscan::python::test::TopicWindowFusion;

constexpr std::string_view kImuEndpoint{"baseline.camera_imu.imu"};
constexpr std::string_view kControlEndpoint{"baseline.camera_imu.control"};

// Reproduce the nested camera/IMU topology independently for Python plan comparison.
// Representative test: test_camera_imu_topic_plan_matches_independent_cpp_authoring.
class FusionGraph final : public holoscan::Graph {
 public:
  holoscan::Output<std::uint32_t> control{"control"};

 protected:
  void compose() override {
    const holoscan::ClockKey clock = add_clock<holoscan::RealtimeClock>("capture-clock");
    set_default_clock(clock);

    holoscan::TopicOptions imu_options;
    imu_options.endpoint_name = kImuEndpoint;
    imu_options.capture_time = holoscan::RequireCaptureTime{.clock = clock};
    const auto imu =
        import_topic<std::uint32_t>("holomq://" + std::string{kImuEndpoint}, imu_options);

    const auto camera = op<TopicCameraSource>("camera");
    const auto fusion = op<TopicWindowFusion>("window-fusion");
    add_flow(camera->frame, fusion->camera, holoscan::ConnectionOptions{.queue_depth = 4U});
    add_flow(imu.output(), fusion->imu, holoscan::ConnectionOptions{.queue_depth = 32U});
    export_output(control, fusion->control);
  }
};

holoscan::ExecutionPlan camera_plan() {
  holoscan::Graph graph{"holoscan-camera-imu-control"};
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  holoscan::TopicOptions control_options;
  control_options.endpoint_name = kControlEndpoint;
  const auto control = graph.export_topic<std::uint32_t>(
      "holomq://" + std::string{kControlEndpoint}, control_options);
  const auto fusion = graph.embed<FusionGraph>("fusion");
  graph.add_flow(fusion->control, control.input(), holoscan::ConnectionOptions{.queue_depth = 8U});
  return holoscan::compile(graph);
}

holoscan::ExecutionPlan warning_plan() {
  holoscan::Graph graph{"generated-endpoint"};
  (void)graph.import_topic<std::uint32_t>("holomq://generated");
  return holoscan::compile(graph);
}

holoscan::ExecutionPlan tensor_plan() {
  constexpr std::array<std::int64_t, 2U> max_shape{2, 4};
  holoscan::TopicOptions options;
  options.endpoint_name = "tensor-input";
  options.tensor = holoscan::TensorBoundaryContract{
      .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 2U,
                         .strides = holoscan::TensorStridePolicy::kContiguousOnly},
      .bounds = holoscan::tensor_bounds(max_shape),
  };
  holoscan::Graph graph{"tensor-topic"};
  (void)graph.import_topic<holoscan::Tensor>("holomq://tensor-input", options);
  return holoscan::compile(graph);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: python_core_topic_plan <camera|tensor|warning>\n";
    return 64;
  }
  const std::string_view mode{argv[1]};
  if (mode == "camera") {
    std::cout << camera_plan().json();
    return 0;
  }
  if (mode == "warning") {
    std::cout << warning_plan().json();
    return 0;
  }
  if (mode == "tensor") {
    std::cout << tensor_plan().json();
    return 0;
  }
  std::cerr << "unknown mode\n";
  return 64;
}
