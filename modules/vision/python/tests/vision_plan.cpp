// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <iostream>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/vision/resize_op.hpp>

int main() {
  holoscan::Graph graph{"vision-python-parity"};
  auto resize = graph.op<holoscan::vision::ResizeOp>("resize");
  resize.set_params(holoscan::vision::ResizeOpParams{
      .width = 3,
      .height = 2,
      .max_batch_size = 2U,
      .interpolation = holoscan::vision::InterpolationMode::kCubic,
  });
  graph.set_default_clock(graph.add_clock<holoscan::ManualClock>("compile-clock"));

  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "resize",
      .output_port = "output",
      .device = holoscan::DeviceId{0},
  });
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json();
    return 1;
  }
  std::cout << plan.json();
  return 0;
}
