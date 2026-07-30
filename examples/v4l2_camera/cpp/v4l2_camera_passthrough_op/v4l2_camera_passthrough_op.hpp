/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <gxf/multimedia/video.hpp>
#include <gxf/std/tensor.hpp>
#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

/**
 * @brief Operator to passthrough the V4L2 buffer to the visualizer.
 *
 * Provide a single output port to connect to the visualizer "receivers" multi-port.
 * Input may be either a Tensor or a VideoBuffer.
 *
 * Note that C++ operator wrapped for Python supports the C++ VideoBuffer component type,
 * whereas a pure Python operator may not support VideoBuffer objects.
 */
class V4L2CameraPassthroughOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(V4L2CameraPassthroughOp)
  V4L2CameraPassthroughOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<nvidia::gxf::Entity>("input");
    spec.output<nvidia::gxf::Entity>("output");
  }
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto maybe_entity = op_input.receive<nvidia::gxf::Entity>("input");
    if (!maybe_entity) {
      HOLOSCAN_LOG_ERROR("Failed to receive message - {}", maybe_entity.error().what());
      return;
    }

    auto entity = maybe_entity.value();

    op_output.emit(entity, "output");
  }
};

}  // namespace holoscan::ops
