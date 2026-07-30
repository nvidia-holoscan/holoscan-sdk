/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_OPERATORS_PING_TENSOR_RX_PING_TENSOR_RX_HPP
#define HOLOSCAN_OPERATORS_PING_TENSOR_RX_PING_TENSOR_RX_HPP

#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

/**
 * @brief Simple tensor receive operator.
 *
 * This is an example of a native operator with one input port.
 *
 * This operator is intended for use in test cases and example applications.
 *
 * On each tick, it receives a TensorMap and loops over each tensor in the map. For each, it will
 * print the tensor's name and shape.
 *
 * ==Named Inputs==
 *
 * - **in** : nvidia::gxf::Tensor(s)
 *   - One or more received tensors (i.e. a TensorMap).
 */
class PingTensorRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTensorRxOp)

  PingTensorRxOp() = default;

  void setup(OperatorSpec& spec) override;

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<bool> receive_as_tensormap_;
  size_t count_ = 1;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_PING_TENSOR_RX_PING_TENSOR_RX_HPP */
