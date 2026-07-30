/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_OPERATORS_RX_DTYPE_TEST_HPP
#define HOLOSCAN_OPERATORS_RX_DTYPE_TEST_HPP

#include <string>

#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

/**
 * @brief Type information receiver operator.
 *
 * On each tick, it receives a std::any and prints the type name.
 *
 * ==Named Inputs==
 *
 * - **in** : <data_type>
 *   - Receives value as std::any type.
 */
class DataTypeRxTestOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DataTypeRxTestOp)

  DataTypeRxTestOp() = default;

  void setup(OperatorSpec& spec) override;

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_RX_DTYPE_TEST_HPP */
