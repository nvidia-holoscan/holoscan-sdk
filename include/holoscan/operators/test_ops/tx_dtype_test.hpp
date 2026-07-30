/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_OPERATORS_TX_DTYPE_TEST_HPP
#define HOLOSCAN_OPERATORS_TX_DTYPE_TEST_HPP

#include <string>

#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

/**
 * @brief Variable type transmitter operator.
 *
 * On each tick, it transmits a fixed value of a specified `data_type` on the output port.
 *
 * ==Named Outputs==
 *
 * - **out** : <data_type>
 *   - A fixed value corresponding to the chosen `data_type`.
 *
 * ==Parameters==
 *
 * - **data_type_**: A string representing the data type for the generated tensor. Must be one of
 *   "int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
 *   "float", "double", "complex<float>", or "complex<double>", "bool" , "std::string" or
 *   "std::unordered_map<std::string, std::string>". Also supports  "std::vector<T>" and
 *   "std::vector<std::vector<T>>" for the types T above. Additionally supports
 *   "std::shared_ptr<T>" types for these types.
 */
class DataTypeTxTestOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DataTypeTxTestOp)

  DataTypeTxTestOp() = default;

  void setup(OperatorSpec& spec) override;

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<std::string> data_type_;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_TX_DTYPE_TEST_HPP */
