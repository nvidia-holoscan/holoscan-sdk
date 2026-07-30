/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TESTS_SYSTEM_TENSOR_COMPARE_OP_HPP
#define TESTS_SYSTEM_TENSOR_COMPARE_OP_HPP

#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

namespace holoscan {
namespace ops {

class TensorCompareOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TensorCompareOp)

  TensorCompareOp() = default;

  void setup(OperatorSpec& spec) override;

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override;
};

}  // namespace ops
}  // namespace holoscan

#endif /* TESTS_SYSTEM_TENSOR_COMPARE_OP_HPP */
