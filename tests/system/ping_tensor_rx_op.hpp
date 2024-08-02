/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SYSTEM_PING_TENSOR_RX_OP_HPP
#define SYSTEM_PING_TENSOR_RX_OP_HPP

#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

namespace holoscan {
namespace ops {

class PingTensorRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTensorRxOp)

  PingTensorRxOp() = default;

  void setup(OperatorSpec& spec) override;

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override;

 private:
  int count_ = 1;
  Parameter<std::string> tensor_name_;
};

}  // namespace ops
}  // namespace holoscan

#endif /* SYSTEM_PING_TENSOR_RX_OP_HPP */
