/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef WRAP_HOLOSCAN_AS_GXF_EXTENSION_PING_VARIANT_RX_NATIVE_OP_PING_VARIANT_RX_NATIVE_OP_HPP
#define WRAP_HOLOSCAN_AS_GXF_EXTENSION_PING_VARIANT_RX_NATIVE_OP_PING_VARIANT_RX_NATIVE_OP_HPP

#include <memory>

#include "holoscan/holoscan.hpp"

namespace myops {

class PingVarRxNativeOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingVarRxNativeOp)

  PingVarRxNativeOp() = default;

  void setup(holoscan::OperatorSpec& spec) override;

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 private:
  int count_ = 1;
  holoscan::Parameter<std::shared_ptr<holoscan::BooleanCondition>> boolean_condition_;
};

}  // namespace myops

#endif  // WRAP_HOLOSCAN_AS_GXF_EXTENSION_PING_VARIANT_RX_NATIVE_OP_PING_VARIANT_RX_NATIVE_OP_HPP
