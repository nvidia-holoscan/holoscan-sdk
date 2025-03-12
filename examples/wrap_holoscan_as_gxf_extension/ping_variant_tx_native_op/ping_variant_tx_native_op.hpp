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

#ifndef WRAP_HOLOSCAN_AS_GXF_EXTENSION_PING_VARIANT_TX_NATIVE_OP_PING_VARIANT_TX_NATIVE_OP_HPP
#define WRAP_HOLOSCAN_AS_GXF_EXTENSION_PING_VARIANT_TX_NATIVE_OP_PING_VARIANT_TX_NATIVE_OP_HPP

#include <memory>
#include <string>
#include <vector>

#include "holoscan/holoscan.hpp"
#include "holoscan/utils/operator_runner.hpp"

#include "../ping_variant_custom_native_res/ping_variant_custom_native_res.hpp"

namespace myops {

class PingVarTxNativeOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingVarTxNativeOp)

  PingVarTxNativeOp() = default;

  void initialize() override;

  void setup(holoscan::OperatorSpec& spec) override;

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 private:
  holoscan::Parameter<std::shared_ptr<myres::PingVarCustomNativeRes>> custom_resource_;
  // Additional parameters for test purposes.
  holoscan::Parameter<int> numeric_;
  holoscan::Parameter<std::vector<float>> numeric_array_;
  holoscan::Parameter<int> optional_numeric_;
  holoscan::Parameter<std::vector<int>> optional_numeric_array_;
  holoscan::Parameter<bool> boolean_;
  holoscan::Parameter<void*> optional_void_ptr_;
  holoscan::Parameter<std::string> string_;
  holoscan::Parameter<std::shared_ptr<holoscan::Resource>> optional_resource_;

  std::shared_ptr<holoscan::ops::OperatorRunner> op_int_generator_;
  std::shared_ptr<holoscan::ops::OperatorRunner> op_processing_;
};

}  // namespace myops

#endif /* WRAP_HOLOSCAN_AS_GXF_EXTENSION_PING_VARIANT_TX_NATIVE_OP_PING_VARIANT_TX_NATIVE_OP_HPP */
