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

#ifndef SYSTEM_PING_TENSOR_TX_OP_HPP
#define SYSTEM_PING_TENSOR_TX_OP_HPP

#include <memory>
#include <string>

#include <holoscan/holoscan.hpp>

namespace holoscan {
namespace ops {

class PingTensorTxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTensorTxOp)

  PingTensorTxOp() = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext&, OutputContext& op_output, ExecutionContext& context) override;

  nvidia::gxf::PrimitiveType element_type() {
    if (element_type_.has_value()) { return element_type_.value(); }
    element_type_ = primitive_type(data_type_.get());
    return element_type_.value();
  }

 private:
  nvidia::gxf::PrimitiveType primitive_type(const std::string& data_type);
  std::optional<nvidia::gxf::PrimitiveType> element_type_;

  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<std::string> storage_type_;
  Parameter<int32_t> batch_size_;
  Parameter<int32_t> rows_;
  Parameter<int32_t> columns_;
  Parameter<int32_t> channels_;
  Parameter<std::string> data_type_;
  Parameter<std::string> tensor_name_;
};

}  // namespace ops
}  // namespace holoscan

#endif /* SYSTEM_PING_TENSOR_TX_OP_HPP */
