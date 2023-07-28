/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TESTS_CORE_PING_RX_OP_HPP
#define TESTS_CORE_PING_RX_OP_HPP

#include <string>
#include <unordered_map>
#include <vector>

#include <holoscan/holoscan.hpp>

namespace holoscan {

enum class MessageType {
  BOOL,
  FLOAT,
  INT32,
  UINT32,
  STRING,
  VEC_BOOL,
  VEC_FLOAT,
  VEC_DOUBLE_LARGE,
  VEC_STRING,
  SHARED_VEC_STRING,
  VEC_VEC_BOOL,
  VEC_VEC_FLOAT,
  VEC_VEC_STRING,
  VEC_INPUTSPEC,
};

static const std::unordered_map<MessageType, std::string> message_type_name_map{
    {MessageType::BOOL, "bool"},
    {MessageType::FLOAT, "float"},
    {MessageType::INT32, "int32_t"},
    {MessageType::UINT32, "uint32_t"},
    {MessageType::STRING, "std::string"},
    {MessageType::VEC_BOOL, "std::vector<bool>"},
    {MessageType::VEC_FLOAT, "std::vector<float>"},
    {MessageType::VEC_STRING, "std::vector<std::string>"},
    {MessageType::SHARED_VEC_STRING, "std::shared_ptr<std::vector<std::string>>"},
    {MessageType::VEC_VEC_BOOL, "std::vector<std::vector<bool>>"},
    {MessageType::VEC_VEC_FLOAT, "std::vector<std::vector<float>>"},
    {MessageType::VEC_VEC_STRING, "std::vector<std::vector<std::string>>"},
    {MessageType::VEC_INPUTSPEC, "std::vector<holoscan::ops::HolovizOp::InputSpec>"},
    {MessageType::VEC_DOUBLE_LARGE, "std::vector<double> (large buffer size)"},
};

namespace ops {

class PingMessageRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMessageRxOp)

  PingMessageRxOp() = default;

  void initialize() override;

  void set_message_type(MessageType type) { type_ = type; }

  void setup(OperatorSpec& spec) override;

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override;

 private:
  MessageType type_ = MessageType::FLOAT;
};

}  // namespace ops
}  // namespace holoscan

#endif /* TESTS_CORE_PING_RX_OP_HPP */
