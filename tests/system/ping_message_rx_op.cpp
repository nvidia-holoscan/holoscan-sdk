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

#include "ping_message_rx_op.hpp"

#include <memory>
#include <string>
#include <vector>

#include "holoscan/operators/holoviz/codecs.hpp"
#include "holoscan/operators/holoviz/holoviz.hpp"

namespace holoscan {
namespace ops {

void PingMessageRxOp::initialize() {
  // Note: overwrite=true set here since both PingMessageTxOp and PingMessageRxOp register this type
  register_codec<std::vector<HolovizOp::InputSpec>>(
      std::string("std::vector<holoscan::ops::HolovizOp::InputSpec>"), true);

  // parent class initialize() call must be after the argument additions above
  Operator::initialize();
}

void PingMessageRxOp::setup(OperatorSpec& spec) {
  switch (type_) {
    case MessageType::BOOL:
      spec.input<bool>("in");
      break;
    case MessageType::FLOAT:
      spec.input<float>("in");
      break;
    case MessageType::INT32:
      spec.input<int32_t>("in");
      break;
    case MessageType::UINT32:
      spec.input<uint32_t>("in");
      break;
    case MessageType::STRING:
      spec.input<std::string>("in");
      break;
    case MessageType::VEC_BOOL:
      spec.input<std::vector<bool>>("in");
      break;
    case MessageType::VEC_FLOAT:
      spec.input<std::vector<float>>("in");
      break;
    case MessageType::VEC_STRING:
      spec.input<std::vector<std::string>>("in");
      break;
    case MessageType::SHARED_VEC_STRING:
      spec.input<std::shared_ptr<std::vector<std::string>>>("in");
      break;
    case MessageType::VEC_VEC_BOOL:
      spec.input<std::vector<std::vector<bool>>>("in");
      break;
    case MessageType::VEC_VEC_FLOAT:
      spec.input<std::vector<std::vector<float>>>("in");
      break;
    case MessageType::VEC_VEC_STRING:
      spec.input<std::vector<std::vector<std::string>>>("in");
      break;
    case MessageType::VEC_INPUTSPEC:
      spec.input<std::vector<HolovizOp::InputSpec>>("in");
      break;
    case MessageType::VEC_DOUBLE_LARGE:
      spec.input<std::vector<double>>("in");
      break;
    default:
      throw std::runtime_error("unsupported type");
  }
}

void PingMessageRxOp::compute(InputContext& op_input, OutputContext&, ExecutionContext&) {
  // NOTE: Values in PingMessageRxOp::compute and PingMessageTxOp::compute must remain consistent.
  //       If any value is changed in PingMessageTxOp, please also update the check here.
  bool valid_value = false;

  switch (type_) {
    case MessageType::BOOL: {
      auto value = op_input.receive<bool>("in");
      if (value) { valid_value = value.value() == true; }
      break;
    }
    case MessageType::FLOAT: {
      auto value = op_input.receive<float>("in");
      if (value) { valid_value = value.value() == 3.5; }
      break;
    }
    case MessageType::INT32: {
      auto value = op_input.receive<int32_t>("in");
      if (value) { valid_value = value.value() == -128573; }
      break;
    }
    case MessageType::UINT32: {
      auto value = op_input.receive<uint32_t>("in");
      if (value) { valid_value = value.value() == 128573; }
      break;
    }
    case MessageType::STRING: {
      auto value = op_input.receive<std::string>("in");
      if (value) { valid_value = value.value() == std::string("abcdefgh"); }
      break;
    }
    case MessageType::VEC_BOOL: {
      auto value = op_input.receive<std::vector<bool>>("in");
      if (value) {
        std::vector<bool> result = value.value();
        valid_value = result.size() == 5;
        valid_value &= result[0] == false;
        valid_value &= result[1] == true;
        valid_value &= result[2] == false;
        valid_value &= result[3] == true;
        valid_value &= result[4] == true;
      }
      break;
    }
    case MessageType::VEC_FLOAT: {
      auto value = op_input.receive<std::vector<float>>("in");
      if (value) {
        std::vector<float> result = value.value();
        valid_value = result.size() == 4;
        valid_value &= result[0] == 0.5;
        valid_value &= result[1] == 1.5;
        valid_value &= result[2] == 2.5;
        valid_value &= result[3] == 3.5;
      }
      break;
    }
    case MessageType::VEC_STRING: {
      auto value = op_input.receive<std::vector<std::string>>("in");
      if (value) {
        std::vector<std::string> result = value.value();
        valid_value = result.size() == 4;
        valid_value &= result[0] == std::string("a");
        valid_value &= result[1] == std::string("bcd");
        valid_value &= result[2] == std::string("ef");
        valid_value &= result[3] == std::string("ghijk");
      }
      break;
    }
    case MessageType::SHARED_VEC_STRING: {
      auto value = op_input.receive<std::shared_ptr<std::vector<std::string>>>("in");
      if (value) {
        auto result = value.value();
        valid_value = result->size() == 4;
        valid_value &= result->at(0) == std::string("a");
        valid_value &= result->at(1) == std::string("bcd");
        valid_value &= result->at(2) == std::string("ef");
        valid_value &= result->at(3) == std::string("ghijk");
      }
      break;
    }
    case MessageType::VEC_VEC_BOOL: {
      auto value = op_input.receive<std::vector<std::vector<bool>>>("in");
      if (value) {
        std::vector<std::vector<bool>> result = value.value();
        valid_value = result.size() == 2;

        std::vector<bool> res0 = result[0];
        valid_value &= res0.size() == 3;
        valid_value &= res0[0] == false;
        valid_value &= res0[1] == true;
        valid_value &= res0[2] == false;

        std::vector<bool> res1 = result[1];
        valid_value &= res1.size() == 2;
        valid_value &= res1[0] == true;
        valid_value &= res1[1] == true;
      }
      break;
    }
    case MessageType::VEC_VEC_FLOAT: {
      auto value = op_input.receive<std::vector<std::vector<float>>>("in");
      if (value) {
        std::vector<std::vector<float>> result = value.value();
        valid_value = result.size() == 2;

        std::vector<float> res0 = result[0];
        valid_value &= res0.size() == 3;
        valid_value &= res0[0] == 0.5;
        valid_value &= res0[1] == 1.5;
        valid_value &= res0[2] == 2.5;

        std::vector<float> res1 = result[1];
        valid_value &= res1.size() == 2;
        valid_value &= res1[0] == 3.5;
        valid_value &= res1[1] == 4.5;
      }
      break;
    }
    case MessageType::VEC_VEC_STRING: {
      auto value = op_input.receive<std::vector<std::vector<std::string>>>("in");
      if (value) {
        std::vector<std::vector<std::string>> result = value.value();
        valid_value = result.size() == 2;

        std::vector<std::string> res0 = result[0];
        valid_value &= res0.size() == 2;
        valid_value &= res0[0] == std::string("a");
        valid_value &= res0[1] == std::string("bcd");
        std::vector<std::string> res1 = result[1];
        valid_value &= res1.size() == 2;
        valid_value &= res1[0] == std::string("ef");
        valid_value &= res1[1] == std::string("ghijk");
      }
      break;
    }
    case MessageType::VEC_INPUTSPEC: {
      auto value = op_input.receive<std::vector<HolovizOp::InputSpec>>("in");
      if (value) {
        std::vector<HolovizOp::InputSpec> result = value.value();
        valid_value = result.size() == 2;

        HolovizOp::InputSpec res0 = result[0];
        valid_value &= res0.tensor_name_ == std::string("tensor1");
        valid_value &= res0.type_ == HolovizOp::InputType::COLOR;
        valid_value &= res0.views_.size() == 0;
        HolovizOp::InputSpec res1 = result[1];
        valid_value &= res1.tensor_name_ == std::string("tensor2");
        valid_value &= res1.type_ == HolovizOp::InputType::TRIANGLES;
        valid_value &= res1.views_.size() == 1;
      }
      break;
    }
    case MessageType::VEC_DOUBLE_LARGE: {
      auto value = op_input.receive<std::vector<double>>("in");
      if (value) {
        std::vector<double> result = value.value();
        valid_value = result.size() == 1'000'000;
        if (valid_value) {
          valid_value &= result[0] == 0.0;
          valid_value &= result[999999] == 999999.0;
        }
      }
      break;
    }
    default: {
      throw std::runtime_error("unsupported type");
    }
  }
  if (valid_value) {
    HOLOSCAN_LOG_INFO("Found expected value in deserialized message.");
  } else {
    HOLOSCAN_LOG_ERROR("Found unexpected value in deserialized message.");
  }
}
}  // namespace ops
}  // namespace holoscan
