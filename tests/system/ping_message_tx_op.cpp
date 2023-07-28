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

#include "ping_message_rx_op.hpp"  // for MessageType enum
#include "ping_message_tx_op.hpp"

#include <memory>
#include <string>
#include <vector>

#include "holoscan/operators/holoviz/codecs.hpp"
#include "holoscan/operators/holoviz/holoviz.hpp"

namespace holoscan {
namespace ops {

void PingMessageTxOp::initialize() {
  register_codec<std::vector<HolovizOp::InputSpec>>(
      std::string("std::vector<holoscan::ops::HolovizOp::InputSpec>"), true);

  // parent class initialize() call must be after the argument additions above
  Operator::initialize();
}

void PingMessageTxOp::setup(OperatorSpec& spec) {
  switch (type_) {
    case MessageType::BOOL:
      spec.output<bool>("out");
      break;
    case MessageType::FLOAT:
      spec.output<float>("out");
      break;
    case MessageType::INT32:
      spec.output<int32_t>("out");
      break;
    case MessageType::UINT32:
      spec.output<uint32_t>("out");
      break;
    case MessageType::STRING:
      spec.output<std::string>("out");
      break;
    case MessageType::VEC_BOOL:
      spec.output<std::vector<bool>>("out");
      break;
    case MessageType::VEC_FLOAT:
      spec.output<std::vector<float>>("out");
      break;
    case MessageType::VEC_STRING:
      spec.output<std::vector<std::string>>("out");
      break;
    case MessageType::SHARED_VEC_STRING:
      spec.output<std::shared_ptr<std::vector<std::string>>>("out");
      break;
    case MessageType::VEC_VEC_BOOL:
      spec.output<std::vector<std::vector<bool>>>("out");
      break;
    case MessageType::VEC_VEC_FLOAT:
      spec.output<std::vector<std::vector<float>>>("out");
      break;
    case MessageType::VEC_VEC_STRING:
      spec.output<std::vector<std::vector<std::string>>>("out");
      break;
    case MessageType::VEC_INPUTSPEC:
      spec.output<std::vector<HolovizOp::InputSpec>>("out");
      break;
    case MessageType::VEC_DOUBLE_LARGE:
      spec.output<std::vector<double>>("out");
      break;
    default:
      throw std::runtime_error("unsupported type");
  }
}

void PingMessageTxOp::compute(InputContext&, OutputContext& op_output, ExecutionContext&) {
  // NOTE: Values in PingMessageTxOp::compute and PingMessageRxOp::compute must remain consistent.
  //       If any value is changed here, please make the corresponding change in PingMessageRxOp.
  switch (type_) {
    case MessageType::BOOL: {
      bool value = true;
      op_output.emit(value, "out");
      break;
    }
    case MessageType::FLOAT: {
      float value = 3.5;
      op_output.emit(value, "out");
      break;
    }
    case MessageType::INT32: {
      int32_t value = -128573;
      op_output.emit(value, "out");
      break;
    }
    case MessageType::UINT32: {
      uint32_t value = 128573;
      op_output.emit(value, "out");
      break;
    }
    case MessageType::STRING: {
      std::string value{"abcdefgh"};
      op_output.emit(value, "out");
      break;
    }
    case MessageType::VEC_BOOL: {
      std::vector<bool> value{false, true, false, true, true};
      op_output.emit(value, "out");
      break;
    }
    case MessageType::VEC_FLOAT: {
      std::vector<float> value{0.5, 1.5, 2.5, 3.5};
      op_output.emit(value, "out");
      break;
    }
    case MessageType::VEC_STRING: {
      std::vector<std::string> value{"a", "bcd", "ef", "ghijk"};
      op_output.emit(value, "out");
      break;
    }
    case MessageType::SHARED_VEC_STRING: {
      std::vector<std::string> value_{"a", "bcd", "ef", "ghijk"};
      auto value = std::make_shared<std::vector<std::string>>(value_);
      op_output.emit(value, "out");
      break;
    }
    case MessageType::VEC_VEC_BOOL: {
      std::vector<std::vector<bool>> value{{false, true, false}, {true, true}};
      op_output.emit(value, "out");
      break;
    }
    case MessageType::VEC_VEC_FLOAT: {
      std::vector<std::vector<float>> value{{0.5, 1.5, 2.5}, {3.5, 4.5}};
      op_output.emit(value, "out");
      break;
    }
    case MessageType::VEC_VEC_STRING: {
      std::vector<std::vector<std::string>> value{{"a", "bcd"}, {"ef", "ghijk"}};
      op_output.emit(value, "out");
      break;
    }
    case MessageType::VEC_INPUTSPEC: {
      std::vector<HolovizOp::InputSpec> specs;
      specs.reserve(2);

      HolovizOp::InputSpec spec1{"tensor1", HolovizOp::InputType::COLOR};

      HolovizOp::InputSpec spec2{"tensor2", HolovizOp::InputType::TRIANGLES};
      HolovizOp::InputSpec::View v2{0.1, 0.1, 0.7, 0.8};
      v2.matrix_ = std::array<float, 16>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
      std::vector<HolovizOp::InputSpec::View> views{v2};
      spec2.views_ = views;

      specs.push_back(spec1);
      specs.push_back(spec2);

      op_output.emit(specs, "out");
      break;
    }
    case MessageType::VEC_DOUBLE_LARGE: {
      // setting size large enough to exceed kDefaultUcxSerializationBufferSize
      std::vector<double> value(1'000'000);
      HOLOSCAN_LOG_INFO("created large double vec of size: {}", value.size());
      for (size_t i = 0; i < value.size(); i++) { value[i] = i; }
      HOLOSCAN_LOG_INFO("finished setting double vec value");
      op_output.emit(value, "out");
      break;
    }
    default: {
      throw std::runtime_error("unsupported type");
    }
  }
}
}  // namespace ops
}  // namespace holoscan
