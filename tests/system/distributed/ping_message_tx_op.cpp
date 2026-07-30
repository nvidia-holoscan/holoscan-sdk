/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ping_message_tx_op.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/executors/gxf/gxf_executor.hpp>
#include <holoscan/operators/holoviz/codecs.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include "ping_message_rx_op.hpp"  // for MessageType enum

namespace holoscan {
namespace ops {

void PingMessageTxOp::initialize() {
  holoscan::gxf::GXFExecutor::register_codec<std::vector<HolovizOp::InputSpec>>(
      std::string("std::vector<holoscan::ops::HolovizOp::InputSpec>"), true);

  // parent class initialize() call must be after the argument additions above
  Operator::initialize();
}

void PingMessageTxOp::setup(OperatorSpec& spec) {
  // Use generic output that works with UCX serialization for all types
  // The actual type is handled at runtime by the serialization layer
  spec.output<nvidia::gxf::Entity>("out");
}

void PingMessageTxOp::compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
                              [[maybe_unused]] ExecutionContext& context) {
  // NOTE: Values in PingMessageTxOp::compute and PingMessageRxOp::compute must remain consistent.
  //       If any value is changed here, please make the corresponding change in PingMessageRxOp.

  // Get current type and cycle to next
  MessageType current_type = types_[current_index_];
  current_index_ = (current_index_ + 1) % types_.size();

  HOLOSCAN_LOG_INFO("Transmitting test case: {}", message_type_name_map.at(current_type));

  // store metadata with a few types
  // (value serialization uses the same codecs as for holoscan::Message, so just test a few here)
  auto meta = metadata();
  meta->set("bool", true);
  meta->set("string", std::string("defg"));
  meta->set("vec", std::vector<float>{1.0, 1.0, 3.0});

  switch (current_type) {
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
      spec2.views_ = std::move(views);

      specs.push_back(std::move(spec1));
      specs.push_back(std::move(spec2));

      op_output.emit(std::move(specs), "out");
      break;
    }
    case MessageType::VEC_DOUBLE_LARGE: {
      // setting size large enough to exceed kDefaultUcxSerializationBufferSize
      std::vector<double> value(1'000'000);
      HOLOSCAN_LOG_INFO("created large double vec of size: {}", value.size());
      for (size_t i = 0; i < value.size(); i++) {
        value[i] = i;
      }
      HOLOSCAN_LOG_INFO("finished setting double vec value");
      op_output.emit(value, "out");
      break;
    }
    case MessageType::CAMERA_POSE: {
      std::array<float, 16> value_{1., 0., 0., 0., 0., 2., 0., 0., 0., 0., 3., 0., 0., 0., 0., 0.};
      auto value = std::make_shared<std::array<float, 16>>(value_);
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
