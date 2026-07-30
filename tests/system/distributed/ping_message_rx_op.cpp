/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ping_message_rx_op.hpp"

#include <memory>
#include <string>
#include <vector>

#include <holoscan/core/executors/gxf/gxf_executor.hpp>
#include <holoscan/operators/holoviz/codecs.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

namespace holoscan {
namespace ops {

void PingMessageRxOp::initialize() {
  // Note: overwrite=true set here since both PingMessageTxOp and PingMessageRxOp register this type
  holoscan::gxf::GXFExecutor::register_codec<std::vector<HolovizOp::InputSpec>>(
      std::string("std::vector<holoscan::ops::HolovizOp::InputSpec>"), true);

  // parent class initialize() call must be after the argument additions above
  Operator::initialize();
}

void PingMessageRxOp::setup(OperatorSpec& spec) {
  // Use generic input that works with UCX serialization for all types
  // The actual type is handled at runtime by the serialization layer
  spec.input<nvidia::gxf::Entity>("in");
}

void PingMessageRxOp::compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
                              [[maybe_unused]] ExecutionContext& context) {
  // NOTE: Values in PingMessageRxOp::compute and PingMessageTxOp::compute must remain consistent.
  //       If any value is changed in PingMessageTxOp, please also update the check here.

  // Get current type and cycle to next
  if (types_.empty()) {
    HOLOSCAN_LOG_ERROR("No message types configured");
    return;
  }
  MessageType current_type = types_[current_index_];
  current_index_ = (current_index_ + 1) % types_.size();

  bool valid_value = false;

  switch (current_type) {
    case MessageType::BOOL: {
      auto value = op_input.receive<bool>("in");
      if (value) {
        valid_value = value.value() == true;
      }
      break;
    }
    case MessageType::FLOAT: {
      auto value = op_input.receive<float>("in");
      if (value) {
        valid_value = value.value() == 3.5;
      }
      break;
    }
    case MessageType::INT32: {
      auto value = op_input.receive<int32_t>("in");
      if (value) {
        valid_value = value.value() == -128573;
      }
      break;
    }
    case MessageType::UINT32: {
      auto value = op_input.receive<uint32_t>("in");
      if (value) {
        valid_value = value.value() == 128573;
      }
      break;
    }
    case MessageType::STRING: {
      auto value = op_input.receive<std::string>("in");
      if (value) {
        valid_value = value.value() == std::string("abcdefgh");
      }
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
    case MessageType::CAMERA_POSE: {
      auto value = op_input.receive<std::shared_ptr<std::array<float, 16>>>("in");
      if (value) {
        auto result = value.value();
        valid_value = result->size() == 16;
        valid_value &= result->at(0) == 1.0;
        valid_value &= result->at(1) == 0.0;
        valid_value &= result->at(5) == 2.0;
        valid_value &= result->at(10) == 3.0;
        valid_value &= result->at(15) == 0.0;
      }
      break;
    }
    default: {
      throw std::runtime_error("unsupported type");
    }
  }
  if (is_metadata_enabled()) {
    // validate the metadata (values were set by PingMessageTxOp)
    auto meta = metadata();
    valid_value &= meta->get<bool>("bool") == true;
    valid_value &= meta->get<std::string>("string") == std::string("defg");
    auto vec = meta->get<std::vector<float>>("vec");
    valid_value &= vec[0] == 1.0;
    valid_value &= vec[1] == 1.0;
    valid_value &= vec[2] == 3.0;
  }
  if (valid_value) {
    HOLOSCAN_LOG_INFO("Found expected value in deserialized message for test case: {}",
                      message_type_name_map.at(current_type));
  } else {
    HOLOSCAN_LOG_ERROR("FAILED test case: {} - Found unexpected value in deserialized message.",
                       message_type_name_map.at(current_type));
  }
}
}  // namespace ops
}  // namespace holoscan
