/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TESTS_CORE_PING_RX_OP_HPP
#define TESTS_CORE_PING_RX_OP_HPP

#include <string>
#include <unordered_map>
#include <utility>
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
  CAMERA_POSE,
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
    {MessageType::CAMERA_POSE, "std::shared_ptr<std::array<float, 16>>"},
};

namespace ops {

class PingMessageRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMessageRxOp)

  PingMessageRxOp() = default;

  void initialize() override;

  void set_message_types(std::vector<MessageType> types) { types_ = std::move(types); }

  void setup(OperatorSpec& spec) override;

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override;

 private:
  std::vector<MessageType> types_ = {MessageType::FLOAT};
  size_t current_index_ = 0;
};

}  // namespace ops
}  // namespace holoscan

#endif /* TESTS_CORE_PING_RX_OP_HPP */
