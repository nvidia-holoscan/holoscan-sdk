/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

class StringTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(StringTxOp)

  StringTxOp() = default;

  void setup(OperatorSpec& spec) override { spec.output<std::shared_ptr<std::string>>("out"); }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto value = std::make_shared<std::string>(message_);
    if (verbose_) {
      HOLOSCAN_LOG_INFO("{}: sending message", name());
    }
    op_output.emit(std::move(value), "out");
  };

  void set_message(const std::string& message, bool verbose = false) {
    message_ = message;
    verbose_ = verbose;
  }

 private:
  std::string message_{};
  bool verbose_ = false;
};

}  // namespace holoscan::ops
