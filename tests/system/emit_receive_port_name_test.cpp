/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>

#include <string>

#include <holoscan/holoscan.hpp>

namespace holoscan {

// Do not pollute holoscan namespace with utility classes
namespace {

///////////////////////////////////////////////////////////////////////////////
// Utility Operators
///////////////////////////////////////////////////////////////////////////////

class PingTxImplicitOutputPortNameOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxImplicitOutputPortNameOp)

  PingTxImplicitOutputPortNameOp() = default;

  void setup(OperatorSpec& spec) override { spec.output<int>("out"); }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    int value = index_++;
    HOLOSCAN_LOG_INFO("Emitting value: {}", value);
    op_output.emit(value);  // intentionally not specifying the port name
  }

 private:
  int index_ = 0;
};

class PingRxImplicitInputPortNameOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRxImplicitInputPortNameOp)

  PingRxImplicitInputPortNameOp() = default;

  void setup(OperatorSpec& spec) override { spec.input<int>("in"); }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto maybe_value = op_input.receive<int>();  // intentionally not specifying the port name
    if (!maybe_value) {
      auto error_msg = fmt::format("Operator '{}' failed to receive message from port 'in': {}",
                                   name(),
                                   maybe_value.error().what());
      HOLOSCAN_LOG_ERROR(error_msg);
      throw std::runtime_error(error_msg);
    }
    int value = maybe_value.value();
    HOLOSCAN_LOG_INFO("Rx message value: {}", value);
  }
};

class SimpleOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SimpleOp)

  SimpleOp() = default;

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("I am here - {}", name());
  }
};

///////////////////////////////////////////////////////////////////////////////
// Utility Applications
///////////////////////////////////////////////////////////////////////////////

class EmptyPortNameApp : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto tx =
        make_operator<PingTxImplicitOutputPortNameOp>("PingTx", make_condition<CountCondition>(1));
    auto rx = make_operator<PingRxImplicitInputPortNameOp>("PingRx");

    auto dummy1 = make_operator<SimpleOp>("Dummy1");
    auto dummy2 = make_operator<SimpleOp>("Dummy2");

    add_flow(tx, rx);
    add_flow(tx, dummy1);
    add_flow(rx, dummy2);
  }
};

}  // namespace

///////////////////////////////////////////////////////////////////////////////
// Tests
///////////////////////////////////////////////////////////////////////////////

TEST(EmitReceivePortName, EmptyPortName) {
  auto app = make_application<EmptyPortNameApp>();

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Emitting value: 0") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("I am here - Dummy1") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Rx message value: 0") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("I am here - Dummy2") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

}  // namespace holoscan
