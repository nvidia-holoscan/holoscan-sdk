/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <gxf/core/gxf.h>

#include <memory>
#include <string>

#include <holoscan/holoscan.hpp>

#include "../config.hpp"
#include "common/assert.hpp"

using namespace std::string_literals;

static HoloscanTestConfig test_config;

namespace holoscan {

// Do not pollute holoscan namespace with utility classes
namespace {

// class PingRxNoCondition : public PingRxOp {
//  public:
class PingRxNoCondition : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRxNoCondition)

  PingRxNoCondition() = default;
  void setup(OperatorSpec& spec) override { spec.input<int>("in").condition(ConditionType::kNone); }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto message = op_input.receive<int>("in");
    if (message) {
      HOLOSCAN_LOG_INFO("PingRxNoCondition: received value {}", message.value());
    } else {
      HOLOSCAN_LOG_INFO("PingRxNoCondition: no value received");
    }
  }
};

class PingTxNoCondition : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxNoCondition)

  PingTxNoCondition() = default;

  void setup(OperatorSpec& spec) override {
    spec.output<int>("out").condition(ConditionType::kNone);
  }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    op_output.emit<int>(count_++, "out");
  }

 private:
  int count_ = 1;
};

}  // namespace

// Ping test app app without any message conditions on the input or output ports
class PingOpMissingMessageConditions : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<PingTxNoCondition>("tx", make_condition<CountCondition>(10));
    auto rx = make_operator<PingRxNoCondition>("rx", make_condition<CountCondition>(20));

    add_flow(tx, rx);
  }
};

// Test that message conditions can be added via the constructor
// Cases test that specifying as any of std::string, YAML::Node or const char* works.
class PingOpWithMessageConditions : public holoscan::Application {
 public:
  enum class TestArgType { kCharPtr, kString, kYAMLNode };

  void compose() override {
    using namespace holoscan;
    std::shared_ptr<DownstreamMessageAffordableCondition> tx_msg_cond;
    std::shared_ptr<MessageAvailableCondition> rx_msg_cond;
    switch (test_arg_type_) {
      case TestArgType::kCharPtr:
        tx_msg_cond = make_condition<DownstreamMessageAffordableCondition>(
            "tx_msg_cond", Arg("transmitter", "out"));
        rx_msg_cond =
            make_condition<MessageAvailableCondition>("rx_msg_cond", Arg("receiver", "in"));
        break;
      case TestArgType::kString:
        tx_msg_cond = make_condition<DownstreamMessageAffordableCondition>(
            "tx_msg_cond", Arg("transmitter", std::string("out")));
        rx_msg_cond = make_condition<MessageAvailableCondition>("rx_msg_cond",
                                                                Arg("receiver", std::string("in")));
        break;
      case TestArgType::kYAMLNode:
        tx_msg_cond = make_condition<DownstreamMessageAffordableCondition>(
            "tx_msg_cond", from_config("downstream_affordable"));
        rx_msg_cond = make_condition<MessageAvailableCondition>("rx_msg_cond",
                                                                from_config("message_available"));
        break;
    }
    auto tx =
        make_operator<PingTxNoCondition>("tx", make_condition<CountCondition>(10), tx_msg_cond);
    auto rx = make_operator<PingRxNoCondition>("rx", rx_msg_cond);

    add_flow(tx, rx);
  }

  void set_test_arg_type(TestArgType test_arg_type) { test_arg_type_ = test_arg_type; }

 private:
  TestArgType test_arg_type_{TestArgType::kString};
};

TEST(MessageConditionViaConstructorTests, TestPingOpMissingMessageConditions) {
  auto app = make_application<PingOpMissingMessageConditions>();

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  // No input port condition so "rx" should compute even if no message was received
  EXPECT_TRUE(log_output.find("PingRxNoCondition: no value received") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

class ArgTypeParameterizedTestFixture
    : public ::testing::TestWithParam<PingOpWithMessageConditions::TestArgType> {};

INSTANTIATE_TEST_CASE_P(MessageConditionArgTypeTests, ArgTypeParameterizedTestFixture,
                        ::testing::Values(PingOpWithMessageConditions::TestArgType::kCharPtr,
                                          PingOpWithMessageConditions::TestArgType::kString,
                                          PingOpWithMessageConditions::TestArgType::kYAMLNode));

TEST_P(ArgTypeParameterizedTestFixture, TestPingOpWithMessageConditions) {
  auto test_arg_type = GetParam();
  auto app = make_application<PingOpWithMessageConditions>();
  app->set_test_arg_type(test_arg_type);

  const std::string config_file = test_config.get_test_data_file("message_conditions.yaml");
  app->config(config_file);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  // "rx" should never have called compute without a value present
  EXPECT_TRUE(log_output.find("PingRxNoCondition: no value received") == std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  // Value of 10 must have been received
  EXPECT_TRUE(log_output.find("PingRxNoCondition: received value 10") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  // Value of 11 must not have been received
  EXPECT_TRUE(log_output.find("PingRxNoCondition: received value 11") == std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

}  // namespace holoscan
