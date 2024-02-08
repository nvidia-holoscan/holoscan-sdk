/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>

#include <holoscan/holoscan.hpp>

#include "../config.hpp"
#include "common/assert.hpp"

#include "ping_rx_op.hpp"
#include "ping_tx_op.hpp"

using namespace std::string_literals;

static HoloscanTestConfig test_config;

namespace holoscan {

// Do not pollute holoscan namespace with utility classes
namespace {

/// @brief integer forwarding with matching input and output port name
class ForwardTestOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ForwardTestOp)

  ForwardTestOp() = default;

  void setup(OperatorSpec& spec) override {
    // intentionally use matching name for the input and output port
    spec.input<int>("data");
    spec.output<int>("data");
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    auto value = op_input.receive<int>("data").value();
    op_output.emit(value, "data");
  }
};

/// @brief version of ForwardTestOp with duplicate outputs
class ForwardTestOpTwoOutputs : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ForwardTestOpTwoOutputs)

  ForwardTestOpTwoOutputs() = default;

  void setup(OperatorSpec& spec) override {
    // intentionally use matching name for the input and output port
    spec.input<int>("data");
    spec.output<int>("data");
    spec.output<int>("data2");
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    auto value = op_input.receive<int>("data").value();
    op_output.emit(value, "data");
    op_output.emit(value, "data2");
  }
};

/// @brief version of ForwardTestOp with two inputs
class ForwardTestOpTwoInputs : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ForwardTestOpTwoInputs)

  ForwardTestOpTwoInputs() = default;

  void setup(OperatorSpec& spec) override {
    // intentionally use matching name for the input and output port
    spec.input<int>("data");
    spec.input<int>("data2");
    spec.output<int>("data");
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    auto value = op_input.receive<int>("data").value();
    auto value2 = op_input.receive<int>("data2").value();
    op_output.emit(value + value2, "data");
  }
};

}  // namespace

class NativeOpApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<ops::PingMultiTxOp>("tx", make_condition<CountCondition>(10));
    auto rx = make_operator<ops::PingMultiRxOp>("rx");

    add_flow(tx, rx, {{"out1", "receivers"}, {"out2", "receivers"}});
  }
};

class NativeForwardOpApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<ops::PingMultiTxOp>("tx", make_condition<CountCondition>(10));
    // ForwardOp tests case where the input and output port have the same name
    auto mx = make_operator<ForwardTestOp>("mx");
    auto rx = make_operator<ops::PingMultiRxOp>("rx");

    add_flow(tx, rx, {{"out1", "receivers"}});
    add_flow(tx, mx, {{"out2", "data"}});
    add_flow(mx, rx, {{"data", "receivers"}});
  }
};

/// @brief version of NativeForwardOpApp with a dangling (unconnected) output port
class NativeForwardOpAppDanglingOutput : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<ops::PingMultiTxOp>("tx", make_condition<CountCondition>(10));
    // ForwardOp tests case where the input and output port have the same name
    auto mx = make_operator<ForwardTestOpTwoOutputs>("mx");
    auto rx = make_operator<ops::PingMultiRxOp>("rx");

    // intentionally don't connect data2 output of mx to check that segfault does not occur
    add_flow(tx, rx, {{"out1", "receivers"}});
    add_flow(tx, mx, {{"out2", "data"}});
    add_flow(mx, rx, {{"data", "receivers"}});
  }
};

/// @brief version of NativeForwardOpApp with a dangling (unconnected) output port
class NativeForwardOpAppDanglingInput : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<ops::PingMultiTxOp>("tx", make_condition<CountCondition>(10));
    // ForwardOp tests case where the input and output port have the same name
    auto mx = make_operator<ForwardTestOpTwoInputs>("mx");
    auto rx = make_operator<ops::PingMultiRxOp>("rx");

    // intentionally don't connect data2 input of mx to check that segfault does not occur
    add_flow(tx, rx, {{"out1", "receivers"}});
    add_flow(tx, mx, {{"out2", "data"}});
    add_flow(mx, rx, {{"data", "receivers"}});
  }
};

TEST(NativeOperatorPingApp, TestNativeOperatorPingApp) {
  auto app = make_application<NativeOpApp>();

  const std::string config_file = test_config.get_test_data_file("minimal.yaml");
  app->config(config_file);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("value1: 1") != std::string::npos);
  EXPECT_TRUE(log_output.find("value2: 100") != std::string::npos);
}

TEST(NativeOperatorPingApp, TestNativeOperatorForwardApp) {
  auto app = make_application<NativeForwardOpApp>();

  const std::string config_file = test_config.get_test_data_file("minimal.yaml");
  app->config(config_file);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("value1: 1") != std::string::npos);
  EXPECT_TRUE(log_output.find("value2: 100") != std::string::npos);
}

TEST(NativeOperatorPingApp, TestNativeForwardOpAppDanglingOutput) {
  auto app = make_application<NativeForwardOpAppDanglingOutput>();

  const std::string config_file = test_config.get_test_data_file("minimal.yaml");
  app->config(config_file);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  // string tested here is from GXF itself, so may have to update it as GXF is updated
  EXPECT_TRUE(log_output.find("Connection not found for Tx data2") != std::string::npos);
}

TEST(NativeOperatorPingApp, TestNativeForwardOpAppDanglingInput) {
  auto app = make_application<NativeForwardOpAppDanglingInput>();

  const std::string config_file = test_config.get_test_data_file("minimal.yaml");
  app->config(config_file);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  // No error will be logged in the dangling input case, but we can test that the
  // app will deadlock so that value1 is not printed
  EXPECT_TRUE(log_output.find("value1: ") == std::string::npos);
}

}  // namespace holoscan
