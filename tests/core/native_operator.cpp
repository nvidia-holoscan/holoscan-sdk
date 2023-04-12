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

#include "../config.hpp"
#include <holoscan/holoscan.hpp>
#include "common/assert.hpp"

using namespace std::string_literals;

static HoloscanTestConfig test_config;

namespace holoscan {

namespace ops {

class PingTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxOp)

  PingTxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.output<int>("out1");
    spec.output<int>("out2");
  }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    auto value1 = std::make_shared<int>(1);
    op_output.emit(value1, "out1");

    auto value2 = std::make_shared<int>(100);
    op_output.emit(value2, "out2");
  };
};

class PingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRxOp)

  PingRxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(receivers_, "receivers", "Input Receivers", "List of input receivers.", {});
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    auto value_vector = op_input.receive<std::vector<int>>("receivers");

    HOLOSCAN_LOG_INFO("Rx message received (count: {}, size: {})", count_++, value_vector.size());

    HOLOSCAN_LOG_INFO("Rx message value1: {}", *(value_vector[0].get()));
    HOLOSCAN_LOG_INFO("Rx message value2: {}", *(value_vector[1].get()));
  };

 private:
  Parameter<std::vector<IOSpec*>> receivers_;
  int count_ = 1;
};
}  // namespace ops


class NativeOpApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(10));
    auto rx = make_operator<ops::PingRxOp>("rx");

    add_flow(tx, rx, {{"out1", "receivers"}, {"out2", "receivers"}});
  }
};

TEST(NativeOperatorApp, TestNativeOperatorApp) {
  load_env_log_level();

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

}  // namespace holoscan
