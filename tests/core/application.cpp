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

class MinimalOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MinimalOp)

  MinimalOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(value_, "value", "value", "value stored by the operator", 2.5);
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    HOLOSCAN_LOG_INFO("MinimalOp: count: {}", count_++);
    HOLOSCAN_LOG_INFO("MinimalOp: value: {}", value_.get());
  };

 private:
  int count_ = 0;
  Parameter<double> value_;
};

}  // namespace ops

class MinimalApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto op = make_operator<ops::MinimalOp>(
        "min_op", make_condition<CountCondition>(3), from_config("value"));
    add_operator(op);
  }
};

TEST(Application, TestMinimalNativeOperatorApp) {
  load_env_log_level();

  auto app = make_application<MinimalApp>();

  const std::string config_file = test_config.get_test_data_file("minimal.yaml");
  app->config(config_file);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("value: 5.3") != std::string::npos);
}

}  // namespace holoscan
