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

TEST(MinimalNativeOperatorApp, TestMinimalNativeOperatorApp) {
  auto app = make_application<MinimalApp>();

  const std::string config_file = test_config.get_test_data_file("minimal.yaml");
  app->config(config_file);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("value: 5.3") != std::string::npos);
}

TEST(MinimalNativeOperatorApp, TestMinimalNativeOperatorAppMultiThread) {
  // use TRACE log level to be able to check detailed messages in the output
  set_log_level(LogLevel::TRACE);

  auto app = make_application<MinimalApp>();

  const std::string config_file = test_config.get_test_data_file("minimal.yaml");
  app->config(config_file);

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  // configure and assign the scheduler
  app->scheduler(app->make_scheduler<MultiThreadScheduler>(
      "multithread-scheduler", app->from_config("multithread_scheduler")));

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("value: 5.3") != std::string::npos);
  // check that the expected parameters were sent onto GXF
  EXPECT_TRUE(log_output.find("setting GXF parameter 'worker_thread_number'") != std::string::npos);
  EXPECT_TRUE(log_output.find("setting GXF parameter 'stop_on_deadlock'") != std::string::npos);
  EXPECT_TRUE(log_output.find("setting GXF parameter 'check_recession_period_ms'") !=
              std::string::npos);
  EXPECT_TRUE(log_output.find("setting GXF parameter 'max_duration_ms'") != std::string::npos);
}

}  // namespace holoscan
