/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <unordered_set>

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

  void initialize() override {
    HOLOSCAN_LOG_INFO("MinimalOp::initialize() - default value before Operator::initialize(): {}",
                      value_.default_value());
    if (value_.has_value()) {
      HOLOSCAN_LOG_INFO("MinimalOp::initialize() - has value before Operator::initialize(): {}",
                        value_.get());
    } else {
      HOLOSCAN_LOG_INFO("MinimalOp::initialize() - has no value before Operator::initialize()");
    }
    Operator::initialize();
    HOLOSCAN_LOG_INFO("MinimalOp::initialize() - value after Operator::initialize(): {}",
                      value_.get());
  }

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

class ComplexValueParameterOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ComplexValueParameterOp)

  ComplexValueParameterOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(cplx_value_,
               "cplx_value",
               "complex value",
               "complex value stored by the operator",
               {2.5, -3.0});
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    HOLOSCAN_LOG_INFO("ComplexValueParameterOp: count: {}", count_++);
    auto cval = cplx_value_.get();
    HOLOSCAN_LOG_INFO("ComplexValueParameterOp: value: {}{}{}j",
                      cval.real(),
                      cval.imag() >= 0 ? "+" : "",
                      cval.imag());
  };

 private:
  int count_ = 0;
  Parameter<std::complex<double>> cplx_value_;
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

  EXPECT_TRUE(
      log_output.find("MinimalOp::initialize() - has no value before Operator::initialize()") !=
      std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find(
                  "MinimalOp::initialize() - default value before Operator::initialize(): 2.5") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(
      log_output.find("MinimalOp::initialize() - value after Operator::initialize(): 5.3") !=
      std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  EXPECT_TRUE(log_output.find("value: 5.3") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(MinimalNativeOperatorApp, TestMinimalNativeOperatorAppMultiThread) {
  const char* env_orig = std::getenv("HOLOSCAN_LOG_LEVEL");

  // Unset HOLOSCAN_LOG_LEVEL environment variable so that the log level is not overridden
  unsetenv("HOLOSCAN_LOG_LEVEL");

  // use TRACE log level to be able to check detailed messages in the output
  auto log_level_orig = log_level();
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
  EXPECT_TRUE(log_output.find("value: 5.3") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  // check that the expected parameters were sent onto GXF
  EXPECT_TRUE(log_output.find("setting GXF parameter 'worker_thread_number'") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("setting GXF parameter 'stop_on_deadlock'") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("setting GXF parameter 'check_recession_period_ms'") !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("setting GXF parameter 'max_duration_ms'") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // restore the original environment variable
  if (env_orig) {
    setenv("HOLOSCAN_LOG_LEVEL", env_orig, 1);
  } else {
    unsetenv("HOLOSCAN_LOG_LEVEL");
  }

  // restore the log level
  set_log_level(log_level_orig);
}

TEST(MinimalNativeOperatorApp, TestConfigKeys) {
  auto app = make_application<MinimalApp>();

  const std::string config_file = test_config.get_test_data_file("config_keys_test.yaml");
  app->config(config_file);

  std::unordered_set<std::string> keys = app->config_keys();

  // verify that all keys from minimal.yaml are present
  EXPECT_TRUE(keys.find("value") != keys.end());
  EXPECT_TRUE(keys.find("multithread_scheduler") != keys.end());
  EXPECT_TRUE(keys.find("multithread_scheduler.worker_thread_number") != keys.end());
  EXPECT_TRUE(keys.find("multithread_scheduler.stop_on_deadlock") != keys.end());
  EXPECT_TRUE(keys.find("a") != keys.end());
  EXPECT_TRUE(keys.find("a.more") != keys.end());
  EXPECT_TRUE(keys.find("a.more.deeply") != keys.end());
  EXPECT_TRUE(keys.find("a.more.deeply.nested") != keys.end());
  EXPECT_TRUE(keys.find("a.more.deeply.nested.value") != keys.end());

  // verify that no additional keys are present
  EXPECT_EQ(keys.size(), 9);
}

class ComplexValueApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto op = make_operator<ops::ComplexValueParameterOp>(
        "cplx_op", make_condition<CountCondition>(3), from_config("cplx_value"));
    add_operator(op);
  }
};

TEST(MinimalNativeOperatorApp, TestComplexValueApp) {
  auto app = make_application<ComplexValueApp>();

  const std::string config_file = test_config.get_test_data_file("minimal.yaml");
  app->config(config_file);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("value: 5.3+2j") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(MinimalNativeOperatorApp, TestComplexValueAppDefault) {
  auto app = make_application<ComplexValueApp>();

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  // did not provide a config file, so the default value will have been used
  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("value: 2.5-3j") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

}  // namespace holoscan
