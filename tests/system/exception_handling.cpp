/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <yaml-cpp/yaml.h>

#include <memory>
#include <string>

#include <holoscan/holoscan.hpp>
#include "../config.hpp"
#include "common/assert.hpp"

static HoloscanTestConfig test_config;

namespace {

enum class ThrowMethod : uint8_t { kStart, kStop, kCompute, kInitialize, kNone };

class MethodParmeterizedTestFixture : public ::testing::TestWithParam<ThrowMethod> {};

}  // namespace

// need convert for ThrowMethod to be able to use it as a Parameter
template <>
struct YAML::convert<ThrowMethod> {
  static Node encode(const ThrowMethod& rhs) {
    Node node;
    node = static_cast<uint8_t>(rhs);
    return node;
  }

  static bool decode(const Node& node, ThrowMethod& rhs) {
    if (!node.IsScalar()) return false;
    uint8_t throw_method = node.as<uint8_t>();
    if (throw_method <= static_cast<uint8_t>(ThrowMethod::kNone)) {
      rhs = static_cast<ThrowMethod>(throw_method);
      return true;
    }
    return false;
  }
};

namespace holoscan {

namespace ops {

class MinimalThrowOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MinimalThrowOp)

  MinimalThrowOp() = default;

  void initialize() override {
    register_converter<ThrowMethod>();

    Operator::initialize();

    if (throw_type_.get() == ThrowMethod::kInitialize) {
      throw std::runtime_error("Exception occurred in MinimalThrowOp::initialize");
    }
  };

  void start() override {
    if (throw_type_.get() == ThrowMethod::kStart) {
      throw std::runtime_error("Exception occurred in MinimalThrowOp::start");
    }
  };

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    if (throw_type_.get() == ThrowMethod::kCompute) {
      throw std::runtime_error("Exception occurred in MinimalThrowOp::compute");
    }
  };

  void stop() override {
    if (throw_type_.get() == ThrowMethod::kStop) {
      throw std::runtime_error("Exception occurred in MinimalThrowOp::stop");
    }
  };

  void setup(OperatorSpec& spec) override {
    spec.param(
        throw_type_, "throw_type", "Throw Type", "Specifies which method throws the exception");
  }

 private:
  Parameter<ThrowMethod> throw_type_;
};

}  // namespace ops

class MinimalThrowApp : public holoscan::Application {
 public:
  /**
   * @brief Construct a new MinimalThrowApp object
   *
   * @param throw_type enum controlling which method (if any) throws an exception
   */
  explicit MinimalThrowApp(ThrowMethod throw_type) : throw_type_(throw_type) {}

  void compose() override {
    using namespace holoscan;
    auto op = make_operator<ops::MinimalThrowOp>(
        "min_op", make_condition<CountCondition>(3), Arg("throw_type", throw_type_));
    add_operator(op);
  }

 private:
  ThrowMethod throw_type_ = ThrowMethod::kNone;
};

INSTANTIATE_TEST_CASE_P(MinimalNativeOperatorAppTests, MethodParmeterizedTestFixture,
                        ::testing::Values(ThrowMethod::kStart, ThrowMethod::kStop,
                                          ThrowMethod::kCompute, ThrowMethod::kInitialize,
                                          ThrowMethod::kNone));

TEST_P(MethodParmeterizedTestFixture, TestMethodExceptionHandling) {
  ThrowMethod throw_method = GetParam();
  auto app = make_application<MinimalThrowApp>(throw_method);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  if (throw_method == ThrowMethod::kNone) {
    EXPECT_NO_THROW({ app->run(); });
  } else {
    EXPECT_THROW({ app->run(); }, std::runtime_error);
  }

  std::string log_output = testing::internal::GetCapturedStderr();
  if ((throw_method != ThrowMethod::kNone)) {
    EXPECT_TRUE(log_output.find("Exception occurred in MinimalThrowOp") != std::string::npos);
    if (throw_method != ThrowMethod::kInitialize) {
      // exception in initialize is before graph start, so this would not be printed
      EXPECT_TRUE(log_output.find("Graph execution error: ") != std::string::npos);
    }
  }
}

}  // namespace holoscan
