/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

static HoloscanTestConfig test_config;

namespace holoscan {

namespace ops {

class MinimalThrowOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MinimalThrowOp)

  MinimalThrowOp() = default;

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    throw std::runtime_error("Exception occurred in MinimalThrowOp::compute");
  };
};

}  // namespace ops

class MinimalThrowApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto op = make_operator<ops::MinimalThrowOp>("min_op", make_condition<CountCondition>(3));
    add_operator(op);
  }
};

TEST(MinimalNativeOperatorApp, TestComputeMethodExceptionHandling) {
  load_env_log_level();

  auto app = make_application<MinimalThrowApp>();

  const std::string config_file = test_config.get_test_data_file("minimal.yaml");
  app->config(config_file);

  EXPECT_EXIT(
      app->run(), testing::ExitedWithCode(1), "Exception occurred in MinimalThrowOp::compute");
}

}  // namespace holoscan
