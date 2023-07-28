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

#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

namespace holoscan {

// Do not pollute holoscan namespace with utility classes
namespace {

///////////////////////////////////////////////////////////////////////////////
// Utility Operators
///////////////////////////////////////////////////////////////////////////////

class OneOutOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(OneOutOp)

  OneOutOp() = default;

  void setup(OperatorSpec& spec) override { spec.output<gxf::Entity>("out"); }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext& context) override {
    auto out_message = gxf::Entity::New(&context);
    op_output.emit(out_message);

    HOLOSCAN_LOG_INFO("{} count {}", name(), count_++);
  }

 private:
  int count_ = 1;
};

class TwoInOneOutOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TwoInOneOutOp)

  TwoInOneOutOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<gxf::Entity>("in0").condition(ConditionType::kNone);
    spec.input<gxf::Entity>("in1").condition(ConditionType::kNone);
    spec.output<gxf::Entity>("out");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto in_message1 = op_input.receive<gxf::Entity>("in0");
    auto in_message2 = op_input.receive<gxf::Entity>("in1");

    auto out_message = gxf::Entity::New(&context);
    op_output.emit(out_message);

    HOLOSCAN_LOG_INFO("{} count {}", name(), count_++);
  }

 private:
  int count_ = 1;
};

class OneInOneOutOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(OneInOneOutOp)

  OneInOneOutOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<gxf::Entity>("in");
    spec.output<gxf::Entity>("out");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto in_message = op_input.receive<gxf::Entity>("in");

    auto out_message = gxf::Entity::New(&context);
    op_output.emit(out_message);

    HOLOSCAN_LOG_INFO("{} count {}", name(), count_++);
  }

 private:
  int count_ = 1;
};

///////////////////////////////////////////////////////////////////////////////
// Utility Applications
///////////////////////////////////////////////////////////////////////////////

class CycleWithSourceApp : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto one_out =
        make_operator<OneOutOp>("OneOut", make_condition<CountCondition>("count-condition", 1));
    auto two_in_one_out =
        make_operator<TwoInOneOutOp>("TwoInOneOut", make_condition<CountCondition>(10));
    auto one_in_one_out = make_operator<OneInOneOutOp>("OneInOneOut");

    add_flow(one_out, two_in_one_out, {{"out", "in0"}});
    add_flow(two_in_one_out, one_in_one_out, {{"out", "in"}});
    add_flow(one_in_one_out, two_in_one_out, {{"out", "in1"}});
  }
};

}  // namespace

///////////////////////////////////////////////////////////////////////////////
// Tests
///////////////////////////////////////////////////////////////////////////////

TEST(Graphs, CycleWithSource) {
  auto app = make_application<CycleWithSourceApp>();

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  // EXPECT_TRUE(log_output.find("OneInOneOut count 10") != std::string::npos);
  EXPECT_TRUE(log_output.find("Failed to initialize fragment") != std::string::npos);
}

}  // namespace holoscan
