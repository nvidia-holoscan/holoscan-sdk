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

#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/ping_rx/ping_rx.hpp>

namespace holoscan {

// Do not pollute holoscan namespace with utility classes
namespace {

///////////////////////////////////////////////////////////////////////////////
// Flow Tracking Passthrough Operators
///////////////////////////////////////////////////////////////////////////////

class OneInOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(OneInOp)

  OneInOp() = default;

  void setup(OperatorSpec& spec) override { spec.input<gxf::Entity>("in"); }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto in_message = op_input.receive<gxf::Entity>("in");

    HOLOSCAN_LOG_INFO("OneInOp count {}", count_++);
  }

 private:
  int count_ = 1;
};

class OneOutOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(OneOutOp)

  OneOutOp() = default;

  void setup(OperatorSpec& spec) override { spec.output<gxf::Entity>("out"); }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
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
    op_output.emit(in_message.value());
    HOLOSCAN_LOG_INFO("{} count {}", name(), count_++);
  }

 private:
  int count_ = 1;
};

///////////////////////////////////////////////////////////////////////////////
// Flow Tracking Passthrough Application
///////////////////////////////////////////////////////////////////////////////

/* PassthroughApp
 *
 * OneOut--->OneInOneOut--->OneIn
 */
class PassthroughApp : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto one_out =
        make_operator<OneOutOp>("OneOut", make_condition<CountCondition>("count-condition", 3));
    auto one_in = make_operator<OneInOp>("OneIn");
    auto one_in_one_out = make_operator<OneInOneOutOp>("OneInOneOut");

    add_flow(one_out, one_in_one_out, {{"out", "in"}});
    add_flow(one_in_one_out, one_in, {{"out", "in"}});
  }
};

}  // namespace

TEST(Graphs, TestFlowTrackingWithEntityPassthrough) {
  auto app = make_application<PassthroughApp>();
  auto& tracker = app->track(0, 0, 0);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStdout();

  app->run();

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(log_output.find("OneOut,OneInOneOut,OneIn") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

}  // namespace holoscan
