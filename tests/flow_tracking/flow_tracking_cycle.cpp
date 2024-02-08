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
#include <holoscan/operators/ping_rx/ping_rx.hpp>

namespace holoscan {

// Do not pollute holoscan namespace with utility classes
namespace {

///////////////////////////////////////////////////////////////////////////////
// Utility Operators
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

class ThreeInOneOutOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ThreeInOneOutOp)

  ThreeInOneOutOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<gxf::Entity>("in0");
    spec.input<gxf::Entity>("in1");
    spec.input<gxf::Entity>("in2").condition(ConditionType::kNone);  // cycle in-port
    spec.output<gxf::Entity>("out");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto in_message1 = op_input.receive<gxf::Entity>("in0");
    auto in_message2 = op_input.receive<gxf::Entity>("in1");
    auto in_message3 = op_input.receive<gxf::Entity>("in2");

    auto out_message = gxf::Entity::New(&context);
    op_output.emit(out_message);

    HOLOSCAN_LOG_INFO("{} count {}", name(), count_++);
  }

 private:
  int count_ = 1;
};

class TwoInTwoOutOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TwoInTwoOutOp)

  TwoInTwoOutOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<gxf::Entity>("in0").condition(ConditionType::kNone);
    spec.input<gxf::Entity>("in1").condition(ConditionType::kNone);
    spec.output<gxf::Entity>("out0");
    spec.output<gxf::Entity>("out1");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto in_message1 = op_input.receive<gxf::Entity>("in0");
    auto in_message2 = op_input.receive<gxf::Entity>("in1");

    auto out_message = gxf::Entity::New(&context);
    op_output.emit(out_message, "out0");
    op_output.emit(out_message, "out1");

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

class OneOptionalInOneOutOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(OneOptionalInOneOutOp)

  OneOptionalInOneOutOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<gxf::Entity>("in").condition(ConditionType::kNone);
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

// The ASCII graphs are Greg's excellent additions

/* CycleWithSourceApp
 *
 * OneOut--->TwoInOneOut--->OneInOneOut
 *             ^               |
 *             |               |
 *             +---------------+
 */
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

/* MiddleCycleApp
 *
 * OneOut--->TwoInOneOut--->OneInOneOut--->rx
 *              ^            |
 *              |            |
 *              +------------+
 */
class MiddleCycleApp : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto one_out =
        make_operator<OneOutOp>("OneOut", make_condition<CountCondition>("count-condition", 1));
    auto two_in_one_out =
        make_operator<TwoInOneOutOp>("TwoInOneOut", make_condition<CountCondition>(10));
    auto one_in_one_out = make_operator<OneInOneOutOp>("OneInOneOut");
    auto rx = make_operator<OneInOp>("PingRx");

    add_flow(one_out, two_in_one_out, {{"out", "in0"}});
    add_flow(two_in_one_out, one_in_one_out, {{"out", "in"}});
    add_flow(one_in_one_out, two_in_one_out, {{"out", "in1"}});
    add_flow(one_in_one_out, rx);
  }
};

/* CycleWithLeaf
 *
 * root--->middle--->leaf
 *   ^       |
 *   |       |
 *   +-------+
 */
class CycleWithLeaf : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto root = make_operator<OneOptionalInOneOutOp>(
        "root", make_condition<CountCondition>("count-condition", 5));
    auto middle = make_operator<OneInOneOutOp>("middle");
    auto rx = make_operator<OneInOp>("leaf");

    add_flow(root, middle);
    add_flow(middle, root);
    add_flow(middle, rx);
  }
};

/* TwoRootsOneCycle
 *
 *             +--------+
 *             |        ^
 *             |        |
 * root2--->middle2--->last
 *             ^
 *             |
 *             |
 * root1--->middle1
 */
class TwoRootsOneCycle : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;

    auto root1 =
        make_operator<OneOutOp>("root1", make_condition<CountCondition>("count-condition", 5));

    auto root2 =
        make_operator<OneOutOp>("root2", make_condition<CountCondition>("count-condition", 5));

    auto middle1 = make_operator<OneInOneOutOp>("middle1");
    auto middle2 = make_operator<ThreeInOneOutOp>("middle2");

    auto last = make_operator<OneInOneOutOp>("last");

    add_flow(root2, middle2, {{"out", "in0"}});
    add_flow(middle2, last);
    add_flow(last, middle2, {{"out", "in2"}});

    add_flow(root1, middle1);
    add_flow(middle1, middle2, {{"out", "in1"}});
  }
};

/* TwoCyclesVariant1
 *
 * start--->middle--->end
 *   ^       |  ^      |
 *   |       |  |      |
 *   +-------+  +------+
 *
 * middle node is triggered first in this case as start and end have mandatory input ports
 */
class TwoCyclesVariant1 : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;

    auto start = make_operator<OneInOneOutOp>("start", make_condition<CountCondition>(5));

    auto middle = make_operator<TwoInTwoOutOp>("middle");

    auto end = make_operator<OneInOneOutOp>("end");

    // First cycle
    add_flow(start, middle, {{"out", "in0"}});
    add_flow(middle, end, {{"out0", "in"}});

    // Second cycle
    add_flow(middle, start, {{"out1", "in"}});
    add_flow(end, middle, {{"out", "in1"}});
  }
};

/* TwoCyclesVariant2
 *
 * same layout as TwoCyclesVariant1
 *
 * start--->middle--->end
 *   ^       |  ^      |
 *   |       |  |      |
 *   +-------+  +------+
 *
 * The difference is that start is triggered first in this case as start and end have optional
 * input ports.
 */
class TwoCyclesVariant2 : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;

    auto start = make_operator<OneOptionalInOneOutOp>("start", make_condition<CountCondition>(5));

    auto middle = make_operator<TwoInTwoOutOp>("middle");

    auto end = make_operator<OneOptionalInOneOutOp>("end");

    // First cycle
    add_flow(start, middle, {{"out", "in0"}});
    add_flow(middle, end, {{"out0", "in"}});

    // Second cycle
    add_flow(middle, start, {{"out1", "in"}});
    add_flow(end, middle, {{"out", "in1"}});
  }
};

}  // namespace

TEST(Graphs, TestFlowTrackingForCycleWithSource) {
  auto app = make_application<CycleWithSourceApp>();
  auto& tracker = app->track(0, 0, 0);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStdout();

  app->run();

  EXPECT_EQ(app->graph().has_cycle().size(), 1);

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(log_output.find("OneOut,TwoInOneOut,OneInOneOut,TwoInOneOut") != std::string::npos);
  EXPECT_TRUE(log_output.find("TwoInOneOut,OneInOneOut,TwoInOneOut") != std::string::npos);
}

TEST(Graphs, TestFlowTrackingForMiddleCycle) {
  auto app = make_application<MiddleCycleApp>();
  auto& tracker = app->track(0, 0, 0);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStdout();

  app->run();

  EXPECT_EQ(app->graph().has_cycle().size(), 1);

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(log_output.find("OneOut,TwoInOneOut,OneInOneOut,PingRx") != std::string::npos);
  EXPECT_TRUE(log_output.find("OneOut,TwoInOneOut,OneInOneOut,TwoInOneOut") != std::string::npos);
  EXPECT_TRUE(log_output.find("TwoInOneOut,OneInOneOut,PingRx") != std::string::npos);
  EXPECT_TRUE(log_output.find("TwoInOneOut,OneInOneOut,TwoInOneOut") != std::string::npos);
}

TEST(Graphs, TestFlowTrackingForCycleWithLeaf) {
  auto app = make_application<CycleWithLeaf>();
  auto& tracker = app->track(0, 0, 0);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStdout();

  app->run();

  EXPECT_EQ(app->graph().has_cycle().size(), 1);

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(log_output.find("root,middle,leaf") != std::string::npos);
  EXPECT_TRUE(log_output.find("root,middle,root") != std::string::npos);
}

TEST(Graphs, TestFlowTrackingForTwoRootsOneCycle) {
  auto app = make_application<TwoRootsOneCycle>();
  auto& tracker = app->track(0, 0, 0);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStdout();

  app->run();

  EXPECT_EQ(app->graph().has_cycle().size(), 1);

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(log_output.find("middle2,last,middle2") != std::string::npos);
  EXPECT_TRUE(log_output.find("root1,middle1,middle2,last,middle2") != std::string::npos);
  EXPECT_TRUE(log_output.find("root2,middle2,last,middle2") != std::string::npos);
}

TEST(Graphs, TestFlowTrackingForTwoCyclesVariant1) {
  auto app = make_application<TwoCyclesVariant1>();
  auto& tracker = app->track(0, 0, 0);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStdout();

  app->run();

  EXPECT_EQ(app->graph().has_cycle().size(), 2);

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(log_output.find("middle,end,middle") != std::string::npos);
  EXPECT_TRUE(log_output.find("middle,start,middle") != std::string::npos);
}

TEST(Graphs, TestFlowTrackingForTwoCyclesVariant2) {
  auto app = make_application<TwoCyclesVariant2>();
  auto& tracker = app->track(0, 0, 0);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStdout();

  app->run();

  EXPECT_EQ(app->graph().has_cycle().size(), 2);

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(log_output.find("middle,end,middle") != std::string::npos);
  EXPECT_TRUE(log_output.find("middle,start,middle") != std::string::npos);

  // The following two paths have only two messages even though 5 messages are sent from the start
  // This is because no more than 2 messages could travel the following two loops.
  // The origin of the rest of the messages become middle node and they travel in the above two
  // loops.
  EXPECT_TRUE(log_output.find("start,middle,end,middle") != std::string::npos);
  EXPECT_TRUE(log_output.find("start,middle,start") != std::string::npos);
}

}  // namespace holoscan
