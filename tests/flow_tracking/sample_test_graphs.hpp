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

#ifndef TEST_FLOW_TRACKING_SAMPLE_TEST_GRAPHS_HPP
#define TEST_FLOW_TRACKING_SAMPLE_TEST_GRAPHS_HPP

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/ping_rx/ping_rx.hpp>

///////////////////////////////////////////////////////////////////////////////
// Utility Operators
///////////////////////////////////////////////////////////////////////////////
namespace holoscan {

namespace {

// Not polluting the Holoscan namespace with sample operartors and applications
// by using an anonymous namespace.

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

class ThreePathsOneRootOneLeaf : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;

    auto root = make_operator<OneOutOp>("root", make_condition<CountCondition>(5));

    auto middle1 = make_operator<OneInOneOutOp>("middle1");
    auto middle2 = make_operator<OneInOneOutOp>("middle2");
    auto middle3 = make_operator<OneInOneOutOp>("middle3");

    auto middle4 = make_operator<ThreeInOneOutOp>("middle4");

    auto leaf = make_operator<OneInOp>("leaf");

    add_flow(root, middle1);
    add_flow(root, middle2);
    add_flow(root, middle3);

    add_flow(middle1, middle4, {{"out", "in0"}});
    add_flow(middle2, middle4, {{"out", "in1"}});
    add_flow(middle3, middle4, {{"out", "in2"}});

    add_flow(middle4, leaf);
  }
};

}  // namespace
}  // namespace holoscan

#endif /* TEST_FLOW_TRACKING_SAMPLE_TEST_GRAPHS_HPP */
