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

#ifndef SYSTEM_DISTRIBUTED_UTILITY_APPS_HPP
#define SYSTEM_DISTRIBUTED_UTILITY_APPS_HPP

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include <holoscan/holoscan.hpp>

namespace holoscan {

// Do not pollute holoscan namespace with utility classes
namespace {

///////////////////////////////////////////////////////////////////////////////
// Utility Operators
///////////////////////////////////////////////////////////////////////////////

class SingleOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SingleOp)

  SingleOp() = default;

  void setup(OperatorSpec& spec) override { spec.param(id_, "id", "id", "id", 0L); }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    HOLOSCAN_LOG_INFO("SingleOp {}.{}: {} - {}", fragment()->name(), name(), id_.get(), index_++);
    // sleep for 0.1 seconds
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  int index() const { return index_; }

 private:
  Parameter<int64_t> id_{0};
  int index_ = 1;
};

class PingTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxOp)

  PingTxOp() = default;

  void setup(OperatorSpec& spec) override { spec.output<gxf::Entity>("out"); }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext& context) override {
    auto out_message = gxf::Entity::New(&context);

    op_output.emit(out_message);
    HOLOSCAN_LOG_INFO("Tx {}.{} message sent", fragment()->name(), name());
  };
};

class PingTxTwoOutputsOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxTwoOutputsOp)

  PingTxTwoOutputsOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.output<gxf::Entity>("out1");
    spec.output<gxf::Entity>("out2");
  }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext& context) override {
    auto out_message1 = gxf::Entity::New(&context);
    auto out_message2 = gxf::Entity::New(&context);

    op_output.emit(out_message1, "out1");
    op_output.emit(out_message2, "out2");
    HOLOSCAN_LOG_INFO("Tx {}.{} message sent", fragment()->name(), name());
  };
};

class PingMxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMxOp)

  PingMxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<gxf::Entity>("in");
    spec.output<gxf::Entity>("out");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto in_message = op_input.receive<gxf::Entity>("in");
    auto out_message = gxf::Entity::New(&context);

    // TODO: Send a tensor. For now, the output message is just an empty Entity.
    // out_message.add(tensor, "tensor");

    op_output.emit(out_message);
    HOLOSCAN_LOG_INFO("Mx {}.{} message sent", fragment()->name(), name());
  };
};

class PingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRxOp)

  PingRxOp() = default;

  void setup(OperatorSpec& spec) override { spec.input<gxf::Entity>("in"); }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    auto in_message = op_input.receive<gxf::Entity>("in");
    HOLOSCAN_LOG_INFO("Rx {}.{} message received count: {}", fragment()->name(), name(), count_++);
  };

 private:
  int count_ = 1;
};

class PingRxTwoInputsOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRxTwoInputsOp)

  PingRxTwoInputsOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<gxf::Entity>("in1");
    spec.input<gxf::Entity>("in2");
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    auto in_message1 = op_input.receive<gxf::Entity>("in1");
    auto in_message2 = op_input.receive<gxf::Entity>("in2");
    HOLOSCAN_LOG_INFO("Rx {}.{} message received count: {}", fragment()->name(), name(), count_++);
  };

 private:
  int count_ = 1;
};

class PingMultiReceiversParamRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMultiReceiversParamRxOp)

  PingMultiReceiversParamRxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<std::vector<gxf::Entity>>("receivers", IOSpec::kAnySize);
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    auto value_vector = op_input.receive<std::vector<gxf::Entity>>("receivers").value();
    HOLOSCAN_LOG_INFO("RxParam {}.{} message received (count: {}, size: {})",
                      fragment()->name(),
                      name(),
                      count_++,
                      value_vector.size());
  };

 private:
  int count_ = 1;
};

class BroadcastOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(BroadcastOp)

  BroadcastOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<gxf::Entity>("in");
    spec.output<gxf::Entity>("out1");
    spec.output<gxf::Entity>("out2");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto in_message = op_input.receive<gxf::Entity>("in");

    op_output.emit(in_message.value(), "out1");
    op_output.emit(in_message.value(), "out2");
    HOLOSCAN_LOG_INFO("Broadcast {}.{} message sent", fragment()->name(), name());
  };
};

///////////////////////////////////////////////////////////////////////////////
// Utility Fragments
///////////////////////////////////////////////////////////////////////////////

class SingleOpFragment : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;
    auto op = make_operator<SingleOp>("op", make_condition<CountCondition>(10));

    add_operator(op);
  }
};

class OneTxFragment : public holoscan::Fragment {
 public:
  explicit OneTxFragment(int64_t count = 10) : count_(count) {}

  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<PingTxOp>("tx", make_condition<CountCondition>(count_));

    add_operator(tx);
  }

 private:
  int64_t count_ = 10;
};

class OneTwoOutputsTxFragment : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<PingTxTwoOutputsOp>("tx", make_condition<CountCondition>(10));

    add_operator(tx);
  }
};

class TwoTxsFragment : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx1 = make_operator<PingTxOp>("tx1", make_condition<CountCondition>(10));
    auto tx2 = make_operator<PingTxOp>("tx2", make_condition<CountCondition>(10));

    add_operator(tx1);
    add_operator(tx2);
  }
};

class TwoRxsFragment : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;
    auto rx1 = make_operator<PingRxOp>("rx1");
    auto rx2 = make_operator<PingRxOp>("rx2");

    add_operator(rx1);
    add_operator(rx2);
  }
};

class OneMxFragment : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;
    auto mx = make_operator<PingMxOp>("mx");

    add_operator(mx);
  }
};

class BroadcastFragment : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;
    auto broadcast = make_operator<BroadcastOp>("broadcast");

    add_operator(broadcast);
  }
};

class OneRxFragment : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;
    auto rx = make_operator<PingRxOp>("rx");

    add_operator(rx);
  }
};

class OneMultiReceiversParamRxFragment : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;
    auto rx = make_operator<PingMultiReceiversParamRxOp>("rx", make_condition<CountCondition>(10));

    add_operator(rx);
  }
};

class OneTwoInputsRxFragment : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;
    auto rx = make_operator<PingRxTwoInputsOp>("rx");

    add_operator(rx);
  }
};

class TwoOperatorFragment : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<PingTxOp>("tx", make_condition<CountCondition>(10));
    auto rx = make_operator<PingRxOp>("rx");

    add_flow(tx, rx);
  }
};

class OneTxTwoOutputsOneMxFragment : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<PingTxTwoOutputsOp>("tx", make_condition<CountCondition>(10));
    auto forward = make_operator<PingMxOp>("forward");

    add_flow(tx, forward, {{"out2", "in"}});
  }
};

class OneTxOneMxFragment : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<PingTxOp>("tx", make_condition<CountCondition>(10));
    auto forward = make_operator<PingMxOp>("forward");

    add_flow(tx, forward, {{"out", "in"}});
  }
};

///////////////////////////////////////////////////////////////////////////////
// Utility Applications
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief Test application that has two fragments with one operator each.
 *
 * The TwoParallelFragmentsApp class creates two SingleOpFragments named "fragment1" and
 * "fragment2". The fragments are created in parallel and are independent from each other. It does
 * not establish any connection between the two fragments.
 */
class TwoParallelFragmentsApp : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto fragment1 = make_fragment<SingleOpFragment>("fragment1");
    auto fragment2 = make_fragment<SingleOpFragment>("fragment2");

    add_fragment(fragment1);
    add_fragment(fragment2);
  }
};

/**
 * @brief Test application that has two fragments with multi inputs/outputs operator.
 *
 * The TwoMultiInputsOutputsFragmentsApp class creates two fragments with one operator each.
 * The first fragment has a PingTxTwoOutputsOp operator that has two outputs. The second fragment
 * has a PingRxTwoInputsOp operator that has two inputs. The two fragments are connected with
 * two flows, one for each output of the PingTxTwoOutputsOp operator.
 *
 * The graph configuration for this application is as follows:
 *
 * - Fragment (fragment1)
 *   - Operator (tx)
 *     - Output port: out1
 *     - Output port: out2
 * - Fragment (fragment2)
 *   - Operator (rx)
 *     - Input port: in1
 *     - Input port: in2
 *
 * The following connections are established:
 *
 * - fragment1.tx.out1 -> fragment2.rx.in1
 * - fragment1.tx.out2 -> fragment2.rx.in2
 */
class TwoMultiInputsOutputsFragmentsApp : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto fragment1 = make_fragment<OneTwoOutputsTxFragment>("fragment1");
    auto fragment2 = make_fragment<OneTwoInputsRxFragment>("fragment2");

    add_flow(fragment1, fragment2, {{"tx.out1", "rx.in1"}});
    add_flow(fragment1, fragment2, {{"tx.out2", "rx.in2"}});
  }
};

/**
 * @brief Test application that has two fragments with multiple operators in fragment1 connected
 *
 * The TwoMultipleSingleOutputOperatorsApp creates two fragments with mixed intra and inter-fragment
 * connections.
 *
 * The graph configuration for this application is as follows:
 *
 * - Fragment (fragment1)
 *   - Operator (tx)
 *     - Output port: out1
 *     - Output port: out2
 *   - Operator (forward)
 *     - Input port: in
 *     - Output port: out
 * - Fragment (fragment2)
 *   - Operator (rx)
 *     - Input port: in1
 *     - Input port: in2
 *
 * The following connections are established within fragment1:
 *
 * - tx.out2 -> forward.in
 *
 * The following connections are established between fragments:
 *
 * - fragment1.tx.out1 -> fragment2.rx.in1
 * - fragment1.forward.out -> fragment2.rx.in2
 */
class TwoMultipleSingleOutputOperatorsApp : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto fragment1 = make_fragment<OneTxTwoOutputsOneMxFragment>("fragment1");
    auto fragment2 = make_fragment<OneTwoInputsRxFragment>("fragment2");

    add_flow(fragment1, fragment2, {{"tx.out1", "rx.in1"}});
    add_flow(fragment1, fragment2, {{"forward.out", "rx.in2"}});
  }
};

/**
 * @brief Test application that has two fragments with multiple operators in fragment1 connected
 *
 * The TwoMultipleSingleOutputOperatorsBroadcastApp creates two fragments with mixed intra and
 * inter-fragment connections where an implicit broadcast is needed for the tx operator in
 * fragment1.
 *
 * The graph configuration for this application is as follows:
 *
 * - Fragment (fragment1)
 *   - Operator (tx)
 *     - Output port: out
 *   - Operator (forward)
 *     - Input port: in
 *     - Output port: out
 * - Fragment (fragment2)
 *   - Operator (rx)
 *     - Input port: in1
 *     - Input port: in2
 *
 * The following connections are established within fragment1:
 *
 * - tx.out -> forward.in
 *
 * The following connections are established between fragments:
 *
 * - fragment1.tx.out -> fragment2.rx.in1
 * - fragment1.forward.out -> fragment2.rx.in2
 */
class TwoMultipleSingleOutputOperatorsBroadcastApp : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto fragment1 = make_fragment<OneTxOneMxFragment>("fragment1");
    auto fragment2 = make_fragment<OneTwoInputsRxFragment>("fragment2");

    add_flow(fragment1, fragment2, {{"tx.out", "rx.in1"}});
    add_flow(fragment1, fragment2, {{"forward.out", "rx.in2"}});
  }
};

/**
 * @brief Test application that broadcasts an output to two ports on the same receiver.
 *
 * The OneTxBroadcastOneRxTwoInputs class creates two fragments.
 * The first fragment has a OneTxFragment operator that has a single output. The second fragment
 * has a OneTwoInputsRxFragment operator that has two inputs. The two fragments are connected with
 * two flows, one for each input of the OneTwoInputsRxFragment operator.
 *
 * The graph configuration for this application is as follows:
 *
 * - Fragment (fragment1)
 *   - Operator (tx)
 *     - Output port: out
 * - Fragment (fragment2)
 *   - Operator (rx)
 *     - Input port: in1
 *     - Input port: in2
 *
 * The following connections are established:
 *
 * - fragment1.tx.out -> fragment2.rx.in1
 * - fragment1.tx.out -> fragment2.rx.in2
 */
class OneTxBroadcastOneRxTwoInputs : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto fragment1 = make_fragment<OneTxFragment>("fragment1");
    auto fragment2 = make_fragment<OneTwoInputsRxFragment>("fragment2");

    add_flow(fragment1, fragment2, {{"tx.out", "rx.in1"}, {"tx.out", "rx.in2"}});
  }
};

/**
 * @brief Test application that has two fragments with multi inputs/outputs operator.
 *
 * The TwoMultiInputsOutputsFragmentsApp2 class creates two fragments with one operator each.
 * The first fragment has a PingTxOp operator that has one output. The second fragment
 * has a PingRxTwoInputsOp operator that has two inputs. The two fragments are connected with
 * two flows, one output of the PingTxOp operator to each input of the PingRxTwoInputsOp operator.
 *
 * The graph configuration for this application is as follows:
 *
 * - Fragment (fragment1)
 *   - Operator (tx)
 *     - Output port: out
 * - Fragment (fragment2)
 *   - Operator (rx)
 *     - Input port: in1
 *     - Input port: in2
 *
 * The following connections are established:
 *
 * - fragment1.tx.out -> fragment2.rx.in1
 * - fragment1.tx.out -> fragment2.rx.in2
 */
class TwoMultiInputsOutputsFragmentsApp2 : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto fragment1 = make_fragment<OneTxFragment>("fragment1");
    auto fragment2 = make_fragment<OneTwoInputsRxFragment>("fragment2");

    add_flow(fragment1, fragment2, {{"tx.out", "rx.in1"}, {"tx.out", "rx.in2"}});
  }
};

/**
 * @brief Test that UcxTransmitter/UcxReceiver works with MultiThreadedScheduler.
 *
 * UcxTransmitter/UcxReceiver doesn't work with GreedyScheduler with the following graph.
 *
 * - Fragment (fragment1)
 *   - Operator (op1)
 *     - Output port: out
 *   - Operator (op2)
 *     - Output port: out
 * - Fragment (fragment2)
 *   - Operator (op3)
 *     - Input ports
 *       - in1
 *       - in2
 *
 * With the following graph connections, due to how UcxTransmitter/UcxReceiver works,
 * UCX connections between op1 and op3 and between op2 and op3 are not established
 * (resulting in a deadlock).
 *
 * - op1.out -> op3.in1
 * - op2.out -> op3.in2
 */
class UCXConnectionApp : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto fragment1 = make_fragment<TwoTxsFragment>("fragment1");
    auto fragment2 = make_fragment<OneTwoInputsRxFragment>("fragment2");

    add_flow(fragment1, fragment2, {{"tx1", "rx.in1"}, {"tx2", "rx.in2"}});
  }
};

/**
 * @brief Test application similar UCXConnectionApp, but with one transmitter and two receivers.
 *
 * The UCXConnectionApp2 class creates two fragments with one operator each.
 * The first fragment has a PingTxTwoOutputsOp operator that has two outputs. The second fragment
 * has two PingRxOp operators that each have one input. The two fragments are connected with
 * two flows, one for each output of the PingTxTwoOutputsOp operator.
 *
 * The graph configuration for this application is as follows:
 *
 * - Fragment (fragment1)
 *   - Operator (tx)
 *     - Output port: out1
 *     - Output port: out2
 * - Fragment (fragment2)
 *   - Operator (rx1)
 *     - Input port: in
 *   - Operator (rx2)
 *     - Input port: in
 *
 * The following connections are established:
 *
 * - fragment1.tx.out1 -> fragment2.rx1.in
 * - fragment1.tx.out2 -> fragment2.rx2.in
 */
class UCXConnectionApp2 : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto fragment1 = make_fragment<OneTwoOutputsTxFragment>("fragment1");
    auto fragment2 = make_fragment<TwoRxsFragment>("fragment2");

    add_flow(fragment1, fragment2, {{"tx.out1", "rx1"}, {"tx.out2", "rx2"}});
  }
};

/**
 * @brief The UCXLinearPipelineApp is an application class designed to test the UcxTransmitter and
 * UcxReceiver in a simple linear pipeline context.
 *
 * The purpose of UCXLinearPipelineApp is to test the linear data flow within the holoscan
 * framework. It uses a UcxTransmitter and a UcxReceiver in a configuration where three fragments
 * are connected in a linear pipeline. The data flow from the transmitter in fragment1 through an
 * intermediary fragment2 to the receiver in fragment3.
 *
 * The graph configuration for this application is as follows:
 *
 * - Fragment (fragment1)
 *   - Operator (tx)
 *     - Output port: out
 * - Fragment (fragment2)
 *   - Operator (mx)
 *     - Input port: in
 *     - Output port: out
 * - Fragment (fragment3)
 *   - Operator (rx)
 *     - Input port: in
 *
 * The following connections are established:
 *
 * - fragment1.tx -> fragment2.mx
 * - fragment2.mx -> fragment3.rx
 *
 */
class UCXLinearPipelineApp : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    int64_t count = 20;
    auto fragment1 = make_fragment<OneTxFragment>("fragment1", count);
    auto fragment2 = make_fragment<OneMxFragment>("fragment2");
    auto fragment3 = make_fragment<OneRxFragment>("fragment3");

    add_flow(fragment1, fragment2, {{"tx", "mx"}});
    add_flow(fragment2, fragment3, {{"mx", "rx"}});
  }
};

/**
 * @brief An application class for testing broadcasting capabilities in a UCX context.
 *
 * The UCXBroadcastApp class is designed to simulate a broadcast scenario using UcxTransmitter and
 * UcxReceiver. It creates a data flow that starts with a single transmitter in one fragment
 * (fragment1), passes through a broadcaster in a second fragment (fragment2), and then proceeds to
 * two separate receivers in two different fragments (fragment3 and fragment4). This models a
 * typical scenario where data from a single source is required by multiple destinations.
 *
 * The graph configuration for this application is as follows:
 *
 * - Fragment (fragment1)
 *   - Operator (tx)
 *     - Output port: out
 * - Fragment (fragment2)
 *   - Operator (broadcast)
 *     - Input port: in
 *     - Output ports
 *       - out1
 *       - out2
 * - Fragment (fragment3)
 *   - Operator (rx)
 *     - Input port: in
 * - Fragment (fragment4)TwoMultiInputsOutputsFragmentsApp
 *   - Operator (rx)
 *     - Input port: in
 *
 * The following connections are established:
 *
 * - fragment1.tx -> fragment2.broadcast
 * - fragment2.broadcast.out1 -> fragment3.rx
 * - fragment2.broadcast.out2 -> fragment4.rx
 */
class UCXBroadcastApp : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto fragment1 = make_fragment<OneTxFragment>("fragment1");
    auto fragment2 = make_fragment<BroadcastFragment>("fragment2");
    auto fragment3 = make_fragment<OneRxFragment>("fragment3");
    auto fragment4 = make_fragment<OneRxFragment>("fragment4");

    add_flow(fragment1, fragment2, {{"tx", "broadcast"}});
    add_flow(fragment2, fragment3, {{"broadcast.out1", "rx"}});
    add_flow(fragment2, fragment4, {{"broadcast.out2", "rx"}});
  }
};

/**
 * @brief Application class for testing broadcasting in a multi-receiver context.
 *
 * The UCXBroadCastMultiReceiverApp class is a test case for UcxTransmitter and UcxReceiver when
 * used in a multi-receiver context. The topology is comprised of four fragments, where fragment1
 * serves as a broadcaster, fragment2 as a multi-receivers, fragment3 as a simple transmitter, and
 * fragment4 as a receiver. Unlike UCXConnectionApp, this class employs
 * broadcasting to a multitude of receivers, as opposed to point-to-point communication.
 *
 * The graph configuration for this application is as follows:
 *
 * - Fragment (fragment1)
 *   - Operator (tx)
 *     - Output port: out
 *   - Operator (rx)
 *     - Input port: in
 * - Fragment (fragment2)
 *   - Operator (rx)
 *     - Input port: receivers
 * - Fragment (fragment3)
 *   - Operator (tx)
 *     - Output port: out
 * - Fragment (fragment4)
 *   - Operator (rx)
 *     - Input port: in
 *
 * The following connections are established:
 *
 * - fragment1.tx -> fragment1.rx
 * - fragment1.tx -> fragment2.rx.receivers
 * - fragment1.tx -> fragment4.rx
 * - fragment3.tx -> fragment2.rx.receivers
 *
 */
class UCXBroadCastMultiReceiverApp : public holoscan::Application {
 public:
  // Inherit the constructor
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto fragment1 = make_fragment<TwoOperatorFragment>("fragment1");
    auto fragment2 = make_fragment<OneMultiReceiversParamRxFragment>("fragment2");
    auto fragment3 = make_fragment<OneTxFragment>("fragment3");
    auto fragment4 = make_fragment<OneRxFragment>("fragment4");

    add_flow(fragment1, fragment2, {{"tx", "rx.receivers"}});
    add_flow(fragment1, fragment4, {{"tx", "rx"}});
    add_flow(fragment3, fragment2, {{"tx", "rx.receivers"}});
  }
};

}  // namespace

}  // namespace holoscan

#endif /* SYSTEM_DISTRIBUTED_UTILITY_APPS_HPP */
