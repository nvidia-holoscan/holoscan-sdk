/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>

#include <holoscan/holoscan.hpp>

class SimpleOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SimpleOp)

  SimpleOp() = default;

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    std::cout << "I am here - " << name() << std::endl;
  }
};

class PingTx : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTx)

  PingTx() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.output<int>("output"); }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    value_++;
    std::cout << "Sending value " << value_ << " - " << name() << std::endl;

    op_output.emit(value_, "output");
  }

  int get_value() const { return value_; }

 private:
  int value_ = 0;
};

class PingRx : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRx)

  PingRx() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.input<int>("input"); }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto msg = op_input.receive<int>();
    if (msg) {
      std::cout << "Received value " << *msg << " - " << name() << std::endl;
    }
  }
};

class MixedFlowApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Define the operators
    auto node1 = make_operator<PingTx>("node1", make_condition<CountCondition>(2));
    auto node2 = make_operator<PingRx>("node2");
    auto alarm = make_operator<SimpleOp>("alarm");
    auto node3 = make_operator<PingRx>("node3");

    // Define the mixed flows workflow
    //
    // Node Graph:
    //
    //             node1 (from 'output' port)
    //         /     |     \
    //       node2 alarm  node3

    // The connections from `node1` to either `node2` or `node3` require an explicit port name
    // because the `node1` operator has two output ports: `output` and `__output_exec__`
    // (implicitly added by the framework when dynamic flows are used).

    // When connecting a specific output port to the input port of the next operator,
    // you must explicitly specify the port name.
    add_flow(node1, node2, {{"output", "input"}});
    add_flow(node1, alarm);
    add_flow(node1, node3, {{"output", "input"}});

    set_dynamic_flows(node1, [node2, node3, alarm](const std::shared_ptr<Operator>& op) {
      auto simple_op = std::static_pointer_cast<PingTx>(op);
      if (simple_op->get_value() % 2 == 1) {
        simple_op->add_dynamic_flow("output", node2, "input");
      } else {
        simple_op->add_dynamic_flow("output", node3, "input");
      }
      // Since the `node1` operator has three outgoing flows, we need to specify the output
      // port name explicitly to the `add_dynamic_flow()` function.
      //
      // `Operator::kOutputExecPortName` is the default execution output port name for
      // operators which would be used by Holoscan internally to connect the output of the
      // operator to the input of the next operator.
      //
      // There is `Operator::kInputExecPortName` which is similar to `kOutputExecPortName` but
      // for the input port.
      //
      // Here we are using `kOutputExecPortName` to signal self to trigger the alarm
      // operator.
      simple_op->add_dynamic_flow(Operator::kOutputExecPortName, alarm);
    });
  }
};

int main() {
  auto app = holoscan::make_application<MixedFlowApp>();
  app->run();
  return 0;
}

// Expected output:
// (node1 alternates between sending to node2 and node3, while alarm is always executed)
//
// Sending value 1 - node1
// I am here - alarm
// Received value 1 - node2
// Sending value 2 - node1
// Received value 2 - node3
// I am here - alarm
