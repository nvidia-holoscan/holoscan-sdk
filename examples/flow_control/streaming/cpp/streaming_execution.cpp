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

class PingTxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxOp)

  PingTxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.output<int>("output"); }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    value_++;
    std::cout << name() << " - "
              << "Sending value " << value_ << std::endl;
    op_output.emit(value_, "output");
  }

 private:
  int value_ = 0;
};

class PingMxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMxOp)

  PingMxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<int>("input");
    spec.output<int>("output");
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto value = op_input.receive<int>("input").value();
    std::cout << name() << " - "
              << "Received value " << value << std::endl;
    op_output.emit(value, "output");
  }
};

class PingRxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRxOp)

  PingRxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.input<int>("input"); }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto value = op_input.receive<int>("input").value();
    std::cout << name() << " - "
              << "Received value " << value << std::endl;
  }
};

class StreamExecutionApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    using namespace std::chrono_literals;

    // Define the operators

    // You can uncomment the following lines to add a periodic condition to the node1 operator
    auto node1 = make_operator<PingTxOp>(
        "node1" /* , make_condition<PeriodicCondition>("periodic-condition", 0.2s) */);
    auto node2 = make_operator<PingMxOp>("node2");
    auto node3 = make_operator<PingMxOp>("node3");
    auto node4 = make_operator<PingRxOp>("node4");

    // Define the streaming workflow
    //
    // Node Graph:
    //                   (cycle)
    //                   -------
    //                   \     /
    //     <|start|>  ->  node1  ->  node2  ->  node3  ->  node4
    //             (triggered 5 times)
    add_flow(start_op(), node1);
    add_flow(node1, node1);
    // The following line requires an explicit port name because the
    // `node1` operator has two output ports: `output` and `__output_exec__` (implicitly added
    //  by the framework when add_dynamic_flow is used).
    // When connecting a specific output port to the input port of the next operator,
    // you must explicitly specify the port name.
    add_flow(node1, node2, {{"output", "input"}});
    add_flow(node2, node3);
    add_flow(node3, node4);

    set_dynamic_flows(node1, [node2](const std::shared_ptr<Operator>& op) {
      static int iteration = 0;
      ++iteration;
      std::cout << "#iteration: " << iteration << std::endl;

      if (iteration <= 5) {
        op->add_dynamic_flow("output", node2);
        // Signal self to trigger the next iteration.
        // Since the `node1` operator has two outgoing flows, we need to specify the output port
        // name explicitly to the `add_dynamic_flow()` function.
        //
        // `Operator::kOutputExecPortName` is the default execution output port name for operators
        // which would be used by Holoscan internally to connect the output of the operator to the
        // input of the next operator.
        //
        // There is `Operator::kInputExecPortName` which is similar to `kOutputExecPortName` but
        // for the input port.
        //
        // Here we are using `kOutputExecPortName` to signal the operator to trigger itself in
        // the next iteration.
        op->add_dynamic_flow(Operator::kOutputExecPortName, op);
      } else {
        iteration = 0;
      }
    });
  }
};

int main() {
  using namespace holoscan;

  auto app = holoscan::make_application<StreamExecutionApp>();
  app->run();
  return 0;
}

// Expected output:
//
// node1 - Sending value 1
// #iteration: 1
// node2 - Received value 1
// node3 - Received value 1
// node4 - Received value 1
// node1 - Sending value 2
// #iteration: 2
// node2 - Received value 2
// node3 - Received value 2
// node4 - Received value 2
// node1 - Sending value 3
// #iteration: 3
// node2 - Received value 3
// node3 - Received value 3
// node4 - Received value 3
// node1 - Sending value 4
// #iteration: 4
// node2 - Received value 4
// node3 - Received value 4
// node4 - Received value 4
// node1 - Sending value 5
// #iteration: 5
// node2 - Received value 5
// node3 - Received value 5
// node4 - Received value 5
// node1 - Sending value 6
// #iteration: 6
