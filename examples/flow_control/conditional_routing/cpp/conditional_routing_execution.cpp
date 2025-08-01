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
    op_output.emit(value_);
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

class ConditionalRoutingApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Define the operators
    auto node1 = make_operator<PingTx>("node1", make_condition<CountCondition>(2));
    auto node2 = make_operator<PingRx>("node2");
    auto node3 = make_operator<PingRx>("node3");

    // Define the conditional routing workflow
    //
    // Node Graph:
    //
    //             node1 (launch twice, emitting data to 'output' port)
    //            /     \
    //          node2 node3

    add_flow(node1, node2);
    add_flow(node1, node3);

    // // If you want to add all the next flows, you can use the following code:
    // set_dynamic_flows(
    //     node1, [](const std::shared_ptr<Operator>& op) { op->add_dynamic_flow(op->next_flows());
    //     });

    // // This is another way to add dynamic flows based on the next operator name
    // set_dynamic_flows(node1, [](const std::shared_ptr<Operator>& op) {
    //   auto simple_op = std::static_pointer_cast<PingTx>(op);
    //   static const auto& node2_flow = op->find_flow_info(
    //       [](const auto& flow) { return flow->next_operator->name() == "node2"; });
    //   static const auto& node3_flow = op->find_flow_info(
    //       [](const auto& flow) { return flow->next_operator->name() == "node3"; });

    //   if (simple_op->get_value() % 2 == 1) {
    //     simple_op->add_dynamic_flow(node2_flow);
    //   } else {
    //     simple_op->add_dynamic_flow(node3_flow);
    //   }
    // });

    set_dynamic_flows(node1, [node2, node3](const std::shared_ptr<Operator>& op) {
      auto simple_op = std::static_pointer_cast<PingTx>(op);
      if (simple_op->get_value() % 2 == 1) {
        simple_op->add_dynamic_flow(node2);
      } else {
        simple_op->add_dynamic_flow(node3);
      }
    });
  }
};

int main() {
  auto app = holoscan::make_application<ConditionalRoutingApp>();
  app->run();
  return 0;
}

// Expected output:
//
// Sending value 1 - node1
// Received value 1 - node2
// Sending value 2 - node1
// Received value 2 - node3
