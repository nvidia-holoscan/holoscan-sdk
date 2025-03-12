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
    value_++;
    std::cout << "I am here - " << name() << std::endl;
  }

  int get_value() const { return value_; }

 private:
  int value_ = 0;
};

class ConditionalExecutionApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Define the operators
    auto node1 = make_operator<SimpleOp>("node1", make_condition<CountCondition>(2));
    auto node2 = make_operator<SimpleOp>("node2");
    auto node3 = make_operator<SimpleOp>("node3");
    auto node4 = make_operator<SimpleOp>("node4");
    auto node5 = make_operator<SimpleOp>("node5");

    // Define the conditional workflow
    //
    // Node Graph:
    //
    //       node1 (launch twice)
    //       /   \
    //   node2   node4
    //     |       |
    //   node3   node5

    add_flow(node1, node2);
    add_flow(node2, node3);
    add_flow(node1, node4);
    add_flow(node4, node5);

    // // If you want to add all the next flows, you can use the following code:
    // set_dynamic_flows(
    //     node1, [](const std::shared_ptr<Operator>& op) { op->add_dynamic_flow(op->next_flows());
    //     });

    set_dynamic_flows(node1, [node2, node4](const std::shared_ptr<Operator>& op) {
      auto simple_op = std::static_pointer_cast<SimpleOp>(op);
      if (simple_op->get_value() % 2 == 1) {
        simple_op->add_dynamic_flow(node2);
      } else {
        simple_op->add_dynamic_flow(node4);
      }
    });

    //// This is another way to add dynamic flows based on the next operator name
    // set_dynamic_flows(node1, [](const std::shared_ptr<Operator>& op) {
    //   auto simple_op = std::static_pointer_cast<SimpleOp>(op);
    //   static const auto& node2_flow = op->find_flow_info(
    //       [](const auto& flow) { return flow->next_operator->name() == "node2"; });
    //   static const auto& node4_flow = op->find_flow_info(
    //       [](const auto& flow) { return flow->next_operator->name() == "node4"; });
    //   //static const auto& all_next_flows = op->find_all_flow_info(
    //   //    [](const auto& flow) { return true; });

    //   //std::cout << "All next flows: ";
    //   //for (const auto& flow : all_next_flows) {
    //   //  std::cout << flow->next_operator->name() << " ";
    //   //}
    //   //std::cout << std::endl;

    //   if (simple_op->get_value() % 2 == 1) {
    //     simple_op->add_dynamic_flow(node2_flow);
    //   } else {
    //     simple_op->add_dynamic_flow(node4_flow);
    //   }
    // });
  }
};

int main() {
  auto app = holoscan::make_application<ConditionalExecutionApp>();
  app->run();
  return 0;
}

// Expected output:
//
// I am here - node1
// I am here - node2
// I am here - node3
// I am here - node1
// I am here - node4
// I am here - node5
