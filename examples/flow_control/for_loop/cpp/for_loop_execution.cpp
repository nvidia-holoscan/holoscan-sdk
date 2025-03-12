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
    int index = metadata()->get<int>("index", -1);
    std::cout << "I am here - " << name() << " (index: " << index << ")" << std::endl;
  }

  int get_value() const { return value_; }

 private:
  int value_ = 0;
};

class ForLoopExecutionApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Define the operators
    auto node1 = make_operator<SimpleOp>("node1");
    auto node2 = make_operator<SimpleOp>("node2");
    auto node3 = make_operator<SimpleOp>("node3");
    auto node4 = make_operator<SimpleOp>("node4");

    // Define the for-loop workflow
    //
    // Node Graph:
    //
    //     <|start|>
    //         |
    //       node1
    //       / ^  \
    //   node2 |  node4
    //     |   |
    //   node3 |
    //     |   |(loop)
    //     \__/
    add_flow(start_op(), node1);
    add_flow(node1, node2);
    add_flow(node2, node3);
    add_flow(node3, node1);
    add_flow(node1, node4);

    // As node1 operates in a loop, the following metadata policy will be (automatically)
    // internally configured to permit metadata updates during the application's execution:
    //   node1->metadata_policy(holoscan::MetadataPolicy::kUpdate);
    set_dynamic_flows(node1, [node2, node4](const std::shared_ptr<Operator>& op) {
      static int index = 0;

      op->metadata()->set("index", index);

      if (index < 3) {
        op->add_dynamic_flow(node2);
        ++index;
      } else {
        index = 0;
        op->add_dynamic_flow(node4);
      }
    });
  }
};

int main() {
  auto app = holoscan::make_application<ForLoopExecutionApp>();
  app->run();
  return 0;
}

// Expected output:
// (node1 will loop 3 times through node2->node3 before going to node4)
//
// I am here - node1 (index: -1)
// I am here - node2 (index: 0)
// I am here - node3 (index: 0)
// I am here - node1 (index: 0)
// I am here - node2 (index: 1)
// I am here - node3 (index: 1)
// I am here - node1 (index: 1)
// I am here - node2 (index: 2)
// I am here - node3 (index: 2)
// I am here - node1 (index: 2)
// I am here - node4 (index: 3)
