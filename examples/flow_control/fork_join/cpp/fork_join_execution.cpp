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

#include <iostream>

#include <holoscan/holoscan.hpp>

class SimpleOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SimpleOp)

  SimpleOp() = default;

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("executing operator: {}", name());
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
};

class ForkJoinExecutionApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Define the operators
    auto node1 = make_operator<SimpleOp>("node1");
    auto node2 = make_operator<SimpleOp>("node2");
    auto node3 = make_operator<SimpleOp>("node3");
    auto node4 = make_operator<SimpleOp>("node4");
    auto node5 = make_operator<SimpleOp>("node5");
    auto node6 = make_operator<SimpleOp>("node6");
    auto node7 = make_operator<SimpleOp>("node7");
    auto node8 = make_operator<SimpleOp>("node8");

    // Define the fork-join workflow
    //                <|start|>
    //                    |
    //                  node1
    //         /    /     |     \     \
    //      node2 node3 node4 node5 node6
    //        \     \     |     /     /
    //         \     \    |    /     /
    //          \     \   |   /     /
    //           \     \  |  /     /
    //                  node7
    //                    |
    //                  node8
    add_flow(start_op(), node1);
    add_flow(node1, node2);
    add_flow(node1, node3);
    add_flow(node1, node4);
    add_flow(node1, node5);
    add_flow(node1, node6);
    add_flow(node2, node7);
    add_flow(node3, node7);
    add_flow(node4, node7);
    add_flow(node5, node7);
    add_flow(node6, node7);
    add_flow(node7, node8);
  }
};

int main() {
  using namespace holoscan;

  auto app = make_application<ForkJoinExecutionApp>();
  auto scheduler = app->make_scheduler<EventBasedScheduler>(
      "myscheduler", Arg("worker_thread_number", 5L), Arg("stop_on_deadlock", true));
  app->scheduler(scheduler);
  app->run();
  return 0;
}

// Expected output:
// (node2 to node6 are executed in parallel so the output is not deterministic)
//
// executing operator: node1
// executing operator: node2
// executing operator: node3
// executing operator: node4
// executing operator: node5
// executing operator: node6
// executing operator: node7
// executing operator: node8
