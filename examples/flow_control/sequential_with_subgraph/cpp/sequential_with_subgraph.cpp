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

#include <string>

#include <holoscan/holoscan.hpp>

class SimpleOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SimpleOp)

  SimpleOp() = default;

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("I am here - {}", name());
  }
};

class SequentialSubgraph : public holoscan::Subgraph {
 public:
  SequentialSubgraph(holoscan::Fragment* frag, const std::string& name) : Subgraph(frag, name) {}

  void compose() override {
    using namespace holoscan;

    // Define the operators inside the subgraph
    auto node2 = make_operator<SimpleOp>("node2");
    auto node3 = make_operator<SimpleOp>("node3");

    // Define the sequential workflow inside the subgraph
    add_flow(node2, node3);

    // Expose node2's input execution port as the subgraph's input execution interface port
    add_input_exec_interface_port("exec_in", node2);

    // Expose node3's output execution port as the subgraph's output execution interface port
    add_output_exec_interface_port("exec_out", node3);
  }
};

class SequentialWithSubgraphApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Define operators outside the subgraph
    auto node1 = make_operator<SimpleOp>("node1");
    auto node4 = make_operator<SimpleOp>("node4");

    // Create the subgraph containing node2 and node3
    auto sequential_sg = make_subgraph<SequentialSubgraph>("sequential_sg");

    // Define the sequential workflow using control flow
    // start_op() -> node1 -> sequential_sg (node2 -> node3) -> node4
    add_flow(start_op(), node1);
    add_flow(node1, sequential_sg);  // Auto-resolves to exec_in
    add_flow(sequential_sg, node4);  // Auto-resolves to exec_out
  }
};

int main() {
  auto app = holoscan::make_application<SequentialWithSubgraphApp>();
  app->run();
  return 0;
}

// Expected output:
//
// I am here - node1
// I am here - node2
// I am here - node3
// I am here - node4
