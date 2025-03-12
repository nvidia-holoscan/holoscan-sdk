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
    std::cout << "I am here - " << name() << std::endl;
  }
};

class SequentialExecutionApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Define the operators
    auto node1 = make_operator<SimpleOp>("node1");
    auto node2 = make_operator<SimpleOp>("node2");
    auto node3 = make_operator<SimpleOp>("node3");

    // Define the sequential workflow
    add_flow(start_op(), node1);
    add_flow(node1, node2);
    add_flow(node2, node3);
  }
};

int main() {
  auto app = holoscan::make_application<SequentialExecutionApp>();
  app->run();
  return 0;
}

// Expected output:
//
// I am here - node1
// I am here - node2
// I am here - node3
