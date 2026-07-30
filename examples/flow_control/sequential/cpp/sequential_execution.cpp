/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
    std::cout << "I am here - " << name() << '\n';
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
