/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/holoscan.hpp>
#include <iostream>

namespace holoscan::ops {

class HelloWorldOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(HelloWorldOp)

  HelloWorldOp() = default;

  void setup(OperatorSpec& spec) override {}

  void compute([[maybe_unused]] InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    std::cout << '\n';
    std::cout << "Hello World!\n";
    std::cout << '\n';
  }
};

}  // namespace holoscan::ops

class HelloWorldApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Define the operators
    auto hello = make_operator<ops::HelloWorldOp>("hello", make_condition<CountCondition>(1));

    // Define the one-operator workflow
    add_operator(hello);
  }
};

int main() {
  auto app = holoscan::make_application<HelloWorldApp>();
  app->run();

  return 0;
}
