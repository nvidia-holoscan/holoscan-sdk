/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

class PingMxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMxOp)

  PingMxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in").condition(holoscan::ConditionType::kNone);
    spec.output<int>("out");
    spec.param(multiplier_, "multiplier", "Multiplier", "Multiply the input by this value", 2);
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto value = op_input.receive<int>("in");

    int out_value = 1;
    if (value) {
      // Multiply the value by the multiplier parameter
      out_value = value.value() * multiplier_;
    }
    std::cout << "[" << name() << "] Middle message output value: " << out_value << std::endl;

    op_output.emit(out_value);
  };

 private:
  Parameter<int> multiplier_;
};

}  // namespace holoscan::ops

class MyPingApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto mx1 = make_operator<ops::PingMxOp>(
        "mx1", Arg("multiplier", 3), make_condition<CountCondition>(5));
    auto mx2 = make_operator<ops::PingMxOp>(
        "mx2", Arg("multiplier", 3), make_condition<CountCondition>(5));
    auto mx3 = make_operator<ops::PingMxOp>(
        "mx3", Arg("multiplier", 3), make_condition<CountCondition>(5));
    // Define the workflow:  tx -> mx -> rx
    add_flow(mx1, mx2);
    add_flow(mx2, mx3);
    add_flow(mx3, mx1);
  }
};

int main() {
  auto app = holoscan::make_application<MyPingApp>();
  app->run();

  return 0;
}
