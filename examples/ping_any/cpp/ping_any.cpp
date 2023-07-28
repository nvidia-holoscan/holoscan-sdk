/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cassert>
#include <memory>
#include <vector>
#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

class PingTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxOp)

  PingTxOp() = default;

  void setup(OperatorSpec& spec) override { spec.output<int>("out"); }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    auto value = index_++;
    op_output.emit(value, "out");
  };
  int index_ = 1;
};

class PingMxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMxOp)

  PingMxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<std::any>("in");
    spec.output<int>("out");
    spec.param(multiplier_, "multiplier", "Multiplier", "Multiply the input by this value", 2);
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    auto value = op_input.receive<std::any>("in").value();

    // Received value must be an int because MX's `in` is connected TX's `out`
    assert(typeid(int) == value.type());

    HOLOSCAN_LOG_INFO("Middle message received (count: {})", count_++);

    HOLOSCAN_LOG_INFO("Middle message value: {}", std::any_cast<int>(value));

    int out_value = std::any_cast<int>(value) * multiplier_;

    op_output.emit(out_value, "out");
  };

 private:
  int count_ = 1;
  Parameter<int> multiplier_;
};

class PingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRxOp)

  PingRxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(receivers_, "receivers", "Input Receivers", "List of input receivers.", {});
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    auto value_vector = op_input.receive<std::vector<int>>("receivers").value();

    HOLOSCAN_LOG_INFO("Rx message received (count: {}, size: {})", count_++, value_vector.size());

    HOLOSCAN_LOG_INFO("Rx message value: {}", value_vector[0]);
  };

 private:
  Parameter<std::vector<IOSpec*>> receivers_;
  int count_ = 1;
};

}  // namespace holoscan::ops

class MyPingApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Define the tx, mx, rx operators, allowing the tx operator to execute 10 times
    auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(10));
    auto mx = make_operator<ops::PingMxOp>("mx", Arg("multiplier", 3));
    auto rx = make_operator<ops::PingRxOp>("rx");

    // Define the workflow
    add_flow(tx, mx, {{"out", "in"}});
    add_flow(mx, rx, {{"out", "receivers"}});
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<MyPingApp>();
  app->run();

  return 0;
}
