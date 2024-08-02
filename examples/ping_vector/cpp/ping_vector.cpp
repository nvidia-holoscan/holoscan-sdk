/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <vector>
#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

class PingTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxOp)

  PingTxOp() = default;

  void setup(OperatorSpec& spec) override { spec.output<std::vector<int>>("out"); }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    auto value1 = index_++;

    std::vector<int> output;
    for (int i = 0; i < 5; i++) { output.push_back(value1++); }

    op_output.emit(output, "out");
  };
  int index_ = 1;
};

class PingMxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMxOp)

  PingMxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<std::vector<int>>("in");
    spec.output<std::vector<int>>("out1");
    spec.output<std::vector<int>>("out2");
    spec.output<std::vector<int>>("out3");
    spec.param(multiplier_, "multiplier", "Multiplier", "Multiply the input by this value", 2);
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    auto values1 = op_input.receive<std::vector<int>>("in").value();

    HOLOSCAN_LOG_INFO("Middle message received (count: {})", count_++);

    std::vector<int> values2;
    std::vector<int> values3;
    for (int i = 0; i < values1.size(); i++) {
      HOLOSCAN_LOG_INFO("Middle message value: {}", values1[i]);
      values2.push_back(values1[i] * multiplier_);
      values3.push_back(values1[i] * multiplier_ * multiplier_);
    }

    op_output.emit(values1, "out1");
    op_output.emit(values2, "out2");
    op_output.emit(values3, "out3");
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
    spec.input<std::vector<int>>("in");
    spec.input<std::vector<int>>("dup_in");
    // // Since Holoscan SDK v2.3, users can define a multi-receiver input port using 'spec.input()'
    // // with 'IOSpec::kAnySize'.
    // // The old way is to use 'spec.param()' with 'Parameter<std::vector<IOSpec*>> receivers_;'.
    // spec.param(receivers_, "receivers", "Input Receivers", "List of input receivers.", {});
    spec.input<std::vector<std::vector<int>>>("receivers", IOSpec::kAnySize);
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    auto receiver_vector = op_input.receive<std::vector<std::vector<int>>>("receivers").value();
    auto input_vector = op_input.receive<std::vector<int>>("in").value();
    auto dup_input_vector = op_input.receive<std::vector<int>>("dup_in").value();

    HOLOSCAN_LOG_INFO(
        "Rx message received (count: {}, input vector size: {}, duplicated input vector size: {}, "
        "receiver size: {})",
        count_++,
        input_vector.size(),
        dup_input_vector.size(),
        receiver_vector.size());

    for (int i = 0; i < input_vector.size(); i++) {
      HOLOSCAN_LOG_INFO("Rx message input value[{}]: {}", i, input_vector[i]);
    }

    for (int i = 0; i < dup_input_vector.size(); i++) {
      HOLOSCAN_LOG_INFO("Rx message duplicated input value[{}]: {}", i, dup_input_vector[i]);
    }

    for (int i = 0; i < receiver_vector.size(); i++) {
      for (int j = 0; j < receiver_vector[i].size(); j++) {
        HOLOSCAN_LOG_INFO("Rx message receiver value[{}][{}]: {}", i, j, receiver_vector[i][j]);
      }
    }
  };

 private:
  // // Since Holoscan SDK v2.3, the following line is no longer needed.
  // Parameter<std::vector<IOSpec*>> receivers_;
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
    add_flow(mx, rx, {{"out1", "in"}, {"out1", "dup_in"}});
    add_flow(mx, rx, {{"out2", "receivers"}, {"out3", "receivers"}});
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<MyPingApp>();
  app->run();

  return 0;
}
