/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/holoscan.hpp"

class ValueData {
 public:
  ValueData() = default;
  explicit ValueData(int value) : data_(value) {
    HOLOSCAN_LOG_TRACE("ValueData::ValueData(): {}", data_);
  }
  ~ValueData() { HOLOSCAN_LOG_TRACE("ValueData::~ValueData(): {}", data_); }

  void data(int value) { data_ = value; }

  int data() const { return data_; }

 private:
  int data_;
};

namespace holoscan::ops {

class PingTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxOp)

  PingTxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.output<ValueData>("out1");
    spec.output<ValueData>("out2");
  }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    auto value1 = std::make_shared<ValueData>(index_++);
    op_output.emit(value1, "out1");

    auto value2 = std::make_shared<ValueData>(index_++);
    op_output.emit(value2, "out2");
  };
  int index_ = 1;
};

class PingMxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMxOp)

  PingMxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<ValueData>("in1");
    spec.input<ValueData>("in2");
    spec.output<ValueData>("out1");
    spec.output<ValueData>("out2");
    spec.param(multiplier_, "multiplier", "Multiplier", "Multiply the input by this value", 2);
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    auto value1 = op_input.receive<ValueData>("in1");
    auto value2 = op_input.receive<ValueData>("in2");

    HOLOSCAN_LOG_INFO("Middle message received (count: {})", count_++);

    HOLOSCAN_LOG_INFO("Middle message value1: {}", value1->data());
    HOLOSCAN_LOG_INFO("Middle message value2: {}", value2->data());

    // Multiply the values by the multiplier parameter
    value1->data(value1->data() * multiplier_);
    value2->data(value2->data() * multiplier_);

    op_output.emit(value1, "out1");
    op_output.emit(value2, "out2");
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
    auto value_vector = op_input.receive<std::vector<ValueData>>("receivers");

    HOLOSCAN_LOG_INFO("Rx message received (count: {}, size: {})", count_++, value_vector.size());

    HOLOSCAN_LOG_INFO("Rx message value1: {}", value_vector[0]->data());
    HOLOSCAN_LOG_INFO("Rx message value2: {}", value_vector[1]->data());
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
    add_flow(tx, mx, {{"out1", "in1"}, {"out2", "in2"}});
    add_flow(mx, rx, {{"out1", "receivers"}, {"out2", "receivers"}});
  }
};

int main(int argc, char** argv) {
  holoscan::load_env_log_level();
  auto app = holoscan::make_application<MyPingApp>();
  app->run();

  return 0;
}
