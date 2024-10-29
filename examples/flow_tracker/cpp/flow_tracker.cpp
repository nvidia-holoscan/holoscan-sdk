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

class ValueData {
 public:
  ValueData() = default;
  explicit ValueData(int value) : data_(value) {
    HOLOSCAN_LOG_TRACE("ValueData::ValueData(): {}", data_);
  }
  ~ValueData() { HOLOSCAN_LOG_TRACE("ValueData::~ValueData(): {}", data_); }

  // Use default copy constructor
  ValueData(const ValueData&) = default;

  // Use default move constructor
  ValueData(ValueData&&) noexcept = default;

  // Use default copy assignment operator
  ValueData& operator=(const ValueData&) = default;

  // Use default move assignment operator
  ValueData& operator=(ValueData&&) noexcept = default;

  void data(int value) { data_ = value; }

  [[nodiscard]] int data() const { return data_; }

 private:
  int data_;
};

namespace holoscan::ops {

class PingTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxOp)

  PingTxOp() = default;

  void setup(OperatorSpec& spec) override { spec.output<ValueData>("out"); }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto value1 = ValueData(index_++);
    op_output.emit(value1, "out");
  };

 private:
  int index_ = 1;
};

class PingMxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingMxOp)

  PingMxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<ValueData>("in");
    spec.output<ValueData>("out");
    spec.param(multiplier_, "multiplier", "Multiplier", "Multiply the input by this value", 2);
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto value = op_input.receive<ValueData>("in").value();

    HOLOSCAN_LOG_INFO("Middle message received (count: {})", count_++);

    HOLOSCAN_LOG_INFO("Middle message value1: {}", value.data());

    // Multiply the values by the multiplier parameter
    value.data(value.data() * multiplier_);

    op_output.emit(value, "out");
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
    // // Since Holoscan SDK v2.3, users can define a multi-receiver input port using 'spec.input()'
    // // with 'IOSpec::kAnySize'.
    // // The old way is to use 'spec.param()' with 'Parameter<std::vector<IOSpec*>> receivers_;'.
    // spec.param(receivers_, "receivers", "Input Receivers", "List of input receivers.", {});
    spec.input<std::vector<ValueData>>("receivers", IOSpec::kAnySize);
  }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto value_vector = op_input.receive<std::vector<ValueData>>("receivers").value();

    HOLOSCAN_LOG_INFO("Rx message received (count: {}, size: {})", count_++, value_vector.size());

    for (int i = 0; i < value_vector.size(); i++) {
      HOLOSCAN_LOG_INFO("Rx message value{}: {}", i, value_vector[i].data());
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
    auto root1 = make_operator<ops::PingTxOp>("root1", make_condition<CountCondition>(10));
    auto root2 = make_operator<ops::PingTxOp>("root2", make_condition<CountCondition>(10));
    auto middle1 = make_operator<ops::PingMxOp>("middle1", Arg("multiplier", 2));
    auto middle2 = make_operator<ops::PingMxOp>("middle2", Arg("multiplier", 3));
    auto leaf1 = make_operator<ops::PingRxOp>("leaf1");
    auto leaf2 = make_operator<ops::PingRxOp>("leaf2");

    // Create the workflow graph
    add_flow(root1, middle1);
    add_flow(middle1, leaf1, {{"out", "receivers"}});
    add_flow(middle1, leaf2, {{"out", "receivers"}});

    add_flow(root2, middle2);
    add_flow(middle2, leaf2, {{"out", "receivers"}});
  }
};

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
  auto app = holoscan::make_application<MyPingApp>();
  // Skip 2 messages at the start and 3 messages at the end
  auto& tracker = app->track(2, 3, 0);
  tracker.enable_logging();
  app->run();

  // Print all the metrics to the standard output
  tracker.print();

  // The code below shows how individual metrics can be accessed via the API
  // Get all the path strings
  auto path_strings = tracker.get_path_strings();
  // Print maximum end-to-end latency for every path
  HOLOSCAN_LOG_INFO("Maximum Latencies:");
  for (auto& path_string : path_strings) {
    HOLOSCAN_LOG_INFO("Path: {} -- {} ms",
                      path_string,
                      tracker.get_metric(path_string, holoscan::DataFlowMetric::kMaxE2ELatency));
  }

  auto root_message_number = tracker.get_metric(holoscan::DataFlowMetric::kNumSrcMessages);

  for (auto& root_message_number_pair : root_message_number) {
    HOLOSCAN_LOG_INFO("Root-transmitter: {}, number of messages: {}",
                      root_message_number_pair.first,
                      root_message_number_pair.second);
  }
  return 0;
}
