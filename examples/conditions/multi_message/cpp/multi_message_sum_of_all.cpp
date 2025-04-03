/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "./common_ops.hpp"
#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

class SumOfAllThrottledRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SumOfAllThrottledRxOp)

  SumOfAllThrottledRxOp() = default;

  void setup(OperatorSpec& spec) override {
    // Using size argument to explicitly set the receiver message queue size for each input.
    spec.input<std::shared_ptr<std::string>>("in1", IOSpec::IOSize(10));
    spec.input<std::shared_ptr<std::string>>("in2", IOSpec::IOSize(10));
    spec.input<std::shared_ptr<std::string>>("in3", IOSpec::IOSize(10));

    // Use kMultiMessageAvailableTimeout to consider all three ports together. In this
    // "SumOfAll" mode, it only matters that `min_sum` messages have arrived across all the ports
    // {"in1", "in2", "in3"}, but it does not matter which ports the messages arrived on. The
    // "execution_frequency" is set to 30ms, so the operator can run once 30 ms has elapsed even
    // if 20 messages have not arrived. Use ConditionType::kMultiMessageAvailable instead if the
    // timeout interval is not desired.
    ArgList multi_message_args{
        holoscan::Arg("execution_frequency", std::string{"30ms"}),
        holoscan::Arg("min_sum", static_cast<size_t>(20)),
        holoscan::Arg("sampling_mode",
                      MultiMessageAvailableTimeoutCondition::SamplingMode::kSumOfAll)};
    spec.multi_port_condition(
        ConditionType::kMultiMessageAvailableTimeout, {"in1", "in2", "in3"}, multi_message_args);
  }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    // We haven't set a specific number of messages per port, so loop over each
    // input until no more messages are in the queue.
    auto in_value1 = op_input.receive<std::shared_ptr<std::string>>("in1");
    std::vector<std::string> msgs_input1;
    while (in_value1) {
      msgs_input1.push_back(*in_value1.value());
      in_value1 = op_input.receive<std::shared_ptr<std::string>>("in1");
    }

    auto in_value2 = op_input.receive<std::shared_ptr<std::string>>("in2");
    std::vector<std::string> msgs_input2;
    while (in_value2) {
      msgs_input2.push_back(*in_value2.value());
      in_value2 = op_input.receive<std::shared_ptr<std::string>>("in2");
    }

    auto in_value3 = op_input.receive<std::shared_ptr<std::string>>("in3");
    std::vector<std::string> msgs_input3;
    while (in_value3) {
      msgs_input3.push_back(*in_value3.value());
      in_value3 = op_input.receive<std::shared_ptr<std::string>>("in3");
    }

    HOLOSCAN_LOG_INFO("messages received on in1: {}", msgs_input1);
    HOLOSCAN_LOG_INFO("messages received on in2: {}", msgs_input2);
    HOLOSCAN_LOG_INFO("messages received on in3: {}", msgs_input3);
  };
};

}  // namespace holoscan::ops

class MultiMessageThrottledApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    using namespace std::chrono_literals;

    auto tx1 = make_operator<ops::StringTxOp>(
        "tx1", make_condition<PeriodicCondition>("periodic-condition1", 4ms));
    tx1->set_message("tx1");

    auto tx2 = make_operator<ops::StringTxOp>(
        "tx2", make_condition<PeriodicCondition>("periodic-condition2", 8ms));
    tx2->set_message("tx2");

    auto tx3 = make_operator<ops::StringTxOp>(
        "tx3", make_condition<PeriodicCondition>("periodic-condition3", 16ms));
    tx3->set_message("tx3");

    auto multi_rx_timeout = make_operator<ops::SumOfAllThrottledRxOp>(
        "multi_rx_timeout", make_condition<CountCondition>("count-condition", 5));

    add_flow(tx1, multi_rx_timeout, {{"out", "in1"}});
    add_flow(tx2, multi_rx_timeout, {{"out", "in2"}});
    add_flow(tx3, multi_rx_timeout, {{"out", "in3"}});
  }
};

int main() {
  auto app = holoscan::make_application<MultiMessageThrottledApp>();

  // use the event-based scheduler so multiple operators can run simultaneously
  app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
      "event-based-scheduler", holoscan::Arg("worker_thread_number", static_cast<int64_t>(4))));

  app->run();

  return 0;
}
