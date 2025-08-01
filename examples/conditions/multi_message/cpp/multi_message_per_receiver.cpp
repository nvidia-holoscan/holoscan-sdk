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

class PerReceiverRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PerReceiverRxOp)

  PerReceiverRxOp() = default;

  void setup(OperatorSpec& spec) override {
    // Using size argument to explicitly set the receiver message queue size for each input.
    spec.input<std::shared_ptr<std::string>>("in1");
    spec.input<std::shared_ptr<std::string>>("in2", IOSpec::IOSize(2));
    spec.input<std::shared_ptr<std::string>>("in3");

    // Configure a MultiMessageAvailableCondition in "PerReceiver" mode so the operator will run
    // only when 1, 2 and 1 messages have arrived on ports "in1", "in2" and "in3", respectively.
    // This per-receiver mode is equivalent to putting a MessageAvailable condition on each input
    // individually.
    ArgList multi_message_args{
        holoscan::Arg("min_sizes", std::vector<size_t>{1, 2, 1}),
        holoscan::Arg("sampling_mode", MultiMessageAvailableCondition::SamplingMode::kPerReceiver)};
    spec.multi_port_condition(
        ConditionType::kMultiMessageAvailable, {"in1", "in2", "in3"}, multi_message_args);
  }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    std::string msg1{};
    std::string msg2{};
    std::string msg3{};
    std::string msg4{};
    auto in_value1 = op_input.receive<std::shared_ptr<std::string>>("in1");
    if (in_value1) {
      msg1 = *in_value1.value();
    }

    // receive twice from in2 because there are 2 messages due to size = 2
    auto in_value2 = op_input.receive<std::shared_ptr<std::string>>("in2");
    if (in_value2) {
      msg2 = *in_value2.value();
    }
    auto in_value3 = op_input.receive<std::shared_ptr<std::string>>("in2");
    if (in_value3) {
      msg3 = *in_value3.value();
    }

    auto in_value4 = op_input.receive<std::shared_ptr<std::string>>("in3");
    if (in_value4) {
      msg4 = *in_value4.value();
    }

    HOLOSCAN_LOG_INFO("message received on in1: {}", msg1);
    HOLOSCAN_LOG_INFO("first message received on in2: {}", msg2);
    HOLOSCAN_LOG_INFO("second message received on in2: {}", msg3);
    HOLOSCAN_LOG_INFO("message received on in3: {}", msg4);
  };
};

}  // namespace holoscan::ops

class MultiMessageApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    using namespace std::chrono_literals;

    auto tx1 = make_operator<ops::StringTxOp>(
        "tx1", make_condition<PeriodicCondition>("periodic-condition1", 0.05s));
    tx1->set_message("Hello from tx1");

    auto tx2 = make_operator<ops::StringTxOp>(
        "tx2", make_condition<PeriodicCondition>("periodic-condition2", 0.025s));
    tx2->set_message("Hello from tx2");

    auto tx3 = make_operator<ops::StringTxOp>(
        "tx3", make_condition<PeriodicCondition>("periodic-condition3", 0.1s));
    tx3->set_message("Hello from tx3");

    auto multi_rx = make_operator<ops::PerReceiverRxOp>(
        "multi_rx", make_condition<CountCondition>("count-condition4", 4));

    add_flow(tx1, multi_rx, {{"out", "in1"}});
    add_flow(tx2, multi_rx, {{"out", "in2"}});
    add_flow(tx3, multi_rx, {{"out", "in3"}});
  }
};

int main() {
  auto app = holoscan::make_application<MultiMessageApp>();

  // use the event-based scheduler so multiple operators can run simultaneously
  app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
      "event-based-scheduler", holoscan::Arg("worker_thread_number", static_cast<int64_t>(4))));

  app->run();

  return 0;
}
