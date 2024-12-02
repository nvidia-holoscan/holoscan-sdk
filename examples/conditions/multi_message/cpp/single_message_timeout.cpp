/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

class RxTimeoutOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RxTimeoutOp)

  RxTimeoutOp() = default;

  void setup(OperatorSpec& spec) override {
    // Set a condition to allow execution once 5 messages have arrived or at least 250 ms has
    // elapsed since the prior time operator::compute was called.
    spec.input<std::shared_ptr<std::string>>("in", IOSpec::IOSize(5))
        .condition(ConditionType::kMultiMessageAvailableTimeout,
                   holoscan::Arg("execution_frequency", std::string{"250ms"}),
                   holoscan::Arg("min_sum", static_cast<size_t>(5)));
  }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    // We haven't set a specific number of messages per port, so loop over each
    // input until no more messages are in the queue.
    auto in_value = op_input.receive<std::shared_ptr<std::string>>("in");
    if (!in_value) { HOLOSCAN_LOG_INFO("No message available"); }
    std::vector<std::string> msgs_input;
    size_t message_count = 0;
    while (in_value) {
      msgs_input.push_back(*in_value.value());
      message_count++;
      in_value = op_input.receive<std::shared_ptr<std::string>>("in");
    }

    HOLOSCAN_LOG_INFO("{} messages received on in: {}", message_count, msgs_input);
  };
};

}  // namespace holoscan::ops

class RxTimeoutApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    using namespace std::chrono_literals;

    auto tx = make_operator<ops::StringTxOp>(
        "tx",
        make_condition<PeriodicCondition>("periodic-condition1", 100ms),
        make_condition<CountCondition>("count", 18));
    tx->set_message("hello from tx", true);

    auto rx_timeout = make_operator<ops::RxTimeoutOp>("rx_timeout");
    add_flow(tx, rx_timeout);
  }
};

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
  auto app = holoscan::make_application<RxTimeoutApp>();

  // Add a timeout slightly less than the execution_frequency so any final messages have time to
  // arrive after tx stops calling compute. If the deadlock timeout here is > execution_frequency
  // than the receive operator will continue to call compute indefinitely with 0 messages at the
  // execution frequency.
  app->scheduler(app->make_scheduler<holoscan::GreedyScheduler>(
      "greedy-scheduler", holoscan::Arg("stop_on_deadlock_timeout", static_cast<int64_t>(245))));

  app->run();

  return 0;
}
