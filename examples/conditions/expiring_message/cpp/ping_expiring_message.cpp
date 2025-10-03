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

#include "holoscan/holoscan.hpp"
#include "holoscan/core/conditions/gxf/expiring_message.hpp"

namespace holoscan::ops {

class TimestampPingTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TimestampPingTxOp)

  TimestampPingTxOp() = default;

  void setup(OperatorSpec& spec) override { spec.output<std::shared_ptr<std::string>>("out"); }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto value =
        std::make_shared<std::string>(fmt::format("ExpiringMessageAvailable ping: {}", index_));
    ++index_;

    // retrieve the scheduler used for this application via it's fragment
    auto scheduler = fragment_->scheduler();
    auto clock = scheduler->clock();
    auto timestamp = clock->timestamp();

    // emitting a timestamp is necessary for this port to be connected to an input port that is
    // using a ExpiringMessageAvailableCondition
    op_output.emit(value, "out", timestamp);
  };

 private:
  int index_ = 1;
};

class TimestampPingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TimestampPingRxOp)

  TimestampPingRxOp() = default;

  void setup(OperatorSpec& spec) override {
    ArgList expiring_message_arglist{Arg("max_batch_size", static_cast<int64_t>(5)),
                                     Arg("max_delay_ns", static_cast<int64_t>(1'000'000'000))};
    spec.input<std::shared_ptr<std::string>>("in")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   Arg("capacity", static_cast<int64_t>(5)),
                   Arg("policy", static_cast<int64_t>(1)))
        .condition(ConditionType::kExpiringMessageAvailable, expiring_message_arglist);
  }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("TimestampPingRxOp::compute() called");

    while (true) {
      auto in_value = op_input.receive<std::shared_ptr<std::string>>("in");

      // receive timestamp (must be called after receive is called for this port)
      auto in_timestamp = op_input.get_acquisition_timestamp("in");
      if (!in_timestamp.has_value()) {
        std::string error_msg = fmt::format(
            "Operator '{}' failed to find timestamp in message received from port 'in': {}",
            name());
        HOLOSCAN_LOG_ERROR(error_msg);
        throw std::runtime_error(error_msg);
      }
      HOLOSCAN_LOG_INFO("Rx message acquisition timestamp: {}", in_timestamp.value());

      if (!in_value) {
        break;
      }

      auto message = in_value.value();
      if (message) {
        HOLOSCAN_LOG_INFO("Rx message received: {}", message->c_str());
      } else {
        HOLOSCAN_LOG_INFO("Rx message received: nullptr");
      }
    }
  };
};

}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    using namespace std::chrono_literals;
    // Configure the operators. Here we use CountCondition to terminate
    // execution after a specific number of messages have been sent.
    // PeriodicCondition is used so that each subsequent message is
    // sent only after a period of 10 milliseconds has elapsed.
    auto tx = make_operator<ops::TimestampPingTxOp>(
        "tx",
        make_condition<CountCondition>("count-condition", 8),
        make_condition<PeriodicCondition>("periodic-condition", 0.01s));

    auto rx = make_operator<ops::TimestampPingRxOp>("rx");

    add_flow(tx, rx);
  }
};

int main() {
  auto app = holoscan::make_application<App>();
  auto& tracker = app->track(0, 0, 0);
  tracker.enable_logging();
  app->run();
  tracker.print();
  return 0;
}
