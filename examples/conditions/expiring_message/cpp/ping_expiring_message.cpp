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

class PingTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxOp)

  PingTxOp() = default;

  void setup(OperatorSpec& spec) override { spec.output<std::shared_ptr<std::string>>("out"); }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto value =
        std::make_shared<std::string>(fmt::format("ExpiringMessageAvailable ping: {}", index_));
    ++index_;

    // retrieve the scheduler used for this application via it's fragment
    auto scheduler = fragment_->scheduler();
    // To get the clock we currently have to cast the scheduler to gxf::GXFScheduler.
    // TODO(unknown): Refactor C++ lib so the clock method is on Scheduler rather than
    //   GXFScheduler. That would allow us to avoid this dynamic_pointer_cast, but might require
    //   adding renaming Clock->GXFClock and then adding a new holoscan::Clock independent of GXF.
    auto gxf_scheduler = std::dynamic_pointer_cast<gxf::GXFScheduler>(scheduler);
    auto clock = gxf_scheduler->clock();
    auto timestamp = clock->timestamp();

    // emitting a timestamp is necessary for this port to be connected to an input port that is
    // using a ExpiringMessageAvailableCondition
    op_output.emit(value, "out", timestamp);
  };

 private:
  int index_ = 1;
};

class PingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRxOp)

  PingRxOp() = default;

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
    HOLOSCAN_LOG_INFO("PingRxOp::compute() called");

    while (true) {
      auto in_value = op_input.receive<std::shared_ptr<std::string>>("in");

      if (!in_value) { break; }

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
    auto tx = make_operator<ops::PingTxOp>(
        "tx",
        make_condition<CountCondition>("count-condition", 8),
        make_condition<PeriodicCondition>("periodic-condition", 0.01s));

    auto rx = make_operator<ops::PingRxOp>("rx");

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
