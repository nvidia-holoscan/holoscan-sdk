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

#include <fmt/core.h>
#include <unistd.h>

#include <memory>
#include <string>
#include <vector>

#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

class PingTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxOp)

  PingTxOp() = default;

  void setup(OperatorSpec& spec) override {
    // Note: Setting ConditionType::kNone overrides the default of
    //   ConditionType::kDownstreamMessageAffordable. This means that the operator will be triggered
    //   regardless of whether any operators connected downstream have space in their queues.
    spec.output<int64_t>("out").condition(ConditionType::kNone);
    spec.param(initial_value_,
               "initial_value",
               "Initial value",
               "Initial value to emit",
               static_cast<int64_t>(0));
    spec.param(increment_,
               "increment",
               "Increment",
               "Integer amount to increment the value by on each subsequent call",
               static_cast<int64_t>(1));
  }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto value = initial_value_.get() + count_ * increment_.get();
    op_output.emit(value, "out");
    count_ += 1;
  };

 private:
  Parameter<int64_t> initial_value_;
  Parameter<int64_t> increment_;
  int64_t count_ = 0;
};

class IncrementOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(IncrementOp)

  IncrementOp() = default;

  /* Setup the input and output ports with custom settings on the input port.
   *
   *  Notes
   *  =====
   *  For policy:
   *    - IOSpec::QueuePolicy::kPop = pop the oldest value in favor of the new one when the
   *      queue is full
   *    - IOSpec::QueuePolicy::kReject = reject the new value when the queue is full
   *    - IOSpec::QueuePolicy::kFault = fault if queue is full (default)
   *
   *  For capacity:
   *    When capacity > 1, even once messages stop arriving, this entity will continue to
   *    call ``compute`` for each remaining item in the queue.
   *
   *  The ``condition`` method call here is the same as the default setting, and is shown
   *  only for completeness. `min_size` = 1 means that this operator will not call compute
   *  unless there is at least one message in the queue.
   *
   *  One could also set the receiver's capacity and policy via the connector method:
   *
   *          .connector(IOSpec::ConnectorType::kDoubleBuffer,
   *                     Arg("capacity", static_cast<uint64_t>(1)),
   *                     Arg("policy", static_cast<uint64_t>(1)))  // 1 = reject
   *
   *  but that is less flexible as `IOSpec::ConnectorType::kDoubleBuffer` is appropriate for
   *  within-fragment connections, but will not work if the operator was connected to a different
   *  fragment. By passing the capacity and policy as arguments to the OperatorSpec::input method,
   *  the SDK can still select the appropriate default receiver type depending on whether the
   *  connection is within-fragment or across fragments (or whether an annotated variants of the
   *  receiver class is needed for data flow tracking).
   */
  void setup(OperatorSpec& spec) override {
    spec.input<int64_t>("in", IOSpec::IOSize(1), IOSpec::QueuePolicy::kReject)
        .condition(  // arguments to condition here are the same as the defaults
            ConditionType::kMessageAvailable,
            Arg("min_size", static_cast<uint64_t>(1)),
            Arg("front_stage_max_size", static_cast<size_t>(1)));
    spec.output<int64_t>("out");

    spec.param(increment_,
               "increment",
               "Increment",
               "Integer amount to increment the input value by",
               static_cast<int64_t>(0));
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    int64_t value = op_input.receive<int64_t>("in").value();
    // increment value by the specified increment
    int64_t new_value = value + increment_.get();
    op_output.emit(new_value, "out");
  };

 private:
  Parameter<int64_t> increment_;
};

class PingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRxOp)

  PingRxOp() = default;

  void setup(OperatorSpec& spec) override { spec.input<int64_t>("in"); }

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto value = op_input.receive<int64_t>("in").value();
    HOLOSCAN_LOG_INFO("receiver '{}' received value: {}", name(), value);
  };
};

}  // namespace holoscan::ops

/* This application has a single transmitter connected to two parallel branches
 *
 *  The geometry of the application is as shown below:
 *
 *     increment1--rx1
 *    /
 *  tx
 *    \
 *     increment2--rx2
 *
 *  The top branch is forced via a PeriodicCondition to run at a slower rate than
 *  the source. It is currently configured to discard any extra messages that arrive
 *  at increment1 before it is ready to execute again, but different behavior could be
 *  achieved via other settings to policy and/or queue sizes.
 */
class MultiRateApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Configure the operators. Here we use CountCondition to terminate
    // execution after a specific number of messages have been sent and a
    // PeriodicCondition to control how often messages are sent.
    int64_t source_rate_hz = 60;                                // messages sent per second
    int64_t period_source_ns = 1'000'000'000 / source_rate_hz;  // period in nanoseconds
    auto tx = make_operator<ops::PingTxOp>(
        "tx",
        make_condition<CountCondition>("count", 100),
        make_condition<PeriodicCondition>("tx-period", period_source_ns));

    // first branch will have a periodic condition so it can't run faster than 5 Hz
    int64_t branch1_hz = 5;
    int64_t period_ns1 = 1'000'000'000 / branch1_hz;
    auto increment1 = make_operator<ops::IncrementOp>(
        "increment1", make_condition<PeriodicCondition>("increment1-period", period_ns1));
    // could set the queue policy here if it wasn't already set to `IOSpec::QueuePolicy::kReject`
    // in IncrementOp::setup() method as follows:
    //   increment1->queue_policy("in", IOSpec::IOType::kInput, IOSpec::QueuePolicy::kReject);
    auto rx1 = make_operator<ops::PingRxOp>("rx1");
    add_flow(tx, increment1);
    add_flow(increment1, rx1);

    // second branch is the same, but no periodic condition so will tick on every received message
    auto increment2 = make_operator<ops::IncrementOp>("increment2");
    // could set the queue policy here if it wasn't already set to `IOSpec::QueuePolicy::kReject`
    // in IncrementOp::setup() method as follows:
    //   increment2->queue_policy("in", IOSpec::IOType::kInput, IOSpec::QueuePolicy::kReject);
    auto rx2 = make_operator<ops::PingRxOp>("rx2");
    add_flow(tx, increment2);
    add_flow(increment2, rx2);
  }
};

int main([[maybe_unused]] int argc, char** argv) {
  auto app = holoscan::make_application<MultiRateApp>();

  // Get the configuration
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path += "/multi_branch_pipeline.yaml";
  app->config(config_path);

  auto scheduler = app->from_config("scheduler").as<std::string>();
  if (scheduler == "multi_thread") {
    // use MultiThreadScheduler instead of the default GreedyScheduler
    app->scheduler(app->make_scheduler<holoscan::MultiThreadScheduler>(
        "multithread-scheduler", app->from_config("multi_thread_scheduler")));
  } else if (scheduler == "event_based") {
    // use EventBasedScheduler instead of the default GreedyScheduler
    app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
        "event-based-scheduler", app->from_config("event_based_scheduler")));
  } else if (scheduler == "greedy") {
    app->scheduler(app->make_scheduler<holoscan::GreedyScheduler>(
        "greedy-scheduler", app->from_config("greedy_scheduler")));
  } else if (scheduler != "default") {
    throw std::runtime_error(fmt::format(
        "unrecognized scheduler option '{}', should be one of {{'multi_thread', 'event_based', "
        "'greedy', 'default'}}",
        scheduler));
  }

  app->run();

  return 0;
}
