/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <utility>
#include <vector>

#include "holoscan/holoscan.hpp"
#include "holoscan/operators/ping_rx/ping_rx.hpp"
#include "holoscan/operators/ping_tx/ping_tx.hpp"

namespace holoscan::ops {

/**
 * @class RoundRobinBroadcastOp
 * @brief A Holoscan operator that broadcasts input messages to multiple output ports in a
 * round-robin fashion.
 *
 * The output ports will be named "output001", "output002", ...,
 * `fmt::format("output{:03d}", num_broadcast)`.
 *
 * This operator receives messages on a single input port and broadcasts them to multiple output
 * ports in a round-robin manner. The number of output ports is determined by the "num_broadcast"
 * parameter.
 */
class RoundRobinBroadcastOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RoundRobinBroadcastOp)

  RoundRobinBroadcastOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<std::any>("input");

    if (num_broadcast_ < 1) {
      throw std::runtime_error("Must have num_broadcast >= 1 (at least one output port).");
    }
    // Create output ports output001, output002, ...
    for (int i = 0; i < num_broadcast_; ++i) {
      std::string port_name = output_name(i);
      spec.output<std::any>(std::move(port_name));
    }
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto message = op_input.receive<std::any>("input");
    if (!message) { throw std::runtime_error("No message found on port named 'input'"); }
    auto out_port_name = output_name(port_index_);
    auto value = message.value();

    // (The specific casting logic below is needed for general compatibility for use with
    //  nvidia::gxf::Codelets which cannot handle the std::any type. We pass GXF Entities through
    //  as an entity rather than std::any. It is not strictly necessary for this example app, but
    //  we provide this so the operator is generally reusable outside of the context of this
    //  example).
    if (value.type() == typeid(holoscan::gxf::Entity)) {
      // emit as entity
      auto entity = std::any_cast<holoscan::gxf::Entity>(value);
      op_output.emit(entity, out_port_name.c_str());
    } else {
      // emit as std::any
      op_output.emit(value, out_port_name.c_str());
    }
    port_index_ = (port_index_ + 1) % num_broadcast_;
  }

  /**
   * @brief Get the name of the output port associated with a given linear port index.
   *
   * @param index The linear index of the output port (valid range: [0, num_broadcast_)).
   * @return The name of the output port.
   */
  std::string static output_name(int index) { return fmt::format("output{:03d}", index + 1); }

 private:
  int num_broadcast_ = 4;  ///< the number of output ports to create
  int port_index_ = 0;     ///< index of the output port to send the next message to
};

class SlowOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SlowOp)

  SlowOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.output<int>("out");
    spec.param(delay_,
               "delay",
               "Delay",
               "Amount of delay (in seconds) before incrementing the input value",
               1.0 / 15.0);
    spec.param(
        increment_, "increment", "Increment", "Integer amount to increment the input value by", 0);
    spec.param(silent_, "silent", "Silent mode?", "Whether to log info on receive", false);
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto value = op_input.receive<int>("in").value();

    // increment value by the specified increment
    int new_value = value + increment_.get();

    double delay = delay_.get();
    bool silent = silent_.get();
    if (delay > 0) {
      if (!silent) { HOLOSCAN_LOG_INFO("{}: now waiting {} s", name(), delay); }
      // sleep for the specified time (rounded down to the nearest microsecond)
      int delay_us = static_cast<int>(delay * 1000000);
      usleep(delay_us);
      if (!silent) { HOLOSCAN_LOG_INFO("{}: finished waiting", name()); }
    }
    if (!silent) { HOLOSCAN_LOG_INFO("{}: sending new value ({})", name(), new_value); }
    op_output.emit(new_value, "out");
  };

 private:
  Parameter<double> delay_;
  Parameter<int> increment_;
  Parameter<bool> silent_;
};

/**
 * @class GatherOneOp
 * @brief A Holoscan operator that has "num_gather" input ports and a single output port.
 * This operator only checks a specific input port for messages on any given `compute` call. Which
 * input port is checked varies across compute calls in a round-robin fashion.
 *
 * For the way add_flow calls are made in `RoundRobinApp`, this ensures that the original order of
 * frames produced by `PingTxOp` is preserved.
 *
 * The input ports will be named "input001", "input002", ...,
 * `fmt::format("input{:03d}", num_gather)`.
 */
class GatherOneOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GatherOneOp)

  GatherOneOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    // Create input ports input001, input002, ...
    std::vector<std::string> all_input_port_names;
    all_input_port_names.reserve(num_gather_);
    for (int i = 0; i < num_gather_; ++i) {
      std::string port_name = input_name(i);
      spec.input<std::any>(port_name);
      all_input_port_names.emplace_back(port_name);
    }
    // create output port
    spec.output<std::any>("output");

    // Use kMultiMessageAvailable to allow the operator to tick as long a message has arrived on
    // one of the input ports. Above, we already stored the list of input port names in
    // `all_input_port_names` for use here.

    // TODO(grelee): ideally would create a custom native condition type called
    // `RoundRobinMessageAvailableCondition` which would check a specific input port in turn so
    // we don't have to have the if(message) {} logic in `compute` to check if the received
    // message wasd on the desired port.
    ArgList multi_message_args{
        holoscan::Arg("min_sum", static_cast<size_t>(1)),
        holoscan::Arg("sampling_mode", MultiMessageAvailableCondition::SamplingMode::kSumOfAll)};
    spec.multi_port_condition(
        ConditionType::kMultiMessageAvailable, all_input_port_names, multi_message_args);
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto message = op_input.receive<std::any>(input_name(port_index_).c_str());
    if (message) {
      // increment the input port index to check next time
      port_index_ = (port_index_ + 1) % num_gather_;

      // output the message
      // (The specific casting logic below is needed for general compatibility for use with
      //  nvidia::gxf::Codelets which cannot handle the std::any type. We pass GXF Entities through
      //  as an entity rather than std::any. It is not strictly necessary for this example app, but
      //  we provide this so the operator is generally reusable outside of the context of this
      //  example).
      auto value = message.value();
      if (value.type() == typeid(holoscan::gxf::Entity)) {
        // emit as entity
        auto entity = std::any_cast<holoscan::gxf::Entity>(value);
        op_output.emit(entity, "output");
      } else {
        // emit as std::any
        op_output.emit(value, "output");
      }
    }
  }

  /**
   * @brief Get the name of the input port associated with a given linear port index.
   *
   * @param index The linear index of the input port (valid range: [0, num_gather_)).
   * @return The name of the input port.
   */
  std::string static input_name(int index) { return fmt::format("input{:03d}", index + 1); }

 private:
  int num_gather_ = 4;
  int port_index_ = 0;
};

}  // namespace holoscan::ops

/**
 * @brief Round-robin broadcast application.
 *
 * Please see ../README.md for a detailed description and diagram of the operators involved.
 */
class RoundRobinApp : public holoscan::Application {
 public:
  void set_num_broadcast(int num_broadcast) { num_broadcast_ = num_broadcast; }
  void set_count(int64_t count) { count_ = count; }
  void set_delay(double delay) { delay_ = delay; }
  void set_silent(bool silent) { silent_ = silent; }

  void compose() override {
    using namespace holoscan;
    using namespace std::chrono_literals;

    // transmit messages at a 60Hz rate
    auto tx = make_operator<ops::PingTxOp>(
        "tx",
        make_condition<CountCondition>("tx-count", count_),
        // 60 Hz = 0.016666 second period
        make_condition<PeriodicCondition>("tx-periodic", 0.016666666666666s));

    auto round_robin = make_operator<ops::RoundRobinBroadcastOp>(
        "round_robin", Arg("num_broadcast", num_broadcast_));

    auto gather = make_operator<ops::GatherOneOp>("gather", Arg("num_gather", num_broadcast_));

    auto rx = make_operator<ops::PingRxOp>("rx");

    add_flow(tx, round_robin);
    for (int i = 0; i < num_broadcast_; ++i) {
      std::string delay_name = fmt::format("delay{:02d}", i);
      auto del_op = make_operator<ops::SlowOp, std::string>(
          std::move(delay_name), Arg{"delay", delay_}, Arg{"increment", 0}, Arg{"silent", silent_});
      add_flow(round_robin, del_op, {{round_robin->output_name(i), "in"}});
      add_flow(del_op, gather, {{"out", gather->input_name(i)}});
    }
    add_flow(gather, rx);
  }

 private:
  int num_broadcast_ = 4;
  int64_t count_ = 100;
  double delay_ = 1.0 / 15.0;
  bool silent_ = false;
};

int main([[maybe_unused]] int argc, char** argv) {
  auto app = holoscan::make_application<RoundRobinApp>();

  // Get the configuration
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path /= std::filesystem::path("round_robin.yaml");
  if (argc >= 2) { config_path = argv[1]; }
  app->config(config_path);

  // Turn on data flow tracking if it is specified in the YAML
  auto tracking = app->from_config("tracking").as<bool>();
  holoscan::DataFlowTracker* tracker = nullptr;
  if (tracking) { tracker = &app->track(0, 0, 0); }

  // set customizable application parameters via the YAML
  auto scheduler = app->from_config("scheduler").as<std::string>();
  if (scheduler == "event_based") {
    // use EventBasedScheduler instead of the default GreedyScheduler
    app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
        "event-based-scheduler", app->from_config("event_based_scheduler")));
  } else if (scheduler == "greedy") {
    app->scheduler(app->make_scheduler<holoscan::GreedyScheduler>(
        "greedy-scheduler", app->from_config("greedy_scheduler")));
  } else {
    throw std::runtime_error(
        fmt::format("unrecognized scheduler option '{}', should be one of ('event_based', "
                    "'greedy')",
                    scheduler));
  }

  app->run();

  // Print all the results of data flow tracking
  if (tracking) { tracker->print(); }

  return 0;
}
