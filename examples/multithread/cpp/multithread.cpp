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

#include <fmt/core.h>
#include <unistd.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

class PingTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingTxOp)

  PingTxOp() = default;

  void setup(OperatorSpec& spec) override { spec.output<std::shared_ptr<int>>("out"); }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    int value = 0;
    op_output.emit(value, "out");
  };
};

class DelayOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DelayOp)

  DelayOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.output<int>("out_val");
    spec.output<std::string>("out_name");
    spec.param(
        delay_, "delay", "Delay", "Amount of delay before incrementing the input value", 0.5);
    spec.param(
        increment_, "increment", "Increment", "Integer amount to increment the input value by", 1);
    spec.param(silent_, "silent", "Silent mode?", "Whether to log info on receive", false);
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    auto value = op_input.receive<int>("in").value();

    // increment value by the specified increment
    int new_value = value + increment_.get();
    auto nm = std::string(name_);

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
    op_output.emit(new_value, "out_val");
    op_output.emit(std::move(nm), "out_name");
  };

 private:
  Parameter<double> delay_;
  Parameter<int> increment_;
  Parameter<bool> silent_;
};

class PingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRxOp)

  explicit PingRxOp(bool silent) : silent_(silent) {}

  void setup(OperatorSpec& spec) override {
    // // Since Holoscan SDK v2.3, users can define a multi-receiver input port using 'spec.input()'
    // // with 'IOSpec::kAnySize'.
    // // The old way is to use 'spec.param()' with 'Parameter<std::vector<IOSpec*>> receivers_;'.
    // spec.param(
    //     names_, "names", "Input receivers for names", "List of input receivers for names.", {});
    // spec.param(
    //     values_, "values", "Input receivers for values", "List of input receivers for values.",
    //     {});
    spec.input<std::vector<std::string>>("names", IOSpec::kAnySize);
    spec.input<std::vector<int>>("values", IOSpec::kAnySize);
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    std::vector<int> value_vector;
    std::vector<std::string> name_vector;
    value_vector = op_input.receive<std::vector<int>>("values").value();
    name_vector = op_input.receive<std::vector<std::string>>("names").value();
    if (!silent_) {
      HOLOSCAN_LOG_INFO("number of received names: {}", name_vector.size());
      HOLOSCAN_LOG_INFO("number of received values: {}", value_vector.size());
    }
    int total = 0;
    for (auto vp : value_vector) { total += vp; }
    if (!silent_) { HOLOSCAN_LOG_INFO("sum of received values: {}", total); }
  };

 private:
  // // Since Holoscan SDK v2.3, the following lines are no longer needed.
  // Parameter<std::vector<IOSpec*>> names_;
  // Parameter<std::vector<IOSpec*>> values_;
  bool silent_ = false;
};

}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  void set_num_delays(int num_delays) { num_delays_ = num_delays; }
  void set_count(int64_t count) { count_ = count; }
  void set_delay(double delay) { delay_ = delay; }
  void set_delay_step(double delay_step) { delay_step_ = delay_step; }
  void set_silent(bool silent) { silent_ = silent; }

  void compose() override {
    using namespace holoscan;

    auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(count_));
    auto rx = make_operator<ops::PingRxOp>("rx", silent_);
    for (int i = 0; i < num_delays_; ++i) {
      std::string delay_name = fmt::format("mx{}", i);
      auto del_op = make_operator<ops::DelayOp, std::string>(std::move(delay_name),
                                                             Arg{"delay", delay_ + delay_step_ * i},
                                                             Arg{"increment", i},
                                                             Arg{"silent", silent_});
      add_flow(tx, del_op);
      add_flow(del_op, rx, {{"out_val", "values"}, {"out_name", "names"}});
    }
  }

 private:
  int num_delays_ = 32;
  int64_t count_ = 1;
  double delay_ = 0.2;
  double delay_step_ = 0.05;
  bool silent_ = false;
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  // Get the configuration
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path += "/multithread.yaml";
  app->config(config_path);

  // Turn on data flow tracking if it is specified in the YAML
  bool tracking = app->from_config("tracking").as<bool>();
  holoscan::DataFlowTracker* tracker = nullptr;
  if (tracking) { tracker = &app->track(0, 0, 0); }

  // set customizable application parameters via the YAML
  int num_delay_ops = app->from_config("num_delay_ops").as<int>();
  double delay = app->from_config("delay").as<double>();
  double delay_step = app->from_config("delay_step").as<double>();
  int count = app->from_config("count").as<int>();
  bool silent = app->from_config("silent").as<bool>();
  app->set_num_delays(num_delay_ops);
  app->set_delay(delay);
  app->set_delay_step(delay_step);
  app->set_count(count);
  app->set_silent(silent);

  std::string scheduler = app->from_config("scheduler").as<std::string>();
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
        "unrecognized scheduler option '{}', should be one of ('multi_thread', 'event_based', "
        "'greedy', 'default')",
        scheduler));
  }

  app->run();

  // Print all the results of data flow tracking
  if (tracking) { tracker->print(); }

  return 0;
}
