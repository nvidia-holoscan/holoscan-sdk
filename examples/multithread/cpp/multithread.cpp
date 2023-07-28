/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

  void setup(OperatorSpec& spec) override { spec.output<std::shared_ptr<int>>("out"); }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    auto value = std::make_shared<int>(0);
    op_output.emit(value, "out");
  };
};

class DelayOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DelayOp)

  DelayOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<std::shared_ptr<int>>("in");
    spec.output<std::shared_ptr<int>>("out_val");
    spec.output<std::string>("out_name");
    spec.param(
        delay_, "delay", "Delay", "Amount of delay before incrementing the input value", 0.5);
    spec.param(
        increment_, "increment", "Increment", "Integer amount to increment the input value by", 1);
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    auto value = op_input.receive<std::shared_ptr<int>>("in").value();

    HOLOSCAN_LOG_INFO("{}: now waiting {} s", name(), delay_.get());

    // sleep for the specified time (rounded down to the nearest microsecond)
    int delay_us = static_cast<int>(delay_ * 1000000);
    usleep(delay_us);
    HOLOSCAN_LOG_INFO("{}: finished waiting", name());

    // increment value by the specified increment
    auto new_value = std::make_shared<int>(*value + increment_.get());
    auto nm = std::string(name());

    HOLOSCAN_LOG_INFO("{}: sending new value ({})", name(), *new_value);
    op_output.emit(new_value, "out_val");
    op_output.emit(nm, "out_name");
  };

 private:
  Parameter<double> delay_;
  Parameter<int> increment_;
};

class PingRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingRxOp)

  PingRxOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(
        names_, "names", "Input receivers for names", "List of input receivers for names.", {});
    spec.param(
        values_, "values", "Input receivers for values", "List of input receivers for values.", {});
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    auto value_vector = op_input.receive<std::vector<std::shared_ptr<int>>>("values").value();
    auto name_vector = op_input.receive<std::vector<std::string>>("names").value();

    HOLOSCAN_LOG_INFO("number of received names: {}", name_vector.size());
    HOLOSCAN_LOG_INFO("number of received values: {}", value_vector.size());
    int total = 0;
    for (auto vp : value_vector) { total += *vp; }
    HOLOSCAN_LOG_INFO("sum of received values: {}", total);
  };

 private:
  Parameter<std::vector<IOSpec*>> names_;
  Parameter<std::vector<IOSpec*>> values_;
};

}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  void set_num_delays(int num_delays) { num_delays_ = num_delays; }
  void set_delay(double delay) { delay_ = delay; }
  void set_delay_step(double delay_step) { delay_step_ = delay_step; }

  void compose() override {
    using namespace holoscan;

    auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(1));
    auto rx = make_operator<ops::PingRxOp>("rx");
    for (int i = 0; i < num_delays_; ++i) {
      std::string delay_name = fmt::format("mx{}", i);
      auto del_op = make_operator<ops::DelayOp, std::string>(
          delay_name, Arg{"delay", delay_ + delay_step_ * i}, Arg{"increment", i});
      add_flow(tx, del_op);
      add_flow(del_op, rx, {{"out_val", "values"}, {"out_name", "names"}});
    }
  }

 private:
  int num_delays_ = 32;
  double delay_ = 0.2;
  double delay_step_ = 0.05;
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
  app->set_num_delays(num_delay_ops);
  app->set_delay(delay);
  app->set_delay_step(delay_step);

  bool multithreaded = app->from_config("multithreaded").as<bool>();
  if (multithreaded) {
    // use MultiThreadScheduler instead of the default GreedyScheduler
    app->scheduler(app->make_scheduler<holoscan::MultiThreadScheduler>(
        "multithread-scheduler", app->from_config("scheduler")));
  }

  app->run();

  // Print all the results of data flow tracking
  if (tracking) { tracker->print(); }

  return 0;
}
