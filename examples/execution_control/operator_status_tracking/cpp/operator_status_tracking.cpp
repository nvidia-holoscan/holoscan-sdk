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

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <magic_enum.hpp>

// A simple operator that processes a fixed number of times and then completes
class FiniteSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FiniteSourceOp)

  FiniteSourceOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<int>("out");
    spec.param(max_count_, "max_count", "Maximum Count", "Maximum number of times to emit data", 5);
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    if (count_ < max_count_.get()) {
      std::cout << "[" << name() << "] Emitting data: " << count_ << std::endl;
      op_output.emit(count_, "out");

      // Simulate some processing time
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    } else {
      std::cout << "[" << name() << "] Additional iterations after completion: " << count_
                << std::endl;
    }

    count_++;
  }

  void stop() override { std::cout << "[" << name() << "] Stopping operator" << std::endl; }

 private:
  holoscan::Parameter<int> max_count_;
  int count_ = 0;
};

// An operator that processes data
class ProcessorOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ProcessorOp)

  ProcessorOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.output<int>("out");
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto data = op_input.receive<int>("in").value();

    // Process the data
    int result = data * 2;
    std::cout << "[" << name() << "] Processing data: " << data << " -> " << result << std::endl;

    // Emit the processed data
    op_output.emit(result, "out");

    // Simulate some processing time
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  void stop() override { std::cout << "[" << name() << "] Stopping operator" << std::endl; }
};

// An operator that consumes data
class ConsumerOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ConsumerOp)

  ConsumerOp() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.input<int>("in"); }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto data = op_input.receive<int>("in").value();

    // Consume the data
    std::cout << "[" << name() << "] Consuming data: " << data << std::endl;

    // Simulate some processing time
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
  }

  void stop() override { std::cout << "[" << name() << "] Stopping operator" << std::endl; }
};

// A dedicated monitoring operator that runs independently of the main pipeline
class MonitorOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MonitorOp)

  MonitorOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    // No inputs or outputs - this operator runs independently
    spec.param(monitored_operators_,
               "monitored_operators",
               "Monitored Operators",
               "Names of operators to monitor",
               {});
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    static int consecutive_idle_count = 0;

    std::cerr << "[" << name() << "] Operator status summary:" << std::endl;

    bool is_pipeline_idle = true;
    for (const auto& op_name : monitored_operators_.get()) {
      auto maybe_status = context.get_operator_status(op_name);
      if (!maybe_status) { throw std::runtime_error(maybe_status.error().what()); }
      auto status = maybe_status.value();
      std::cerr << "  - " << op_name << ": " << magic_enum::enum_name(status) << std::endl;
      if (status != holoscan::OperatorStatus::kIdle) { is_pipeline_idle = false; }
    }

    if (is_pipeline_idle) {
      consecutive_idle_count++;

      // Currently, there's no way to retrieve the computed SchedulingCondition status from the
      // operator (i.e., whether the computed scheduling condition status is NEVER).
      // Instead, we use `consecutive_idle_count` to determine if all operators, except the monitor
      // operator, have completed.
      // If `consecutive_idle_count` is equal to or greater than the hardcoded value 3,
      // we consider them completed, as there are cases where all operators are idle but not yet
      // completed.
      // A better approach for checking an operator's computed scheduling condition will be
      // available in a future release.
      if (consecutive_idle_count >= 3) {
        std::cerr << "[" << name() << "] All operators have completed." << std::endl;
        // Stop the monitor operator which is the only operator keeping the application alive
        stop_execution();  // the application will terminate through a deadlock
      }
    } else {
      consecutive_idle_count = 0;
    }
  }

 private:
  holoscan::Parameter<std::vector<std::string>> monitored_operators_;
};

class OperatorStatusTrackingApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Define the operators
    auto source = make_operator<FiniteSourceOp>(
        "source", make_condition<CountCondition>(10), Arg("max_count", 5));
    auto processor = make_operator<ProcessorOp>("processor");
    auto consumer = make_operator<ConsumerOp>("consumer");

    // Create the independent monitor operator with its own scheduling condition
    // This will continue running even if the main pipeline stops
    auto monitor = make_operator<MonitorOp>(
        "monitor",
        make_condition<PeriodicCondition>(std::chrono::milliseconds(50)),
        Arg("monitored_operators", std::vector<std::string>{"source", "processor", "consumer"}));

    // Define the workflow for the main pipeline
    add_flow(source, processor, {{"out", "in"}});
    add_flow(processor, consumer, {{"out", "in"}});

    // The monitor is not connected to any other operators - it runs independently
    add_operator(monitor);

    // Print information about the execution context API
    std::cout << "This example demonstrates the Operator Status Tracking API." << std::endl;
    std::cout << "The source operator will emit 5 values and then stop executing after 10 "
              << "iterations." << std::endl;
    std::cout << "The monitor operator runs independently and tracks the status of all operators."
              << std::endl;
    std::cout << "When operators complete, the monitor will be terminated." << std::endl;
    std::cout << "-------------------------------------------------------------------" << std::endl;
  }
};

int main() {
  auto app = holoscan::make_application<OperatorStatusTrackingApp>();
  holoscan::ArgList scheduler_args{holoscan::Arg("worker_thread_number", static_cast<int64_t>(2))};
  app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>("EBS", scheduler_args));
  app->run();

  std::cout << "-------------------------------------------------------------------" << std::endl;
  std::cout << "Application completed. All operators have finished processing." << std::endl;
  return 0;
}
