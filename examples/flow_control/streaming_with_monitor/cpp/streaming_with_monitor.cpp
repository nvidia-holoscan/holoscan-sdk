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

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <utility>

#include <holoscan/holoscan.hpp>

class GenSignalOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GenSignalOp)

  GenSignalOp() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.output<int>("output"); }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    value_++;
    HOLOSCAN_LOG_INFO("{} - Sending value {}", name(), value_);
    op_output.emit(value_, "output");
  }

 private:
  int value_ = 0;
};

class ProcessSignalOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ProcessSignalOp)

  ProcessSignalOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<int>("input");
    spec.output<int>("output");
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto value = op_input.receive<int>("input").value();
    HOLOSCAN_LOG_INFO("{} - Received value {}", name(), value);
    // Simulate the processing time
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    op_output.emit(value, "output");
  }
};

class ExecutionThrottlerOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ExecutionThrottlerOp)

  ExecutionThrottlerOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<std::any>("input");
    // Set the queue policy to kPop to prevent push failures when emitting a message to the output
    // port. Additionally, configure the condition to kNone for the execution throttler operator to
    // ensure it executes immediately upon receiving a message, even if the downstream operator is
    // not ready to receive it due to the detect event operator's lengthy (3ms) computation.
    spec.output<std::any>("output", holoscan::IOSpec::QueuePolicy::kPop)
        .condition(holoscan::ConditionType::kNone);
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto in_message = op_input.receive<std::any>("input");
    // Forward the input message to the output port.
    if (in_message) {
      auto value = in_message.value();
      op_output.emit(std::move(value), "output");
    }
  }
};

class DetectEventOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DetectEventOp)

  DetectEventOp() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.input<int>("input"); }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto value = op_input.receive<int>("input").value();

    metadata()->set("value", value);
    if (value % 2 == 1) {
      metadata()->set("is_detected", true);
    } else {
      metadata()->set("is_detected", false);
    }

    HOLOSCAN_LOG_INFO("{} - Received value {} (set metadata (is_detected: {}))",
                      name(),
                      value,
                      metadata()->get<bool>("is_detected", false));
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
  }
};

class ReportGenOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ReportGenOp)

  ReportGenOp() = default;

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    int value = metadata()->get<int>("value", -1);
    bool is_detected = metadata()->get<bool>("is_detected", false);
    HOLOSCAN_LOG_INFO(
        "{} - Input value {}, Metadata value (is_detected: {})", name(), value, is_detected);
  }
};

class VisualizeOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VisualizeOp)

  VisualizeOp() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.input<int>("input"); }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto value = op_input.receive<int>("input").value();
    HOLOSCAN_LOG_INFO("{} - Received value {}", name(), value);
  }
};

class StreamExecutionWithMonitorApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    using namespace std::chrono_literals;

    // Define the operators

    auto gen_signal = make_operator<GenSignalOp>("gen_signal");
    auto process_signal = make_operator<ProcessSignalOp>("process_signal");
    auto execution_throttler = make_operator<ExecutionThrottlerOp>("execution_throttler");
    auto detect_event = make_operator<DetectEventOp>("detect_event");
    // Configure the queue policy to kPop to ensure the input port avoids a push failure
    // when the execution throttler (lacking DownstreamMessageAffordableCondition on the output
    // port) sends a message while the input port queue is full.
    detect_event->queue_policy("input", IOSpec::IOType::kInput, IOSpec::QueuePolicy::kPop);
    auto report_generation = make_operator<ReportGenOp>("report_generation");
    auto visualize = make_operator<VisualizeOp>("visualize");

    // Define the streaming with monitor workflow
    //
    // Node Graph:
    //                                                               -> report_generation
    //                                                              |
    //               (cycle)                                        | (is_detected:true)
    //               -------         => execution_throttler => detect_event
    //               \     /         |                              | (is_detected:false)
    // <|start|> -> gen_signal => process_signal => visualize       |
    //              (triggered 5 times)                              -> (ignored)
    add_flow(start_op(), gen_signal);
    add_flow(gen_signal, gen_signal);
    add_flow(gen_signal, process_signal, {{"output", "input"}});
    add_flow(process_signal, execution_throttler);
    add_flow(execution_throttler, detect_event);
    add_flow(detect_event, report_generation);
    add_flow(process_signal, visualize);

    set_dynamic_flows(gen_signal, [process_signal](const std::shared_ptr<Operator>& op) {
      static int iteration = 0;
      ++iteration;
      HOLOSCAN_LOG_INFO("#iteration: {}", iteration);

      if (iteration <= 10) {
        op->add_dynamic_flow("output", process_signal);
        // Signal to trigger itself in the next iteration
        op->add_dynamic_flow(Operator::kOutputExecPortName, op);
      } else {
        iteration = 0;
      }
    });

    set_dynamic_flows(detect_event, [report_generation](const std::shared_ptr<Operator>& op) {
      // If the detect event operator detects an event, add a dynamic flow to the report generation
      // operator.
      bool is_detected = op->metadata()->get<bool>("is_detected", false);
      if (is_detected) { op->add_dynamic_flow(report_generation); }
    });
  }
};

int main() {
  using namespace holoscan;

  auto app = holoscan::make_application<StreamExecutionWithMonitorApp>();
  auto scheduler = app->make_scheduler<EventBasedScheduler>(
      "myscheduler", Arg("worker_thread_number", 2L), Arg("stop_on_deadlock", true));
  app->scheduler(scheduler);
  app->run();
  return 0;
}

// Expected output:
// (The output is not deterministic, so the order of the logs may vary.)
//
// gen_signal - Sending value 1
// #iteration: 1
// process_signal - Received value 1
// gen_signal - Sending value 2
// #iteration: 2
// visualize - Received value 1
// process_signal - Received value 2
// detect_event - Received value 1 (set metadata (is_detected: true))
// gen_signal - Sending value 3
// #iteration: 3
// visualize - Received value 2
// process_signal - Received value 3
// gen_signal - Sending value 4
// #iteration: 4
// visualize - Received value 3
// process_signal - Received value 4
// gen_signal - Sending value 5
// #iteration: 5
// report_generation - Input value 1, Metadata value (is_detected: true)
// detect_event - Received value 3 (set metadata (is_detected: true))
// visualize - Received value 4
// process_signal - Received value 5
// gen_signal - Sending value 6
// #iteration: 6
// visualize - Received value 5
// process_signal - Received value 6
// gen_signal - Sending value 7
// #iteration: 7
// visualize - Received value 6
// process_signal - Received value 7
// gen_signal - Sending value 8
// #iteration: 8
// report_generation - Input value 3, Metadata value (is_detected: true)
// detect_event - Received value 6 (set metadata (is_detected: false))
// visualize - Received value 7
// process_signal - Received value 8
// gen_signal - Sending value 9
// #iteration: 9
// visualize - Received value 8
// process_signal - Received value 9
// gen_signal - Sending value 10
// #iteration: 10
// detect_event - Received value 8 (set metadata (is_detected: false))
// visualize - Received value 9
// process_signal - Received value 10
// gen_signal - Sending value 11
// #iteration: 11
// visualize - Received value 10
// detect_event - Received value 10 (set metadata (is_detected: false))
