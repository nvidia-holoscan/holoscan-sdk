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

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

namespace holoscan {

namespace {

///////////////////////////////////////////////////////////////////////////////
// Async Buffer Tracking Operators for Cyclic Graphs
///////////////////////////////////////////////////////////////////////////////

class AsyncCyclicSourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AsyncCyclicSourceOp)

  AsyncCyclicSourceOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<int>("feedback");
    spec.output<int>("out");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    // Try to receive feedback message (may not be available initially)
    auto feedback_message = op_input.receive<int>("feedback");

    int out_value;
    if (!feedback_message) {
      HOLOSCAN_LOG_INFO("{} did not receive feedback, emitting new message", name());
      // Create a new value when no feedback is available
      out_value = count_;
    } else {
      HOLOSCAN_LOG_INFO("{} received feedback {}, creating new message from it {}",
                        name(),
                        feedback_message.value(),
                        count_);
      // Create a new value from the received value
      out_value = feedback_message.value() + 1;
    }

    op_output.emit(out_value);
    count_++;
  }

 private:
  int count_ = 1;
};

class AsyncCyclicMiddleOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AsyncCyclicMiddleOp)

  AsyncCyclicMiddleOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.output<int>("out");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto in_value = op_input.receive<int>("in");
    if (!in_value) {
      HOLOSCAN_LOG_INFO("{} did not receive valid input", name());
      return;
    }

    // Create a new value from the received value to avoid passthrough issues in cycles
    int out_value = in_value.value() + 1;
    op_output.emit(out_value);
    HOLOSCAN_LOG_INFO("{} forwarding value {} -> {}", name(), in_value.value(), out_value);
    count_++;
  }

 private:
  int count_ = 1;
};

class AsyncCyclicMiddleDualOutputOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AsyncCyclicMiddleDualOutputOp)

  AsyncCyclicMiddleDualOutputOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.output<int>("out_feedback");
    spec.output<int>("out_forward");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto in_value = op_input.receive<int>("in");
    if (!in_value) {
      HOLOSCAN_LOG_INFO("{} did not receive valid input", name());
      return;
    }

    // Create a new value from the received value to avoid passthrough issues in cycles
    int out_value = in_value.value() + 1;
    op_output.emit(out_value, "out_feedback");
    op_output.emit(out_value, "out_forward");
    HOLOSCAN_LOG_INFO("{} forwarding value {} -> {}", name(), in_value.value(), out_value);
    count_++;
  }

 private:
  int count_ = 1;
};

class AsyncCyclicFeedbackOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AsyncCyclicFeedbackOp)

  AsyncCyclicFeedbackOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.output<int>("feedback");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto in_value = op_input.receive<int>("in");
    if (!in_value) {
      HOLOSCAN_LOG_INFO("{} did not receive valid input", name());
      return;
    }

    // Create a new value from the received value to avoid passthrough issues in cycles
    int out_value = in_value.value() + 1;
    op_output.emit(out_value);
    HOLOSCAN_LOG_INFO("{} sending feedback value {} -> {}", name(), in_value.value(), out_value);
    count_++;
  }

 private:
  int count_ = 1;
};

class AsyncCyclicDualInputSourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AsyncCyclicDualInputSourceOp)

  AsyncCyclicDualInputSourceOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<int>("feedback_local");
    spec.input<int>("feedback_remote");
    spec.output<int>("out");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    // Try to receive feedback messages (may not be available initially)
    auto feedback_local = op_input.receive<int>("feedback_local");
    auto feedback_remote = op_input.receive<int>("feedback_remote");

    int out_value;
    if (!feedback_local && !feedback_remote) {
      HOLOSCAN_LOG_INFO("{} did not receive any feedback, emitting new message", name());
      // Create a new value when no feedback is available
      out_value = count_;
    } else if (feedback_local && !feedback_remote) {
      HOLOSCAN_LOG_INFO("{} received local feedback {}, creating new message {}",
                        name(),
                        feedback_local.value(),
                        count_);
      // Use local feedback
      out_value = feedback_local.value() + 1;
    } else if (!feedback_local && feedback_remote) {
      HOLOSCAN_LOG_INFO("{} received remote feedback {}, creating new message {}",
                        name(),
                        feedback_remote.value(),
                        count_);
      // Use remote feedback
      out_value = feedback_remote.value() + 1;
    } else {
      HOLOSCAN_LOG_INFO("{} received both feedbacks (local: {}, remote: {}), using local {}",
                        name(),
                        feedback_local.value(),
                        feedback_remote.value(),
                        count_);
      // Use their summation when both are available
      out_value = feedback_local.value() + feedback_remote.value() + 1;
    }
    op_output.emit(out_value);
    count_++;
  }

 private:
  int count_ = 1;
};

}  // namespace
///////////////////////////////////////////////////////////////////////////////
// Application Classes for Cyclic Graphs
///////////////////////////////////////////////////////////////////////////////

/* ComplexCycleApp - Application with multiple interconnected cycles
 *
 * First Cycle:
 *   Source1 -----> Middle1 -----> Feedback1
 *     ^               |                |
 *     |_______________|                |
 *                                      |
 *                                      |
 *      ---------------------------------
 *      |
 *   Source2 -----> Middle2 -----> Feedback2
 *     ^                              |
 *     |______________________________|
 *
 * Second Cycle above
 */
class ComplexCycleApp : public Application {
 public:
  void compose() override {
    // First cycle: Source1 -> Middle1 -> Source1 (via feedback)
    auto source1 = make_operator<AsyncCyclicSourceOp>(
        "CyclicSource1",
        make_condition<PeriodicCondition>("source1-periodic",
                                          Arg("recess_period", std::string("10ms"))),
        make_condition<CountCondition>("source1-count", 50));

    auto middle1 = make_operator<AsyncCyclicMiddleDualOutputOp>(
        "CyclicMiddle1",
        make_condition<PeriodicCondition>("middle1-periodic",
                                          Arg("recess_period", std::string("15ms"))),
        make_condition<CountCondition>("middle1-count", 20));

    auto feedback1 = make_operator<AsyncCyclicFeedbackOp>(
        "CyclicFeedback1",
        make_condition<PeriodicCondition>("feedback1-periodic",
                                          Arg("recess_period", std::string("20ms"))),
        make_condition<CountCondition>("feedback1-count", 20));

    // Second cycle: Source2 -> Middle2 -> Source2 (via feedback)
    // Source2 uses dual-input operator to receive from both Middle2 and Feedback1
    auto source2 = make_operator<AsyncCyclicDualInputSourceOp>(
        "CyclicSource2",
        make_condition<PeriodicCondition>("source2-periodic",
                                          Arg("recess_period", std::string("14ms"))),
        make_condition<CountCondition>("source2-count", 50));

    auto middle2 = make_operator<AsyncCyclicMiddleOp>(
        "CyclicMiddle2",
        make_condition<PeriodicCondition>("middle2-periodic",
                                          Arg("recess_period", std::string("18ms"))),
        make_condition<CountCondition>("middle2-count", 20));

    auto feedback2 = make_operator<AsyncCyclicFeedbackOp>(
        "CyclicFeedback2",
        make_condition<PeriodicCondition>("feedback2-periodic",
                                          Arg("recess_period", std::string("22ms"))),
        make_condition<CountCondition>("feedback2-count", 20));

    // First cycle
    add_flow(source1, middle1, {{"out", "in"}}, IOSpec::ConnectorType::kAsyncBuffer);
    add_flow(middle1, source1, {{"out_feedback", "feedback"}}, IOSpec::ConnectorType::kAsyncBuffer);
    add_flow(middle1, feedback1, {{"out_forward", "in"}}, IOSpec::ConnectorType::kAsyncBuffer);

    // Connection from first cycle to second cycle (Feedback1 -> Source2)
    add_flow(
        feedback1, source2, {{"feedback", "feedback_remote"}}, IOSpec::ConnectorType::kAsyncBuffer);

    // Second cycle with local feedback
    add_flow(source2, middle2, {{"out", "in"}}, IOSpec::ConnectorType::kAsyncBuffer);
    add_flow(middle2, feedback2, {{"out", "in"}}, IOSpec::ConnectorType::kAsyncBuffer);
    add_flow(
        feedback2, source2, {{"feedback", "feedback_local"}}, IOSpec::ConnectorType::kAsyncBuffer);
  }
};

/* TestSimpleCycle - Simple 3-node cycle with async buffers
 *
 * CyclicSource ---> CyclicMiddle ---> CyclicFeedback
 *      ^                                      |
 *      |______________________________________|
 *
 * This tests async buffer tracking with a circular dependency.
 * The feedback creates a cycle in the graph topology.
 */
TEST(AsyncBufferCycleTracking, TestSimpleCycle) {
  auto app = make_application<Application>();

  // Source operator with feedback input and output
  auto source_op = app->make_operator<AsyncCyclicSourceOp>(
      "CyclicSource",
      app->make_condition<PeriodicCondition>("source-periodic",
                                             Arg("recess_period", std::string("10ms"))),
      app->make_condition<CountCondition>("source-count", 15));

  // Middle operator
  auto middle_op = app->make_operator<AsyncCyclicMiddleOp>(
      "CyclicMiddle",
      app->make_condition<PeriodicCondition>("middle-periodic",
                                             Arg("recess_period", std::string("20ms"))),
      app->make_condition<CountCondition>("middle-count", 20));

  // Feedback operator that closes the cycle
  auto feedback_op = app->make_operator<AsyncCyclicFeedbackOp>(
      "CyclicFeedback",
      app->make_condition<PeriodicCondition>("feedback-periodic",
                                             Arg("recess_period", std::string("40ms"))),
      app->make_condition<CountCondition>("feedback-count", 25));

  // Create the cyclic flow with async buffers
  app->add_flow(source_op, middle_op, {{"out", "in"}}, IOSpec::ConnectorType::kAsyncBuffer);
  app->add_flow(middle_op, feedback_op, {{"out", "in"}}, IOSpec::ConnectorType::kAsyncBuffer);
  app->add_flow(
      feedback_op, source_op, {{"feedback", "feedback"}}, IOSpec::ConnectorType::kAsyncBuffer);

  auto& tracker = app->track(0, 0, 0);

  // Capture output to check tracking behavior with cycles
  testing::internal::CaptureStdout();

  app->run();

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();

  // Verify that the flow tracking works with cyclic topology
  EXPECT_TRUE(log_output.find("CyclicFeedback_old,CyclicSource") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // CyclicSource -> CyclicMiddle -> CyclicFeedback(_old) -> CyclicSource
  // (_old) is optional. So, we should have at least 1 path. We might get two paths if source
  // operator is executed faster, depending on the system.

  EXPECT_GE(tracker.get_num_paths(), 1)
      << "Expected at least 1 path, but got " << tracker.get_num_paths() << "\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

/* TestTwoNodeCycle - Minimal 2-node cycle
 *
 * OpA <---> OpB (bidirectional async buffer connections)
 *
 * This tests async buffer tracking with the simplest possible cycle.
 */
TEST(AsyncBufferCycleTracking, TestTwoNodeCycle) {
  auto app = make_application<Application>();

  // First operator with input and output
  auto op_a = app->make_operator<AsyncCyclicSourceOp>(
      "CyclicOpA",
      app->make_condition<PeriodicCondition>("opa-periodic",
                                             Arg("recess_period", std::string("6ms"))),
      app->make_condition<CountCondition>("opa-count", 15));

  // Second operator with input and output
  auto op_b = app->make_operator<AsyncCyclicFeedbackOp>(
      "CyclicOpB",
      app->make_condition<PeriodicCondition>("opb-periodic",
                                             Arg("recess_period", std::string("24ms"))),
      app->make_condition<CountCondition>("opb-count", 25));

  // Create 2-node cycle with async buffers
  app->add_flow(op_a, op_b, {{"out", "in"}}, IOSpec::ConnectorType::kAsyncBuffer);
  app->add_flow(op_b, op_a, {{"feedback", "feedback"}}, IOSpec::ConnectorType::kAsyncBuffer);

  auto& tracker = app->track(0, 0, 0);

  // Capture output to check tracking behavior with minimal cycle
  testing::internal::CaptureStdout();

  app->run();

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();

  // Verify that the flow tracking works with 2-node cycle
  EXPECT_TRUE(log_output.find("CyclicOpB_old,CyclicOpA") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  EXPECT_GE(tracker.get_num_paths(), 2)
      << "Expected at least 2 paths, but got " << tracker.get_num_paths() << "\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

// This variation flips the period values of the two operators.
TEST(AsyncBufferCycleTracking, TestTwoNodeCycleVariation) {
  auto app = make_application<Application>();

  // First operator with input and output
  auto op_a = app->make_operator<AsyncCyclicSourceOp>(
      "CyclicOpA",
      app->make_condition<PeriodicCondition>("opa-periodic",
                                             Arg("recess_period", std::string("24ms"))),
      app->make_condition<CountCondition>("opa-count", 15));

  // Second operator with input and output
  auto op_b = app->make_operator<AsyncCyclicFeedbackOp>(
      "CyclicOpB",
      app->make_condition<PeriodicCondition>("opb-periodic",
                                             Arg("recess_period", std::string("6ms"))),
      app->make_condition<CountCondition>("opb-count", 75));

  // Create 2-node cycle with async buffers
  app->add_flow(op_a, op_b, {{"out", "in"}}, IOSpec::ConnectorType::kAsyncBuffer);
  app->add_flow(op_b, op_a, {{"feedback", "feedback"}}, IOSpec::ConnectorType::kAsyncBuffer);

  auto& tracker = app->track(0, 0, 0);

  // Capture output to check tracking behavior with minimal cycle
  testing::internal::CaptureStdout();

  app->run();

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();

  // Verify that the flow tracking works with 2-node cycle
  EXPECT_TRUE(log_output.find("CyclicOpA_old,CyclicOpB") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // We have only one path because first and last operators of a path is same in a cycle, and for
  // first and last operators of a path, we don't distinguish between "old" or original message.
  EXPECT_GE(tracker.get_num_paths(), 1)
      << "Expected at least 2 paths, but got " << tracker.get_num_paths() << "\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

/* TestComplexCycle - Complex graph with multiple cycles using default (greedy) scheduler
 *
 * This tests async buffer tracking with multiple independent cycles using the default scheduler.
 */
TEST(AsyncBufferCycleTracking, TestComplexCycle) {
  auto app = make_application<ComplexCycleApp>();

  auto& tracker = app->track(0, 0, 0);

  app->run();

  // at least two paths are expected
  EXPECT_GE(tracker.get_num_paths(), 2);

  testing::internal::CaptureStdout();

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();

  // check for "_old" printout
  EXPECT_TRUE(log_output.find("_old") != std::string::npos) << "=== LOG ===\n"
                                                            << log_output << "\n===========\n";
}

/* TestComplexCycleEventBased - Complex graph with multiple cycles using event-based scheduler
 *
 * This tests async buffer tracking with multiple independent cycles using the event-based
 * scheduler with 2 worker threads for concurrent execution.
 */
TEST(AsyncBufferCycleTracking, TestComplexCycleEventBased) {
  auto app = make_application<ComplexCycleApp>();

  // Configure event-based scheduler with 3 worker threads
  app->scheduler(
      app->make_scheduler<EventBasedScheduler>("event-scheduler", Arg("worker_thread_number", 3L)));

  auto& tracker = app->track(0, 0, 0);

  // Capture output to check tracking behavior with complex cycles
  testing::internal::CaptureStdout();

  app->run();

  tracker.print();
  std::string log_output = testing::internal::GetCapturedStdout();

  EXPECT_GE(tracker.get_num_paths(), 3);

  EXPECT_TRUE(log_output.find("_old") != std::string::npos) << "=== LOG ===\n"
                                                            << log_output << "\n===========\n";
}

}  // namespace holoscan
