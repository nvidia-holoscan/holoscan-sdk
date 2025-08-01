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

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <holoscan/holoscan.hpp>

namespace holoscan {

// Test operators for operator status tracking
class TestSourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TestSourceOp)

  TestSourceOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.output<int>("out");
    spec.param(max_count_, "max_count", "Maximum Count", "Maximum number of times to emit data", 3);
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("TestSourceOp::compute: count_ = {}", count_);
    if (count_ < max_count_.get()) {
      op_output.emit(count_, "out");
    }
    count_++;
  }

 private:
  Parameter<int> max_count_;
  int count_ = 0;
};

class TestProcessorOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TestProcessorOp)

  TestProcessorOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.output<int>("out");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("TestProcessorOp::compute");
    auto data = op_input.receive<int>("in").value();
    int result = data * 2;
    op_output.emit(result, "out");
  }
};

class TestConsumerOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TestConsumerOp)

  TestConsumerOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<int>("in");
    spec.param(
        stop_after_, "stop_after", "Stop After", "Stop after receiving this many messages", 2);
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("TestConsumerOp::compute");
    auto data = op_input.receive<int>("in").value();
    received_values_.push_back(data);

    // Stop execution after receiving a certain number of values
    if (received_values_.size() >= static_cast<size_t>(stop_after_.get())) {
      stop_execution();
    }
  }

  const std::vector<int>& received_values() const { return received_values_; }

 private:
  Parameter<int> stop_after_;
  std::vector<int> received_values_;
};

class TestMonitorOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TestMonitorOp)

  TestMonitorOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(monitored_operators_,
               "monitored_operators",
               "Monitored Operators",
               "Names of operators to monitor",
               {});
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("TestMonitorOp::compute");

    // Check the status of all monitored operators
    bool is_pipeline_idle = true;
    for (const auto& op_name : monitored_operators_.get()) {
      auto maybe_status = context.get_operator_status(op_name);
      if (maybe_status) {
        operator_statuses_[op_name] = maybe_status.value();
        // If any operator is not idle, the pipeline is not idle
        if (maybe_status.value() != OperatorStatus::kIdle) {
          is_pipeline_idle = false;
        }
      } else {
        is_pipeline_idle = false;
      }
    }

    // If all operators are idle, increment the idle count
    if (is_pipeline_idle) {
      idle_count_++;
      // Stop the application if all operators have been idle for a while
      if (idle_count_ >= 3) {
        HOLOSCAN_LOG_INFO(
            "TestMonitorOp::compute: All operators have been idle for {} iterations. Stopping "
            "execution",
            idle_count_);
        fragment()->stop_execution();
      }
    } else {
      // Reset the idle count if any operator is not idle
      idle_count_ = 0;
    }
  }

  const std::unordered_map<std::string, OperatorStatus>& operator_statuses() const {
    return operator_statuses_;
  }

 private:
  Parameter<std::vector<std::string>> monitored_operators_;
  std::unordered_map<std::string, OperatorStatus> operator_statuses_;
  int idle_count_ = 0;
};

class OperatorStatusApp : public Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Create operators
    source_ = make_operator<TestSourceOp>(
        "source", make_condition<CountCondition>(10), Arg("max_count", 3));
    processor_ = make_operator<TestProcessorOp>("processor");
    consumer_ = make_operator<TestConsumerOp>("consumer", Arg("stop_after", 2));

    // Create monitor with its own scheduling condition
    monitor_ = make_operator<TestMonitorOp>(
        "monitor",
        make_condition<PeriodicCondition>(std::chrono::milliseconds(10)),
        Arg("monitored_operators", std::vector<std::string>{"source", "processor", "consumer"}));

    // Define the workflow
    add_flow(source_, processor_, {{"out", "in"}});
    add_flow(processor_, consumer_, {{"out", "in"}});

    // Add monitor (not connected to other operators)
    add_operator(monitor_);
  }

  std::shared_ptr<TestSourceOp> source_;
  std::shared_ptr<TestProcessorOp> processor_;
  std::shared_ptr<TestConsumerOp> consumer_;
  std::shared_ptr<TestMonitorOp> monitor_;
};

TEST(OperatorStatus, TestOperatorStatusTracking) {
  auto app = make_application<OperatorStatusApp>();
  app->scheduler(app->make_scheduler<EventBasedScheduler>(
      "eventbased", Arg("worker_thread_number", static_cast<int64_t>(2))));

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";

  // Verify that the consumer received the expected values
  auto consumer = app->consumer_;
  const auto& received_values = consumer->received_values();

  // Should have received 2 values (0*2 and 1*2)
  ASSERT_EQ(received_values.size(), 2);
  EXPECT_EQ(received_values[0], 0);
  EXPECT_EQ(received_values[1], 2);

  // Verify that the monitor collected operator statuses
  auto monitor = app->monitor_;
  const auto& statuses = monitor->operator_statuses();

  // Should have status entries for all operators
  EXPECT_TRUE(statuses.find("source") != statuses.end());
  EXPECT_TRUE(statuses.find("processor") != statuses.end());
  EXPECT_TRUE(statuses.find("consumer") != statuses.end());
}

TEST(OperatorStatus, TestStopExecution) {
  // This test verifies:
  // 1. The consumer operator stops itself after receiving 2 values using stop_execution()
  // 2. The monitor operator stops the entire application when all operators have been idle for 3
  // iterations
  //    using fragment()->stop_execution()
  auto app = make_application<OperatorStatusApp>();
  app->scheduler(app->make_scheduler<MultiThreadScheduler>(
      "multithread", Arg("worker_thread_number", static_cast<int64_t>(2))));

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";

  // Verify that the consumer stopped after receiving 2 values
  auto consumer = app->consumer_;
  const auto& received_values = consumer->received_values();

  ASSERT_EQ(received_values.size(), 2);

  // Verify that the monitor also stopped itself
  auto monitor = app->monitor_;
  const auto& statuses = monitor->operator_statuses();

  // Monitor should have collected statuses before stopping
  EXPECT_FALSE(statuses.empty());
}

TEST(OperatorStatus, TestFindOperator) {
  auto app = make_application<OperatorStatusApp>();

  // Run the application
  app->scheduler(app->make_scheduler<EventBasedScheduler>(
      "eventbased", Arg("worker_thread_number", static_cast<int64_t>(2))));

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";

  // Get the monitor operator
  auto monitor = app->monitor_;

  // Test that the execution context can find operators by name
  auto execution_context = monitor->execution_context();
  ASSERT_NE(execution_context, nullptr);

  const std::vector<std::string> op_names{"source", "processor", "consumer"};
  for (const auto& op_name : op_names) {
    auto found_op = execution_context->find_operator(op_name);
    EXPECT_NE(found_op, nullptr);
    EXPECT_EQ(found_op->name(), op_name);
  }

  // Test finding non-existent operator
  auto not_found = execution_context->find_operator("non_existent");
  EXPECT_EQ(not_found, nullptr);
}

class AsyncTestOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AsyncTestOp)

  AsyncTestOp() = default;

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    // Access the async_condition method
    auto async_cond = async_condition();
    ASSERT_NE(async_cond, nullptr);

    // Test setting the event state to EVENT_NEVER
    // This is equivalent to calling stop_execution() which internally does:
    // async_condition()->event_state(AsynchronousEventState::EVENT_NEVER);
    async_cond->event_state(AsynchronousEventState::EVENT_NEVER);
  }
};

class AsyncTestApp : public Application {
 public:
  void compose() override {
    op_ = make_operator<AsyncTestOp>("async_op");
    add_operator(op_);
  }

  std::shared_ptr<AsyncTestOp> op_;
};

TEST(OperatorStatus, TestAsyncCondition) {
  // This test verifies that we can access and use the async_condition method
  auto app = make_application<AsyncTestApp>();
  app->scheduler(app->make_scheduler<EventBasedScheduler>(
      "eventbased", Arg("worker_thread_number", static_cast<int64_t>(2))));

  // The test passes if the application runs without errors
  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

class ExecutionContextTestOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ExecutionContextTestOp)
  ExecutionContextTestOp() = default;

  void initialize() override {
    Operator::initialize();

    // Test execution_context() in initialize()
    auto context = execution_context();
    ASSERT_NE(context, nullptr);

    // Store the context for later verification
    initialize_context_ = context;

    // Test find_operator() in initialize() - should return nullptr since we're not connected yet
    auto self = context->find_operator(name());
    ASSERT_NE(self, nullptr);

    HOLOSCAN_LOG_INFO("ExecutionContextTestOp::initialize: execution_context is valid");
  }

  void start() override {
    Operator::start();

    // Test execution_context() in start()
    auto context = execution_context();
    ASSERT_NE(context, nullptr);

    // Store the context for later verification
    start_context_ = context;

    // Test find_operator() in start()
    auto self = context->find_operator(name());
    ASSERT_NE(self, nullptr);

    HOLOSCAN_LOG_INFO("ExecutionContextTestOp::start: execution_context is valid");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    // Test execution_context() in compute()
    auto exec_context = execution_context();
    ASSERT_NE(exec_context, nullptr);

    // Store the context for later verification - make a copy to ensure it persists
    compute_context_ = exec_context;

    // Test find_operator() in compute()
    auto self = exec_context->find_operator(name());
    ASSERT_NE(self, nullptr);

    HOLOSCAN_LOG_INFO("ExecutionContextTestOp::compute: execution_context is valid");

    // Stop execution after storing the context
    stop_execution();
  }

  void stop() override {
    // Test execution_context() in stop()
    auto context = execution_context();
    ASSERT_NE(context, nullptr);

    // Store the context for later verification
    stop_context_ = context;

    // Test find_operator() in stop()
    auto self = context->find_operator(name());
    ASSERT_NE(self, nullptr);

    HOLOSCAN_LOG_INFO("ExecutionContextTestOp::stop: execution_context is valid");

    Operator::stop();
  }

  // Getters to verify contexts after the application has run
  std::shared_ptr<ExecutionContext> initialize_context() const { return initialize_context_; }
  std::shared_ptr<ExecutionContext> start_context() const { return start_context_; }
  std::shared_ptr<ExecutionContext> compute_context() const { return compute_context_; }
  std::shared_ptr<ExecutionContext> stop_context() const { return stop_context_; }

 private:
  std::shared_ptr<ExecutionContext> initialize_context_;
  std::shared_ptr<ExecutionContext> start_context_;
  std::shared_ptr<ExecutionContext> compute_context_;
  std::shared_ptr<ExecutionContext> stop_context_;
};

class ExecutionContextTestApp : public Application {
 public:
  void compose() override {
    op_ = make_operator<ExecutionContextTestOp>("context_test_op");
    add_operator(op_);
  }

  std::shared_ptr<ExecutionContextTestOp> op_;
};

TEST(OperatorStatus, TestExecutionContextInLifecycleMethods) {
  // This test verifies that execution_context() returns a non-null pointer and that
  // calling execution_context() and find_operator() methods work in initialize(), start(),
  // stop(), and compute() methods

  auto app = make_application<ExecutionContextTestApp>();
  app->scheduler(app->make_scheduler<EventBasedScheduler>(
      "eventbased", Arg("worker_thread_number", static_cast<int64_t>(2))));

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";

  // Verify that execution_context() returned non-null in all lifecycle methods
  auto op = app->op_;
  EXPECT_NE(op->initialize_context(), nullptr);
  EXPECT_NE(op->start_context(), nullptr);
  EXPECT_NE(op->compute_context(), nullptr);
  EXPECT_NE(op->stop_context(), nullptr);
}

class SourceOp : public Operator {
 public:
  SourceOp() = default;

  void setup(OperatorSpec& spec) override { spec.output<int>("out"); }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    op_output.emit(42, "out");
    stop_execution();
  }
};

class SinkOp : public Operator {
 public:
  SinkOp() = default;

  void setup(OperatorSpec& spec) override { spec.input<int>("in"); }

  void initialize() override {
    Operator::initialize();

    // Try to find the source operator during initialization
    auto context = execution_context();
    ASSERT_NE(context, nullptr);

    auto source_op = context->find_operator("source_op");
    // Store the result for later verification
    found_source_in_initialize_ = (source_op != nullptr);

    HOLOSCAN_LOG_INFO("SinkOp::initialize: found_source = {}", found_source_in_initialize_);
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    // Try to find the source operator during compute
    auto exec_context = execution_context();
    ASSERT_NE(exec_context, nullptr);

    auto source_op = exec_context->find_operator("source_op");
    // Store the result for later verification
    found_source_in_compute_ = (source_op != nullptr);

    HOLOSCAN_LOG_INFO("SinkOp::compute: found_source = {}", found_source_in_compute_);

    // Receive the value from the source
    auto value = op_input.receive<int>("in").value();
    EXPECT_EQ(value, 42);
  }

  // Getters to verify results after the application has run
  bool found_source_in_initialize() const { return found_source_in_initialize_; }
  bool found_source_in_compute() const { return found_source_in_compute_; }

 private:
  bool found_source_in_initialize_ = false;
  bool found_source_in_compute_ = false;
};

class FindOperatorsApp : public Application {
 public:
  void compose() override {
    source_ = make_operator<SourceOp>("source_op");
    sink_ = make_operator<SinkOp>("sink_op");

    // Connect the operators
    add_flow(source_, sink_, {{"out", "in"}});
  }

  std::shared_ptr<SourceOp> source_;
  std::shared_ptr<SinkOp> sink_;
};

TEST(OperatorStatus, TestFindOtherOperators) {
  // This test verifies that execution_context()->find_operator() can be used to find
  // other operators in the application

  auto app = make_application<FindOperatorsApp>();
  app->scheduler(app->make_scheduler<EventBasedScheduler>(
      "eventbased", Arg("worker_thread_number", static_cast<int64_t>(2))));

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";

  // Verify that the sink operator could find the source operator
  auto sink = app->sink_;
  EXPECT_TRUE(sink->found_source_in_initialize());
  EXPECT_TRUE(sink->found_source_in_compute());
}

}  // namespace holoscan
