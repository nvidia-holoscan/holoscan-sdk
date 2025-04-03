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
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include <holoscan/holoscan.hpp>

// A simple operator that can be controlled by the ControllerOp
class SimpleOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SimpleOp)

  SimpleOp() = default;

  // Constructor with notification callback
  template <typename... ArgsT>
  explicit SimpleOp(std::function<void()> notification_callback, ArgsT&&... args)
      : SimpleOp(std::forward<ArgsT>(args)...) {
    notification_callback_ = notification_callback;
  }

  void initialize() override {
    // Call parent's initialize method
    Operator::initialize();

    // Set the operator to WAIT state to wait for the controller to change the state to READY
    async_condition()->event_state(holoscan::AsynchronousEventState::WAIT);
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override {
    std::cout << "[" << name() << "] Executing compute method" << std::endl;

    // Set async condition to WAIT state to wait for the controller to change the state to READY
    // by setting the event state to EVENT_DONE
    async_condition()->event_state(holoscan::AsynchronousEventState::WAIT);

    // Call the notification callback so notifying the controller that the execution of the
    // operator is done
    notification_callback_();
  }

 private:
  std::function<void()> notification_callback_;  ///< callback function to notify the controller
};

// A controller operator that manages the execution of other operators
class ControllerOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ControllerOp)

  ControllerOp() = default;

  void initialize() override {
    // Call parent's initialize method
    Operator::initialize();

    // Get all operators in the application except this one
    auto all_ops = fragment()->graph().get_nodes();
    for (auto& op : all_ops) {
      if (op->name() != name()) { op_map_[op->name()] = op; }
    }
  }

  /**
   * @brief Wait for the controller to be started.
   *
   * Blocks the calling thread until the controller's start method has been called
   * and the is_started_ flag is set to true.
   */
  void wait_for_start() {
    std::unique_lock<std::mutex> lock(mutex_);
    start_cv_.wait(lock, [this]() { return is_started_; });
  }

  void start() override {
    // Set this operator's async condition to EVENT_WAITING to prevent its compute method
    // from being called until the controller sets the event state to EVENT_DONE.
    // Note that at least one operator's event state (in this case, the controller) needs to be set
    // to EVENT_WAITING instead of WAIT to prevent the application from being terminated at start by
    // the deadlock detector.
    async_condition()->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);

    is_started_ = true;
    start_cv_.notify_all();
  }

  void stop() override { std::cout << "[" << name() << "] Stopping controller" << std::endl; }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override {
    // If the event state is set to EVENT_DONE, the compute method will be called.
    // Setting the event state to EVENT_DONE triggers the scheduler,
    // which updates the operator's scheduling condition type to READY, leading to the execution of
    // the compute method.
    std::cout << "[" << name() << "] Stopping controller execution" << std::endl;
    // Stop this operator's execution (event state is set to EVENT_NEVER)
    stop_execution();
  }

  // Method to be called by SimpleOp operators to notify this controller
  void notify_callback() {
    is_operator_executed_ = true;
    operator_execution_cv_.notify_all();
  }

  void execute_operator(const std::string& op_name) {
    auto op_it = op_map_.find(op_name);
    if (op_it == op_map_.end()) {
      std::cout << "[" << name() << "] Operator " << op_name << " not found" << std::endl;
      return;
    }

    auto op = op_it->second;
    // Set the operator to EVENT_DONE to signal the scheduler that the operator is ready to execute
    op->async_condition()->event_state(holoscan::AsynchronousEventState::EVENT_DONE);

    // Wait for a notification from the operator
    {
      std::unique_lock<std::mutex> lock(mutex_);
      operator_execution_cv_.wait(lock, [this]() { return is_operator_executed_; });
      is_operator_executed_ = false;
    }
  }

  void shutdown() {
    // Set all operators' event states to EVENT_NEVER to change their scheduling condition type to
    // NEVER.
    std::cout << "[" << name() << "] Shutting down controller" << std::endl;

    for (auto& [op_name, op] : op_map_) {
      op->async_condition()->event_state(holoscan::AsynchronousEventState::EVENT_NEVER);
    }

    // Set the controller to EVENT_DONE to stop the application (this will trigger the scheduler to
    // get notified and make the scheduling condition type of the operator to be READY, executing
    // the compute method)
    async_condition()->event_state(holoscan::AsynchronousEventState::EVENT_DONE);
  }

  std::unordered_map<std::string, std::shared_ptr<holoscan::Operator>> op_map_;
  std::mutex mutex_;
  std::condition_variable operator_execution_cv_;
  std::condition_variable start_cv_;
  bool is_operator_executed_ = false;
  bool is_started_ = false;
};

class AsyncOperatorExecutionControlApp : public holoscan::Application {
 public:
  std::shared_ptr<ControllerOp> get_controller() {
    // Wait for the controller to have a non-null pointer
    while (!controller_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    controller_->wait_for_start();
    return controller_;
  }

  void compose() override {
    using namespace holoscan;

    // Create the controller operator
    controller_ = make_operator<ControllerOp>("controller");

    // Create multiple SimpleOp operators, all using the controller as callback
    auto notify_callback = [controller = controller_]() { controller->notify_callback(); };
    auto op1 = make_operator<SimpleOp>("op1", notify_callback);
    auto op2 = make_operator<SimpleOp>("op2", notify_callback);
    auto op3 = make_operator<SimpleOp>("op3", notify_callback);

    // Add all operators to the application (no flows between them since controller manages
    // execution)
    add_operator(controller_);
    add_operator(op1);
    add_operator(op2);
    add_operator(op3);

    // Print information about the example
    std::cout << "This example demonstrates async operator execution control." << std::endl;
    std::cout << "The controller operator runs in a separate thread and synchronizes" << std::endl;
    std::cout << "the execution of multiple SimpleOp operators using async_condition()."
              << std::endl;
    std::cout << "-------------------------------------------------------------------" << std::endl;
    std::cout << "Key concepts demonstrated:" << std::endl;
    std::cout << "1. Using asynchronous conditions to control operator execution states"
              << std::endl;
    std::cout << "2. External execution control from outside the Holoscan runtime" << std::endl;
    std::cout << "3. Notification callbacks for coordination between operators" << std::endl;
    std::cout << "4. Manual scheduling of operators in a specific order (op3 → op2 → op1)"
              << std::endl;
    std::cout << "5. Graceful shutdown of an application with async operators" << std::endl;
    std::cout << "-------------------------------------------------------------------" << std::endl;
    std::cout << "Execution flow:" << std::endl;
    std::cout << "- All operators start in WAIT state except the controller (EVENT_WAITING)"
              << std::endl;
    std::cout << "- Main thread gets controller and executes operators in sequence" << std::endl;
    std::cout << "- Each operator signals completion via the notification callback" << std::endl;
    std::cout << "- The controller then shuts down the application" << std::endl;
    std::cout << "-------------------------------------------------------------------" << std::endl;
  }

 private:
  std::shared_ptr<ControllerOp> controller_;
};

int main() {
  auto app = holoscan::make_application<AsyncOperatorExecutionControlApp>();
  // This example works with any scheduler (GreedyScheduler, MultiThreadScheduler, etc.)
  // with any number of worker threads.
  // Here we use EventBasedScheduler for demonstration purposes.
  holoscan::ArgList scheduler_args{holoscan::Arg("worker_thread_number", static_cast<int64_t>(3))};
  app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>("EBS", scheduler_args));
  auto future = app->run_async();

  auto controller = app->get_controller();

  // Executing operators outside of the Holoscan runtime
  controller->execute_operator("op3");
  controller->execute_operator("op2");
  controller->execute_operator("op1");

  // Shutting down the application
  controller->shutdown();

  // Waiting for the application to complete
  future.get();

  std::cout << "-------------------------------------------------------------------" << std::endl;
  std::cout << "Application completed. All operators have finished execution." << std::endl;
  return 0;
}
