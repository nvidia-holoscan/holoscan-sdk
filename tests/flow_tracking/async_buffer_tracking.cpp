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
// Async Buffer Tracking Operators with Periodic Conditions
///////////////////////////////////////////////////////////////////////////////

class AsyncPeriodicTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AsyncPeriodicTxOp)

  AsyncPeriodicTxOp() = default;

  void setup(OperatorSpec& spec) override { spec.output<gxf::Entity>("out"); }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto out_message = gxf::Entity::New(&context);
    op_output.emit(out_message);
    HOLOSCAN_LOG_INFO("{} transmitting message {}", name(), count_++);
  }

 private:
  int count_ = 1;
};

class AsyncPeriodicRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AsyncPeriodicRxOp)

  AsyncPeriodicRxOp() = default;

  void setup(OperatorSpec& spec) override { spec.input<gxf::Entity>("in"); }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto in_message = op_input.receive<gxf::Entity>("in");
    if (in_message) {
      HOLOSCAN_LOG_INFO("{} receiving message {}", name(), count_++);
    }
  }

 private:
  int count_ = 1;
};

class AsyncPeriodicMiddleOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AsyncPeriodicMiddleOp)

  AsyncPeriodicMiddleOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<gxf::Entity>("in");
    spec.output<gxf::Entity>("out");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto in_message = op_input.receive<gxf::Entity>("in");
    if (in_message) {
      op_output.emit(in_message.value());
      HOLOSCAN_LOG_INFO("{} forwarding message {}", name(), count_++);
    }
  }

 private:
  int count_ = 1;
};

///////////////////////////////////////////////////////////////////////////////
// Async Buffer Tracking Applications with Different Periods
///////////////////////////////////////////////////////////////////////////////

/* AsyncBufferPeriodicTrackingApp - Tests async buffer with different periodic rates
 *
 * AsyncTx(10ms) ---> AsyncMiddle(20ms) ---> AsyncRx(30ms)
 *
 * This tests the key behavior: when receiver reads slower than transmitter writes,
 * the same message UID may be received multiple times. The fix ensures we don't
 * deannotate the same message repeatedly.
 */
class AsyncBufferPeriodicTrackingApp : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;

    // Transmitter runs fastest (10ms period)
    auto tx_op = make_operator<AsyncPeriodicTxOp>(
        "AsyncTx",
        make_condition<PeriodicCondition>("tx-periodic", Arg("recess_period", std::string("10ms"))),
        make_condition<CountCondition>("tx-count", 10));

    // Middle operator runs at medium speed (20ms period)
    auto middle_op = make_operator<AsyncPeriodicMiddleOp>(
        "AsyncMiddle",
        make_condition<PeriodicCondition>("middle-periodic",
                                          Arg("recess_period", std::string("20ms"))),
        make_condition<CountCondition>("middle-count", 20));

    // Receiver runs slowest (30ms period) - will read same message multiple times
    auto rx_op = make_operator<AsyncPeriodicRxOp>(
        "AsyncRx",
        make_condition<PeriodicCondition>("rx-periodic", Arg("recess_period", std::string("30ms"))),
        make_condition<CountCondition>("rx-count", 30));

    // Use async buffer connectors
    add_flow(tx_op, middle_op, {{"out", "in"}}, IOSpec::ConnectorType::kAsyncBuffer);
    add_flow(middle_op, rx_op, {{"out", "in"}}, IOSpec::ConnectorType::kAsyncBuffer);
  }
};

/* AsyncBufferPeriodicMultiPathApp - Multiple paths with different rates
 *
 * Path 1: AsyncTx1(15ms) ---> AsyncMiddle1(25ms) ---> AsyncRx1(35ms)
 * Path 2: AsyncTx2(12ms) ---> AsyncMiddle2(22ms) ---> AsyncRx2(32ms)
 */
class AsyncBufferPeriodicMultiPathApp : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;

    // First path - 15ms, 25ms, 35ms periods
    auto tx_op1 = make_operator<AsyncPeriodicTxOp>(
        "AsyncTx1",
        make_condition<PeriodicCondition>("tx1-periodic",
                                          Arg("recess_period", std::string("15ms"))),
        make_condition<CountCondition>("tx1-count", 10));

    auto middle_op1 = make_operator<AsyncPeriodicMiddleOp>(
        "AsyncMiddle1",
        make_condition<PeriodicCondition>("middle1-periodic",
                                          Arg("recess_period", std::string("25ms"))),
        make_condition<CountCondition>("middle1-count", 15));

    auto rx_op1 = make_operator<AsyncPeriodicRxOp>(
        "AsyncRx1",
        make_condition<PeriodicCondition>("rx1-periodic",
                                          Arg("recess_period", std::string("35ms"))),
        make_condition<CountCondition>("rx1-count", 20));

    // Second path - 12ms, 22ms, 32ms periods
    auto tx_op2 = make_operator<AsyncPeriodicTxOp>(
        "AsyncTx2",
        make_condition<PeriodicCondition>("tx2-periodic",
                                          Arg("recess_period", std::string("12ms"))),
        make_condition<CountCondition>("tx2-count", 10));

    auto middle_op2 = make_operator<AsyncPeriodicMiddleOp>(
        "AsyncMiddle2",
        make_condition<PeriodicCondition>("middle2-periodic",
                                          Arg("recess_period", std::string("22ms"))),
        make_condition<CountCondition>("middle2-count", 15));

    auto rx_op2 = make_operator<AsyncPeriodicRxOp>(
        "AsyncRx2",
        make_condition<PeriodicCondition>("rx2-periodic",
                                          Arg("recess_period", std::string("32ms"))),
        make_condition<CountCondition>("rx2-count", 20));

    // Use async buffer connectors for all connections
    add_flow(tx_op1, middle_op1, {{"out", "in"}}, IOSpec::ConnectorType::kAsyncBuffer);
    add_flow(middle_op1, rx_op1, {{"out", "in"}}, IOSpec::ConnectorType::kAsyncBuffer);
    add_flow(tx_op2, middle_op2, {{"out", "in"}}, IOSpec::ConnectorType::kAsyncBuffer);
    add_flow(middle_op2, rx_op2, {{"out", "in"}}, IOSpec::ConnectorType::kAsyncBuffer);
  }
};

}  // namespace

TEST(AsyncBufferTracking, TestAsyncBufferPeriodicTrackingGreedyScheduler) {
  auto app = make_application<AsyncBufferPeriodicTrackingApp>();

  // By default, greedy scheduled will be used

  auto& tracker = app->track(0, 0, 0);

  // capture output so that we can check for duplicate UID handling
  testing::internal::CaptureStdout();

  app->run();

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();

  // Verify that the flow tracking captured the async buffer path
  EXPECT_TRUE(log_output.find("AsyncTx,AsyncMiddle,AsyncRx") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify that Data Flow Tracking Results are present
  EXPECT_TRUE(log_output.find("Data Flow Tracking Results") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // CRITICAL TEST: Verify that "_old" suffix appears on the upstream operator (AsyncMiddle)
  // When AsyncRx reads the same message UID multiple times (due to slower period),
  // the last operator in the path (AsyncMiddle) before the receiver gets "_old" appended
  // Expected path: "AsyncTx,AsyncMiddle_old,AsyncRx"
  EXPECT_TRUE(log_output.find("AsyncMiddle_old") != std::string::npos)
      << "Expected to find 'AsyncMiddle_old' in path for duplicate message reads\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(AsyncBufferTracking, TestAsyncBufferPeriodicTrackingEventBasedScheduler) {
  auto app = make_application<AsyncBufferPeriodicTrackingApp>();

  // Use event-based scheduler with 3 threads (one per operator: AsyncTx, AsyncMiddle, AsyncRx)
  app->scheduler(
      app->make_scheduler<EventBasedScheduler>("event_scheduler", Arg("worker_thread_number", 3L)));

  auto& tracker = app->track(0, 0, 0);

  // capture output so that we can check for duplicate UID handling
  testing::internal::CaptureStdout();

  app->run();

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();

  // Verify that the flow tracking captured the async buffer path
  EXPECT_TRUE(log_output.find("AsyncTx,AsyncMiddle,AsyncRx") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify that Data Flow Tracking Results are present
  EXPECT_TRUE(log_output.find("Data Flow Tracking Results") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // CRITICAL TEST: Verify that "_old" suffix appears on the upstream operator (AsyncMiddle)
  // When AsyncRx reads the same message UID multiple times (due to slower period),
  // the last operator in the path (AsyncMiddle) before the receiver gets "_old" appended
  // Expected path: "AsyncTx,AsyncMiddle_old,AsyncRx"
  EXPECT_TRUE(log_output.find("AsyncMiddle_old") != std::string::npos)
      << "Expected to find 'AsyncMiddle_old' in path for duplicate message reads\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(AsyncBufferTracking, TestAsyncBufferMultiPathPeriodicGreedyScheduler) {
  auto app = make_application<AsyncBufferPeriodicMultiPathApp>();

  // By default, greedy scheduled will be used

  auto& tracker = app->track(0, 0, 0);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStdout();

  app->run();

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();

  // Verify that the flow tracking captured both async buffer paths
  EXPECT_TRUE(log_output.find("AsyncTx1,AsyncMiddle1,AsyncRx1") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  EXPECT_TRUE(log_output.find("AsyncTx2,AsyncMiddle2,AsyncRx2") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // CRITICAL TEST: Verify that "_old" suffix appears on upstream operators in both paths
  // Path 1: AsyncTx1,AsyncMiddle1_old,AsyncRx1 (AsyncMiddle1 gets _old when AsyncRx1 rereads)
  // Path 2: AsyncTx2,AsyncMiddle2_old,AsyncRx2 (AsyncMiddle2 gets _old when AsyncRx2 rereads)
  EXPECT_TRUE(log_output.find("AsyncMiddle1_old") != std::string::npos ||
              log_output.find("AsyncMiddle2_old") != std::string::npos)
      << "Expected to find 'AsyncMiddle1_old' or 'AsyncMiddle2_old' in paths\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(AsyncBufferTracking, TestAsyncBufferMultiPathPeriodicEventBasedScheduler) {
  auto app = make_application<AsyncBufferPeriodicMultiPathApp>();

  // Use event-based scheduler with 6 threads (one per operator across both paths)
  app->scheduler(
      app->make_scheduler<EventBasedScheduler>("event_scheduler", Arg("worker_thread_number", 6L)));

  auto& tracker = app->track(0, 0, 0);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStdout();

  app->run();

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();

  // Verify that the flow tracking captured both async buffer paths
  EXPECT_TRUE(log_output.find("AsyncTx1,AsyncMiddle1,AsyncRx1") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  EXPECT_TRUE(log_output.find("AsyncTx2,AsyncMiddle2,AsyncRx2") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // CRITICAL TEST: Verify that "_old" suffix appears on upstream operators in both paths
  // Path 1: AsyncTx1,AsyncMiddle1_old,AsyncRx1 (AsyncMiddle1 gets _old when AsyncRx1 rereads)
  // Path 2: AsyncTx2,AsyncMiddle2_old,AsyncRx2 (AsyncMiddle2 gets _old when AsyncRx2 rereads)
  EXPECT_TRUE(log_output.find("AsyncMiddle1_old") != std::string::npos ||
              log_output.find("AsyncMiddle2_old") != std::string::npos)
      << "Expected to find 'AsyncMiddle1_old' or 'AsyncMiddle2_old' in paths\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

}  // namespace holoscan
