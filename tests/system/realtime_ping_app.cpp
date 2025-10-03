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

#include <holoscan/holoscan.hpp>

#include "ping_rx_op.hpp"
#include "ping_tx_op.hpp"

using namespace std::string_literals;

namespace holoscan {

class RealtimePingFirstInFirstOutApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<ops::PingMultiTxOp>("tx", make_condition<CountCondition>(10));
    auto rx = make_operator<ops::PingMultiRxOp>("rx");

    auto pool1 = make_thread_pool("pool1", 0);
    pool1->add_realtime(tx, holoscan::SchedulingPolicy::kFirstInFirstOut, true, {0}, 1);
    pool1->add(rx, true);

    add_flow(tx, rx, {{"out1", "receivers"}, {"out2", "receivers"}});
  }
};

class RealtimePingRoundRobinApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<ops::PingMultiTxOp>("tx", make_condition<CountCondition>(10));
    auto rx = make_operator<ops::PingMultiRxOp>("rx");

    auto pool1 = make_thread_pool("pool1", 0);
    pool1->add_realtime(tx, holoscan::SchedulingPolicy::kRoundRobin, true, {0}, 1);
    pool1->add(rx, true);

    add_flow(tx, rx, {{"out1", "receivers"}, {"out2", "receivers"}});
  }
};

class RealtimePingDeadlineApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<ops::PingMultiTxOp>("tx", make_condition<CountCondition>(10));
    auto rx = make_operator<ops::PingMultiRxOp>("rx");

    auto pool1 = make_thread_pool("pool1", 0);
    pool1->add_realtime(
        tx, holoscan::SchedulingPolicy::kDeadline, true, {0}, 0, 1000000, 10000000, 10000000);
    pool1->add(rx, true);

    add_flow(tx, rx, {{"out1", "receivers"}, {"out2", "receivers"}});
  }
};

TEST(RealtimePingFirstInFirstOutApp, TestRealtimePingFirstInFirstOutApp) {
  auto app = make_application<RealtimePingFirstInFirstOutApp>();

  app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
      "event-based",
      holoscan::Arg("worker_thread_number", static_cast<int64_t>(3)),
      holoscan::Arg("max_duration_ms", static_cast<int64_t>(10000))));

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("value1: 1") != std::string::npos) << "=== LOG ===\n"
                                                                 << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("value2: 100") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(RealtimePingRoundRobinApp, TestRealtimePingRoundRobinApp) {
  auto app = make_application<RealtimePingRoundRobinApp>();

  app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
      "event-based",
      holoscan::Arg("worker_thread_number", static_cast<int64_t>(3)),
      holoscan::Arg("max_duration_ms", static_cast<int64_t>(10000))));

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("value1: 1") != std::string::npos) << "=== LOG ===\n"
                                                                 << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("value2: 100") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(RealtimePingDeadlineApp, TestRealtimePingDeadlineApp) {
  auto app = make_application<RealtimePingDeadlineApp>();

  app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
      "event-based",
      holoscan::Arg("worker_thread_number", static_cast<int64_t>(3)),
      holoscan::Arg("max_duration_ms", static_cast<int64_t>(10000))));

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  if (log_output.find("Failed to set SCHED_DEADLINE policy") != std::string::npos) {
    GTEST_SKIP() << "Deadline test case skipped. Environment has insufficient permissions to set "
                    " the SCHED_DEADLINE policy.";
  }

  EXPECT_TRUE(log_output.find("value1: 1") != std::string::npos) << "=== LOG ===\n"
                                                                 << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("value2: 100") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

}  // namespace holoscan
