/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <unistd.h>

#include <string>
#include <vector>

#include "sample_test_graphs.hpp"

namespace holoscan {

TEST(Graphs, TestFlowTrackingForCycleWithSource) {
  auto app = make_application<CycleWithSourceApp>();
  // Skip 1 message at the beginning so that one path does not have any messages tracked
  auto& tracker = app->track(1, 0, 0);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStdout();

  app->run();

  EXPECT_EQ(app->graph().has_cycle().size(), 1);

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(log_output.find("OneOut,TwoInOneOut,OneInOneOut,TwoInOneOut") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("No messages tracked for this path.") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("TwoInOneOut,OneInOneOut,TwoInOneOut") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(Graphs, TestFlowTrackingForMiddleCycle) {
  auto app = make_application<MiddleCycleApp>();
  auto& tracker = app->track(0, 0, 0);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStdout();

  app->run();

  EXPECT_EQ(app->graph().has_cycle().size(), 1);

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(log_output.find("OneOut,TwoInOneOut,OneInOneOut,PingRx") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("OneOut,TwoInOneOut,OneInOneOut,TwoInOneOut") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("TwoInOneOut,OneInOneOut,PingRx") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("TwoInOneOut,OneInOneOut,TwoInOneOut") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(Graphs, TestFlowTrackingForCycleWithLeaf) {
  auto app = make_application<CycleWithLeaf>();
  auto& tracker = app->track(0, 0, 0);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStdout();

  app->run();

  EXPECT_EQ(app->graph().has_cycle().size(), 1);

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(log_output.find("root,middle,leaf") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("root,middle,root") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(Graphs, TestFlowTrackingForTwoRootsOneCycle) {
  auto app = make_application<TwoRootsOneCycle>();
  auto& tracker = app->track(0, 0, 0);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStdout();

  app->run();

  EXPECT_EQ(app->graph().has_cycle().size(), 1);

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(log_output.find("middle2,last,middle2") != std::string::npos);
  EXPECT_TRUE(log_output.find("root1,middle1,middle2,last,middle2") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("root2,middle2,last,middle2") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(Graphs, TestFlowTrackingForTwoCyclesVariant1) {
  auto app = make_application<TwoCyclesVariant1>();
  auto& tracker = app->track(0, 0, 0);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStdout();

  app->run();

  EXPECT_EQ(app->graph().has_cycle().size(), 2);

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(log_output.find("middle,end,middle") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("middle,start,middle") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(Graphs, TestFlowTrackingForTwoCyclesVariant2) {
  auto app = make_application<TwoCyclesVariant2>();
  auto& tracker = app->track(0, 0, 0);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStdout();

  app->run();

  EXPECT_EQ(app->graph().has_cycle().size(), 2);

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();

  // this app has two cycles, and middle node is not triggered first, as start has an optional input
  // port.
  EXPECT_TRUE(log_output.find("start,middle,end,middle") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("start,middle,start") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(Graphs, DISABLED_TestSingleCycleTwoRoots) {
  using namespace holoscan;
  Fragment fragment;
  fragment.name("test_fragment");
  auto root1 = fragment.make_operator<OneOptionalInOneOutOp>(
      "root1", fragment.make_condition<CountCondition>("root1-count", 5));
  auto root2 = fragment.make_operator<OneOptionalInOneOutOp>(
      "root2", fragment.make_condition<CountCondition>("root2-count", 5));
  auto middle = fragment.make_operator<OneInOneOutOp>("middle");

  fragment.add_flow(root1, middle);
  fragment.add_flow(middle, root2);
  fragment.add_flow(root2, root1);

  auto& tracker = fragment.track(0, 0, 0);

  fragment.scheduler(fragment.make_scheduler<holoscan::EventBasedScheduler>(
      "event-based-scheduler", holoscan::Arg{"worker_thread_number", static_cast<int64_t>(2)}));

  testing::internal::CaptureStdout();

  fragment.run();

  // there will be two paths in most cases:
  // 1. root1 -> middle -> root2 -> root1
  // 2. root2 -> root1 -> middle -> root2

  // Check number of available CPUs
  auto available_cpus = sysconf(_SC_NPROCESSORS_ONLN);

  if (available_cpus < 2) {
    // If there is only one CPU, then two roots cannot normally run truly concurrently.
    // Therefore, we expect only one path.
    // However, there are certain cases like preempted execution or hardware
    // multithreading, when two roots could run concurrently. For those cases, there might
    // be two paths, instead of expected 1.
    // So, we add two conservative test case below: >=1 and <= 2 i.e., either 1 or 2
    EXPECT_GE(tracker.get_num_paths(), 1);
    EXPECT_LE(tracker.get_num_paths(), 2);
    return;
  }
  EXPECT_EQ(tracker.get_num_paths(), 2);

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(
      log_output.find(
          "test_fragment.root1,test_fragment.middle,test_fragment.root2,test_fragment.root1") !=
      std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(
      log_output.find(
          "test_fragment.root2,test_fragment.root1,test_fragment.middle,test_fragment.root2") !=
      std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST(Graphs, DISABLED_TestSingleCycleTwoRootsOnePath) {
  using namespace holoscan;
  Fragment fragment;
  fragment.name("test_fragment");

  // In this case, we run root2 just 2 times. Therefore, the cyclic path starting from
  // root2 will not be able to form a full cycle at runtime. root2 will run the
  // following two times:
  // 1. independently as a root operator.
  // 2. as part of the root1's cyclic path.
  // So, this application will only have one (cyclic) path.
  auto root1 = fragment.make_operator<OneOptionalInOneOutOp>(
      "root1", fragment.make_condition<CountCondition>("root1-count", 10));
  auto root2 = fragment.make_operator<OneOptionalInOneOutOp>(
      "root2", fragment.make_condition<CountCondition>("root2-count", 2));
  auto middle = fragment.make_operator<OneInOneOutOp>("middle");

  fragment.add_flow(root1, middle);
  fragment.add_flow(middle, root2);
  fragment.add_flow(root2, root1);

  fragment.scheduler(fragment.make_scheduler<holoscan::EventBasedScheduler>(
      "event-based-scheduler", holoscan::Arg{"worker_thread_number", static_cast<int64_t>(2)}));

  auto& tracker = fragment.track(0, 0, 0);

  fragment.run();

  // it will be either root1->middle->root2->root1 or root2->root1->middle->root2
  // but it cannot be both (i.e., 2 paths), as root2 can only run twice
  EXPECT_EQ(tracker.get_num_paths(), 1);
}

}  // namespace holoscan
