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

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "sample_test_graphs.hpp"

namespace holoscan {

TEST(Graphs, TestFlowTrackingForCycleWithSource) {
  auto app = make_application<CycleWithSourceApp>();
  auto& tracker = app->track(0, 0, 0);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStdout();

  app->run();

  EXPECT_EQ(app->graph().has_cycle().size(), 1);

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(log_output.find("OneOut,TwoInOneOut,OneInOneOut,TwoInOneOut") != std::string::npos)
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
  EXPECT_TRUE(log_output.find("middle,end,middle") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("middle,start,middle") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // The following two paths have only two messages even though 5 messages are sent from the start
  // This is because no more than 2 messages could travel the following two loops.
  // The origin of the rest of the messages become middle node and they travel in the above two
  // loops.
  EXPECT_TRUE(log_output.find("start,middle,end,middle") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("start,middle,start") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

}  // namespace holoscan
