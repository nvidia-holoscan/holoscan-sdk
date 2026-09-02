/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

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

}  // namespace holoscan
