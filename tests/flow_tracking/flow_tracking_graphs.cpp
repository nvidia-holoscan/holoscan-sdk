/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <string>

#include "sample_test_graphs.hpp"

namespace holoscan {

TEST(Graphs, TestFlowTrackingForSingleOperatorFragment) {
  auto app = make_application<SingleOperatorApp>();
  auto& tracker = app->track(0, 0, 0);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStdout();

  app->run();

  tracker.print();

  std::string log_output = testing::internal::GetCapturedStdout();

  // The single operator "solo" is both root and leaf, so the tracked path
  // should consist of just that one operator.
  EXPECT_EQ(tracker.get_num_paths(), 1) << "=== LOG ===\n" << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find(": solo") != std::string::npos) << "=== LOG ===\n"
                                                              << log_output << "\n===========\n";
}

}  // namespace holoscan
