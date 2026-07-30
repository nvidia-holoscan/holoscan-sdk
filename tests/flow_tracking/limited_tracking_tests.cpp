/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <string>

#include "sample_test_graphs.hpp"

namespace holoscan {

TEST(Graphs, TestThreePathsOneRootOneLeaf) {
  auto app1 = make_application<ThreePathsOneRootOneLeaf>();
  auto& tracker1 = app1->track(0, 0, 0, true);  // enabled limited tracking

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStdout();

  app1->run();

  tracker1.print();

  std::string log_output1 = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(log_output1.find("root,leaf") != std::string::npos)
      << "=== LOG ===\n"
      << log_output1 << "\n===========\n";

  EXPECT_TRUE(log_output1.find("Number of messages: 15") != std::string::npos)
      << "=== LOG ===\n"
      << log_output1 << "\n===========\n";

  // Try with default option below
  auto app2 = make_application<ThreePathsOneRootOneLeaf>();
  auto& tracker2 = app2->track(0, 0, 0);  // default option

  testing::internal::CaptureStdout();

  app2->run();

  tracker2.print();

  std::string log_output2 = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(log_output2.find("root,middle1,middle4,leaf") != std::string::npos)
      << "=== LOG ===\n"
      << log_output2 << "\n===========\n";

  EXPECT_TRUE(log_output2.find("root,middle2,middle4,leaf") != std::string::npos)
      << "=== LOG ===\n"
      << log_output2 << "\n===========\n";

  EXPECT_TRUE(log_output2.find("root,middle3,middle4,leaf") != std::string::npos)
      << "=== LOG ===\n"
      << log_output2 << "\n===========\n";
}

}  // namespace holoscan
