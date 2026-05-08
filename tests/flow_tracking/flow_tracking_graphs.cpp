/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
