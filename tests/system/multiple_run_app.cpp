/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <string>

#include "holoscan/holoscan.hpp"
#include "holoscan/operators/ping_rx/ping_rx.hpp"
#include "holoscan/operators/ping_tx/ping_tx.hpp"

class MyPingApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    // Define the tx and rx operators, allowing the tx operator to execute 10 times
    auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(10));
    auto rx = make_operator<ops::PingRxOp>("rx");

    // Define the workflow:  tx -> rx
    add_flow(tx, rx);
  }
};

TEST(MultipleRunApp, TestMultipleRunApp) {
  constexpr int num_runs = 100;

  auto app = holoscan::make_application<MyPingApp>();

  // Capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  // Run the app multiple times
  for (int i = 0; i < num_runs; i++) {
    app->run();
  }

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}

TEST(MultipleRunApp, TestMultipleRunAppAsync) {
  constexpr int num_runs = 100;

  auto app = holoscan::make_application<MyPingApp>();

  // Capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  // Run the app multiple times
  for (int i = 0; i < num_runs; i++) {
    auto future = app->run_async();
    future.get();
  }

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("error") == std::string::npos) << "=== LOG ===\n"
                                                             << log_output << "\n===========\n";
}
