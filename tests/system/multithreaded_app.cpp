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
#include <utility>

#include "holoscan/holoscan.hpp"
#include <holoscan/operators/bayer_demosaic/bayer_demosaic.hpp>

#include "env_wrapper.hpp"
#include "ping_tensor_rx_op.hpp"
#include "ping_tensor_tx_op.hpp"

constexpr int NUM_RX = 3;
constexpr int NUM_ITER = 100;

class PingMultithreadApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<ops::PingTensorTxOp>(
        "tx", Arg("rows", 64), Arg("columns", 32), make_condition<CountCondition>(NUM_ITER));

    for (int index = 1; index <= NUM_RX; ++index) {
      const auto rx_name = fmt::format("rx{}", index);
      auto rx = make_operator<ops::PingTensorRxOp>(rx_name);
      add_flow(tx, rx);
    }
  }
};

TEST(MultithreadedApp, TestSendingTensorToMultipleOperators) {
  // Issue 4272363
  using namespace holoscan;

  EnvVarWrapper wrapper({
      std::make_pair("HOLOSCAN_LOG_LEVEL", "DEBUG"),
      std::make_pair("HOLOSCAN_EXECUTOR_LOG_LEVEL", "INFO"),  // quiet multi_thread_scheduler.cpp
  });

  auto app = make_application<PingMultithreadApp>();

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  // configure and assign the scheduler
  app->scheduler(app->make_scheduler<MultiThreadScheduler>(
      "multithread-scheduler",
      Arg{"worker_thread_number", static_cast<int64_t>(NUM_RX)},
      Arg{"stop_on_deadlock_timeout", 100L}));  // should be > check_recession_period_ms

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("null data") == std::string::npos);
  for (int i = 1; i < NUM_RX; ++i) {
    EXPECT_TRUE(log_output.find(fmt::format(
                    "Rx message value - name:rx{}, data[0]:{}, nbytes:2048", i, NUM_ITER)) !=
                std::string::npos);
  }
  // Check that the last rx operator received the expected value and print the log if it didn't
  EXPECT_TRUE(log_output.find(fmt::format(
                  "Rx message value - name:rx{}, data[0]:{}, nbytes:2048", NUM_RX, NUM_ITER)) !=
              std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}
