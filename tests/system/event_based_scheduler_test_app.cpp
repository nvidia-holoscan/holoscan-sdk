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
#include <utility>
#include <vector>

#include "holoscan/holoscan.hpp"
#include "holoscan/operators/ping_tensor_rx/ping_tensor_rx.hpp"
#include "holoscan/operators/ping_tensor_tx/ping_tensor_tx.hpp"
#include "holoscan/core/schedulers/event_based_scheduler.hpp"

#include "env_wrapper.hpp"

constexpr int NUM_RX = 3;
constexpr int NUM_ITER = 100;

class PingEventBasedApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<ops::PingTensorTxOp>("tx",
                                                 Arg("rows", 64),
                                                 Arg("columns", 32),
                                                 Arg("storage_type", "device"),
                                                 make_condition<CountCondition>(NUM_ITER));

    for (int index = 1; index <= NUM_RX; ++index) {
      const auto rx_name = fmt::format("rx{}", index);
      auto rx = make_operator<ops::PingTensorRxOp>(rx_name);
      add_flow(tx, rx);
    }
  }
};

TEST(EventBasedSchedulerApp, TestSendingTensorToMultipleOperatorsWithoutPinCores) {
  // Test EventBasedScheduler without pin_cores parameter
  using namespace holoscan;

  EnvVarWrapper wrapper({
      std::make_pair("HOLOSCAN_LOG_LEVEL", "DEBUG"),
      std::make_pair("HOLOSCAN_EXECUTOR_LOG_LEVEL", "INFO"),
  });

  auto app = make_application<PingEventBasedApp>();

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  // configure and assign the scheduler without pin_cores
  app->scheduler(app->make_scheduler<EventBasedScheduler>(
      "event-based-scheduler",
      Arg{"worker_thread_number", static_cast<int64_t>(NUM_RX)},
      Arg{"stop_on_deadlock_timeout", 100L}));

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("null data") == std::string::npos) << "=== LOG ===\n"
                                                                 << log_output << "\n===========\n";
  for (int i = 1; i <= NUM_RX; ++i) {
    EXPECT_TRUE(log_output.find(fmt::format("rx{} received message", i)) != std::string::npos)
        << "=== LOG ===\n"
        << log_output << "\n===========\n";
  }
}

TEST(EventBasedSchedulerApp, TestSendingTensorToMultipleOperatorsWithPinCores) {
  // Test EventBasedScheduler with pin_cores parameter
  using namespace holoscan;

  EnvVarWrapper wrapper({
      std::make_pair("HOLOSCAN_LOG_LEVEL", "DEBUG"),
      std::make_pair("HOLOSCAN_EXECUTOR_LOG_LEVEL", "INFO"),
  });

  auto app = make_application<PingEventBasedApp>();

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStderr();

  // configure and assign the scheduler with pin_cores
  app->scheduler(app->make_scheduler<EventBasedScheduler>(
      "event-based-scheduler",
      Arg{"worker_thread_number", static_cast<int64_t>(NUM_RX)},
      Arg{"stop_on_deadlock_timeout", 100L},
      Arg{"pin_cores", std::vector<uint32_t>{0, 1, 2}}));

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("null data") == std::string::npos) << "=== LOG ===\n"
                                                                 << log_output << "\n===========\n";
  for (int i = 1; i <= NUM_RX; ++i) {
    EXPECT_TRUE(log_output.find(fmt::format("rx{} received message", i)) != std::string::npos)
        << "=== LOG ===\n"
        << log_output << "\n===========\n";
  }

  // Check that the worker threads are pinned to the specified cores
  EXPECT_TRUE(log_output.find("pinned to CPU cores: 0,1,2") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}
