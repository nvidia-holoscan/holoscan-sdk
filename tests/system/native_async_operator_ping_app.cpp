/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <gxf/core/gxf.h>

#include <string>

#include <holoscan/holoscan.hpp>

#include "../config.hpp"
#include "common/assert.hpp"

#include <holoscan/operators/async_ping_rx/async_ping_rx.hpp>
#include <holoscan/operators/async_ping_tx/async_ping_tx.hpp>
#include <holoscan/operators/ping_rx/ping_rx.hpp>
#include <holoscan/operators/ping_tx/ping_tx.hpp>

using namespace std::string_literals;

static HoloscanTestConfig test_config;

namespace holoscan {

class AsyncRxApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(5));
    auto rx = make_operator<ops::AsyncPingRxOp>("rx", Arg("delay", 10L));

    add_flow(tx, rx);
  }
};

class AsyncTxApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<ops::AsyncPingTxOp>("tx", Arg("delay", 10L), Arg("count", 5UL));
    auto rx = make_operator<ops::PingRxOp>("rx");

    add_flow(tx, rx);
  }
};

class AsyncTxRxApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<ops::AsyncPingTxOp>("tx", Arg("delay", 10L), Arg("count", 5UL));
    auto rx = make_operator<ops::AsyncPingRxOp>("rx", Arg("delay", 10L));

    add_flow(tx, rx);
  }
};

class ParameterizedAsyncPingTestFixture : public ::testing::TestWithParam<bool> {};

INSTANTIATE_TEST_CASE_P(AsyncPingApps, ParameterizedAsyncPingTestFixture,
                        ::testing::Values(false, true));

TEST_P(ParameterizedAsyncPingTestFixture, TestAsyncRxApp) {
  auto app = make_application<AsyncRxApp>();

  auto multithreaded = GetParam();
  if (multithreaded) {
    app->scheduler(app->make_scheduler<holoscan::MultiThreadScheduler>(
        "multithread-scheduler",
        holoscan::Arg("stop_on_deadlock", false),
        holoscan::Arg("max_duration_ms", 1000L)));
  }

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Async ping rx thread entering") != std::string::npos);
  EXPECT_TRUE(log_output.find("Async ping rx thread exiting") != std::string::npos);
  EXPECT_TRUE(log_output.find("Rx message value: 5") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  EXPECT_TRUE(log_output.find("Async ping tx thread entering") == std::string::npos);
  EXPECT_TRUE(log_output.find("Async ping tx thread exiting") == std::string::npos);
}

TEST_P(ParameterizedAsyncPingTestFixture, TestAsyncTxApp) {
  auto app = make_application<AsyncTxApp>();

  auto multithreaded = GetParam();
  if (multithreaded) {
    app->scheduler(app->make_scheduler<holoscan::MultiThreadScheduler>(
        "multithread-scheduler", holoscan::Arg("stop_on_deadlock", true)));
  }

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Async ping tx thread entering") != std::string::npos);
  EXPECT_TRUE(log_output.find("Async ping tx thread exiting") != std::string::npos);
  EXPECT_TRUE(log_output.find("Rx message value: 5") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  EXPECT_TRUE(log_output.find("Async ping rx thread entering") == std::string::npos);
  EXPECT_TRUE(log_output.find("Async ping rx thread exiting") == std::string::npos);
}

TEST_P(ParameterizedAsyncPingTestFixture, TestAsyncTxRxApp) {
  auto app = make_application<AsyncTxRxApp>();

  auto multithreaded = GetParam();
  if (multithreaded) {
    app->scheduler(app->make_scheduler<holoscan::MultiThreadScheduler>(
        "multithread-scheduler", holoscan::Arg("stop_on_deadlock", true)));
  }

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Async ping tx thread entering") != std::string::npos);
  EXPECT_TRUE(log_output.find("Async ping rx thread entering") != std::string::npos);
  EXPECT_TRUE(log_output.find("Async ping tx thread exiting") != std::string::npos);
  EXPECT_TRUE(log_output.find("Async ping rx thread exiting") != std::string::npos);
  EXPECT_TRUE(log_output.find("Rx message value: 5") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

}  // namespace holoscan
