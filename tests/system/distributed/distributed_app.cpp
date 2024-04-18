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
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "../env_wrapper.hpp"
#include "utility_apps.hpp"
namespace holoscan {

///////////////////////////////////////////////////////////////////////////////
// Tests
///////////////////////////////////////////////////////////////////////////////

TEST(DistributedApp, TestTwoParallelFragmentsApp) {
  auto app = make_application<TwoParallelFragmentsApp>();

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("SingleOp fragment1.op: 0 - 10") != std::string::npos);
  EXPECT_TRUE(log_output.find("SingleOp fragment2.op: 0 - 10") != std::string::npos);
}

TEST(DistributedApp, TestTwoMultiInputsOutputsFragmentsApp) {
  auto app = make_application<TwoMultiInputsOutputsFragmentsApp>();

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("received count: 10") != std::string::npos);
}

TEST(DistributedApp, TestTwoMultipleSingleOutputOperatorsApp) {
  auto app = make_application<TwoMultipleSingleOutputOperatorsApp>();

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("received count: 10") != std::string::npos);
}

TEST(DistributedApp, TestTwoMultipleSingleOutputOperatorsBroadcastApp) {
  auto app = make_application<TwoMultipleSingleOutputOperatorsBroadcastApp>();

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("received count: 10") != std::string::npos);
}

TEST(DistributedApp, TestOneTxBroadcastOneRxTwoInputs) {
  auto app = make_application<OneTxBroadcastOneRxTwoInputs>();

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("received count: 10") != std::string::npos);
}

TEST(DistributedApp, TestTwoMultiInputsOutputsFragmentsApp2) {
  auto app = make_application<TwoMultiInputsOutputsFragmentsApp2>();

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("received count: 10") != std::string::npos);
}

TEST(DistributedApp, TestUCXConnectionApp) {
  auto app = make_application<UCXConnectionApp>();

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("received count: 10") != std::string::npos);
}

TEST(DistributedApp, TestUCXConnectionApp2) {
  auto app = make_application<UCXConnectionApp2>();

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("received count: 10") != std::string::npos);
}

TEST(DistributedApp, TestUCXLinearPipelineApp) {
  auto app = make_application<UCXLinearPipelineApp>();

  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("received count: 20") != std::string::npos);
}

TEST(DistributedApp, TestUCXBroadcastApp) {
  auto app = make_application<UCXBroadcastApp>();

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Rx fragment3.rx message received count: 10") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("Rx fragment4.rx message received count: 10") != std::string::npos);
}

TEST(DistributedApp, TestUCXBroadCastMultiReceiverApp) {
  auto app = make_application<UCXBroadCastMultiReceiverApp>();

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("RxParam fragment2.rx message received (count: 10, size: 2)") !=
              std::string::npos);
  EXPECT_TRUE(log_output.find("Rx fragment4.rx message received count: 10") != std::string::npos);
}

TEST(DistributedApp, TestDriverTerminationWithConnectionFailure) {
  const char* env_orig = std::getenv("HOLOSCAN_MAX_CONNECTION_RETRY_COUNT");

  // Set retry count to 1 to save time
  const char* new_env_var = "1";
  setenv("HOLOSCAN_MAX_CONNECTION_RETRY_COUNT", new_env_var, 1);

  // Test that the driver terminates when both the driver and the worker are started but the
  // connection to the driver from the worker fails (wrong IP address such as '22' which is usually
  // used for SSH and not bindable so we can safely assume that the connection will fail).
  //
  // Note:: This test will hang if the port number 22 is bindable.
  const std::vector<std::string> args{
      "test_app", "--driver", "--worker", "--address", "127.0.0.1:22"};

  auto app = make_application<UCXLinearPipelineApp>(args);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  app->run();
  // The driver should terminate after the connection failure (after 1 retry)

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Failed to connect to driver") != std::string::npos);

  // restore the original environment variable
  if (env_orig) {
    setenv("HOLOSCAN_MAX_CONNECTION_RETRY_COUNT", env_orig, 1);
  } else {
    unsetenv("HOLOSCAN_MAX_CONNECTION_RETRY_COUNT");
  }
}

}  // namespace holoscan
