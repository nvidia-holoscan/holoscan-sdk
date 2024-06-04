/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <filesystem>
#include <string>
#include <utility>

#include <holoscan/operators/ping_rx/ping_rx.hpp>
#include <holoscan/operators/ping_tx/ping_tx.hpp>
#include "holoscan/holoscan.hpp"

#include "env_wrapper.hpp"

class SimplePingApp : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto tx = make_operator<ops::PingTxOp>("tx", make_resource<CountCondition>(100));
    auto rx = make_operator<ops::PingRxOp>("rx");
    add_flow(tx, rx, {{"out", "in"}});
  }
};

TEST(JobStatisticsApp, TestJobStatisticsDisabled) {
  using namespace holoscan;

  EnvVarWrapper wrapper({
      std::make_pair("HOLOSCAN_ENABLE_GXF_JOB_STATISTICS", "0"),
  });

  auto app = make_application<SimplePingApp>();

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStdout();
  app->run();

  std::string log_output = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(log_output.find("Job Statistics Report") == std::string::npos);
}

TEST(JobStatisticsApp, TestJobStatisticsEnabled) {
  using namespace holoscan;

  EnvVarWrapper wrapper({
      std::make_pair("HOLOSCAN_ENABLE_GXF_JOB_STATISTICS", "TRUE"),
  });

  auto app = make_application<SimplePingApp>();

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStdout();
  app->run();

  std::string console_output = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(console_output.find("Job Statistics Report") != std::string::npos);

  // Codelet statistics report is disabled by default
  EXPECT_TRUE(console_output.find("Codelet Statistics Report") == std::string::npos);
}

TEST(JobStatisticsApp, TestJobStatisticsEnabledCountSet) {
  using namespace holoscan;

  EnvVarWrapper wrapper({
      std::make_pair("HOLOSCAN_ENABLE_GXF_JOB_STATISTICS", "1"),
      std::make_pair("HOLOSCAN_GXF_JOB_STATISTICS_COUNT", "35"),
      std::make_pair("HOLOSCAN_LOG_LEVEL", "DEBUG"),
  });

  auto app = make_application<SimplePingApp>();

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();

  app->run();

  std::string console_output = testing::internal::GetCapturedStdout();
  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(console_output.find("Job Statistics Report") != std::string::npos);

  // Rely on DEBUG level log output to detect the event_history_count that was set as the
  // value is not shown in the report itself.
  EXPECT_TRUE(log_output.find("event_history_count: 35") != std::string::npos);
}

TEST(JobStatisticsApp, TestJobStatisticsCodeletReportEnabled) {
  using namespace holoscan;

  EnvVarWrapper wrapper({
      std::make_pair("HOLOSCAN_ENABLE_GXF_JOB_STATISTICS", "true"),
      std::make_pair("HOLOSCAN_GXF_JOB_STATISTICS_CODELET", "1"),
  });

  auto app = make_application<SimplePingApp>();

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStdout();
  app->run();

  std::string console_output = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(console_output.find("Job Statistics Report") != std::string::npos);
  EXPECT_TRUE(console_output.find("Codelet Statistics Report") != std::string::npos);
}

TEST(JobStatisticsApp, TestJobStatisticsFilePathSet) {
  using namespace holoscan;

  std::string file_path = "temp_job_stats.json";
  EXPECT_FALSE(std::filesystem::exists(file_path));

  EnvVarWrapper wrapper({
      std::make_pair("HOLOSCAN_ENABLE_GXF_JOB_STATISTICS", "true"),
      std::make_pair("HOLOSCAN_GXF_JOB_STATISTICS_PATH", file_path),
  });

  auto app = make_application<SimplePingApp>();

  // capture output to check that the expected messages were logged
  testing::internal::CaptureStdout();
  app->run();

  std::string console_output = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(console_output.find("Job Statistics Report") != std::string::npos);

  // check that the expected JSON file was created
  EXPECT_TRUE(std::filesystem::exists(file_path));
  std::filesystem::remove(file_path);
}
