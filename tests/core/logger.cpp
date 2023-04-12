/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdlib>
#include <string>

#include "../config.hpp"
#include <holoscan/holoscan.hpp>

using namespace std::string_literals;

static HoloscanTestConfig test_config;

namespace holoscan {

class LevelParameterizedTestFixture : public ::testing::TestWithParam<LogLevel> {};

TEST(Logger, TestLoggingPattern) {
  auto orig_level = log_level();

  set_log_level(LogLevel::INFO);
  // log without the actual message (%v)
  set_log_pattern("[thread %t]");

  testing::internal::CaptureStderr();
  HOLOSCAN_LOG_INFO("my_message");

  // can explicitly flush the log (not required for this test case)
  Logger::flush();

  // test that the specified pattern includes the thread, but omits the message
  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("[thread") != std::string::npos);
  EXPECT_TRUE(log_output.find("my_message") == std::string::npos);

  // restore original log level
  set_log_level(orig_level);

  // restore default logging pattern or test cases run afterwards may fail
  set_log_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%n] %v");
}

TEST(Logger, TestLoggingFlushLevel) {
  auto default_level = Logger::flush_level();

  Logger::flush_on(LogLevel::WARN);
  EXPECT_EQ(Logger::flush_level(), LogLevel::WARN);

  Logger::flush_on(default_level);
}

TEST(Logger, TestLoggingBacktrace) {
  bool default_backtrace = Logger::should_backtrace();

  Logger::enable_backtrace(32);
  EXPECT_TRUE(Logger::should_backtrace());

  Logger::disable_backtrace();
  EXPECT_FALSE(Logger::should_backtrace());

  Logger::dump_backtrace();

  if (default_backtrace) {
    Logger::enable_backtrace(32);
  } else {
    Logger::disable_backtrace();
  }
}

TEST_P(LevelParameterizedTestFixture, TestLoggingGetSet) {
  // check the current logging level
  auto orig_level = log_level();

  // set and get the desired logging levelf
  auto desired_level = GetParam();
  set_log_level(desired_level);
  EXPECT_EQ(log_level(), desired_level);

  // restore the original level
  set_log_level(orig_level);
}

INSTANTIATE_TEST_CASE_P(LoggingTests, LevelParameterizedTestFixture,
                        ::testing::Values(LogLevel::TRACE, LogLevel::DEBUG, LogLevel::INFO,
                                          LogLevel::WARN, LogLevel::ERROR, LogLevel::CRITICAL,
                                          LogLevel::OFF));

TEST_P(LevelParameterizedTestFixture, TestLoadEnvLogLevel) {
  const char* env_orig = std::getenv("HOLOSCAN_LOG_LEVEL");

  auto desired_level = GetParam();
  switch (desired_level) {
    case LogLevel::TRACE:
      setenv("HOLOSCAN_LOG_LEVEL", "TRACE", 1);
      break;
    case LogLevel::DEBUG:
      setenv("HOLOSCAN_LOG_LEVEL", "DEBUG", 1);
      break;
    case LogLevel::INFO:
      setenv("HOLOSCAN_LOG_LEVEL", "INFO", 1);
      break;
    case LogLevel::WARN:
      setenv("HOLOSCAN_LOG_LEVEL", "WARN", 1);
      break;
    case LogLevel::ERROR:
      setenv("HOLOSCAN_LOG_LEVEL", "ERROR", 1);
      break;
    case LogLevel::CRITICAL:
      setenv("HOLOSCAN_LOG_LEVEL", "CRITICAL", 1);
      break;
    case LogLevel::OFF:
      setenv("HOLOSCAN_LOG_LEVEL", "OFF", 1);
      break;
  }

  load_env_log_level();
  EXPECT_EQ(log_level(), desired_level);

  // restore the original log level
  if (env_orig) {
    setenv("HOLOSCAN_LOG_LEVEL", env_orig, 1);
  } else {
    unsetenv("HOLOSCAN_LOG_LEVEL");
  }
}

TEST_P(LevelParameterizedTestFixture, TestLoggingMacros) {
  // check the current logging level
  auto orig_level = log_level();

  auto desired_level = GetParam();
  set_log_level(desired_level);

  // test that expected logging messages were logged
  testing::internal::CaptureStderr();

  // Log both visible and unvisible messages
  switch (desired_level) {
    case LogLevel::TRACE:
      HOLOSCAN_LOG_TRACE("visible message");
      break;
    case LogLevel::DEBUG:
      HOLOSCAN_LOG_DEBUG("visible message");
      HOLOSCAN_LOG_TRACE("unlogged message");
      break;
    case LogLevel::INFO:
      HOLOSCAN_LOG_INFO("visible message");
      HOLOSCAN_LOG_DEBUG("unlogged message");
      HOLOSCAN_LOG_TRACE("unlogged message");
      break;
    case LogLevel::WARN:
      HOLOSCAN_LOG_WARN("visible message");
      HOLOSCAN_LOG_INFO("unlogged message");
      HOLOSCAN_LOG_DEBUG("unlogged message");
      HOLOSCAN_LOG_TRACE("unlogged message");
      break;
    case LogLevel::ERROR:
      HOLOSCAN_LOG_ERROR("visible message");
      HOLOSCAN_LOG_WARN("unlogged message");
      HOLOSCAN_LOG_INFO("unlogged message");
      HOLOSCAN_LOG_DEBUG("unlogged message");
      HOLOSCAN_LOG_TRACE("unlogged message");
      break;
    case LogLevel::CRITICAL:
      HOLOSCAN_LOG_CRITICAL("visible message");
      HOLOSCAN_LOG_ERROR("unlogged message");
      HOLOSCAN_LOG_WARN("unlogged message");
      HOLOSCAN_LOG_INFO("unlogged message");
      HOLOSCAN_LOG_DEBUG("unlogged message");
      HOLOSCAN_LOG_TRACE("unlogged message");
      break;
    case LogLevel::OFF:
      HOLOSCAN_LOG_CRITICAL("unlogged message");
      HOLOSCAN_LOG_ERROR("unlogged message");
      HOLOSCAN_LOG_WARN("unlogged message");
      HOLOSCAN_LOG_INFO("unlogged message");
      HOLOSCAN_LOG_DEBUG("unlogged message");
      HOLOSCAN_LOG_TRACE("unlogged message");
      break;
  }

  std::string log_output = testing::internal::GetCapturedStderr();

  switch (desired_level) {
    case LogLevel::TRACE:
      EXPECT_TRUE(log_output.find("trace") != std::string::npos);
      break;
    case LogLevel::DEBUG:
      EXPECT_TRUE(log_output.find("debug") != std::string::npos);
      EXPECT_TRUE(log_output.find("unlogged") == std::string::npos);
      break;
    case LogLevel::INFO:
      EXPECT_TRUE(log_output.find("info") != std::string::npos);
      EXPECT_TRUE(log_output.find("unlogged") == std::string::npos);
      break;
    case LogLevel::WARN:
      EXPECT_TRUE(log_output.find("warn") != std::string::npos);
      EXPECT_TRUE(log_output.find("unlogged") == std::string::npos);
      break;
    case LogLevel::ERROR:
      EXPECT_TRUE(log_output.find("error") != std::string::npos);
      EXPECT_TRUE(log_output.find("unlogged") == std::string::npos);
      break;
    case LogLevel::CRITICAL:
      EXPECT_TRUE(log_output.find("critical") != std::string::npos);
      EXPECT_TRUE(log_output.find("unlogged") == std::string::npos);
      break;
    case LogLevel::OFF:
      EXPECT_TRUE(log_output.find("unlogged") == std::string::npos);
      break;
  }

  // restore the original log level
  set_log_level(orig_level);
}

}  // namespace holoscan
