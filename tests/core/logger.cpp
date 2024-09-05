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

#include <cstdlib>
#include <memory>
#include <string>

#include <common/logger/spdlog_logger.hpp>
#include <holoscan/holoscan.hpp>
#include "../config.hpp"

using namespace std::string_literals;

static HoloscanTestConfig test_config;

namespace holoscan {

class LevelParameterizedTestFixture : public ::testing::TestWithParam<LogLevel> {};

TEST(Logger, TestLoggingPattern) {
  auto orig_level = log_level();
  const char* env_orig = std::getenv("HOLOSCAN_LOG_LEVEL");
  const char* env_format = std::getenv("HOLOSCAN_LOG_FORMAT");

  // Unset HOLOSCAN_LOG_LEVEL environment variable so that the log level is not overridden
  unsetenv("HOLOSCAN_LOG_LEVEL");

  set_log_level(LogLevel::INFO);
  // log without the actual message (%v)
  set_log_pattern("[thread %t]");

  testing::internal::CaptureStderr();
  HOLOSCAN_LOG_INFO("my_message");

  // test that the specified pattern includes the thread, but omits the message
  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("[thread") != std::string::npos) << "=== LOG ===\n"
                                                               << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("my_message") == std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // restore the original log level
  set_log_level(orig_level);

  // restore the original environment variable
  if (env_orig) {
    setenv("HOLOSCAN_LOG_LEVEL", env_orig, 1);
  } else {
    unsetenv("HOLOSCAN_LOG_LEVEL");
  }

  // restore the original log format
  if (env_format) {
    setenv("HOLOSCAN_LOG_FORMAT", env_format, 1);
  } else {
    unsetenv("HOLOSCAN_LOG_FORMAT");
  }
}

TEST(Logger, TestDefaultLogPattern) {
  auto orig_level = log_level();
  const char* env_orig = std::getenv("HOLOSCAN_LOG_LEVEL");
  const char* env_format = std::getenv("HOLOSCAN_LOG_FORMAT");

  std::shared_ptr<Application> app;

  // set default log level to INFO
  setenv("HOLOSCAN_LOG_LEVEL", "INFO", 1);

  // 1. HOLOSCAN_LOG_FORMAT is set to SHORT and no set_log_pattern() is called
  //    before Application::Application().
  //    => expected log pattern is SHORT ("[%^%l%$] %v")
  Logger::log_pattern_set_by_user = false;
  setenv("HOLOSCAN_LOG_FORMAT", "SHORT", 1);
  app = holoscan::make_application<Application>();
  EXPECT_EQ(Logger::pattern(), "[%^%l%$] %v");

  // 2. HOLOSCAN_LOG_FORMAT is set to SHORT and set_log_pattern("FULL") is called
  //    before Application::Application().
  //    => expected log pattern is SHORT (environment variable has higher priority)
  Logger::log_pattern_set_by_user = false;
  setenv("HOLOSCAN_LOG_FORMAT", "SHORT", 1);
  set_log_pattern("FULL");
  app = holoscan::make_application<Application>();
  EXPECT_EQ(Logger::pattern(), "[%^%l%$] %v");

  // 3. HOLOSCAN_LOG_FORMAT is set to FULL and set_log_pattern("SHORT") is called
  //    after Application::Application().
  //    => expected log pattern is FULL (environment variable has higher priority)
  Logger::log_pattern_set_by_user = false;
  setenv("HOLOSCAN_LOG_FORMAT", "FULL", 1);
  app = holoscan::make_application<Application>();
  set_log_pattern("SHORT");
  EXPECT_EQ(Logger::pattern(), "[%Y-%m-%d %H:%M:%S.%e] [%t] [%n] [%^%l%$] [%s:%#] %v");

  // 4. HOLOSCAN_LOG_FORMAT is not set and set_log_pattern("LONG") is called
  //    after Application::Application().
  //    => expected log pattern is LONG ("[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] [%s:%#] %v")
  Logger::log_pattern_set_by_user = false;
  unsetenv("HOLOSCAN_LOG_FORMAT");
  app = holoscan::make_application<Application>();
  set_log_pattern("LONG");
  EXPECT_EQ(Logger::pattern(), "[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] [%s:%#] %v");

  // 5. HOLOSCAN_LOG_FORMAT is not set.
  //    => expected log pattern is DEFAULT (after Application::Application()).
  Logger::log_pattern_set_by_user = false;
  unsetenv("HOLOSCAN_LOG_FORMAT");
  set_log_pattern("DEFAULT");
  app = holoscan::make_application<Application>();
  EXPECT_EQ(Logger::pattern(), "[%^%l%$] [%s:%#] %v");

  // Unset HOLOSCAN_LOG_LEVEL environment variable so that the log level is not overridden
  unsetenv("HOLOSCAN_LOG_LEVEL");

  // restore the original log level
  set_log_level(orig_level);

  // restore the original environment variable
  if (env_orig) {
    setenv("HOLOSCAN_LOG_LEVEL", env_orig, 1);
  } else {
    unsetenv("HOLOSCAN_LOG_LEVEL");
  }

  // restore the original log format
  if (env_format) {
    setenv("HOLOSCAN_LOG_FORMAT", env_format, 1);
  } else {
    unsetenv("HOLOSCAN_LOG_FORMAT");
  }
}

TEST_P(LevelParameterizedTestFixture, TestLoggingGetSet) {
  const char* env_orig = std::getenv("HOLOSCAN_LOG_LEVEL");

  // Unset HOLOSCAN_LOG_LEVEL environment variable so that the log level is not overridden
  unsetenv("HOLOSCAN_LOG_LEVEL");

  // check the current logging level
  auto orig_level = log_level();

  // set and get the desired logging levelf
  auto desired_level = GetParam();
  set_log_level(desired_level);
  EXPECT_EQ(log_level(), desired_level);

  // restore the original level
  set_log_level(orig_level);

  // restore the original environment variable
  if (env_orig) {
    setenv("HOLOSCAN_LOG_LEVEL", env_orig, 1);
  } else {
    unsetenv("HOLOSCAN_LOG_LEVEL");
  }
}

INSTANTIATE_TEST_CASE_P(LoggingTests, LevelParameterizedTestFixture,
                        ::testing::Values(LogLevel::TRACE, LogLevel::DEBUG, LogLevel::INFO,
                                          LogLevel::WARN, LogLevel::ERROR, LogLevel::CRITICAL,
                                          LogLevel::OFF));

TEST_P(LevelParameterizedTestFixture, TestLoadEnvLogLevel) {
  // check the current logging level
  auto orig_level = log_level();
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

  // test that the desired logging level was loaded from the environment
  holoscan::set_log_level(LogLevel::INFO);  // set an arbitrary level to load from the environment
  EXPECT_EQ(log_level(), desired_level);

  // Unset HOLOSCAN_LOG_LEVEL environment variable so that the log level is not overridden
  unsetenv("HOLOSCAN_LOG_LEVEL");

  // restore the original log level
  set_log_level(orig_level);

  // restore the original log level
  if (env_orig) {
    setenv("HOLOSCAN_LOG_LEVEL", env_orig, 1);
  } else {
    unsetenv("HOLOSCAN_LOG_LEVEL");
  }
}

TEST_P(LevelParameterizedTestFixture, TestLoggingMacros) {
  const char* env_orig = std::getenv("HOLOSCAN_LOG_LEVEL");

  // Unset HOLOSCAN_LOG_LEVEL environment variable so that the log level is not overridden
  unsetenv("HOLOSCAN_LOG_LEVEL");

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
      EXPECT_TRUE(log_output.find("trace") != std::string::npos) << "=== LOG ===\n"
                                                                 << log_output << "\n===========\n";
      break;
    case LogLevel::DEBUG:
      EXPECT_TRUE(log_output.find("debug") != std::string::npos) << "=== LOG ===\n"
                                                                 << log_output << "\n===========\n";
      EXPECT_TRUE(log_output.find("unlogged") == std::string::npos)
          << "=== LOG ===\n"
          << log_output << "\n===========\n";
      break;
    case LogLevel::INFO:
      EXPECT_TRUE(log_output.find("info") != std::string::npos) << "=== LOG ===\n"
                                                                << log_output << "\n===========\n";
      EXPECT_TRUE(log_output.find("unlogged") == std::string::npos)
          << "=== LOG ===\n"
          << log_output << "\n===========\n";
      break;
    case LogLevel::WARN:
      EXPECT_TRUE(log_output.find("warn") != std::string::npos) << "=== LOG ===\n"
                                                                << log_output << "\n===========\n";
      EXPECT_TRUE(log_output.find("unlogged") == std::string::npos)
          << "=== LOG ===\n"
          << log_output << "\n===========\n";
      break;
    case LogLevel::ERROR:
      EXPECT_TRUE(log_output.find("error") != std::string::npos) << "=== LOG ===\n"
                                                                 << log_output << "\n===========\n";
      EXPECT_TRUE(log_output.find("unlogged") == std::string::npos)
          << "=== LOG ===\n"
          << log_output << "\n===========\n";
      break;
    case LogLevel::CRITICAL:
      EXPECT_TRUE(log_output.find("critical") != std::string::npos)
          << "=== LOG ===\n"
          << log_output << "\n===========\n";
      EXPECT_TRUE(log_output.find("unlogged") == std::string::npos)
          << "=== LOG ===\n"
          << log_output << "\n===========\n";
      break;
    case LogLevel::OFF:
      EXPECT_TRUE(log_output.find("unlogged") == std::string::npos)
          << "=== LOG ===\n"
          << log_output << "\n===========\n";
      break;
  }

  // restore the original log level
  set_log_level(orig_level);

  // restore the original environment variable
  if (env_orig) {
    setenv("HOLOSCAN_LOG_LEVEL", env_orig, 1);
  } else {
    unsetenv("HOLOSCAN_LOG_LEVEL");
  }
}

TEST(Logger, TestDefaultLogLevel) {
  auto orig_level = log_level();
  const char* env_orig = std::getenv("HOLOSCAN_LOG_LEVEL");

  std::shared_ptr<Application> app;

  // 1. HOLOSCAN_LOG_LEVEL is set to TRACE and no set_log_level() is called
  //    before Application::Application().
  //    => expected log level is TRACE
  Logger::set_level(LogLevel::INFO);
  Logger::log_level_set_by_user = false;
  setenv("HOLOSCAN_LOG_LEVEL", "TRACE", 1);
  app = holoscan::make_application<Application>();
  EXPECT_EQ(log_level(), LogLevel::TRACE);

  // 2. HOLOSCAN_LOG_LEVEL is set to TRACE and set_log_level(LogLevel::WARN) is called
  //    before Application::Application().
  //    => expected log level is TRACE (environment variable has higher priority)
  Logger::set_level(LogLevel::INFO);
  Logger::log_level_set_by_user = false;
  setenv("HOLOSCAN_LOG_LEVEL", "TRACE", 1);
  set_log_level(LogLevel::WARN);
  app = holoscan::make_application<Application>();
  EXPECT_EQ(log_level(), LogLevel::TRACE);

  // 3. HOLOSCAN_LOG_LEVEL is set to ERROR and set_log_level(LogLevel::WARN) is called
  //    after Application::Application().
  //    => expected log level is ERROR (environment variable has higher priority)
  Logger::set_level(LogLevel::INFO);
  Logger::log_level_set_by_user = false;
  setenv("HOLOSCAN_LOG_LEVEL", "ERROR", 1);
  app = holoscan::make_application<Application>();
  set_log_level(LogLevel::WARN);
  EXPECT_EQ(log_level(), LogLevel::ERROR);

  // 4. HOLOSCAN_LOG_LEVEL is not set and set_log_level(LogLevel::WARN) is called
  //    after Application::Application().
  //    => expected log level is WARN
  Logger::set_level(LogLevel::INFO);
  Logger::log_level_set_by_user = false;
  unsetenv("HOLOSCAN_LOG_LEVEL");
  app = holoscan::make_application<Application>();
  set_log_level(LogLevel::WARN);
  EXPECT_EQ(log_level(), LogLevel::WARN);

  // 5. HOLOSCAN_LOG_LEVEL is not set.
  //    => expected log level is INFO (after Application::Application()).
  Logger::log_level_set_by_user = false;
  unsetenv("HOLOSCAN_LOG_LEVEL");
  app = holoscan::make_application<Application>();
  EXPECT_EQ(log_level(), LogLevel::INFO);

  // Unset HOLOSCAN_LOG_LEVEL environment variable so that the log level is not overridden
  unsetenv("HOLOSCAN_LOG_LEVEL");

  // restore the original log level
  set_log_level(orig_level);

  // restore the original environment variable
  if (env_orig) {
    setenv("HOLOSCAN_LOG_LEVEL", env_orig, 1);
  } else {
    unsetenv("HOLOSCAN_LOG_LEVEL");
  }
}

////////////////////////////////////////////////////////////////////////////////
// Test cases for SpdlogLogger
////////////////////////////////////////////////////////////////////////////////

namespace {
constexpr const char* kLogTestString = "Test log message";
constexpr const char* kLogFilePath = "/tmp/test_log_file";
}  // namespace

class MockSpdlogLogger : public nvidia::logger::SpdlogLogger {
 public:
  explicit MockSpdlogLogger(const char* name) : SpdlogLogger(name) {}

  static std::shared_ptr<MockSpdlogLogger> create() {
    return std::make_shared<MockSpdlogLogger>("mock_logger");
  }
};

class RedirectLogTest : public ::testing::Test {
 protected:
  void SetUp() override { file_ = fopen(kLogFilePath, "wr"); }

  void TearDown() override {
    // Close the file
    fclose(file_);
  }

  std::string get_file_content() {
    // Read the file content from kLogFilePath
    std::string file_content;
    std::ifstream file;
    file.open(kLogFilePath);
    getline(file, file_content);
    file.close();
    return file_content;
  }

  std::FILE* file_ = nullptr;
};

TEST_F(RedirectLogTest, Trace) {
  auto logger = MockSpdlogLogger::create();

  logger->redirect(HOLOSCAN_LOG_LEVEL_TRACE, file_);
  logger->level(HOLOSCAN_LOG_LEVEL_TRACE);
  logger->log(__FILE__, __LINE__, "test", HOLOSCAN_LOG_LEVEL_TRACE, kLogTestString);
  std::string file_content = get_file_content();

  EXPECT_NE(file_content.find(std::string(kLogTestString)), std::string::npos) << file_content;
  EXPECT_NE(file_content.find(std::string("trace")), std::string::npos) << file_content;
}

TEST_F(RedirectLogTest, Debug) {
  auto logger = MockSpdlogLogger::create();

  logger->redirect(HOLOSCAN_LOG_LEVEL_DEBUG, file_);
  logger->level(HOLOSCAN_LOG_LEVEL_DEBUG);
  logger->log(__FILE__, __LINE__, "test", HOLOSCAN_LOG_LEVEL_DEBUG, kLogTestString);
  // Log info message
  logger->log(__FILE__, __LINE__, "test", HOLOSCAN_LOG_LEVEL_INFO, kLogTestString);
  std::string file_content = get_file_content();

  EXPECT_NE(file_content.find(std::string(kLogTestString)), std::string::npos) << file_content;
  EXPECT_NE(file_content.find(std::string("debug")), std::string::npos) << file_content;
  // Info log message should not be written to the file
  EXPECT_EQ(file_content.find(std::string("info")), std::string::npos) << file_content;
}

TEST_F(RedirectLogTest, Info) {
  auto logger = MockSpdlogLogger::create();

  logger->redirect(HOLOSCAN_LOG_LEVEL_INFO, file_);
  logger->level(HOLOSCAN_LOG_LEVEL_INFO);
  logger->log(__FILE__, __LINE__, "test", HOLOSCAN_LOG_LEVEL_INFO, kLogTestString);
  std::string file_content = get_file_content();

  EXPECT_NE(file_content.find(std::string(kLogTestString)), std::string::npos) << file_content;
  EXPECT_NE(file_content.find(std::string("info")), std::string::npos) << file_content;
}

TEST_F(RedirectLogTest, Warn) {
  auto logger = MockSpdlogLogger::create();

  logger->redirect(HOLOSCAN_LOG_LEVEL_WARN, file_);
  logger->level(HOLOSCAN_LOG_LEVEL_WARN);
  logger->log(__FILE__, __LINE__, "test", HOLOSCAN_LOG_LEVEL_WARN, kLogTestString);
  std::string file_content = get_file_content();

  EXPECT_NE(file_content.find(std::string(kLogTestString)), std::string::npos) << file_content;
  EXPECT_NE(file_content.find(std::string("warn")), std::string::npos) << file_content;
}

TEST_F(RedirectLogTest, Error) {
  auto logger = MockSpdlogLogger::create();

  logger->redirect(HOLOSCAN_LOG_LEVEL_ERROR, file_);
  logger->level(HOLOSCAN_LOG_LEVEL_ERROR);
  logger->log(__FILE__, __LINE__, "test", HOLOSCAN_LOG_LEVEL_ERROR, kLogTestString);
  std::string file_content = get_file_content();

  EXPECT_NE(file_content.find(std::string(kLogTestString)), std::string::npos) << file_content;
  EXPECT_NE(file_content.find(std::string("error")), std::string::npos) << file_content;
}

TEST_F(RedirectLogTest, Critical) {
  auto logger = MockSpdlogLogger::create();

  logger->redirect(HOLOSCAN_LOG_LEVEL_CRITICAL, file_);
  logger->level(HOLOSCAN_LOG_LEVEL_CRITICAL);
  logger->log(__FILE__, __LINE__, "test", HOLOSCAN_LOG_LEVEL_CRITICAL, kLogTestString);
  std::string file_content = get_file_content();

  EXPECT_NE(file_content.find(std::string(kLogTestString)), std::string::npos) << file_content;
  EXPECT_NE(file_content.find(std::string("critical")), std::string::npos) << file_content;
}

TEST_F(RedirectLogTest, NullPointer) {
  auto logger = MockSpdlogLogger::create();

  logger->redirect(HOLOSCAN_LOG_LEVEL_CRITICAL, nullptr);
  logger->level(HOLOSCAN_LOG_LEVEL_CRITICAL);
  logger->log(__FILE__, __LINE__, "test", HOLOSCAN_LOG_LEVEL_CRITICAL, kLogTestString);
  std::string file_content = get_file_content();

  // No log message should be written to the file
  EXPECT_EQ(file_content.find(std::string(kLogTestString)), std::string::npos) << file_content;
  EXPECT_EQ(file_content.find(std::string("critical")), std::string::npos) << file_content;
}

}  // namespace holoscan
