/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <optional>
#include <string>

#include <holoscan/logger/logger.hpp>
#include <holoscan/logger/spdlog_logger.hpp>

using namespace std::string_literals;

namespace holoscan {

// Parameterized fixture that runs each test body once per LogLevel variant (TRACE through OFF).
// Instantiated below the test definitions via INSTANTIATE_TEST_CASE_P.
class LevelParameterizedTestFixture : public ::testing::TestWithParam<LogLevel> {};

namespace {

// Returns a copy of the named environment variable's value, or nullopt if unset. Used by tests
// to snapshot global logger state so it can be restored on exit.
std::optional<std::string> get_env_copy(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return std::nullopt;
  }
  return std::string{value};
}

// Restores a previously snapshotted environment variable. If the snapshot was nullopt (absent
// before the test), the variable is unset rather than set to an empty string.
void restore_env(const char* name, const std::optional<std::string>& value) {
  if (value) {
    setenv(name, value->c_str(), 1);
  } else {
    unsetenv(name);
  }
}

}  // namespace

TEST(Logger, TestLoggingPattern) {
  auto orig_level = log_level();
  const std::string orig_pattern = Logger::pattern();
  const bool orig_pattern_set_by_user = Logger::log_pattern_set_by_user;
  const auto env_orig = get_env_copy("HOLOSCAN_LOG_LEVEL");
  const auto env_format = get_env_copy("HOLOSCAN_LOG_FORMAT");

  // Unset HOLOSCAN_LOG_LEVEL environment variable so that the log level is not overridden
  unsetenv("HOLOSCAN_LOG_LEVEL");

  set_log_level(LogLevel::INFO);
  // Key setup: pattern includes thread id (%t) but omits %v (the message body).
  set_log_pattern("[thread %t]");

  testing::internal::CaptureStderr();
  HOLOSCAN_LOG_INFO("my_message");

  // Key assertion: output contains "[thread" but omits "my_message" because %v is absent.
  const std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("[thread") != std::string::npos) << "=== LOG ===\n"
                                                               << log_output << "\n===========\n";
  EXPECT_TRUE(log_output.find("my_message") == std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // restore the original log level
  set_log_level(orig_level);

  // restore the original environment variable
  restore_env("HOLOSCAN_LOG_LEVEL", env_orig);

  // restore the original log format
  restore_env("HOLOSCAN_LOG_FORMAT", env_format);

  set_log_pattern(orig_pattern);
  Logger::log_pattern_set_by_user = orig_pattern_set_by_user;
}

TEST_P(LevelParameterizedTestFixture, TestLoggingGetSet) {
  const auto env_orig = get_env_copy("HOLOSCAN_LOG_LEVEL");

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
  restore_env("HOLOSCAN_LOG_LEVEL", env_orig);
}

INSTANTIATE_TEST_CASE_P(LoggingTests, LevelParameterizedTestFixture,
                        ::testing::Values(LogLevel::TRACE, LogLevel::DEBUG, LogLevel::INFO,
                                          LogLevel::WARN, LogLevel::ERROR, LogLevel::CRITICAL,
                                          LogLevel::OFF));

TEST_P(LevelParameterizedTestFixture, TestLoadEnvLogLevel) {
  // check the current logging level
  auto orig_level = log_level();
  const auto env_orig = get_env_copy("HOLOSCAN_LOG_LEVEL");

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

  // Key assertion: set_log_level reads HOLOSCAN_LOG_LEVEL from the environment and applies it,
  //                overriding the INFO level passed as argument.
  holoscan::set_log_level(LogLevel::INFO);  // set an arbitrary level to load from the environment
  EXPECT_EQ(log_level(), desired_level);

  // Unset HOLOSCAN_LOG_LEVEL environment variable so that the log level is not overridden
  unsetenv("HOLOSCAN_LOG_LEVEL");

  // restore the original log level
  set_log_level(orig_level);

  // restore the original log level
  restore_env("HOLOSCAN_LOG_LEVEL", env_orig);
}

TEST_P(LevelParameterizedTestFixture, TestLoggingMacros) {
  const auto env_orig = get_env_copy("HOLOSCAN_LOG_LEVEL");

  // Unset HOLOSCAN_LOG_LEVEL environment variable so that the log level is not overridden
  unsetenv("HOLOSCAN_LOG_LEVEL");

  // check the current logging level
  auto orig_level = log_level();

  auto desired_level = GetParam();
  set_log_level(desired_level);
  // Key setup: active level set to desired_level; only macros at or above this level emit output.
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

  // Key assertion: captured output contains the level keyword for the visible message and
  //                must not contain "unlogged" for any below-threshold call.
  const std::string log_output = testing::internal::GetCapturedStderr();

  switch (desired_level) {
    case LogLevel::TRACE:
#if HOLOSCAN_LOG_ACTIVE_LEVEL <= HOLOSCAN_LOG_LEVEL_TRACE
      EXPECT_TRUE(log_output.find("trace") != std::string::npos) << "=== LOG ===\n"
                                                                 << log_output << "\n===========\n";
#else
      EXPECT_TRUE(log_output.empty()) << "=== LOG ===\n" << log_output << "\n===========\n";
#endif
      break;
    case LogLevel::DEBUG:
#if HOLOSCAN_LOG_ACTIVE_LEVEL <= HOLOSCAN_LOG_LEVEL_DEBUG
      EXPECT_TRUE(log_output.find("debug") != std::string::npos) << "=== LOG ===\n"
                                                                 << log_output << "\n===========\n";
#else
      EXPECT_TRUE(log_output.empty()) << "=== LOG ===\n" << log_output << "\n===========\n";
#endif
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
  restore_env("HOLOSCAN_LOG_LEVEL", env_orig);
}

// REQ: EXPECTED-020
TEST(Logger, CompileTimeThresholdDoesNotEvaluateDisabledTraceArguments) {
#if HOLOSCAN_LOG_ACTIVE_LEVEL <= HOLOSCAN_LOG_LEVEL_TRACE
  int trace_argument_evaluations = 0;
#else
  const int trace_argument_evaluations = 0;
#endif
  const auto orig_level = log_level();
  set_log_level(LogLevel::OFF);

  HOLOSCAN_LOG_TRACE("TRACE_COMPILE_OUT_MARKER {}", ++trace_argument_evaluations);

#if HOLOSCAN_LOG_ACTIVE_LEVEL <= HOLOSCAN_LOG_LEVEL_TRACE
  EXPECT_EQ(trace_argument_evaluations, 1);
#else
  EXPECT_EQ(trace_argument_evaluations, 0);
#endif

  set_log_level(orig_level);
}

// REQ: EXPECTED-022
TEST(Logger, SourceLocationCapturesMacroCallSite) {
  const auto orig_level = log_level();
  const std::string orig_pattern = Logger::pattern();
  const bool orig_pattern_set_by_user = Logger::log_pattern_set_by_user;
  const auto env_level = get_env_copy("HOLOSCAN_LOG_LEVEL");
  const auto env_format = get_env_copy("HOLOSCAN_LOG_FORMAT");
  unsetenv("HOLOSCAN_LOG_LEVEL");
  unsetenv("HOLOSCAN_LOG_FORMAT");
  set_log_level(LogLevel::ERROR);
  set_log_pattern("[%s:%#:%!] %v");

  testing::internal::CaptureStderr();
  const int expected_line = __LINE__ + 1;
  HOLOSCAN_LOG_ERROR("holoscan-source-location-marker");
  const std::string log_output = testing::internal::GetCapturedStderr();

  EXPECT_NE(log_output.find("logger.cpp:" + std::to_string(expected_line)), std::string::npos)
      << log_output;
  EXPECT_NE(log_output.find("SourceLocationCapturesMacroCallSite"), std::string::npos)
      << log_output;
  EXPECT_NE(log_output.find("holoscan-source-location-marker"), std::string::npos) << log_output;

  set_log_level(orig_level);
  set_log_pattern(orig_pattern);
  Logger::log_pattern_set_by_user = orig_pattern_set_by_user;
  restore_env("HOLOSCAN_LOG_LEVEL", env_level);
  restore_env("HOLOSCAN_LOG_FORMAT", env_format);
}

////////////////////////////////////////////////////////////////////////////////
// Test cases for SpdlogLogger
////////////////////////////////////////////////////////////////////////////////

namespace {
constexpr const char* kLogTestString = "Test log message";
constexpr const char* kLogFilePath = "/tmp/test_log_file";
}  // namespace

// Thin SpdlogLogger subclass that exposes a public create() factory. The base class constructor
// is protected; this facade lets tests call log() and redirect() without an operator graph.
class MockSpdlogLogger : public holoscan::logger::SpdlogLogger {
 public:
  explicit MockSpdlogLogger(const char* name) : SpdlogLogger(name) {}

  static std::shared_ptr<MockSpdlogLogger> create() {
    return std::make_shared<MockSpdlogLogger>("mock_logger");
  }
};

// Fixture that opens a temporary file at kLogFilePath and points SpdlogLogger::redirect() at it.
// get_file_content() reads back the first line for level-keyword assertions. TearDown closes it.
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
  const std::string file_content = get_file_content();

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
  const std::string file_content = get_file_content();

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
  const std::string file_content = get_file_content();

  EXPECT_NE(file_content.find(std::string(kLogTestString)), std::string::npos) << file_content;
  EXPECT_NE(file_content.find(std::string("info")), std::string::npos) << file_content;
}

TEST_F(RedirectLogTest, Warn) {
  auto logger = MockSpdlogLogger::create();

  logger->redirect(HOLOSCAN_LOG_LEVEL_WARN, file_);
  logger->level(HOLOSCAN_LOG_LEVEL_WARN);
  logger->log(__FILE__, __LINE__, "test", HOLOSCAN_LOG_LEVEL_WARN, kLogTestString);
  const std::string file_content = get_file_content();

  EXPECT_NE(file_content.find(std::string(kLogTestString)), std::string::npos) << file_content;
  EXPECT_NE(file_content.find(std::string("warn")), std::string::npos) << file_content;
}

TEST_F(RedirectLogTest, Error) {
  auto logger = MockSpdlogLogger::create();

  logger->redirect(HOLOSCAN_LOG_LEVEL_ERROR, file_);
  logger->level(HOLOSCAN_LOG_LEVEL_ERROR);
  logger->log(__FILE__, __LINE__, "test", HOLOSCAN_LOG_LEVEL_ERROR, kLogTestString);
  const std::string file_content = get_file_content();

  EXPECT_NE(file_content.find(std::string(kLogTestString)), std::string::npos) << file_content;
  EXPECT_NE(file_content.find(std::string("error")), std::string::npos) << file_content;
}

TEST_F(RedirectLogTest, Critical) {
  auto logger = MockSpdlogLogger::create();

  logger->redirect(HOLOSCAN_LOG_LEVEL_CRITICAL, file_);
  logger->level(HOLOSCAN_LOG_LEVEL_CRITICAL);
  logger->log(__FILE__, __LINE__, "test", HOLOSCAN_LOG_LEVEL_CRITICAL, kLogTestString);
  const std::string file_content = get_file_content();

  EXPECT_NE(file_content.find(std::string(kLogTestString)), std::string::npos) << file_content;
  EXPECT_NE(file_content.find(std::string("critical")), std::string::npos) << file_content;
}

TEST_F(RedirectLogTest, NullPointer) {
  auto logger = MockSpdlogLogger::create();

  logger->redirect(HOLOSCAN_LOG_LEVEL_CRITICAL, nullptr);
  logger->level(HOLOSCAN_LOG_LEVEL_CRITICAL);
  logger->log(__FILE__, __LINE__, "test", HOLOSCAN_LOG_LEVEL_CRITICAL, kLogTestString);
  const std::string file_content = get_file_content();

  // No log message should be written to the file
  EXPECT_EQ(file_content.find(std::string(kLogTestString)), std::string::npos) << file_content;
  EXPECT_EQ(file_content.find(std::string("critical")), std::string::npos) << file_content;
}

}  // namespace holoscan
