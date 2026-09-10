/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <holomq/logger.hpp>
#include <holomq/publisher.hpp>

#if defined(HOLOMQ_TRANSPORT_IOX2)
#include <holomq/context_builder.hpp>

#include "iox2/log.hpp"
#endif

#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

namespace {

struct LogEntry {
  holomq::LogLevel level{};
  std::string file;
  int line = 0;
  std::string function;
  std::string message;
};

std::vector<LogEntry> g_entries;

void test_log_handler(const holomq::LogLevel level, const char* file, const int line,
                      const char* function, const char* message, void* /*user_data*/) noexcept {
  g_entries.push_back(LogEntry{
      level,
      file != nullptr ? file : "",
      line,
      function != nullptr ? function : "",
      message != nullptr ? message : "",
  });
}

std::optional<std::string> get_env_copy(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return std::nullopt;
  }
  return std::string{value};
}

void restore_env(const char* name, const std::optional<std::string>& value) {
  if (value) {
    setenv(name, value->c_str(), 1);
  } else {
    unsetenv(name);
  }
}

}  // namespace

class LoggerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_entries.clear();
    holomq::reset_log_handler();
    holomq::disable_transport_logging();
    holomq::set_log_level(holomq::LogLevel::Trace);
    saved_env_ = get_env_copy("HOLOMQ_LOG_LEVEL");
    unsetenv("HOLOMQ_LOG_LEVEL");
  }

  void TearDown() override {
    holomq::disable_transport_logging();
    holomq::reset_log_handler();
    restore_env("HOLOMQ_LOG_LEVEL", saved_env_);
  }

  std::optional<std::string> saved_env_;
};

TEST_F(LoggerTest, HandlerReceivesFormattedMessage) {
  holomq::set_log_handler(test_log_handler, nullptr);
  HOLOMQ_LOG_INFO("hello {}", 42);

  ASSERT_EQ(g_entries.size(), 1U);
  EXPECT_EQ(g_entries[0].level, holomq::LogLevel::Info);
  EXPECT_EQ(g_entries[0].message, "hello 42");
  EXPECT_FALSE(g_entries[0].file.empty());
}

TEST_F(LoggerTest, LogLevelFiltersMessages) {
  holomq::set_log_handler(test_log_handler, nullptr);
  holomq::set_log_level(holomq::LogLevel::Warn);

  HOLOMQ_LOG_INFO("suppressed");
  HOLOMQ_LOG_WARN("visible");

  ASSERT_EQ(g_entries.size(), 1U);
  EXPECT_EQ(g_entries[0].level, holomq::LogLevel::Warn);
  EXPECT_EQ(g_entries[0].message, "visible");
}

TEST_F(LoggerTest, ResetHandlerStopsCapture) {
  holomq::set_log_handler(test_log_handler, nullptr);
  HOLOMQ_LOG_INFO("first");
  holomq::reset_log_handler();
  HOLOMQ_LOG_INFO("second");

  ASSERT_EQ(g_entries.size(), 1U);
  EXPECT_EQ(g_entries[0].message, "first");
}

TEST_F(LoggerTest, LogLiteralIsNoexceptFriendly) {
  holomq::set_log_handler(test_log_handler, nullptr);
  holomq::log_literal(holomq::LogLevel::Error, "literal message");

  ASSERT_EQ(g_entries.size(), 1U);
  EXPECT_EQ(g_entries[0].level, holomq::LogLevel::Error);
  EXPECT_EQ(g_entries[0].message, "literal message");
}

TEST_F(LoggerTest, PublisherSendFlowControlOutcomesAreTraceNotErrors) {
  holomq::set_log_handler(test_log_handler, nullptr);

  holomq::detail::log_publisher_send_error(
      holomq::Error{holomq::ErrorCode::kNotReady, "subscriber discovery pending"});
  holomq::detail::log_publisher_send_error(
      holomq::Error{holomq::ErrorCode::kBackpressured, "queue capacity unavailable"});
  holomq::detail::log_publisher_send_error(
      holomq::Error{holomq::ErrorCode::kFailure, "transport invariant failed"});

  ASSERT_EQ(g_entries.size(), 3U);
  EXPECT_EQ(g_entries[0].level, holomq::LogLevel::Trace);
  EXPECT_EQ(g_entries[1].level, holomq::LogLevel::Trace);
  EXPECT_EQ(g_entries[2].level, holomq::LogLevel::Error);
  EXPECT_NE(g_entries[0].message.find("kNotReady"), std::string::npos);
  EXPECT_NE(g_entries[1].message.find("kBackpressured"), std::string::npos);
  EXPECT_NE(g_entries[2].message.find("kFailure"), std::string::npos);
}

TEST_F(LoggerTest, EnvOverridesLogLevel) {
  setenv("HOLOMQ_LOG_LEVEL", "ERROR", 1);
  holomq::set_log_handler(test_log_handler, nullptr);
  holomq::set_log_level(holomq::LogLevel::Trace);

  HOLOMQ_LOG_WARN("suppressed");
  HOLOMQ_LOG_ERROR("visible");

  ASSERT_EQ(g_entries.size(), 1U);
  EXPECT_EQ(g_entries[0].level, holomq::LogLevel::Error);
}

#if defined(HOLOMQ_TRANSPORT_IOX2)
TEST_F(LoggerTest, EnvOverridesEnableTransportLoggingLevel) {
  setenv("HOLOMQ_LOG_LEVEL", "ERROR", 1);
  holomq::set_log_handler(test_log_handler, nullptr);
  holomq::enable_transport_logging(holomq::LogLevel::Trace);

  iox2::log(iox2::LogLevel::Warn, "test_origin", "suppressed");
  iox2::log(iox2::LogLevel::Error, "test_origin", "visible");

  ASSERT_EQ(g_entries.size(), 1U);
  EXPECT_EQ(g_entries[0].level, holomq::LogLevel::Error);
  EXPECT_NE(g_entries[0].message.find("visible"), std::string::npos);
}
#endif

#if defined(HOLOMQ_TRANSPORT_IOX2)
TEST_F(LoggerTest, TransportLoggingForwardsIox2Messages) {
  holomq::set_log_handler(test_log_handler, nullptr);
  holomq::enable_transport_logging(holomq::LogLevel::Trace);

  iox2::log(iox2::LogLevel::Warn, "test_origin", "transport message");

  ASSERT_GE(g_entries.size(), 1U);
  const auto& entry = g_entries.back();
  EXPECT_EQ(entry.level, holomq::LogLevel::Warn);
  EXPECT_EQ(entry.file, "iox2");
  EXPECT_EQ(entry.function, "test_origin");
  EXPECT_NE(entry.message.find("transport message"), std::string::npos);
}

TEST_F(LoggerTest, TransportLoggingCanBeEnabledAfterContextCreation) {
  auto context = holomq::ContextBuilder().create();
  ASSERT_TRUE(context.has_value());

  holomq::set_log_handler(test_log_handler, nullptr);
  holomq::enable_transport_logging(holomq::LogLevel::Trace);

  iox2::log(iox2::LogLevel::Warn, "test_origin", "late transport message");

  ASSERT_GE(g_entries.size(), 1U);
  const auto& entry = g_entries.back();
  EXPECT_EQ(entry.level, holomq::LogLevel::Warn);
  EXPECT_EQ(entry.file, "iox2");
  EXPECT_EQ(entry.function, "test_origin");
  EXPECT_NE(entry.message.find("late transport message"), std::string::npos);
}
#endif
