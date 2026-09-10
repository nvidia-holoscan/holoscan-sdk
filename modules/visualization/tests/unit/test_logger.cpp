// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <thread>

#include <holoviz/logger.hpp>

#include <gtest/gtest.h>

namespace holoviz {
namespace {

// LogHandler is a plain noexcept function pointer, so the blocking handler coordinates with
// the test through file-scope atomics (reset by the fixture) and through user_data.
std::atomic<bool> g_handler_entered{false};
std::atomic<bool> g_release_handler{false};
std::atomic<int> g_blocking_handler_calls{0};
std::atomic<int> g_replacement_handler_calls{0};

void blocking_handler(LogLevel, const char*, int, const char*, const char*,
                      void* user_data) noexcept {
  g_blocking_handler_calls.fetch_add(1, std::memory_order_acq_rel);
  g_handler_entered.store(true, std::memory_order_release);
  while (!g_release_handler.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  // Touch user_data as the LAST step: the teardown guarantee is precisely that the owner
  // cannot have invalidated it yet, because set/reset has not been allowed to return.
  static_cast<std::atomic<int>*>(user_data)->fetch_add(1, std::memory_order_acq_rel);
}

void replacement_handler(LogLevel, const char*, int, const char*, const char*, void*) noexcept {
  g_replacement_handler_calls.fetch_add(1, std::memory_order_acq_rel);
}

[[nodiscard]] bool wait_until_true(const std::atomic<bool>& flag,
                                   std::chrono::milliseconds budget) {
  const auto deadline = std::chrono::steady_clock::now() + budget;
  while (!flag.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

class LoggerHandlerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Pin the effective level regardless of the environment: set_log_level() lets
    // HOLOVIZ_LOG_LEVEL override its argument, so clear it first. No test-spawned threads
    // exist yet, so unsetenv() is safe here.
    ::unsetenv("HOLOVIZ_LOG_LEVEL");
    set_log_level(LogLevel::kInfo);
    g_handler_entered.store(false);
    g_release_handler.store(false);
    g_blocking_handler_calls.store(0);
    g_replacement_handler_calls.store(0);
  }
  void TearDown() override {
    // Open the gate before taking the exclusive handler lock so a failed assertion mid-test
    // cannot deadlock teardown behind a still-blocked shared-lock holder.
    g_release_handler.store(true, std::memory_order_release);
    reset_log_handler();
  }
};

// Regression test for the g_handler_mutex teardown guarantee (logger.cpp):
// reset_log_handler() must not return while a previously installed handler is still
// executing (and can therefore still access user_data).
//
// Determinism: a regressed implementation (no exclusive lock held across the swap) returns
// from reset essentially immediately, so the observation window catches it reliably. A
// correct implementation can never set reset_returned before the gate opens, so the
// assertion cannot false-fail.
TEST_F(LoggerHandlerTest, ResetDoesNotReturnWhileCallbackInFlight) {
  std::atomic<int> user_data_touches{0};
  set_log_handler(&blocking_handler, &user_data_touches);

  std::thread logger_thread([] { log_literal(LogLevel::kError, "blocking message"); });
  ASSERT_TRUE(wait_until_true(g_handler_entered, std::chrono::seconds(5)))
      << "handler was never entered";

  std::atomic<bool> reset_returned{false};
  std::thread reset_thread([&reset_returned] {
    reset_log_handler();
    reset_returned.store(true, std::memory_order_release);
  });

  const auto observe_until = std::chrono::steady_clock::now() + std::chrono::milliseconds(150);
  while (std::chrono::steady_clock::now() < observe_until) {
    ASSERT_FALSE(reset_returned.load(std::memory_order_acquire))
        << "reset_log_handler() returned while the callback was still running";
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  g_release_handler.store(true, std::memory_order_release);
  logger_thread.join();
  reset_thread.join();

  EXPECT_TRUE(reset_returned.load(std::memory_order_acquire));
  // user_data was accessed exactly once, from inside the still-protected callback.
  EXPECT_EQ(user_data_touches.load(), 1);
  EXPECT_EQ(g_blocking_handler_calls.load(), 1);

  // Dispatch after reset must not reach the old handler.
  log_literal(LogLevel::kError, "after reset");
  EXPECT_EQ(g_blocking_handler_calls.load(), 1);
}

// Same guarantee for the replacement path: set_log_handler() takes the same exclusive lock,
// so it must not return while the previous handler is still executing.
TEST_F(LoggerHandlerTest, ReplaceDoesNotReturnWhileCallbackInFlight) {
  std::atomic<int> user_data_touches{0};
  set_log_handler(&blocking_handler, &user_data_touches);

  std::thread logger_thread([] { log_literal(LogLevel::kError, "blocking message"); });
  ASSERT_TRUE(wait_until_true(g_handler_entered, std::chrono::seconds(5)))
      << "handler was never entered";

  std::atomic<bool> replace_returned{false};
  std::thread replace_thread([&replace_returned] {
    set_log_handler(&replacement_handler, nullptr);
    replace_returned.store(true, std::memory_order_release);
  });

  const auto observe_until = std::chrono::steady_clock::now() + std::chrono::milliseconds(150);
  while (std::chrono::steady_clock::now() < observe_until) {
    ASSERT_FALSE(replace_returned.load(std::memory_order_acquire))
        << "set_log_handler() returned while the previous callback was still running";
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  g_release_handler.store(true, std::memory_order_release);
  logger_thread.join();
  replace_thread.join();

  EXPECT_TRUE(replace_returned.load(std::memory_order_acquire));
  EXPECT_EQ(user_data_touches.load(), 1);
  EXPECT_EQ(g_blocking_handler_calls.load(), 1);

  // Dispatch after the swap reaches only the new handler.
  log_literal(LogLevel::kError, "after swap");
  EXPECT_EQ(g_blocking_handler_calls.load(), 1);
  EXPECT_EQ(g_replacement_handler_calls.load(), 1);
}

// Level-parse coverage. The first-dispatch env initialization is once-only and therefore
// not order-independently testable in-process; set_log_level() re-reads HOLOVIZ_LOG_LEVEL
// on every call, so it pins the parser deterministically. setenv()/unsetenv() are safe:
// no test-spawned threads run during this test.
TEST(LoggerLevelParseTest, SetLogLevelEnvOverrideParsesAllTokensCaseInsensitively) {
  struct Case {
    const char* env;
    LogLevel expected;
  };
  const Case cases[] = {
      {"trace", LogLevel::kTrace},
      {"DEBUG", LogLevel::kDebug},
      {"iNfO", LogLevel::kInfo},
      {"warn", LogLevel::kWarn},
      {"ERROR", LogLevel::kError},
      {"critical", LogLevel::kCritical},
      {"off", LogLevel::kOff},
  };
  for (const auto& c : cases) {
    ASSERT_EQ(::setenv("HOLOVIZ_LOG_LEVEL", c.env, 1), 0);
    // Request a different level than the env asks for, so passing proves the env token won.
    set_log_level(c.expected == LogLevel::kOff ? LogLevel::kTrace : LogLevel::kOff);
    EXPECT_EQ(log_level(), c.expected) << "env token: " << c.env;
  }

  // Unrecognized token falls back to the requested level (full-match only: no prefixes).
  ASSERT_EQ(::setenv("HOLOVIZ_LOG_LEVEL", "WARNING", 1), 0);
  set_log_level(LogLevel::kDebug);
  EXPECT_EQ(log_level(), LogLevel::kDebug);

  ASSERT_EQ(::unsetenv("HOLOVIZ_LOG_LEVEL"), 0);
  set_log_level(LogLevel::kInfo);
  EXPECT_EQ(log_level(), LogLevel::kInfo);
}

}  // namespace
}  // namespace holoviz
