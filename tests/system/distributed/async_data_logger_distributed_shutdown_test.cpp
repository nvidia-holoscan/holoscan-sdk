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

#include <atomic>
#include <chrono>
#include <csignal>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <holoscan/core/executors/gxf/gxf_executor.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/ping_rx/ping_rx.hpp>
#include <holoscan/operators/ping_tx/ping_tx.hpp>

#include "../async_data_logger_test_helpers.hpp"

namespace holoscan {
namespace {

/**
 * @brief Test fragment that uses a slow async logger with built-in ping operators.
 *
 * This is the fragment version of SlowLoggerTestApp for use in distributed applications.
 */
class SlowLoggerTestFragment : public Fragment {
 public:
  void set_num_iterations(int n) { num_iterations_ = n; }
  void set_shutdown_wait_period_ms(int64_t ms) { shutdown_wait_period_ms_ = ms; }
  void set_process_delay_ms(int64_t ms) { process_delay_ms_ = ms; }

  void compose() override {
    using namespace holoscan;

    // Use built-in ping operators
    auto tx = make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>(num_iterations_));
    auto rx = make_operator<ops::PingRxOp>("rx");

    // Create a slow async logger with the configured shutdown timeout
    auto logger = make_resource<SlowAsyncLogger>(
        "slow_logger",
        Arg("log_inputs", true),
        Arg("log_outputs", false),
        Arg("shutdown_wait_period_ms", shutdown_wait_period_ms_),
        Arg("enable_large_data_queue", false));  // Disable large data queue for simpler testing
    logger->set_process_delay(std::chrono::milliseconds(process_delay_ms_));
    slow_logger_ = logger;

    // Add the logger to the fragment
    add_data_logger(logger);

    add_flow(tx, rx);
  }

  std::shared_ptr<SlowAsyncLogger> get_slow_logger() const { return slow_logger_; }

 private:
  int num_iterations_ = 20;
  int64_t shutdown_wait_period_ms_ = 500;  // 500ms timeout
  int64_t process_delay_ms_ = 100;         // 100ms per entry
  std::shared_ptr<SlowAsyncLogger> slow_logger_;
};

/**
 * @brief Distributed test application that uses SlowLoggerTestFragment.
 *
 * This creates a multi-fragment application to test signal handling
 * in the distributed (AppWorker) code path.
 */
class SlowLoggerDistributedTestApp : public Application {
 public:
  explicit SlowLoggerDistributedTestApp(const std::vector<std::string>& argv = {})
      : Application(argv) {}

  void set_num_iterations(int n) { num_iterations_ = n; }
  void set_shutdown_wait_period_ms(int64_t ms) { shutdown_wait_period_ms_ = ms; }
  void set_process_delay_ms(int64_t ms) { process_delay_ms_ = ms; }

  void compose() override {
    using namespace holoscan;

    auto fragment = make_fragment<SlowLoggerTestFragment>("slow_logger_fragment");
    slow_logger_fragment_ = std::dynamic_pointer_cast<SlowLoggerTestFragment>(fragment);
    slow_logger_fragment_->set_num_iterations(num_iterations_);
    slow_logger_fragment_->set_shutdown_wait_period_ms(shutdown_wait_period_ms_);
    slow_logger_fragment_->set_process_delay_ms(process_delay_ms_);

    add_fragment(fragment);
  }

  std::shared_ptr<SlowAsyncLogger> get_slow_logger() const {
    return slow_logger_fragment_ ? slow_logger_fragment_->get_slow_logger() : nullptr;
  }

 private:
  int num_iterations_ = 20;
  int64_t shutdown_wait_period_ms_ = 500;  // 500ms timeout
  int64_t process_delay_ms_ = 100;         // 100ms per entry
  std::shared_ptr<SlowLoggerTestFragment> slow_logger_fragment_;
};

}  // namespace

class AsyncDataLoggerDistributedShutdownTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Reset GXFExecutor static interrupt flags between tests to prevent state leakage
    // from previous tests that raised SIGINT
    holoscan::gxf::GXFExecutor::reset_interrupt_flags();
  }
  void TearDown() override {
    // Clean up interrupt flags after each test
    holoscan::gxf::GXFExecutor::reset_interrupt_flags();
  }
};

/**
 * @brief Test that interrupt signal triggers data logger shutdown in distributed applications.
 *
 * This test verifies the signal handler in the AppWorker code path (distributed applications):
 * 1. The signal handler calls shutdown_data_loggers() on each fragment before the watchdog
 * 2. The app terminates gracefully without the watchdog force-killing it
 * 3. Entries are processed during the shutdown drain period (not just before SIGINT)
 *
 * Note: We intentionally do not use EXPECT_EXIT here because GoogleTest's death tests use fork(),
 * which is unsafe in multi-threaded contexts. Since Holoscan applications spawn multiple threads,
 * using EXPECT_EXIT would trigger warnings and potentially cause undefined behavior.
 * Instead, we rely on Holoscan's signal handler to consume the SIGINT gracefully.
 */
TEST_F(AsyncDataLoggerDistributedShutdownTest,
       DistributedAppInterruptSignalTriggersDataLoggerShutdown) {
  // Timing parameters for threshold calculation
  constexpr int64_t pre_signal_wait_ms = 500;
  constexpr int64_t shutdown_wait_period_ms = 4000;
  constexpr int64_t process_delay_ms = 50;
  constexpr int num_iterations = 200;

  // Calculate expected entry counts
  constexpr size_t max_pre_signal = pre_signal_wait_ms / process_delay_ms;  // ~10 entries
  constexpr size_t min_with_shutdown_drain = max_pre_signal + 5;            // require > 15
  constexpr size_t max_total_entries = num_iterations;                      // 200 entries

  // Pass --driver --worker --fragments=all to exercise the AppWorker code path
  std::vector<std::string> args{"test_app", "--driver", "--worker", "--fragments=all"};
  auto app = make_application<SlowLoggerDistributedTestApp>(args);
  app->set_num_iterations(num_iterations);  // Many iterations so app doesn't finish naturally
  app->set_process_delay_ms(process_delay_ms);
  app->set_shutdown_wait_period_ms(shutdown_wait_period_ms);

  // Run the distributed app in a separate thread (run_async doesn't support distributed apps)
  std::atomic<bool> app_finished{false};
  std::thread app_thread([&app, &app_finished]() {
    app->run();
    app_finished.store(true);
  });

  // Wait a bit for the app to start and generate some log entries
  std::this_thread::sleep_for(std::chrono::milliseconds(pre_signal_wait_ms));

  // Send SIGINT to trigger the signal handler (simulates Ctrl+C)
  HOLOSCAN_LOG_INFO("Sending SIGINT to test distributed app interrupt handler...");
  std::raise(SIGINT);

  // Wait for the app thread to finish (should complete via signal handler, not naturally)
  // Poll with timeout to avoid hanging if something goes wrong
  auto start_wait = std::chrono::steady_clock::now();
  while (!app_finished.load() &&
         std::chrono::steady_clock::now() - start_wait < std::chrono::seconds(10)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  ASSERT_TRUE(app_finished.load())
      << "Distributed app did not shut down within expected time after SIGINT";

  app_thread.join();

  auto logger = app->get_slow_logger();
  ASSERT_NE(logger, nullptr);

  size_t processed = logger->get_entries_processed();
  HOLOSCAN_LOG_INFO("Distributed app interrupt test: processed {} entries (expected range: {}-{})",
                    processed,
                    min_with_shutdown_drain,
                    max_total_entries);

  // Verify processed count is within expected bounds:
  // - More than min_with_shutdown_drain (~15): proves shutdown drain occurred
  EXPECT_GT(processed, min_with_shutdown_drain)
      << "Expected more entries to be processed during shutdown drain period";

  // - Less than max_total_entries (200): proves interrupt stopped app early
  EXPECT_LT(processed, max_total_entries)
      << "Expected interrupt to stop distributed app before completion";
}

}  // namespace holoscan
