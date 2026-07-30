/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <holoscan/core/executors/gxf/gxf_executor.hpp>
#include <holoscan/core/schedulers/gxf/greedy_scheduler.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/ping_rx/ping_rx.hpp>
#include <holoscan/operators/ping_tx/ping_tx.hpp>

#include "async_data_logger_test_helpers.hpp"

namespace holoscan {
namespace {

/**
 * @brief Test application that uses a slow async logger with built-in ping operators.
 */
class SlowLoggerTestApp : public Application {
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

class AsyncDataLoggerShutdownTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Reset GXFExecutor static interrupt flags between tests to prevent state leakage
    // from previous tests that raised SIGINT
    holoscan::gxf::GXFExecutor::reset_interrupt_flags();
  }
  void TearDown() override {
    // Clean up interrupt flags after each test
    holoscan::gxf::GXFExecutor::reset_interrupt_flags();

    // Allow time for any detached countdown threads to see the cancelled flag
    // before starting the next test. This prevents race conditions where
    // a thread from a previous test might interfere with the next test.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
};

/**
 * @brief Test that shutdown timeout correctly limits waiting time and drops remaining entries.
 *
 * This test configures:
 * - 20 iterations (generating ~40 log entries for inputs/outputs)
 * - 100ms processing delay per entry
 * - 0.5s shutdown timeout
 *
 * Expected behavior:
 * - Not all entries will be processed before timeout
 * - The dropped count should be > 0
 * - Shutdown should complete within a reasonable time (not hang)
 */
TEST_F(AsyncDataLoggerShutdownTest, TimeoutDropsRemainingEntries) {
  auto app = make_application<SlowLoggerTestApp>();
  app->set_num_iterations(20);            // Generate ~40 log entries
  app->set_process_delay_ms(100);         // 100ms per entry = ~4 seconds to process all
  app->set_shutdown_wait_period_ms(500);  // Only wait 500ms during shutdown

  auto start_time = std::chrono::steady_clock::now();
  app->run();
  auto end_time = std::chrono::steady_clock::now();

  auto duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  auto logger = app->get_slow_logger();
  ASSERT_NE(logger, nullptr);

  size_t processed = logger->get_entries_processed();
  size_t dropped = logger->get_data_dropped_count();

  HOLOSCAN_LOG_INFO(
      "Test results: processed={}, dropped={}, duration={}ms", processed, dropped, duration_ms);

  // Verify that some entries were dropped due to timeout
  // (not all 40 entries could be processed in 0.5s with 100ms delay each)
  EXPECT_GT(dropped, 0) << "Expected some entries to be dropped due to timeout";

  // Verify that some entries were processed
  EXPECT_GT(processed, 0) << "Expected some entries to be processed";

  // Verify that shutdown didn't take too long (timeout should have kicked in)
  // Allow some margin for test execution overhead
  EXPECT_LT(duration_ms, 5000) << "Shutdown took too long, timeout may not have worked";
}

/**
 * @brief Test that with infinite timeout (-1), all entries are processed.
 *
 * This test uses a shorter number of iterations and faster processing to ensure
 * it completes in reasonable time while still verifying the infinite timeout behavior.
 */
TEST_F(AsyncDataLoggerShutdownTest, InfiniteTimeoutProcessesAllEntries) {
  auto app = make_application<SlowLoggerTestApp>();
  app->set_num_iterations(5);            // Generate ~10 log entries
  app->set_process_delay_ms(10);         // 10ms per entry = ~100ms to process all
  app->set_shutdown_wait_period_ms(-1);  // Wait indefinitely

  app->run();

  auto logger = app->get_slow_logger();
  ASSERT_NE(logger, nullptr);

  size_t dropped = logger->get_data_dropped_count();

  // With infinite timeout, no entries should be dropped
  EXPECT_EQ(dropped, 0) << "Expected no entries to be dropped with infinite timeout";
}

/**
 * @brief Verifies visible shutdown logging (resource name, drain progress, completion summary).
 *
 * Ensures that INFO-level logs are emitted during async logger shutdown so that
 * users see progress instead of a silent wait that can look like a hang.
 */
TEST_F(AsyncDataLoggerShutdownTest, ShutdownEmitsVisibleProgressLogs) {
  auto app = make_application<SlowLoggerTestApp>();
  app->set_num_iterations(30);
  app->set_process_delay_ms(200);
  app->set_shutdown_wait_period_ms(-1);

  testing::internal::CaptureStderr();
  app->run();
  std::string log_output = testing::internal::GetCapturedStderr();

  EXPECT_NE(log_output.find("slow_logger"), std::string::npos)
      << "Expected logger resource name in shutdown logs\n"
      << log_output;
  EXPECT_NE(log_output.find("shutdown requested, draining asynchronous log queues"),
            std::string::npos)
      << "Expected shutdown entry log\n"
      << log_output;
  EXPECT_NE(log_output.find("Draining log queues"), std::string::npos)
      << "Expected periodic drain progress log\n"
      << log_output;
  EXPECT_NE(log_output.find("shutdown queue drain complete"), std::string::npos)
      << "Expected shutdown completion summary log\n"
      << log_output;
}

/**
 * @brief Test that zero timeout immediately stops without processing remaining entries.
 */
TEST_F(AsyncDataLoggerShutdownTest, ZeroTimeoutStopsImmediately) {
  auto app = make_application<SlowLoggerTestApp>();
  app->set_num_iterations(20);          // Generate ~40 log entries
  app->set_process_delay_ms(100);       // 100ms per entry
  app->set_shutdown_wait_period_ms(0);  // Don't wait at all

  auto start_time = std::chrono::steady_clock::now();
  app->run();
  auto end_time = std::chrono::steady_clock::now();

  auto duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  auto logger = app->get_slow_logger();
  ASSERT_NE(logger, nullptr);

  size_t dropped = logger->get_data_dropped_count();

  HOLOSCAN_LOG_INFO("Zero timeout test: dropped={}, duration={}ms", dropped, duration_ms);

  // With zero timeout, we expect entries to be dropped
  EXPECT_GT(dropped, 0) << "Expected entries to be dropped with zero timeout";
}

/**
 * @brief Test that interrupt signal (Ctrl+C) triggers data logger shutdown before watchdog.
 *
 * This test verifies that when SIGINT is received:
 * 1. The signal handler calls shutdown_data_loggers() before starting the watchdog
 * 2. The app terminates gracefully without the watchdog force-killing it
 * 3. Entries are processed during the shutdown drain period (not just before SIGINT)
 *
 * Test scenario timing:
 * - The operators run quickly (no PeriodicCondition), so the scheduler will reach deadlock
 *   and stop BEFORE SIGINT is sent (200 iterations complete in < 500ms)
 * - However, the data logger has a slow process_delay_ms (50ms per entry), so the logger
 *   queue is still being drained when SIGINT arrives
 * - This tests the case where SIGINT arrives AFTER scheduler termination but WHILE the
 *   data logger is still processing queued entries
 * - The signal handler's interrupt() will return false (graph already stopped), but
 *   shutdown_data_loggers() is still called to drain remaining entries
 *
 * Note: We intentionally do not use EXPECT_EXIT here because GoogleTest's death tests use fork(),
 * which is unsafe in multi-threaded contexts. Since Holoscan applications spawn multiple threads,
 * using EXPECT_EXIT would trigger warnings and potentially cause undefined behavior.
 * Instead, we rely on Holoscan's signal handler to consume the SIGINT gracefully.
 */
TEST_F(AsyncDataLoggerShutdownTest, InterruptSignalTriggersDataLoggerShutdown) {
  // Timing parameters for threshold calculation:
  // - pre_signal_wait_ms: time before SIGINT is sent
  // - shutdown_wait_period_ms: time allowed for drain during shutdown
  // - process_delay_ms: time to process each log entry (slows down LOGGER, not operators)
  // - num_iterations: total iterations (generates 1 log entry per iteration)
  //
  // Note: The operators run without delay, so scheduler will reach deadlock quickly.
  // The process_delay_ms only affects how fast the logger drains its queue.
  constexpr int64_t pre_signal_wait_ms = 500;
  constexpr int64_t shutdown_wait_period_ms = 2000;
  constexpr int64_t process_delay_ms = 50;
  constexpr int num_iterations = 200;

  // Calculate expected entry counts:
  // - max_pre_signal: maximum entries that could be processed before SIGINT
  // - min_with_shutdown_drain: threshold that proves shutdown drain occurred
  // - max_total_entries: maximum possible entries
  constexpr size_t max_pre_signal = pre_signal_wait_ms / process_delay_ms;  // ~10 entries
  constexpr size_t min_with_shutdown_drain = max_pre_signal + 5;            // require > 15
  constexpr size_t max_total_entries = num_iterations;                      // 200 entries

  auto app = make_application<SlowLoggerTestApp>();
  app->set_num_iterations(num_iterations);  // Operators complete quickly, logger drains slowly
  app->set_process_delay_ms(process_delay_ms);
  app->set_shutdown_wait_period_ms(shutdown_wait_period_ms);

  // Run the app asynchronously
  auto future = app->run_async();

  // Wait a bit for the app to start and generate some log entries
  std::this_thread::sleep_for(std::chrono::milliseconds(pre_signal_wait_ms));

  // Send SIGINT to trigger the signal handler (simulates Ctrl+C)
  HOLOSCAN_LOG_INFO("Sending SIGINT to test interrupt handler...");
  std::raise(SIGINT);

  // Wait for the app to finish (should complete via signal handler, not naturally)
  auto status = future.wait_for(std::chrono::seconds(10));
  ASSERT_EQ(status, std::future_status::ready)
      << "App did not shut down within expected time after SIGINT";

  auto logger = app->get_slow_logger();
  ASSERT_NE(logger, nullptr);

  size_t processed = logger->get_entries_processed();
  HOLOSCAN_LOG_INFO("Interrupt test: processed {} entries (expected range: {}-{})",
                    processed,
                    min_with_shutdown_drain,
                    max_total_entries);

  // Verify processed count is within expected bounds:
  // - More than min_with_shutdown_drain (~15): proves shutdown drain occurred
  EXPECT_GT(processed, min_with_shutdown_drain)
      << "Expected more entries to be processed during shutdown drain period";

  // - Less than max_total_entries (200): proves interrupt stopped app early
  EXPECT_LT(processed, max_total_entries) << "Expected interrupt to stop app before completion";
}

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
TEST_F(AsyncDataLoggerShutdownTest, DistributedAppInterruptSignalTriggersDataLoggerShutdown) {
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

/**
 * @brief Slow test application for testing interrupt during active execution.
 *
 * Uses a PeriodicCondition to throttle operator execution, ensuring the scheduler
 * is still actively running when SIGINT is sent.
 */
class SlowExecutionTestApp : public Application {
 public:
  void set_execution_period(const std::string& period) { execution_period_ = period; }
  void set_num_iterations(int n) { num_iterations_ = n; }

  void compose() override {
    using namespace holoscan;

    // Use a PeriodicCondition to throttle execution - this ensures the scheduler
    // is still running when we send SIGINT
    // recess_period is a string like "10ms", "100ms", "1s", etc.
    auto tx = make_operator<ops::PingTxOp>(
        "tx",
        make_condition<CountCondition>(num_iterations_),
        make_condition<PeriodicCondition>("periodic", Arg("recess_period", execution_period_)));
    auto rx = make_operator<ops::PingRxOp>("rx");

    add_flow(tx, rx);
  }

 private:
  std::string execution_period_ = "10ms";  // 10ms between executions
  int num_iterations_ = 100;
};

/**
 * @brief Test that GxfGraphInterrupt() provides immediate scheduler shutdown.
 *
 * This test validates that with the GxfGraphInterrupt() approach, the scheduler
 * stops quickly rather than waiting for stop_on_deadlock_timeout. The key improvement
 * is that shutdown should complete in ~1-2 seconds, not the full stop_on_deadlock_timeout.
 *
 * We verify this by:
 * 1. Configuring a LONG stop_on_deadlock_timeout (30 seconds)
 * 2. Running an app with slow execution (PeriodicCondition) so it won't finish naturally
 * 3. Sending SIGINT while the scheduler is actively running
 * 4. Measuring total shutdown time
 * 5. Verifying it completes quickly (< 5 seconds), NOT 30+ seconds
 *
 * Historical context: The old EVENT_NEVER soft-stop approach would wait for
 * stop_on_deadlock_timeout before the scheduler would exit. GxfGraphInterrupt()
 * bypasses this wait entirely by directly signaling the scheduler to stop.
 */
TEST_F(AsyncDataLoggerShutdownTest, InterruptStopsSchedulerImmediately) {
  // Configuration:
  // - Long stop_on_deadlock_timeout to prove we don't wait for it
  // - Slow execution (10ms period) so scheduler is still running when SIGINT sent
  // - Many iterations so app won't finish naturally before SIGINT
  constexpr int64_t stop_on_deadlock_timeout_ms = 30000;  // 30 seconds - we should NOT wait this!
  const std::string execution_period = "10ms";            // 10ms between operator executions
  constexpr int64_t pre_signal_wait_ms = 500;             // Wait 500ms before sending SIGINT
  constexpr int num_iterations = 1000;  // Would take ~10 seconds to complete naturally

  auto app = make_application<SlowExecutionTestApp>();
  app->set_num_iterations(num_iterations);
  app->set_execution_period(execution_period);

  // Configure scheduler with a long stop_on_deadlock_timeout.
  // With the old soft-stop approach (EVENT_NEVER), shutdown would wait this long.
  // With GxfGraphInterrupt(), shutdown should be immediate regardless of this value.
  auto scheduler = app->make_scheduler<GreedyScheduler>(
      "greedy-scheduler",
      Arg("stop_on_deadlock", true),
      Arg("stop_on_deadlock_timeout", stop_on_deadlock_timeout_ms));
  app->scheduler(scheduler);

  auto future = app->run_async();

  // Wait for app to start running and process some messages
  std::this_thread::sleep_for(std::chrono::milliseconds(pre_signal_wait_ms));

  // Record time before signal and send SIGINT
  auto before_signal = std::chrono::steady_clock::now();
  HOLOSCAN_LOG_INFO("Sending SIGINT to test immediate scheduler stop...");
  std::raise(SIGINT);

  // Wait for app to finish - use a longer timeout since we're not racing the force exit
  auto status = future.wait_for(std::chrono::seconds(15));
  ASSERT_EQ(status, std::future_status::ready)
      << "App did not shut down within expected time after SIGINT";

  auto after_shutdown = std::chrono::steady_clock::now();
  auto shutdown_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(after_shutdown - before_signal).count();

  HOLOSCAN_LOG_INFO(
      "Immediate stop test: shutdown completed in {} ms (stop_on_deadlock_timeout was {} ms)",
      shutdown_time_ms,
      stop_on_deadlock_timeout_ms);

  // With GxfGraphInterrupt(), scheduler should stop almost immediately.
  // Total time should be roughly: scheduler stop (~immediate) + cleanup overhead
  // Should be well under 10 seconds total, NOT 30+ seconds (stop_on_deadlock_timeout).
  //
  // This proves that GxfGraphInterrupt() bypasses the stop_on_deadlock_timeout wait entirely.
  // The old EVENT_NEVER soft-stop approach would have waited the full 30 seconds.
  EXPECT_LT(shutdown_time_ms, 10000)
      << "Shutdown took too long (" << shutdown_time_ms << " ms) - GxfGraphInterrupt() should "
      << "provide immediate scheduler stop, not wait for stop_on_deadlock_timeout ("
      << stop_on_deadlock_timeout_ms << " ms)";

  // Verify we stopped early (didn't process all iterations)
  // With 10ms period and 500ms wait, we should have processed ~50 iterations, not 10000
  EXPECT_LT(shutdown_time_ms, stop_on_deadlock_timeout_ms)
      << "Shutdown time should be much less than stop_on_deadlock_timeout";
}

/**
 * @brief Test that interrupt signal works correctly when no data loggers are present.
 *
 * This test verifies the signal handler path when there are no data loggers to drain.
 * The signal handler only calls GxfGraphInterrupt() - the main thread handles all cleanup
 * (including data logger shutdown if any) after its GxfGraphWait() returns.
 *
 * Uses SlowExecutionTestApp (which has no data loggers) with PeriodicCondition to ensure
 * SIGINT arrives while the scheduler is still actively running.
 */
TEST_F(AsyncDataLoggerShutdownTest, InterruptWorksWithoutDataLoggers) {
  // Configuration:
  // - Slow execution (10ms period) so scheduler is still running when SIGINT sent
  // - 1000 iterations at 10ms = 10 seconds, well beyond the 500ms pre-signal wait
  // - Long stop_on_deadlock_timeout to ensure we're testing GxfGraphInterrupt(), not natural
  // timeout
  const std::string execution_period = "10ms";            // 10ms between operator executions
  constexpr int64_t pre_signal_wait_ms = 500;             // Wait 500ms before sending SIGINT
  constexpr int num_iterations = 1000;                    // Would take ~10 seconds naturally
  constexpr int64_t stop_on_deadlock_timeout_ms = 30000;  // 30 seconds - ensures we test interrupt

  // SlowExecutionTestApp has no data loggers - just PingTx/PingRx with PeriodicCondition
  auto app = make_application<SlowExecutionTestApp>();
  app->set_num_iterations(num_iterations);
  app->set_execution_period(execution_period);

  // Configure scheduler with long timeout to ensure we're testing interrupt, not natural completion
  auto scheduler = app->make_scheduler<GreedyScheduler>(
      "greedy-scheduler",
      Arg("stop_on_deadlock", true),
      Arg("stop_on_deadlock_timeout", stop_on_deadlock_timeout_ms));
  app->scheduler(scheduler);

  auto future = app->run_async();

  // Wait for app to start running and process some messages
  std::this_thread::sleep_for(std::chrono::milliseconds(pre_signal_wait_ms));

  // Record time before signal and send SIGINT
  auto before_signal = std::chrono::steady_clock::now();
  HOLOSCAN_LOG_INFO("Sending SIGINT to test interrupt without data loggers...");
  std::raise(SIGINT);

  // Wait for app to finish - use a longer timeout since we're not racing the force exit
  auto status = future.wait_for(std::chrono::seconds(15));
  ASSERT_EQ(status, std::future_status::ready)
      << "App without data loggers did not shut down within expected time after SIGINT";

  auto after_shutdown = std::chrono::steady_clock::now();
  auto shutdown_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(after_shutdown - before_signal).count();

  HOLOSCAN_LOG_INFO(
      "No-logger test: shutdown completed in {} ms (stop_on_deadlock_timeout was {} ms)",
      shutdown_time_ms,
      stop_on_deadlock_timeout_ms);

  // With GxfGraphInterrupt(), scheduler should stop almost immediately.
  // Without data loggers, there's no drain time - just scheduler stop + deactivate.
  // Should complete well under 10 seconds, NOT 30+ seconds (stop_on_deadlock_timeout).
  EXPECT_LT(shutdown_time_ms, 10000)
      << "Shutdown took too long (" << shutdown_time_ms << " ms) - GxfGraphInterrupt() should "
      << "provide immediate scheduler stop";

  // Verify we stopped early (didn't wait for stop_on_deadlock_timeout)
  EXPECT_LT(shutdown_time_ms, stop_on_deadlock_timeout_ms)
      << "Shutdown time should be much less than stop_on_deadlock_timeout";
}

}  // namespace holoscan
