/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "../env_wrapper.hpp"
#include "distributed_app_fixture.hpp"
#include "utility_apps.hpp"

namespace holoscan {

///////////////////////////////////////////////////////////////////////////////
// Test fixture that unsets HOLOSCAN_DISTRIBUTED_APP_SCHEDULER to allow
// user-specified schedulers to take effect
///////////////////////////////////////////////////////////////////////////////

class DistributedAppSchedulerTest : public DistributedApp {
 protected:
  void SetUp() override {
    // Call parent SetUp first
    DistributedApp::SetUp();

    // Save original value of HOLOSCAN_DISTRIBUTED_APP_SCHEDULER before unsetting
    const char* env_value = std::getenv("HOLOSCAN_DISTRIBUTED_APP_SCHEDULER");
    if (env_value != nullptr) {
      orig_scheduler_env_ = env_value;
      orig_scheduler_env_was_set_ = true;
    } else {
      orig_scheduler_env_was_set_ = false;
    }

    // Unset HOLOSCAN_DISTRIBUTED_APP_SCHEDULER so user-specified schedulers are respected
    // (The CTest environment sets this to multi_thread for SYSTEM_DISTRIBUTED_TEST and
    //  to event_based for SYSTEM_DISTRIBUTED_EBS_TEST).
    unsetenv("HOLOSCAN_DISTRIBUTED_APP_SCHEDULER");
  }

  void TearDown() override {
    // Restore original HOLOSCAN_DISTRIBUTED_APP_SCHEDULER value
    if (orig_scheduler_env_was_set_) {
      setenv("HOLOSCAN_DISTRIBUTED_APP_SCHEDULER", orig_scheduler_env_.c_str(), 1);
    }

    // Call parent TearDown
    DistributedApp::TearDown();
  }

  std::string orig_scheduler_env_;
  bool orig_scheduler_env_was_set_ = false;
};

namespace {

///////////////////////////////////////////////////////////////////////////////
// Test Fragments that set their own scheduler
///////////////////////////////////////////////////////////////////////////////

class TxFragmentWithGreedyScheduler : public holoscan::Fragment {
 public:
  explicit TxFragmentWithGreedyScheduler(int64_t stop_on_deadlock_timeout = 5000)
      : stop_on_deadlock_timeout_(stop_on_deadlock_timeout) {}

  void compose() override {
    using namespace holoscan;

    // Set GreedyScheduler with custom arguments
    auto greedy =
        make_scheduler<GreedyScheduler>("custom-greedy",
                                        Arg("stop_on_deadlock", true),
                                        Arg("stop_on_deadlock_timeout", stop_on_deadlock_timeout_),
                                        Arg("check_recession_period_ms", 5.0));
    scheduler(greedy);

    auto tx = make_operator<PingTxOp>("tx", make_condition<CountCondition>(5));
    add_operator(tx);
  }

 private:
  int64_t stop_on_deadlock_timeout_;
};

class RxFragmentWithMultiThreadScheduler : public holoscan::Fragment {
 public:
  explicit RxFragmentWithMultiThreadScheduler(int64_t worker_thread_number = 4)
      : worker_thread_number_(worker_thread_number) {}

  void compose() override {
    using namespace holoscan;

    // Set MultiThreadScheduler with custom arguments
    auto multithread =
        make_scheduler<MultiThreadScheduler>("custom-multithread",
                                             Arg("stop_on_deadlock", true),
                                             Arg("stop_on_deadlock_timeout", 8000L),
                                             Arg("worker_thread_number", worker_thread_number_));
    scheduler(multithread);

    auto rx = make_operator<PingRxOp>("rx");
    add_operator(rx);
  }

 private:
  int64_t worker_thread_number_;
};

class RxFragmentWithEventBasedScheduler : public holoscan::Fragment {
 public:
  void compose() override {
    using namespace holoscan;

    // Set EventBasedScheduler with custom arguments
    auto event_based = make_scheduler<EventBasedScheduler>("custom-event-based",
                                                           Arg("stop_on_deadlock", true),
                                                           Arg("stop_on_deadlock_timeout", 7000L),
                                                           Arg("worker_thread_number", 2L));
    scheduler(event_based);

    auto rx = make_operator<PingRxOp>("rx");
    add_operator(rx);
  }
};

///////////////////////////////////////////////////////////////////////////////
// Test Applications
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief Application where fragment1 uses GreedyScheduler and fragment2 uses MultiThreadScheduler.
 */
class AppWithMixedSchedulers : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto fragment1 = make_fragment<TxFragmentWithGreedyScheduler>("fragment1");
    auto fragment2 = make_fragment<RxFragmentWithMultiThreadScheduler>("fragment2");

    add_flow(fragment1, fragment2, {{"tx", "rx"}});
  }
};

/**
 * @brief Application where both fragments use GreedyScheduler.
 */
class AppWithGreedySchedulers : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;
    auto fragment1 = make_fragment<TxFragmentWithGreedyScheduler>("fragment1", 6000);
    auto fragment2 = make_fragment<TxFragmentWithGreedyScheduler>("fragment2", 6200);

    // Connect them - note that TxFragmentWithGreedyScheduler only has tx, so we need a different
    // fragment for rx. Let's use OneTxFragment -> OneRxFragment pattern.
    add_fragment(fragment1);
    add_fragment(fragment2);
  }
};

/**
 * @brief Application that sets scheduler at Application level.
 * Fragments without their own scheduler should inherit this.
 */
class AppWithApplicationScheduler : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;

    // Set scheduler at Application level
    auto app_scheduler =
        make_scheduler<MultiThreadScheduler>("app-multithread",
                                             Arg("stop_on_deadlock", true),
                                             Arg("stop_on_deadlock_timeout", 6000L),
                                             Arg("worker_thread_number", 3L));
    scheduler(app_scheduler);

    // These fragments don't set their own scheduler
    auto fragment1 = make_fragment<OneTxFragment>("fragment1");
    auto fragment2 = make_fragment<OneRxFragment>("fragment2");

    add_flow(fragment1, fragment2, {{"tx", "rx"}});
  }
};

/**
 * @brief Application that sets scheduler at Application level, but one fragment overrides it.
 */
class AppWithMixedSchedulerSources : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;

    // Set scheduler at Application level
    auto app_scheduler =
        make_scheduler<MultiThreadScheduler>("app-multithread",
                                             Arg("stop_on_deadlock", true),
                                             Arg("stop_on_deadlock_timeout", 9000L),
                                             Arg("worker_thread_number", 3L));
    scheduler(app_scheduler);

    // fragment1 doesn't set its own scheduler - should inherit from Application
    auto fragment1 = make_fragment<OneTxFragment>("fragment1");
    // fragment2 sets its own GreedyScheduler - should use its own
    auto fragment2 = make_fragment<RxFragmentWithEventBasedScheduler>("fragment2");

    add_flow(fragment1, fragment2, {{"tx", "rx"}});
  }
};

/**
 * @brief Application using GreedyScheduler at app level for distributed app.
 */
class AppWithGreedySchedulerAtAppLevel : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;

    // Set GreedyScheduler at Application level
    auto app_scheduler = make_scheduler<GreedyScheduler>("app-greedy",
                                                         Arg("stop_on_deadlock", true),
                                                         Arg("stop_on_deadlock_timeout", 11000L),
                                                         Arg("check_recession_period_ms", 2.0));
    scheduler(app_scheduler);

    auto fragment1 = make_fragment<OneTxFragment>("fragment1");
    auto fragment2 = make_fragment<OneRxFragment>("fragment2");

    add_flow(fragment1, fragment2, {{"tx", "rx"}});
  }
};

/**
 * @brief Application that sets a scheduler clock at Application level.
 */
class AppWithApplicationSchedulerClock : public holoscan::Application {
 public:
  using Application::Application;

  void compose() override {
    using namespace holoscan;

    auto clock = make_resource<SyntheticClock>("app-synthetic-clock",
                                               Arg("initial_timestamp", int64_t{1234}));
    // Use GreedyScheduler to test that it gets overridden to EventBasedScheduler
    // for multi-fragment apps, while still cloning the clock resource
    auto app_scheduler = make_scheduler<GreedyScheduler>("app-greedy-with-clock",
                                                         Arg("stop_on_deadlock", true),
                                                         Arg("stop_on_deadlock_timeout", 6000L),
                                                         Arg("check_recession_period_ms", 2.0),
                                                         Arg("clock", clock));
    // Explicit cast needed for clang compatibility (implicit shared_ptr<Derived> to
    // shared_ptr<Base> conversion not resolved in overload resolution)
    scheduler(std::static_pointer_cast<Scheduler>(app_scheduler));

    auto fragment1 = make_fragment<OneTxFragment>("fragment1", 2);
    auto fragment2 = make_fragment<OneRxFragment>("fragment2");

    add_flow(fragment1, fragment2, {{"tx", "rx"}});
  }
};

}  // namespace

///////////////////////////////////////////////////////////////////////////////
// Helper function to verify scheduler type on a fragment
///////////////////////////////////////////////////////////////////////////////

template <typename SchedulerT>
bool is_scheduler_type(const std::shared_ptr<Fragment>& fragment) {
  auto scheduler = fragment->scheduler();
  return std::dynamic_pointer_cast<SchedulerT>(scheduler) != nullptr;
}

/**
 * @brief Get the stop_on_deadlock_timeout value from a scheduler.
 *
 * Works with EventBasedScheduler, MultiThreadScheduler, and GreedyScheduler.
 * Returns -1 if the scheduler type is unknown or parameter is not set.
 */
int64_t get_scheduler_deadlock_timeout(const std::shared_ptr<Fragment>& fragment) {
  auto scheduler = fragment->scheduler();
  try {
    if (auto* ebs = dynamic_cast<EventBasedScheduler*>(scheduler.get())) {
      return ebs->stop_on_deadlock_timeout();
    } else if (auto* mts = dynamic_cast<MultiThreadScheduler*>(scheduler.get())) {
      return mts->stop_on_deadlock_timeout();
    } else if (auto* gs = dynamic_cast<GreedyScheduler*>(scheduler.get())) {
      return gs->stop_on_deadlock_timeout();
    }
  } catch (const std::runtime_error&) {
    // Parameter not set
  }
  return -1;
}

///////////////////////////////////////////////////////////////////////////////
// Tests for respecting user-defined scheduler on fragments
///////////////////////////////////////////////////////////////////////////////

TEST_F(DistributedAppSchedulerTest, TestGreedySchedulerOverriddenToEventBased) {
  // Test that GreedyScheduler is automatically overridden to EventBasedScheduler
  // for multi-fragment apps (Greedy causes deadlock in distributed mode)
  auto app = make_application<AppWithMixedSchedulers>();

  testing::internal::CaptureStderr();

  try {
    app->run();
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Exception: {}", e.what());
  }

  std::string log_output = testing::internal::GetCapturedStderr();

  // Verify the app ran successfully
  EXPECT_TRUE(log_output.find("Rx fragment2.rx message received count:") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify warning about GreedyScheduler override was logged
  EXPECT_TRUE(log_output.find("GreedyScheduler is not supported for multi-fragment") !=
              std::string::npos)
      << "Expected warning about GreedyScheduler override\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify scheduler types via dynamic_pointer_cast on fragments
  auto& fragment_graph = app->fragment_graph();
  auto fragments = fragment_graph.get_nodes();

  for (const auto& fragment : fragments) {
    if (fragment->name() == "fragment1") {
      // GreedyScheduler should have been overridden to EventBasedScheduler
      EXPECT_TRUE(is_scheduler_type<EventBasedScheduler>(fragment))
          << "Expected fragment1 to use EventBasedScheduler (Greedy should be overridden)";
    } else if (fragment->name() == "fragment2") {
      EXPECT_TRUE(is_scheduler_type<MultiThreadScheduler>(fragment))
          << "Expected fragment2 to use MultiThreadScheduler, but it doesn't";
      // Also verify the custom scheduler name appears in logs
      EXPECT_TRUE(log_output.find("custom-multithread") != std::string::npos)
          << "Expected 'custom-multithread' scheduler name in logs\n"
          << "=== LOG ===\n"
          << log_output << "\n===========\n";
    }
  }
}

TEST_F(DistributedAppSchedulerTest, TestRxFragmentWithMultiThreadSchedulerIsRespected) {
  auto app = make_application<AppWithMixedSchedulers>();

  testing::internal::CaptureStderr();

  try {
    app->run();
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Exception: {}", e.what());
  }

  std::string log_output = testing::internal::GetCapturedStderr();

  // Verify the app ran successfully (messages were received)
  EXPECT_TRUE(log_output.find("Rx fragment2.rx message received count:") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify scheduler types directly
  auto& fragment_graph = app->fragment_graph();
  auto fragments = fragment_graph.get_nodes();

  bool found_fragment2 = false;
  for (const auto& fragment : fragments) {
    if (fragment->name() == "fragment2") {
      found_fragment2 = true;
      EXPECT_TRUE(is_scheduler_type<MultiThreadScheduler>(fragment))
          << "Expected fragment2 to use MultiThreadScheduler";
    }
  }
  EXPECT_TRUE(found_fragment2) << "fragment2 not found in fragment graph";
}

///////////////////////////////////////////////////////////////////////////////
// Tests for Application::scheduler propagation to fragments
///////////////////////////////////////////////////////////////////////////////

TEST_F(DistributedAppSchedulerTest, TestApplicationSchedulerPropagatedToFragments) {
  auto app = make_application<AppWithApplicationScheduler>();

  testing::internal::CaptureStderr();

  try {
    app->run();
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Exception: {}", e.what());
  }

  std::string log_output = testing::internal::GetCapturedStderr();

  // Verify the app ran successfully
  EXPECT_TRUE(log_output.find("Rx fragment2.rx message received count:") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify debug message about using scheduler type from Application appears
  // (This requires DEBUG log level which is set in the fixture)
  EXPECT_TRUE(log_output.find("Using scheduler type from Application") != std::string::npos)
      << "Expected debug message about using Application scheduler type not found\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify all fragments use MultiThreadScheduler (from Application)
  auto& fragment_graph = app->fragment_graph();
  auto fragments = fragment_graph.get_nodes();

  for (const auto& fragment : fragments) {
    EXPECT_TRUE(is_scheduler_type<MultiThreadScheduler>(fragment))
        << "Expected " << fragment->name() << " to use MultiThreadScheduler from Application";
  }

  // Verify MultiThreadScheduler is being used (new instance created per fragment)
  EXPECT_TRUE(log_output.find("multithread-scheduler") != std::string::npos)
      << "Expected 'multithread-scheduler' scheduler in logs\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify that arguments from app scheduler are propagated to fragments
  // AppWithApplicationScheduler sets worker_thread_number=3, so we should see this arg copied
  EXPECT_TRUE(
      log_output.find("Copying argument 'worker_thread_number' from Application scheduler") !=
      std::string::npos)
      << "Expected 'worker_thread_number' argument to be copied from app scheduler\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

TEST_F(DistributedAppSchedulerTest, TestApplicationSchedulerClockClonedPerFragment) {
  // This test verifies that when an Application uses GreedyScheduler with a custom clock:
  // 1. GreedyScheduler is overridden to EventBasedScheduler (with warning)
  // 2. The clock resource is still cloned per fragment
  auto app = make_application<AppWithApplicationSchedulerClock>();

  testing::internal::CaptureStderr();

  try {
    app->run();
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Exception: {}", e.what());
  }

  std::string log_output = testing::internal::GetCapturedStderr();

  // Verify the app ran successfully
  EXPECT_TRUE(log_output.find("Rx fragment2.rx message received count:") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify warning about GreedyScheduler override was logged
  EXPECT_TRUE(log_output.find("GreedyScheduler is not supported for multi-fragment") !=
              std::string::npos)
      << "Expected warning about GreedyScheduler override\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify per-fragment clock cloning happened for app scheduler clock resource
  EXPECT_TRUE(log_output.find("Cloned clock resource 'app-synthetic-clock'") != std::string::npos)
      << "Expected app scheduler clock to be cloned per fragment\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify all fragments use EventBasedScheduler (Greedy was overridden)
  auto& fragment_graph = app->fragment_graph();
  auto fragments = fragment_graph.get_nodes();
  for (const auto& fragment : fragments) {
    EXPECT_TRUE(is_scheduler_type<EventBasedScheduler>(fragment))
        << "Expected " << fragment->name()
        << " to use EventBasedScheduler (Greedy should be overridden)";
  }
}

TEST_F(DistributedAppSchedulerTest, TestFragmentSchedulerOverridesApplicationScheduler) {
  auto app = make_application<AppWithMixedSchedulerSources>();

  testing::internal::CaptureStderr();

  try {
    app->run();
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Exception: {}", e.what());
  }

  std::string log_output = testing::internal::GetCapturedStderr();

  // Verify the app ran successfully
  EXPECT_TRUE(log_output.find("Rx fragment2.rx message received count:") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // fragment1 should use Application scheduler type (debug message should appear for fragment1)
  EXPECT_TRUE(log_output.find("fragment1") != std::string::npos &&
              log_output.find("Using scheduler type from Application") != std::string::npos)
      << "Expected fragment1 to use Application scheduler type\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify scheduler types directly via dynamic_pointer_cast
  auto& fragment_graph = app->fragment_graph();
  auto fragments = fragment_graph.get_nodes();

  for (const auto& fragment : fragments) {
    if (fragment->name() == "fragment1") {
      // fragment1 should inherit MultiThreadScheduler from Application
      EXPECT_TRUE(is_scheduler_type<MultiThreadScheduler>(fragment))
          << "Expected fragment1 to use MultiThreadScheduler from Application";
    } else if (fragment->name() == "fragment2") {
      // fragment2 sets its own EventBasedScheduler
      EXPECT_TRUE(is_scheduler_type<EventBasedScheduler>(fragment))
          << "Expected fragment2 to use its own EventBasedScheduler";
      // Verify custom scheduler name appears
      EXPECT_TRUE(log_output.find("custom-event-based") != std::string::npos)
          << "Expected 'custom-event-based' scheduler name in logs\n"
          << "=== LOG ===\n"
          << log_output << "\n===========\n";
    }
  }
}

TEST_F(DistributedAppSchedulerTest, TestApplicationWithGreedySchedulerOverridden) {
  // Test that Application-level GreedyScheduler is overridden to EventBasedScheduler
  // for multi-fragment apps (Greedy causes deadlock in distributed mode)
  auto app = make_application<AppWithGreedySchedulerAtAppLevel>();

  testing::internal::CaptureStderr();

  try {
    app->run();
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Exception: {}", e.what());
  }

  std::string log_output = testing::internal::GetCapturedStderr();

  // Verify the app ran successfully
  EXPECT_TRUE(log_output.find("Rx fragment2.rx message received count:") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify warning about GreedyScheduler override was logged for each fragment
  size_t warning_count = 0;
  size_t pos = 0;
  std::string warning_str = "GreedyScheduler is not supported for multi-fragment";
  while ((pos = log_output.find(warning_str, pos)) != std::string::npos) {
    ++warning_count;
    pos += warning_str.length();
  }
  EXPECT_GE(warning_count, 2)
      << "Expected warning about GreedyScheduler override for each fragment\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify all fragments use EventBasedScheduler (Greedy was overridden)
  auto& fragment_graph = app->fragment_graph();
  auto fragments = fragment_graph.get_nodes();

  for (const auto& fragment : fragments) {
    EXPECT_TRUE(is_scheduler_type<EventBasedScheduler>(fragment))
        << "Expected " << fragment->name()
        << " to use EventBasedScheduler (Greedy should be overridden)";
  }
}

///////////////////////////////////////////////////////////////////////////////
// Test with driver/worker mode
///////////////////////////////////////////////////////////////////////////////

TEST_F(DistributedAppSchedulerTest, TestApplicationSchedulerPropagatedInWorkerMode) {
  std::vector<std::string> args{"app", "--driver", "--worker", "--fragments=all"};
  auto app = make_application<AppWithApplicationScheduler>(args);

  testing::internal::CaptureStderr();

  try {
    app->run();
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Exception: {}", e.what());
  }

  std::string log_output = testing::internal::GetCapturedStderr();

  // Verify the app ran successfully
  EXPECT_TRUE(log_output.find("Rx fragment2.rx message received count:") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify debug message about using scheduler type from Application appears
  EXPECT_TRUE(log_output.find("Using scheduler type from Application") != std::string::npos)
      << "Expected debug message about using Application scheduler type in worker mode\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify all fragments use MultiThreadScheduler (from Application)
  auto& fragment_graph = app->fragment_graph();
  auto fragments = fragment_graph.get_nodes();

  for (const auto& fragment : fragments) {
    EXPECT_TRUE(is_scheduler_type<MultiThreadScheduler>(fragment))
        << "Expected " << fragment->name() << " to use MultiThreadScheduler from Application";
  }
}

TEST_F(DistributedAppSchedulerTest, TestMixedSchedulersInWorkerMode) {
  std::vector<std::string> args{"app", "--driver", "--worker", "--fragments=all"};
  auto app = make_application<AppWithMixedSchedulers>(args);

  testing::internal::CaptureStderr();

  try {
    app->run();
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Exception: {}", e.what());
  }

  std::string log_output = testing::internal::GetCapturedStderr();

  // Verify the app ran successfully (messages were received)
  EXPECT_TRUE(log_output.find("Rx fragment2.rx message received count:") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify scheduler types via dynamic_pointer_cast on fragments
  // Note: GreedyScheduler is overridden to EventBasedScheduler in multi-fragment apps
  auto& fragment_graph = app->fragment_graph();
  auto fragments = fragment_graph.get_nodes();

  for (const auto& fragment : fragments) {
    if (fragment->name() == "fragment1") {
      // GreedyScheduler is not supported in multi-fragment apps; expect EventBasedScheduler
      EXPECT_TRUE(is_scheduler_type<EventBasedScheduler>(fragment))
          << "Expected fragment1 to use EventBasedScheduler (Greedy overridden) in worker mode";
    } else if (fragment->name() == "fragment2") {
      EXPECT_TRUE(is_scheduler_type<MultiThreadScheduler>(fragment))
          << "Expected fragment2 to use MultiThreadScheduler in worker mode";
    }
  }

  // Verify warning about GreedyScheduler override appears
  EXPECT_TRUE(log_output.find("GreedyScheduler is not supported for multi-fragment") !=
              std::string::npos)
      << "Expected GreedyScheduler override warning in logs\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify custom multithread scheduler name appears in logs
  EXPECT_TRUE(log_output.find("custom-multithread") != std::string::npos)
      << "Expected 'custom-multithread' scheduler name in logs\n"
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

///////////////////////////////////////////////////////////////////////////////
// Test that environment variable overrides user-set scheduler arguments
///////////////////////////////////////////////////////////////////////////////

TEST_F(DistributedAppSchedulerTest, TestEnvVarOverridesUserSchedulerArgs) {
  // Check if HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT is set in the environment
  // (CTest sets this to 3000 or 6000 depending on architecture)
  const char* env_timeout = std::getenv("HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT");
  if (env_timeout == nullptr) {
    GTEST_SKIP() << "HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT not set, skipping env override test";
  }

  int64_t expected_timeout = std::stol(env_timeout);
  constexpr int64_t user_set_timeout = 6000L;  // AppWithApplicationScheduler sets this value

  // AppWithApplicationScheduler sets stop_on_deadlock_timeout=6000, but if
  // HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT is set, that value should be used instead
  auto app = make_application<AppWithApplicationScheduler>();

  testing::internal::CaptureStderr();

  try {
    app->run();
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Exception: {}", e.what());
  }

  std::string log_output = testing::internal::GetCapturedStderr();

  // Verify the app ran successfully
  EXPECT_TRUE(log_output.find("Rx fragment2.rx message received count:") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";

  // Verify that the environment variable value is used, not the user-set value
  // by directly querying the scheduler's stop_on_deadlock_timeout parameter
  auto& fragment_graph = app->fragment_graph();
  auto fragments = fragment_graph.get_nodes();

  for (const auto& fragment : fragments) {
    int64_t actual_timeout = get_scheduler_deadlock_timeout(fragment);
    EXPECT_EQ(actual_timeout, expected_timeout)
        << "Fragment '" << fragment->name() << "': Expected env var "
        << "HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT=" << expected_timeout
        << " to override user-set value of " << user_set_timeout << ", but got " << actual_timeout;
    EXPECT_NE(actual_timeout, user_set_timeout)
        << "Fragment '" << fragment->name()
        << "': User-set stop_on_deadlock_timeout=" << user_set_timeout
        << " should have been overridden by env var";
  }
}

}  // namespace holoscan
