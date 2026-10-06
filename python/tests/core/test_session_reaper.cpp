/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "../../holoscan/core/session_reaper.hpp"
#include "pipeline_test_operators.hpp"

namespace holoscan::python::test {
namespace {

std::atomic<std::size_t> retirement_notifications{};
std::atomic<bool> hold_retirement_notification{};
std::atomic<bool> release_retirement_notification{};
std::mutex notification_mutex;
std::condition_variable notification_ready;

void record_retirement_notification() noexcept {
  if (hold_retirement_notification.load(std::memory_order_acquire)) {
    release_retirement_notification.wait(false, std::memory_order_acquire);
  }
  {
    const std::lock_guard lock(notification_mutex);
    retirement_notifications.fetch_add(1U, std::memory_order_relaxed);
  }
  notification_ready.notify_all();
}

class RetirementTestStateGuard {
 public:
  RetirementTestStateGuard() = default;
  RetirementTestStateGuard(const RetirementTestStateGuard&) = delete;
  RetirementTestStateGuard& operator=(const RetirementTestStateGuard&) = delete;

  ~RetirementTestStateGuard() noexcept {
    release_retirement_notification.store(true, std::memory_order_release);
    release_retirement_notification.notify_all();
    holoscan::python::detail::RunSessionReaper::destroy_retired();
    holoscan::python::detail::RunSessionReaper::set_retirement_notifier(nullptr);
    hold_retirement_notification.store(false, std::memory_order_release);
  }
};

// REQ: PY-API-005
TEST(RunSessionReaperTest, StopsAndJoinsAnAbandonedSession) {
  using namespace std::chrono_literals;
  using holoscan::python::detail::RunSessionReaper;

  PipelineObservation::reset();
  retirement_notifications.store(0U, std::memory_order_relaxed);
  hold_retirement_notification.store(true, std::memory_order_release);
  release_retirement_notification.store(false, std::memory_order_release);
  RunSessionReaper::set_retirement_notifier(&record_retirement_notification);
  const RetirementTestStateGuard retirement_state_guard;

  // Key setup: Run the canonical pipeline and transfer its live session to the reaper.
  //
  //   source --uint32--> doubler --uint32--> sink
  holoscan::Graph graph{"session-reaper"};
  auto source = graph.op<CounterSource>("source");
  source.set_params(CounterSourceParams{.limit = 1'000U});
  const auto doubler = graph.op<DoubleIt>("doubler");
  const auto sink = graph.op<Sink>("sink");
  graph.add_flow(source->out, doubler->in, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(doubler->out, sink->in, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  ASSERT_TRUE(plan.ok());
  holoscan::RunSession session = holoscan::run_async(plan);
  ASSERT_TRUE(PipelineObservation::wait_for(2U, 5s));

  auto& reaper = RunSessionReaper::instance();
  reaper.enqueue(std::make_unique<RunSessionReaper::Node>(std::move(session)));

  // Key assertion: The ordinary successful queue path stops and physically joins the abandoned
  // session within the bounded wait. Resource-exhaustion behavior is a qualified fallback, not an
  // allocation-free callback-path promise.
  EXPECT_TRUE(reaper.wait_until_idle_for(5s));
  EXPECT_EQ(retirement_notifications.load(std::memory_order_relaxed), 0U);
  RunSessionReaper::set_retirement_notifier(nullptr);
  std::atomic<bool> release_triggered{};
  const std::jthread release_after_close{[&] {
    std::this_thread::sleep_for(50ms);
    release_triggered.store(true, std::memory_order_release);
    release_retirement_notification.store(true, std::memory_order_release);
    release_retirement_notification.notify_all();
  }};

  // Key assertion: Closing notifier admission still waits for the callback that was admitted
  // before the close. Interpreter shutdown therefore cannot overtake a GIL acquisition already
  // in flight on the worker.
  EXPECT_TRUE(RunSessionReaper::wait_for_existing_for(5s));
  EXPECT_TRUE(release_triggered.load(std::memory_order_acquire));
  {
    std::unique_lock lock(notification_mutex);
    EXPECT_TRUE(notification_ready.wait_for(
        lock, 5s, [] { return retirement_notifications.load(std::memory_order_relaxed) == 1U; }));
  }
  EXPECT_EQ(retirement_notifications.load(std::memory_order_relaxed), 1U);
}

}  // namespace
}  // namespace holoscan::python::test
