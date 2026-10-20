/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <utility>

#include <holoscan/core/run_session.hpp>

namespace holoscan::python::detail {

/// Process-lifetime cleanup owner for sessions finalized from a forbidden wait context.
/// A hosted-language session can enter callbacks or release language-owned state on this worker;
/// adapters must therefore make teardown safe when their interpreter is unavailable or finalizing.
/// Exit-time draining is bounded and may deliberately abandon a non-cooperating session.
/// Representative: RunSessionReaperTest.StopsAndJoinsAnAbandonedSession.
class RunSessionReaper final {
 public:
  using RetirementNotifier = void (*)() noexcept;

  struct Retirement {
    virtual ~Retirement() noexcept = default;
    virtual void request_stop() noexcept = 0;
    virtual void join() = 0;
  };

  struct Node final : Retirement {
    explicit Node(RunSession&& value) noexcept : session(std::move(value)) {}

    // Language-owned state the session can still call into, such as Python constructor
    // callbacks. Declared first so it is destroyed after the session; it is responsible for its
    // own interpreter safety.
    std::shared_ptr<void> language_state;
    void request_stop() noexcept override { session.request_stop(); }
    void join() override { session.wait(); }
    RunSession session;
  };

  [[nodiscard]] static RunSessionReaper& instance();

  /// Queue one preallocated ownership node. The deque may allocate; failure is logged and the
  /// process-lifetime node is intentionally retained rather than invoking fatal session cleanup.
  void enqueue(std::unique_ptr<Retirement> node) noexcept;

  [[nodiscard]] bool wait_until_idle_for(std::chrono::nanoseconds timeout);

  /// Wait for an already-created process-local reaper and its admitted retirement notifications.
  [[nodiscard]] static bool wait_for_existing_for(std::chrono::nanoseconds timeout) noexcept;

  /// Destroy joined sessions on a Python-owned thread that already holds the GIL.
  static void destroy_retired() noexcept;

  /// Install or close the language-adapter handoff invoked after a session is physically joined.
  static void set_retirement_notifier(RetirementNotifier notifier) noexcept;

 private:
  RunSessionReaper();

  void worker_loop() noexcept;

  [[nodiscard]] bool wait_until_drained_for(std::chrono::nanoseconds timeout);

  std::mutex mutex_;
  std::condition_variable ready_;
  std::condition_variable idle_;
  std::deque<std::unique_ptr<Retirement>> queue_;
  std::deque<std::unique_ptr<Retirement>> retired_;
  std::size_t pending_{};
  std::size_t retirement_notifications_in_flight_{};
};

}  // namespace holoscan::python::detail
