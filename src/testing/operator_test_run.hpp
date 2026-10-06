/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "controlled_clock_run.hpp"

namespace holoscan::testing::detail {

struct TestResultState {
  ExecutionPlan plan;
  RunTermination termination{RunTermination::kFailed};
  std::vector<Diagnostic> diagnostics;
  std::map<std::string, std::shared_ptr<CaptureStorage>, std::less<>> captures;
  std::shared_ptr<ControlledClockAttempt> clock;
  bool clock_requested{};
  bool retention_required{};
  bool observed_completion{};
  bool watchdog_expired{};
  bool diagnostics_copy_failed{};
  bool expect_stop{};
};

struct FrameReleaseState {
  std::mutex mutex;
  NotificationSender sender;
  bool armed{};
  // The original RunningFixture closes its driver even if startup never reached Arm.
  bool driver_closed{};
  // Stop closes admission while startup may still Arm the original, live fixture.
  bool stop_requested{};

  [[nodiscard]] expected<void, Error> post(
      std::optional<std::chrono::steady_clock::time_point> deadline = {}) noexcept {
    const std::lock_guard lock(mutex);
    if (driver_closed || stop_requested)
      return make_unexpected(Error{.code = ErrorCode::kClosed});
    // Rearm can hold this mutex across expiration; do not admit a late notification.
    if (deadline && std::chrono::steady_clock::now() >= *deadline)
      return make_unexpected(Error{.code = ErrorCode::kTimeout});
    if (!armed)
      return make_unexpected(Error{.code = ErrorCode::kNotReady});
    return sender.post();
  }
};

struct CaptureProgress {
  std::mutex mutex;
  std::condition_variable changed;
  std::size_t count{};
};

struct AttemptGate {
  std::mutex mutex;
  bool active{};
};

class ActiveAttempt {
 public:
  explicit ActiveAttempt(std::shared_ptr<AttemptGate> gate) : gate_(std::move(gate)) {
    const std::lock_guard lock(gate_->mutex);
    if (gate_->active)
      throw std::logic_error{"operator test already has an active attempt"};
    gate_->active = true;
  }
  ~ActiveAttempt() {
    const std::lock_guard lock(gate_->mutex);
    gate_->active = false;
  }
  ActiveAttempt(const ActiveAttempt&) = delete;
  ActiveAttempt& operator=(const ActiveAttempt&) = delete;

 private:
  std::shared_ptr<AttemptGate> gate_;
};

using CaptureHandoffs =
    std::vector<std::pair<std::shared_ptr<CaptureStorage>, std::shared_ptr<CaptureStorage>>>;

// Stable heap storage: the watchdog never borrows the movable public wrapper or its builder.
struct RunningFixtureState {
  explicit RunningFixtureState(std::shared_ptr<AttemptGate> gate);
  ~RunningFixtureState() noexcept;
  void start_watchdog(std::chrono::milliseconds timeout);
  void finish();
  void complete_cold() noexcept;
  void close_frame_driver() noexcept;

  std::unique_ptr<ActiveAttempt> lease;
  std::shared_ptr<TestResultState> result = std::make_shared<TestResultState>();
  CaptureHandoffs captures;
  std::map<std::string, std::shared_ptr<CaptureProgress>, std::less<>> progress;
  OperatorRef tested_operator;
  std::shared_ptr<FrameReleaseState> frames;
  std::optional<RunSession> session;
  std::mutex watchdog_mutex;
  std::condition_variable_any watchdog_changed;
  std::atomic<bool> watchdog_expired{};
  std::jthread watchdog;
  bool joined{};
};

}  // namespace holoscan::testing::detail
