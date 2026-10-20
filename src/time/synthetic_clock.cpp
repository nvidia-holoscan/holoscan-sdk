/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/time/synthetic_clock.hpp>

#include <chrono>
#include <condition_variable>
#include <mutex>

#include "clock_internal.hpp"

namespace holoscan {

SyntheticClock::SyntheticClock(int64_t initial_timestamp_ns)
    : current_time_ns_(initial_timestamp_ns) {}

SyntheticClock::~SyntheticClock() {
  // Stop the clock so no thread is left blocked on a clock that will never advance again.
  stop();
}

double SyntheticClock::time() const {
  return timestamp_to_time(timestamp());
}

int64_t SyntheticClock::timestamp() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return current_time_ns_;
}

expected<void, Error> SyntheticClock::sleep_until(int64_t target_time_ns) {
  std::unique_lock<std::mutex> lock(mutex_);
  ++active_sleepers_;
  condition_variable_.wait(
      lock, [this, target_time_ns] { return current_time_ns_ >= target_time_ns || stopped_; });
  const bool reached = current_time_ns_ >= target_time_ns;
  if (--active_sleepers_ == 0) {
    // Let a concurrent stop() / destructor observe that all sleepers have left their wait.
    condition_variable_.notify_all();
  }
  if (!reached) {
    return make_unexpected(Error{ErrorCode::kClosed, {}});
  }
  return {};
}

expected<void, Error> SyntheticClock::sleep_for(int64_t duration_ns) {
  std::unique_lock<std::mutex> lock(mutex_);
  const int64_t target_time_ns = detail::saturating_add(current_time_ns_, duration_ns);
  ++active_sleepers_;
  condition_variable_.wait(
      lock, [this, target_time_ns] { return current_time_ns_ >= target_time_ns || stopped_; });
  const bool reached = current_time_ns_ >= target_time_ns;
  if (--active_sleepers_ == 0) {
    condition_variable_.notify_all();
  }
  if (!reached) {
    return make_unexpected(Error{ErrorCode::kClosed, {}});
  }
  return {};
}

// advance_to() / advance_by() wake every blocked sleeper with a single notify_all() so each can
// re-check its own target. With many concurrent sleepers and a high advance rate this is O(waiters)
// wakeups per advance; that is fine for coarse replay/simulation stepping. If fine-grained,
// high-fanout stepping ever becomes a hot path, switch to a target-ordered wakeup (e.g. wake only
// sleepers whose target has been reached). Notifying outside the lock is intentional.
void SyntheticClock::advance_to(int64_t target_time_ns) {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    current_time_ns_ = target_time_ns;
  }
  condition_variable_.notify_all();
}

void SyntheticClock::advance_by(int64_t delta_ns) {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    current_time_ns_ = detail::saturating_add(current_time_ns_, delta_ns);
  }
  condition_variable_.notify_all();
}

void SyntheticClock::advance_by(std::chrono::nanoseconds delta) {
  advance_by(delta.count());
}

void SyntheticClock::stop() {
  std::unique_lock<std::mutex> lock(mutex_);
  stopped_ = true;
  condition_variable_.notify_all();
  // Block until every woken sleeper has left its wait(), so the mutex and condition variable are no
  // longer in use by any other thread and can be safely destroyed.
  condition_variable_.wait(lock, [this] { return active_sleepers_ == 0; });
}

}  // namespace holoscan
