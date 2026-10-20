/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <mutex>

#include "busy_wait_pause.hpp"

namespace holoscan::detail {

/// How long a waiter spins before it waits, and how long each timed wait lasts.
struct SettledLockPolicy {
  std::chrono::nanoseconds spin{std::chrono::microseconds{2}};
  std::chrono::nanoseconds slice{std::chrono::microseconds{50}};
};

/// Acquire `mutex`, or stop without it as soon as `settled()` reports that the work the lock would
/// protect is already done.
///
/// The caller never waits without a timeout: after a bounded pause spin it waits in timed slices
/// and re-checks `settled()` between them, so a real-time thread always has a timer armed and a
/// holder that finishes the work releases the waiter without handing over the lock.
///
/// @return An owned lock, or an unowned one when `settled()` became true first.
template <typename Settled>
[[nodiscard]] std::unique_lock<std::timed_mutex> lock_unless_settled(
    // Recheck the same predicate as an lvalue throughout the bounded wait.
    // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    std::timed_mutex& mutex, Settled&& settled, const SettledLockPolicy policy = {}) noexcept {
  std::unique_lock<std::timed_mutex> lock{mutex, std::try_to_lock};
  if (lock.owns_lock()) {
    return lock;
  }
  const auto spin_until = std::chrono::steady_clock::now() + policy.spin;
  do {
    busy_wait_pause();
    if (settled()) {
      return lock;
    }
    if (lock.try_lock()) {
      return lock;
    }
  } while (std::chrono::steady_clock::now() < spin_until);
  while (!lock.try_lock_for(policy.slice)) {
    if (settled()) {
      return lock;
    }
  }
  return lock;
}

}  // namespace holoscan::detail
