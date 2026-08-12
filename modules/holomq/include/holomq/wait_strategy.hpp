/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>

namespace holomq {

/// Blocking behavior after the initial userspace spin phase (\ref WaitStrategy::spin_duration).
enum class WaitMode : std::uint8_t {
  /// Poll for \ref WaitStrategy::spin_duration, then return. \ref WaitStrategy::timeout is ignored
  /// (a warning is logged when set).
  Return,
  /// Poll for \ref WaitStrategy::spin_duration, then \c sleep(timeout) in userspace regardless of
  /// event wake.
  TimedWait,
  /// Poll for \ref WaitStrategy::spin_duration, then block on the kernel event listener. Requires
  /// event wake on the subscriber (and symmetric publisher setting). Uses \ref
  /// WaitStrategy::timeout
  /// when set; blocks indefinitely when unset.
  KernelWait,
};

/// Parameters for blocking waits on \ref Subscriber and \ref WaitSet.
struct WaitStrategy {
  /// Blocking behavior after the initial spin phase.
  WaitMode mode{WaitMode::TimedWait};
  /// Userspace poll duration before the post-spin behavior selected by \ref WaitMode.
  std::chrono::nanoseconds spin_duration{50'000};
  /// Upper bound for \ref WaitMode::TimedWait sleep, or for \ref WaitMode::KernelWait kernel wait.
  /// Empty uses a 1 ms sleep for \ref WaitMode::TimedWait and indefinite kernel wait for \ref
  /// WaitMode::KernelWait. Ignored for \ref WaitMode::Return.
  std::optional<std::chrono::nanoseconds> timeout{};
};

/// Outcome of \ref WaitSet::wait and \ref WaitSet::wait_for_ready.
enum class WaitResult : std::uint8_t {
  /// Wait completed; check \ref WaitSet::ready_handles and \ref Subscriber::try_take (may be empty
  /// after kernel
  /// spurious wake on IOX2).
  ready,
  /// Timed wait expired with no attachment ready (\ref WaitMode::Return after the spin phase,
  /// \ref WaitStrategy::timeout on \ref WaitMode::TimedWait or \ref WaitMode::KernelWait, or IOX2
  /// timed kernel/listener wait with no notification).
  timeout,
  /// Reserved coarse failure value for compatibility. Current wait APIs report failures through
  /// their \ref holomq::expected error instead of returning this value.
  error,
};

}  // namespace holomq
