/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#if defined(__linux__)
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <ctime>
#else
#include <semaphore>
#endif

#include <holoscan/core/graph_ids.hpp>

#include "../time/clock_internal.hpp"
#include "busy_wait_pause.hpp"

namespace holoscan::detail {

// Durable wake tokens close the readiness-scan-to-controller-wait race without making the
// notification producer acquire the controller's wait mutex.
//
// On Linux the state is one private futex word: bit 31 marks a parked owner and the low bits count
// pending tokens. A producer adds a token with one compare-and-swap and enters the kernel only when
// its update cleared the parked bit. Before parking, the owner checks for a token for at most
// kSpinBeforePark, pausing the CPU between checks but never yielding it. In a saturated pipeline
// the next token often arrives within that window, and taking it there saves both the park and
// the producer's kernel wake. The spin never extends past the deadline and never repeats after a
// timeout. The owner then waits with FUTEX_WAIT_BITSET on an absolute CLOCK_MONOTONIC deadline,
// so no retry or spurious return moves the deadline. At most one thread parks at a time (the
// partition owner or the controller); other threads may drain or try a zero-timeout wait
// concurrently.
// Isolate the hot futex word from diagnostic and failure-accounting writes.
// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
class RuntimeWakeState final {
 public:
  /// Index of the partition owner this state wakes, for diagnostic accounting only.
  static constexpr std::uint32_t kNoAccountingPartition = std::numeric_limits<std::uint32_t>::max();
  std::uint32_t accounting_partition{kNoAccountingPartition};

  /// Add one durable token.
  void signal() noexcept { static_cast<void>(signal_observed()); }

  /// Add one durable token and return whether a parked waiter had to be woken in the kernel.
  [[nodiscard]] bool signal_observed() noexcept;

  /// Publish at most one coalesced wake hint until the next controller consumption.
  ///
  /// Coalescing is appropriate when exact witness cardinality lives in another bounded source and
  /// the wake merely asks a partition owner to rescan that source. A hint is published only when no
  /// token is pending, so concurrent producers cannot open a second coalesced publication window.
  /// Counted signal() calls may still add independent durable tokens.
  [[nodiscard]] bool signal_coalesced() noexcept { return signal_coalesced(nullptr); }

  /// As signal_coalesced(), also reporting whether publishing had to wake a parked waiter.
  [[nodiscard]] bool signal_coalesced(bool* kernel_wake) noexcept;

  /// Consume one token, parking until `deadline` (steady_clock, CLOCK_MONOTONIC on Linux) when none
  /// is pending. Return false on the deadline, or on an unexpected wait failure (see failed()).
  [[nodiscard]] bool wait_until(std::chrono::steady_clock::time_point deadline) noexcept;

  /// Consume one token, waiting at most `timeout`. A zero timeout only tries and never parks.
  [[nodiscard]] bool wait_for(std::chrono::nanoseconds timeout) noexcept {
    if (timeout <= std::chrono::nanoseconds::zero()) {
      return try_take();
    }
    return wait_until(std::chrono::steady_clock::now() + timeout);
  }

  /// Consume every pending token without entering the kernel; return how many.
  [[nodiscard]] std::size_t drain() noexcept;

  /// Return whether a token is pending, without consuming it.
  [[nodiscard]] bool pending() const noexcept;

  /// Return whether a wait failed for a reason other than its deadline.
  [[nodiscard]] bool failed() const noexcept { return failed_.load(std::memory_order_relaxed); }

  /// Return whether an owner is parked in wait_until(), so that the next signal must wake it. The
  /// portable fallback cannot observe its waiter and always returns false.
  [[nodiscard]] bool parked() const noexcept;

 private:
  [[nodiscard]] bool try_take() noexcept;

  std::atomic<bool> failed_{false};
#if defined(__linux__)
  static constexpr std::uint32_t kParked = 1U << 31U;
  static constexpr std::uint32_t kTokens = kParked - 1U;
  // Long enough to catch a token from a producer one pipeline stage ahead, short enough that a
  // periodic owner waiting for its next tick spends little CPU before parking.
  static constexpr std::chrono::nanoseconds kSpinBeforePark{2'000};

  // Take one token from `observed` (reloaded on contention); false when none is pending.
  [[nodiscard]] bool take_token(std::uint32_t& observed) noexcept {
    while ((observed & kTokens) != 0U) {
      // A pending token implies the parked bit is clear, so one decrement takes it.
      if (word_.compare_exchange_weak(
              observed, observed - 1U, std::memory_order_acquire, std::memory_order_acquire)) {
        return true;
      }
    }
    return false;
  }

  static void wake_parked(std::atomic<std::uint32_t>& word) noexcept;

  alignas(64) std::atomic<std::uint32_t> word_{0U};
  static_assert(sizeof(std::atomic<std::uint32_t>) == sizeof(std::uint32_t) &&
                std::atomic<std::uint32_t>::is_always_lock_free);
#else
  // Portable fallback: a token remains pending until a consumer has acquired its matching permit.
  std::atomic<std::size_t> pending_{0U};
  std::counting_semaphore<> semaphore_{0};
#endif
};

#if defined(__linux__)
inline void RuntimeWakeState::wake_parked(std::atomic<std::uint32_t>& word) noexcept {
  // Wake every parked waiter; production parks at most one, and waking more is harmless.
  static_cast<void>(syscall(SYS_futex,
                            reinterpret_cast<std::uint32_t*>(&word),
                            FUTEX_WAKE | FUTEX_PRIVATE_FLAG,
                            std::numeric_limits<int>::max(),
                            nullptr,
                            nullptr,
                            0));
}

inline bool RuntimeWakeState::signal_observed() noexcept {
  std::uint32_t observed = word_.load(std::memory_order_relaxed);
  std::uint32_t next{};
  do {
    next = (observed & kTokens) + 1U;  // add a token and clear the parked bit
  } while (!word_.compare_exchange_weak(
      observed, next, std::memory_order_release, std::memory_order_relaxed));
  if ((observed & kParked) == 0U) {
    return false;
  }
  wake_parked(word_);
  return true;
}

inline bool RuntimeWakeState::signal_coalesced(bool* kernel_wake) noexcept {
  std::uint32_t observed = word_.load(std::memory_order_relaxed);
  do {
    if ((observed & kTokens) != 0U) {
      return false;
    }
  } while (!word_.compare_exchange_weak(
      observed, 1U, std::memory_order_release, std::memory_order_relaxed));
  const bool parked = (observed & kParked) != 0U;
  if (parked) {
    wake_parked(word_);
  }
  if (kernel_wake != nullptr) {
    *kernel_wake = parked;
  }
  return true;
}

inline bool RuntimeWakeState::try_take() noexcept {
  std::uint32_t observed = word_.load(std::memory_order_acquire);
  return take_token(observed);
}

inline bool RuntimeWakeState::wait_until(
    const std::chrono::steady_clock::time_point deadline) noexcept {
  // libstdc++ steady_clock is CLOCK_MONOTONIC, the clock FUTEX_WAIT_BITSET uses by default.
  const auto since_epoch = deadline.time_since_epoch();
  const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(since_epoch);
  const timespec absolute{
      .tv_sec = static_cast<time_t>(seconds.count()),
      .tv_nsec = static_cast<long>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(since_epoch - seconds).count())};
  std::uint32_t observed = word_.load(std::memory_order_acquire);
  const auto spin_end = std::min(std::chrono::steady_clock::now() + kSpinBeforePark, deadline);
  while (std::chrono::steady_clock::now() < spin_end) {
    if (take_token(observed)) {
      return true;
    }
    busy_wait_pause();
    observed = word_.load(std::memory_order_acquire);
  }
  for (;;) {
    if (take_token(observed)) {
      return true;
    }
    // Parking only changes the wait marker; take_token acquires any producer publication.
    if (observed == 0U &&
        !word_.compare_exchange_weak(
            observed, kParked, std::memory_order_relaxed, std::memory_order_relaxed)) {
      continue;  // a token arrived; observed was reloaded
    }
    const long result = syscall(SYS_futex,
                                reinterpret_cast<std::uint32_t*>(&word_),
                                FUTEX_WAIT_BITSET | FUTEX_PRIVATE_FLAG,
                                kParked,
                                &absolute,
                                nullptr,
                                FUTEX_BITSET_MATCH_ANY);
    const int error = result == 0 ? 0 : errno;
    if (error == ETIMEDOUT) {
      // Clear the parked bit; a token that raced the timeout is taken rather than left behind.
      observed = word_.fetch_and(~kParked, std::memory_order_acquire) & ~kParked;
      return take_token(observed);
    }
    if (error != 0 && error != EAGAIN && error != EINTR) {
      word_.fetch_and(~kParked, std::memory_order_relaxed);
      failed_.store(true, std::memory_order_relaxed);
      return false;
    }
    observed = word_.load(std::memory_order_acquire);  // woken, changed, or interrupted
  }
}

inline std::size_t RuntimeWakeState::drain() noexcept {
  return word_.fetch_and(kParked, std::memory_order_acquire) & kTokens;
}

inline bool RuntimeWakeState::pending() const noexcept {
  return (word_.load(std::memory_order_acquire) & kTokens) != 0U;
}

inline bool RuntimeWakeState::parked() const noexcept {
  return (word_.load(std::memory_order_acquire) & kParked) != 0U;
}
#else
inline bool RuntimeWakeState::signal_observed() noexcept {
  pending_.fetch_add(1U, std::memory_order_relaxed);
  semaphore_.release();
  return false;
}

inline bool RuntimeWakeState::signal_coalesced(bool* kernel_wake) noexcept {
  std::size_t expected = 0U;
  if (!pending_.compare_exchange_strong(
          expected, 1U, std::memory_order_acq_rel, std::memory_order_relaxed)) {
    return false;
  }
  semaphore_.release();
  if (kernel_wake != nullptr) {
    *kernel_wake = false;
  }
  return true;
}

inline bool RuntimeWakeState::try_take() noexcept {
  if (!semaphore_.try_acquire()) {
    return false;
  }
  static_cast<void>(pending_.fetch_sub(1U, std::memory_order_relaxed));
  return true;
}

inline bool RuntimeWakeState::wait_until(
    const std::chrono::steady_clock::time_point deadline) noexcept {
  if (!semaphore_.try_acquire_until(deadline)) {
    return false;
  }
  static_cast<void>(pending_.fetch_sub(1U, std::memory_order_relaxed));
  return true;
}

inline std::size_t RuntimeWakeState::drain() noexcept {
  const std::size_t available = pending_.load(std::memory_order_acquire);
  std::size_t consumed = 0U;
  while (consumed < available && semaphore_.try_acquire()) {
    // Decrement only after consuming the matching permit: a producer may have incremented
    // pending_ but not released its permit yet, and must remain visible to a later drain.
    static_cast<void>(pending_.fetch_sub(1U, std::memory_order_relaxed));
    ++consumed;
  }
  return consumed;
}

inline bool RuntimeWakeState::pending() const noexcept {
  return pending_.load(std::memory_order_acquire) != 0U;
}

inline bool RuntimeWakeState::parked() const noexcept {
  return false;
}
#endif

/// Map each compiler-frozen Operator owner to its exact partition wake state.
///
/// The router is materialized on the cold RunSession path. Direct-retention callbacks then perform
/// one indexed lookup instead of broadcasting progress to every partition owner.
class RuntimePartitionWakeRouter final {
 public:
  void clear() noexcept { wakes_.clear(); }

  void reset(std::size_t size) { wakes_.assign(size, nullptr); }

  [[nodiscard]] bool empty() const noexcept { return wakes_.empty(); }

  [[nodiscard]] bool bind(VertexId owner, RuntimeWakeState* wake) noexcept {
    if (!owner || wake == nullptr || owner.value >= wakes_.size() ||
        wakes_[owner.value] != nullptr) {
      return false;
    }
    wakes_[owner.value] = wake;
    return true;
  }

  [[nodiscard]] RuntimeWakeState* find(VertexId owner) const noexcept {
    return owner && owner.value < wakes_.size() ? wakes_[owner.value] : nullptr;
  }

 private:
  std::vector<RuntimeWakeState*> wakes_;
};

/// Bound one positive source-wake delta without overflowing signed clock coordinates.
constexpr std::chrono::nanoseconds bounded_source_wait_duration(
    std::int64_t now_ns, std::int64_t wake_ns, std::chrono::nanoseconds maximum) noexcept {
  if (maximum.count() <= 0 || wake_ns <= now_ns) {
    return std::chrono::nanoseconds::zero();
  }
  const std::int64_t limit = saturating_add(now_ns, maximum.count());
  if (limit == std::numeric_limits<std::int64_t>::max() && now_ns > limit - maximum.count()) {
    // A saturated upper bound proves both coordinates are close enough for
    // the exact positive difference to remain representable and below the cap.
    return std::chrono::nanoseconds{wake_ns - now_ns};
  }
  if (wake_ns >= limit) {
    return maximum;
  }
  // The comparison above proves this positive difference is smaller than the
  // representable maximum, including when the capped addition saturated.
  return std::chrono::nanoseconds{wake_ns - now_ns};
}

}  // namespace holoscan::detail
