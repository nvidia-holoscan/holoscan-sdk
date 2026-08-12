/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <semaphore>
#include <vector>

#include <holoscan/core/graph.hpp>

#include "../time/clock_internal.hpp"

namespace holoscan::detail {

// Durable wake tokens close the readiness-scan-to-controller-wait race without making the
// notification producer acquire the controller's wait mutex.
class RuntimeWakeState final {
 public:
  void signal() noexcept {
    pending_.fetch_add(1U, std::memory_order_relaxed);
    semaphore_.release();
  }

  /// Publish at most one coalesced wake hint until the next controller consumption.
  ///
  /// Coalescing is appropriate when exact witness cardinality lives in another bounded source and
  /// the wake merely asks a partition owner to rescan that source. The durable pending count stays
  /// nonzero while the producer publishes the corresponding semaphore permit, so concurrent
  /// producers cannot open a second coalesced publication window. Counted signal() calls may still
  /// add independent durable tokens.
  [[nodiscard]] bool signal_coalesced() noexcept {
    std::size_t expected = 0U;
    if (!pending_.compare_exchange_strong(
            expected, 1U, std::memory_order_acq_rel, std::memory_order_relaxed)) {
      return false;
    }
    semaphore_.release();
    return true;
  }

  [[nodiscard]] bool wait_for(std::chrono::nanoseconds timeout) noexcept {
    if (!semaphore_.try_acquire_for(timeout)) {
      return false;
    }
    static_cast<void>(pending_.fetch_sub(1U, std::memory_order_relaxed));
    return true;
  }

  [[nodiscard]] std::size_t drain() noexcept {
    const std::size_t available = pending_.load(std::memory_order_acquire);
    if (available == 0U) {
      return 0U;
    }
    std::size_t consumed = 0U;
    while (consumed < available && semaphore_.try_acquire()) {
      // Decrement only after consuming the matching permit: a producer may
      // have incremented pending_ but not released its permit yet, and must
      // remain visible to a later drain instead of being double-counted.
      static_cast<void>(pending_.fetch_sub(1U, std::memory_order_relaxed));
      ++consumed;
    }
    return consumed;
  }

 private:
  // Accepted physical-source capacity bounds the number of signals between controller drains.
  // A token remains pending until a consumer has acquired its matching semaphore permit.
  std::atomic<std::size_t> pending_{0U};
  std::counting_semaphore<> semaphore_{0};
};

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
