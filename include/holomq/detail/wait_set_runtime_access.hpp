/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/expected.hpp>
#include <holomq/wait_set.hpp>
#include <holomq/wait_set_handle.hpp>
#include <holomq/wait_strategy.hpp>

#include <atomic>
#include <memory>
#include <utility>

namespace holomq::detail {

/// Private Runtime access to \ref WaitSet entry points that are not public API.
class WaitSetRuntimeAccess {
 public:
  /// The Runtime channel pump used this to skip the pre-block readiness scan after what it took to
  /// be an exhaustive drain. That invariant does not hold (readiness can appear between the drain
  /// and the block), so the entry point now performs the checked wait and has no callers. It is
  /// retained for the exported ABI baseline until a change can absorb the export removal.
  [[nodiscard]] static auto wait_assuming_no_ready(WaitSet& wait_set,
                                                   const WaitStrategy& strategy) noexcept
      -> expected<WaitResult, Error> {
    return wait_set.wait_impl_assuming_no_ready(strategy);
  }

  /// Install, or with \c nullptr remove, the readiness mask of one attached subscriber.
  ///
  /// While the mask is set, the subscriber's queued samples do not make a wait report ready and
  /// its handle is absent from \ref WaitSet::ready_handles. Its kernel notifications are still
  /// drained and still end a blocked wait, which then reports \ref WaitResult::ready with no handle
  /// for it. The Runtime pump sets the mask while the subscriber holds a sample it cannot store, so
  /// samples queued behind that one stop waking it. The owner may set or clear the flag from any
  /// thread; installing it is serialized with attach and wait on the waiting thread.
  /// @retval ErrorCode::kInvalidArgument \p handle is not an attached subscriber.
  [[nodiscard]] static auto set_readiness_mask(
      WaitSet& wait_set, const WaitSetHandle handle,
      std::shared_ptr<const std::atomic<bool>> mask) noexcept -> expected<void, Error> {
    return wait_set.set_readiness_mask_impl(handle, std::move(mask));
  }

  /// Attach a reusable wake that \ref signal_wake posts from any thread.
  ///
  /// A posted wake ends a blocked kernel wait, or the next one, which reports
  /// \ref WaitResult::ready with no ready handle for the wake. Unlike \ref WaitTrigger it does not
  /// latch, so later waits block again. At most one wake can be attached to a wait set.
  [[nodiscard]] static auto attach_wake(WaitSet& wait_set) -> expected<void, Error> {
    return wait_set.attach_wake_impl();
  }

  /// Post the wake attached by \ref attach_wake. Posts coalesce until a wait drains them.
  ///
  /// Safe to call from any thread, concurrently with a wait, after \ref attach_wake has returned.
  [[nodiscard]] static auto signal_wake(const WaitSet& wait_set) noexcept -> expected<void, Error> {
    return wait_set.signal_wake_impl();
  }
};

}  // namespace holomq::detail
