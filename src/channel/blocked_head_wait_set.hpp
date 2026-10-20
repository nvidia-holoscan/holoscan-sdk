/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/channel/endpoint_interfaces.hpp>

#include <atomic>
#include <cstdint>
#include <memory>

namespace holoscan::channel_internal {

/// Source-private capability of the Runtime pump's wait set for a subscriber whose staged sample
/// cannot be stored.
///
/// Every queued sample behind such a sample makes the wait report its subscriber ready, although
/// none can be received until the consumer frees storage. A readiness mask hides those samples,
/// and a wake ends the pump's wait once storage is freed.
class BlockedHeadWaitSet {
 public:
  virtual ~BlockedHeadWaitSet() = default;

  /// @brief Install, or with \c nullptr remove, the readiness mask of one attachment.
  /// @param handle Wait-set-local identity returned by attach.
  /// @param mask Flag that its owner may set from any thread to hide the attachment's queued
  /// samples from waits. Its notifications still end a blocked wait, as a wake with no handle.
  /// @return Empty success, or \ref ErrorCode::kInvalidArgument for an unknown attachment.
  [[nodiscard]] virtual expected<void, Error> set_readiness_mask(
      std::uint64_t handle, std::shared_ptr<const std::atomic<bool>> mask) noexcept = 0;

  /// @brief Attach the reusable wake that \ref wake posts. At most one may be attached.
  /// @return Empty success, or a checked setup error.
  [[nodiscard]] virtual expected<void, Error> attach_wake() = 0;

  /// @brief End the current or next wait as a wake with no ready handle. Posts coalesce.
  /// @return Empty success, or the failed post. Safe from any thread after \ref attach_wake.
  [[nodiscard]] virtual expected<void, Error> wake() const noexcept = 0;
};

/// Return the blocked-head capability of a wait set, or null when it has none.
[[nodiscard]] inline BlockedHeadWaitSet* blocked_head_wait_set(
    InlinePayloadWaitSet& wait_set) noexcept {
  return dynamic_cast<BlockedHeadWaitSet*>(&wait_set);
}

}  // namespace holoscan::channel_internal
