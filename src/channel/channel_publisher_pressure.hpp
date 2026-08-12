/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <holoscan/channel/types.hpp>

namespace holoscan::channel_internal {

enum class PublisherPressureMode : std::uint8_t {
  kFollowService,
  kRejectWhenFull,
  kBoundedRetry,
};

struct PublisherPressureOptions {
  PublisherPressureMode mode{PublisherPressureMode::kFollowService};
  std::uint64_t bounded_retry_budget_ns{};

  [[nodiscard]] static constexpr PublisherPressureOptions follow_service() noexcept { return {}; }

  [[nodiscard]] static constexpr PublisherPressureOptions reject_when_full() noexcept {
    return {.mode = PublisherPressureMode::kRejectWhenFull};
  }

  [[nodiscard]] static constexpr PublisherPressureOptions bounded_retry(
      const std::uint64_t budget_ns) noexcept {
    return {
        .mode = PublisherPressureMode::kBoundedRetry,
        .bounded_retry_budget_ns = budget_ns,
    };
  }

  [[nodiscard]] constexpr PublisherPressureOptions for_queue(
      const ChannelQueuePolicy queue_policy) const noexcept {
    return queue_policy == ChannelQueuePolicy::kBlocking ? *this : follow_service();
  }

  [[nodiscard]] constexpr bool enables_graph_retry_boundary(
      const std::uint32_t publisher_history_depth) const noexcept {
    return publisher_history_depth == 0U && selects_graph_retry_boundary();
  }

 private:
  [[nodiscard]] constexpr bool selects_graph_retry_boundary() const noexcept {
    return mode == PublisherPressureMode::kRejectWhenFull ||
           mode == PublisherPressureMode::kBoundedRetry;
  }
};

}  // namespace holoscan::channel_internal
