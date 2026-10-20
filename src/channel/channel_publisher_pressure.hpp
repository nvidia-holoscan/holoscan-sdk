/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cstdint>

#include <holoscan/channel/types.hpp>

namespace holoscan::channel_internal {

enum class PublisherPressureMode : std::uint8_t {
  kFollowService,
  kRejectWhenFull,
  kBoundedRetry,
};

/// Receives one call per sample that HoloMQ replaced in a drop-oldest subscriber queue during a
/// send by this publisher. HoloMQ reports replacements synchronously inside the send, on the
/// sending thread. Runtime uses it to count process-local deliveries that no drain will receive.
struct ReplacementObserver {
  void* state{};
  void (*replaced)(void*) noexcept {};

  void operator()() const noexcept {
    if (replaced != nullptr) {
      replaced(state);
    }
  }
};

struct PublisherPressureOptions {
  PublisherPressureMode mode{PublisherPressureMode::kFollowService};
  std::uint64_t bounded_retry_budget_ns{};
  /// Optional flag that ends a bounded retry early; see
  /// `holomq::PublisherPubSubOptions::bounded_retry_cancellation`.
  const std::atomic<bool>* bounded_retry_cancellation{};
  /// Optional replacement counter; only drop-oldest publishers report replacements.
  ReplacementObserver replacement_observer{};

  [[nodiscard]] static constexpr PublisherPressureOptions follow_service() noexcept { return {}; }

  [[nodiscard]] static constexpr PublisherPressureOptions reject_when_full() noexcept {
    return {.mode = PublisherPressureMode::kRejectWhenFull};
  }

  [[nodiscard]] static constexpr PublisherPressureOptions bounded_retry(
      const std::uint64_t budget_ns,
      const std::atomic<bool>* const cancellation = nullptr) noexcept {
    return {
        .mode = PublisherPressureMode::kBoundedRetry,
        .bounded_retry_budget_ns = budget_ns,
        .bounded_retry_cancellation = cancellation,
    };
  }

  [[nodiscard]] constexpr PublisherPressureOptions for_queue(
      const ChannelQueuePolicy queue_policy) const noexcept {
    return queue_policy == ChannelQueuePolicy::kBlocking
               ? *this
               : PublisherPressureOptions{.replacement_observer = replacement_observer};
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
