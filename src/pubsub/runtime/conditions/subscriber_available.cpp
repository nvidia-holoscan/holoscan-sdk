/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/pubsub/runtime/conditions/subscriber_available.hpp>

#include <fmt/format.h>
#include <algorithm>
#include <limits>

#include <stdexcept>
#include <string>

#include <holoscan/core/component_spec.hpp>

namespace holoscan {

void SubscriberAvailableCondition::setup(ComponentSpec& spec) {
  spec.param(transmitter_,
             "transmitter",
             "Transmitter",
             "The pub/sub transmitter checked for matched subscriber readiness.");
  spec.param(min_subscriber_count_,
             "min_subscriber_count",
             "Minimum subscriber count",
             "The condition is READY when matched_subscriber_count() is at least this value.",
             static_cast<uint64_t>(1));
  spec.param(require_pubsub_connector_,
             "require_pubsub_connector",
             "Require pub/sub connector",
             "If true, the configured transmitter must be a PubSubTransmitter.",
             true);
  spec.param(poll_period_ms_,
             "poll_period_ms",
             "Poll period [ms]",
             "Polling interval used while waiting for matched subscribers.",
             int64_t(100));
  spec.param(stabilization_ms_,
             "stabilization_ms",
             "Stabilization period [ms]",
             "Optional post-match stabilization delay before transitioning to READY.",
             int64_t(0));
  spec.param(latch_ready_,
             "latch_ready",
             "Latch ready",
             "If true, remain READY after the first successful subscriber match.",
             false);
  spec.param(ready_on_shutdown_,
             "ready_on_shutdown",
             "Ready on shutdown",
             "Reserved policy flag for coordinated shutdown behavior.",
             false);
}

void SubscriberAvailableCondition::initialize() {
  Condition::initialize();
  resolve_pubsub_transmitter();
}

void SubscriberAvailableCondition::resolve_pubsub_transmitter() {
  auto tx = transmitter_.get();
  if (!tx) {
    throw std::runtime_error(
        fmt::format("Condition '{}' requires a valid transmitter parameter", name()));
  }

  pubsub_transmitter_ = std::dynamic_pointer_cast<PubSubTransmitter>(tx);
  if (!pubsub_transmitter_) {
    if (require_pubsub_connector_.get()) {
      throw std::runtime_error(
          fmt::format("Condition '{}' expected a PubSubTransmitter but got a non-pubsub "
                      "transmitter for parameter 'transmitter'",
                      name()));
    }

    HOLOSCAN_LOG_DEBUG(
        "Condition '{}': transmitter is not pub/sub and "
        "require_pubsub_connector=false, treating condition as READY",
        name());
  }
}

void SubscriberAvailableCondition::update_state(int64_t timestamp) {
  if (latch_ready_.get() && latched_ready_.load()) {
    current_state_.store(SchedulingStatusType::kReady);
    target_timestamp_.store(timestamp);
    return;
  }

  // TODO(grelee): ready_on_shutdown_ is reserved for future coordinated-shutdown support
  // when the runtime exposes a native-condition shutdown signal.

  bool is_matched = true;
  if (pubsub_transmitter_) {
    const uint64_t min_count = min_subscriber_count_.get();
    if (min_count == 1) {
      is_matched = pubsub_transmitter_->has_matched_subscribers();
    } else {
      is_matched = pubsub_transmitter_->matched_subscriber_count() >= min_count;
    }
  }

  // TODO(grelee): Replace poll loop with kWaitEvent + notify_scheduler() driven by
  // PubSubTransmitter match-event listener (on_subscription_matched from the DDS backend).
  // Keep poll_period_ms only as a fallback for backends that cannot deliver the event.
  constexpr int64_t kMaxMs = std::numeric_limits<int64_t>::max() / 1000000LL;
  const int64_t poll_period_ns =
      std::min(std::max<int64_t>(1, poll_period_ms_.get()), kMaxMs) * 1000000LL;
  const int64_t stabilization_ns =
      std::min(std::max<int64_t>(0, stabilization_ms_.get()), kMaxMs) * 1000000LL;

  SchedulingStatusType next_state = SchedulingStatusType::kReady;
  int64_t next_target_timestamp = timestamp;

  if (!is_matched) {
    matched_since_timestamp_ = -1;
    next_state = SchedulingStatusType::kWaitTime;
    next_target_timestamp = (timestamp > std::numeric_limits<int64_t>::max() - poll_period_ns)
                                ? std::numeric_limits<int64_t>::max()
                                : timestamp + poll_period_ns;
  } else if (stabilization_ns > 0) {
    if (matched_since_timestamp_ < 0) {
      matched_since_timestamp_ = timestamp;
      HOLOSCAN_LOG_TRACE(
          "SubscriberAvailableCondition '{}': subscriber match detected at timestamp {}, "
          "stabilizing for {} ms",
          name(),
          timestamp,
          stabilization_ms_.get());
    }

    const int64_t stable_until =
        (matched_since_timestamp_ > std::numeric_limits<int64_t>::max() - stabilization_ns)
            ? std::numeric_limits<int64_t>::max()
            : matched_since_timestamp_ + stabilization_ns;
    if (timestamp < stable_until) {
      next_state = SchedulingStatusType::kWaitTime;
      next_target_timestamp = stable_until;
    } else {
      next_state = SchedulingStatusType::kReady;
      next_target_timestamp = timestamp;
      if (current_state_.load() != SchedulingStatusType::kReady) {
        HOLOSCAN_LOG_TRACE(
            "SubscriberAvailableCondition '{}': stabilization complete at timestamp {}",
            name(),
            timestamp);
        if (latch_ready_.get()) {
          latched_ready_.store(true);
        }
      }
    }
  } else {
    matched_since_timestamp_ = timestamp;
    next_state = SchedulingStatusType::kReady;
    next_target_timestamp = timestamp;
    if (current_state_.load() != SchedulingStatusType::kReady) {
      HOLOSCAN_LOG_TRACE(
          "SubscriberAvailableCondition '{}': subscriber match detected at timestamp {}",
          name(),
          timestamp);
      if (latch_ready_.get()) {
        latched_ready_.store(true);
      }
    }
  }

  if (current_state_.load() != next_state) {
    current_state_.store(next_state);
  }
  target_timestamp_.store(next_target_timestamp);
}

void SubscriberAvailableCondition::check([[maybe_unused]] int64_t timestamp,
                                         SchedulingStatusType* status_type,
                                         int64_t* target_timestamp) const {
  if (status_type == nullptr) {
    throw std::runtime_error(
        fmt::format("Condition '{}' received nullptr for status_type", name()));
  }
  if (target_timestamp == nullptr) {
    throw std::runtime_error(
        fmt::format("Condition '{}' received nullptr for target_timestamp", name()));
  }
  *status_type = current_state_.load();
  *target_timestamp = target_timestamp_.load();
}

void SubscriberAvailableCondition::on_execute(int64_t timestamp) {
  update_state(timestamp);
}

}  // namespace holoscan
