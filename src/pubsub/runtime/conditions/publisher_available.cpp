/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <holoscan/pubsub/runtime/conditions/publisher_available.hpp>

#include <fmt/format.h>
#include <algorithm>
#include <limits>

#include <stdexcept>
#include <string>

#include <holoscan/core/component_spec.hpp>

namespace holoscan {

void PublisherAvailableCondition::setup(ComponentSpec& spec) {
  spec.param(receiver_,
             "receiver",
             "Receiver",
             "The pub/sub receiver checked for matched publisher readiness.");
  spec.param(min_publisher_count_,
             "min_publisher_count",
             "Minimum publisher count",
             "The condition is READY when matched_publisher_count() is at least this value.",
             static_cast<uint64_t>(1));
  spec.param(require_pubsub_connector_,
             "require_pubsub_connector",
             "Require pub/sub connector",
             "If true, the configured receiver must be a PubSubReceiver.",
             true);
  spec.param(poll_period_ms_,
             "poll_period_ms",
             "Poll period [ms]",
             "Polling interval used while waiting for matched publishers.",
             int64_t(100));
  spec.param(latch_ready_,
             "latch_ready",
             "Latch ready",
             "If true, remain READY after the first successful publisher match.",
             false);
}

void PublisherAvailableCondition::initialize() {
  Condition::initialize();
  resolve_pubsub_receiver();
}

void PublisherAvailableCondition::resolve_pubsub_receiver() {
  auto rx = receiver_.get();
  if (!rx) {
    throw std::runtime_error(
        fmt::format("Condition '{}' requires a valid receiver parameter", name()));
  }

  pubsub_receiver_ = std::dynamic_pointer_cast<PubSubReceiver>(rx);
  if (!pubsub_receiver_) {
    if (require_pubsub_connector_.get()) {
      throw std::runtime_error(
          fmt::format("Condition '{}' expected a PubSubReceiver but got a non-pubsub "
                      "receiver for parameter 'receiver'",
                      name()));
    }

    HOLOSCAN_LOG_DEBUG(
        "Condition '{}': receiver is not pub/sub and "
        "require_pubsub_connector=false, treating condition as READY",
        name());
  }
}

void PublisherAvailableCondition::update_state(int64_t timestamp) {
  if (latch_ready_.get() && latched_ready_.load()) {
    current_state_.store(SchedulingStatusType::kReady);
    target_timestamp_.store(timestamp);
    return;
  }

  bool is_ready = true;
  if (pubsub_receiver_) {
    const uint64_t min_count = min_publisher_count_.get();
    if (min_count == 1) {
      is_ready = pubsub_receiver_->has_matched_publishers();
    } else {
      is_ready = pubsub_receiver_->matched_publisher_count() >= min_count;
    }
  }

  // TODO(grelee): Replace poll loop with kWaitEvent + notify_scheduler() driven by
  // PubSubReceiver match-event listener (on_publication_matched from the DDS backend).
  // Keep poll_period_ms only as a fallback for backends that cannot deliver the event.
  const auto next_state = is_ready ? SchedulingStatusType::kReady : SchedulingStatusType::kWaitTime;
  if (current_state_.load() != SchedulingStatusType::kReady &&
      next_state == SchedulingStatusType::kReady) {
    HOLOSCAN_LOG_TRACE("PublisherAvailableCondition '{}': publisher match detected at timestamp {}",
                       name(),
                       timestamp);
    if (latch_ready_.get()) {
      latched_ready_.store(true);
    }
  }
  if (current_state_.load() != next_state) {
    current_state_.store(next_state);
  }
  if (current_state_.load() == SchedulingStatusType::kReady) {
    target_timestamp_.store(timestamp);
  } else {
    constexpr int64_t kMaxMs = std::numeric_limits<int64_t>::max() / 1000000LL;
    const int64_t poll_period_ns =
        std::min(std::max<int64_t>(1, poll_period_ms_.get()), kMaxMs) * 1000000LL;
    target_timestamp_.store((timestamp > std::numeric_limits<int64_t>::max() - poll_period_ns)
                                ? std::numeric_limits<int64_t>::max()
                                : timestamp + poll_period_ns);
  }
}

void PublisherAvailableCondition::check([[maybe_unused]] int64_t timestamp,
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

void PublisherAvailableCondition::on_execute(int64_t timestamp) {
  update_state(timestamp);
}

}  // namespace holoscan
