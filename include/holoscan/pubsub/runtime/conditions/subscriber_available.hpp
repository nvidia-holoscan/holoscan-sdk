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

#ifndef HOLOSCAN_PUBSUB_RUNTIME_CONDITIONS_SUBSCRIBER_AVAILABLE_HPP
#define HOLOSCAN_PUBSUB_RUNTIME_CONDITIONS_SUBSCRIBER_AVAILABLE_HPP

#include <atomic>
#include <cstdint>
#include <memory>

#include <holoscan/core/condition.hpp>
#include <holoscan/core/resources/gxf/pubsub_transmitter.hpp>
#include <holoscan/core/resources/gxf/transmitter.hpp>

namespace holoscan {

/**
 * @brief Native condition that gates execution on pub/sub subscriber match readiness.
 *
 * This condition is ready when the configured transmitter has at least
 * `min_subscriber_count` matched subscribers.
 *
 * ==Parameters==
 *
 * - **transmitter** (`std::shared_ptr<holoscan::Transmitter>`, required): The transmitter to
 *   monitor. In most applications this is provided as an output port name
 *   (for example `Arg("transmitter", "out")`) and resolved by the framework.
 * - **min_subscriber_count** (`uint64_t`, optional, default `1`): Minimum number of matched
 *   subscribers required for READY state.
 * - **require_pubsub_connector** (`bool`, optional, default `true`): If true, the transmitter
 *   must resolve to a `PubSubTransmitter`, otherwise initialization throws.
 * - **poll_period_ms** (`int64_t`, optional, default `100`): Polling interval used while
 *   waiting for matched subscribers. The condition returns `WAIT_TIME` until the next poll.
 * - **stabilization_ms** (`int64_t`, optional, default `0`): Optional post-match stabilization
 *   delay. When greater than zero, the condition waits for the match criterion to remain
 *   satisfied for this duration before transitioning to READY.
 * - **latch_ready** (`bool`, optional, default `false`): If true, once the condition becomes
 *   ready it remains ready for the lifetime of the condition rather than returning to wait on
 *   later disconnects.
 * - **ready_on_shutdown** (`bool`, optional, default `false`): Reserved policy flag for future
 *   coordinated-shutdown behavior.
 */
class SubscriberAvailableCondition : public Condition {
 public:
  HOLOSCAN_CONDITION_FORWARD_ARGS(SubscriberAvailableCondition)
  SubscriberAvailableCondition() = default;

  void setup(ComponentSpec& spec) override;
  void initialize() override;

  void update_state(int64_t timestamp) override;
  void check(int64_t timestamp, SchedulingStatusType* status_type,
             int64_t* target_timestamp) const override;
  void on_execute(int64_t timestamp) override;

 private:
  void resolve_pubsub_transmitter();

  Parameter<std::shared_ptr<Transmitter>> transmitter_;
  Parameter<uint64_t> min_subscriber_count_;
  Parameter<bool> require_pubsub_connector_;
  Parameter<int64_t> poll_period_ms_;
  Parameter<int64_t> stabilization_ms_;
  Parameter<bool> latch_ready_;
  Parameter<bool> ready_on_shutdown_;

  std::shared_ptr<PubSubTransmitter> pubsub_transmitter_;
  std::atomic<SchedulingStatusType> current_state_{SchedulingStatusType::kWaitTime};
  std::atomic<int64_t> target_timestamp_{0};
  int64_t matched_since_timestamp_ = -1;
  std::atomic<bool> latched_ready_{false};
};

}  // namespace holoscan

#endif /* HOLOSCAN_PUBSUB_RUNTIME_CONDITIONS_SUBSCRIBER_AVAILABLE_HPP */
