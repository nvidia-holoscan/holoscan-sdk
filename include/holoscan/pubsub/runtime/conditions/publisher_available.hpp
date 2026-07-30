/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_PUBSUB_RUNTIME_CONDITIONS_PUBLISHER_AVAILABLE_HPP
#define HOLOSCAN_PUBSUB_RUNTIME_CONDITIONS_PUBLISHER_AVAILABLE_HPP

#include <atomic>
#include <cstdint>
#include <memory>

#include <holoscan/core/condition.hpp>
#include <holoscan/core/resources/gxf/pubsub_receiver.hpp>
#include <holoscan/core/resources/gxf/receiver.hpp>

namespace holoscan {

/**
 * @brief Native condition that gates execution on pub/sub publisher match readiness.
 *
 * This condition is ready when the configured receiver has at least
 * `min_publisher_count` matched publishers.
 *
 * ==Parameters==
 *
 * - **receiver** (`std::shared_ptr<holoscan::Receiver>`, required): The receiver to monitor.
 *   In most applications this is provided as an input port name
 *   (for example `Arg("receiver", "in")`) and resolved by the framework.
 * - **min_publisher_count** (`uint64_t`, optional, default `1`): Minimum number of matched
 *   publishers required for READY state.
 * - **require_pubsub_connector** (`bool`, optional, default `true`): If true, the receiver
 *   must resolve to a `PubSubReceiver`, otherwise initialization throws.
 * - **poll_period_ms** (`int64_t`, optional, default `100`): Polling interval used while
 *   waiting for matched publishers. The condition returns `WAIT_TIME` until the next poll.
 * - **latch_ready** (`bool`, optional, default `false`): If true, once the condition becomes
 *   ready it remains ready for the lifetime of the condition rather than returning to wait on
 *   later disconnects.
 */
class PublisherAvailableCondition : public Condition {
 public:
  HOLOSCAN_CONDITION_FORWARD_ARGS(PublisherAvailableCondition)
  PublisherAvailableCondition() = default;

  void setup(ComponentSpec& spec) override;
  void initialize() override;

  void update_state(int64_t timestamp) override;
  void check(int64_t timestamp, SchedulingStatusType* status_type,
             int64_t* target_timestamp) const override;
  void on_execute(int64_t timestamp) override;

 private:
  void resolve_pubsub_receiver();

  Parameter<std::shared_ptr<Receiver>> receiver_;
  Parameter<uint64_t> min_publisher_count_;
  Parameter<bool> require_pubsub_connector_;
  Parameter<int64_t> poll_period_ms_;
  Parameter<bool> latch_ready_;

  std::shared_ptr<PubSubReceiver> pubsub_receiver_;
  std::atomic<SchedulingStatusType> current_state_{SchedulingStatusType::kWaitTime};
  std::atomic<int64_t> target_timestamp_{0};
  std::atomic<bool> latched_ready_{false};
};

}  // namespace holoscan

#endif /* HOLOSCAN_PUBSUB_RUNTIME_CONDITIONS_PUBLISHER_AVAILABLE_HPP */
