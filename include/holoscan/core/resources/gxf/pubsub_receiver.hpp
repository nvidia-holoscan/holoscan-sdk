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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_PUBSUB_RECEIVER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_PUBSUB_RECEIVER_HPP

#include <optional>
#include <string>
#include <vector>

#include <gxf/pubsub/pubsub_receiver.hpp>
#include <gxf/pubsub/qos_profile.hpp>

#include "./receiver.hpp"

namespace holoscan {

/**
 * @brief Pub/Sub receiver class for topic-based subscriptions.
 *
 * PubSubReceiver is used to receive messages from publishers on a specific topic.
 * It wraps the GXF `nvidia::gxf::PubSubReceiver` component, which is backend-agnostic
 * and works with different transport implementations (gRPC+UCX, DDS, etc.).
 *
 * ## Features
 *
 * - **Topic-based subscription**: Receives messages from all publishers on the topic
 * - **Dynamic discovery**: Automatically discovers matching publishers via PubSubContext
 * - **QoS support**: Configurable reliability, durability, and history policies
 * - **Double-buffered queue**: Messages arrive in backstage, moved to main stage on sync
 *
 * ## Usage
 *
 * Application authors typically don't use this class directly. It is configured
 * automatically when using pub/sub connectors with `ConnectorType::kPubSub`.
 *
 * ==Parameters==
 *
 * - **topic_name** (std::string): The topic to subscribe to.
 * - **capacity** (uint64_t, optional): The capacity of the double-buffer queue (default: 1).
 * - **policy** (uint64_t, optional): Queue policy when full - 0: pop, 1: reject, 2: fault
 *   (default: 2).
 *
 * @see nvidia::gxf::PubSubReceiver for the underlying GXF component
 * @see PubSubContext for the network context managing discovery and transport
 */
class PubSubReceiver : public Receiver {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(PubSubReceiver, Receiver)
  PubSubReceiver() = default;
  PubSubReceiver(const std::string& name, nvidia::gxf::PubSubReceiver* component);

  [[nodiscard]] const char* gxf_typename() const override { return "nvidia::gxf::PubSubReceiver"; }

  void setup(ComponentSpec& spec) override;
  void initialize() override;

  /**
   * @brief Get the underlying GXF PubSubReceiver component.
   * @return Pointer to the GXF PubSubReceiver, or nullptr if not initialized.
   */
  [[nodiscard]] nvidia::gxf::PubSubReceiver* get() const;

  /**
   * @brief Get the topic name this receiver subscribes to.
   * @return The topic name string.
   */
  [[nodiscard]] std::string topic_name() const;

  /**
   * @brief Get the number of matched publishers.
   * @return Number of publishers currently matched to this subscriber.
   */
  [[nodiscard]] size_t matched_publisher_count() const;

  /**
   * @brief Check if there are any matched publishers.
   * @return true if at least one publisher is matched.
   */
  [[nodiscard]] bool has_matched_publishers() const;

  /**
   * @brief Get the GIDs of all currently matched publishers.
   * @return Snapshot vector of publisher GIDs (taken under lock).
   */
  [[nodiscard]] std::vector<nvidia::gxf::PublisherGid> matched_publisher_gids() const;

  /**
   * @brief Set the QoS profile for this receiver.
   *
   * Must be called before initialize(). The profile is forwarded to the underlying
   * GXF PubSubReceiver during initialization. If called after initialize(),
   * the update is ignored and a warning is logged.
   * If not set, QoSProfile::Default() is used (best-effort, volatile, keep-last 10).
   *
   * @param qos The QoS profile to apply.
   */
  void qos(const nvidia::gxf::QoSProfile& qos);

  /**
   * @brief Get the QoS profile set on this receiver (if any).
   * @return The QoS profile, or std::nullopt if none was explicitly set.
   */
  [[nodiscard]] const std::optional<nvidia::gxf::QoSProfile>& qos() const { return qos_; }

 private:
  Parameter<std::string> topic_name_;
  Parameter<uint64_t> capacity_;
  Parameter<uint64_t> policy_;
  std::optional<nvidia::gxf::QoSProfile> qos_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_PUBSUB_RECEIVER_HPP */
