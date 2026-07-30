/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_PUBSUB_TRANSMITTER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_PUBSUB_TRANSMITTER_HPP

#include <optional>
#include <string>
#include <vector>

#include <gxf/pubsub/pubsub_transmitter.hpp>
#include <gxf/pubsub/qos_profile.hpp>

#include "./transmitter.hpp"

namespace holoscan {

/**
 * @brief Pub/Sub transmitter class for topic-based publishing.
 *
 * PubSubTransmitter is used to emit messages to subscribers on a specific topic.
 * It wraps the GXF `nvidia::gxf::PubSubTransmitter` component, which is backend-agnostic
 * and works with different transport implementations (gRPC+UCX, DDS, etc.).
 *
 * ## Features
 *
 * - **Topic-based publishing**: Sends messages to all subscribers on the topic
 * - **Dynamic discovery**: Automatically discovers matching subscribers via PubSubContext
 * - **QoS support**: Configurable reliability, durability, and history policies
 * - **Double-buffered queue**: Messages pushed to backstage, published on sync
 *
 * ## Usage
 *
 * Application authors typically don't use this class directly. It is configured
 * automatically when using pub/sub connectors with `ConnectorType::kPubSub`.
 *
 * ==Parameters==
 *
 * - **topic_name** (std::string): The topic to publish to.
 * - **capacity** (uint64_t, optional): The capacity of the double-buffer queue (default: 1).
 * - **policy** (uint64_t, optional): Queue policy when full - 0: pop, 1: reject, 2: fault
 *   (default: 2).
 *
 * @see nvidia::gxf::PubSubTransmitter for the underlying GXF component
 * @see PubSubContext for the network context managing discovery and transport
 */
class PubSubTransmitter : public Transmitter {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(PubSubTransmitter, Transmitter)
  PubSubTransmitter() = default;
  PubSubTransmitter(const std::string& name, nvidia::gxf::PubSubTransmitter* component);

  [[nodiscard]] const char* gxf_typename() const override {
    return "nvidia::gxf::PubSubTransmitter";
  }

  void setup(ComponentSpec& spec) override;
  void initialize() override;

  /**
   * @brief Get the underlying GXF PubSubTransmitter component.
   * @return Pointer to the GXF PubSubTransmitter, or nullptr if not initialized.
   */
  [[nodiscard]] nvidia::gxf::PubSubTransmitter* get() const;

  /**
   * @brief Get the topic name this transmitter publishes to.
   * @return The topic name string.
   */
  [[nodiscard]] std::string topic_name() const;

  /**
   * @brief Get the number of matched subscribers.
   * @return Number of subscribers currently matched to this publisher.
   */
  [[nodiscard]] size_t matched_subscriber_count() const;

  /**
   * @brief Check if there are any matched subscribers.
   * @return true if at least one subscriber is matched.
   */
  [[nodiscard]] bool has_matched_subscribers() const;

  /**
   * @brief Get the GIDs of all currently matched subscribers.
   * @return Snapshot vector of subscriber GIDs (taken under lock).
   */
  [[nodiscard]] std::vector<nvidia::gxf::SubscriberGid> matched_subscriber_gids() const;

  /**
   * @brief Set the QoS profile for this transmitter.
   *
   * Must be called before initialize(). The profile is forwarded to the underlying
   * GXF PubSubTransmitter during initialization. If called after initialize(),
   * the update is ignored and a warning is logged.
   * If not set, QoSProfile::Default() is used (best-effort, volatile, keep-last 10).
   *
   * @param qos The QoS profile to apply.
   */
  void qos(const nvidia::gxf::QoSProfile& qos);

  /**
   * @brief Get the QoS profile set on this transmitter (if any).
   * @return The QoS profile, or std::nullopt if none was explicitly set.
   */
  [[nodiscard]] const std::optional<nvidia::gxf::QoSProfile>& qos() const { return qos_; }

  Parameter<std::string> topic_name_;
  Parameter<uint64_t> capacity_;
  Parameter<uint64_t> policy_;

 private:
  std::optional<nvidia::gxf::QoSProfile> qos_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_PUBSUB_TRANSMITTER_HPP */
