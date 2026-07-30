/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_PUBSUB_FASTDDS_PUBSUB_FASTDDS_DISCOVERY_HPP
#define HOLOSCAN_PUBSUB_FASTDDS_PUBSUB_FASTDDS_DISCOVERY_HPP

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantListener.hpp>
#include <fastdds/rtps/builtin/data/ParticipantBuiltinTopicData.hpp>
#include <fastdds/rtps/builtin/data/PublicationBuiltinTopicData.hpp>
#include <fastdds/rtps/builtin/data/SubscriptionBuiltinTopicData.hpp>
#include <fastdds/rtps/participant/ParticipantDiscoveryInfo.hpp>
#include <fastdds/rtps/reader/ReaderDiscoveryStatus.hpp>
#include <fastdds/rtps/writer/WriterDiscoveryStatus.hpp>

#include <gxf/pubsub/endpoint_info.hpp>
#include <gxf/pubsub/pubsub_discovery.hpp>

namespace holoscan {

// Forward declarations
class FastDdsPubSubContext;

/**
 * @brief DDS-based implementation of PubSubDiscovery.
 *
 * Wraps FastDDS's built-in SPDP/SEDP discovery protocols.
 *
 * The `announce_publisher` and `announce_subscriber` methods don't actually
 * register with an external service; instead, they track local endpoints
 * while DDS handles the actual discovery via SEDP.
 *
 * Discovery callbacks are driven by DDS listener events.
 *
 * ## Discovery Model
 *
 * DDS discovery happens automatically when DataWriters/DataReaders are created:
 * - SPDP (Simple Participant Discovery Protocol): Finds other participants
 * - SEDP (Simple Endpoint Discovery Protocol): Exchanges topic/endpoint info
 *
 * This class provides the PubSubDiscovery interface on top of these mechanisms.
 *
 * ## Usage
 *
 * ```cpp
 * auto dds_ctx = make_resource<FastDdsPubSubContext>("dds_context", ...);
 * auto discovery = std::make_shared<FastDdsDiscovery>(dds_ctx.get());
 *
 * discovery->set_on_publisher_discovered([](const PublisherInfo& info) {
 *     // Handle newly discovered publisher
 * });
 *
 * discovery->initialize();
 *
 * // Announce local endpoints (tracked, DDS handles actual discovery)
 * discovery->announce_publisher(pub_info);
 * ```
 */
class FastDdsDiscovery : public nvidia::gxf::PubSubDiscovery {
 public:
  /**
   * @brief Construct a FastDdsDiscovery with a FastDdsPubSubContext.
   *
   * @param context FastDdsPubSubContext providing DomainParticipant
   */
  explicit FastDdsDiscovery(FastDdsPubSubContext* context);

  /**
   * @brief Destructor - calls shutdown() if needed.
   */
  ~FastDdsDiscovery() override;

  // Delete copy operations
  FastDdsDiscovery(const FastDdsDiscovery&) = delete;
  FastDdsDiscovery& operator=(const FastDdsDiscovery&) = delete;

  //----------------------------------------------------------------------------
  // PubSubDiscovery Lifecycle
  //----------------------------------------------------------------------------

  nvidia::gxf::Expected<void> initialize() override;
  nvidia::gxf::Expected<void> shutdown() override;
  bool is_initialized() const override;

  //----------------------------------------------------------------------------
  // Discovery Model
  //----------------------------------------------------------------------------

  /// DDS uses decentralized passive discovery (SPDP/SEDP).
  /// announce_publisher()/announce_subscriber() are local bookkeeping only;
  /// actual network discovery happens when DataWriters/DataReaders are created.
  nvidia::gxf::DiscoveryModel discovery_model() const override {
    return nvidia::gxf::DiscoveryModel::kDecentralizedPassive;
  }

  //----------------------------------------------------------------------------
  // PubSubDiscovery Registration
  //----------------------------------------------------------------------------

  /**
   * @brief Announce a local publisher (no-op for DDS).
   *
   * Local endpoint tracking is handled by the shared TopicRegistry
   * (set by PubSubContext via set_topic_registry()). The actual DDS
   * discovery happens when FastDdsTransport::create_publisher_endpoint()
   * creates the DataWriter.
   */
  nvidia::gxf::Expected<void> announce_publisher(const nvidia::gxf::PublisherInfo& info) override;

  /**
   * @brief Announce a local subscriber (no-op for DDS).
   *
   * Local endpoint tracking is handled by the shared TopicRegistry
   * (set by PubSubContext via set_topic_registry()). The actual DDS
   * discovery happens when FastDdsTransport::create_subscriber_endpoint()
   * creates the DataReader.
   */
  nvidia::gxf::Expected<void> announce_subscriber(const nvidia::gxf::SubscriberInfo& info) override;

  //----------------------------------------------------------------------------
  // PubSubDiscovery Deregistration
  //----------------------------------------------------------------------------

  nvidia::gxf::Expected<void> remove_publisher(const nvidia::gxf::PublisherGid& gid) override;

  nvidia::gxf::Expected<void> remove_subscriber(const nvidia::gxf::SubscriberGid& gid) override;

  //----------------------------------------------------------------------------
  // PubSubDiscovery Query
  //----------------------------------------------------------------------------

  /**
   * @brief Query discovered publishers on a topic.
   *
   * @param topic_name Topic to query (empty string returns all publishers)
   * @return List of publisher info, or error
   */
  nvidia::gxf::Expected<std::vector<nvidia::gxf::PublisherInfo>> query_publishers(
      const std::string& topic_name) override;

  /**
   * @brief Query discovered subscribers on a topic.
   *
   * @param topic_name Topic to query (empty string returns all subscribers)
   * @return List of subscriber info, or error
   */
  nvidia::gxf::Expected<std::vector<nvidia::gxf::SubscriberInfo>> query_subscribers(
      const std::string& topic_name) override;

  /**
   * @brief Get all known topic names.
   *
   * @return List of topic names with at least one publisher or subscriber
   */
  nvidia::gxf::Expected<std::vector<std::string>> get_all_topics() override;

  //----------------------------------------------------------------------------
  // PubSubDiscovery Callbacks
  //----------------------------------------------------------------------------

  void set_on_publisher_discovered(PublisherDiscoveredCallback callback) override;
  void set_on_subscriber_discovered(SubscriberDiscoveredCallback callback) override;
  void set_on_publisher_lost(PublisherLostCallback callback) override;
  void set_on_subscriber_lost(SubscriberLostCallback callback) override;

  //----------------------------------------------------------------------------
  // Native Buffer Capability Advertisement
  //----------------------------------------------------------------------------

  /// Set the local NativeBufferCapability to be serialized into participant UserData.
  /// Must be called before initialize() (the participant is already created by that point,
  /// so the capability is set during FastDdsPubSubContext::initialize()).
  void set_local_native_capability(const nvidia::gxf::NativeBufferCapability& cap) {
    std::lock_guard<std::mutex> lock(mutex_);
    local_native_capability_ = cap;
  }

  nvidia::gxf::NativeBufferCapability local_native_capability(int device_id = -1) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (device_id < 0 || !local_native_capability_.supports_native_buffers()) {
      return local_native_capability_;
    }
    // Build a capability for the requested device by looking up its UUID.
    return capability_for_device(device_id);
  }

  /// Serialize NativeBufferCapability into participant UserData QoS bytes.
  static std::vector<uint8_t> serialize_native_capability(
      const nvidia::gxf::NativeBufferCapability& cap);

  /// Parse NativeBufferCapability from participant UserData QoS bytes.
  static nvidia::gxf::NativeBufferCapability parse_native_capability(
      const std::vector<uint8_t>& user_data);

 private:
  /// DDS listener for discovery events
  class DiscoveryListener;

  /// Helper to convert DDS GUID to GXF Gid
  static nvidia::gxf::Gid guid_to_gid(const eprosima::fastdds::rtps::GUID_t& guid);

  FastDdsPubSubContext* context_;
  bool initialized_ = false;

  // Discovered remote endpoints (from DDS SEDP listener callbacks).
  // Local endpoints are tracked by the shared TopicRegistry (set by PubSubContext
  // via set_topic_registry()), so we don't need separate local_publishers_/
  // local_subscribers_ maps.
  std::unordered_map<nvidia::gxf::Gid, nvidia::gxf::NativeBufferCapability>
      discovered_participant_capabilities_;
  std::unordered_map<nvidia::gxf::Gid, nvidia::gxf::PublisherInfo> discovered_publishers_;
  std::unordered_map<nvidia::gxf::Gid, nvidia::gxf::SubscriberInfo> discovered_subscribers_;

  mutable std::mutex mutex_;

  // Callbacks
  PublisherDiscoveredCallback on_publisher_discovered_;
  SubscriberDiscoveredCallback on_subscriber_discovered_;
  PublisherLostCallback on_publisher_lost_;
  SubscriberLostCallback on_subscriber_lost_;

  // DDS listener
  std::unique_ptr<DiscoveryListener> listener_;

  /// Build a capability for a specific device by querying its UUID via CUDA.
  /// Returns the default capability (with empty UUID) if the query fails.
  nvidia::gxf::NativeBufferCapability capability_for_device(int device_id) const;

  // Local native buffer capability (set before initialize)
  nvidia::gxf::NativeBufferCapability local_native_capability_;
};

}  // namespace holoscan

#endif  // HOLOSCAN_PUBSUB_FASTDDS_PUBSUB_FASTDDS_DISCOVERY_HPP
