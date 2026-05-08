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

#ifndef HOLOSCAN_PUBSUB_FASTDDS_PUBSUB_FASTDDS_TRANSPORT_HPP
#define HOLOSCAN_PUBSUB_FASTDDS_PUBSUB_FASTDDS_TRANSPORT_HPP

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/publisher/DataWriterListener.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/DataReaderListener.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/topic/Topic.hpp>
#include <fastdds/dds/topic/TypeSupport.hpp>

#include <gxf/pubsub/pubsub_native_buffer.hpp>
#include <gxf/pubsub/pubsub_transport.hpp>

#include <holoscan/pubsub/common/sidecar_dispatch_queue.hpp>
#include <holoscan/pubsub/fastdds/pubsub/fastdds_holoscan_entity_type_support.hpp>
#include <holoscan/pubsub/fastdds/pubsub/fastdds_qos_profiles.hpp>

namespace holoscan {

// Forward declarations
class FastDdsPubSubContext;

/**
 * @brief DDS-based implementation of PubSubTransport.
 *
 * Uses FastDDS DataWriter/DataReader for message transport.
 *
 * ## Topic-Based Transport
 *
 * DDS is a **topic-based** transport (TransportModel::kTopicBased):
 * - One `DataWriter::write()` fans out to all matching subscribers
 * - `PubSubContext` calls `send(topic_name, ...)` once per publish
 * - DDS handles reliability, fragmentation, and transport internally
 *
 * ## Connection Model
 *
 * DDS handles connections via SPDP/SEDP discovery:
 * - `connect_to()` is a no-op (DDS auto-discovers endpoints)
 * - `disconnect_from()` is a no-op (DDS handles endpoint lifecycle)
 * - `is_connected_to()` always returns true for known endpoints
 *
 * ## Endpoint Lifecycle
 *
 * `PubSubContext` calls the generic lifecycle methods automatically:
 * - `create_publisher_endpoint()` → creates DDS DataWriter for a topic
 * - `create_subscriber_endpoint()` → creates DDS DataReader for a topic
 * - `remove_publisher_endpoint()` → deletes DDS DataWriter
 * - `remove_subscriber_endpoint()` → deletes DDS DataReader
 *
 * ## Usage
 *
 * ```cpp
 * auto dds_ctx = make_resource<FastDdsPubSubContext>("dds_context", ...);
 * auto transport = std::make_shared<FastDdsTransport>(dds_ctx.get());
 *
 * transport->set_on_receive([](const Gid& src, std::vector<uint8_t>&& payload,
 *                              const MessageMetadata& meta) {
 *     // Handle received message
 * });
 *
 * transport->initialize();
 *
 * // Endpoints are created automatically by PubSubContext during
 * // register_publisher() / register_subscriber().
 * ```
 */
class FastDdsTransport : public nvidia::gxf::PubSubTransport {
 public:
  /**
   * @brief Construct a FastDdsTransport with a FastDdsPubSubContext.
   *
   * @param context FastDdsPubSubContext providing DomainParticipant
   */
  explicit FastDdsTransport(FastDdsPubSubContext* context);

  /**
   * @brief Destructor - calls shutdown() if needed.
   */
  ~FastDdsTransport() override;

  // Delete copy operations
  FastDdsTransport(const FastDdsTransport&) = delete;
  FastDdsTransport& operator=(const FastDdsTransport&) = delete;

  //----------------------------------------------------------------------------
  // PubSubTransport Lifecycle
  //----------------------------------------------------------------------------

  nvidia::gxf::Expected<void> initialize() override;
  nvidia::gxf::Expected<void> shutdown() override;
  bool is_initialized() const override;

  //----------------------------------------------------------------------------
  // Transport Model
  //----------------------------------------------------------------------------

  /**
   * @brief DDS is a topic-based transport.
   *
   * PubSubContext uses this to call send(topic_name, ...) once per publish
   * instead of calling send(gid, ...) once per subscriber.
   */
  nvidia::gxf::TransportModel transport_model() const override {
    return nvidia::gxf::TransportModel::kTopicBased;
  }

  //----------------------------------------------------------------------------
  // Backend Capabilities
  //----------------------------------------------------------------------------

  /// SEDP handles topic matching + QoS compatibility at the RTPS protocol level
  bool native_topic_matching() const override { return true; }

  /// RTPS-level reliability, durability, and deadline enforcement
  bool native_qos_enforcement() const override { return true; }

  /// RTPS supports multicast for discovery and data
  bool supports_multicast() const override { return true; }

  /// DDS DataWriters/DataReaders manage connections internally
  bool requires_explicit_connections() const override { return false; }

  //----------------------------------------------------------------------------
  // Native Buffer (CUDA IPC) Capabilities
  //----------------------------------------------------------------------------

  bool supports_native_buffers() const override { return native_buffers_enabled_; }
  bool supports_mixed_local_remote_fanout() const override { return native_buffers_enabled_; }
  bool supports_native_profile(const std::string& profile) const override {
    return profile == "cuda_ipc_same_gpu_v1";
  }

  /// Send a native descriptor on the sidecar topic "{topic}/_native_desc"
  nvidia::gxf::Expected<void> send_native_descriptor(
      const std::string& topic_name, const nvidia::gxf::NativeDescriptorPayload& descriptor,
      const nvidia::gxf::MessageMetadata& metadata) override;

  /// Enable native buffer support (called by FastDdsPubSubNetworkContext)
  void set_native_buffers_enabled(bool enabled);

  //----------------------------------------------------------------------------
  // PubSubTransport Connection Management (mostly no-ops for DDS)
  //----------------------------------------------------------------------------

  /**
   * @brief No-op for DDS - discovery is automatic via SPDP/SEDP.
   */
  nvidia::gxf::Expected<void> connect_to(const nvidia::gxf::EndpointInfo& remote_endpoint) override;

  /**
   * @brief No-op for DDS - endpoint lifecycle managed by DDS.
   */
  nvidia::gxf::Expected<void> disconnect_from(const nvidia::gxf::Gid& remote_gid) override;

  /**
   * @brief Returns true for any endpoint (DDS handles discovery).
   */
  bool is_connected_to(const nvidia::gxf::Gid& remote_gid) const override;

  //----------------------------------------------------------------------------
  // Topic-Based Endpoint Lifecycle (called by PubSubContext)
  //----------------------------------------------------------------------------

  /**
   * @brief Create a DataWriter for a topic.
   *
   * Called automatically by PubSubContext::register_publisher().
   * Maps the QoSProfile to FastDDS DataWriterQos via dds_qos::apply_writer_qos().
   *
   * @param topic_name Name of the topic to publish to.
   * @param publisher_gid GID to associate with this writer.
   * @param qos QoS profile to apply to the DataWriter.
   * @return Success or error if topic/writer creation fails.
   */
  nvidia::gxf::Expected<void> create_publisher_endpoint(
      const std::string& topic_name, const nvidia::gxf::Gid& publisher_gid,
      const nvidia::gxf::QoSProfile& qos = nvidia::gxf::QoSProfile{}) override;

  /**
   * @brief Create a DataReader for a topic.
   *
   * Called automatically by PubSubContext::register_subscriber().
   * Maps the QoSProfile to FastDDS DataReaderQos via dds_qos::apply_reader_qos().
   *
   * @param topic_name Name of the topic to subscribe to.
   * @param subscriber_gid GID to associate with this reader.
   * @param qos QoS profile to apply to the DataReader.
   * @return Success or error if topic/reader creation fails.
   */
  nvidia::gxf::Expected<void> create_subscriber_endpoint(
      const std::string& topic_name, const nvidia::gxf::Gid& subscriber_gid,
      const nvidia::gxf::QoSProfile& qos = nvidia::gxf::QoSProfile{}) override;

  /**
   * @brief Remove a DataWriter.
   *
   * Called automatically by PubSubContext::unregister_publisher()
   * and PubSubContext::deinitialize(). For reliable QoS, calls
   * wait_for_acknowledgments() before deletion.
   *
   * @param publisher_gid GID of the writer to remove.
   * @return Success or error if writer not found.
   */
  nvidia::gxf::Expected<void> remove_publisher_endpoint(
      const nvidia::gxf::Gid& publisher_gid) override;

  /**
   * @brief Remove a DataReader.
   *
   * Called automatically by PubSubContext::unregister_subscriber()
   * and PubSubContext::deinitialize().
   *
   * @param subscriber_gid GID of the reader to remove.
   * @return Success or error if reader not found.
   */
  nvidia::gxf::Expected<void> remove_subscriber_endpoint(
      const nvidia::gxf::Gid& subscriber_gid) override;

  //----------------------------------------------------------------------------
  // PubSubTransport Data Plane
  //----------------------------------------------------------------------------

  // Bring base class overloads into scope (topic-based send, move-semantic variants)
  using PubSubTransport::send;

  /**
   * @brief Send payload via DDS DataWriter (GID-based fallback).
   *
   * This is the GID-based send required by the PubSubTransport interface.
   * For kTopicBased transports, PubSubContext calls the topic-based send()
   * overload instead, so this is a fallback only.
   *
   * @param destination_gid IGNORED - DDS publishes to topic, not endpoint.
   * @param payload Serialized message (already staged to host memory).
   * @param metadata Must contain publisher_gid identifying the writer.
   * @return Success or error if no writer exists for the publisher.
   */
  nvidia::gxf::Expected<void> send(const nvidia::gxf::Gid& destination_gid,
                                   const std::vector<uint8_t>& payload,
                                   const nvidia::gxf::MessageMetadata& metadata) override;

  /**
   * @brief Send payload via DDS DataWriter (topic-based, preferred).
   *
   * Looks up the DataWriter directly by topic name — no GID→topic mapping.
   * This is the primary send path for DDS since transport_model() returns kTopicBased.
   *
   * @param topic_name Topic to publish on.
   * @param payload Serialized message.
   * @param metadata Message metadata (publisher_gid for diagnostics).
   * @return Success or error if no writer exists for the topic.
   */
  nvidia::gxf::Expected<void> send(const std::string& topic_name,
                                   const std::vector<uint8_t>& payload,
                                   const nvidia::gxf::MessageMetadata& metadata) override;

  void set_on_receive(ReceiveCallback callback) override;
  void set_on_connection_established(ConnectionEstablishedCallback callback) override;
  void set_on_connection_lost(ConnectionLostCallback callback) override;

  //----------------------------------------------------------------------------
  // PubSubTransport Metrics
  //----------------------------------------------------------------------------

  size_t get_send_queue_size() const override;
  size_t get_receive_queue_size() const override;
  size_t get_connection_count() const override;

 private:
  /// DDS listener for DataReader events (receives messages)
  class ReaderListener;

  /// DDS listener for DataWriter events (optional, for matched subscriber tracking)
  class WriterListener;

  /// DDS listener for sidecar DataReader events (receives native descriptors)
  class SidecarReaderListener;

  /// Get or create a DDS Topic
  eprosima::fastdds::dds::Topic* get_or_create_topic(const std::string& topic_name);

  /// Get or create a sidecar DataWriter for native descriptor messages
  eprosima::fastdds::dds::DataWriter* get_or_create_sidecar_writer(const std::string& topic_name);

  /// Create a sidecar DataReader for native descriptor messages
  nvidia::gxf::Expected<void> create_sidecar_reader(const std::string& topic_name,
                                                    const nvidia::gxf::Gid& subscriber_gid);

  /// Shared send implementation used by both GID-based and topic-based send()
  nvidia::gxf::Expected<void> send_impl(const std::string& topic_name,
                                        const std::vector<uint8_t>& payload,
                                        const nvidia::gxf::MessageMetadata& metadata);
  void enqueue_sidecar_receive(nvidia::gxf::Gid publisher_gid, std::vector<uint8_t>&& payload,
                               nvidia::gxf::MessageMetadata metadata);

  FastDdsPubSubContext* context_;
  bool initialized_ = false;

  // Type support (registered once per participant, managed by DDS via TypeSupport wrapper)
  // Note: We store the type name to use for creating topics after registration
  std::string registered_type_name_;
  bool type_registered_ = false;

  // DDS Publisher and Subscriber (one each, shared by all writers/readers)
  eprosima::fastdds::dds::Publisher* publisher_ = nullptr;
  eprosima::fastdds::dds::Subscriber* subscriber_ = nullptr;

  // Writers by publisher GID
  struct WriterInfo {
    eprosima::fastdds::dds::DataWriter* writer = nullptr;
    eprosima::fastdds::dds::Topic* topic = nullptr;
    std::string topic_name;
  };
  std::unordered_map<nvidia::gxf::Gid, WriterInfo> writers_;

  // Readers by subscriber GID
  struct ReaderInfo {
    eprosima::fastdds::dds::DataReader* reader = nullptr;
    eprosima::fastdds::dds::Topic* topic = nullptr;
    std::string topic_name;
    nvidia::gxf::Gid subscriber_gid;
  };
  std::unordered_map<nvidia::gxf::Gid, ReaderInfo> readers_;

  // Topics by name (shared between writers and readers)
  std::unordered_map<std::string, eprosima::fastdds::dds::Topic*> topics_;

  mutable std::mutex endpoints_mutex_;

  // Callbacks (receive_callback_ is guarded by callback_mutex_)
  mutable std::mutex callback_mutex_;
  ReceiveCallback receive_callback_;
  ConnectionEstablishedCallback connection_established_callback_;
  ConnectionLostCallback connection_lost_callback_;

  // Listeners
  std::unique_ptr<ReaderListener> reader_listener_;
  std::unique_ptr<WriterListener> writer_listener_;

  // Type support (stored to ensure proper lifetime)
  eprosima::fastdds::dds::TypeSupport type_support_;

  // Native buffer (CUDA IPC) state
  bool native_buffers_enabled_ = false;

  // Sidecar writers for native descriptor messages (keyed by base topic name)
  struct SidecarWriterInfo {
    eprosima::fastdds::dds::DataWriter* writer = nullptr;
    eprosima::fastdds::dds::Topic* topic = nullptr;
  };
  std::unordered_map<std::string, SidecarWriterInfo> sidecar_writers_;

  // Sidecar readers for native descriptor messages
  struct SidecarReaderInfo {
    eprosima::fastdds::dds::DataReader* reader = nullptr;
    eprosima::fastdds::dds::Topic* topic = nullptr;
  };
  std::unordered_map<std::string, SidecarReaderInfo> sidecar_readers_;
  std::unique_ptr<SidecarReaderListener> sidecar_reader_listener_;

  std::unique_ptr<SidecarDispatchQueue> sidecar_dispatch_queue_;
};

}  // namespace holoscan

#endif  // HOLOSCAN_PUBSUB_FASTDDS_PUBSUB_FASTDDS_TRANSPORT_HPP
