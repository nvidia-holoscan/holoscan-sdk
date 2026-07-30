/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_PUBSUB_IN_MEMORY_NETWORK_CONTEXTS_GXF_IN_MEMORY_PUBSUB_SESSION_HPP
#define HOLOSCAN_PUBSUB_IN_MEMORY_NETWORK_CONTEXTS_GXF_IN_MEMORY_PUBSUB_SESSION_HPP

#include <gxf/core/gxf.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <gxf/pubsub/endpoint_info.hpp>
#include <gxf/pubsub/in_memory_discovery.hpp>
#include <gxf/pubsub/in_memory_serializer.hpp>
#include <gxf/pubsub/in_memory_transport.hpp>
#include <gxf/pubsub/pubsub_discovery.hpp>
#include <gxf/pubsub/pubsub_entity_serializer.hpp>
#include <gxf/pubsub/pubsub_transport.hpp>

#include <holoscan/core/resources/gxf/serialization_buffer.hpp>
#include <holoscan/core/resources/gxf/std_entity_serializer.hpp>

namespace holoscan {

/// @brief Per-participant callback and GID tracking for a shared in-memory pub/sub session.
///
/// Each PubSubContext that joins a session gets its own SessionParticipant,
/// which stores the callbacks that PubSubContext::init_context() registers on
/// the discovery and transport frontends.
struct InMemorySessionParticipant {
  mutable std::mutex mutex;
  bool discovery_active = false;
  bool transport_active = false;
  bool participant_closed = false;
  nvidia::gxf::PubSubDiscovery::PublisherDiscoveredCallback publisher_discovered_callback;
  nvidia::gxf::PubSubDiscovery::SubscriberDiscoveredCallback subscriber_discovered_callback;
  nvidia::gxf::PubSubDiscovery::PublisherLostCallback publisher_lost_callback;
  nvidia::gxf::PubSubDiscovery::SubscriberLostCallback subscriber_lost_callback;
  nvidia::gxf::PubSubTransport::ReceiveCallback receive_callback;
  nvidia::gxf::PubSubTransport::ConnectionEstablishedCallback connection_established_callback;
  nvidia::gxf::PubSubTransport::ConnectionLostCallback connection_lost_callback;
  std::unordered_set<nvidia::gxf::PublisherGid> publisher_gids;
  std::unordered_set<nvidia::gxf::SubscriberGid> subscriber_gids;
};

/// @brief Process-global shared state for a group of in-memory pub/sub participants.
///
/// `InMemoryPubSubSession` consolidates the `SharedInMemoryPubSubBackend` and
/// `SharedInMemoryPubSubSession` into a single, reusable class.  It owns a shared
/// `InMemoryDiscovery` and `InMemoryTransport`, and provides per-participant
/// discovery/transport frontends that multiplex callbacks so that each PubSubContext's
/// `init_context()` wiring is independent. This was implemented to enable test cases
/// that involve launching multiple GXF contexts from the same process.
///
/// ## Usage
///
/// ```cpp
/// auto session = InMemoryPubSubSession::get_or_create("my_test_session");
/// session->join(gxf_context);
/// auto [discovery, transport] = session->create_frontends();
/// // ... inject into PubSubContext via set_discovery()/set_transport() ...
/// ```
///
/// ## Thread Safety
///
/// All public methods are thread-safe.
class InMemoryPubSubSession : public std::enable_shared_from_this<InMemoryPubSubSession> {
 public:
  /// Get or create a session by ID. Thread-safe.
  /// Returns the same shared_ptr for the same session_id within the process.
  static std::shared_ptr<InMemoryPubSubSession> get_or_create(const std::string& session_id);

  /// Release all sessions and reset shared state for tests.
  /// Must only be called when no session frontends are active.
  static void reset_all_for_testing();

  ~InMemoryPubSubSession();

  /// Create a paired discovery + transport frontend that share the same participant state.
  /// The discovery and transport frontends must be used together for a single PubSubContext —
  /// they share per-participant callback state so that PubSubContext::init_context() wiring
  /// is independent per context.
  struct Frontends {
    std::shared_ptr<nvidia::gxf::PubSubDiscovery> discovery;
    std::shared_ptr<nvidia::gxf::PubSubTransport> transport;
  };
  Frontends create_frontends();

  /// Join this session from a specific GXF context.
  void join(gxf_context_t context);

  /// Leave this session from a specific GXF context.
  void leave(gxf_context_t context);

  /// Returns true if more than one distinct GXF context has joined.
  bool is_multi_context() const;

  const std::string& session_id() const { return session_id_; }

  // ---- Internal methods used by frontends ----

  [[nodiscard]] nvidia::gxf::Expected<void> ensure_backends();

  [[nodiscard]] nvidia::gxf::Expected<void> retain_participant(
      const std::shared_ptr<InMemorySessionParticipant>& participant);
  [[nodiscard]] nvidia::gxf::Expected<void> release_participant(
      const std::shared_ptr<InMemorySessionParticipant>& participant);
  [[nodiscard]] nvidia::gxf::Expected<void> replay_existing_publishers(
      const std::shared_ptr<InMemorySessionParticipant>& participant) const;
  [[nodiscard]] nvidia::gxf::Expected<void> replay_existing_subscribers(
      const std::shared_ptr<InMemorySessionParticipant>& participant) const;

  [[nodiscard]] nvidia::gxf::Expected<void> announce_publisher(
      const std::shared_ptr<InMemorySessionParticipant>& participant,
      const nvidia::gxf::PublisherInfo& info);
  [[nodiscard]] nvidia::gxf::Expected<void> announce_subscriber(
      const std::shared_ptr<InMemorySessionParticipant>& participant,
      const nvidia::gxf::SubscriberInfo& info);
  [[nodiscard]] nvidia::gxf::Expected<void> remove_publisher(
      const std::shared_ptr<InMemorySessionParticipant>& participant,
      const nvidia::gxf::PublisherGid& gid);
  [[nodiscard]] nvidia::gxf::Expected<void> remove_subscriber(
      const std::shared_ptr<InMemorySessionParticipant>& participant,
      const nvidia::gxf::SubscriberGid& gid);

  [[nodiscard]] nvidia::gxf::Expected<std::vector<nvidia::gxf::PublisherInfo>> query_publishers(
      const std::string& topic_name) const;
  [[nodiscard]] nvidia::gxf::Expected<std::vector<nvidia::gxf::SubscriberInfo>> query_subscribers(
      const std::string& topic_name) const;
  [[nodiscard]] nvidia::gxf::Expected<std::vector<std::string>> get_all_topics() const;

  [[nodiscard]] bool wait_for_subscribers(const std::string& topic_name, size_t expected_count,
                                          std::chrono::milliseconds timeout) const;

  [[nodiscard]] nvidia::gxf::Expected<void> send(
      const nvidia::gxf::Gid& destination_gid, const std::vector<uint8_t>& payload,
      const nvidia::gxf::MessageMetadata& metadata) const;
  [[nodiscard]] nvidia::gxf::Expected<void> send(
      const nvidia::gxf::Gid& destination_gid, std::vector<uint8_t>&& payload,
      const nvidia::gxf::MessageMetadata& metadata) const;

  size_t send_queue_size() const;

 private:
  explicit InMemoryPubSubSession(const std::string& session_id);

  std::shared_ptr<nvidia::gxf::InMemoryDiscovery> discovery() const;
  std::shared_ptr<nvidia::gxf::InMemoryTransport> transport() const;

  std::vector<std::shared_ptr<InMemorySessionParticipant>> snapshot_participants();
  void cleanup_expired_participants_locked();
  void reset_locked();

  void notify_publisher_discovered(const nvidia::gxf::PublisherInfo& info);
  void notify_subscriber_discovered(const nvidia::gxf::SubscriberInfo& info);
  void notify_publisher_lost(const nvidia::gxf::PublisherGid& gid);
  void notify_subscriber_lost(const nvidia::gxf::SubscriberGid& gid);

  static nvidia::gxf::PubSubTransport::ReceiveCallback make_receive_callback(
      const std::shared_ptr<InMemorySessionParticipant>& participant);

  struct SessionRegistry;
  static SessionRegistry& registry();

  std::string session_id_;

  mutable std::mutex mutex_;
  mutable std::mutex subscriber_wait_mutex_;
  mutable std::condition_variable subscriber_wait_cv_;
  std::unordered_map<uint64_t, std::weak_ptr<InMemorySessionParticipant>> participants_;
  std::shared_ptr<nvidia::gxf::InMemoryDiscovery> discovery_;
  std::shared_ptr<nvidia::gxf::InMemoryTransport> transport_;
  uint64_t next_participant_id_ = 0;

  mutable std::mutex context_mutex_;
  std::set<gxf_context_t> contexts_;
};

/// @brief PubSubDiscovery frontend that delegates to a shared InMemoryPubSubSession.
///
/// Each instance maintains its own per-participant callbacks while forwarding
/// endpoint operations to the shared InMemoryDiscovery backend.
class SessionDiscoveryFrontend final : public nvidia::gxf::PubSubDiscovery {
 public:
  SessionDiscoveryFrontend(std::shared_ptr<InMemoryPubSubSession> session,
                           std::shared_ptr<InMemorySessionParticipant> participant);

  [[nodiscard]] nvidia::gxf::Expected<void> initialize() override;
  [[nodiscard]] nvidia::gxf::Expected<void> shutdown() override;
  bool is_initialized() const override;

  nvidia::gxf::DiscoveryModel discovery_model() const override {
    return nvidia::gxf::DiscoveryModel::kDecentralizedPassive;
  }

  [[nodiscard]] nvidia::gxf::Expected<void> announce_publisher(
      const nvidia::gxf::PublisherInfo& info) override;
  [[nodiscard]] nvidia::gxf::Expected<void> announce_subscriber(
      const nvidia::gxf::SubscriberInfo& info) override;
  [[nodiscard]] nvidia::gxf::Expected<void> remove_publisher(
      const nvidia::gxf::PublisherGid& gid) override;
  [[nodiscard]] nvidia::gxf::Expected<void> remove_subscriber(
      const nvidia::gxf::SubscriberGid& gid) override;

  [[nodiscard]] nvidia::gxf::Expected<std::vector<nvidia::gxf::PublisherInfo>> query_publishers(
      const std::string& topic_name) override;
  [[nodiscard]] nvidia::gxf::Expected<std::vector<nvidia::gxf::SubscriberInfo>> query_subscribers(
      const std::string& topic_name) override;
  [[nodiscard]] nvidia::gxf::Expected<std::vector<std::string>> get_all_topics() override;

  void set_on_publisher_discovered(
      nvidia::gxf::PubSubDiscovery::PublisherDiscoveredCallback callback) override;
  void set_on_subscriber_discovered(
      nvidia::gxf::PubSubDiscovery::SubscriberDiscoveredCallback callback) override;
  void set_on_publisher_lost(nvidia::gxf::PubSubDiscovery::PublisherLostCallback callback) override;
  void set_on_subscriber_lost(
      nvidia::gxf::PubSubDiscovery::SubscriberLostCallback callback) override;
  [[nodiscard]] nvidia::gxf::Expected<void> replay_registered_endpoints();

 private:
  std::shared_ptr<InMemoryPubSubSession> session_;
  std::shared_ptr<InMemorySessionParticipant> participant_;
  std::atomic<bool> initialized_{false};
};

/// @brief PubSubTransport frontend that delegates to a shared InMemoryPubSubSession.
///
/// Each instance maintains its own per-participant connection state while
/// forwarding send operations to the shared InMemoryTransport backend.
class SessionTransportFrontend final : public nvidia::gxf::PubSubTransport {
 public:
  SessionTransportFrontend(std::shared_ptr<InMemoryPubSubSession> session,
                           std::shared_ptr<InMemorySessionParticipant> participant);

  using nvidia::gxf::PubSubTransport::send;

  [[nodiscard]] nvidia::gxf::Expected<void> initialize() override;
  [[nodiscard]] nvidia::gxf::Expected<void> shutdown() override;
  bool is_initialized() const override;

  nvidia::gxf::TransportModel transport_model() const override {
    return nvidia::gxf::TransportModel::kEndpointAddressed;
  }
  bool native_topic_matching() const override { return false; }
  bool native_qos_enforcement() const override { return false; }
  bool supports_multicast() const override { return false; }
  bool requires_explicit_connections() const override { return true; }

  [[nodiscard]] nvidia::gxf::Expected<void> connect_to(
      const nvidia::gxf::EndpointInfo& remote_endpoint) override;
  [[nodiscard]] nvidia::gxf::Expected<void> disconnect_from(
      const nvidia::gxf::Gid& remote_gid) override;
  bool is_connected_to(const nvidia::gxf::Gid& remote_gid) const override;

  [[nodiscard]] nvidia::gxf::Expected<void> send(
      const nvidia::gxf::Gid& destination_gid, const std::vector<uint8_t>& payload,
      const nvidia::gxf::MessageMetadata& metadata) override;
  [[nodiscard]] nvidia::gxf::Expected<void> send(
      const nvidia::gxf::Gid& destination_gid, std::vector<uint8_t>&& payload,
      const nvidia::gxf::MessageMetadata& metadata) override;

  void set_on_receive(nvidia::gxf::PubSubTransport::ReceiveCallback callback) override;
  void set_on_connection_established(
      nvidia::gxf::PubSubTransport::ConnectionEstablishedCallback callback) override;
  void set_on_connection_lost(
      nvidia::gxf::PubSubTransport::ConnectionLostCallback callback) override;

  size_t get_send_queue_size() const override;
  size_t get_receive_queue_size() const override { return 0UL; }
  size_t get_connection_count() const override;

 private:
  std::shared_ptr<InMemoryPubSubSession> session_;
  std::shared_ptr<InMemorySessionParticipant> participant_;
  mutable std::mutex mutex_;
  std::unordered_map<nvidia::gxf::Gid, nvidia::gxf::EndpointInfo> connections_;
  std::atomic<bool> initialized_{false};
};

/// @brief Adapter that bridges Holoscan's StdEntitySerializer with GXF's PubSubEntitySerializer.
///
/// `StdPubSubEntitySerializer` wraps a Holoscan `StdEntitySerializer` (which uses
/// `SerializationBuffer`) and presents it via the GXF `PubSubEntitySerializer`
/// interface (which works with `std::vector<uint8_t>`).
///
class StdPubSubEntitySerializer final : public nvidia::gxf::PubSubEntitySerializer {
 public:
  StdPubSubEntitySerializer(std::shared_ptr<StdEntitySerializer> serializer,
                            std::shared_ptr<SerializationBuffer> serialize_buffer,
                            std::shared_ptr<SerializationBuffer> deserialize_buffer,
                            size_t buffer_size);

  nvidia::gxf::Expected<std::vector<uint8_t>> serialize(
      nvidia::gxf::Entity entity, nvidia::gxf::Handle<nvidia::gxf::Allocator> allocator =
                                      nvidia::gxf::Handle<nvidia::gxf::Allocator>()) override;

  nvidia::gxf::Expected<nvidia::gxf::Entity> deserialize(
      const std::vector<uint8_t>& data, gxf_context_t context,
      nvidia::gxf::Handle<nvidia::gxf::Allocator> allocator) override;

  size_t estimate_size(nvidia::gxf::Entity entity) override;

  const char* name() const override { return "StdPubSubEntitySerializer"; }

 private:
  std::shared_ptr<StdEntitySerializer> serializer_;
  std::shared_ptr<SerializationBuffer> serialize_buffer_;
  std::shared_ptr<SerializationBuffer> deserialize_buffer_;
  size_t buffer_size_;
  std::mutex mutex_;
};

}  // namespace holoscan

#endif  // HOLOSCAN_PUBSUB_IN_MEMORY_NETWORK_CONTEXTS_GXF_IN_MEMORY_PUBSUB_SESSION_HPP
