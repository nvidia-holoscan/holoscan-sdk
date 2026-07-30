/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/pubsub/in_memory/network_contexts/gxf/in_memory_pubsub_session.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace holoscan {

// =============================================================================
// InMemoryPubSubSession — registry helpers
// =============================================================================

struct InMemoryPubSubSession::SessionRegistry {
  std::mutex mutex;
  std::unordered_map<std::string, std::weak_ptr<InMemoryPubSubSession>> sessions;
};

InMemoryPubSubSession::SessionRegistry& InMemoryPubSubSession::registry() {
  // Intentionally leaked to avoid static-destruction-order issues: sessions may be
  // torn down from other translation units' static destructors.
  static auto* registry = new SessionRegistry();
  return *registry;
}

std::shared_ptr<InMemoryPubSubSession> InMemoryPubSubSession::get_or_create(
    const std::string& session_id) {
  auto& registry_state = registry();
  std::lock_guard<std::mutex> lock(registry_state.mutex);

  for (auto it = registry_state.sessions.begin(); it != registry_state.sessions.end();) {
    if (it->second.expired()) {
      it = registry_state.sessions.erase(it);
    } else {
      ++it;
    }
  }

  auto it = registry_state.sessions.find(session_id);
  if (it != registry_state.sessions.end()) {
    if (auto existing = it->second.lock()) {
      return existing;
    }
    // Expired — will be replaced below
  }

  // Can't use make_shared because constructor is private
  auto session = std::shared_ptr<InMemoryPubSubSession>(new InMemoryPubSubSession(session_id));
  registry_state.sessions[session_id] = session;
  return session;
}

void InMemoryPubSubSession::reset_all_for_testing() {
  std::vector<std::shared_ptr<InMemoryPubSubSession>> live_sessions;
  auto& registry_state = registry();
  {
    std::lock_guard<std::mutex> registry_lock(registry_state.mutex);
    for (auto& [id, weak_session] : registry_state.sessions) {
      if (auto session = weak_session.lock()) {
        live_sessions.push_back(std::move(session));
      }
    }

    for (auto& session : live_sessions) {
      std::lock_guard<std::mutex> session_lock(session->mutex_);
      session->cleanup_expired_participants_locked();
      if (!session->participants_.empty()) {
        throw std::logic_error(
            "InMemoryPubSubSession::reset_all_for_testing() requires all frontends to be shut "
            "down");
      }
    }

    registry_state.sessions.clear();
  }

  for (auto& session : live_sessions) {
    std::lock_guard<std::mutex> session_lock(session->mutex_);
    session->reset_locked();
  }
}

InMemoryPubSubSession::InMemoryPubSubSession(const std::string& session_id)
    : session_id_(session_id) {}

InMemoryPubSubSession::~InMemoryPubSubSession() {
  std::lock_guard<std::mutex> lock(mutex_);
  reset_locked();
}

// =============================================================================
// InMemoryPubSubSession — frontend creation
// =============================================================================

InMemoryPubSubSession::Frontends InMemoryPubSubSession::create_frontends() {
  auto participant = std::make_shared<InMemorySessionParticipant>();
  return {
      std::make_shared<SessionDiscoveryFrontend>(shared_from_this(), participant),
      std::make_shared<SessionTransportFrontend>(shared_from_this(), participant),
  };
}

// =============================================================================
// InMemoryPubSubSession — context tracking
// =============================================================================

void InMemoryPubSubSession::join(gxf_context_t context) {
  std::lock_guard<std::mutex> lock(context_mutex_);
  contexts_.insert(context);
}

void InMemoryPubSubSession::leave(gxf_context_t context) {
  std::lock_guard<std::mutex> lock(context_mutex_);
  contexts_.erase(context);
}

bool InMemoryPubSubSession::is_multi_context() const {
  std::lock_guard<std::mutex> lock(context_mutex_);
  return contexts_.size() > 1;
}

// =============================================================================
// InMemoryPubSubSession — backend management
// =============================================================================

nvidia::gxf::Expected<void> InMemoryPubSubSession::ensure_backends() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!discovery_) {
    discovery_ = std::make_shared<nvidia::gxf::InMemoryDiscovery>();
    auto result = discovery_->initialize();
    if (!result) {
      return result;
    }
  }

  if (!transport_) {
    transport_ = std::make_shared<nvidia::gxf::InMemoryTransport>();
    auto result = transport_->initialize();
    if (!result) {
      return result;
    }
  }

  return {};
}

nvidia::gxf::Expected<void> InMemoryPubSubSession::retain_participant(
    const std::shared_ptr<InMemorySessionParticipant>& participant) {
  // Lock order: mutex_ first, then participant->mutex (matches notify_* paths).
  std::lock_guard<std::mutex> lock(mutex_);

  if (!discovery_) {
    discovery_ = std::make_shared<nvidia::gxf::InMemoryDiscovery>();
    auto result = discovery_->initialize();
    if (!result) {
      return result;
    }
  }

  if (!transport_) {
    transport_ = std::make_shared<nvidia::gxf::InMemoryTransport>();
    auto result = transport_->initialize();
    if (!result) {
      return result;
    }
  }

  {
    std::lock_guard<std::mutex> participant_lock(participant->mutex);
    participant->participant_closed = false;
  }
  cleanup_expired_participants_locked();
  auto id = next_participant_id_++;
  participants_[id] = participant;
  return {};
}

nvidia::gxf::Expected<void> InMemoryPubSubSession::release_participant(
    const std::shared_ptr<InMemorySessionParticipant>& participant) {
  std::vector<nvidia::gxf::PublisherGid> publishers;
  std::vector<nvidia::gxf::SubscriberGid> subscribers;

  // Lock order: mutex_ first, then participant->mutex (matches retain_participant
  // and notify_* paths).  Setting participant_closed and removing from participants_
  // must be atomic under mutex_ to prevent a concurrent retain_participant from
  // re-adding a closed participant.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    {
      std::lock_guard<std::mutex> participant_lock(participant->mutex);
      participant->participant_closed = true;
      participant->discovery_active = false;
      participant->transport_active = false;
      publishers.assign(participant->publisher_gids.begin(), participant->publisher_gids.end());
      subscribers.assign(participant->subscriber_gids.begin(), participant->subscriber_gids.end());
    }
    // Explicitly remove this participant's entry by matching the shared_ptr,
    // rather than relying on weak_ptr expiration (which won't happen while
    // the frontend objects still hold their shared_ptr to the participant).
    for (auto it = participants_.begin(); it != participants_.end(); ++it) {
      if (it->second.lock() == participant) {
        participants_.erase(it);
        break;
      }
    }
    // Also clean up any other expired entries.
    cleanup_expired_participants_locked();
  }

  // Remove publishers/subscribers outside mutex_ — these call notify_* which
  // re-acquire mutex_ via snapshot_participants().
  for (const auto& gid : publishers) {
    auto result = remove_publisher(participant, gid);
    if (!result) {
      return result;
    }
  }
  for (const auto& gid : subscribers) {
    auto result = remove_subscriber(participant, gid);
    if (!result) {
      return result;
    }
  }

  // Re-check under mutex_ whether we were the last participant and should
  // tear down the backends.  Another participant may have joined while we
  // were removing publishers/subscribers above.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    cleanup_expired_participants_locked();
    if (participants_.empty()) {
      reset_locked();
    }
  }
  return {};
}

nvidia::gxf::Expected<void> InMemoryPubSubSession::announce_publisher(
    const std::shared_ptr<InMemorySessionParticipant>& participant,
    const nvidia::gxf::PublisherInfo& info) {
  {
    std::lock_guard<std::mutex> lock(participant->mutex);
    if (participant->participant_closed) {
      return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
    }
    participant->publisher_gids.insert(info.gid);
  }

  auto discovery_copy = discovery();
  if (!discovery_copy) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }

  auto result = discovery_copy->announce_publisher(info);
  if (!result) {
    return result;
  }
  notify_publisher_discovered(info);
  return {};
}

nvidia::gxf::Expected<void> InMemoryPubSubSession::announce_subscriber(
    const std::shared_ptr<InMemorySessionParticipant>& participant,
    const nvidia::gxf::SubscriberInfo& info) {
  {
    std::lock_guard<std::mutex> lock(participant->mutex);
    if (participant->participant_closed) {
      return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
    }
    participant->subscriber_gids.insert(info.gid);
  }

  auto discovery_copy = discovery();
  auto transport_copy = transport();
  if (!discovery_copy || !transport_copy) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }

  auto result = discovery_copy->announce_subscriber(info);
  if (!result) {
    return result;
  }

  transport_copy->register_subscriber_endpoint(info.gid, make_receive_callback(participant));
  notify_subscriber_discovered(info);
  subscriber_wait_cv_.notify_all();
  return {};
}

nvidia::gxf::Expected<void> InMemoryPubSubSession::remove_publisher(
    const std::shared_ptr<InMemorySessionParticipant>& participant,
    const nvidia::gxf::PublisherGid& gid) {
  {
    std::lock_guard<std::mutex> lock(participant->mutex);
    participant->publisher_gids.erase(gid);
  }

  auto discovery_copy = discovery();
  if (!discovery_copy) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }

  auto result = discovery_copy->remove_publisher(gid);
  if (!result) {
    return result;
  }
  notify_publisher_lost(gid);
  return {};
}

nvidia::gxf::Expected<void> InMemoryPubSubSession::remove_subscriber(
    const std::shared_ptr<InMemorySessionParticipant>& participant,
    const nvidia::gxf::SubscriberGid& gid) {
  {
    std::lock_guard<std::mutex> lock(participant->mutex);
    participant->subscriber_gids.erase(gid);
  }

  auto discovery_copy = discovery();
  auto transport_copy = transport();
  if (!discovery_copy || !transport_copy) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }

  auto result = discovery_copy->remove_subscriber(gid);
  if (!result) {
    return result;
  }

  transport_copy->unregister_subscriber_endpoint(gid);
  notify_subscriber_lost(gid);
  subscriber_wait_cv_.notify_all();
  return {};
}

nvidia::gxf::Expected<std::vector<nvidia::gxf::PublisherInfo>>
InMemoryPubSubSession::query_publishers(const std::string& topic_name) const {
  auto discovery_copy = discovery();
  if (!discovery_copy) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }
  return discovery_copy->query_publishers(topic_name);
}

nvidia::gxf::Expected<std::vector<nvidia::gxf::SubscriberInfo>>
InMemoryPubSubSession::query_subscribers(const std::string& topic_name) const {
  auto discovery_copy = discovery();
  if (!discovery_copy) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }
  return discovery_copy->query_subscribers(topic_name);
}

nvidia::gxf::Expected<std::vector<std::string>> InMemoryPubSubSession::get_all_topics() const {
  auto discovery_copy = discovery();
  if (!discovery_copy) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }
  return discovery_copy->get_all_topics();
}

bool InMemoryPubSubSession::wait_for_subscribers(const std::string& topic_name,
                                                 size_t expected_count,
                                                 std::chrono::milliseconds timeout) const {
  std::unique_lock<std::mutex> lock(subscriber_wait_mutex_);
  return subscriber_wait_cv_.wait_for(lock, timeout, [&] {
    auto subscribers = query_subscribers(topic_name);
    return subscribers.has_value() && subscribers->size() >= expected_count;
  });
}

nvidia::gxf::Expected<void> InMemoryPubSubSession::send(
    const nvidia::gxf::Gid& destination_gid, const std::vector<uint8_t>& payload,
    const nvidia::gxf::MessageMetadata& metadata) const {
  auto transport_copy = transport();
  if (!transport_copy) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }
  return transport_copy->send(destination_gid, payload, metadata);
}

nvidia::gxf::Expected<void> InMemoryPubSubSession::send(
    const nvidia::gxf::Gid& destination_gid, std::vector<uint8_t>&& payload,
    const nvidia::gxf::MessageMetadata& metadata) const {
  auto transport_copy = transport();
  if (!transport_copy) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }
  return transport_copy->send(destination_gid, std::move(payload), metadata);
}

size_t InMemoryPubSubSession::send_queue_size() const {
  auto transport_copy = transport();
  return transport_copy ? transport_copy->get_send_queue_size() : 0UL;
}

// =============================================================================
// InMemoryPubSubSession — private helpers
// =============================================================================

nvidia::gxf::Expected<void> InMemoryPubSubSession::replay_existing_publishers(
    const std::shared_ptr<InMemorySessionParticipant>& participant) const {
  auto discovery_copy = discovery();
  if (!discovery_copy) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }

  nvidia::gxf::PubSubDiscovery::PublisherDiscoveredCallback publisher_callback;
  {
    std::lock_guard<std::mutex> lock(participant->mutex);
    publisher_callback = participant->publisher_discovered_callback;
  }
  if (!publisher_callback) {
    return {};
  }

  auto topics_result = discovery_copy->get_all_topics();
  if (!topics_result) {
    return nvidia::gxf::Unexpected{topics_result.error()};
  }

  for (const auto& topic : topics_result.value()) {
    auto publishers_result = discovery_copy->query_publishers(topic);
    if (!publishers_result) {
      return nvidia::gxf::Unexpected{publishers_result.error()};
    }
    for (const auto& info : publishers_result.value()) {
      publisher_callback(info);
    }
  }

  return {};
}

nvidia::gxf::Expected<void> InMemoryPubSubSession::replay_existing_subscribers(
    const std::shared_ptr<InMemorySessionParticipant>& participant) const {
  auto discovery_copy = discovery();
  if (!discovery_copy) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }

  nvidia::gxf::PubSubDiscovery::SubscriberDiscoveredCallback subscriber_callback;
  {
    std::lock_guard<std::mutex> lock(participant->mutex);
    subscriber_callback = participant->subscriber_discovered_callback;
  }
  if (!subscriber_callback) {
    return {};
  }

  auto topics_result = discovery_copy->get_all_topics();
  if (!topics_result) {
    return nvidia::gxf::Unexpected{topics_result.error()};
  }

  for (const auto& topic : topics_result.value()) {
    auto subscribers_result = discovery_copy->query_subscribers(topic);
    if (!subscribers_result) {
      return nvidia::gxf::Unexpected{subscribers_result.error()};
    }
    for (const auto& info : subscribers_result.value()) {
      subscriber_callback(info);
    }
  }

  return {};
}

std::shared_ptr<nvidia::gxf::InMemoryDiscovery> InMemoryPubSubSession::discovery() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return discovery_;
}

std::shared_ptr<nvidia::gxf::InMemoryTransport> InMemoryPubSubSession::transport() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return transport_;
}

std::vector<std::shared_ptr<InMemorySessionParticipant>>
InMemoryPubSubSession::snapshot_participants() {
  std::lock_guard<std::mutex> lock(mutex_);
  cleanup_expired_participants_locked();

  std::vector<std::shared_ptr<InMemorySessionParticipant>> snapshot;
  snapshot.reserve(participants_.size());
  for (const auto& [id, weak_participant] : participants_) {
    (void)id;
    if (auto participant = weak_participant.lock()) {
      snapshot.push_back(std::move(participant));
    }
  }
  return snapshot;
}

void InMemoryPubSubSession::cleanup_expired_participants_locked() {
  for (auto it = participants_.begin(); it != participants_.end();) {
    if (it->second.expired()) {
      it = participants_.erase(it);
    } else {
      ++it;
    }
  }
}

void InMemoryPubSubSession::reset_locked() {
  participants_.clear();
  if (transport_) {
    transport_->shutdown();
    transport_.reset();
  }
  if (discovery_) {
    discovery_->clear();
    discovery_->shutdown();
    discovery_.reset();
  }
  subscriber_wait_cv_.notify_all();
}

void InMemoryPubSubSession::notify_publisher_discovered(const nvidia::gxf::PublisherInfo& info) {
  for (const auto& participant : snapshot_participants()) {
    nvidia::gxf::PubSubDiscovery::PublisherDiscoveredCallback callback;
    {
      std::lock_guard<std::mutex> lock(participant->mutex);
      if (!participant->discovery_active) {
        continue;
      }
      callback = participant->publisher_discovered_callback;
    }
    if (callback) {
      callback(info);
    }
  }
}

void InMemoryPubSubSession::notify_subscriber_discovered(const nvidia::gxf::SubscriberInfo& info) {
  for (const auto& participant : snapshot_participants()) {
    nvidia::gxf::PubSubDiscovery::SubscriberDiscoveredCallback callback;
    {
      std::lock_guard<std::mutex> lock(participant->mutex);
      if (!participant->discovery_active) {
        continue;
      }
      callback = participant->subscriber_discovered_callback;
    }
    if (callback) {
      callback(info);
    }
  }
}

void InMemoryPubSubSession::notify_publisher_lost(const nvidia::gxf::PublisherGid& gid) {
  for (const auto& participant : snapshot_participants()) {
    nvidia::gxf::PubSubDiscovery::PublisherLostCallback callback;
    {
      std::lock_guard<std::mutex> lock(participant->mutex);
      if (!participant->discovery_active) {
        continue;
      }
      callback = participant->publisher_lost_callback;
    }
    if (callback) {
      callback(gid);
    }
  }
}

void InMemoryPubSubSession::notify_subscriber_lost(const nvidia::gxf::SubscriberGid& gid) {
  for (const auto& participant : snapshot_participants()) {
    nvidia::gxf::PubSubDiscovery::SubscriberLostCallback callback;
    {
      std::lock_guard<std::mutex> lock(participant->mutex);
      if (!participant->discovery_active) {
        continue;
      }
      callback = participant->subscriber_lost_callback;
    }
    if (callback) {
      callback(gid);
    }
  }
}

nvidia::gxf::PubSubTransport::ReceiveCallback InMemoryPubSubSession::make_receive_callback(
    const std::shared_ptr<InMemorySessionParticipant>& participant) {
  std::weak_ptr<InMemorySessionParticipant> weak_participant = participant;
  return [weak_participant](const nvidia::gxf::Gid& source_gid,
                            std::vector<uint8_t>&& payload,
                            const nvidia::gxf::MessageMetadata& metadata) {
    auto locked = weak_participant.lock();
    if (!locked) {
      return;
    }

    nvidia::gxf::PubSubTransport::ReceiveCallback callback;
    {
      std::lock_guard<std::mutex> lock(locked->mutex);
      if (!locked->transport_active) {
        return;
      }
      callback = locked->receive_callback;
    }
    if (callback) {
      callback(source_gid, std::move(payload), metadata);
    }
  };
}

// =============================================================================
// SessionDiscoveryFrontend
// =============================================================================

SessionDiscoveryFrontend::SessionDiscoveryFrontend(
    std::shared_ptr<InMemoryPubSubSession> session,
    std::shared_ptr<InMemorySessionParticipant> participant)
    : session_(std::move(session)), participant_(std::move(participant)) {}

nvidia::gxf::Expected<void> SessionDiscoveryFrontend::initialize() {
  if (initialized_.exchange(true)) {
    return nvidia::gxf::Unexpected{GXF_FAILURE};
  }
  auto result = session_->retain_participant(participant_);
  if (!result) {
    initialized_ = false;
    return result;
  }
  {
    std::lock_guard<std::mutex> lock(participant_->mutex);
    participant_->discovery_active = true;
  }
  return {};
}

nvidia::gxf::Expected<void> SessionDiscoveryFrontend::shutdown() {
  if (!initialized_.exchange(false)) {
    return {};
  }
  bool should_release_participant = false;
  // Clear callbacks so the participant stops receiving discovery notifications
  // immediately, even if the transport frontend still holds a shared_ptr.
  {
    std::lock_guard<std::mutex> lock(participant_->mutex);
    participant_->discovery_active = false;
    should_release_participant = !participant_->transport_active;
    participant_->publisher_discovered_callback = nullptr;
    participant_->subscriber_discovered_callback = nullptr;
    participant_->publisher_lost_callback = nullptr;
    participant_->subscriber_lost_callback = nullptr;
  }
  if (should_release_participant) {
    return session_->release_participant(participant_);
  }
  return {};
}

bool SessionDiscoveryFrontend::is_initialized() const {
  return initialized_.load();
}

nvidia::gxf::Expected<void> SessionDiscoveryFrontend::announce_publisher(
    const nvidia::gxf::PublisherInfo& info) {
  if (!initialized_.load()) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }
  return session_->announce_publisher(participant_, info);
}

nvidia::gxf::Expected<void> SessionDiscoveryFrontend::announce_subscriber(
    const nvidia::gxf::SubscriberInfo& info) {
  if (!initialized_.load()) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }
  return session_->announce_subscriber(participant_, info);
}

nvidia::gxf::Expected<void> SessionDiscoveryFrontend::remove_publisher(
    const nvidia::gxf::PublisherGid& gid) {
  if (!initialized_.load()) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }
  return session_->remove_publisher(participant_, gid);
}

nvidia::gxf::Expected<void> SessionDiscoveryFrontend::remove_subscriber(
    const nvidia::gxf::SubscriberGid& gid) {
  if (!initialized_.load()) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }
  return session_->remove_subscriber(participant_, gid);
}

nvidia::gxf::Expected<std::vector<nvidia::gxf::PublisherInfo>>
SessionDiscoveryFrontend::query_publishers(const std::string& topic_name) {
  if (!initialized_.load()) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }
  return session_->query_publishers(topic_name);
}

nvidia::gxf::Expected<std::vector<nvidia::gxf::SubscriberInfo>>
SessionDiscoveryFrontend::query_subscribers(const std::string& topic_name) {
  if (!initialized_.load()) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }
  return session_->query_subscribers(topic_name);
}

nvidia::gxf::Expected<std::vector<std::string>> SessionDiscoveryFrontend::get_all_topics() {
  if (!initialized_.load()) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }
  return session_->get_all_topics();
}

void SessionDiscoveryFrontend::set_on_publisher_discovered(
    nvidia::gxf::PubSubDiscovery::PublisherDiscoveredCallback callback) {
  std::lock_guard<std::mutex> lock(participant_->mutex);
  participant_->publisher_discovered_callback = std::move(callback);
}

void SessionDiscoveryFrontend::set_on_subscriber_discovered(
    nvidia::gxf::PubSubDiscovery::SubscriberDiscoveredCallback callback) {
  std::lock_guard<std::mutex> lock(participant_->mutex);
  participant_->subscriber_discovered_callback = std::move(callback);
}

void SessionDiscoveryFrontend::set_on_publisher_lost(
    nvidia::gxf::PubSubDiscovery::PublisherLostCallback callback) {
  std::lock_guard<std::mutex> lock(participant_->mutex);
  participant_->publisher_lost_callback = std::move(callback);
}

void SessionDiscoveryFrontend::set_on_subscriber_lost(
    nvidia::gxf::PubSubDiscovery::SubscriberLostCallback callback) {
  std::lock_guard<std::mutex> lock(participant_->mutex);
  participant_->subscriber_lost_callback = std::move(callback);
}

nvidia::gxf::Expected<void> SessionDiscoveryFrontend::replay_registered_endpoints() {
  if (!initialized_.load()) {
    return {};
  }
  auto publishers_result = session_->replay_existing_publishers(participant_);
  if (!publishers_result) {
    return publishers_result;
  }
  return session_->replay_existing_subscribers(participant_);
}

// =============================================================================
// SessionTransportFrontend
// =============================================================================

SessionTransportFrontend::SessionTransportFrontend(
    std::shared_ptr<InMemoryPubSubSession> session,
    std::shared_ptr<InMemorySessionParticipant> participant)
    : session_(std::move(session)), participant_(std::move(participant)) {}

nvidia::gxf::Expected<void> SessionTransportFrontend::initialize() {
  if (initialized_.exchange(true)) {
    return nvidia::gxf::Unexpected{GXF_FAILURE};
  }
  auto result = session_->ensure_backends();
  if (!result) {
    initialized_ = false;
    return result;
  }
  {
    std::lock_guard<std::mutex> lock(participant_->mutex);
    participant_->transport_active = true;
  }
  return {};
}

nvidia::gxf::Expected<void> SessionTransportFrontend::shutdown() {
  if (!initialized_.exchange(false)) {
    return {};
  }
  bool should_release_participant = false;
  // Clear transport callbacks so the participant stops receiving messages
  // immediately after shutdown.
  {
    std::lock_guard<std::mutex> lock(participant_->mutex);
    participant_->transport_active = false;
    should_release_participant = !participant_->discovery_active;
    participant_->receive_callback = nullptr;
    participant_->connection_established_callback = nullptr;
    participant_->connection_lost_callback = nullptr;
  }
  if (should_release_participant) {
    return session_->release_participant(participant_);
  }
  return {};
}

bool SessionTransportFrontend::is_initialized() const {
  return initialized_.load();
}

nvidia::gxf::Expected<void> SessionTransportFrontend::connect_to(
    const nvidia::gxf::EndpointInfo& remote_endpoint) {
  if (!initialized_.load()) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }

  nvidia::gxf::PubSubTransport::ConnectionEstablishedCallback callback;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    connections_[remote_endpoint.gid] = remote_endpoint;
  }
  {
    std::lock_guard<std::mutex> participant_lock(participant_->mutex);
    callback = participant_->connection_established_callback;
  }

  if (callback) {
    callback(remote_endpoint.gid);
  }
  return {};
}

nvidia::gxf::Expected<void> SessionTransportFrontend::disconnect_from(
    const nvidia::gxf::Gid& remote_gid) {
  if (!initialized_.load()) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }

  nvidia::gxf::PubSubTransport::ConnectionLostCallback callback;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    connections_.erase(remote_gid);
  }
  {
    std::lock_guard<std::mutex> participant_lock(participant_->mutex);
    callback = participant_->connection_lost_callback;
  }

  if (callback) {
    callback(remote_gid);
  }
  return {};
}

bool SessionTransportFrontend::is_connected_to(const nvidia::gxf::Gid& remote_gid) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return connections_.find(remote_gid) != connections_.end();
}

nvidia::gxf::Expected<void> SessionTransportFrontend::send(
    const nvidia::gxf::Gid& destination_gid, const std::vector<uint8_t>& payload,
    const nvidia::gxf::MessageMetadata& metadata) {
  if (!initialized_.load()) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }
  return session_->send(destination_gid, payload, metadata);
}

nvidia::gxf::Expected<void> SessionTransportFrontend::send(
    const nvidia::gxf::Gid& destination_gid, std::vector<uint8_t>&& payload,
    const nvidia::gxf::MessageMetadata& metadata) {
  if (!initialized_.load()) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }
  return session_->send(destination_gid, std::move(payload), metadata);
}

void SessionTransportFrontend::set_on_receive(
    nvidia::gxf::PubSubTransport::ReceiveCallback callback) {
  std::lock_guard<std::mutex> lock(participant_->mutex);
  participant_->receive_callback = std::move(callback);
}

void SessionTransportFrontend::set_on_connection_established(
    nvidia::gxf::PubSubTransport::ConnectionEstablishedCallback callback) {
  std::lock_guard<std::mutex> lock(participant_->mutex);
  participant_->connection_established_callback = std::move(callback);
}

void SessionTransportFrontend::set_on_connection_lost(
    nvidia::gxf::PubSubTransport::ConnectionLostCallback callback) {
  std::lock_guard<std::mutex> lock(participant_->mutex);
  participant_->connection_lost_callback = std::move(callback);
}

size_t SessionTransportFrontend::get_send_queue_size() const {
  return session_->send_queue_size();
}

size_t SessionTransportFrontend::get_connection_count() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return connections_.size();
}

// =============================================================================
// StdPubSubEntitySerializer
// =============================================================================

StdPubSubEntitySerializer::StdPubSubEntitySerializer(
    std::shared_ptr<StdEntitySerializer> serializer,
    std::shared_ptr<SerializationBuffer> serialize_buffer,
    std::shared_ptr<SerializationBuffer> deserialize_buffer, size_t buffer_size)
    : serializer_(std::move(serializer)),
      serialize_buffer_(std::move(serialize_buffer)),
      deserialize_buffer_(std::move(deserialize_buffer)),
      buffer_size_(buffer_size) {}

nvidia::gxf::Expected<std::vector<uint8_t>> StdPubSubEntitySerializer::serialize(
    nvidia::gxf::Entity entity, nvidia::gxf::Handle<nvidia::gxf::Allocator> allocator) {
  (void)allocator;

  std::lock_guard<std::mutex> lock(mutex_);
  auto* serializer = serializer_ ? serializer_->get() : nullptr;
  auto* buffer = serialize_buffer_ ? serialize_buffer_->get() : nullptr;
  if (serializer == nullptr || buffer == nullptr) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }

  auto resize_result = buffer->resize(buffer_size_, nvidia::gxf::MemoryStorageType::kSystem);
  if (!resize_result) {
    return nvidia::gxf::Unexpected{resize_result.error()};
  }

  uint64_t bytes_written = 0;
  const auto result = serializer->serialize_entity_abi(entity.eid(), buffer, &bytes_written);
  if (result != GXF_SUCCESS) {
    return nvidia::gxf::Unexpected{result};
  }

  return std::vector<uint8_t>(buffer->data(), buffer->data() + bytes_written);
}

nvidia::gxf::Expected<nvidia::gxf::Entity> StdPubSubEntitySerializer::deserialize(
    const std::vector<uint8_t>& data, gxf_context_t context,
    nvidia::gxf::Handle<nvidia::gxf::Allocator> allocator) {
  (void)allocator;

  std::lock_guard<std::mutex> lock(mutex_);
  auto* serializer = serializer_ ? serializer_->get() : nullptr;
  auto* buffer = deserialize_buffer_ ? deserialize_buffer_->get() : nullptr;
  if (serializer == nullptr || buffer == nullptr) {
    return nvidia::gxf::Unexpected{GXF_RESOURCE_NOT_INITIALIZED};
  }
  if (context != nullptr && serializer_->gxf_context() != nullptr &&
      serializer_->gxf_context() != context) {
    return nvidia::gxf::Unexpected{GXF_CONTEXT_INVALID};
  }

  auto resize_result = buffer->resize(data.size(), nvidia::gxf::MemoryStorageType::kSystem);
  if (!resize_result) {
    return nvidia::gxf::Unexpected{resize_result.error()};
  }

  if (!data.empty()) {
    size_t bytes_written = 0;
    const auto write_result = buffer->write_abi(data.data(), data.size(), &bytes_written);
    if (write_result != GXF_SUCCESS) {
      return nvidia::gxf::Unexpected{write_result};
    }
    if (bytes_written != data.size()) {
      return nvidia::gxf::Unexpected{GXF_FAILURE};
    }
  }

  return serializer->deserialize_entity_header_abi(buffer);
}

size_t StdPubSubEntitySerializer::estimate_size(nvidia::gxf::Entity entity) {
  (void)entity;
  return buffer_size_;
}

}  // namespace holoscan
