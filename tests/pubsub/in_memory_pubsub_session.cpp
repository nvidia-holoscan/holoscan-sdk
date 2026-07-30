/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "../utils.hpp"

#define private public
#include <gxf/pubsub/in_memory_serializer.hpp>
#include <holoscan/pubsub/in_memory/network_contexts/gxf/in_memory_pubsub_network_context.hpp>
#include <holoscan/pubsub/in_memory/network_contexts/gxf/in_memory_pubsub_session.hpp>
#undef private

#include <gxf/pubsub/pubsub_context.hpp>
#include <holoscan/core/resources/gxf/serialization_buffer.hpp>
#include <holoscan/core/resources/gxf/std_entity_serializer.hpp>
#include <holoscan/core/resources/gxf/unbounded_allocator.hpp>

namespace holoscan {

namespace {

class TestStdEntitySerializerResource : public StdEntitySerializer {
 public:
  void set_gxf_serializer(nvidia::gxf::StdEntitySerializer* serializer) { gxf_cptr_ = serializer; }
};

class TestSerializationBufferResource : public SerializationBuffer {
 public:
  void set_gxf_buffer(nvidia::gxf::SerializationBuffer* buffer) { gxf_cptr_ = buffer; }
};

template <typename T>
void expect_gxf_error(const nvidia::gxf::Expected<T>& result, gxf_result_t expected_error) {
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), expected_error);
}

nvidia::gxf::PublisherInfo make_publisher_info(const std::string& topic_name,
                                               const std::string& node_name = "publisher_node") {
  nvidia::gxf::PublisherInfo info;
  info.gid = nvidia::gxf::Gid::generate();
  info.topic_name = topic_name;
  info.node_name = node_name;
  return info;
}

nvidia::gxf::SubscriberInfo make_subscriber_info(const std::string& topic_name,
                                                 const std::string& node_name = "subscriber_node") {
  nvidia::gxf::SubscriberInfo info;
  info.gid = nvidia::gxf::Gid::generate();
  info.topic_name = topic_name;
  info.node_name = node_name;
  return info;
}

}  // namespace

class InMemoryPubSubSessionTest : public ::testing::Test {
 protected:
  void SetUp() override { InMemoryPubSubSession::reset_all_for_testing(); }
  void TearDown() override { InMemoryPubSubSession::reset_all_for_testing(); }
};

class InMemoryPubSubSessionWithGXFContextTest : public TestWithGXFContext {
 protected:
  void SetUp() override {
    TestWithGXFContext::SetUp();
    auto extension_manager = F.executor().extension_manager();
    ASSERT_TRUE(extension_manager->load_extension("libgxf_pubsub.so", true));
    InMemoryPubSubSession::reset_all_for_testing();
  }

  void TearDown() override { InMemoryPubSubSession::reset_all_for_testing(); }
};

// =============================================================================
// Session lifecycle tests
// =============================================================================

TEST_F(InMemoryPubSubSessionTest, GetOrCreateReturnsSameSessionForSameId) {
  auto session1 = InMemoryPubSubSession::get_or_create("test_session");
  auto session2 = InMemoryPubSubSession::get_or_create("test_session");
  EXPECT_EQ(session1.get(), session2.get());
  EXPECT_EQ(session1->session_id(), "test_session");
}

TEST_F(InMemoryPubSubSessionTest, GetOrCreateReturnsDifferentSessionsForDifferentIds) {
  auto session_a = InMemoryPubSubSession::get_or_create("session_a");
  auto session_b = InMemoryPubSubSession::get_or_create("session_b");
  EXPECT_NE(session_a.get(), session_b.get());
  EXPECT_EQ(session_a->session_id(), "session_a");
  EXPECT_EQ(session_b->session_id(), "session_b");
}

TEST_F(InMemoryPubSubSessionTest, ExpiredSessionIsRecreated) {
  // Verify that after a session expires, get_or_create produces a fresh instance.
  // We can't compare raw pointers (allocator may reuse the address), so instead
  // we verify the new session has clean state by checking that backends created
  // under the old session are not present in the new one.
  {
    auto session = InMemoryPubSubSession::get_or_create("ephemeral");
    // Create backends so the session has non-trivial state
    EXPECT_TRUE(session->ensure_backends().has_value());
  }
  // Session should be expired now (weak_ptr in registry); backends were reset in destructor.
  auto session2 = InMemoryPubSubSession::get_or_create("ephemeral");
  EXPECT_EQ(session2->session_id(), "ephemeral");
  // The new session should have no topics registered (fresh backends)
  auto topics = session2->get_all_topics();
  // get_all_topics returns error when backends haven't been created yet
  EXPECT_FALSE(topics.has_value());
}

TEST_F(InMemoryPubSubSessionTest, ResetAllClearsRegistry) {
  auto session = InMemoryPubSubSession::get_or_create("to_reset");
  auto* raw_ptr = session.get();

  InMemoryPubSubSession::reset_all_for_testing();

  auto session2 = InMemoryPubSubSession::get_or_create("to_reset");
  // After reset_all, a new session is created even though we still hold the old one
  // (reset_all clears the registry, so get_or_create won't find the old weak_ptr)
  EXPECT_NE(session2.get(), raw_ptr);
}

TEST_F(InMemoryPubSubSessionTest, ResetAllForTestingRejectsActiveParticipants) {
  auto session = InMemoryPubSubSession::get_or_create("active_reset");
  auto frontends = session->create_frontends();

  ASSERT_TRUE(frontends.discovery->initialize().has_value());
  ASSERT_TRUE(frontends.transport->initialize().has_value());

  EXPECT_THROW(InMemoryPubSubSession::reset_all_for_testing(), std::logic_error);

  ASSERT_TRUE(frontends.discovery->shutdown().has_value());
  ASSERT_TRUE(frontends.transport->shutdown().has_value());
}

TEST_F(InMemoryPubSubSessionTest, ResetAllForTestingDoesNotOrphanConcurrentGetOrCreate) {
  auto blocking_session = InMemoryPubSubSession::get_or_create("blocking_reset");

  std::promise<void> reset_started;
  auto resetter = std::async(std::launch::async, [&] {
    reset_started.set_value();
    InMemoryPubSubSession::reset_all_for_testing();
  });

  std::future<std::shared_ptr<InMemoryPubSubSession>> creator;
  std::future_status creator_status{};
  {
    std::unique_lock<std::mutex> blocking_lock(blocking_session->mutex_);
    reset_started.get_future().wait();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    creator = std::async(std::launch::async, [] {
      return InMemoryPubSubSession::get_or_create("concurrent_get_or_create");
    });
    creator_status = creator.wait_for(std::chrono::milliseconds(50));

    EXPECT_EQ(creator_status, std::future_status::timeout)
        << "get_or_create() should remain blocked until reset_all_for_testing() completes";
  }

  ASSERT_EQ(resetter.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  EXPECT_NO_THROW(resetter.get());

  ASSERT_EQ(creator.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  auto concurrent_session = creator.get();
  auto after_reset = InMemoryPubSubSession::get_or_create("concurrent_get_or_create");
  EXPECT_EQ(after_reset.get(), concurrent_session.get())
      << "reset_all_for_testing() orphaned a concurrently created session";
}

// =============================================================================
// Context tracking tests
// =============================================================================

TEST_F(InMemoryPubSubSessionTest, SingleContextIsNotMultiContext) {
  auto session = InMemoryPubSubSession::get_or_create("ctx_test");
  EXPECT_FALSE(session->is_multi_context());

  // Use dummy pointers as gxf_context_t (void*)
  int dummy_a = 1;
  session->join(reinterpret_cast<gxf_context_t>(&dummy_a));
  EXPECT_FALSE(session->is_multi_context());
}

TEST_F(InMemoryPubSubSessionTest, TwoContextsIsMultiContext) {
  auto session = InMemoryPubSubSession::get_or_create("ctx_test");

  int dummy_a = 1;
  int dummy_b = 2;
  session->join(reinterpret_cast<gxf_context_t>(&dummy_a));
  session->join(reinterpret_cast<gxf_context_t>(&dummy_b));
  EXPECT_TRUE(session->is_multi_context());
}

TEST_F(InMemoryPubSubSessionTest, LeaveReducesContextCount) {
  auto session = InMemoryPubSubSession::get_or_create("ctx_test");

  int dummy_a = 1;
  int dummy_b = 2;
  session->join(reinterpret_cast<gxf_context_t>(&dummy_a));
  session->join(reinterpret_cast<gxf_context_t>(&dummy_b));
  EXPECT_TRUE(session->is_multi_context());

  session->leave(reinterpret_cast<gxf_context_t>(&dummy_a));
  EXPECT_FALSE(session->is_multi_context());
}

// =============================================================================
// Frontend creation tests
// =============================================================================

TEST_F(InMemoryPubSubSessionTest, CreateFrontendsReturnsNonNull) {
  auto session = InMemoryPubSubSession::get_or_create("frontend_test");
  auto frontends = session->create_frontends();
  EXPECT_NE(frontends.discovery, nullptr);
  EXPECT_NE(frontends.transport, nullptr);
}

TEST_F(InMemoryPubSubSessionTest, FrontendDiscoveryInitializeAndShutdown) {
  auto session = InMemoryPubSubSession::get_or_create("discovery_lifecycle");
  auto frontends = session->create_frontends();

  EXPECT_FALSE(frontends.discovery->is_initialized());

  auto result = frontends.discovery->initialize();
  EXPECT_TRUE(result.has_value());
  EXPECT_TRUE(frontends.discovery->is_initialized());

  auto shutdown_result = frontends.discovery->shutdown();
  EXPECT_TRUE(shutdown_result.has_value());
  EXPECT_FALSE(frontends.discovery->is_initialized());
}

TEST_F(InMemoryPubSubSessionTest, FrontendTransportInitializeAndShutdown) {
  auto session = InMemoryPubSubSession::get_or_create("transport_lifecycle");
  auto frontends = session->create_frontends();

  EXPECT_FALSE(frontends.transport->is_initialized());

  auto result = frontends.transport->initialize();
  EXPECT_TRUE(result.has_value());
  EXPECT_TRUE(frontends.transport->is_initialized());

  auto shutdown_result = frontends.transport->shutdown();
  EXPECT_TRUE(shutdown_result.has_value());
  EXPECT_FALSE(frontends.transport->is_initialized());
}

TEST_F(InMemoryPubSubSessionTest, DoubleInitializeFails) {
  auto session = InMemoryPubSubSession::get_or_create("double_init");
  auto frontends = session->create_frontends();

  EXPECT_TRUE(frontends.discovery->initialize().has_value());
  EXPECT_FALSE(frontends.discovery->initialize().has_value());  // second init fails

  EXPECT_TRUE(frontends.transport->initialize().has_value());
  EXPECT_FALSE(frontends.transport->initialize().has_value());  // second init fails
}

// =============================================================================
// Session isolation tests
// =============================================================================

TEST_F(InMemoryPubSubSessionTest, DifferentSessionsHaveIndependentBackends) {
  auto session_a = InMemoryPubSubSession::get_or_create("isolated_a");
  auto session_b = InMemoryPubSubSession::get_or_create("isolated_b");

  auto frontends_a = session_a->create_frontends();
  auto frontends_b = session_b->create_frontends();

  EXPECT_TRUE(frontends_a.discovery->initialize().has_value());
  EXPECT_TRUE(frontends_b.discovery->initialize().has_value());

  // Register a publisher on session A
  nvidia::gxf::PublisherInfo pub_info;
  pub_info.gid = nvidia::gxf::Gid::generate();
  pub_info.topic_name = "/test/topic";
  pub_info.node_name = "node_a";
  EXPECT_TRUE(frontends_a.discovery->announce_publisher(pub_info).has_value());

  // Session A should see the publisher
  auto pubs_a = session_a->query_publishers("/test/topic");
  ASSERT_TRUE(pubs_a.has_value());
  EXPECT_EQ(pubs_a->size(), 1UL);

  // Session B should NOT see the publisher
  auto pubs_b = session_b->query_publishers("/test/topic");
  ASSERT_TRUE(pubs_b.has_value());
  EXPECT_EQ(pubs_b->size(), 0UL);

  frontends_a.discovery->shutdown();
  frontends_b.discovery->shutdown();
}

// =============================================================================
// Multiple frontends per session (callback multiplexing)
// =============================================================================

TEST_F(InMemoryPubSubSessionTest, TwoFrontendsReceiveDiscoveryCallbacks) {
  auto session = InMemoryPubSubSession::get_or_create("multiplex");

  auto frontends_1 = session->create_frontends();
  auto frontends_2 = session->create_frontends();

  // Track callbacks received by each frontend
  int discovered_count_1 = 0;
  int discovered_count_2 = 0;

  frontends_1.discovery->set_on_publisher_discovered(
      [&discovered_count_1](const nvidia::gxf::PublisherInfo&) { ++discovered_count_1; });
  frontends_2.discovery->set_on_publisher_discovered(
      [&discovered_count_2](const nvidia::gxf::PublisherInfo&) { ++discovered_count_2; });

  EXPECT_TRUE(frontends_1.discovery->initialize().has_value());
  EXPECT_TRUE(frontends_2.discovery->initialize().has_value());

  // Announce a publisher via frontend 1
  nvidia::gxf::PublisherInfo pub_info;
  pub_info.gid = nvidia::gxf::Gid::generate();
  pub_info.topic_name = "/multiplex/topic";
  pub_info.node_name = "publisher_node";
  EXPECT_TRUE(frontends_1.discovery->announce_publisher(pub_info).has_value());

  // Both frontends should have received the discovery callback
  EXPECT_EQ(discovered_count_1, 1);
  EXPECT_EQ(discovered_count_2, 1);

  frontends_1.discovery->shutdown();
  frontends_2.discovery->shutdown();
}

TEST_F(InMemoryPubSubSessionTest,
       LateJoinerReceivesExistingPublisherAfterCallbackInstalledPostInitialize) {
  auto session = InMemoryPubSubSession::get_or_create("late_join_replay");

  auto frontends_1 = session->create_frontends();
  auto frontends_2 = session->create_frontends();

  ASSERT_TRUE(frontends_1.discovery->initialize().has_value());

  nvidia::gxf::PublisherInfo pub_info;
  pub_info.gid = nvidia::gxf::Gid::generate();
  pub_info.topic_name = "/late_join/topic";
  pub_info.node_name = "publisher_node";
  ASSERT_TRUE(frontends_1.discovery->announce_publisher(pub_info).has_value());

  ASSERT_TRUE(frontends_2.discovery->initialize().has_value());

  int discovered_count = 0;
  std::vector<std::string> discovered_topics;
  frontends_2.discovery->set_on_publisher_discovered(
      [&](const nvidia::gxf::PublisherInfo& discovered) {
        ++discovered_count;
        discovered_topics.push_back(discovered.topic_name);
      });
  auto replay_frontend = std::dynamic_pointer_cast<SessionDiscoveryFrontend>(frontends_2.discovery);
  ASSERT_NE(replay_frontend, nullptr);
  ASSERT_TRUE(replay_frontend->replay_registered_endpoints().has_value());

  EXPECT_EQ(discovered_count, 1)
      << "Late joiners should receive already-registered endpoints once callbacks are installed";
  ASSERT_EQ(discovered_topics.size(), 1UL);
  EXPECT_EQ(discovered_topics.front(), "/late_join/topic");

  frontends_2.discovery->shutdown();
  frontends_1.discovery->shutdown();
}

TEST_F(InMemoryPubSubSessionTest, WaitForSubscribersUnblocksOnAnnouncement) {
  auto session = InMemoryPubSubSession::get_or_create("wait_for_subscribers");
  auto frontends = session->create_frontends();

  ASSERT_TRUE(frontends.discovery->initialize().has_value());
  ASSERT_TRUE(frontends.transport->initialize().has_value());

  auto waiter = std::async(std::launch::async, [&] {
    return session->wait_for_subscribers("/wait/topic", 1UL, std::chrono::milliseconds(500));
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  nvidia::gxf::SubscriberInfo sub_info;
  sub_info.gid = nvidia::gxf::Gid::generate();
  sub_info.topic_name = "/wait/topic";
  sub_info.node_name = "subscriber_node";
  ASSERT_TRUE(frontends.discovery->announce_subscriber(sub_info).has_value());

  EXPECT_EQ(waiter.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  EXPECT_TRUE(waiter.get());

  ASSERT_TRUE(frontends.discovery->shutdown().has_value());
  ASSERT_TRUE(frontends.transport->shutdown().has_value());
}

TEST_F(InMemoryPubSubSessionTest, DeserializeRejectsContextMismatch) {
  auto serializer = std::make_shared<TestStdEntitySerializerResource>();
  auto serialize_buffer = std::make_shared<TestSerializationBufferResource>();
  auto deserialize_buffer = std::make_shared<TestSerializationBufferResource>();

  auto gxf_serializer = std::make_unique<nvidia::gxf::StdEntitySerializer>();
  auto gxf_serialize_buffer = std::make_unique<nvidia::gxf::SerializationBuffer>();
  auto gxf_deserialize_buffer = std::make_unique<nvidia::gxf::SerializationBuffer>();

  serializer->set_gxf_serializer(gxf_serializer.get());
  serialize_buffer->set_gxf_buffer(gxf_serialize_buffer.get());
  deserialize_buffer->set_gxf_buffer(gxf_deserialize_buffer.get());

  int serializer_context_token = 0;
  int other_context_token = 0;
  serializer->gxf_context(reinterpret_cast<gxf_context_t>(&serializer_context_token));

  StdPubSubEntitySerializer pubsub_serializer(
      serializer, serialize_buffer, deserialize_buffer, 1024);

  auto result =
      pubsub_serializer.deserialize({}, reinterpret_cast<gxf_context_t>(&other_context_token), {});

  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), GXF_CONTEXT_INVALID);
}

TEST_F(InMemoryPubSubSessionWithGXFContextTest, DeserializeCopiesViaWriteAbiWithoutConstCast) {
  auto allocator = F.make_resource<UnboundedAllocator>("deserialize_allocator");
  allocator->initialize();

  auto serializer = F.make_resource<StdEntitySerializer>("deserialize_serializer");
  serializer->initialize();

  auto serialize_buffer =
      F.make_resource<SerializationBuffer>("serialize_buffer",
                                           Arg("allocator", allocator),
                                           Arg("buffer_size", static_cast<size_t>(256)));
  auto deserialize_buffer =
      F.make_resource<SerializationBuffer>("deserialize_buffer",
                                           Arg("allocator", allocator),
                                           Arg("buffer_size", static_cast<size_t>(256)));
  serialize_buffer->initialize();
  deserialize_buffer->initialize();
  auto* deserialize_gxf_buffer = deserialize_buffer->get();

  StdPubSubEntitySerializer pubsub_serializer(
      serializer, std::move(serialize_buffer), std::move(deserialize_buffer), 256);

  std::vector<uint8_t> payload = {1, 2, 3, 4, 5, 6, 7, 8};
  auto result = pubsub_serializer.deserialize(payload, serializer->gxf_context(), {});

  EXPECT_FALSE(result.has_value());
  ASSERT_NE(deserialize_gxf_buffer, nullptr);
  EXPECT_EQ(deserialize_gxf_buffer->size(), payload.size());
}

// =============================================================================
// Transport send/receive through session
// =============================================================================

TEST_F(InMemoryPubSubSessionTest, TransportSendToRegisteredSubscriber) {
  auto session = InMemoryPubSubSession::get_or_create("transport_test");

  auto frontends = session->create_frontends();
  EXPECT_TRUE(frontends.discovery->initialize().has_value());
  EXPECT_TRUE(frontends.transport->initialize().has_value());

  // Register a subscriber with a receive callback
  nvidia::gxf::SubscriberInfo sub_info;
  sub_info.gid = nvidia::gxf::Gid::generate();
  sub_info.topic_name = "/transport/topic";
  sub_info.node_name = "subscriber_node";

  std::vector<uint8_t> received_payload;
  frontends.transport->set_on_receive([&received_payload](const nvidia::gxf::Gid&,
                                                          std::vector<uint8_t>&& payload,
                                                          const nvidia::gxf::MessageMetadata&) {
    received_payload = std::move(payload);
  });

  EXPECT_TRUE(frontends.discovery->announce_subscriber(sub_info).has_value());

  // Send a message to the subscriber
  std::vector<uint8_t> test_payload = {1, 2, 3, 4, 5};
  nvidia::gxf::MessageMetadata metadata{};
  auto send_result = session->send(sub_info.gid, test_payload, metadata);
  EXPECT_TRUE(send_result.has_value());

  EXPECT_EQ(received_payload, test_payload);

  frontends.discovery->shutdown();
  frontends.transport->shutdown();
}

// =============================================================================
// EnsureBackends idempotency
// =============================================================================

TEST_F(InMemoryPubSubSessionTest, EnsureBackendsIsIdempotent) {
  auto session = InMemoryPubSubSession::get_or_create("idempotent_test");

  auto result1 = session->ensure_backends();
  EXPECT_TRUE(result1.has_value());

  auto result2 = session->ensure_backends();
  EXPECT_TRUE(result2.has_value());
}

// =============================================================================
// Transport connection management
// =============================================================================

TEST_F(InMemoryPubSubSessionTest, TransportConnectionManagement) {
  auto session = InMemoryPubSubSession::get_or_create("connection_test");
  auto frontends = session->create_frontends();
  EXPECT_TRUE(frontends.transport->initialize().has_value());

  nvidia::gxf::EndpointInfo endpoint;
  endpoint.gid = nvidia::gxf::Gid::generate();
  endpoint.node_name = "remote_node";

  EXPECT_FALSE(frontends.transport->is_connected_to(endpoint.gid));
  EXPECT_EQ(frontends.transport->get_connection_count(), 0UL);

  EXPECT_TRUE(frontends.transport->connect_to(endpoint).has_value());
  EXPECT_TRUE(frontends.transport->is_connected_to(endpoint.gid));
  EXPECT_EQ(frontends.transport->get_connection_count(), 1UL);

  EXPECT_TRUE(frontends.transport->disconnect_from(endpoint.gid).has_value());
  EXPECT_FALSE(frontends.transport->is_connected_to(endpoint.gid));
  EXPECT_EQ(frontends.transport->get_connection_count(), 0UL);

  frontends.transport->shutdown();
}

// =============================================================================
// Transport rejects operations when not initialized
// =============================================================================

TEST_F(InMemoryPubSubSessionTest, TransportRejectsOperationsWhenNotInitialized) {
  auto session = InMemoryPubSubSession::get_or_create("uninit_test");
  auto frontends = session->create_frontends();
  // Don't initialize transport

  nvidia::gxf::EndpointInfo endpoint;
  endpoint.gid = nvidia::gxf::Gid::generate();
  EXPECT_FALSE(frontends.transport->connect_to(endpoint).has_value());
  EXPECT_FALSE(frontends.transport->disconnect_from(endpoint.gid).has_value());

  std::vector<uint8_t> payload = {1, 2, 3};
  nvidia::gxf::MessageMetadata metadata{};
  EXPECT_FALSE(frontends.transport->send(endpoint.gid, payload, metadata).has_value());
}

// =============================================================================
// Participant lifecycle and cleanup tests
// =============================================================================

// After all frontends shut down, the session's backends should be reset.
// Verify by checking that a new frontend after shutdown gets fresh backends
// (a publisher announced before shutdown is no longer visible).
TEST_F(InMemoryPubSubSessionTest, BackendsResetAfterLastFrontendShutdown) {
  auto session = InMemoryPubSubSession::get_or_create("reset_test");
  auto frontends = session->create_frontends();

  ASSERT_TRUE(frontends.discovery->initialize().has_value());
  ASSERT_TRUE(frontends.transport->initialize().has_value());

  // Announce a publisher
  nvidia::gxf::PublisherInfo pub_info;
  pub_info.gid = nvidia::gxf::Gid::generate();
  pub_info.topic_name = "/reset/topic";
  pub_info.node_name = "node";
  ASSERT_TRUE(frontends.discovery->announce_publisher(pub_info).has_value());

  auto pubs = session->query_publishers("/reset/topic");
  ASSERT_TRUE(pubs.has_value());
  EXPECT_EQ(pubs->size(), 1UL);

  // Shut down both frontends (discovery calls release_participant)
  frontends.discovery->shutdown();
  frontends.transport->shutdown();

  // Release the shared_ptrs so participant weak_ptrs can expire
  frontends.discovery.reset();
  frontends.transport.reset();

  // Create new frontends — backends should have been reset, so the old
  // publisher should not be visible.
  auto frontends2 = session->create_frontends();
  ASSERT_TRUE(frontends2.discovery->initialize().has_value());

  auto pubs2 = session->query_publishers("/reset/topic");
  ASSERT_TRUE(pubs2.has_value());
  EXPECT_EQ(pubs2->size(), 0UL) << "Backends should have been reset after last participant left";

  frontends2.discovery->shutdown();
}

// After discovery shutdown, the participant should no longer receive
// discovery callbacks from other participants in the same session.
TEST_F(InMemoryPubSubSessionTest, ShutdownFrontendStopsReceivingCallbacks) {
  auto session = InMemoryPubSubSession::get_or_create("callback_cleanup");

  auto frontends_1 = session->create_frontends();
  auto frontends_2 = session->create_frontends();

  int discovered_count_1 = 0;
  int discovered_count_2 = 0;

  frontends_1.discovery->set_on_publisher_discovered(
      [&discovered_count_1](const nvidia::gxf::PublisherInfo&) { ++discovered_count_1; });
  frontends_2.discovery->set_on_publisher_discovered(
      [&discovered_count_2](const nvidia::gxf::PublisherInfo&) { ++discovered_count_2; });

  ASSERT_TRUE(frontends_1.discovery->initialize().has_value());
  ASSERT_TRUE(frontends_2.discovery->initialize().has_value());

  // Shut down frontend 1
  frontends_1.discovery->shutdown();

  // Announce a publisher via frontend 2 — should NOT notify shutdown frontend 1
  nvidia::gxf::PublisherInfo pub_info;
  pub_info.gid = nvidia::gxf::Gid::generate();
  pub_info.topic_name = "/callback/topic";
  pub_info.node_name = "node";
  ASSERT_TRUE(frontends_2.discovery->announce_publisher(pub_info).has_value());

  EXPECT_EQ(discovered_count_1, 0) << "Shutdown frontend should not receive discovery callbacks";
  EXPECT_EQ(discovered_count_2, 1) << "Active frontend should still receive discovery callbacks";

  frontends_2.discovery->shutdown();
}

TEST_F(InMemoryPubSubSessionTest, DiscoveryShutdownPreventsLateDispatch) {
  auto session = InMemoryPubSubSession::get_or_create("late_dispatch");
  auto frontends = session->create_frontends();

  int discovered_count = 0;
  frontends.discovery->set_on_publisher_discovered(
      [&discovered_count](const nvidia::gxf::PublisherInfo&) { ++discovered_count; });

  ASSERT_TRUE(frontends.discovery->initialize().has_value());

  auto discovery_frontend =
      std::dynamic_pointer_cast<SessionDiscoveryFrontend>(frontends.discovery);
  ASSERT_NE(discovery_frontend, nullptr);
  {
    std::lock_guard<std::mutex> lock(discovery_frontend->participant_->mutex);
    // Simulate the shutdown window where a copied callback may still exist
    // but the participant is no longer discovery-active.
    discovery_frontend->participant_->discovery_active = false;
  }

  session->notify_publisher_discovered(make_publisher_info("/late_dispatch/topic"));

  EXPECT_EQ(discovered_count, 0);
  ASSERT_TRUE(frontends.discovery->shutdown().has_value());
}

// After transport shutdown, the participant should no longer receive
// messages via the transport receive callback.
TEST_F(InMemoryPubSubSessionTest, ShutdownTransportStopsReceivingMessages) {
  auto session = InMemoryPubSubSession::get_or_create("transport_cleanup");

  auto frontends_1 = session->create_frontends();
  auto frontends_2 = session->create_frontends();

  ASSERT_TRUE(frontends_1.discovery->initialize().has_value());
  ASSERT_TRUE(frontends_1.transport->initialize().has_value());
  ASSERT_TRUE(frontends_2.discovery->initialize().has_value());
  ASSERT_TRUE(frontends_2.transport->initialize().has_value());

  // Set up receive callback on frontend 1
  int receive_count = 0;
  frontends_1.transport->set_on_receive(
      [&receive_count](const nvidia::gxf::Gid&,
                       std::vector<uint8_t>&&,
                       const nvidia::gxf::MessageMetadata&) { ++receive_count; });

  // Register subscriber on frontend 1
  nvidia::gxf::SubscriberInfo sub_info;
  sub_info.gid = nvidia::gxf::Gid::generate();
  sub_info.topic_name = "/transport_cleanup/topic";
  sub_info.node_name = "sub_node";
  ASSERT_TRUE(frontends_1.discovery->announce_subscriber(sub_info).has_value());

  // Send a message before shutdown — should be received
  std::vector<uint8_t> payload = {1, 2, 3};
  nvidia::gxf::MessageMetadata metadata{};
  ASSERT_TRUE(frontends_2.transport->send(sub_info.gid, payload, metadata).has_value());
  EXPECT_EQ(receive_count, 1);

  // Shut down transport on frontend 1
  frontends_1.transport->shutdown();

  // Send another message — should NOT be received (callback cleared)
  ASSERT_TRUE(frontends_2.transport->send(sub_info.gid, payload, metadata).has_value());
  EXPECT_EQ(receive_count, 1) << "Shutdown transport should not receive messages";

  frontends_1.discovery->shutdown();
  frontends_2.discovery->shutdown();
}

// With two frontends, shutting down one should not affect the other.
TEST_F(InMemoryPubSubSessionTest, PartialShutdownLeavesOtherFrontendFunctional) {
  auto session = InMemoryPubSubSession::get_or_create("partial_shutdown");

  auto frontends_1 = session->create_frontends();
  auto frontends_2 = session->create_frontends();

  ASSERT_TRUE(frontends_1.discovery->initialize().has_value());
  ASSERT_TRUE(frontends_2.discovery->initialize().has_value());

  // Shut down frontend 1 completely
  frontends_1.discovery->shutdown();
  frontends_1.transport->shutdown();

  // Frontend 2 should still be fully functional
  nvidia::gxf::PublisherInfo pub_info;
  pub_info.gid = nvidia::gxf::Gid::generate();
  pub_info.topic_name = "/partial/topic";
  pub_info.node_name = "node";
  EXPECT_TRUE(frontends_2.discovery->announce_publisher(pub_info).has_value());

  auto pubs = session->query_publishers("/partial/topic");
  ASSERT_TRUE(pubs.has_value());
  EXPECT_EQ(pubs->size(), 1UL);

  frontends_2.discovery->shutdown();
}

TEST_F(InMemoryPubSubSessionTest, DiscoveryShutdownDoesNotReleaseParticipantWhileTransportActive) {
  auto session = InMemoryPubSubSession::get_or_create("coordinated_shutdown");
  auto frontends = session->create_frontends();

  ASSERT_TRUE(frontends.discovery->initialize().has_value());
  ASSERT_TRUE(frontends.transport->initialize().has_value());

  nvidia::gxf::SubscriberInfo sub_info;
  sub_info.gid = nvidia::gxf::Gid::generate();
  sub_info.topic_name = "/coordinated_shutdown/topic";
  sub_info.node_name = "sub_node";
  ASSERT_TRUE(frontends.discovery->announce_subscriber(sub_info).has_value());

  auto subscribers_before_shutdown = session->query_subscribers("/coordinated_shutdown/topic");
  ASSERT_TRUE(subscribers_before_shutdown.has_value());
  EXPECT_EQ(subscribers_before_shutdown->size(), 1UL);

  ASSERT_TRUE(frontends.discovery->shutdown().has_value());

  auto subscribers_after_discovery_shutdown =
      session->query_subscribers("/coordinated_shutdown/topic");
  ASSERT_TRUE(subscribers_after_discovery_shutdown.has_value());
  EXPECT_EQ(subscribers_after_discovery_shutdown->size(), 1UL);

  ASSERT_TRUE(frontends.transport->shutdown().has_value());
  EXPECT_FALSE(session->query_subscribers("/coordinated_shutdown/topic").has_value());
}

TEST_F(InMemoryPubSubSessionTest, DiscoveryFrontendRejectsOperationsAfterShutdown) {
  auto session = InMemoryPubSubSession::get_or_create("discovery_post_shutdown_ops");
  auto frontends = session->create_frontends();

  ASSERT_TRUE(frontends.discovery->initialize().has_value());
  ASSERT_TRUE(frontends.transport->initialize().has_value());

  const auto publisher = make_publisher_info("/post_shutdown/publisher");
  const auto subscriber = make_subscriber_info("/post_shutdown/subscriber");
  ASSERT_TRUE(frontends.discovery->announce_publisher(publisher).has_value());
  ASSERT_TRUE(frontends.discovery->announce_subscriber(subscriber).has_value());

  ASSERT_TRUE(frontends.discovery->shutdown().has_value());

  expect_gxf_error(
      frontends.discovery->announce_publisher(make_publisher_info("/post_shutdown/new_publisher")),
      GXF_RESOURCE_NOT_INITIALIZED);
  expect_gxf_error(frontends.discovery->announce_subscriber(
                       make_subscriber_info("/post_shutdown/new_subscriber")),
                   GXF_RESOURCE_NOT_INITIALIZED);
  expect_gxf_error(frontends.discovery->remove_publisher(publisher.gid),
                   GXF_RESOURCE_NOT_INITIALIZED);
  expect_gxf_error(frontends.discovery->remove_subscriber(subscriber.gid),
                   GXF_RESOURCE_NOT_INITIALIZED);
  expect_gxf_error(frontends.discovery->query_publishers("/post_shutdown/publisher"),
                   GXF_RESOURCE_NOT_INITIALIZED);
  expect_gxf_error(frontends.discovery->query_subscribers("/post_shutdown/subscriber"),
                   GXF_RESOURCE_NOT_INITIALIZED);
  expect_gxf_error(frontends.discovery->get_all_topics(), GXF_RESOURCE_NOT_INITIALIZED);

  ASSERT_TRUE(frontends.transport->shutdown().has_value());
}

TEST_F(InMemoryPubSubSessionTest, ReleaseParticipantRejectsLateAnnouncements) {
  auto session = InMemoryPubSubSession::get_or_create("late_announcement_rejection");
  auto frontends_1 = session->create_frontends();
  auto frontends_2 = session->create_frontends();

  ASSERT_TRUE(frontends_1.discovery->initialize().has_value());
  ASSERT_TRUE(frontends_2.discovery->initialize().has_value());

  auto discovery_frontend =
      std::dynamic_pointer_cast<SessionDiscoveryFrontend>(frontends_1.discovery);
  ASSERT_NE(discovery_frontend, nullptr);
  auto participant = discovery_frontend->participant_;
  ASSERT_NE(participant, nullptr);

  ASSERT_TRUE(session->release_participant(participant).has_value());

  const auto late_publisher = make_publisher_info("/late_announcement/topic");
  auto late_result = session->announce_publisher(participant, late_publisher);
  EXPECT_FALSE(late_result.has_value());

  auto publishers = session->query_publishers(late_publisher.topic_name);
  ASSERT_TRUE(publishers.has_value());
  EXPECT_TRUE(publishers->empty());

  ASSERT_TRUE(frontends_1.discovery->shutdown().has_value());
  ASSERT_TRUE(frontends_2.discovery->shutdown().has_value());
}

TEST_F(InMemoryPubSubSessionTest, NetworkContextDestructorClearsSerializerDelegate) {
  auto serializer = std::make_shared<nvidia::gxf::InMemorySerializer>(
      nvidia::gxf::SerializerMode::kFullSerialization);

  {
    auto network_context = std::make_unique<InMemoryPubSubNetworkContext>();
    network_context->serializer_ = serializer;
    network_context->delegate_serializer_ =
        std::make_shared<StdPubSubEntitySerializer>(nullptr, nullptr, nullptr, 0);
    serializer->set_delegate(network_context->delegate_serializer_.get());
    ASSERT_NE(serializer->delegate_, nullptr);
  }

  EXPECT_EQ(serializer->delegate_, nullptr);
}

TEST_F(InMemoryPubSubSessionTest, SessionModeInitContextInitializesInjectedFrontends) {
  auto context = std::make_unique<nvidia::gxf::PubSubContext>();
  auto session = InMemoryPubSubSession::get_or_create("session_init_context");
  auto frontends = session->create_frontends();
  auto discovery_frontend =
      std::dynamic_pointer_cast<SessionDiscoveryFrontend>(frontends.discovery);
  auto transport_frontend =
      std::dynamic_pointer_cast<SessionTransportFrontend>(frontends.transport);
  ASSERT_NE(discovery_frontend, nullptr);
  ASSERT_NE(transport_frontend, nullptr);
  EXPECT_FALSE(discovery_frontend->is_initialized());
  EXPECT_FALSE(transport_frontend->is_initialized());

  context->set_discovery(frontends.discovery);
  context->set_transport(frontends.transport);
  context->set_serializer(std::make_shared<nvidia::gxf::InMemorySerializer>());

  ASSERT_EQ(context->initialize(), GXF_SUCCESS);
  EXPECT_TRUE(discovery_frontend->is_initialized());
  EXPECT_TRUE(transport_frontend->is_initialized());
  ASSERT_EQ(context->init_context(), GXF_SUCCESS);
  EXPECT_TRUE(discovery_frontend->is_initialized());
  EXPECT_TRUE(transport_frontend->is_initialized());

  ASSERT_EQ(context->deinitialize(), GXF_SUCCESS);
  ASSERT_TRUE(frontends.discovery->shutdown().has_value());
  ASSERT_TRUE(frontends.transport->shutdown().has_value());
}

}  // namespace holoscan
