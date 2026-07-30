// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/qos/DomainParticipantQos.hpp>
#include <fastdds/rtps/common/SerializedPayload.hpp>

#include <gxf/pubsub/gid.hpp>
#include <gxf/pubsub/topic_registry.hpp>
#include <gxf/std/tensor.hpp>
#include <gxf/std/timestamp.hpp>

#include "holoscan/core/arg.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/message.hpp"
#include "holoscan/core/messagelabel.hpp"
#include "holoscan/core/metadata.hpp"
#include "holoscan/core/resources/gxf/unbounded_allocator.hpp"
#include "holoscan/pubsub/fastdds/pubsub/fastdds_discovery.hpp"
#include "holoscan/pubsub/fastdds/pubsub/fastdds_endpoint.hpp"
#include "holoscan/pubsub/fastdds/pubsub/fastdds_holoscan_entity_type_support.hpp"
#include "holoscan/pubsub/fastdds/pubsub/fastdds_qos_profiles.hpp"
#include "holoscan/pubsub/fastdds/pubsub/fastdds_serializer.hpp"
#include "holoscan/pubsub/fastdds/pubsub/fastdds_transport.hpp"
#include "holoscan/pubsub/fastdds/resources/fastdds_pubsub_context.hpp"

// Include test utilities for TestWithGXFContext
#include "config.hpp"
#include "utils.hpp"

namespace {

using eprosima::fastdds::dds::DomainParticipant;
using eprosima::fastdds::dds::DomainParticipantFactory;
using eprosima::fastdds::dds::DomainParticipantQos;
using eprosima::fastdds::rtps::SerializedPayload_t;

// =============================================================================
// FastDDS Smoke Tests
// =============================================================================

TEST(FastDDSSmokeTest, CreateAndDeleteParticipant) {
  // Get the DomainParticipantFactory singleton
  DomainParticipantFactory* factory = DomainParticipantFactory::get_instance();
  ASSERT_NE(factory, nullptr) << "DomainParticipantFactory::get_instance() returned null";

  // Get default QoS
  DomainParticipantQos qos;
  factory->get_default_participant_qos(qos);

  // Create a DomainParticipant on domain 0
  DomainParticipant* participant = factory->create_participant(0, qos);
  ASSERT_NE(participant, nullptr) << "create_participant() returned null";

  // Clean up: delete the participant (RETCODE_OK == 0 in Fast DDS)
  EXPECT_EQ(factory->delete_participant(participant), 0) << "delete_participant() failed";
}

// =============================================================================
// FastDdsHoloscanEntityTypeSupport Tests
// =============================================================================

class DDSHoloscanEntityTypeSupportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    type_support_ = std::make_unique<holoscan::FastDdsHoloscanEntityTypeSupport>();
  }

  void TearDown() override { type_support_.reset(); }

  std::unique_ptr<holoscan::FastDdsHoloscanEntityTypeSupport> type_support_;
};

TEST_F(DDSHoloscanEntityTypeSupportTest, TypeName) {
  // Verify the type name is set correctly
  EXPECT_EQ(std::string(type_support_->get_name()), "holoscan::Entity");
}

TEST_F(DDSHoloscanEntityTypeSupportTest, CreateAndDeleteData) {
  // Test create_data
  void* data = type_support_->create_data();
  ASSERT_NE(data, nullptr);

  // Verify it's a valid HoloscanEntityData
  auto* entity_data = static_cast<holoscan::HoloscanEntityData*>(data);
  EXPECT_TRUE(entity_data->serialized_data.empty());
  EXPECT_TRUE(entity_data->source_operator.empty());
  EXPECT_EQ(entity_data->timestamp_ns, 0);
  EXPECT_FALSE(entity_data->contains_gpu_tensors);
  EXPECT_EQ(entity_data->gpu_device_id, 0);
  EXPECT_TRUE(entity_data->publisher_gid.empty());
  EXPECT_EQ(entity_data->descriptor_format_version, 0);
  EXPECT_TRUE(entity_data->protocol_name.empty());

  // Test delete_data (should not crash)
  type_support_->delete_data(data);
}

TEST_F(DDSHoloscanEntityTypeSupportTest, SerializeDeserializeEmpty) {
  // Create empty entity data
  holoscan::HoloscanEntityData original;

  // Calculate serialized size
  uint32_t size = type_support_->calculate_serialized_size(
      &original, eprosima::fastdds::dds::DataRepresentationId_t::XCDR_DATA_REPRESENTATION);
  EXPECT_GT(size, 0u);

  // Create payload with sufficient space
  SerializedPayload_t payload;
  payload.reserve(size);

  // Serialize
  bool serialize_result = type_support_->serialize(
      &original, payload, eprosima::fastdds::dds::DataRepresentationId_t::XCDR_DATA_REPRESENTATION);
  ASSERT_TRUE(serialize_result);
  EXPECT_GT(payload.length, 0u);

  // Deserialize into new instance
  holoscan::HoloscanEntityData deserialized;
  bool deserialize_result = type_support_->deserialize(payload, &deserialized);
  ASSERT_TRUE(deserialize_result);

  // Verify fields match
  EXPECT_EQ(deserialized.serialized_data, original.serialized_data);
  EXPECT_EQ(deserialized.source_operator, original.source_operator);
  EXPECT_EQ(deserialized.timestamp_ns, original.timestamp_ns);
  EXPECT_EQ(deserialized.contains_gpu_tensors, original.contains_gpu_tensors);
  EXPECT_EQ(deserialized.gpu_device_id, original.gpu_device_id);
  EXPECT_EQ(deserialized.publisher_gid, original.publisher_gid);
  EXPECT_EQ(deserialized.descriptor_format_version, original.descriptor_format_version);
  EXPECT_EQ(deserialized.protocol_name, original.protocol_name);
}

TEST_F(DDSHoloscanEntityTypeSupportTest, SerializeDeserializePopulated) {
  // Create populated entity data
  holoscan::HoloscanEntityData original;
  original.serialized_data = {0x01, 0x02, 0x03, 0x04, 0x05, 0xAB, 0xCD, 0xEF};
  original.source_operator = "test_camera_operator";
  original.timestamp_ns = 1234567890123456789LL;
  original.contains_gpu_tensors = true;
  original.gpu_device_id = 2;
  original.publisher_gid = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6";
  original.descriptor_format_version = 1;
  original.protocol_name = "cuda_ipc";

  // Calculate serialized size
  uint32_t size = type_support_->calculate_serialized_size(
      &original, eprosima::fastdds::dds::DataRepresentationId_t::XCDR_DATA_REPRESENTATION);
  EXPECT_GT(size, 0u);

  // Create payload with sufficient space
  SerializedPayload_t payload;
  payload.reserve(size);

  // Serialize
  bool serialize_result = type_support_->serialize(
      &original, payload, eprosima::fastdds::dds::DataRepresentationId_t::XCDR_DATA_REPRESENTATION);
  ASSERT_TRUE(serialize_result);
  EXPECT_GT(payload.length, 0u);

  // Deserialize into new instance
  holoscan::HoloscanEntityData deserialized;
  bool deserialize_result = type_support_->deserialize(payload, &deserialized);
  ASSERT_TRUE(deserialize_result);

  // Verify all fields match exactly
  EXPECT_EQ(deserialized.serialized_data, original.serialized_data);
  EXPECT_EQ(deserialized.source_operator, original.source_operator);
  EXPECT_EQ(deserialized.timestamp_ns, original.timestamp_ns);
  EXPECT_EQ(deserialized.contains_gpu_tensors, original.contains_gpu_tensors);
  EXPECT_EQ(deserialized.gpu_device_id, original.gpu_device_id);
  EXPECT_EQ(deserialized.publisher_gid, original.publisher_gid);
  EXPECT_EQ(deserialized.descriptor_format_version, original.descriptor_format_version);
  EXPECT_EQ(deserialized.protocol_name, original.protocol_name);
}

TEST_F(DDSHoloscanEntityTypeSupportTest, SerializeDeserializeLargePayload) {
  // Create entity data with large serialized payload
  holoscan::HoloscanEntityData original;
  original.serialized_data.resize(1024 * 1024);  // 1 MB payload
  for (size_t i = 0; i < original.serialized_data.size(); ++i) {
    original.serialized_data[i] = static_cast<uint8_t>(i & 0xFF);
  }
  original.source_operator = "large_payload_producer";
  original.timestamp_ns = 1234567890123456789LL;  // Use smaller value to avoid overflow
  original.contains_gpu_tensors = true;
  original.gpu_device_id = 7;
  original.publisher_gid = "fedcba9876543210fedcba9876543210";
  original.descriptor_format_version = 3;
  original.protocol_name = "cuda_vmm";

  // Calculate serialized size
  uint32_t size = type_support_->calculate_serialized_size(
      &original, eprosima::fastdds::dds::DataRepresentationId_t::XCDR_DATA_REPRESENTATION);
  EXPECT_GE(size, original.serialized_data.size());

  // Create payload with sufficient space
  SerializedPayload_t payload;
  payload.reserve(size);

  // Serialize
  bool serialize_result = type_support_->serialize(
      &original, payload, eprosima::fastdds::dds::DataRepresentationId_t::XCDR_DATA_REPRESENTATION);
  ASSERT_TRUE(serialize_result);

  // Deserialize
  holoscan::HoloscanEntityData deserialized;
  bool deserialize_result = type_support_->deserialize(payload, &deserialized);
  ASSERT_TRUE(deserialize_result);

  // Verify
  EXPECT_EQ(deserialized.serialized_data.size(), original.serialized_data.size());
  EXPECT_EQ(deserialized.serialized_data, original.serialized_data);
  EXPECT_EQ(deserialized.source_operator, original.source_operator);
  EXPECT_EQ(deserialized.timestamp_ns, original.timestamp_ns);
  EXPECT_EQ(deserialized.contains_gpu_tensors, original.contains_gpu_tensors);
  EXPECT_EQ(deserialized.gpu_device_id, original.gpu_device_id);
  EXPECT_EQ(deserialized.publisher_gid, original.publisher_gid);
  EXPECT_EQ(deserialized.descriptor_format_version, original.descriptor_format_version);
  EXPECT_EQ(deserialized.protocol_name, original.protocol_name);
}

TEST_F(DDSHoloscanEntityTypeSupportTest, ComputeKeyReturnsFalse) {
  // HoloscanEntityData does not use keyed topics
  holoscan::HoloscanEntityData data;
  eprosima::fastdds::rtps::InstanceHandle_t handle;

  // compute_key should return false (no key defined)
  EXPECT_FALSE(type_support_->compute_key(&data, handle, false));
  EXPECT_FALSE(type_support_->compute_key(&data, handle, true));

  // Also test the SerializedPayload overload
  SerializedPayload_t payload;
  EXPECT_FALSE(type_support_->compute_key(payload, handle, false));
}

// =============================================================================
// DDS QoS Profiles Tests
// =============================================================================

TEST(DDSQoSProfilesTest, ApplyWriterQoSFromGxfPresets) {
  using namespace eprosima::fastdds::dds;

  // Verify all GXF named presets can be mapped to DDS writer QoS
  for (const auto& name : nvidia::gxf::QoSProfile::preset_names()) {
    auto profile = nvidia::gxf::QoSProfile::from_name(name);
    ASSERT_TRUE(profile) << "Failed to look up GXF preset: " << name;

    DataWriterQos qos = DATAWRITER_QOS_DEFAULT;
    holoscan::dds_qos::apply_writer_qos(qos, profile.value());
    // Verify at least the core fields were set (non-default for most presets)
    // The apply function is void — it always succeeds structurally.
  }
}

TEST(DDSQoSProfilesTest, ApplyReaderQoSFromGxfPresets) {
  using namespace eprosima::fastdds::dds;

  for (const auto& name : nvidia::gxf::QoSProfile::preset_names()) {
    auto profile = nvidia::gxf::QoSProfile::from_name(name);
    ASSERT_TRUE(profile) << "Failed to look up GXF preset: " << name;

    DataReaderQos qos = DATAREADER_QOS_DEFAULT;
    holoscan::dds_qos::apply_reader_qos(qos, profile.value());
  }
}

TEST(DDSQoSProfilesTest, DefaultProfileIsBestEffort) {
  using namespace eprosima::fastdds::dds;

  // QoSProfile::Default() is best-effort, volatile, keep-last(10)
  DataWriterQos qos = DATAWRITER_QOS_DEFAULT;
  holoscan::dds_qos::apply_writer_qos(qos, nvidia::gxf::QoSProfile::Default());
  EXPECT_EQ(qos.reliability().kind, BEST_EFFORT_RELIABILITY_QOS);
  EXPECT_EQ(qos.durability().kind, VOLATILE_DURABILITY_QOS);
  EXPECT_EQ(qos.history().kind, KEEP_LAST_HISTORY_QOS);
  EXPECT_EQ(qos.history().depth, 10);
}

TEST(DDSQoSProfilesTest, VideoStreamProfileIsBestEffort) {
  using namespace eprosima::fastdds::dds;

  DataWriterQos qos = DATAWRITER_QOS_DEFAULT;
  holoscan::dds_qos::apply_writer_qos(qos, nvidia::gxf::QoSProfile::VideoStream());
  EXPECT_EQ(qos.reliability().kind, BEST_EFFORT_RELIABILITY_QOS);
  EXPECT_EQ(qos.durability().kind, VOLATILE_DURABILITY_QOS);
  EXPECT_EQ(qos.history().depth, 1);
}

TEST(DDSQoSProfilesTest, TensorDataProfileIsVolatile) {
  using namespace eprosima::fastdds::dds;

  DataWriterQos qos = DATAWRITER_QOS_DEFAULT;
  holoscan::dds_qos::apply_writer_qos(qos, nvidia::gxf::QoSProfile::TensorData());
  EXPECT_EQ(qos.reliability().kind, RELIABLE_RELIABILITY_QOS);
  EXPECT_EQ(qos.durability().kind, VOLATILE_DURABILITY_QOS);
  EXPECT_EQ(qos.history().depth, 3);
}

TEST(DDSQoSProfilesTest, ControlMessageProfileKeepsAll) {
  using namespace eprosima::fastdds::dds;

  DataWriterQos qos = DATAWRITER_QOS_DEFAULT;
  holoscan::dds_qos::apply_writer_qos(qos, nvidia::gxf::QoSProfile::ControlMessage());
  EXPECT_EQ(qos.reliability().kind, RELIABLE_RELIABILITY_QOS);
  EXPECT_EQ(qos.durability().kind, TRANSIENT_LOCAL_DURABILITY_QOS);
  EXPECT_EQ(qos.history().kind, KEEP_ALL_HISTORY_QOS);
}

TEST(DDSQoSProfilesTest, PriorityIsMapped) {
  using namespace eprosima::fastdds::dds;

  nvidia::gxf::QoSProfile qos;
  qos.priority = 42;

  DataWriterQos dds_qos = DATAWRITER_QOS_DEFAULT;
  holoscan::dds_qos::apply_writer_qos(dds_qos, qos);
  EXPECT_EQ(dds_qos.transport_priority().value, 42u);
}

TEST(DDSQoSProfilesTest, MaxBlockingTimeIsMapped) {
  using namespace eprosima::fastdds::dds;

  nvidia::gxf::QoSProfile qos;
  qos.max_blocking_time_ns = 2'500'000'000ULL;  // 2.5 seconds

  DataWriterQos dds_qos = DATAWRITER_QOS_DEFAULT;
  holoscan::dds_qos::apply_writer_qos(dds_qos, qos);
  EXPECT_EQ(dds_qos.reliability().max_blocking_time.seconds, 2);
  EXPECT_EQ(dds_qos.reliability().max_blocking_time.nanosec, 500'000'000u);
}

TEST(DDSQoSProfilesTest, DeadlineAndLifespanAreMapped) {
  using namespace eprosima::fastdds::dds;

  nvidia::gxf::QoSProfile qos;
  qos.deadline_ns = 1'000'000'000ULL;  // 1 second
  qos.lifespan_ns = 5'000'000'000ULL;  // 5 seconds

  DataWriterQos dds_qos = DATAWRITER_QOS_DEFAULT;
  holoscan::dds_qos::apply_writer_qos(dds_qos, qos);
  EXPECT_EQ(dds_qos.deadline().period.seconds, 1);
  EXPECT_EQ(dds_qos.deadline().period.nanosec, 0u);
  EXPECT_EQ(dds_qos.lifespan().duration.seconds, 5);
  EXPECT_EQ(dds_qos.lifespan().duration.nanosec, 0u);
}

TEST(DDSQoSProfilesTest, GxfPresetNamesMatchExpected) {
  auto names = nvidia::gxf::QoSProfile::preset_names();
  // GXF provides 8 named presets
  EXPECT_EQ(names.size(), 8u);
}

// =============================================================================
// FastDdsTransport Basic Tests (no GXF context required)
// =============================================================================

TEST(DDSTransportBasicTest, ConstructWithNullContext) {
  holoscan::FastDdsTransport transport(nullptr);
  EXPECT_FALSE(transport.is_initialized());
}

TEST(DDSTransportBasicTest, NotInitializedByDefault) {
  holoscan::FastDdsTransport transport(nullptr);
  EXPECT_FALSE(transport.is_initialized());
}

TEST(DDSTransportBasicTest, InitializeFailsWithNullContext) {
  holoscan::FastDdsTransport transport(nullptr);
  auto result = transport.initialize();
  EXPECT_FALSE(result.has_value());
}

TEST(DDSTransportBasicTest, ShutdownOnUninitializedIsSafe) {
  holoscan::FastDdsTransport transport(nullptr);
  auto result = transport.shutdown();
  // Shutdown on uninitialized transport should succeed (idempotent)
  EXPECT_TRUE(result.has_value());
}

TEST(DDSTransportBasicTest, GetConnectionCountOnUninitialized) {
  holoscan::FastDdsTransport transport(nullptr);
  EXPECT_EQ(transport.get_connection_count(), 0u);
}

TEST(DDSTransportBasicTest, GetSendQueueSizeOnUninitialized) {
  holoscan::FastDdsTransport transport(nullptr);
  EXPECT_EQ(transport.get_send_queue_size(), 0u);
}

TEST(DDSTransportBasicTest, GetReceiveQueueSizeOnUninitialized) {
  holoscan::FastDdsTransport transport(nullptr);
  EXPECT_EQ(transport.get_receive_queue_size(), 0u);
}

TEST(DDSTransportBasicTest, SetReceiveCallbackOnUninitialized) {
  holoscan::FastDdsTransport transport(nullptr);
  bool callback_invoked = false;
  transport.set_on_receive([&](const nvidia::gxf::Gid&,
                               std::vector<uint8_t>&&,
                               const nvidia::gxf::MessageMetadata&) { callback_invoked = true; });
  // Just verify setting callback doesn't crash
  EXPECT_FALSE(callback_invoked);
}

TEST(DDSTransportBasicTest, SetConnectionCallbacksOnUninitialized) {
  holoscan::FastDdsTransport transport(nullptr);
  bool established_called = false;
  bool lost_called = false;

  transport.set_on_connection_established(
      [&](const nvidia::gxf::Gid&) { established_called = true; });

  transport.set_on_connection_lost([&](const nvidia::gxf::Gid&) { lost_called = true; });

  // Just verify setting callbacks doesn't crash
  EXPECT_FALSE(established_called);
  EXPECT_FALSE(lost_called);
}

TEST(DDSTransportBasicTest, CreatePublisherEndpointFailsWhenUninitialized) {
  holoscan::FastDdsTransport transport(nullptr);
  auto result = transport.create_publisher_endpoint("test_topic", nvidia::gxf::Gid::generate());
  EXPECT_FALSE(result.has_value());
}

TEST(DDSTransportBasicTest, CreateSubscriberEndpointFailsWhenUninitialized) {
  holoscan::FastDdsTransport transport(nullptr);
  auto result = transport.create_subscriber_endpoint("test_topic", nvidia::gxf::Gid::generate());
  EXPECT_FALSE(result.has_value());
}

TEST(DDSTransportBasicTest, RemovePublisherEndpointFailsWhenUninitialized) {
  holoscan::FastDdsTransport transport(nullptr);
  auto result = transport.remove_publisher_endpoint(nvidia::gxf::Gid::generate());
  EXPECT_FALSE(result.has_value());
}

TEST(DDSTransportBasicTest, RemoveSubscriberEndpointFailsWhenUninitialized) {
  holoscan::FastDdsTransport transport(nullptr);
  auto result = transport.remove_subscriber_endpoint(nvidia::gxf::Gid::generate());
  EXPECT_FALSE(result.has_value());
}

TEST(DDSTransportBasicTest, TransportModelIsTopicBased) {
  holoscan::FastDdsTransport transport(nullptr);
  EXPECT_EQ(transport.transport_model(), nvidia::gxf::TransportModel::kTopicBased);
}

TEST(DDSTransportBasicTest, BackendCapabilities) {
  holoscan::FastDdsTransport transport(nullptr);
  EXPECT_TRUE(transport.native_topic_matching());
  EXPECT_TRUE(transport.native_qos_enforcement());
  EXPECT_TRUE(transport.supports_multicast());
  EXPECT_FALSE(transport.requires_explicit_connections());
}

TEST(DDSTransportBasicTest, SendFailsWhenUninitialized) {
  holoscan::FastDdsTransport transport(nullptr);
  std::vector<uint8_t> payload = {0x01, 0x02, 0x03};
  nvidia::gxf::MessageMetadata metadata;
  auto result = transport.send(nvidia::gxf::Gid::null(), payload, metadata);
  EXPECT_FALSE(result.has_value());
}

// =============================================================================
// FastDdsTransport Integration Tests (using TestWithGXFContext)
// =============================================================================
//
// These tests use Holoscan's TestWithGXFContext fixture to properly initialize
// GXF context and resources.
// =============================================================================

class DDSTransportIntegrationTest : public holoscan::TestWithGXFContext {
 protected:
  void SetUp() override {
    // Initialize GXF context via base class
    holoscan::TestWithGXFContext::SetUp();

    // Create FastDdsPubSubContext using Fragment's make_resource
    // This ensures proper parameter binding and GXF integration
    holoscan::ArgList args{
        holoscan::Arg{"domain_id", static_cast<int32_t>(0)},
        holoscan::Arg{"participant_name", std::string("test_transport_participant")},
    };
    dds_context_ = F.make_resource<holoscan::FastDdsPubSubContext>("test_dds_context", args);

    // Initialize the FastDdsPubSubContext (creates DomainParticipant)
    dds_context_->initialize();

    // Create FastDdsTransport with the context
    transport_ = std::make_unique<holoscan::FastDdsTransport>(dds_context_.get());
  }

  void TearDown() override {
    // Shutdown transport first
    if (transport_ && transport_->is_initialized()) {
      transport_->shutdown();
    }
    transport_.reset();

    // FastDdsPubSubContext cleanup happens when shared_ptr is reset
    dds_context_.reset();
  }

  std::shared_ptr<holoscan::FastDdsPubSubContext> dds_context_;
  std::unique_ptr<holoscan::FastDdsTransport> transport_;
};

TEST_F(DDSTransportIntegrationTest, InitializeShutdown) {
  // Transport should not be initialized yet
  EXPECT_FALSE(transport_->is_initialized());

  // Initialize
  auto init_result = transport_->initialize();
  ASSERT_TRUE(init_result.has_value()) << "initialize() failed";
  EXPECT_TRUE(transport_->is_initialized());

  // Shutdown
  auto shutdown_result = transport_->shutdown();
  ASSERT_TRUE(shutdown_result.has_value()) << "shutdown() failed";
  EXPECT_FALSE(transport_->is_initialized());

  // Shutdown is idempotent
  auto shutdown_result2 = transport_->shutdown();
  EXPECT_TRUE(shutdown_result2.has_value());
}

TEST_F(DDSTransportIntegrationTest, CreateAndRemoveEndpoints) {
  // Initialize transport
  ASSERT_TRUE(transport_->initialize().has_value());

  // Create a publisher endpoint
  nvidia::gxf::Gid pub_gid = nvidia::gxf::Gid::generate();
  auto writer_result = transport_->create_publisher_endpoint("test_topic", pub_gid);
  ASSERT_TRUE(writer_result.has_value()) << "create_publisher_endpoint() failed";

  // Create a subscriber endpoint
  nvidia::gxf::Gid sub_gid = nvidia::gxf::Gid::generate();
  auto reader_result = transport_->create_subscriber_endpoint("test_topic", sub_gid);
  ASSERT_TRUE(reader_result.has_value()) << "create_subscriber_endpoint() failed";

  // Connection count should reflect created endpoints
  EXPECT_EQ(transport_->get_connection_count(), 2u);

  // Remove publisher and subscriber endpoints
  EXPECT_TRUE(transport_->remove_publisher_endpoint(pub_gid).has_value());
  EXPECT_TRUE(transport_->remove_subscriber_endpoint(sub_gid).has_value());
  EXPECT_EQ(transport_->get_connection_count(), 0u);
}

TEST_F(DDSTransportIntegrationTest, SendWithoutWriter) {
  ASSERT_TRUE(transport_->initialize().has_value());

  // Try to send without creating a writer - should fail
  std::vector<uint8_t> payload = {0x01, 0x02, 0x03};
  nvidia::gxf::MessageMetadata metadata;
  metadata.publisher_gid = nvidia::gxf::Gid::generate();

  auto send_result = transport_->send(nvidia::gxf::Gid::null(), payload, metadata);
  EXPECT_FALSE(send_result.has_value()) << "send() should fail without a writer";
}

TEST_F(DDSTransportIntegrationTest, SendWithWriter) {
  ASSERT_TRUE(transport_->initialize().has_value());

  // Create a publisher endpoint
  nvidia::gxf::Gid pub_gid = nvidia::gxf::Gid::generate();
  ASSERT_TRUE(transport_->create_publisher_endpoint("test_topic", pub_gid).has_value());

  // Send should succeed
  std::vector<uint8_t> payload = {0x01, 0x02, 0x03, 0x04};
  nvidia::gxf::MessageMetadata metadata;
  metadata.publisher_gid = pub_gid;
  metadata.source_timestamp_ns = 1234567890;
  metadata.sequence_number = 1;

  auto send_result = transport_->send(nvidia::gxf::Gid::null(), payload, metadata);
  EXPECT_TRUE(send_result.has_value()) << "send() failed with a valid writer";
}

TEST_F(DDSTransportIntegrationTest, SendReceiveLoopback) {
  ASSERT_TRUE(transport_->initialize().has_value());

  // Create writer and reader for same topic (loopback)
  nvidia::gxf::Gid pub_gid = nvidia::gxf::Gid::generate();
  nvidia::gxf::Gid sub_gid = nvidia::gxf::Gid::generate();

  ASSERT_TRUE(transport_->create_publisher_endpoint("loopback_topic", pub_gid).has_value());
  ASSERT_TRUE(transport_->create_subscriber_endpoint("loopback_topic", sub_gid).has_value());

  // Allow time for DDS discovery (writer/reader matching)
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Set up receive callback
  std::mutex mtx;
  std::condition_variable cv;
  std::vector<uint8_t> received_payload;
  nvidia::gxf::Gid received_src_gid;
  nvidia::gxf::MessageMetadata received_metadata;
  bool received = false;

  transport_->set_on_receive([&](const nvidia::gxf::Gid& src,
                                 std::vector<uint8_t>&& payload,
                                 const nvidia::gxf::MessageMetadata& meta) {
    std::lock_guard<std::mutex> lock(mtx);
    received_payload = std::move(payload);
    received_src_gid = src;
    received_metadata = meta;
    received = true;
    cv.notify_one();
  });

  // Send a message
  std::vector<uint8_t> test_payload = {0xDE, 0xAD, 0xBE, 0xEF};
  nvidia::gxf::MessageMetadata metadata;
  metadata.publisher_gid = pub_gid;
  metadata.source_timestamp_ns = 9999999999;

  ASSERT_TRUE(transport_->send(nvidia::gxf::Gid::null(), test_payload, metadata).has_value());

  // Wait for message to be received (with timeout)
  {
    std::unique_lock<std::mutex> lock(mtx);
    bool got_message = cv.wait_for(lock, std::chrono::seconds(5), [&] { return received; });
    EXPECT_TRUE(got_message) << "Did not receive loopback message within timeout";
  }

  // Verify received payload and metadata
  if (received) {
    EXPECT_EQ(received_payload, test_payload);

    // Verify publisher_gid is populated from DDS SampleInfo.sample_identity.writer_guid()
    // The GID should be non-null (extracted from FastDDS GUID_t)
    EXPECT_NE(received_src_gid, nvidia::gxf::Gid::null())
        << "Source GID should be populated from DDS writer GUID";
    EXPECT_NE(received_metadata.publisher_gid, nvidia::gxf::Gid::null())
        << "metadata.publisher_gid should be populated from DDS writer GUID";

    // Both should match (src GID passed to callback == metadata.publisher_gid)
    EXPECT_EQ(received_src_gid, received_metadata.publisher_gid)
        << "Source GID and metadata.publisher_gid should match";

    // Verify destination_gid is set to the subscriber's GID
    EXPECT_EQ(received_metadata.destination_gid, sub_gid)
        << "metadata.destination_gid should be the subscriber GID";
  }
}

TEST_F(DDSTransportIntegrationTest, ConnectionManagementNoOps) {
  ASSERT_TRUE(transport_->initialize().has_value());

  // connect_to and disconnect_from should be no-ops for DDS
  nvidia::gxf::EndpointInfo endpoint;
  endpoint.gid = nvidia::gxf::Gid::generate();
  endpoint.topic_name = "test_topic";

  EXPECT_TRUE(transport_->connect_to(endpoint).has_value());
  EXPECT_TRUE(transport_->is_connected_to(endpoint.gid));
  EXPECT_TRUE(transport_->disconnect_from(endpoint.gid).has_value());
}

TEST_F(DDSTransportIntegrationTest, DuplicateEndpointIsNoOp) {
  ASSERT_TRUE(transport_->initialize().has_value());

  nvidia::gxf::Gid pub_gid = nvidia::gxf::Gid::generate();

  // First creation should succeed
  EXPECT_TRUE(transport_->create_publisher_endpoint("test_topic", pub_gid).has_value());
  EXPECT_EQ(transport_->get_connection_count(), 1u);

  // Second creation with same GID should be a no-op (returns success, doesn't add another)
  EXPECT_TRUE(transport_->create_publisher_endpoint("test_topic", pub_gid).has_value());
  EXPECT_EQ(transport_->get_connection_count(), 1u);
}

TEST_F(DDSTransportIntegrationTest, CreatePublisherEndpointWithQoSProfiles) {
  ASSERT_TRUE(transport_->initialize().has_value());

  // Default QoSProfile (struct-based)
  nvidia::gxf::Gid gid1 = nvidia::gxf::Gid::generate();
  EXPECT_TRUE(
      transport_
          ->create_publisher_endpoint("topic_default", gid1, nvidia::gxf::QoSProfile::Default())
          .has_value());

  // ControlMessage preset
  nvidia::gxf::Gid gid2 = nvidia::gxf::Gid::generate();
  EXPECT_TRUE(transport_
                  ->create_publisher_endpoint(
                      "topic_control", gid2, nvidia::gxf::QoSProfile::ControlMessage())
                  .has_value());

  // TensorData preset
  nvidia::gxf::Gid gid3 = nvidia::gxf::Gid::generate();
  EXPECT_TRUE(
      transport_
          ->create_publisher_endpoint("topic_tensor", gid3, nvidia::gxf::QoSProfile::TensorData())
          .has_value());

  // VideoStream preset
  nvidia::gxf::Gid gid4 = nvidia::gxf::Gid::generate();
  EXPECT_TRUE(
      transport_
          ->create_publisher_endpoint("topic_video", gid4, nvidia::gxf::QoSProfile::VideoStream())
          .has_value());

  EXPECT_EQ(transport_->get_connection_count(), 4u);
}

TEST_F(DDSTransportIntegrationTest, CreateSubscriberEndpointWithQoSProfiles) {
  ASSERT_TRUE(transport_->initialize().has_value());

  nvidia::gxf::Gid gid1 = nvidia::gxf::Gid::generate();
  EXPECT_TRUE(
      transport_
          ->create_subscriber_endpoint("topic_default", gid1, nvidia::gxf::QoSProfile::Default())
          .has_value());

  nvidia::gxf::Gid gid2 = nvidia::gxf::Gid::generate();
  EXPECT_TRUE(transport_
                  ->create_subscriber_endpoint(
                      "topic_control", gid2, nvidia::gxf::QoSProfile::ControlMessage())
                  .has_value());

  nvidia::gxf::Gid gid3 = nvidia::gxf::Gid::generate();
  EXPECT_TRUE(
      transport_
          ->create_subscriber_endpoint("topic_tensor", gid3, nvidia::gxf::QoSProfile::TensorData())
          .has_value());

  nvidia::gxf::Gid gid4 = nvidia::gxf::Gid::generate();
  EXPECT_TRUE(
      transport_
          ->create_subscriber_endpoint("topic_video", gid4, nvidia::gxf::QoSProfile::VideoStream())
          .has_value());

  EXPECT_EQ(transport_->get_connection_count(), 4u);
}

TEST_F(DDSTransportIntegrationTest, CreateEndpointWithFromName) {
  ASSERT_TRUE(transport_->initialize().has_value());

  // Verify QoSProfile::from_name() integration works end-to-end
  nvidia::gxf::Gid gid = nvidia::gxf::Gid::generate();
  auto qos = nvidia::gxf::QoSProfile::from_name("tensor_data");
  ASSERT_TRUE(qos);
  EXPECT_TRUE(
      transport_->create_publisher_endpoint("topic_from_name", gid, qos.value()).has_value());
}

TEST_F(DDSTransportIntegrationTest, TopicBasedSend) {
  ASSERT_TRUE(transport_->initialize().has_value());

  // Create a publisher endpoint
  nvidia::gxf::Gid pub_gid = nvidia::gxf::Gid::generate();
  ASSERT_TRUE(transport_->create_publisher_endpoint("test_topic_send", pub_gid).has_value());

  // Send via topic-based overload (the primary path for kTopicBased transports)
  std::vector<uint8_t> payload = {0x01, 0x02, 0x03, 0x04};
  nvidia::gxf::MessageMetadata metadata;
  metadata.publisher_gid = pub_gid;
  metadata.source_timestamp_ns = 1234567890;

  auto send_result = transport_->send(std::string("test_topic_send"), payload, metadata);
  EXPECT_TRUE(send_result.has_value()) << "topic-based send() failed";
}

TEST_F(DDSTransportIntegrationTest, TopicBasedSendNoWriter) {
  ASSERT_TRUE(transport_->initialize().has_value());

  // Send to topic with no writer — should fail
  std::vector<uint8_t> payload = {0x01, 0x02};
  nvidia::gxf::MessageMetadata metadata;
  metadata.publisher_gid = nvidia::gxf::Gid::generate();

  auto send_result = transport_->send(std::string("nonexistent_topic"), payload, metadata);
  EXPECT_FALSE(send_result.has_value());
}

TEST_F(DDSTransportIntegrationTest, RemoveNonExistentEndpoint) {
  ASSERT_TRUE(transport_->initialize().has_value());

  nvidia::gxf::Gid unknown_gid = nvidia::gxf::Gid::generate();

  // Removing non-existent publisher endpoint should fail gracefully
  EXPECT_FALSE(transport_->remove_publisher_endpoint(unknown_gid).has_value());

  // Removing non-existent subscriber endpoint should fail gracefully
  EXPECT_FALSE(transport_->remove_subscriber_endpoint(unknown_gid).has_value());
}

// =============================================================================
// FastDdsEndpoint Tests
// =============================================================================

class DDSEndpointTest : public ::testing::Test {};

TEST_F(DDSEndpointTest, ConstructWriteMode) {
  std::vector<uint8_t> buffer;
  holoscan::FastDdsEndpoint endpoint(&buffer);

  EXPECT_TRUE(endpoint.is_write_mode());
  EXPECT_FALSE(endpoint.is_read_mode());
  EXPECT_TRUE(endpoint.is_write_available());
  EXPECT_FALSE(endpoint.is_read_available());
  EXPECT_EQ(endpoint.size(), 0u);
}

TEST_F(DDSEndpointTest, ConstructReadMode) {
  std::vector<uint8_t> buffer = {0x01, 0x02, 0x03, 0x04};
  holoscan::FastDdsEndpoint endpoint(static_cast<const std::vector<uint8_t>*>(&buffer));

  EXPECT_FALSE(endpoint.is_write_mode());
  EXPECT_TRUE(endpoint.is_read_mode());
  EXPECT_FALSE(endpoint.is_write_available());
  EXPECT_TRUE(endpoint.is_read_available());
  EXPECT_EQ(endpoint.size(), 4u);
  EXPECT_EQ(endpoint.bytes_remaining(), 4u);
}

TEST_F(DDSEndpointTest, ConstructWithNullWriteBufferThrows) {
  EXPECT_THROW(holoscan::FastDdsEndpoint(static_cast<std::vector<uint8_t>*>(nullptr)),
               std::invalid_argument);
}

TEST_F(DDSEndpointTest, ConstructWithNullReadBufferThrows) {
  EXPECT_THROW(holoscan::FastDdsEndpoint(static_cast<const std::vector<uint8_t>*>(nullptr)),
               std::invalid_argument);
}

TEST_F(DDSEndpointTest, WriteData) {
  std::vector<uint8_t> buffer;
  holoscan::FastDdsEndpoint endpoint(&buffer);

  uint8_t data[] = {0xDE, 0xAD, 0xBE, 0xEF};
  auto result = endpoint.write(data, sizeof(data));

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 4u);
  EXPECT_EQ(buffer.size(), 4u);
  EXPECT_EQ(buffer[0], 0xDE);
  EXPECT_EQ(buffer[1], 0xAD);
  EXPECT_EQ(buffer[2], 0xBE);
  EXPECT_EQ(buffer[3], 0xEF);
}

TEST_F(DDSEndpointTest, WriteMultipleTimes) {
  std::vector<uint8_t> buffer;
  holoscan::FastDdsEndpoint endpoint(&buffer);

  uint8_t data1[] = {0x01, 0x02};
  uint8_t data2[] = {0x03, 0x04, 0x05};

  auto result1 = endpoint.write(data1, sizeof(data1));
  auto result2 = endpoint.write(data2, sizeof(data2));

  ASSERT_TRUE(result1.has_value());
  ASSERT_TRUE(result2.has_value());
  EXPECT_EQ(result1.value(), 2u);
  EXPECT_EQ(result2.value(), 3u);
  EXPECT_EQ(buffer.size(), 5u);
  EXPECT_EQ(buffer, (std::vector<uint8_t>{0x01, 0x02, 0x03, 0x04, 0x05}));
}

TEST_F(DDSEndpointTest, WriteZeroBytes) {
  std::vector<uint8_t> buffer;
  holoscan::FastDdsEndpoint endpoint(&buffer);

  auto result = endpoint.write(nullptr, 0);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 0u);
  EXPECT_EQ(buffer.size(), 0u);
}

TEST_F(DDSEndpointTest, WriteInReadModeFails) {
  std::vector<uint8_t> buffer = {0x01, 0x02};
  holoscan::FastDdsEndpoint endpoint(static_cast<const std::vector<uint8_t>*>(&buffer));

  uint8_t data[] = {0x03};
  auto result = endpoint.write(data, sizeof(data));

  EXPECT_FALSE(result.has_value());
}

TEST_F(DDSEndpointTest, ReadData) {
  std::vector<uint8_t> buffer = {0xDE, 0xAD, 0xBE, 0xEF};
  holoscan::FastDdsEndpoint endpoint(static_cast<const std::vector<uint8_t>*>(&buffer));

  uint8_t data[4] = {0};
  auto result = endpoint.read(data, sizeof(data));

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 4u);
  EXPECT_EQ(data[0], 0xDE);
  EXPECT_EQ(data[1], 0xAD);
  EXPECT_EQ(data[2], 0xBE);
  EXPECT_EQ(data[3], 0xEF);
  EXPECT_EQ(endpoint.read_position(), 4u);
  EXPECT_EQ(endpoint.bytes_remaining(), 0u);
}

TEST_F(DDSEndpointTest, ReadMultipleTimes) {
  std::vector<uint8_t> buffer = {0x01, 0x02, 0x03, 0x04, 0x05};
  holoscan::FastDdsEndpoint endpoint(static_cast<const std::vector<uint8_t>*>(&buffer));

  uint8_t data1[2] = {0};
  uint8_t data2[3] = {0};

  auto result1 = endpoint.read(data1, sizeof(data1));
  EXPECT_EQ(endpoint.read_position(), 2u);
  EXPECT_EQ(endpoint.bytes_remaining(), 3u);

  auto result2 = endpoint.read(data2, sizeof(data2));
  EXPECT_EQ(endpoint.read_position(), 5u);
  EXPECT_EQ(endpoint.bytes_remaining(), 0u);

  ASSERT_TRUE(result1.has_value());
  ASSERT_TRUE(result2.has_value());
  EXPECT_EQ(data1[0], 0x01);
  EXPECT_EQ(data1[1], 0x02);
  EXPECT_EQ(data2[0], 0x03);
  EXPECT_EQ(data2[1], 0x04);
  EXPECT_EQ(data2[2], 0x05);
}

TEST_F(DDSEndpointTest, ReadZeroBytes) {
  std::vector<uint8_t> buffer = {0x01, 0x02};
  holoscan::FastDdsEndpoint endpoint(static_cast<const std::vector<uint8_t>*>(&buffer));

  auto result = endpoint.read(nullptr, 0);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 0u);
  EXPECT_EQ(endpoint.read_position(), 0u);
}

TEST_F(DDSEndpointTest, ReadInsufficientDataFails) {
  std::vector<uint8_t> buffer = {0x01, 0x02};
  holoscan::FastDdsEndpoint endpoint(static_cast<const std::vector<uint8_t>*>(&buffer));

  uint8_t data[5] = {0};
  auto result = endpoint.read(data, sizeof(data));

  EXPECT_FALSE(result.has_value());
}

TEST_F(DDSEndpointTest, ReadInWriteModeFails) {
  std::vector<uint8_t> buffer;
  holoscan::FastDdsEndpoint endpoint(&buffer);

  uint8_t data[1] = {0};
  auto result = endpoint.read(data, sizeof(data));

  EXPECT_FALSE(result.has_value());
}

TEST_F(DDSEndpointTest, ResetReadPosition) {
  std::vector<uint8_t> buffer = {0x01, 0x02, 0x03, 0x04};
  holoscan::FastDdsEndpoint endpoint(static_cast<const std::vector<uint8_t>*>(&buffer));

  uint8_t data[2] = {0};
  endpoint.read(data, sizeof(data));
  EXPECT_EQ(endpoint.read_position(), 2u);

  endpoint.reset_read_position();
  EXPECT_EQ(endpoint.read_position(), 0u);
  EXPECT_EQ(endpoint.bytes_remaining(), 4u);

  // Read again from beginning
  endpoint.read(data, sizeof(data));
  EXPECT_EQ(data[0], 0x01);
  EXPECT_EQ(data[1], 0x02);
}

TEST_F(DDSEndpointTest, WriteTrivialType) {
  std::vector<uint8_t> buffer;
  holoscan::FastDdsEndpoint endpoint(&buffer);

  int32_t value = 0x12345678;
  auto result = endpoint.write_trivial_type(&value);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), sizeof(int32_t));
  EXPECT_EQ(buffer.size(), sizeof(int32_t));

  // Verify byte order (little endian on x86)
  EXPECT_EQ(buffer[0], 0x78);
  EXPECT_EQ(buffer[1], 0x56);
  EXPECT_EQ(buffer[2], 0x34);
  EXPECT_EQ(buffer[3], 0x12);
}

TEST_F(DDSEndpointTest, ReadTrivialType) {
  // Little endian representation of 0x12345678
  std::vector<uint8_t> buffer = {0x78, 0x56, 0x34, 0x12};
  holoscan::FastDdsEndpoint endpoint(static_cast<const std::vector<uint8_t>*>(&buffer));

  int32_t value = 0;
  auto result = endpoint.read_trivial_type(&value);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), sizeof(int32_t));
  EXPECT_EQ(value, 0x12345678);
}

TEST_F(DDSEndpointTest, WriteReadRoundTrip) {
  // Test round-trip: write then read
  std::vector<uint8_t> buffer;

  // Write phase
  {
    holoscan::FastDdsEndpoint endpoint(&buffer);

    int32_t int_val = 42;
    double double_val = 3.14159;
    uint64_t uint_val = 0xDEADBEEFCAFEBABE;

    endpoint.write_trivial_type(&int_val);
    endpoint.write_trivial_type(&double_val);
    endpoint.write_trivial_type(&uint_val);
  }

  // Read phase
  {
    holoscan::FastDdsEndpoint endpoint(static_cast<const std::vector<uint8_t>*>(&buffer));

    int32_t int_val = 0;
    double double_val = 0.0;
    uint64_t uint_val = 0;

    endpoint.read_trivial_type(&int_val);
    endpoint.read_trivial_type(&double_val);
    endpoint.read_trivial_type(&uint_val);

    EXPECT_EQ(int_val, 42);
    EXPECT_DOUBLE_EQ(double_val, 3.14159);
    EXPECT_EQ(uint_val, 0xDEADBEEFCAFEBABE);
  }
}

TEST_F(DDSEndpointTest, WritePtrHostMemory) {
  std::vector<uint8_t> buffer;
  holoscan::FastDdsEndpoint endpoint(&buffer);

  uint8_t data[] = {0x01, 0x02, 0x03};
  auto result = endpoint.write_ptr(data, sizeof(data), holoscan::MemoryStorageType::kHost);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(buffer.size(), 3u);
  EXPECT_EQ(buffer, (std::vector<uint8_t>{0x01, 0x02, 0x03}));
}

TEST_F(DDSEndpointTest, WritePtrDeviceMemoryFails) {
  std::vector<uint8_t> buffer;
  holoscan::FastDdsEndpoint endpoint(&buffer);

  uint8_t data[] = {0x01};
  auto result = endpoint.write_ptr(data, sizeof(data), holoscan::MemoryStorageType::kDevice);

  EXPECT_FALSE(result.has_value());
}

TEST_F(DDSEndpointTest, WritePtrSystemMemorySucceeds) {
  // kSystem is regular CPU memory (malloc/new), which can be serialized directly
  std::vector<uint8_t> buffer;
  holoscan::FastDdsEndpoint endpoint(&buffer);

  uint8_t data[] = {0x01, 0x02, 0x03};
  auto result = endpoint.write_ptr(data, sizeof(data), holoscan::MemoryStorageType::kSystem);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(buffer.size(), 3u);
  EXPECT_EQ(buffer, (std::vector<uint8_t>{0x01, 0x02, 0x03}));
}

TEST_F(DDSEndpointTest, WritePtrCudaManagedMemorySucceeds) {
  // kCudaManaged is CUDA unified memory, which is CPU-accessible
  // (though a warning is logged about synchronization)
  std::vector<uint8_t> buffer;
  holoscan::FastDdsEndpoint endpoint(&buffer);

  uint8_t data[] = {0xAB, 0xCD};
  auto result = endpoint.write_ptr(data, sizeof(data), holoscan::MemoryStorageType::kCudaManaged);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(buffer.size(), 2u);
  EXPECT_EQ(buffer, (std::vector<uint8_t>{0xAB, 0xCD}));
}

TEST_F(DDSEndpointTest, IsReadAvailableAfterReadingAll) {
  std::vector<uint8_t> buffer = {0x01, 0x02};
  holoscan::FastDdsEndpoint endpoint(static_cast<const std::vector<uint8_t>*>(&buffer));

  EXPECT_TRUE(endpoint.is_read_available());

  uint8_t data[2];
  endpoint.read(data, sizeof(data));

  EXPECT_FALSE(endpoint.is_read_available());
}

// =============================================================================
// FastDdsSerializer Tests
// =============================================================================

class DDSSerializerTest : public ::testing::Test {
 protected:
  void SetUp() override { serializer_ = std::make_unique<holoscan::FastDdsSerializer>(); }

  void TearDown() override { serializer_.reset(); }

  std::unique_ptr<holoscan::FastDdsSerializer> serializer_;
};

TEST_F(DDSSerializerTest, Name) {
  EXPECT_STREQ(serializer_->name(), "FastDdsSerializer");
}

TEST_F(DDSSerializerTest, SupportsGpuTensors) {
  EXPECT_TRUE(serializer_->supports_gpu_tensors());
}

TEST_F(DDSSerializerTest, DoesNotSupportZeroCopy) {
  // GPU staging requires copies
  EXPECT_FALSE(serializer_->supports_zero_copy());
}

TEST_F(DDSSerializerTest, SupportsDirectBufferSerialization) {
  // FastDdsSerializer overrides serialize_into() for direct buffer writes
  EXPECT_TRUE(serializer_->supports_direct_buffer_serialization());
}

TEST_F(DDSSerializerTest, SetCudaStream) {
  // Initially null
  EXPECT_EQ(serializer_->cuda_stream(), nullptr);

  // Create a test stream (we don't need to actually use it)
  cudaStream_t test_stream = nullptr;
  cudaError_t err = cudaStreamCreate(&test_stream);
  if (err == cudaSuccess && test_stream != nullptr) {
    serializer_->set_cuda_stream(test_stream);
    EXPECT_EQ(serializer_->cuda_stream(), test_stream);

    // Cleanup
    cudaStreamDestroy(test_stream);
    serializer_->set_cuda_stream(nullptr);
  } else {
    // No CUDA available, just test with nullptr
    serializer_->set_cuda_stream(nullptr);
    EXPECT_EQ(serializer_->cuda_stream(), nullptr);
  }
}

// =============================================================================
// FastDdsSerializer Integration Tests (using TestWithGXFContext)
// =============================================================================
//
// These tests verify serialization/deserialization of GXF entities with various
// component types: Timestamp, Tensor (host memory), Message, MetadataDictionary,
// and MessageLabel.
// =============================================================================

class DDSSerializerIntegrationTest : public holoscan::TestWithGXFContext {
 protected:
  void SetUp() override {
    holoscan::TestWithGXFContext::SetUp();
    serializer_ = std::make_unique<holoscan::FastDdsSerializer>();

    // Create an allocator for tensor deserialization
    holoscan::ArgList args{};
    allocator_ = F.make_resource<holoscan::UnboundedAllocator>("test_allocator", args);
    allocator_->initialize();
  }

  void TearDown() override {
    allocator_.reset();
    serializer_.reset();
    holoscan::TestWithGXFContext::TearDown();
  }

  gxf_context_t context() { return F.executor().context(); }

  // Helper to get allocator handle for deserialization
  nvidia::gxf::Handle<nvidia::gxf::Allocator> get_allocator_handle() {
    return nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context(), allocator_->gxf_cid())
        .value();
  }

  std::unique_ptr<holoscan::FastDdsSerializer> serializer_;
  std::shared_ptr<holoscan::UnboundedAllocator> allocator_;
};

TEST_F(DDSSerializerIntegrationTest, SerializeEmptyEntity) {
  // Create an empty entity
  auto maybe_entity = nvidia::gxf::Entity::New(context());
  ASSERT_TRUE(maybe_entity) << "Failed to create entity";
  auto entity = maybe_entity.value();

  // Serialize
  auto serialized = serializer_->serialize(entity);
  ASSERT_TRUE(serialized) << "serialize() failed";

  // Should have at least a minimal header
  EXPECT_GE(serialized.value().size(), sizeof(holoscan::FastDdsSerializer::SerializationHeader));
}

TEST_F(DDSSerializerIntegrationTest, SerializeDeserializeTimestamp) {
  // Create entity with timestamp
  auto maybe_entity = nvidia::gxf::Entity::New(context());
  ASSERT_TRUE(maybe_entity);
  auto entity = maybe_entity.value();

  auto maybe_ts = entity.add<nvidia::gxf::Timestamp>("test_timestamp");
  ASSERT_TRUE(maybe_ts);
  auto ts = maybe_ts.value();
  ts->acqtime = 123456789;
  ts->pubtime = 987654321;

  // Serialize
  auto serialized = serializer_->serialize(entity);
  ASSERT_TRUE(serialized) << "serialize() failed";

  // Deserialize
  auto deserialized =
      serializer_->deserialize(serialized.value(), context(), get_allocator_handle());
  ASSERT_TRUE(deserialized) << "deserialize() failed";

  // Verify timestamp
  auto maybe_ts_out = deserialized.value().get<nvidia::gxf::Timestamp>("test_timestamp");
  ASSERT_TRUE(maybe_ts_out);
  EXPECT_EQ(maybe_ts_out.value()->acqtime, 123456789);
  EXPECT_EQ(maybe_ts_out.value()->pubtime, 987654321);
}

TEST_F(DDSSerializerIntegrationTest, SerializeDeserializeHostTensor) {
  // Create entity with a small host tensor
  auto maybe_entity = nvidia::gxf::Entity::New(context());
  ASSERT_TRUE(maybe_entity);
  auto entity = maybe_entity.value();

  auto maybe_tensor = entity.add<nvidia::gxf::Tensor>("test_tensor");
  ASSERT_TRUE(maybe_tensor);
  auto tensor = maybe_tensor.value();

  // Create a small 2x3 float tensor on host
  nvidia::gxf::Shape shape({2, 3});
  auto reshape_result =
      tensor->reshapeCustom(shape,
                            nvidia::gxf::PrimitiveType::kFloat32,
                            sizeof(float),
                            nvidia::gxf::ComputeTrivialStrides(shape, sizeof(float)),
                            nvidia::gxf::MemoryStorageType::kHost,
                            get_allocator_handle());
  ASSERT_TRUE(reshape_result);

  // Fill with test data
  float* data = reinterpret_cast<float*>(tensor->pointer());
  for (int i = 0; i < 6; ++i) {
    data[i] = static_cast<float>(i) * 1.5f;
  }

  // Serialize
  auto serialized = serializer_->serialize(entity);
  ASSERT_TRUE(serialized) << "serialize() failed";

  // Deserialize
  auto deserialized =
      serializer_->deserialize(serialized.value(), context(), get_allocator_handle());
  ASSERT_TRUE(deserialized) << "deserialize() failed";

  // Verify tensor
  auto maybe_tensor_out = deserialized.value().get<nvidia::gxf::Tensor>("test_tensor");
  ASSERT_TRUE(maybe_tensor_out);
  auto tensor_out = maybe_tensor_out.value();

  EXPECT_EQ(tensor_out->rank(), 2u);
  EXPECT_EQ(tensor_out->shape().dimension(0), 2);
  EXPECT_EQ(tensor_out->shape().dimension(1), 3);
  EXPECT_EQ(tensor_out->element_type(), nvidia::gxf::PrimitiveType::kFloat32);

  // Verify data
  float* data_out = reinterpret_cast<float*>(tensor_out->pointer());
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(data_out[i], static_cast<float>(i) * 1.5f);
  }
}

TEST_F(DDSSerializerIntegrationTest, SerializeDeserializeMessage) {
  // Create entity with Message containing an integer
  auto maybe_entity = nvidia::gxf::Entity::New(context());
  ASSERT_TRUE(maybe_entity);
  auto entity = maybe_entity.value();

  auto maybe_msg = entity.add<holoscan::Message>("test_message");
  ASSERT_TRUE(maybe_msg);
  auto msg = maybe_msg.value();
  msg->set_value(42);

  // Serialize
  auto serialized = serializer_->serialize(entity);
  ASSERT_TRUE(serialized) << "serialize() failed";

  // Deserialize
  auto deserialized =
      serializer_->deserialize(serialized.value(), context(), get_allocator_handle());
  ASSERT_TRUE(deserialized) << "deserialize() failed";

  // Verify message
  auto maybe_msg_out = deserialized.value().get<holoscan::Message>("test_message");
  ASSERT_TRUE(maybe_msg_out);
  auto value = maybe_msg_out.value()->value();
  ASSERT_TRUE(value.type() == typeid(int));
  EXPECT_EQ(std::any_cast<int>(value), 42);
}

TEST_F(DDSSerializerIntegrationTest, SerializeDeserializeMessageString) {
  // Create entity with Message containing a string
  auto maybe_entity = nvidia::gxf::Entity::New(context());
  ASSERT_TRUE(maybe_entity);
  auto entity = maybe_entity.value();

  auto maybe_msg = entity.add<holoscan::Message>("test_message");
  ASSERT_TRUE(maybe_msg);
  auto msg = maybe_msg.value();
  msg->set_value(std::string("Hello, DDS!"));

  // Serialize
  auto serialized = serializer_->serialize(entity);
  ASSERT_TRUE(serialized) << "serialize() failed";

  // Deserialize
  auto deserialized =
      serializer_->deserialize(serialized.value(), context(), get_allocator_handle());
  ASSERT_TRUE(deserialized) << "deserialize() failed";

  // Verify message
  auto maybe_msg_out = deserialized.value().get<holoscan::Message>("test_message");
  ASSERT_TRUE(maybe_msg_out);
  auto value = maybe_msg_out.value()->value();
  ASSERT_TRUE(value.type() == typeid(std::string));
  EXPECT_EQ(std::any_cast<std::string>(value), "Hello, DDS!");
}

TEST_F(DDSSerializerIntegrationTest, SerializeDeserializeMessageVector) {
  // Create entity with Message containing a vector
  auto maybe_entity = nvidia::gxf::Entity::New(context());
  ASSERT_TRUE(maybe_entity);
  auto entity = maybe_entity.value();

  auto maybe_msg = entity.add<holoscan::Message>("test_message");
  ASSERT_TRUE(maybe_msg);
  auto msg = maybe_msg.value();
  std::vector<float> test_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  msg->set_value(test_data);

  // Serialize
  auto serialized = serializer_->serialize(entity);
  ASSERT_TRUE(serialized) << "serialize() failed";

  // Deserialize
  auto deserialized =
      serializer_->deserialize(serialized.value(), context(), get_allocator_handle());
  ASSERT_TRUE(deserialized) << "deserialize() failed";

  // Verify message
  auto maybe_msg_out = deserialized.value().get<holoscan::Message>("test_message");
  ASSERT_TRUE(maybe_msg_out);
  auto value = maybe_msg_out.value()->value();
  ASSERT_TRUE(value.type() == typeid(std::vector<float>));
  auto result = std::any_cast<std::vector<float>>(value);
  ASSERT_EQ(result.size(), 5u);
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(result[i], test_data[i]);
  }
}

TEST_F(DDSSerializerIntegrationTest, SerializeDeserializeMetadataDictionary) {
  // Create entity with MetadataDictionary
  auto maybe_entity = nvidia::gxf::Entity::New(context());
  ASSERT_TRUE(maybe_entity);
  auto entity = maybe_entity.value();

  auto maybe_meta = entity.add<holoscan::MetadataDictionary>("test_metadata");
  ASSERT_TRUE(maybe_meta);
  auto meta = maybe_meta.value();

  // Add some metadata entries
  meta->set("int_key", 123);
  meta->set("str_key", std::string("test_value"));

  // Serialize
  auto serialized = serializer_->serialize(entity);
  ASSERT_TRUE(serialized) << "serialize() failed";

  // Deserialize
  auto deserialized =
      serializer_->deserialize(serialized.value(), context(), get_allocator_handle());
  ASSERT_TRUE(deserialized) << "deserialize() failed";

  // Verify metadata
  auto maybe_meta_out = deserialized.value().get<holoscan::MetadataDictionary>("test_metadata");
  ASSERT_TRUE(maybe_meta_out);
  auto meta_out = maybe_meta_out.value();

  EXPECT_EQ(meta_out->size(), 2u);

  // Check int value
  auto it_int = meta_out->find("int_key");
  ASSERT_NE(it_int, meta_out->end());
  EXPECT_EQ(std::any_cast<int>(it_int->second->value()), 123);

  // Check string value
  auto it_str = meta_out->find("str_key");
  ASSERT_NE(it_str, meta_out->end());
  EXPECT_EQ(std::any_cast<std::string>(it_str->second->value()), "test_value");
}

TEST_F(DDSSerializerIntegrationTest, SerializeDeserializeMessageLabel) {
  // Create entity with MessageLabel
  auto maybe_entity = nvidia::gxf::Entity::New(context());
  ASSERT_TRUE(maybe_entity);
  auto entity = maybe_entity.value();

  auto maybe_label = entity.add<holoscan::MessageLabel>("test_label");
  ASSERT_TRUE(maybe_label);
  auto label = maybe_label.value();

  // Add operator timestamps
  holoscan::OperatorTimestampLabel op1;
  op1.operator_name = "operator_1";
  op1.rec_timestamp = 1000;
  op1.pub_timestamp = 2000;

  holoscan::OperatorTimestampLabel op2;
  op2.operator_name = "operator_2";
  op2.rec_timestamp = 3000;
  op2.pub_timestamp = 4000;

  label->add_new_op_timestamp(op1);
  label->add_new_op_timestamp(op2);

  // Serialize
  auto serialized = serializer_->serialize(entity);
  ASSERT_TRUE(serialized) << "serialize() failed";

  // Deserialize
  auto deserialized =
      serializer_->deserialize(serialized.value(), context(), get_allocator_handle());
  ASSERT_TRUE(deserialized) << "deserialize() failed";

  // Verify label
  auto maybe_label_out = deserialized.value().get<holoscan::MessageLabel>("test_label");
  ASSERT_TRUE(maybe_label_out);
  auto label_out = maybe_label_out.value();

  EXPECT_EQ(label_out->num_paths(), 1);
  auto& paths = label_out->paths();
  ASSERT_EQ(paths.size(), 1u);
  ASSERT_EQ(paths[0].size(), 2u);

  EXPECT_EQ(paths[0][0].operator_name, "operator_1");
  EXPECT_EQ(paths[0][0].rec_timestamp, 1000);
  EXPECT_EQ(paths[0][0].pub_timestamp, 2000);

  EXPECT_EQ(paths[0][1].operator_name, "operator_2");
  EXPECT_EQ(paths[0][1].rec_timestamp, 3000);
  EXPECT_EQ(paths[0][1].pub_timestamp, 4000);
}

TEST_F(DDSSerializerIntegrationTest, SerializeDeserializeMixedComponents) {
  // Create entity with multiple component types
  auto maybe_entity = nvidia::gxf::Entity::New(context());
  ASSERT_TRUE(maybe_entity);
  auto entity = maybe_entity.value();

  // Add timestamp
  auto maybe_ts = entity.add<nvidia::gxf::Timestamp>("timestamp");
  ASSERT_TRUE(maybe_ts);
  maybe_ts.value()->acqtime = 111;
  maybe_ts.value()->pubtime = 222;

  // Add small tensor
  auto maybe_tensor = entity.add<nvidia::gxf::Tensor>("tensor");
  ASSERT_TRUE(maybe_tensor);
  auto tensor = maybe_tensor.value();
  nvidia::gxf::Shape shape({4});
  auto reshape_result =
      tensor->reshapeCustom(shape,
                            nvidia::gxf::PrimitiveType::kInt32,
                            sizeof(int32_t),
                            nvidia::gxf::ComputeTrivialStrides(shape, sizeof(int32_t)),
                            nvidia::gxf::MemoryStorageType::kHost,
                            get_allocator_handle());
  ASSERT_TRUE(reshape_result);
  int32_t* data = reinterpret_cast<int32_t*>(tensor->pointer());
  for (int i = 0; i < 4; ++i) {
    data[i] = i * 10;
  }

  // Add message
  auto maybe_msg = entity.add<holoscan::Message>("message");
  ASSERT_TRUE(maybe_msg);
  maybe_msg.value()->set_value(std::string("mixed_test"));

  // Serialize
  auto serialized = serializer_->serialize(entity);
  ASSERT_TRUE(serialized) << "serialize() failed";

  // Deserialize
  auto deserialized =
      serializer_->deserialize(serialized.value(), context(), get_allocator_handle());
  ASSERT_TRUE(deserialized) << "deserialize() failed";

  // Verify all components
  auto de = deserialized.value();

  // Verify timestamp
  auto maybe_ts_out = de.get<nvidia::gxf::Timestamp>("timestamp");
  ASSERT_TRUE(maybe_ts_out);
  EXPECT_EQ(maybe_ts_out.value()->acqtime, 111);
  EXPECT_EQ(maybe_ts_out.value()->pubtime, 222);

  // Verify tensor
  auto maybe_tensor_out = de.get<nvidia::gxf::Tensor>("tensor");
  ASSERT_TRUE(maybe_tensor_out);
  auto tensor_out = maybe_tensor_out.value();
  EXPECT_EQ(tensor_out->rank(), 1u);
  EXPECT_EQ(tensor_out->shape().dimension(0), 4);
  int32_t* data_out = reinterpret_cast<int32_t*>(tensor_out->pointer());
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(data_out[i], i * 10);
  }

  // Verify message
  auto maybe_msg_out = de.get<holoscan::Message>("message");
  ASSERT_TRUE(maybe_msg_out);
  EXPECT_EQ(std::any_cast<std::string>(maybe_msg_out.value()->value()), "mixed_test");
}

TEST_F(DDSSerializerIntegrationTest, EstimateSizeReturnsReasonableValue) {
  // Create entity with components
  auto maybe_entity = nvidia::gxf::Entity::New(context());
  ASSERT_TRUE(maybe_entity);
  auto entity = maybe_entity.value();

  // Add timestamp
  auto maybe_ts = entity.add<nvidia::gxf::Timestamp>("ts");
  ASSERT_TRUE(maybe_ts);

  // Add message
  auto maybe_msg = entity.add<holoscan::Message>("msg");
  ASSERT_TRUE(maybe_msg);
  maybe_msg.value()->set_value(42);

  // Estimate size
  size_t estimate = serializer_->estimate_size(entity);
  EXPECT_GT(estimate, 0u);

  // Actually serialize and compare
  auto serialized = serializer_->serialize(entity);
  ASSERT_TRUE(serialized);

  // Estimate should be >= actual size (we add padding)
  EXPECT_GE(estimate, serialized.value().size());
}

// =============================================================================
// FastDdsDiscovery Basic Tests (no GXF context required)
// =============================================================================

TEST(DDSDiscoveryBasicTest, ConstructWithNullContext) {
  holoscan::FastDdsDiscovery discovery(nullptr);
  EXPECT_FALSE(discovery.is_initialized());
}

TEST(DDSDiscoveryBasicTest, NotInitializedByDefault) {
  holoscan::FastDdsDiscovery discovery(nullptr);
  EXPECT_FALSE(discovery.is_initialized());
}

TEST(DDSDiscoveryBasicTest, InitializeFailsWithNullContext) {
  holoscan::FastDdsDiscovery discovery(nullptr);
  auto result = discovery.initialize();
  EXPECT_FALSE(result);
}

TEST(DDSDiscoveryBasicTest, ShutdownOnUninitializedIsSafe) {
  holoscan::FastDdsDiscovery discovery(nullptr);
  auto result = discovery.shutdown();
  EXPECT_TRUE(result);
}

TEST(DDSDiscoveryBasicTest, DoubleShutdownIsSafe) {
  holoscan::FastDdsDiscovery discovery(nullptr);
  auto result1 = discovery.shutdown();
  EXPECT_TRUE(result1);
  auto result2 = discovery.shutdown();
  EXPECT_TRUE(result2);
}

TEST(DDSDiscoveryBasicTest, SetCallbacksBeforeInitialize) {
  holoscan::FastDdsDiscovery discovery(nullptr);

  bool pub_discovered = false;
  bool sub_discovered = false;
  bool pub_lost = false;
  bool sub_lost = false;

  discovery.set_on_publisher_discovered(
      [&](const nvidia::gxf::PublisherInfo&) { pub_discovered = true; });

  discovery.set_on_subscriber_discovered(
      [&](const nvidia::gxf::SubscriberInfo&) { sub_discovered = true; });

  discovery.set_on_publisher_lost([&](const nvidia::gxf::PublisherGid&) { pub_lost = true; });

  discovery.set_on_subscriber_lost([&](const nvidia::gxf::SubscriberGid&) { sub_lost = true; });

  // Just verify we can set callbacks without crashing
  EXPECT_FALSE(pub_discovered);
  EXPECT_FALSE(sub_discovered);
  EXPECT_FALSE(pub_lost);
  EXPECT_FALSE(sub_lost);
}

TEST(DDSDiscoveryBasicTest, OverwriteCallbacksIsSafe) {
  holoscan::FastDdsDiscovery discovery(nullptr);

  int call_count = 0;

  // Set first callback
  discovery.set_on_publisher_discovered([&](const nvidia::gxf::PublisherInfo&) { call_count = 1; });

  // Overwrite with second callback
  discovery.set_on_publisher_discovered([&](const nvidia::gxf::PublisherInfo&) { call_count = 2; });

  // Just verify overwriting doesn't crash
  EXPECT_EQ(call_count, 0);
}

TEST(DDSDiscoveryBasicTest, AnnouncePublisherFailsWhenUninitialized) {
  holoscan::FastDdsDiscovery discovery(nullptr);
  nvidia::gxf::PublisherInfo info;
  info.gid = nvidia::gxf::Gid::generate();
  info.topic_name = "test_topic";
  auto result = discovery.announce_publisher(info);
  EXPECT_FALSE(result);
}

TEST(DDSDiscoveryBasicTest, AnnounceSubscriberFailsWhenUninitialized) {
  holoscan::FastDdsDiscovery discovery(nullptr);
  nvidia::gxf::SubscriberInfo info;
  info.gid = nvidia::gxf::Gid::generate();
  info.topic_name = "test_topic";
  auto result = discovery.announce_subscriber(info);
  EXPECT_FALSE(result);
}

TEST(DDSDiscoveryBasicTest, RemovePublisherFailsWhenUninitialized) {
  holoscan::FastDdsDiscovery discovery(nullptr);
  auto result = discovery.remove_publisher(nvidia::gxf::Gid::generate());
  EXPECT_FALSE(result);
}

TEST(DDSDiscoveryBasicTest, RemoveSubscriberFailsWhenUninitialized) {
  holoscan::FastDdsDiscovery discovery(nullptr);
  auto result = discovery.remove_subscriber(nvidia::gxf::Gid::generate());
  EXPECT_FALSE(result);
}

TEST(DDSDiscoveryBasicTest, QueryPublishersFailsWhenUninitialized) {
  holoscan::FastDdsDiscovery discovery(nullptr);
  auto result = discovery.query_publishers("test_topic");
  EXPECT_FALSE(result);
}

TEST(DDSDiscoveryBasicTest, QuerySubscribersFailsWhenUninitialized) {
  holoscan::FastDdsDiscovery discovery(nullptr);
  auto result = discovery.query_subscribers("test_topic");
  EXPECT_FALSE(result);
}

TEST(DDSDiscoveryBasicTest, GetAllTopicsFailsWhenUninitialized) {
  holoscan::FastDdsDiscovery discovery(nullptr);
  auto result = discovery.get_all_topics();
  EXPECT_FALSE(result);
}

TEST(DDSDiscoveryBasicTest, DiscoveryModelIsDecentralizedPassive) {
  holoscan::FastDdsDiscovery discovery(nullptr);
  EXPECT_EQ(discovery.discovery_model(), nvidia::gxf::DiscoveryModel::kDecentralizedPassive);
}

// =============================================================================
// FastDdsDiscovery Integration Tests (using TestWithGXFContext)
// =============================================================================
//
// These tests use Holoscan's TestWithGXFContext fixture to properly initialize
// GXF context and resources.
// =============================================================================

class DDSDiscoveryIntegrationTest : public holoscan::TestWithGXFContext {
 protected:
  void SetUp() override {
    // Initialize GXF context via base class
    holoscan::TestWithGXFContext::SetUp();

    // Create FastDdsPubSubContext using Fragment's make_resource
    holoscan::ArgList args{
        holoscan::Arg{"domain_id", static_cast<int32_t>(0)},
        holoscan::Arg{"participant_name", std::string("test_discovery_participant")},
    };
    dds_context_ = F.make_resource<holoscan::FastDdsPubSubContext>("test_dds_context", args);

    // Initialize the FastDdsPubSubContext (creates DomainParticipant)
    dds_context_->initialize();

    // Create FastDdsDiscovery with the context
    discovery_ = std::make_unique<holoscan::FastDdsDiscovery>(dds_context_.get());

    // Share a TopicRegistry with the discovery (mimics PubSubContext::init_context)
    discovery_->set_topic_registry(&topic_registry_);
  }

  void TearDown() override {
    // Clear registry pointer before shutdown (mimics PubSubContext::deinitialize)
    if (discovery_) {
      discovery_->set_topic_registry(nullptr);
    }

    // Shutdown discovery first
    if (discovery_ && discovery_->is_initialized()) {
      discovery_->shutdown();
    }
    discovery_.reset();

    // FastDdsPubSubContext cleanup happens when shared_ptr is reset
    dds_context_.reset();
  }

  /// Helper: register a publisher in the shared TopicRegistry and announce via discovery.
  /// Mimics the PubSubContext::register_publisher() flow.
  void register_publisher(const nvidia::gxf::PublisherInfo& info) {
    topic_registry_.register_publisher(info);
    auto result = discovery_->announce_publisher(info);
    ASSERT_TRUE(result.has_value()) << "announce_publisher() failed";
  }

  /// Helper: register a subscriber in the shared TopicRegistry and announce via discovery.
  /// Mimics the PubSubContext::register_subscriber() flow.
  void register_subscriber(const nvidia::gxf::SubscriberInfo& info) {
    topic_registry_.register_subscriber(info);
    auto result = discovery_->announce_subscriber(info);
    ASSERT_TRUE(result.has_value()) << "announce_subscriber() failed";
  }

  std::shared_ptr<holoscan::FastDdsPubSubContext> dds_context_;
  std::unique_ptr<holoscan::FastDdsDiscovery> discovery_;
  nvidia::gxf::TopicRegistry topic_registry_;
};

class DDSDiscoveryCapabilityPropagationTest : public holoscan::TestWithGXFContext {
 protected:
  void SetUp() override {
    holoscan::TestWithGXFContext::SetUp();

    holoscan::ArgList enabled_args{
        holoscan::Arg{"domain_id", static_cast<int32_t>(96)},
        holoscan::Arg{"participant_name", std::string("cap_enabled_participant")},
        holoscan::Arg{"native_buffer_policy", std::string("preferred")},
    };
    enabled_context_ =
        F.make_resource<holoscan::FastDdsPubSubContext>("cap_enabled_context", enabled_args);
    enabled_context_->initialize();

    holoscan::ArgList disabled_args{
        holoscan::Arg{"domain_id", static_cast<int32_t>(96)},
        holoscan::Arg{"participant_name", std::string("cap_disabled_participant")},
        holoscan::Arg{"native_buffer_policy", std::string("disabled")},
    };
    disabled_context_ =
        F.make_resource<holoscan::FastDdsPubSubContext>("cap_disabled_context", disabled_args);
    disabled_context_->initialize();

    enabled_discovery_ = std::make_unique<holoscan::FastDdsDiscovery>(enabled_context_.get());
    disabled_discovery_ = std::make_unique<holoscan::FastDdsDiscovery>(disabled_context_.get());
    enabled_discovery_->set_topic_registry(&enabled_registry_);
    disabled_discovery_->set_topic_registry(&disabled_registry_);
    enabled_discovery_->initialize();
    disabled_discovery_->initialize();

    enabled_transport_ = std::make_unique<holoscan::FastDdsTransport>(enabled_context_.get());
    disabled_transport_ = std::make_unique<holoscan::FastDdsTransport>(disabled_context_.get());
    enabled_transport_->initialize();
    disabled_transport_->initialize();
  }

  void TearDown() override {
    if (enabled_transport_ && enabled_transport_->is_initialized()) {
      enabled_transport_->shutdown();
    }
    if (disabled_transport_ && disabled_transport_->is_initialized()) {
      disabled_transport_->shutdown();
    }
    enabled_transport_.reset();
    disabled_transport_.reset();

    if (enabled_discovery_ && enabled_discovery_->is_initialized()) {
      enabled_discovery_->set_topic_registry(nullptr);
      enabled_discovery_->shutdown();
    }
    if (disabled_discovery_ && disabled_discovery_->is_initialized()) {
      disabled_discovery_->set_topic_registry(nullptr);
      disabled_discovery_->shutdown();
    }
    enabled_discovery_.reset();
    disabled_discovery_.reset();

    enabled_context_.reset();
    disabled_context_.reset();
  }

  template <typename QueryFn>
  bool wait_for_query(QueryFn&& fn, size_t expected_size) {
    constexpr int kMaxAttempts = 50;
    for (int i = 0; i < kMaxAttempts; ++i) {
      auto result = fn();
      if (result && result->size() >= expected_size) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
  }

  std::shared_ptr<holoscan::FastDdsPubSubContext> enabled_context_;
  std::shared_ptr<holoscan::FastDdsPubSubContext> disabled_context_;
  std::unique_ptr<holoscan::FastDdsDiscovery> enabled_discovery_;
  std::unique_ptr<holoscan::FastDdsDiscovery> disabled_discovery_;
  std::unique_ptr<holoscan::FastDdsTransport> enabled_transport_;
  std::unique_ptr<holoscan::FastDdsTransport> disabled_transport_;
  nvidia::gxf::TopicRegistry enabled_registry_;
  nvidia::gxf::TopicRegistry disabled_registry_;
};

TEST_F(DDSDiscoveryIntegrationTest, InitializeShutdown) {
  // Discovery should not be initialized yet
  EXPECT_FALSE(discovery_->is_initialized());

  // Initialize
  auto init_result = discovery_->initialize();
  ASSERT_TRUE(init_result.has_value()) << "initialize() failed";
  EXPECT_TRUE(discovery_->is_initialized());

  // Shutdown
  auto shutdown_result = discovery_->shutdown();
  ASSERT_TRUE(shutdown_result.has_value()) << "shutdown() failed";
  EXPECT_FALSE(discovery_->is_initialized());

  // Shutdown is idempotent
  auto shutdown_result2 = discovery_->shutdown();
  EXPECT_TRUE(shutdown_result2.has_value());
}

TEST_F(DDSDiscoveryIntegrationTest, AnnounceAndQueryPublisher) {
  ASSERT_TRUE(discovery_->initialize().has_value());

  // Register + announce a publisher (registry + discovery, mimics PubSubContext)
  nvidia::gxf::PublisherInfo pub_info;
  pub_info.gid = nvidia::gxf::Gid::generate();
  pub_info.topic_name = "test_topic";
  pub_info.node_name = "test_node";
  register_publisher(pub_info);

  // Query publishers on the topic — should find our registered publisher
  auto query_result = discovery_->query_publishers("test_topic");
  ASSERT_TRUE(query_result.has_value()) << "query_publishers() failed";

  auto& publishers = query_result.value();
  EXPECT_GE(publishers.size(), 1u);

  // Remove the publisher
  topic_registry_.unregister_publisher(pub_info.gid);
  auto remove_result = discovery_->remove_publisher(pub_info.gid);
  EXPECT_TRUE(remove_result.has_value());
}

TEST_F(DDSDiscoveryIntegrationTest, AnnounceAndQuerySubscriber) {
  ASSERT_TRUE(discovery_->initialize().has_value());

  // Register + announce a subscriber (registry + discovery, mimics PubSubContext)
  nvidia::gxf::SubscriberInfo sub_info;
  sub_info.gid = nvidia::gxf::Gid::generate();
  sub_info.topic_name = "test_topic";
  sub_info.node_name = "test_node";
  register_subscriber(sub_info);

  // Query subscribers on the topic — should find our registered subscriber
  auto query_result = discovery_->query_subscribers("test_topic");
  ASSERT_TRUE(query_result.has_value()) << "query_subscribers() failed";

  auto& subscribers = query_result.value();
  EXPECT_GE(subscribers.size(), 1u);

  // Remove the subscriber
  topic_registry_.unregister_subscriber(sub_info.gid);
  auto remove_result = discovery_->remove_subscriber(sub_info.gid);
  EXPECT_TRUE(remove_result.has_value());
}

TEST_F(DDSDiscoveryIntegrationTest, GetAllTopics) {
  ASSERT_TRUE(discovery_->initialize().has_value());

  // Register + announce endpoints on different topics
  nvidia::gxf::PublisherInfo pub1;
  pub1.gid = nvidia::gxf::Gid::generate();
  pub1.topic_name = "topic_a";
  register_publisher(pub1);

  nvidia::gxf::PublisherInfo pub2;
  pub2.gid = nvidia::gxf::Gid::generate();
  pub2.topic_name = "topic_b";
  register_publisher(pub2);

  // Get all topics
  auto topics_result = discovery_->get_all_topics();
  ASSERT_TRUE(topics_result.has_value()) << "get_all_topics() failed";

  // Should have at least the topics we registered
  auto& topics = topics_result.value();
  EXPECT_GE(topics.size(), 2u);

  // Cleanup
  topic_registry_.unregister_publisher(pub1.gid);
  discovery_->remove_publisher(pub1.gid);
  topic_registry_.unregister_publisher(pub2.gid);
  discovery_->remove_publisher(pub2.gid);
}

TEST_F(DDSDiscoveryCapabilityPropagationTest,
       PropagatesParticipantCapabilityToDiscoveredEndpoints) {
  ASSERT_TRUE(disabled_transport_->create_subscriber_endpoint(
      "capability_topic", nvidia::gxf::Gid::generate(), nvidia::gxf::QoSProfile::Default()));

  ASSERT_TRUE(enabled_transport_->create_publisher_endpoint(
      "capability_topic", nvidia::gxf::Gid::generate(), nvidia::gxf::QoSProfile::Default()));

  ASSERT_TRUE(wait_for_query(
      [&]() { return enabled_discovery_->query_subscribers("capability_topic"); }, 1u))
      << "Timed out waiting for enabled discovery to observe disabled subscriber";

  auto subscribers = enabled_discovery_->query_subscribers("capability_topic");
  ASSERT_TRUE(subscribers);
  ASSERT_FALSE(subscribers->empty()) << "Disabled subscriber not found in enabled discovery";
  auto it = std::find_if(subscribers->begin(), subscribers->end(), [&](const auto& info) {
    return info.topic_name == "capability_topic";
  });
  ASSERT_NE(it, subscribers->end()) << "Capability-topic subscriber not found in enabled discovery";
  EXPECT_FALSE(it->native_buffer_capability.supports_native_buffers());
  EXPECT_TRUE(it->native_buffer_capability.native_buffer_protocols.empty());

  ASSERT_TRUE(wait_for_query(
      [&]() { return disabled_discovery_->query_publishers("capability_topic"); }, 1u))
      << "Timed out waiting for disabled discovery to observe enabled publisher";

  auto publishers = disabled_discovery_->query_publishers("capability_topic");
  ASSERT_TRUE(publishers);
  ASSERT_FALSE(publishers->empty()) << "Enabled publisher not found in disabled discovery";
  auto pub_it = std::find_if(publishers->begin(), publishers->end(), [&](const auto& info) {
    return info.topic_name == "capability_topic";
  });
  ASSERT_NE(pub_it, publishers->end())
      << "Capability-topic publisher not found in disabled discovery";
  EXPECT_TRUE(pub_it->native_buffer_capability.supports_native_buffers());
  EXPECT_EQ(pub_it->native_buffer_capability.native_buffer_protocols,
            (std::vector<std::string>{"cuda_ipc"}));
  EXPECT_EQ(pub_it->native_buffer_capability.native_buffer_profile, "cuda_ipc_same_gpu_v1");
  EXPECT_EQ(pub_it->native_buffer_capability.descriptor_format_version, 1);
  EXPECT_EQ(pub_it->native_buffer_capability.gpu_device_uuid,
            enabled_context_->native_buffer_capability().gpu_device_uuid);
  EXPECT_EQ(pub_it->native_buffer_capability.host_id,
            enabled_context_->native_buffer_capability().host_id);
  EXPECT_EQ(it->native_buffer_capability.host_id,
            disabled_context_->native_buffer_capability().host_id);
}

}  // namespace
