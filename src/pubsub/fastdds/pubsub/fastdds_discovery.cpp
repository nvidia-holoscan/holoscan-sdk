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

#include <holoscan/pubsub/fastdds/pubsub/fastdds_discovery.hpp>

#include <algorithm>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <fastdds/dds/core/status/StatusMask.hpp>
#include <fastdds/rtps/common/Guid.hpp>

#include <gxf/pubsub/cuda_ipc_eligibility.hpp>
#include <gxf/pubsub/topic_registry.hpp>

#include <holoscan/logger/logger.hpp>
#include <holoscan/pubsub/fastdds/resources/fastdds_pubsub_context.hpp>

namespace holoscan {

//------------------------------------------------------------------------------
// DiscoveryListener - DDS listener that forwards discovery events
//------------------------------------------------------------------------------

class FastDdsDiscovery::DiscoveryListener
    : public eprosima::fastdds::dds::DomainParticipantListener {
 public:
  explicit DiscoveryListener(FastDdsDiscovery* discovery) : discovery_(discovery) {}

  // Participant discovery events
  void on_participant_discovery(eprosima::fastdds::dds::DomainParticipant* /* participant */,
                                eprosima::fastdds::rtps::ParticipantDiscoveryStatus status,
                                const eprosima::fastdds::rtps::ParticipantBuiltinTopicData& info,
                                bool& /* should_be_ignored */) override {
    nvidia::gxf::Gid gid = FastDdsDiscovery::guid_to_gid(info.guid);
    switch (status) {
      case eprosima::fastdds::rtps::ParticipantDiscoveryStatus::DISCOVERED_PARTICIPANT:
      case eprosima::fastdds::rtps::ParticipantDiscoveryStatus::CHANGED_QOS_PARTICIPANT: {
        HOLOSCAN_LOG_DEBUG("FastDdsDiscovery: Discovered participant: {}",
                           info.participant_name.to_string());
        auto capability = FastDdsDiscovery::parse_native_capability(info.user_data.data_vec());
        {
          std::lock_guard<std::mutex> lock(discovery_->mutex_);
          discovery_->discovered_participant_capabilities_[gid] = std::move(capability);
        }
        break;
      }
      case eprosima::fastdds::rtps::ParticipantDiscoveryStatus::REMOVED_PARTICIPANT:
      case eprosima::fastdds::rtps::ParticipantDiscoveryStatus::DROPPED_PARTICIPANT:
        HOLOSCAN_LOG_DEBUG("FastDdsDiscovery: Lost participant: {}",
                           info.participant_name.to_string());
        {
          std::lock_guard<std::mutex> lock(discovery_->mutex_);
          discovery_->discovered_participant_capabilities_.erase(gid);
        }
        break;
      default:
        break;
    }
  }

  // DataWriter (publisher) discovery events
  void on_data_writer_discovery(eprosima::fastdds::dds::DomainParticipant* /* participant */,
                                eprosima::fastdds::rtps::WriterDiscoveryStatus status,
                                const eprosima::fastdds::rtps::PublicationBuiltinTopicData& info,
                                bool& /* should_be_ignored */) override {
    // NOTE: info.guid is the transport-native FastDDS writer GUID discovered via
    // SEDP. We convert it into a framework-level Gid for EndpointInfo storage.
    // This GUID-derived Gid is distinct from any application-assigned publisher
    // Gid that may have been passed to create_publisher_endpoint() locally.
    nvidia::gxf::Gid discovered_writer_gid = FastDdsDiscovery::guid_to_gid(info.guid);

    switch (status) {
      case eprosima::fastdds::rtps::WriterDiscoveryStatus::DISCOVERED_WRITER: {
        // Create PublisherInfo from DDS discovery data
        nvidia::gxf::PublisherInfo pub_info;
        pub_info.gid = discovered_writer_gid;
        pub_info.topic_name = info.topic_name.to_string();
        pub_info.type_name = info.type_name.to_string();
        nvidia::gxf::Gid discovered_participant_gid =
            FastDdsDiscovery::guid_to_gid(info.participant_guid);

        HOLOSCAN_LOG_DEBUG("FastDdsDiscovery: Discovered writer on topic: {}", pub_info.topic_name);

        // Store in discovered publishers
        {
          std::lock_guard<std::mutex> lock(discovery_->mutex_);
          auto cap_it =
              discovery_->discovered_participant_capabilities_.find(discovered_participant_gid);
          if (cap_it != discovery_->discovered_participant_capabilities_.end()) {
            pub_info.native_buffer_capability = cap_it->second;
          }
          discovery_->discovered_publishers_[discovered_writer_gid] = pub_info;
        }

        // Fire callback
        if (discovery_->on_publisher_discovered_) {
          discovery_->on_publisher_discovered_(pub_info);
        }
        break;
      }
      case eprosima::fastdds::rtps::WriterDiscoveryStatus::REMOVED_WRITER: {
        HOLOSCAN_LOG_DEBUG("FastDdsDiscovery: Lost writer: {}", discovered_writer_gid.to_string());

        // Remove from discovered publishers
        {
          std::lock_guard<std::mutex> lock(discovery_->mutex_);
          discovery_->discovered_publishers_.erase(discovered_writer_gid);
        }

        // Fire callback
        if (discovery_->on_publisher_lost_) {
          discovery_->on_publisher_lost_(discovered_writer_gid);
        }
        break;
      }
      default:
        break;
    }
  }

  // DataReader (subscriber) discovery events
  void on_data_reader_discovery(eprosima::fastdds::dds::DomainParticipant* /* participant */,
                                eprosima::fastdds::rtps::ReaderDiscoveryStatus status,
                                const eprosima::fastdds::rtps::SubscriptionBuiltinTopicData& info,
                                bool& /* should_be_ignored */) override {
    // NOTE: info.guid is the transport-native FastDDS reader GUID discovered via
    // SEDP. We convert it into a framework-level Gid for EndpointInfo storage.
    // This GUID-derived Gid is distinct from any application-assigned subscriber
    // Gid that may have been passed to create_subscriber_endpoint() locally.
    nvidia::gxf::Gid discovered_reader_gid = FastDdsDiscovery::guid_to_gid(info.guid);

    switch (status) {
      case eprosima::fastdds::rtps::ReaderDiscoveryStatus::DISCOVERED_READER: {
        // Create SubscriberInfo from DDS discovery data
        nvidia::gxf::SubscriberInfo sub_info;
        sub_info.gid = discovered_reader_gid;
        sub_info.topic_name = info.topic_name.to_string();
        sub_info.type_name = info.type_name.to_string();
        nvidia::gxf::Gid discovered_participant_gid =
            FastDdsDiscovery::guid_to_gid(info.participant_guid);

        HOLOSCAN_LOG_DEBUG("FastDdsDiscovery: Discovered reader on topic: {}", sub_info.topic_name);

        // Store in discovered subscribers
        {
          std::lock_guard<std::mutex> lock(discovery_->mutex_);
          auto cap_it =
              discovery_->discovered_participant_capabilities_.find(discovered_participant_gid);
          if (cap_it != discovery_->discovered_participant_capabilities_.end()) {
            sub_info.native_buffer_capability = cap_it->second;
          }
          discovery_->discovered_subscribers_[discovered_reader_gid] = sub_info;
        }

        // Fire callback
        if (discovery_->on_subscriber_discovered_) {
          discovery_->on_subscriber_discovered_(sub_info);
        }
        break;
      }
      case eprosima::fastdds::rtps::ReaderDiscoveryStatus::REMOVED_READER: {
        HOLOSCAN_LOG_DEBUG("FastDdsDiscovery: Lost reader: {}", discovered_reader_gid.to_string());

        // Remove from discovered subscribers
        {
          std::lock_guard<std::mutex> lock(discovery_->mutex_);
          discovery_->discovered_subscribers_.erase(discovered_reader_gid);
        }

        // Fire callback
        if (discovery_->on_subscriber_lost_) {
          discovery_->on_subscriber_lost_(discovered_reader_gid);
        }
        break;
      }
      default:
        break;
    }
  }

 private:
  FastDdsDiscovery* discovery_;
};

//------------------------------------------------------------------------------
// FastDdsDiscovery Implementation
//------------------------------------------------------------------------------

FastDdsDiscovery::FastDdsDiscovery(FastDdsPubSubContext* context) : context_(context) {}

FastDdsDiscovery::~FastDdsDiscovery() {
  if (initialized_) {
    auto result = shutdown();
    if (!result) {
      HOLOSCAN_LOG_ERROR("FastDdsDiscovery: Failed to shutdown in destructor");
    }
  }
}

nvidia::gxf::Expected<void> FastDdsDiscovery::initialize() {
  if (initialized_) {
    return nvidia::gxf::Success;
  }

  if (!context_) {
    HOLOSCAN_LOG_ERROR("FastDdsDiscovery: No FastDdsPubSubContext provided");
    return nvidia::gxf::Unexpected(GXF_ARGUMENT_NULL);
  }

  auto* participant = context_->participant();
  if (!participant) {
    HOLOSCAN_LOG_ERROR("FastDdsDiscovery: DomainParticipant not available");
    return nvidia::gxf::Unexpected(GXF_PUBSUB_DISCOVERY_FAILED);
  }

  // Create and attach the listener
  listener_ = std::make_unique<DiscoveryListener>(this);

  // Note: The listener is set on the participant. In FastDDS, we can set it
  // during participant creation or later via set_listener(). Since the
  // participant is already created by FastDdsPubSubContext, we set it here.
  //
  // IMPORTANT: We pass StatusMask::none() so the participant listener ONLY
  // receives discovery callbacks (on_participant_discovery, on_data_writer_discovery,
  // on_data_reader_discovery) which are NOT governed by the status mask.
  //
  // Without this, the default StatusMask::all() causes the participant listener
  // to intercept DATA_ON_READERS_STATUS, which prevents the DataReaderListener's
  // on_data_available() from firing (FastDDS listener hierarchy: participant >
  // subscriber > datareader — higher-level listeners steal status events from
  // lower-level ones).
  auto ret = participant->set_listener(listener_.get(), eprosima::fastdds::dds::StatusMask::none());
  if (ret != eprosima::fastdds::dds::RETCODE_OK) {
    HOLOSCAN_LOG_ERROR("FastDdsDiscovery: Failed to set participant listener");
    return nvidia::gxf::Unexpected(GXF_PUBSUB_DISCOVERY_FAILED);
  }

  initialized_ = true;
  HOLOSCAN_LOG_INFO("FastDdsDiscovery: Initialized");
  return nvidia::gxf::Success;
}

nvidia::gxf::Expected<void> FastDdsDiscovery::shutdown() {
  if (!initialized_) {
    return nvidia::gxf::Success;
  }

  // Remove listener from participant
  if (context_ && context_->participant()) {
    context_->participant()->set_listener(nullptr);
  }

  // Clear listener
  listener_.reset();

  // Clear discovered remote endpoints
  // (Local endpoints are owned by the shared TopicRegistry, cleared by PubSubContext)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    discovered_participant_capabilities_.clear();
    discovered_publishers_.clear();
    discovered_subscribers_.clear();
  }

  initialized_ = false;
  HOLOSCAN_LOG_INFO("FastDdsDiscovery: Shutdown complete");
  return nvidia::gxf::Success;
}

bool FastDdsDiscovery::is_initialized() const {
  return initialized_;
}

nvidia::gxf::Expected<void> FastDdsDiscovery::announce_publisher(
    const nvidia::gxf::PublisherInfo& info) {
  if (!initialized_) {
    return nvidia::gxf::Unexpected(GXF_CONTRACT_INVALID_SEQUENCE);
  }

  // Local bookkeeping only — the endpoint is already tracked by the shared
  // TopicRegistry (registered by PubSubContext before calling announce).
  // Actual DDS discovery happens when FastDdsTransport::create_publisher_endpoint()
  // creates the DataWriter.
  HOLOSCAN_LOG_DEBUG("FastDdsDiscovery: Announced local publisher on topic: {}", info.topic_name);
  return nvidia::gxf::Success;
}

nvidia::gxf::Expected<void> FastDdsDiscovery::announce_subscriber(
    const nvidia::gxf::SubscriberInfo& info) {
  if (!initialized_) {
    return nvidia::gxf::Unexpected(GXF_CONTRACT_INVALID_SEQUENCE);
  }

  // Local bookkeeping only — the endpoint is already tracked by the shared
  // TopicRegistry (registered by PubSubContext before calling announce).
  // Actual DDS discovery happens when FastDdsTransport::create_subscriber_endpoint()
  // creates the DataReader.
  HOLOSCAN_LOG_DEBUG("FastDdsDiscovery: Announced local subscriber on topic: {}", info.topic_name);
  return nvidia::gxf::Success;
}

nvidia::gxf::Expected<void> FastDdsDiscovery::remove_publisher(
    const nvidia::gxf::PublisherGid& gid) {
  if (!initialized_) {
    return nvidia::gxf::Unexpected(GXF_CONTRACT_INVALID_SEQUENCE);
  }

  // Local endpoint removal is handled by the shared TopicRegistry
  // (PubSubContext::unregister_publisher calls registry_.unregister_publisher).
  HOLOSCAN_LOG_DEBUG("FastDdsDiscovery: Removed local publisher: {}", gid.to_string());
  return nvidia::gxf::Success;
}

nvidia::gxf::Expected<void> FastDdsDiscovery::remove_subscriber(
    const nvidia::gxf::SubscriberGid& gid) {
  if (!initialized_) {
    return nvidia::gxf::Unexpected(GXF_CONTRACT_INVALID_SEQUENCE);
  }

  // Local endpoint removal is handled by the shared TopicRegistry
  // (PubSubContext::unregister_subscriber calls registry_.unregister_subscriber).
  HOLOSCAN_LOG_DEBUG("FastDdsDiscovery: Removed local subscriber: {}", gid.to_string());
  return nvidia::gxf::Success;
}

nvidia::gxf::Expected<std::vector<nvidia::gxf::PublisherInfo>> FastDdsDiscovery::query_publishers(
    const std::string& topic_name) {
  if (!initialized_) {
    return nvidia::gxf::Unexpected(GXF_CONTRACT_INVALID_SEQUENCE);
  }

  std::vector<nvidia::gxf::PublisherInfo> result;

  // Local publishers from the shared TopicRegistry (thread-safe internally)
  auto* registry = topic_registry();
  if (registry) {
    if (topic_name.empty()) {
      // Collect publishers from all topics
      for (const auto& topic : registry->get_all_topics()) {
        auto pubs = registry->get_publishers(topic);
        result.insert(result.end(), pubs.begin(), pubs.end());
      }
    } else {
      result = registry->get_publishers(topic_name);
    }
  }

  // Discovered remote publishers (protected by FastDdsDiscovery::mutex_)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [gid, info] : discovered_publishers_) {
      if (topic_name.empty() || info.topic_name == topic_name) {
        result.push_back(info);
      }
    }
  }

  return result;
}

nvidia::gxf::Expected<std::vector<nvidia::gxf::SubscriberInfo>> FastDdsDiscovery::query_subscribers(
    const std::string& topic_name) {
  if (!initialized_) {
    return nvidia::gxf::Unexpected(GXF_CONTRACT_INVALID_SEQUENCE);
  }

  std::vector<nvidia::gxf::SubscriberInfo> result;

  // Local subscribers from the shared TopicRegistry (thread-safe internally)
  auto* registry = topic_registry();
  if (registry) {
    if (topic_name.empty()) {
      // Collect subscribers from all topics
      for (const auto& topic : registry->get_all_topics()) {
        auto subs = registry->get_subscribers(topic);
        result.insert(result.end(), subs.begin(), subs.end());
      }
    } else {
      result = registry->get_subscribers(topic_name);
    }
  }

  // Discovered remote subscribers (protected by FastDdsDiscovery::mutex_)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [gid, info] : discovered_subscribers_) {
      if (topic_name.empty() || info.topic_name == topic_name) {
        result.push_back(info);
      }
    }
  }

  return result;
}

nvidia::gxf::Expected<std::vector<std::string>> FastDdsDiscovery::get_all_topics() {
  if (!initialized_) {
    return nvidia::gxf::Unexpected(GXF_CONTRACT_INVALID_SEQUENCE);
  }

  std::set<std::string> topics;

  // Local topics from the shared TopicRegistry (thread-safe internally)
  auto* registry = topic_registry();
  if (registry) {
    auto local_topics = registry->get_all_topics();
    topics.insert(local_topics.begin(), local_topics.end());
  }

  // Discovered remote topics (protected by FastDdsDiscovery::mutex_)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [gid, info] : discovered_publishers_) {
      topics.insert(info.topic_name);
    }
    for (const auto& [gid, info] : discovered_subscribers_) {
      topics.insert(info.topic_name);
    }
  }

  return std::vector<std::string>(topics.begin(), topics.end());
}

void FastDdsDiscovery::set_on_publisher_discovered(PublisherDiscoveredCallback callback) {
  on_publisher_discovered_ = std::move(callback);
}

void FastDdsDiscovery::set_on_subscriber_discovered(SubscriberDiscoveredCallback callback) {
  on_subscriber_discovered_ = std::move(callback);
}

void FastDdsDiscovery::set_on_publisher_lost(PublisherLostCallback callback) {
  on_publisher_lost_ = std::move(callback);
}

void FastDdsDiscovery::set_on_subscriber_lost(SubscriberLostCallback callback) {
  on_subscriber_lost_ = std::move(callback);
}

//------------------------------------------------------------------------------
// Helper Functions
//------------------------------------------------------------------------------

nvidia::gxf::Gid FastDdsDiscovery::guid_to_gid(const eprosima::fastdds::rtps::GUID_t& guid) {
  // GUID_t is 16 bytes: 12 bytes GuidPrefix + 4 bytes EntityId
  // Our Gid is also 16 bytes, so we do a direct copy
  std::array<uint8_t, nvidia::gxf::kGidSize> data;

  // Copy the GUID prefix (12 bytes)
  std::memcpy(data.data(), guid.guidPrefix.value, 12);

  // Copy the EntityId (4 bytes)
  std::memcpy(data.data() + 12, guid.entityId.value, 4);

  return nvidia::gxf::Gid(data);
}

//------------------------------------------------------------------------------
// Native Buffer Capability Serialization
//------------------------------------------------------------------------------
//
// UserData format (binary, little-endian). Version 1
//   [0..3]   magic "NBUF" (0x4655424E)
//   [4]      version (= 1)
//   [5]      num_protocols (uint8_t) — non-zero implies native buffer support
//   [6..]    for each protocol:
//              uint8_t name_len
//              <name bytes>
//   [..]     memory_domain_len (uint8_t)
//   [..]     memory_domain string
//   [..]     gpu_device_uuid (40 bytes, null-padded)
//   [..]     native_buffer_profile_len (uint8_t)
//   [..]     native_buffer_profile string
//   [..]     descriptor_format_version (uint8_t)
//   [..]     host_id_len (uint16_t)
//   [..]     host_id bytes (UTF-8, opaque; stable host id e.g. machine-id; length may be 0)

static constexpr uint32_t kNbufMagic = 0x4655424E;  // "NBUF" little-endian
static constexpr uint8_t kNbufVersion = 1;
static constexpr size_t kUuidFieldSize = 40;

std::vector<uint8_t> FastDdsDiscovery::serialize_native_capability(
    const nvidia::gxf::NativeBufferCapability& cap) {
  std::vector<uint8_t> result;
  result.reserve(128);

  auto append = [&result](const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    result.insert(result.end(), bytes, bytes + size);
  };

  append(&kNbufMagic, sizeof(kNbufMagic));
  append(&kNbufVersion, sizeof(kNbufVersion));

  uint8_t num_protocols =
      static_cast<uint8_t>(std::min(cap.native_buffer_protocols.size(), size_t{255}));
  append(&num_protocols, sizeof(num_protocols));

  for (uint8_t i = 0; i < num_protocols; ++i) {
    const auto& proto = cap.native_buffer_protocols[i];
    uint8_t name_len = static_cast<uint8_t>(std::min(proto.size(), size_t{255}));
    append(&name_len, sizeof(name_len));
    append(proto.data(), name_len);
  }

  // memory_domain
  uint8_t md_len = static_cast<uint8_t>(std::min(cap.memory_domain.size(), size_t{255}));
  append(&md_len, sizeof(md_len));
  append(cap.memory_domain.data(), md_len);

  // GPU UUID (40 bytes, null-padded)
  std::array<uint8_t, kUuidFieldSize> uuid_field{};
  std::memcpy(uuid_field.data(),
              cap.gpu_device_uuid.data(),
              std::min(cap.gpu_device_uuid.size(), kUuidFieldSize));
  append(uuid_field.data(), kUuidFieldSize);

  // native_buffer_profile
  uint8_t prof_len = static_cast<uint8_t>(std::min(cap.native_buffer_profile.size(), size_t{255}));
  append(&prof_len, sizeof(prof_len));
  append(cap.native_buffer_profile.data(), prof_len);

  // descriptor_format_version
  append(&cap.descriptor_format_version, sizeof(cap.descriptor_format_version));

  // host_id
  uint16_t host_len = static_cast<uint16_t>(std::min(cap.host_id.size(), size_t{65535}));
  append(&host_len, sizeof(host_len));
  if (host_len > 0) {
    append(cap.host_id.data(), host_len);
  }

  return result;
}

nvidia::gxf::NativeBufferCapability FastDdsDiscovery::parse_native_capability(
    const std::vector<uint8_t>& user_data) {
  nvidia::gxf::NativeBufferCapability cap;
  size_t offset = 0;

  auto read = [&user_data, &offset](void* dest, size_t size) -> bool {
    if (offset + size > user_data.size())
      return false;
    std::memcpy(dest, user_data.data() + offset, size);
    offset += size;
    return true;
  };

  uint32_t magic = 0;
  if (!read(&magic, sizeof(magic)) || magic != kNbufMagic)
    return cap;

  uint8_t version = 0;
  if (!read(&version, sizeof(version)))
    return cap;
  if (version != kNbufVersion)
    return cap;

  uint8_t num_protocols = 0;
  if (!read(&num_protocols, sizeof(num_protocols)))
    return cap;

  for (uint8_t i = 0; i < num_protocols; ++i) {
    uint8_t name_len = 0;
    if (!read(&name_len, sizeof(name_len)))
      return cap;
    if (offset + name_len > user_data.size())
      return cap;
    std::string proto(reinterpret_cast<const char*>(user_data.data() + offset), name_len);
    offset += name_len;
    cap.native_buffer_protocols.push_back(std::move(proto));
  }

  uint8_t md_len = 0;
  if (!read(&md_len, sizeof(md_len)))
    return cap;
  if (offset + md_len > user_data.size())
    return cap;
  cap.memory_domain = std::string(reinterpret_cast<const char*>(user_data.data() + offset), md_len);
  offset += md_len;

  std::array<uint8_t, kUuidFieldSize> uuid_field{};
  if (!read(uuid_field.data(), kUuidFieldSize))
    return cap;
  // Find null terminator or use full field
  size_t uuid_len = 0;
  while (uuid_len < kUuidFieldSize && uuid_field[uuid_len] != 0)
    uuid_len++;
  cap.gpu_device_uuid = std::string(reinterpret_cast<const char*>(uuid_field.data()), uuid_len);

  uint8_t prof_len = 0;
  if (!read(&prof_len, sizeof(prof_len)))
    return cap;
  if (offset + prof_len > user_data.size())
    return cap;
  cap.native_buffer_profile =
      std::string(reinterpret_cast<const char*>(user_data.data() + offset), prof_len);
  offset += prof_len;

  if (!read(&cap.descriptor_format_version, sizeof(cap.descriptor_format_version)))
    return cap;

  uint16_t host_len = 0;
  if (!read(&host_len, sizeof(host_len)))
    return cap;
  if (host_len > 0) {
    if (offset + host_len > user_data.size())
      return cap;
    cap.host_id = std::string(reinterpret_cast<const char*>(user_data.data() + offset), host_len);
    offset += host_len;
  }

  return cap;
}

nvidia::gxf::NativeBufferCapability FastDdsDiscovery::capability_for_device(int device_id) const {
  // Start from the default capability (carries host_id, protocols, profile, etc.)
  nvidia::gxf::NativeBufferCapability cap = local_native_capability_;

  auto gpu_info = nvidia::gxf::CudaDeviceIpcInfo::query(device_id);

  if (!gpu_info.cuda_ipc_supported) {
    HOLOSCAN_LOG_DEBUG(
        "FastDdsDiscovery::capability_for_device: device {} does not support CUDA IPC; "
        "removing cuda_ipc protocol",
        device_id);
    auto& protos = cap.native_buffer_protocols;
    protos.erase(std::remove(protos.begin(), protos.end(), "cuda_ipc"), protos.end());
    cap.gpu_device_uuid.clear();
    return cap;
  }

  cap.gpu_device_uuid = gpu_info.uuid;
  return cap;
}

}  // namespace holoscan
