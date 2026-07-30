/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/pubsub/fastdds/resources/fastdds_pubsub_context.hpp>

#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/qos/DomainParticipantQos.hpp>
#include <fastdds/dds/topic/TypeSupport.hpp>
#include <fastdds/rtps/attributes/BuiltinTransports.hpp>
#include <fastdds/rtps/transport/UDPv4TransportDescriptor.hpp>
#include <fastdds/rtps/transport/shared_mem/SharedMemTransportDescriptor.hpp>
#include <gxf/pubsub/qos_profile.hpp>

#include <charconv>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/component_spec.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>
#include <holoscan/core/resources/gxf/unbounded_allocator.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/pubsub/common/pubsub_context_utils.hpp>
#include <holoscan/pubsub/fastdds/pubsub/fastdds_discovery.hpp>

namespace holoscan {

namespace {

// Parse "ip:port" string
std::pair<std::string, uint16_t> parse_peer_address(const std::string& peer) {
  auto colon_pos = peer.rfind(':');
  if (colon_pos == std::string::npos) {
    // No port specified, use DDS default discovery port
    return {peer, 7400};
  }
  std::string ip = peer.substr(0, colon_pos);
  const auto* port_begin = peer.data() + colon_pos + 1;
  const auto* port_end = peer.data() + peer.size();
  unsigned long port_val = 0;  // NOLINT(runtime/int)
  auto [ptr, ec] = std::from_chars(port_begin, port_end, port_val);
  if (ec != std::errc{} || ptr != port_end || port_val > 65535) {  // NOLINT(whitespace/braces)
    throw std::invalid_argument("parse_peer_address: invalid port in '" + peer + "'");
  }
  return {ip, static_cast<uint16_t>(port_val)};
}

}  // namespace

FastDdsPubSubContext::~FastDdsPubSubContext() {
  cleanup();
}

void FastDdsPubSubContext::setup(ComponentSpec& spec) {
  spec.param(domain_id_,
             "domain_id",
             "Domain ID",
             "DDS domain ID (participants in same domain can communicate)",
             static_cast<int32_t>(0));
  spec.param(participant_name_,
             "participant_name",
             "Participant Name",
             "Name for this DDS participant (used for debugging)",
             std::string("holoscan_participant"));
  spec.param(discovery_peers_,
             "discovery_peers",
             "Discovery Peers",
             "List of peer addresses for unicast discovery (e.g., '192.168.1.10:7400')",
             std::vector<std::string>{});
  spec.param(transport_profile_,
             "transport_profile",
             "Transport Profile",
             "Transport configuration: 'LARGE_DATA' (default, handles any size), "
             "'DEFAULT' (UDP+SHM 512KB limit), 'SHM_ONLY' (local only, lowest latency). "
             "Note: Transport applies to ALL topics on this participant.",
             std::string("LARGE_DATA"));
  spec.param(default_qos_profile_,
             "default_qos_profile",
             "Default QoS Profile",
             "Default QoS profile name for DDS endpoints, resolved via QoSProfile::from_name(). "
             "Options: 'default', 'control_message', 'tensor_data', 'video_stream', "
             "'sensor_data', 'reliable', 'transient_local', 'bulk_transfer'. Case-insensitive.",
             std::string("default"));
  spec.param(allocator_,
             "allocator",
             "Allocator",
             "Allocator for GPU staging buffers (RMMAllocator recommended, UnboundedAllocator for "
             "simplicity). If not provided, a default UnboundedAllocator is created.");
  spec.param(native_buffer_policy_str_,
             "native_buffer_policy",
             "Native Buffer Policy",
             "Native buffer (CUDA IPC) policy: 'disabled', 'preferred' (default), 'required'",
             std::string("preferred"));
  spec.param(native_buffer_acquire_timeout_ms_,
             "native_buffer_acquire_timeout_ms",
             "Native Buffer Acquire Timeout (ms)",
             "Timeout for CUDA IPC handle acquisition",
             static_cast<int64_t>(500));
  spec.param(native_buffer_export_ttl_ms_,
             "native_buffer_export_ttl_ms",
             "Native Buffer Export TTL (ms)",
             "Max age for stale export eviction",
             static_cast<int64_t>(5000));
}

void FastDdsPubSubContext::initialize() {
  // Call base class to bind parameters from args
  Resource::initialize();

  HOLOSCAN_LOG_DEBUG("FastDdsPubSubContext::initialize");

  // Create CUDA stream for async D2H/H2D copies
  cudaError_t cuda_err = cudaStreamCreate(&cuda_stream_);
  if (cuda_err != cudaSuccess) {
    throw std::runtime_error(std::string("FastDdsPubSubContext: failed to create CUDA stream: ") +
                             cudaGetErrorString(cuda_err));
  }

  // Use the provided allocator or create a default UnboundedAllocator
  if (!allocator_.has_value() || !allocator_.get()) {
    HOLOSCAN_LOG_WARN(
        "FastDdsPubSubContext: No allocator provided. Creating default UnboundedAllocator. "
        "Consider using RMMAllocator for better performance with GPU staging.");

    auto frag = fragment();
    if (!frag) {
      throw std::runtime_error("FastDdsPubSubContext: fragment not set while creating allocator");
    }
    allocator_ = frag->make_resource<UnboundedAllocator>("pubsub__dds_allocator");
    allocator_.get()->initialize();
  }

  host_id_ = load_stable_host_id();
  native_buffer_policy_enum_ = parse_native_buffer_policy(native_buffer_policy_str_.get());

  if (native_buffer_policy_enum_ != nvidia::gxf::NativeBufferPolicy::kDisabled) {
    auto gpu = query_gpu_device_info();
    gpu_device_id_ = gpu.device_id;
    gpu_device_uuid_ = gpu.device_uuid;
    cuda_ipc_supported_ = gpu.cuda_ipc_supported;
    if (!cuda_ipc_supported_ && !gpu_device_uuid_.empty()) {
      HOLOSCAN_LOG_WARN(
          "FastDdsPubSubContext: GPU device={} does not support CUDA IPC; "
          "native buffer will not advertise cuda_ipc protocol",
          gpu_device_id_);
    }
    HOLOSCAN_LOG_INFO("FastDdsPubSubContext: GPU device={}, UUID={}, CUDA IPC={}",
                      gpu_device_id_,
                      gpu_device_uuid_,
                      cuda_ipc_supported_ ? "yes" : "no");
  }

  // Configure DDS participant QoS
  using namespace eprosima::fastdds::dds;
  DomainParticipantQos pqos;
  pqos.name(participant_name_.get());

  // Configure transports (SHM for same-node efficiency, UDP for network)
  configure_transports(pqos);

  // Configure discovery peers (for non-multicast networks)
  if (!discovery_peers_.get().empty()) {
    configure_discovery_peers(pqos);
  }

  // Advertise native buffer capability in participant user data so remote
  // discovery listeners can classify endpoints for native-vs-byte routing.
  auto capability_bytes = FastDdsDiscovery::serialize_native_capability(native_buffer_capability());
  pqos.user_data().data_vec(capability_bytes);

  // Create participant - discovery starts automatically via SPDP/SEDP
  participant_ =
      DomainParticipantFactory::get_instance()->create_participant(domain_id_.get(), pqos);

  if (!participant_) {
    // Cleanup already created resources
    cleanup();
    throw std::runtime_error("FastDdsPubSubContext: failed to create DomainParticipant");
  }

  HOLOSCAN_LOG_INFO("FastDdsPubSubContext: DDS Participant created (domain={}, name='{}')",
                    domain_id_.get(),
                    participant_name_.get());
}

void FastDdsPubSubContext::configure_transports(
    eprosima::fastdds::dds::DomainParticipantQos& pqos) {
  using namespace eprosima::fastdds::rtps;

  // Segment size for large tensor payloads (128 MB supports up to ~100 MB tensors with overhead)
  constexpr uint32_t kLargeShmSegmentSize = 128 * 1024 * 1024;

  const std::string& profile = transport_profile_.get();

  if (profile == "LARGE_DATA") {
    // LARGE_DATA profile configures:
    // - SharedMemTransport (SHM) for same-host communication (default 8.5 MB segment)
    // - TCPv4 for reliable data transfer across hosts
    // - UDPv4 for discovery/metatraffic only
    pqos.setup_transports(BuiltinTransports::LARGE_DATA);

    // CRITICAL: The default LARGE_DATA SHM segment (8.5 MB) is NOT sufficient for
    // large tensor payloads (10-100 MB). We must increase it after setup_transports().
    for (auto& transport : pqos.transport().user_transports) {
      auto shm_transport = std::dynamic_pointer_cast<SharedMemTransportDescriptor>(transport);
      if (shm_transport) {
        shm_transport->segment_size(kLargeShmSegmentSize);
        HOLOSCAN_LOG_DEBUG("FastDdsPubSubContext: increased SHM segment size to {} MB",
                           kLargeShmSegmentSize / (1024 * 1024));
      }
    }
    HOLOSCAN_LOG_DEBUG("FastDdsPubSubContext: using LARGE_DATA transport profile (SHM+TCP+UDP)");

  } else if (profile == "SHM_ONLY") {
    // SHM_ONLY: Shared memory only for local communication
    // Lowest latency, but only works on same node
    pqos.transport().use_builtin_transports = false;
    auto shm_transport = std::make_shared<SharedMemTransportDescriptor>();
    shm_transport->segment_size(kLargeShmSegmentSize);
    pqos.transport().user_transports.push_back(shm_transport);
    HOLOSCAN_LOG_DEBUG("FastDdsPubSubContext: using SHM_ONLY transport profile ({} MB segments)",
                       kLargeShmSegmentSize / (1024 * 1024));

  } else {
    // DEFAULT: Built-in UDP + shared memory (512KB segment limit)
    // WARNING: Will fail for payloads > 512KB!
    if (profile != "DEFAULT") {
      HOLOSCAN_LOG_WARN("FastDdsPubSubContext: unknown transport_profile '{}', using DEFAULT",
                        profile);
    }
    pqos.transport().use_builtin_transports = true;
    HOLOSCAN_LOG_WARN(
        "FastDdsPubSubContext: using DEFAULT transport profile. "
        "WARNING: SHM segment limit is 512KB - large tensor payloads will fail! "
        "Use 'LARGE_DATA' profile for payloads > 512KB.");
  }
}

void FastDdsPubSubContext::configure_discovery_peers(
    eprosima::fastdds::dds::DomainParticipantQos& pqos) {
  using namespace eprosima::fastdds::rtps;
  using namespace eprosima::fastdds::dds;

  // For networks without multicast, configure explicit peer addresses
  auto& builtin = pqos.wire_protocol().builtin;
  builtin.discovery_config.discoveryProtocol = DiscoveryProtocol::SIMPLE;

  // Clear default multicast and unicast locators, then add explicit unicast peers
  builtin.metatrafficMulticastLocatorList.clear();
  builtin.metatrafficUnicastLocatorList.clear();

  for (const auto& peer : discovery_peers_.get()) {
    auto [ip, port] = parse_peer_address(peer);

    Locator_t locator;
    locator.kind = LOCATOR_KIND_UDPv4;
    locator.port = port;

    // Parse IPv4 address into bytes 12-15 of the locator address
    std::istringstream iss(ip);
    std::string octet;
    int i = 12;  // IPv4 address starts at byte 12 in the locator address
    bool valid = true;
    while (std::getline(iss, octet, '.') && i < 16) {
      if (octet.empty()) {
        valid = false;
        break;
      }
      unsigned long val = 0;  // NOLINT(runtime/int)
      const auto* begin = octet.data();
      const auto* end = octet.data() + octet.size();
      auto [ptr, ec] = std::from_chars(begin, end, val);
      if (ec != std::errc{} || ptr != end || val > 255) {  // NOLINT(whitespace/braces)
        valid = false;
        break;
      }
      locator.address[i++] = static_cast<unsigned char>(val);
    }
    if (!valid || i != 16) {
      HOLOSCAN_LOG_ERROR(
          "FastDdsPubSubContext: invalid IPv4 address '{}' in peer '{}', skipping", ip, peer);
      continue;
    }

    builtin.initialPeersList.push_back(locator);
  }

  HOLOSCAN_LOG_INFO("FastDdsPubSubContext: configured {} unicast discovery peers",
                    discovery_peers_.get().size());
}

void FastDdsPubSubContext::cleanup() {
  // Cleanup follows the recommended bottom-up deletion order:
  //   1. Topics managed by this context
  //   2. Any remaining contained entities (safety net)
  //   3. DomainParticipant itself
  //   4. CUDA resources
  //
  // Note: FastDdsTransport and FastDdsDiscovery should be shut down BEFORE this context
  // is destroyed. They clean up their own writers/readers/publisher/subscriber.

  if (participant_) {
    // Step 1: Delete topics that this context owns
    {
      std::lock_guard<std::mutex> lock(topics_mutex_);
      HOLOSCAN_LOG_DEBUG("FastDdsPubSubContext::cleanup: deleting {} topic(s)", topics_.size());
      for (auto& [name, topic] : topics_) {
        if (topic) {
          participant_->delete_topic(topic);
        }
      }
      topics_.clear();
    }

    // Step 2: Safety net - delete any remaining DDS entities that were not
    // cleaned up by FastDdsTransport/FastDdsDiscovery (e.g., if they crashed or were
    // destroyed out of order). delete_contained_entities() handles:
    //   - Remaining DataWriters/DataReaders → Publishers/Subscribers → Topics
    auto ret = participant_->delete_contained_entities();
    if (ret != eprosima::fastdds::dds::RETCODE_OK) {
      HOLOSCAN_LOG_WARN("FastDdsPubSubContext::cleanup: delete_contained_entities returned {}",
                        static_cast<int>(ret));
    }

    // Step 3: Delete participant
    HOLOSCAN_LOG_DEBUG("FastDdsPubSubContext::cleanup: deleting DomainParticipant");
    eprosima::fastdds::dds::DomainParticipantFactory::get_instance()->delete_participant(
        participant_);
    participant_ = nullptr;
  }

  // Note: allocator_ is managed externally (by the application), not cleaned up here

  // Step 4: Destroy CUDA stream
  if (cuda_stream_) {
    HOLOSCAN_LOG_DEBUG("FastDdsPubSubContext::cleanup: destroying CUDA stream");
    auto err = cudaStreamDestroy(cuda_stream_);
    if (err != cudaSuccess) {
      HOLOSCAN_LOG_WARN("FastDdsPubSubContext::cleanup: cudaStreamDestroy failed: {}",
                        cudaGetErrorString(err));
    }
    cuda_stream_ = nullptr;
  }
}

eprosima::fastdds::dds::Topic* FastDdsPubSubContext::get_or_create_topic(
    const std::string& topic_name, const std::string& type_name) {
  if (!participant_) {
    HOLOSCAN_LOG_ERROR("FastDdsPubSubContext::get_or_create_topic: participant not initialized");
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(topics_mutex_);

  // Check if topic already exists
  auto it = topics_.find(topic_name);
  if (it != topics_.end()) {
    return it->second;
  }

  // Create new topic
  auto* topic =
      participant_->create_topic(topic_name, type_name, eprosima::fastdds::dds::TOPIC_QOS_DEFAULT);

  if (!topic) {
    HOLOSCAN_LOG_ERROR(
        "FastDdsPubSubContext: failed to create topic '{}' with type '{}'", topic_name, type_name);
    return nullptr;
  }

  topics_[topic_name] = topic;
  HOLOSCAN_LOG_DEBUG(
      "FastDdsPubSubContext: created topic '{}' with type '{}'", topic_name, type_name);

  return topic;
}

std::vector<std::string> FastDdsPubSubContext::get_discovered_topics() const {
  std::vector<std::string> result;
  if (!participant_) {
    return result;
  }

  std::lock_guard<std::mutex> lock(topics_mutex_);
  result.reserve(topics_.size());
  for (const auto& [name, topic] : topics_) {
    result.push_back(name);
  }
  return result;
}

nvidia::gxf::QoSProfile FastDdsPubSubContext::default_qos() const {
  auto name = default_qos_profile_.get();
  if (name.empty()) {
    return nvidia::gxf::QoSProfile::Default();
  }
  auto result = nvidia::gxf::QoSProfile::from_name(name);
  if (!result) {
    HOLOSCAN_LOG_WARN("FastDdsPubSubContext: unknown QoS preset '{}', using Default", name);
    return nvidia::gxf::QoSProfile::Default();
  }
  return result.value();
}

int64_t FastDdsPubSubContext::native_buffer_acquire_timeout_ms() const {
  return native_buffer_acquire_timeout_ms_.get();
}

int64_t FastDdsPubSubContext::native_buffer_export_ttl_ms() const {
  return native_buffer_export_ttl_ms_.get();
}

nvidia::gxf::NativeBufferCapability FastDdsPubSubContext::native_buffer_capability() const {
  return build_native_buffer_capability(
      host_id_, gpu_device_uuid_, cuda_ipc_supported_, native_buffer_policy_enum_);
}

}  // namespace holoscan
