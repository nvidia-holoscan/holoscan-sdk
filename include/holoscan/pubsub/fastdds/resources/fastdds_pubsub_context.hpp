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

#ifndef HOLOSCAN_PUBSUB_FASTDDS_RESOURCES_FASTDDS_PUBSUB_CONTEXT_HPP
#define HOLOSCAN_PUBSUB_FASTDDS_RESOURCES_FASTDDS_PUBSUB_CONTEXT_HPP

#include <cuda_runtime.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/topic/Topic.hpp>

#include <gxf/pubsub/endpoint_info.hpp>
#include <gxf/pubsub/pubsub_context.hpp>
#include <gxf/pubsub/qos_profile.hpp>

#include <holoscan/core/resource.hpp>

namespace holoscan {

// Forward declarations
class Allocator;

/**
 * @brief Resource managing DDS participant and GPU staging for pub/sub.
 *
 * FastDdsPubSubContext provides:
 * - DDS DomainParticipant lifecycle management
 * - Automatic discovery via SPDP/SEDP (no central registry needed)
 * - Allocator for GPU tensor staging (RMMAllocator or UnboundedAllocator)
 * - CUDA stream for async memory copies
 *
 * ## Discovery Model
 *
 * DDS discovery is automatic:
 * - Participants announce themselves via multicast (or configured unicast peers)
 * - SPDP discovers participants, SEDP discovers topics/endpoints
 * - No startup ordering required - fragments can start in any order
 *
 * ## GPU Staging Allocator
 *
 * For GPU tensor staging (D2H before send, H2D after receive), FastDdsPubSubContext
 * uses an existing Holoscan allocator resource:
 * - **RMMAllocator** (recommended): Provides memory pools for pinned host memory
 *   and device memory. Efficient for repeated allocations of similar sizes.
 * - **UnboundedAllocator** (simple): Dynamic allocation each time without pooling.
 *   Simpler but less performant - suitable for development/testing.
 *
 * ## Usage
 *
 * ```cpp
 * // In Application::compose()
 * auto allocator = make_resource<RMMAllocator>("staging_allocator",
 *     Arg("host_memory_initial_size", "64MB"),
 *     Arg("host_memory_max_size", "256MB"));
 *
 * auto dds_ctx = make_resource<FastDdsPubSubContext>("dds_context",
 *     Arg("domain_id", 0),
 *     Arg("allocator", allocator));
 *
 * // PubSubTransmitter/PubSubReceiver use this context for DDS transport
 * spec.output<Tensor>("out")
 *     .connector(ConnectorType::kPubSub,
 *                Arg("topic", "/sensors/camera"),
 *                Arg("pubsub_context", dds_ctx));
 * ```
 *
 * ==Parameters==
 *
 * - **domain_id** (int32_t, optional): DDS domain ID (default: 0). All participants
 *   in the same domain can communicate.
 * - **participant_name** (std::string, optional): Name for this participant, used
 *   for debugging and discovery (default: "holoscan_participant").
 * - **discovery_peers** (std::vector<std::string>, optional): List of peer addresses
 *   for unicast discovery (e.g., "192.168.1.10:7400"). If empty, uses multicast.
 * - **transport_profile** (std::string, optional): Transport configuration profile:
 *   - "LARGE_DATA" (default): SHM (128MB) + TCP + UDP discovery. Recommended for
 *     mixed payload sizes (1KB to 100MB). Works across network boundaries.
 *   - "SHM_ONLY": Shared memory only (128MB segments). Local-only, lowest latency.
 *   - "DEFAULT": Built-in UDP + SHM (512KB limit). WARNING: Fails for payloads > 512KB!
 *   Note: Transport is configured per-participant and applies to ALL topics.
 * - **default_qos_profile** (std::string, optional): Default QoS profile name for endpoints.
 *   Resolved at runtime via `QoSProfile::from_name()` (case-insensitive).
 *   Used by FastDdsTransport when no per-topic QoS is specified. Options:
 *   - "default" (default): Best-effort, volatile, keep-last(10). General-purpose.
 *   - "control_message": Reliable, transient-local, keep-all(100). Small control msgs.
 *   - "tensor_data": Reliable, volatile, keep-last(3). Large tensor payloads.
 *   - "video_stream": Best-effort, volatile, keep-last(1). High-frequency video.
 *   See `QoSProfile::preset_names()` for the full list.
 * - **allocator** (std::shared_ptr<Allocator>, optional): Allocator for GPU staging
 *   buffers. RMMAllocator recommended for production. If not provided, a default
 *   UnboundedAllocator is created.
 */
class FastDdsPubSubContext : public Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS(FastDdsPubSubContext)
  FastDdsPubSubContext() = default;
  ~FastDdsPubSubContext() override;

  void setup(ComponentSpec& spec) override;
  void initialize() override;

  /**
   * @brief Get the DDS DomainParticipant.
   * @return Pointer to the participant, or nullptr if not initialized.
   */
  eprosima::fastdds::dds::DomainParticipant* participant() const { return participant_; }

  /**
   * @brief Get the allocator for GPU staging.
   *
   * This allocator (typically RMMAllocator or UnboundedAllocator) is used for
   * allocating pinned host memory buffers for GPU D2H/H2D staging operations.
   *
   * @return Shared pointer to the allocator.
   */
  std::shared_ptr<Allocator> allocator() const { return allocator_; }

  /**
   * @brief Get the CUDA stream for staging operations.
   * @return The CUDA stream handle.
   */
  cudaStream_t cuda_stream() const { return cuda_stream_; }

  /**
   * @brief Get or create a DDS topic.
   *
   * Topics are cached and reused within this participant.
   *
   * @param topic_name The topic name (e.g., "/sensors/camera").
   * @param type_name The registered type name (e.g., "holoscan::Entity").
   * @return Pointer to the Topic, or nullptr on failure.
   */
  eprosima::fastdds::dds::Topic* get_or_create_topic(const std::string& topic_name,
                                                     const std::string& type_name);

  /**
   * @brief Get list of discovered topics.
   * @return Vector of topic names currently discovered by this participant.
   */
  std::vector<std::string> get_discovered_topics() const;

  /**
   * @brief Get the DDS domain ID.
   * @return The domain ID.
   */
  int32_t domain_id() const { return domain_id_.get(); }

  /**
   * @brief Get the participant name.
   * @return The participant name.
   */
  std::string participant_name() const { return participant_name_.get(); }

  /**
   * @brief Get the transport profile.
   * @return The transport profile name ("LARGE_DATA", "DEFAULT", or "SHM_ONLY").
   */
  std::string transport_profile() const { return transport_profile_.get(); }

  /**
   * @brief Get the default QoS profile for endpoints.
   *
   * Resolves the `default_qos_profile` string parameter (from YAML or Arg)
   * into a `QoSProfile` struct using `QoSProfile::from_name()`. Falls back
   * to `QoSProfile::Default()` if the name is empty or unrecognized.
   *
   * Used by FastDdsTransport when no per-topic QoS is specified.
   *
   * @return The resolved QoSProfile struct.
   */
  nvidia::gxf::QoSProfile default_qos() const;

  //----------------------------------------------------------------------------
  // Native Buffer (CUDA IPC) Configuration
  //----------------------------------------------------------------------------

  /**
   * @brief Get the native buffer policy.
   */
  nvidia::gxf::NativeBufferPolicy native_buffer_policy() const {
    return native_buffer_policy_enum_;
  }

  /**
   * @brief Get the CUDA IPC handle acquisition timeout (ms).
   */
  int64_t native_buffer_acquire_timeout_ms() const;

  /**
   * @brief Get the stale export TTL (ms).
   */
  int64_t native_buffer_export_ttl_ms() const;

  /**
   * @brief Get the GPU device UUID string.
   */
  const std::string& gpu_device_uuid() const { return gpu_device_uuid_; }

  /**
   * @brief Stable host identifier (e.g. Linux /etc/machine-id) for discovery.
   */
  const std::string& host_id() const { return host_id_; }

  /**
   * @brief Get the GPU device ID.
   */
  int32_t gpu_device_id() const { return gpu_device_id_; }

  /**
   * @brief Get the local native buffer capability for discovery advertisement.
   */
  nvidia::gxf::NativeBufferCapability native_buffer_capability() const;

 private:
  void configure_transports(eprosima::fastdds::dds::DomainParticipantQos& pqos);
  void configure_discovery_peers(eprosima::fastdds::dds::DomainParticipantQos& pqos);
  void cleanup();

  // Parameters
  Parameter<int32_t> domain_id_;
  Parameter<std::string> participant_name_;
  Parameter<std::vector<std::string>> discovery_peers_;
  Parameter<std::string> transport_profile_;
  Parameter<std::string> default_qos_profile_;
  Parameter<std::shared_ptr<Allocator>> allocator_;

  // Native buffer parameters
  Parameter<std::string> native_buffer_policy_str_;
  Parameter<int64_t> native_buffer_acquire_timeout_ms_;
  Parameter<int64_t> native_buffer_export_ttl_ms_;

  // DDS resources
  eprosima::fastdds::dds::DomainParticipant* participant_ = nullptr;
  std::unordered_map<std::string, eprosima::fastdds::dds::Topic*> topics_;
  mutable std::mutex topics_mutex_;

  // GPU staging resources
  cudaStream_t cuda_stream_ = nullptr;

  // Native buffer state
  nvidia::gxf::NativeBufferPolicy native_buffer_policy_enum_{
      nvidia::gxf::NativeBufferPolicy::kPreferred};
  std::string gpu_device_uuid_;
  int32_t gpu_device_id_{0};
  bool cuda_ipc_supported_{false};
  /// Populated at initialize() from HOLOSCAN_HOST_ID (if set) or /etc/machine-id.
  std::string host_id_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_PUBSUB_FASTDDS_RESOURCES_FASTDDS_PUBSUB_CONTEXT_HPP */
