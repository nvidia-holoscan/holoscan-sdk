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

#ifndef HOLOSCAN_PUBSUB_FASTDDS_PUBSUB_FASTDDS_HOLOSCAN_ENTITY_TYPE_SUPPORT_HPP
#define HOLOSCAN_PUBSUB_FASTDDS_PUBSUB_FASTDDS_HOLOSCAN_ENTITY_TYPE_SUPPORT_HPP

#include <cstdint>
#include <string>
#include <vector>

#include <fastdds/dds/topic/TopicDataType.hpp>

namespace holoscan {

/**
 * @brief DDS data type for Holoscan pub/sub messages.
 *
 * This structure is used for two kinds of DDS messages:
 *
 * - **Main-topic messages**: `serialized_data` carries a CodecRegistry-serialized
 *   Holoscan Entity (which may reference GPU tensors). The metadata fields
 *   `contains_gpu_tensors` and `gpu_device_id` describe the entity payload.
 *   `descriptor_format_version` is 0 and `protocol_name` / `publisher_gid` are
 *   empty.
 *
 * - **Sidecar messages** (native buffer descriptors): `serialized_data` carries
 *   native-descriptor bytes produced by `FastDdsTransport::send_native_descriptor()`.
 *   `descriptor_format_version` (> 0) and `protocol_name` identify the wire
 *   format, and `publisher_gid` associates the descriptor with a main-topic
 *   publisher.
 *
 * Common metadata fields:
 * - **source_operator**: Tracing/debugging (identifies the publishing operator)
 * - **timestamp_ns**: Message timestamping (for latency measurement, ordering)
 */
struct HoloscanEntityData {
  /// Message payload bytes. For main-topic messages this is a
  /// CodecRegistry-serialized Holoscan Entity; for sidecar messages this is
  /// a native buffer descriptor (format identified by `descriptor_format_version`
  /// and `protocol_name`).
  std::vector<uint8_t> serialized_data;

  /// Name of the operator that produced this message
  std::string source_operator;

  /// Timestamp in nanoseconds (typically from std::chrono::steady_clock)
  int64_t timestamp_ns = 0;

  /// True if serialized_data contains GPU tensor references that need staging
  bool contains_gpu_tensors = false;

  /// GPU device ID where tensors originated (for device affinity)
  int32_t gpu_device_id = 0;

  /// GID of the main-topic publisher (hex string, e.g. from Gid::to_string()).
  /// Used by sidecar messages (native buffer descriptors) so the receiver can
  /// associate the descriptor with the correct publisher in its local registry.
  /// Empty for regular (main-topic) messages where the DDS writer GUID suffices.
  std::string publisher_gid;

  /// Native descriptor wire format version (0 for regular/main-topic payloads).
  uint8_t descriptor_format_version = 0;

  /// Native descriptor protocol identifier (empty for regular/main-topic payloads).
  std::string protocol_name;
};

/**
 * @brief DDS TopicDataType implementation for HoloscanEntityData.
 *
 * This class implements the FastDDS TopicDataType interface, enabling
 * HoloscanEntityData to be used as a DDS topic type. It handles:
 * - CDR serialization/deserialization using FastCDR
 * - Memory management for data instances
 * - Size calculation for DDS
 *
 * ## Usage
 *
 * ```cpp
 * // Register the type with a DomainParticipant
 * auto type_support = std::make_shared<FastDdsHoloscanEntityTypeSupport>();
 * type_support->register_type(participant);
 *
 * // Create a topic using this type
 * auto topic = participant->create_topic(
 *     "sensor_data",
 *     type_support->get_type_name(),
 *     TOPIC_QOS_DEFAULT);
 * ```
 *
 * ## Serialization Format
 *
 * The CDR serialization format is:
 * ```
 * [4 bytes: serialized_data length]
 * [N bytes: serialized_data content]
 * [4 bytes: source_operator length]
 * [M bytes: source_operator string]
 * [8 bytes: timestamp_ns (int64)]
 * [1 byte:  contains_gpu_tensors (bool)]
 * [4 bytes: gpu_device_id (int32)]
 * [4 bytes: publisher_gid length]
 * [P bytes: publisher_gid string]
 * [1 byte:  descriptor_format_version (uint8)]
 * [4 bytes: protocol_name length]
 * [Q bytes: protocol_name string]
 * ```
 */
class FastDdsHoloscanEntityTypeSupport : public eprosima::fastdds::dds::TopicDataType {
 public:
  /**
   * @brief Construct a new FastDdsHoloscanEntityTypeSupport.
   *
   * Sets the type name to "holoscan::Entity" and configures for variable-size data.
   *
   * Uses a 1MB initial max_serialized_type_size for payload pool allocation.
   * With FastDDS's default PREALLOCATED_WITH_REALLOC_MEMORY_MODE, larger messages
   * (10-100+ MB tensors) will trigger automatic reallocation. The actual size per
   * message is determined by calculate_serialized_size().
   *
   * @note FastDDS 3.4.2 Bug: Documentation says to use 0 for unbounded types, but
   *       this causes a null pointer crash in TopicPayloadPoolProxy::reserve_history().
   */
  FastDdsHoloscanEntityTypeSupport();

  /**
   * @brief Destructor.
   */
  ~FastDdsHoloscanEntityTypeSupport() override = default;

  // Delete copy operations (TopicDataType is typically not copied)
  FastDdsHoloscanEntityTypeSupport(const FastDdsHoloscanEntityTypeSupport&) = delete;
  FastDdsHoloscanEntityTypeSupport& operator=(const FastDdsHoloscanEntityTypeSupport&) = delete;

  // Allow move operations
  FastDdsHoloscanEntityTypeSupport(FastDdsHoloscanEntityTypeSupport&&) = default;
  FastDdsHoloscanEntityTypeSupport& operator=(FastDdsHoloscanEntityTypeSupport&&) = default;

  /**
   * @brief Serialize HoloscanEntityData to CDR format.
   *
   * @param data Pointer to HoloscanEntityData to serialize
   * @param payload Output serialized payload
   * @param data_representation Data representation (CDR encoding)
   * @return true on success, false on failure
   */
  bool serialize(const void* data, eprosima::fastdds::rtps::SerializedPayload_t& payload,
                 eprosima::fastdds::dds::DataRepresentationId_t data_representation) override;

  /**
   * @brief Deserialize CDR format to HoloscanEntityData.
   *
   * @param payload Input serialized payload
   * @param data Pointer to HoloscanEntityData to populate
   * @return true on success, false on failure
   */
  bool deserialize(eprosima::fastdds::rtps::SerializedPayload_t& payload, void* data) override;

  /**
   * @brief Get the serialized size of a HoloscanEntityData instance.
   *
   * @param data Pointer to HoloscanEntityData
   * @param data_representation Data representation (CDR encoding)
   * @return Size in bytes when serialized
   */
  uint32_t calculate_serialized_size(
      const void* data,
      eprosima::fastdds::dds::DataRepresentationId_t data_representation) override;

  /**
   * @brief Create a new HoloscanEntityData instance.
   *
   * @return Pointer to newly allocated HoloscanEntityData
   */
  void* create_data() override;

  /**
   * @brief Delete a HoloscanEntityData instance.
   *
   * @param data Pointer to HoloscanEntityData to delete
   */
  void delete_data(void* data) override;

  /**
   * @brief Check if this type has a key.
   *
   * HoloscanEntityData does not use keyed topics.
   *
   * @return false (no key defined)
   */
  bool compute_key(eprosima::fastdds::rtps::SerializedPayload_t& payload,
                   eprosima::fastdds::rtps::InstanceHandle_t& handle, bool force_md5) override;

  /**
   * @brief Get key from data.
   *
   * @param data Pointer to data
   * @param handle Output instance handle
   * @param force_md5 Force MD5 computation
   * @return false (no key defined)
   */
  bool compute_key(const void* data, eprosima::fastdds::rtps::InstanceHandle_t& handle,
                   bool force_md5) override;

  /**
   * @brief Check if this type is bounded.
   *
   * @return false (variable-size type with serialized_data vector)
   */
  inline bool is_bounded() const override { return false; }

  /**
   * @brief Check if this type is plain (all members have fixed size and no optional).
   *
   * @return false (contains variable-size vector and string)
   */
  inline bool is_plain(eprosima::fastdds::dds::DataRepresentationId_t) const override {
    return false;
  }
};

}  // namespace holoscan

#endif  // HOLOSCAN_PUBSUB_FASTDDS_PUBSUB_FASTDDS_HOLOSCAN_ENTITY_TYPE_SUPPORT_HPP
