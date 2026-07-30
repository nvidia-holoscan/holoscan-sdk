/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/pubsub/fastdds/pubsub/fastdds_holoscan_entity_type_support.hpp>

#include <cstring>
#include <set>
#include <stdexcept>

#include <fastcdr/Cdr.h>
#include <fastcdr/FastBuffer.h>
#include <fastcdr/exceptions/BadParamException.h>
#include <fastcdr/exceptions/NotEnoughMemoryException.h>

#include <holoscan/logger/logger.hpp>

namespace holoscan {

// Type name registered with DDS
static constexpr char kTypeName[] = "holoscan::Entity";

// CDR header size (encapsulation header)
static constexpr uint32_t kCdrHeaderSize = 4;

FastDdsHoloscanEntityTypeSupport::FastDdsHoloscanEntityTypeSupport() {
  set_name(kTypeName);

  // Set initial max size for payload pool allocation.
  //
  // BUG WORKAROUND: FastDDS documentation says to use 0 for unbounded types, but
  // FastDDS 3.4.2 has a bug where TopicPayloadPool::get() returns nullptr when
  // payload_initial_size=0, causing a crash in TopicPayloadPoolProxy::reserve_history().
  //
  // Using 1 MB initial size with is_bounded()=false ensures:
  // - PREALLOCATED_WITH_REALLOC_MEMORY_MODE is used (allows automatic growth)
  // - Small messages use the 1 MB buffer efficiently (reused, not wasted)
  // - Large messages (10-100 MB) trigger automatic reallocation
  max_serialized_type_size = 1 * 1024 * 1024;  // 1 MB initial

  // No key defined for this type
  is_compute_key_provided = false;
}

bool FastDdsHoloscanEntityTypeSupport::serialize(
    const void* data, eprosima::fastdds::rtps::SerializedPayload_t& payload,
    eprosima::fastdds::dds::DataRepresentationId_t /* data_representation */) {
  if (data == nullptr) {
    HOLOSCAN_LOG_ERROR("FastDdsHoloscanEntityTypeSupport::serialize: null data pointer");
    return false;
  }

  const auto* entity_data = static_cast<const HoloscanEntityData*>(data);

  try {
    // Calculate required size
    uint32_t serialized_size = calculate_serialized_size(
        data, eprosima::fastdds::dds::DataRepresentationId_t::XCDR_DATA_REPRESENTATION);

    // Reserve space in payload if needed
    if (payload.max_size < serialized_size) {
      payload.reserve(serialized_size);
    }

    // Create FastCDR buffer and serializer
    eprosima::fastcdr::FastBuffer fastbuffer(reinterpret_cast<char*>(payload.data),
                                             payload.max_size);

    eprosima::fastcdr::Cdr set(
        fastbuffer, eprosima::fastcdr::Cdr::DEFAULT_ENDIAN, eprosima::fastcdr::CdrVersion::XCDRv1);

    // Write CDR encapsulation header
    set.serialize_encapsulation();

    // Serialize fields in order:
    // 1. serialized_data (std::vector<uint8_t>)
    set << entity_data->serialized_data;

    // 2. source_operator (std::string)
    set << entity_data->source_operator;

    // 3. timestamp_ns (int64_t)
    set << entity_data->timestamp_ns;

    // 4. contains_gpu_tensors (bool)
    set << entity_data->contains_gpu_tensors;

    // 5. gpu_device_id (int32_t)
    set << entity_data->gpu_device_id;

    // 6. publisher_gid (std::string)
    set << entity_data->publisher_gid;

    // 7. descriptor_format_version (uint8_t)
    set << entity_data->descriptor_format_version;

    // 8. protocol_name (std::string)
    set << entity_data->protocol_name;

    // Set actual serialized length
    payload.length = static_cast<uint32_t>(set.get_serialized_data_length());

    return true;
  } catch (const eprosima::fastcdr::exception::NotEnoughMemoryException& e) {
    HOLOSCAN_LOG_ERROR("FastDdsHoloscanEntityTypeSupport::serialize: not enough memory - {}",
                       e.what());
    return false;
  } catch (const eprosima::fastcdr::exception::BadParamException& e) {
    HOLOSCAN_LOG_ERROR("FastDdsHoloscanEntityTypeSupport::serialize: bad parameter - {}", e.what());
    return false;
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("FastDdsHoloscanEntityTypeSupport::serialize: exception - {}", e.what());
    return false;
  }
}

bool FastDdsHoloscanEntityTypeSupport::deserialize(
    eprosima::fastdds::rtps::SerializedPayload_t& payload, void* data) {
  if (data == nullptr) {
    HOLOSCAN_LOG_ERROR("FastDdsHoloscanEntityTypeSupport::deserialize: null data pointer");
    return false;
  }

  if (payload.data == nullptr || payload.length == 0) {
    HOLOSCAN_LOG_ERROR("FastDdsHoloscanEntityTypeSupport::deserialize: empty payload");
    return false;
  }

  auto* entity_data = static_cast<HoloscanEntityData*>(data);

  try {
    // Create FastCDR buffer and deserializer
    eprosima::fastcdr::FastBuffer fastbuffer(reinterpret_cast<char*>(payload.data), payload.length);

    eprosima::fastcdr::Cdr deser(
        fastbuffer, eprosima::fastcdr::Cdr::DEFAULT_ENDIAN, eprosima::fastcdr::CdrVersion::XCDRv1);

    // Read CDR encapsulation header
    deser.read_encapsulation();

    // Deserialize fields in order (must match serialize order):
    // 1. serialized_data (std::vector<uint8_t>)
    deser >> entity_data->serialized_data;

    // 2. source_operator (std::string)
    deser >> entity_data->source_operator;

    // 3. timestamp_ns (int64_t)
    deser >> entity_data->timestamp_ns;

    // 4. contains_gpu_tensors (bool)
    deser >> entity_data->contains_gpu_tensors;

    // 5. gpu_device_id (int32_t)
    deser >> entity_data->gpu_device_id;

    // 6. publisher_gid (std::string)
    deser >> entity_data->publisher_gid;

    // 7. descriptor_format_version (uint8_t)
    deser >> entity_data->descriptor_format_version;

    // 8. protocol_name (std::string)
    deser >> entity_data->protocol_name;

    return true;
  } catch (const eprosima::fastcdr::exception::NotEnoughMemoryException& e) {
    HOLOSCAN_LOG_ERROR("FastDdsHoloscanEntityTypeSupport::deserialize: not enough memory - {}",
                       e.what());
    return false;
  } catch (const eprosima::fastcdr::exception::BadParamException& e) {
    HOLOSCAN_LOG_ERROR("FastDdsHoloscanEntityTypeSupport::deserialize: bad parameter - {}",
                       e.what());
    return false;
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("FastDdsHoloscanEntityTypeSupport::deserialize: exception - {}", e.what());
    return false;
  }
}

uint32_t FastDdsHoloscanEntityTypeSupport::calculate_serialized_size(
    const void* data, eprosima::fastdds::dds::DataRepresentationId_t /* data_representation */) {
  if (data == nullptr) {
    // Return minimum size for empty data (header + minimum field sizes + alignment)
    return 64;
  }

  const auto* entity_data = static_cast<const HoloscanEntityData*>(data);

  // Calculate size with conservative alignment estimates.
  // CDR alignment rules can add padding between fields, so we add extra space
  // to ensure the buffer is always large enough.
  //
  // Fields:
  // - CDR encapsulation header: 4 bytes
  // - serialized_data: 4 (length) + N (data)
  // - source_operator: 4 (length) + M (string) + 1 (null terminator)
  // - timestamp_ns: 8 bytes (int64, may need alignment padding)
  // - contains_gpu_tensors: 1 byte (bool)
  // - gpu_device_id: 4 bytes (int32, may need alignment padding)
  // - publisher_gid: 4 (length) + P (string) + 1 (null terminator)
  // - descriptor_format_version: 1 byte
  // - protocol_name: 4 (length) + Q (string) + 1 (null terminator)

  uint32_t size = kCdrHeaderSize;  // 4 bytes

  // serialized_data: 4-byte length prefix + data
  // Add 4 bytes alignment padding (worst case)
  size += 4 + static_cast<uint32_t>(entity_data->serialized_data.size()) + 4;

  // source_operator: 4-byte length prefix + string + null terminator
  // Add 4 bytes alignment padding (worst case)
  size += 4 + static_cast<uint32_t>(entity_data->source_operator.size()) + 1 + 4;

  // timestamp_ns: 8 bytes + up to 7 bytes alignment padding
  size += 8 + 7;

  // contains_gpu_tensors: 1 byte
  size += 1;

  // gpu_device_id: 4 bytes + up to 3 bytes alignment padding
  size += 4 + 3;

  // publisher_gid: 4-byte length prefix + string + null terminator
  // Add 4 bytes alignment padding (worst case)
  size += 4 + static_cast<uint32_t>(entity_data->publisher_gid.size()) + 1 + 4;

  // descriptor_format_version: 1 byte
  size += 1;

  // protocol_name: 4-byte length prefix + string + null terminator
  // Add 4 bytes alignment padding (worst case)
  size += 4 + static_cast<uint32_t>(entity_data->protocol_name.size()) + 1 + 4;

  return size;
}

void* FastDdsHoloscanEntityTypeSupport::create_data() {
  return new HoloscanEntityData();
}

void FastDdsHoloscanEntityTypeSupport::delete_data(void* data) {
  delete static_cast<HoloscanEntityData*>(data);
}

bool FastDdsHoloscanEntityTypeSupport::compute_key(
    eprosima::fastdds::rtps::SerializedPayload_t& /* payload */,
    eprosima::fastdds::rtps::InstanceHandle_t& /* handle */, bool /* force_md5 */) {
  // No key defined for this type
  return false;
}

bool FastDdsHoloscanEntityTypeSupport::compute_key(
    const void* /* data */, eprosima::fastdds::rtps::InstanceHandle_t& /* handle */,
    bool /* force_md5 */) {
  // No key defined for this type
  return false;
}

}  // namespace holoscan
