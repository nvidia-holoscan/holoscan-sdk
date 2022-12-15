/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_STREAM_PLAYBACK_VIDEO_STREAM_SERIALIZER_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_STREAM_PLAYBACK_VIDEO_STREAM_SERIALIZER_HPP_

#include <unordered_map>

#include "common/fixed_vector.hpp"
#include "gxf/serialization/component_serializer.hpp"
#include "gxf/serialization/entity_serializer.hpp"
#include "gxf/serialization/tid_hash.hpp"

namespace nvidia::holoscan::stream_playback {

/// @brief Data marshalling codelet for video stream entities.
///
/// Serializes and deserializes entities with the provided component serializers.
/// Little-endian is used over big-endian for better performance on x86 and arm platforms.
/// Entities are serialized in the following format:
///
///   | Entity Header || Component Header | Component Name | Component | ... | ... | ... |
///
/// Components will be serialized in the order they are added to the entity.
/// Components without serializers will be skipped.
/// Each component will be preceded by a component header and the name of the component.
/// The component itself will be serialized with a component serializer.
/// An entity header will be added at the beginning.
class VideoStreamSerializer : gxf::EntitySerializer {
 public:
#pragma pack(push, 1)
  // Header preceding entities
  struct EntityHeader {
    uint64_t serialized_size;  // Size of the serialized entity in bytes
    uint32_t checksum;         // Checksum to verify the integrity of the message
    uint64_t sequence_number;  // Sequence number of the message
    uint32_t flags;            // Flags to specify delivery options
    uint64_t component_count;  // Number of components in the entity
    uint64_t reserved;         // Bytes reserved for future use
  };
#pragma pack(pop)

#pragma pack(push, 1)
  // Header preceding components
  struct ComponentHeader {
    uint64_t serialized_size;  // Size of the serialized component in bytes
    gxf_tid_t tid;             // Type ID of the component
    uint64_t name_size;        // Size of the component name in bytes
  };
#pragma pack(pop)

  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t initialize() override { return GXF_SUCCESS; }
  gxf_result_t deinitialize() override { return GXF_SUCCESS; }

  gxf_result_t serialize_entity_abi(gxf_uid_t eid, gxf::Endpoint* endpoint,
                                    uint64_t* size) override;
  gxf_result_t deserialize_entity_abi(gxf_uid_t eid, gxf::Endpoint* endpoint) override;
  gxf::Expected<gxf::Entity> deserialize_entity_header_abi(gxf::Endpoint* endpoint) override;

 private:
  // Structure used to organize serializable components
  struct ComponentEntry;

  // Populates a list of component entries using a list of component handles
  gxf::Expected<FixedVector<ComponentEntry, kMaxComponents>> createComponentEntries(
      const FixedVectorBase<gxf::UntypedHandle>& components);
  // Serializes a list of components and writes them to an endpoint
  // Returns the total number of bytes serialized
  gxf::Expected<size_t> serializeComponents(const FixedVectorBase<ComponentEntry>& entries,
                                            gxf::Endpoint* endpoint);
  // Reads from an endpoint and deserializes a list of components
  gxf::Expected<void> deserializeComponents(size_t component_count, gxf::Entity entity,
                                            gxf::Endpoint* endpoint);
  // Searches for a component serializer that supports the given type ID
  // Uses the first valid serializer found and caches it for subsequent lookups
  // Returns an Unexpected if no valid serializer is found
  gxf::Expected<gxf::Handle<gxf::ComponentSerializer>> findComponentSerializer(gxf_tid_t tid);

  gxf::Parameter<FixedVector<gxf::Handle<gxf::ComponentSerializer>, kMaxComponents>>
      component_serializers_;

  // Table that caches type ID with a valid component serializer
  std::unordered_map<gxf_tid_t, gxf::Handle<gxf::ComponentSerializer>, gxf::TidHash>
      serializer_cache_;
  // Sequence number for outgoing messages
  uint64_t outgoing_sequence_number_;
  // Sequence number for incoming messages
  uint64_t incoming_sequence_number_;
};

}  // namespace nvidia::holoscan::stream_playback

#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_STREAM_PLAYBACK_VIDEO_STREAM_SERIALIZER_HPP_
