/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "video_stream_serializer.hpp"

#include <endian.h>

#include <chrono>
#include <cinttypes>
#include <cstring>
#include <string>

namespace nvidia {
namespace holoscan {
namespace stream_playback {

namespace {

// Serializes EntityHeader
gxf::Expected<size_t> SerializeEntityHeader(VideoStreamSerializer::EntityHeader header,
                                            gxf::Endpoint* endpoint) {
  if (!endpoint) { return gxf::Unexpected{GXF_ARGUMENT_NULL}; }
  header.serialized_size = htole64(header.serialized_size);
  header.checksum = htole32(header.checksum);
  header.sequence_number = htole64(header.sequence_number);
  header.flags = htole32(header.flags);
  header.component_count = htole64(header.component_count);
  header.reserved = htole64(header.reserved);
  return endpoint->writeTrivialType(&header).substitute(sizeof(header));
}

// Deserializes EntityHeader
gxf::Expected<VideoStreamSerializer::EntityHeader> DeserializeEntityHeader(
    gxf::Endpoint* endpoint) {
  if (!endpoint) { return gxf::Unexpected{GXF_ARGUMENT_NULL}; }
  VideoStreamSerializer::EntityHeader header;
  return endpoint->readTrivialType(&header).and_then([&]() {
    header.serialized_size = le64toh(header.serialized_size);
    header.checksum = le32toh(header.checksum);
    header.sequence_number = le64toh(header.sequence_number);
    header.flags = le32toh(header.flags);
    header.component_count = le64toh(header.component_count);
    header.reserved = le64toh(header.reserved);
    return header;
  });
}

// Serializes ComponentHeader
gxf::Expected<size_t> SerializeComponentHeader(VideoStreamSerializer::ComponentHeader header,
                                               gxf::Endpoint* endpoint) {
  if (!endpoint) { return gxf::Unexpected{GXF_ARGUMENT_NULL}; }
  header.serialized_size = htole64(header.serialized_size);
  header.tid.hash1 = htole64(header.tid.hash1);
  header.tid.hash2 = htole64(header.tid.hash2);
  header.name_size = htole64(header.name_size);
  return endpoint->writeTrivialType(&header).substitute(sizeof(header));
}

// Deserializes ComponentHeader
gxf::Expected<VideoStreamSerializer::ComponentHeader> DeserializeComponentHeader(
    gxf::Endpoint* endpoint) {
  if (!endpoint) { return gxf::Unexpected{GXF_ARGUMENT_NULL}; }
  VideoStreamSerializer::ComponentHeader header;
  return endpoint->readTrivialType(&header).and_then([&]() {
    header.serialized_size = le64toh(header.serialized_size);
    header.tid.hash1 = le64toh(header.tid.hash1);
    header.tid.hash2 = le64toh(header.tid.hash2);
    header.name_size = le64toh(header.name_size);
    return header;
  });
}

}  // namespace

struct VideoStreamSerializer::ComponentEntry {
  ComponentHeader header = {0, GxfTidNull(), 0};
  gxf::UntypedHandle component = gxf::UntypedHandle::Null();
  gxf::Handle<gxf::ComponentSerializer> serializer = gxf::Handle<gxf::ComponentSerializer>::Null();
};

gxf_result_t VideoStreamSerializer::registerInterface(gxf::Registrar* registrar) {
  if (registrar == nullptr) { return GXF_ARGUMENT_NULL; }
  gxf::Expected<void> result;
  result &=
      registrar->parameter(component_serializers_, "component_serializers", "Component serializers",
                           "List of serializers for serializing and deserializing components");
  return gxf::ToResultCode(result);
}

gxf_result_t VideoStreamSerializer::serialize_entity_abi(gxf_uid_t eid, gxf::Endpoint* endpoint,
                                                         uint64_t* size) {
  if (endpoint == nullptr || size == nullptr) { return GXF_ARGUMENT_NULL; }
  FixedVector<gxf::UntypedHandle, kMaxComponents> components;
  FixedVector<ComponentEntry, kMaxComponents> entries;
  return gxf::ToResultCode(
      gxf::Entity::Shared(context(), eid)
          .map([&](gxf::Entity entity) { return entity.findAll(components); })
          .and_then([&]() { return createComponentEntries(components); })
          .assign_to(entries)
          .and_then([&]() {
            EntityHeader entity_header;
            entity_header.serialized_size = 0;  // How can we compute this before serializing?
            entity_header.checksum = 0x00000000;
            entity_header.sequence_number = outgoing_sequence_number_++;
            entity_header.flags = 0x00000000;
            entity_header.component_count = entries.size();
            entity_header.reserved = 0;
            return SerializeEntityHeader(entity_header, endpoint);
          })
          .assign_to(*size)
          .and_then([&]() { return serializeComponents(entries, endpoint); })
          .map([&](size_t serialized_size) { *size += serialized_size; }));
}

gxf::Expected<gxf::Entity> VideoStreamSerializer::deserialize_entity_header_abi(
    gxf::Endpoint* endpoint) {
  gxf::Entity entity;

  gxf_result_t result = gxf::ToResultCode(
      gxf::Entity::New(context())
          .assign_to(entity)
          .and_then([&]() { return DeserializeEntityHeader(endpoint); })
          .map([&](EntityHeader entity_header) {
            if (entity_header.sequence_number != incoming_sequence_number_) {
              incoming_sequence_number_ = entity_header.sequence_number;
            }
            incoming_sequence_number_++;
            return deserializeComponents(entity_header.component_count, entity, endpoint);
          })
          .substitute(entity));

  if (result != GXF_SUCCESS) { GXF_LOG_ERROR("Deserialize entity header failed"); }
  return entity;
}

gxf_result_t VideoStreamSerializer::deserialize_entity_abi(gxf_uid_t eid, gxf::Endpoint* endpoint) {
  if (endpoint == nullptr) { return GXF_ARGUMENT_NULL; }
  gxf::Entity entity;
  return gxf::ToResultCode(gxf::Entity::Shared(context(), eid)
                               .assign_to(entity)
                               .and_then([&]() { return DeserializeEntityHeader(endpoint); })
                               .map([&](EntityHeader entity_header) {
                                 if (entity_header.sequence_number != incoming_sequence_number_) {
                                   // Note:: This is a workaround for the issue that the frame count
                                   //        is out of the maximum frame index.
                                   //        Modified to support 'repeat' feature in
                                   //        nvidia::holoscan::stream_playback::VideoStreamReplayer
                                   //        which reuses gxf::EntityReplayer.
                                   //        When 'repeat' parameter is 'true' and the frame count
                                   //        is out of the maximum frame index, this error message
                                   //        is printed with nvidia::gxf::StdEntitySerializer but it
                                   //        is actually not a warning so we provide
                                   //        nvidia::holoscan::stream_playback::VideoStreamSerializer
                                   //        to replace nvidia::gxf::StdEntitySerializer and not to
                                   //        print this warning message.
                                   incoming_sequence_number_ = entity_header.sequence_number;
                                 }
                                 incoming_sequence_number_++;
                                 return deserializeComponents(entity_header.component_count, entity,
                                                              endpoint);
                               }));
}

gxf::Expected<FixedVector<VideoStreamSerializer::ComponentEntry, kMaxComponents>>
VideoStreamSerializer::createComponentEntries(
    const FixedVectorBase<gxf::UntypedHandle>& components) {
  FixedVector<ComponentEntry, kMaxComponents> entries;
  for (size_t i = 0; i < components.size(); i++) {
    const auto component = components[i];
    if (!component) { return gxf::Unexpected{GXF_ARGUMENT_OUT_OF_RANGE}; }

    // Check if component is serializable
    auto component_serializer = findComponentSerializer(component->tid());
    if (!component_serializer) {
      GXF_LOG_WARNING("No serializer found for component '%s' with type ID 0x%016zx%016zx",
                      component->name(), component->tid().hash1, component->tid().hash2);
      continue;
    }

    // Create component header
    ComponentHeader component_header;
    component_header.serialized_size = 0;  // How can we compute this before serializing?
    component_header.tid = component->tid();
    component_header.name_size = std::strlen(component->name());

    // Update component list
    const auto result =
        entries.emplace_back(component_header, component.value(), component_serializer.value());
    if (!result) { return gxf::Unexpected{GXF_EXCEEDING_PREALLOCATED_SIZE}; }
  }

  return entries;
}

gxf::Expected<size_t> VideoStreamSerializer::serializeComponents(
    const FixedVectorBase<ComponentEntry>& entries, gxf::Endpoint* endpoint) {
  size_t size = 0;
  for (size_t i = 0; i < entries.size(); i++) {
    const auto& entry = entries[i];
    if (!entry) { return gxf::Unexpected{GXF_ARGUMENT_OUT_OF_RANGE}; }
    const auto result =
        SerializeComponentHeader(entry->header, endpoint)
            .map([&](size_t component_header_size) { size += component_header_size; })
            .and_then(
                [&]() { return endpoint->write(entry->component.name(), entry->header.name_size); })
            .and_then([&]() { size += entry->header.name_size; })
            .and_then(
                [&]() { return entry->serializer->serializeComponent(entry->component, endpoint); })
            .map([&](size_t component_size) { size += component_size; });
    if (!result) { return gxf::ForwardError(result); }
  }
  return size;
}

gxf::Expected<void> VideoStreamSerializer::deserializeComponents(size_t component_count,
                                                                 gxf::Entity entity,
                                                                 gxf::Endpoint* endpoint) {
  for (size_t i = 0; i < component_count; i++) {
    ComponentEntry entry;
    const auto result =
        DeserializeComponentHeader(endpoint)
            .assign_to(entry.header)
            .and_then([&]() { return findComponentSerializer(entry.header.tid); })
            .assign_to(entry.serializer)
            .and_then([&]() -> gxf::Expected<std::string> {
              try {
                std::string name(entry.header.name_size, '\0');
                return gxf::ExpectedOrError(
                    endpoint->read(const_cast<char*>(name.data()), name.size()), name);
              } catch (const std::exception& exception) {
                GXF_LOG_ERROR("Failed to deserialize component name: %s", exception.what());
                return gxf::Unexpected{GXF_OUT_OF_MEMORY};
              }
            })
            .map([&](std::string name) { return entity.add(entry.header.tid, name.c_str()); })
            .assign_to(entry.component)
            .and_then([&]() {
              return entry.serializer->deserializeComponent(entry.component, endpoint);
            });
    if (!result) { return gxf::ForwardError(result); }
  }
  return gxf::Success;
}

gxf::Expected<gxf::Handle<gxf::ComponentSerializer>> VideoStreamSerializer::findComponentSerializer(
    gxf_tid_t tid) {
  // Search cache for valid serializer
  const auto search = serializer_cache_.find(tid);
  if (search != serializer_cache_.end()) { return search->second; }

  // Search serializer list for valid serializer and cache result
  for (size_t i = 0; i < component_serializers_.get().size(); i++) {
    const auto component_serializer = component_serializers_.get()[i];
    if (!component_serializer) { return gxf::Unexpected{GXF_ARGUMENT_OUT_OF_RANGE}; }
    if (component_serializer.value()->isSupported(tid)) {
      serializer_cache_[tid] = component_serializer.value();
      return component_serializer.value();
    }
  }

  return gxf::Unexpected{GXF_QUERY_NOT_FOUND};
}

}  // namespace stream_playback
}  // namespace holoscan
}  // namespace nvidia
