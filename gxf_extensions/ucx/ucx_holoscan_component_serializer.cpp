/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ucx_holoscan_component_serializer.hpp"

#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include <gxf/ucx/ucx_serialization_buffer.hpp>

#include "holoscan/utils/timer.hpp"

namespace nvidia {
namespace gxf {

gxf_result_t UcxHoloscanComponentSerializer::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      allocator_, "allocator", "Memory allocator", "Memory allocator for tensor components");
  return ToResultCode(result);
}

gxf_result_t UcxHoloscanComponentSerializer::initialize() {
  // temporarily disabled due to missing common/endian.h header in the GXF package
  // if (!IsLittleEndian()) {
  //   GXF_LOG_WARNING(
  //       "UcxHoloscanComponentSerializer currently only supports little-endian devices");
  //   return GXF_NOT_IMPLEMENTED;
  // }
  return ToResultCode(configureSerializers() & configureDeserializers());
}

Expected<void> UcxHoloscanComponentSerializer::configureSerializers() {
  Expected<void> result;
  result &= setSerializer<holoscan::Message>([this](void* component, Endpoint* endpoint) {
    return serializeHoloscanMessage(*static_cast<holoscan::Message*>(component), endpoint);
  });
  result &=
      setSerializer<holoscan::MetadataDictionary>([this](void* component, Endpoint* endpoint) {
        return serializeMetadataDictionary(*static_cast<holoscan::MetadataDictionary*>(component),
                                           endpoint);
      });
  result &= setSerializer<holoscan::MessageLabel>([this](void* component, Endpoint* endpoint) {
    return serializeMessageLabel(*static_cast<holoscan::MessageLabel*>(component), endpoint);
  });
  return result;
}

Expected<void> UcxHoloscanComponentSerializer::configureDeserializers() {
  Expected<void> result;
  result &= setDeserializer<holoscan::Message>([this](void* component, Endpoint* endpoint) {
    return deserializeHoloscanMessage(endpoint).assign_to(
        *static_cast<holoscan::Message*>(component));
  });
  result &=
      setDeserializer<holoscan::MetadataDictionary>([this](void* component, Endpoint* endpoint) {
        return deserializeMetadataDictionary(endpoint).assign_to(
            *static_cast<holoscan::MetadataDictionary*>(component));
      });

  result &= setDeserializer<holoscan::MessageLabel>([this](void* component, Endpoint* endpoint) {
    return deserializeMessageLabel(endpoint).assign_to(
        *static_cast<holoscan::MessageLabel*>(component));
  });
  return result;
}

static inline Expected<size_t> serialize_string(const std::string& data, Endpoint* endpoint) {
  holoscan::ContiguousDataHeader header;
  header.size = data.size();
  header.bytes_per_element = header.size > 0 ? sizeof(data[0]) : 1;
  auto size = endpoint->writeTrivialType<holoscan::ContiguousDataHeader>(&header);
  if (!size) {
    return ForwardError(size);
  }
  auto size2 = endpoint->write(data.data(), header.size * header.bytes_per_element);
  if (!size2) {
    return ForwardError(size2);
  }
  return size.value() + size2.value();
}

static inline Expected<std::string> deserialize_string(Endpoint* endpoint) {
  holoscan::ContiguousDataHeader header;
  auto header_size = endpoint->readTrivialType<holoscan::ContiguousDataHeader>(&header);
  if (!header_size) {
    return ForwardError(header_size);
  }
  std::string data;
  data.resize(header.size);
  auto result = endpoint->read(data.data(), header.size * header.bytes_per_element);
  if (!result) {
    return ForwardError(result);
  }
  return data;
}

Expected<size_t> UcxHoloscanComponentSerializer::serializeHoloscanMessage(
    const holoscan::Message& message, Endpoint* endpoint) {
  GXF_LOG_DEBUG("UcxHoloscanComponentSerializer::serializeHoloscanMessage");

  // retrieve the name of the codec corresponding to the data in the Message
  auto index = std::type_index(message.value().type());
  auto& registry = holoscan::gxf::CodecRegistry::get_instance();
  auto maybe_name = registry.index_to_name(index);
  if (!maybe_name) {
    GXF_LOG_ERROR("No codec found for type_index with name: %s", index.name());
    return Unexpected{GXF_FAILURE};
  }
  std::string codec_name = maybe_name.value();

  // serialize the codec_name of the holoscan::Message codec to retrieve
  auto maybe_size = serialize_string(codec_name, endpoint);
  if (!maybe_size) {
    return ForwardError(maybe_size);
  }
  auto total_size = maybe_size.value();

  // serialize the message contents
  auto serialize_func = registry.get_serializer(codec_name);
  maybe_size = serialize_func(message, endpoint);
  if (!maybe_size) {
    return ForwardError(maybe_size);
  }
  total_size += maybe_size.value();
  return total_size;
}

Expected<holoscan::Message> UcxHoloscanComponentSerializer::deserializeHoloscanMessage(
    Endpoint* endpoint) {
  GXF_LOG_DEBUG("UcxHoloscanComponentSerializer::deserializeHoloscanMessage");

  // deserialize the type_name of the holoscan::Message codec to retrieve
  auto maybe_codec_name = deserialize_string(endpoint);
  if (!maybe_codec_name) {
    return ForwardError(maybe_codec_name);
  }

  // deserialize the message contents
  auto& registry = holoscan::gxf::CodecRegistry::get_instance();
  auto deserialize_func = registry.get_deserializer(maybe_codec_name.value());
  return deserialize_func(endpoint);
}

Expected<size_t> UcxHoloscanComponentSerializer::serializeMetadataDictionary(
    const holoscan::MetadataDictionary& metadata, Endpoint* endpoint) {
  GXF_LOG_DEBUG("UcxHoloscanComponentSerializer::serializeMetadataDictionary");

  // store the number of keys to expect
  uint32_t num_items = static_cast<uint32_t>(metadata.size());
  auto maybe_size = endpoint->writeTrivialType<uint32_t>(&num_items);
  if (!maybe_size) {
    return ForwardError(maybe_size);
  }
  size_t total_size = maybe_size.value();

  // serialize all items in the dictionary
  for (const auto& [key, value] : metadata) {
    // serialize the key
    auto maybe_size = serialize_string(key, endpoint);
    if (!maybe_size) {
      return ForwardError(maybe_size);
    }
    total_size += maybe_size.value();

    // serialize the holoscan::Message value
    maybe_size = serializeHoloscanMessage(*value, endpoint);
    if (!maybe_size) {
      return ForwardError(maybe_size);
    }
    total_size += maybe_size.value();
  }
  return total_size;
}

Expected<holoscan::MetadataDictionary>
UcxHoloscanComponentSerializer::deserializeMetadataDictionary(Endpoint* endpoint) {
  GXF_LOG_DEBUG("UcxHoloscanComponentSerializer::deserializeMetadataDictionary");
  holoscan::MetadataDictionary metadata{};

  // determine the size of the dictionary
  uint32_t num_items;
  auto size = endpoint->readTrivialType<uint32_t>(&num_items);

  // deserialize all items in the dictionary, storing them into metadata
  for (size_t i = 0; i < num_items; i++) {
    // deserialize the key
    auto maybe_key = deserialize_string(endpoint);
    if (!maybe_key) {
      return ForwardError(maybe_key);
    }

    // deserialize the value
    auto maybe_value = deserializeHoloscanMessage(endpoint);
    if (!maybe_value) {
      return ForwardError(maybe_value);
    }

    // store the deserialized item in the metadata dictionary
    metadata.set(maybe_key.value(), std::make_shared<holoscan::Message>(maybe_value.value()));
  }
  return metadata;
}

Expected<size_t> UcxHoloscanComponentSerializer::serializeMessageLabel(
    const holoscan::MessageLabel& messagelabel, Endpoint* endpoint) {
  GXF_LOG_DEBUG("UcxHoloscanComponentSerializer::serializeMessageLabel");
  size_t total_size = 0;

  // Get the total number of paths in message label and write it first
  int total_paths = messagelabel.num_paths();
  auto maybe_size = endpoint->writeTrivialType<int>(&total_paths);
  if (!maybe_size) {
    return ForwardError(maybe_size);
  }
  total_size += maybe_size.value();

  // for every path in message label, write the number of operators in the path first.
  // then, write all operator timestamps in a path
  for (const auto& path : messagelabel.paths()) {
    uint32_t num_operators = path.size();
    maybe_size = endpoint->writeTrivialType<uint32_t>(&num_operators);
    if (!maybe_size) {
      return ForwardError(maybe_size);
    }
    total_size += maybe_size.value();
    for (const auto& optimestamp : path) {
      maybe_size = serializeOperatorTimestampLabel(optimestamp, endpoint);
      if (!maybe_size) {
        return ForwardError(maybe_size);
      }
      total_size += maybe_size.value();
    }
  }
  // Check the total_size is less than the serialization buffer size (defaults to
  // holoscan::kDefaultSerializationBufferSize) - This is an UCX transfer limitation for non-tensor
  // data
  auto ucx_buf = dynamic_cast<nvidia::gxf::UcxSerializationBuffer*>(endpoint);
  if (!ucx_buf) {
    GXF_LOG_ERROR("Dynamic cast of Endpoint* to nvidia::gxf::UcxSerializationBuffer* failed");
    return Unexpected{GXF_FAILURE};
  }
  size_t buffer_capacity = ucx_buf->capacity();
  if (total_size > buffer_capacity) {
    GXF_LOG_ERROR(
        "MessageLabel size of %zu bytes exceeds the current UCX serialization buffer capacity of "
        "%zu bytes. You can try to increase the buffer capacity by setting the "
        "HOLOSCAN_UCX_SERIALIZATION_BUFFER_SIZE environment variable, but the limit it is "
        "possible to set will depend on the maximum header size supported by the underlying "
        "ucp_send_am_nbx function of UCX.",
        total_size,
        buffer_capacity);
    return Unexpected{GXF_FAILURE};
  }
  return total_size;
}

Expected<holoscan::MessageLabel> UcxHoloscanComponentSerializer::deserializeMessageLabel(
    Endpoint* endpoint) {
  GXF_LOG_DEBUG("UcxHoloscanComponentSerializer::deserializeHoloscanMessageLabel");
  holoscan::MessageLabel messagelabel;

  int total_paths;
  auto size = endpoint->readTrivialType<int>(&total_paths);
  if (!size) {
    return ForwardError(size);
  }
  for (int i = 0; i < total_paths; i++) {
    uint32_t num_operators;
    auto size = endpoint->readTrivialType<uint32_t>(&num_operators);
    if (!size) {
      return ForwardError(size);
    }
    holoscan::MessageLabel::TimestampedPath path;
    for (int j = 0; j < num_operators; j++) {
      auto maybe_optimestamp = deserializeOperatorTimestampLabel(endpoint);
      if (!maybe_optimestamp) {
        return ForwardError(maybe_optimestamp);
      }
      path.push_back(maybe_optimestamp.value());
    }
    messagelabel.add_new_path(path);
  }
  return messagelabel;
}

Expected<size_t> UcxHoloscanComponentSerializer::serializeOperatorTimestampLabel(
    const holoscan::OperatorTimestampLabel& operatortimestamplabel, Endpoint* endpoint) {
  GXF_LOG_DEBUG("UcxHoloscanComponentSerializer::serializeOperatorTimestampLabel");
  auto maybe_size = serialize_string(operatortimestamplabel.operator_name, endpoint);
  if (!maybe_size) {
    return ForwardError(maybe_size);
  }
  size_t total_size = maybe_size.value();
  maybe_size = endpoint->writeTrivialType<int64_t>(&operatortimestamplabel.rec_timestamp);
  if (!maybe_size) {
    return ForwardError(maybe_size);
  }
  total_size += maybe_size.value();
  maybe_size = endpoint->writeTrivialType<int64_t>(&operatortimestamplabel.pub_timestamp);
  if (!maybe_size) {
    return ForwardError(maybe_size);
  }
  total_size += maybe_size.value();
  return total_size;
}

Expected<holoscan::OperatorTimestampLabel>
UcxHoloscanComponentSerializer::deserializeOperatorTimestampLabel(Endpoint* endpoint) {
  GXF_LOG_DEBUG("UcxHoloscanComponentSerializer::deserializeOperatorTimestampLabel");
  holoscan::OperatorTimestampLabel operatortimestamplabel;

  auto maybe_operator_name = deserialize_string(endpoint);
  if (!maybe_operator_name) {
    return ForwardError(maybe_operator_name);
  }
  operatortimestamplabel.operator_name = maybe_operator_name.value();

  int64_t maybe_rec_timestamp;
  auto size = endpoint->readTrivialType<int64_t>(&maybe_rec_timestamp);
  if (!size) {
    return ForwardError(size);
  }
  operatortimestamplabel.rec_timestamp = maybe_rec_timestamp;

  int64_t maybe_pub_timestamp;
  size = endpoint->readTrivialType<int64_t>(&maybe_pub_timestamp);
  if (!size) {
    return ForwardError(size);
  }
  operatortimestamplabel.pub_timestamp = maybe_pub_timestamp;

  return operatortimestamplabel;
}

}  // namespace gxf
}  // namespace nvidia
