/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <utility>

namespace nvidia {
namespace gxf {

// copy of TensorHeader from UcxComponentSerializer needed for GXFTensor serialize/deserialize
namespace {

#pragma pack(push, 1)
struct TensorHeader {
  MemoryStorageType storage_type;     // CPU or GPU tensor
  PrimitiveType element_type;         // Tensor element type
  uint64_t bytes_per_element;         // Bytes per tensor element
  uint32_t rank;                      // Tensor rank
  int32_t dims[Shape::kMaxRank];      // Tensor dimensions
  uint64_t strides[Shape::kMaxRank];  // Tensor strides
};
#pragma pack(pop)

}  // namespace

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
  result &= setSerializer<holoscan::gxf::GXFTensor>([this](void* component, Endpoint* endpoint) {
    return serializeHoloscanGXFTensor(*static_cast<holoscan::gxf::GXFTensor*>(component), endpoint);
  });
  result &= setSerializer<holoscan::Message>([this](void* component, Endpoint* endpoint) {
    return serializeHoloscanMessage(*static_cast<holoscan::Message*>(component), endpoint);
  });
  return result;
}

Expected<void> UcxHoloscanComponentSerializer::configureDeserializers() {
  Expected<void> result;
  result &= setDeserializer<holoscan::gxf::GXFTensor>([this](void* component, Endpoint* endpoint) {
    return deserializeHoloscanGXFTensor(endpoint).assign_to(
        *static_cast<holoscan::gxf::GXFTensor*>(component));
  });
  result &= setDeserializer<holoscan::Message>([this](void* component, Endpoint* endpoint) {
    return deserializeHoloscanMessage(endpoint).assign_to(
        *static_cast<holoscan::Message*>(component));
  });
  return result;
}

Expected<size_t> UcxHoloscanComponentSerializer::serializeHoloscanGXFTensor(
    const holoscan::gxf::GXFTensor& tensor, Endpoint* endpoint) {
  GXF_LOG_DEBUG("UcxHoloscanComponentSerializer::serializeHoloscanGXFTensor");
  // Implementation matches UcxComponentSerializer::serializeTensor since holoscan::gxf::Tensor
  // inherits from nvidia::gxf::Tensor.
  TensorHeader header;
  header.storage_type = tensor.storage_type();
  header.element_type = tensor.element_type();
  header.bytes_per_element = tensor.bytes_per_element();
  header.rank = tensor.rank();
  for (size_t i = 0; i < Shape::kMaxRank; i++) {
    header.dims[i] = tensor.shape().dimension(i);
    header.strides[i] = tensor.stride(i);
  }
  auto result = endpoint->write_ptr(tensor.pointer(), tensor.size(), tensor.storage_type());
  if (!result) { return ForwardError(result); }
  auto size = endpoint->writeTrivialType<TensorHeader>(&header);
  if (!size) { return ForwardError(size); }
  return sizeof(header);
}

Expected<holoscan::gxf::GXFTensor> UcxHoloscanComponentSerializer::deserializeHoloscanGXFTensor(
    Endpoint* endpoint) {
  GXF_LOG_DEBUG("UcxHoloscanComponentSerializer::deserializeHoloscanGXFTensor");
  // Implementation is as in UcxComponentSerializer::deserializeTensor is private, but with an
  // additional conversion to GXFTensor at the end.
  if (!endpoint) { return Unexpected{GXF_ARGUMENT_NULL}; }

  TensorHeader header;
  auto size = endpoint->readTrivialType<TensorHeader>(&header);
  if (!size) { return ForwardError(size); }

  std::array<int32_t, Shape::kMaxRank> dims;
  std::memcpy(dims.data(), header.dims, sizeof(header.dims));
  Tensor::stride_array_t strides;
  std::memcpy(strides.data(), header.strides, sizeof(header.strides));

  Tensor tensor;
  auto result = tensor.reshapeCustom(Shape(dims, header.rank),
                                     header.element_type,
                                     header.bytes_per_element,
                                     strides,
                                     header.storage_type,
                                     allocator_);
  if (!result) { return ForwardError(result); }
  result = endpoint->write_ptr(tensor.pointer(), tensor.size(), tensor.storage_type());
  if (!result) { return ForwardError(result); }
  return holoscan::gxf::GXFTensor(tensor);
}

Expected<size_t> UcxHoloscanComponentSerializer::serializeHoloscanMessage(
    const holoscan::Message& message, Endpoint* endpoint) {
  GXF_LOG_DEBUG("UcxHoloscanComponentSerializer::serializeHoloscanMessage");

  // retrieve the name of the codec corresponding to the data in the Message
  auto index = std::type_index(message.value().type());
  auto& registry = holoscan::CodecRegistry::get_instance();
  auto maybe_name = registry.index_to_name(index);
  if (!maybe_name) {
    GXF_LOG_ERROR("No codec found for type_index with name: %s", index.name());
    return Unexpected{GXF_FAILURE};
  }
  std::string codec_name = maybe_name.value();

  // serialize the codec_name of the holoscan::Message codec to retrieve
  holoscan::ContiguousDataHeader header;
  header.size = codec_name.size();
  header.bytes_per_element = header.size > 0 ? sizeof(codec_name[0]) : 1;
  size_t total_size = 0;
  auto maybe_size = endpoint->writeTrivialType<holoscan::ContiguousDataHeader>(&header);
  if (!maybe_size) { return ForwardError(maybe_size); }
  total_size += maybe_size.value();
  maybe_size = endpoint->write(codec_name.data(), header.size * header.bytes_per_element);
  if (!maybe_size) { return ForwardError(maybe_size); }
  total_size += maybe_size.value();

  // serialize the message contents
  auto serialize_func = registry.get_serializer(codec_name);
  maybe_size = serialize_func(message, endpoint);
  if (!maybe_size) { return ForwardError(maybe_size); }
  total_size += maybe_size.value();
  return total_size;
}

Expected<holoscan::Message> UcxHoloscanComponentSerializer::deserializeHoloscanMessage(
    Endpoint* endpoint) {
  GXF_LOG_DEBUG("UcxHoloscanComponentSerializer::deserializeHoloscanMessage");

  // deserialize the type_name of the holoscan::Message codec to retrieve
  holoscan::ContiguousDataHeader header;
  auto header_size = endpoint->readTrivialType<holoscan::ContiguousDataHeader>(&header);
  if (!header_size) { return ForwardError(header_size); }
  std::string codec_name;
  codec_name.resize(header.size);
  auto result = endpoint->read(codec_name.data(), header.size * header.bytes_per_element);
  if (!result) { return ForwardError(result); }

  // deserialize the message contents
  auto& registry = holoscan::CodecRegistry::get_instance();
  auto deserialize_func = registry.get_deserializer(codec_name);
  return deserialize_func(endpoint);
}
}  // namespace gxf
}  // namespace nvidia
