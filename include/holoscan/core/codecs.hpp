/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_CODECS_HPP
#define HOLOSCAN_CORE_CODECS_HPP

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "./common.hpp"
#include "./endpoint.hpp"
#include "./errors.hpp"
#include "./expected.hpp"
#include "./type_traits.hpp"

// Note: Currently the GXF UCX extension transmits using little-endian byte order. All hardware
//       supported by Holoscan SDK is also little endian. To support big endian platforms, we
//       would need to update the codecs in this file to properly handle endianness.

namespace holoscan {

// codec template struct so users can define their own codec class
//    i.e. following the design of yaml-cpp's YAML::convert
template <typename T>
struct codec;

//////////////////////////////////////////////////////////////////////////////////////////////////
// Codec type 1: trivial binary types (float, etc.)
//
// Types that can be serialized by writing binary data where the number
// of bytes to be written is based on sizeof(typeT).

template <typename typeT>
static inline expected<size_t, RuntimeError> serialize_trivial_type(const typeT& value,
                                                                    Endpoint* endpoint) {
  return endpoint->write_trivial_type(&value);
}

template <typename typeT>
static inline expected<typeT, RuntimeError> deserialize_trivial_type(Endpoint* endpoint) {
  typeT encoded{};
  auto maybe_value = endpoint->read_trivial_type(&encoded);
  if (!maybe_value) {
    return forward_error(maybe_value);
  }
  if (maybe_value.value() != sizeof(typeT)) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "short trivial type"));
  }
  return encoded;
}

// TODO(unknown): currently not handling integer types separately
template <typename typeT>
struct codec {
  static expected<size_t, RuntimeError> serialize(const typeT& value, Endpoint* endpoint) {
    return serialize_trivial_type<typeT>(value, endpoint);
  }
  static expected<typeT, RuntimeError> deserialize(Endpoint* endpoint) {
    return deserialize_trivial_type<typeT>(endpoint);
  }
};

//////////////////////////////////////////////////////////////////////////////////////////////////
// Codec type 3: basic container types
//
// For vectorT types storing multiple items contiguously in memory
// (e.g. std::vector, std::array, std::string).
//
// It requires that vectorT have methods size(), data(), and operator[]

#pragma pack(push, 1)
struct ContiguousDataHeader {
  size_t size;
  uint8_t bytes_per_element;
};
#pragma pack(pop)

template <typename vectorT>
static inline expected<size_t, RuntimeError> serialize_binary_blob(const vectorT& data,
                                                                   Endpoint* endpoint) {
  using value_type = typename vectorT::value_type;
  if (data.size() > 0 && sizeof(value_type) > std::numeric_limits<uint8_t>::max()) {
    return make_unexpected<RuntimeError>(RuntimeError(
        ErrorCode::kCodecError, "container element size cannot be represented on the wire"));
  }
  if (data.size() > std::numeric_limits<size_t>::max() / sizeof(value_type)) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "container byte size overflow"));
  }

  ContiguousDataHeader header;
  header.size = data.size();
  // Preserve the existing empty-container wire representation for mixed-version deployments.
  header.bytes_per_element = header.size > 0 ? sizeof(value_type) : 1;
  auto size = endpoint->write_trivial_type<ContiguousDataHeader>(&header);
  if (!size) {
    return forward_error(size);
  }
  if (header.size == 0) {
    return size.value();
  }
  auto size2 = endpoint->write(data.data(), header.size * sizeof(value_type));
  if (!size2) {
    return forward_error(size2);
  }
  return size.value() + size2.value();
}

template <typename vectorT>
static inline expected<vectorT, RuntimeError> deserialize_binary_blob(Endpoint* endpoint) {
  using value_type = typename vectorT::value_type;
  ContiguousDataHeader header;
  auto header_size = endpoint->read_trivial_type<ContiguousDataHeader>(&header);
  if (!header_size) {
    return forward_error(header_size);
  }
  if (header_size.value() != sizeof(ContiguousDataHeader)) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "short contiguous data header"));
  }
  if (header.size == 0) {
    if (header.bytes_per_element != 1 && header.bytes_per_element != sizeof(value_type)) {
      return make_unexpected<RuntimeError>(
          RuntimeError(ErrorCode::kCodecError, "invalid empty contiguous data element size"));
    }
    return vectorT{};
  }
  if (header.bytes_per_element != sizeof(value_type)) {
    return make_unexpected<RuntimeError>(RuntimeError(
        ErrorCode::kCodecError, "contiguous data element size does not match destination type"));
  }
  if (header.size > std::numeric_limits<size_t>::max() / sizeof(value_type)) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "contiguous data byte size overflow"));
  }

  vectorT data;
  constexpr size_t kMaxReadChunkBytes = 64 * 1024;
  constexpr size_t kElementsPerChunk = std::max<size_t>(1, kMaxReadChunkBytes / sizeof(value_type));
  size_t elements_read = 0;
  while (elements_read < header.size) {
    const size_t elements_to_read = std::min(kElementsPerChunk, header.size - elements_read);
    const size_t next_size = elements_read + elements_to_read;
    try {
      data.resize(next_size);
    } catch (const std::bad_alloc&) {
      return make_unexpected<RuntimeError>(
          RuntimeError(ErrorCode::kCodecError, "unable to allocate decoded container"));
    } catch (const std::length_error&) {
      return make_unexpected<RuntimeError>(
          RuntimeError(ErrorCode::kCodecError, "decoded container exceeds maximum size"));
    }

    const size_t bytes_to_read = elements_to_read * sizeof(value_type);
    auto result = endpoint->read(data.data() + elements_read, bytes_to_read);
    if (!result) {
      return forward_error(result);
    }
    if (result.value() != bytes_to_read) {
      return make_unexpected<RuntimeError>(
          RuntimeError(ErrorCode::kCodecError, "short contiguous data payload"));
    }
    elements_read = next_size;
  }
  return data;
}

// codec for vector of trivially serializable typeT
template <typename typeT>
struct codec<std::vector<typeT>> {
  static expected<size_t, RuntimeError> serialize(const std::vector<typeT>& value,
                                                  Endpoint* endpoint) {
    return serialize_binary_blob<std::vector<typeT>>(value, endpoint);
  }
  static expected<std::vector<typeT>, RuntimeError> deserialize(Endpoint* endpoint) {
    return deserialize_binary_blob<std::vector<typeT>>(endpoint);
  }
};

// deserialize_array is exactly like deserialize_binary_blob, but without a call to resize()
template <typename arrayT>
static inline expected<arrayT, RuntimeError> deserialize_array(Endpoint* endpoint) {
  using value_type = typename arrayT::value_type;
  constexpr size_t kElementCount = std::tuple_size<arrayT>::value;
  ContiguousDataHeader header;
  auto header_size = endpoint->read_trivial_type<ContiguousDataHeader>(&header);
  if (!header_size) {
    return forward_error(header_size);
  }
  if (header_size.value() != sizeof(ContiguousDataHeader)) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "short fixed array header"));
  }
  if (header.size != kElementCount) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "fixed array element count mismatch"));
  }
  if constexpr (kElementCount == 0) {
    if (header.bytes_per_element != 1 && header.bytes_per_element != sizeof(value_type)) {
      return make_unexpected<RuntimeError>(
          RuntimeError(ErrorCode::kCodecError, "invalid empty fixed array element size"));
    }
    return arrayT{};
  }
  if (header.bytes_per_element != sizeof(value_type)) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "fixed array element size mismatch"));
  }
  arrayT data;
  constexpr size_t kDataSize = kElementCount * sizeof(value_type);
  auto result = endpoint->read(data.data(), kDataSize);
  if (!result) {
    return forward_error(result);
  }
  if (result.value() != kDataSize) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "short fixed array payload"));
  }
  return data;
}

// codec for array of trivially serializable typeT and size N
template <typename typeT, size_t N>
struct codec<std::array<typeT, N>> {
  static expected<size_t, RuntimeError> serialize(const std::array<typeT, N>& value,
                                                  Endpoint* endpoint) {
    return serialize_binary_blob<std::array<typeT, N>>(value, endpoint);
  }
  static expected<std::array<typeT, N>, RuntimeError> deserialize(Endpoint* endpoint) {
    return deserialize_array<std::array<typeT, N>>(endpoint);
  }
};

// codec for std::string
template <>
struct codec<std::string> {
  static expected<size_t, RuntimeError> serialize(const std::string& value, Endpoint* endpoint) {
    return serialize_binary_blob<std::string>(value, endpoint);
  }
  static expected<std::string, RuntimeError> deserialize(Endpoint* endpoint) {
    return deserialize_binary_blob<std::string>(endpoint);
  }
};

// will hold Python cloudpickle strings in this container to differentiate from std::string
struct CloudPickleSerializedObject {
  std::string serialized;
};

// codec for CloudPickleSerializedObject
template <>
struct codec<CloudPickleSerializedObject> {
  static expected<size_t, RuntimeError> serialize(const CloudPickleSerializedObject& value,
                                                  Endpoint* endpoint) {
    return serialize_binary_blob<std::string>(value.serialized, endpoint);
  }
  static expected<CloudPickleSerializedObject, RuntimeError> deserialize(Endpoint* endpoint) {
    auto maybe_string = deserialize_binary_blob<std::string>(endpoint);
    if (!maybe_string) {
      return forward_error(maybe_string);
    }
    CloudPickleSerializedObject cloudpickle_obj{std::move(maybe_string.value())};
    return cloudpickle_obj;
  }
};

//////////////////////////////////////////////////////////////////////////////////////////////////
// Codec type 4: serialization of std::vector<bool> only
//
// Performs bit-packing/unpacking to/from uint8_t type for more efficient serialization.

// codec of std::vector<bool>
template <>
struct codec<std::vector<bool>> {
  static expected<size_t, RuntimeError> serialize(const std::vector<bool>& data,
                                                  Endpoint* endpoint) {
    size_t total_bytes = 0;
    size_t num_bits = data.size();
    size_t num_bytes = (num_bits + 7) / 8;  // the number of bytes needed to store the bits
    auto size = endpoint->write_trivial_type<size_t>(&num_bits);
    if (!size) {
      return forward_error(size);
    }
    total_bytes += size.value();
    std::vector<uint8_t> packed_data(num_bytes, 0);  // Create a vector to store the packed data
    for (size_t i = 0; i < num_bits; ++i) {
      if (data[i]) {
        packed_data[i / 8] |= (1 << (i % 8));  // Pack the bits into the bytes
      }
    }
    auto result = endpoint->write(packed_data.data(), packed_data.size());
    if (!result) {
      return forward_error(result);
    }
    total_bytes += result.value();
    return total_bytes;
  }

  static expected<std::vector<bool>, RuntimeError> deserialize(Endpoint* endpoint) {
    auto num_bits_result = deserialize_trivial_type<size_t>(endpoint);
    if (!num_bits_result) {
      return forward_error(num_bits_result);
    }
    const size_t num_bits = num_bits_result.value();
    size_t num_bytes =
        (num_bits + 7) / 8;  // Calculate the number of bytes needed to store the bits
    std::vector<uint8_t> packed_data(num_bytes, 0);  // Create a vector to store the packed data
    auto result = endpoint->read(packed_data.data(), packed_data.size());
    if (!result) {
      return forward_error(result);
    }
    std::vector<bool> data(num_bits, false);  // Create a vector to store the unpacked data
    for (size_t i = 0; i < num_bits; ++i) {
      if (packed_data[i / 8] & (1 << (i % 8))) {  // Unpack the bits from the bytes
        data[i] = true;
      }
    }
    return data;
  }
};

//////////////////////////////////////////////////////////////////////////////////////////////////
// Codec type 5: serialization of nested container types
//               e.g. std::vector<std::vector<float>>,  std::vector<std::string>
//

template <typename typeT>
static inline expected<size_t, RuntimeError> serialize_vector_of_vectors(const typeT& vectors,
                                                                         Endpoint* endpoint) {
  size_t total_size = 0;

  // header is just the total number of vectors
  size_t num_vectors = vectors.size();
  auto size = endpoint->write_trivial_type<size_t>(&num_vectors);
  if (!size) {
    return forward_error(size);
  }
  total_size += size.value();

  using vectorT = typename typeT::value_type;

  // now transmit each individual vector
  for (const auto& vector : vectors) {
    size = codec<vectorT>::serialize(vector, endpoint);
    if (!size) {
      return forward_error(size);
    }
    total_size += size.value();
  }
  return total_size;
}

template <typename typeT>
static inline expected<typeT, RuntimeError> deserialize_vector_of_vectors(Endpoint* endpoint) {
  auto num_vectors_result = deserialize_trivial_type<size_t>(endpoint);
  if (!num_vectors_result) {
    return forward_error(num_vectors_result);
  }
  const size_t num_vectors = num_vectors_result.value();

  using vectorT = typename typeT::value_type;

  std::vector<vectorT> data;
  data.reserve(num_vectors);
  for (size_t i = 0; i < num_vectors; i++) {
    auto vec = codec<vectorT>::deserialize(endpoint);
    if (!vec) {
      return forward_error(vec);
    }
    data.push_back(vec.value());
  }
  return data;
}

template <typename typeT>
struct codec<std::vector<std::vector<typeT>>> {
  static expected<size_t, RuntimeError> serialize(const std::vector<std::vector<typeT>>& value,
                                                  Endpoint* endpoint) {
    return serialize_vector_of_vectors<std::vector<std::vector<typeT>>>(value, endpoint);
  }
  static expected<std::vector<std::vector<typeT>>, RuntimeError> deserialize(Endpoint* endpoint) {
    return deserialize_vector_of_vectors<std::vector<std::vector<typeT>>>(endpoint);
  }
};

template <>
struct codec<std::vector<std::string>> {
  static expected<size_t, RuntimeError> serialize(const std::vector<std::string>& value,
                                                  Endpoint* endpoint) {
    return serialize_vector_of_vectors<std::vector<std::string>>(value, endpoint);
  }
  static expected<std::vector<std::string>, RuntimeError> deserialize(Endpoint* endpoint) {
    return deserialize_vector_of_vectors<std::vector<std::string>>(endpoint);
  }
};

// codec for shared_ptr types
// Serializes the contents of the shared_ptr. On deserialize, a new shared_ptr to the deserialized
// value is returned.
template <typename typeT>
struct codec<std::shared_ptr<typeT>> {
  static expected<size_t, RuntimeError> serialize(std::shared_ptr<typeT> value,
                                                  Endpoint* endpoint) {
    return codec<typeT>::serialize(*value, endpoint);
  }
  static expected<std::shared_ptr<typeT>, RuntimeError> deserialize(Endpoint* endpoint) {
    auto value = codec<typeT>::deserialize(endpoint);
    if (!value) {
      return forward_error(value);
    }
    return std::make_shared<typeT>(value.value());
  }
};

// codec for std::unordered_map<KeyType, ValueType>
template <typename KeyType, typename ValueType>
struct codec<std::unordered_map<KeyType, ValueType>> {
  static expected<size_t, RuntimeError> serialize(
      const std::unordered_map<KeyType, ValueType>& value, Endpoint* endpoint) {
    size_t total_size = 0;

    // header is the number of key-value pairs
    size_t num_pairs = value.size();
    auto size = endpoint->write_trivial_type<size_t>(&num_pairs);
    if (!size) {
      return forward_error(size);
    }
    total_size += size.value();

    // serialize each key-value pair
    for (const auto& pair : value) {
      // serialize the key
      size = codec<KeyType>::serialize(pair.first, endpoint);
      if (!size) {
        return forward_error(size);
      }
      total_size += size.value();

      // serialize the value
      size = codec<ValueType>::serialize(pair.second, endpoint);
      if (!size) {
        return forward_error(size);
      }
      total_size += size.value();
    }
    return total_size;
  }

  static expected<std::unordered_map<KeyType, ValueType>, RuntimeError> deserialize(
      Endpoint* endpoint) {
    auto num_pairs_result = deserialize_trivial_type<size_t>(endpoint);
    if (!num_pairs_result) {
      return forward_error(num_pairs_result);
    }
    const size_t num_pairs = num_pairs_result.value();

    std::unordered_map<KeyType, ValueType> data;
    data.reserve(num_pairs);

    for (size_t i = 0; i < num_pairs; i++) {
      // deserialize the key
      auto key = codec<KeyType>::deserialize(endpoint);
      if (!key) {
        return forward_error(key);
      }

      // deserialize the value
      auto value = codec<ValueType>::deserialize(endpoint);
      if (!value) {
        return forward_error(value);
      }

      data[key.value()] = value.value();
    }
    return data;
  }
};
}  // namespace holoscan

#endif /* HOLOSCAN_CORE_CODECS_HPP */
