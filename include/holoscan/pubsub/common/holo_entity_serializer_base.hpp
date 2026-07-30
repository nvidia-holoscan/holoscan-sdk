/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PUBSUB_COMMON_INCLUDE_PUBSUB_HOLO_ENTITY_SERIALIZER_BASE_HPP
#define PUBSUB_COMMON_INCLUDE_PUBSUB_HOLO_ENTITY_SERIALIZER_BASE_HPP

#include <atomic>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <typeindex>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <gxf/core/gxf.h>
#include <gxf/core/entity.hpp>
#include <gxf/core/expected.hpp>
#include <gxf/core/handle.hpp>
#include <gxf/pubsub/pubsub_entity_serializer.hpp>
#include <gxf/pubsub/pubsub_native_buffer.hpp>
#include <gxf/std/allocator.hpp>
#include <gxf/std/tensor.hpp>
#include <gxf/std/timestamp.hpp>

#include <holoscan/core/codec_registry.hpp>
#include <holoscan/core/codecs.hpp>
#include <holoscan/core/message.hpp>
#include <holoscan/core/messagelabel.hpp>
#include <holoscan/core/metadata.hpp>
#include <holoscan/logger/logger.hpp>

#include <holoscan/pubsub/common/holoscan_component_payloads.hpp>
#include <holoscan/pubsub/common/native_buffer_protocol.hpp>
#include <holoscan/pubsub/common/native_buffer_protocol_adapter.hpp>

// Error-check macros for endpoint read/write. Use serializer_name_ for log context.
// Defined here and #undef'd at the end of this header to avoid leaking.
#define HOLO_CHECK_READ(expr, error_msg)                                                    \
  do {                                                                                      \
    auto _holo_r = (expr);                                                                  \
    if (!_holo_r) {                                                                         \
      HOLOSCAN_LOG_ERROR(                                                                   \
          "{}::deserialize: {} ({})", serializer_name_, error_msg, _holo_r.error().what()); \
      return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);                      \
    }                                                                                       \
  } while (0)

#define HOLO_CHECK_WRITE(expr, error_msg)                                                 \
  do {                                                                                    \
    auto _holo_r = (expr);                                                                \
    if (!_holo_r) {                                                                       \
      HOLOSCAN_LOG_ERROR(                                                                 \
          "{}::serialize: {} ({})", serializer_name_, error_msg, _holo_r.error().what()); \
      return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);                    \
    }                                                                                     \
  } while (0)

namespace holoscan {

// ---------------------------------------------------------------------------
// Process-global cached GXF type IDs (shared across all template instantiations)
// ---------------------------------------------------------------------------
namespace detail {

struct CachedTypeIds {
  gxf_tid_t timestamp_tid = GxfTidNull();
  gxf_tid_t tensor_tid = GxfTidNull();
  gxf_tid_t message_tid = GxfTidNull();
  gxf_tid_t metadata_tid = GxfTidNull();
  gxf_tid_t message_label_tid = GxfTidNull();
  bool initialized = false;
  gxf_result_t init_result = GXF_SUCCESS;
};

inline std::once_flag& cached_tid_init_flag() {
  static std::once_flag flag;
  return flag;
}

inline CachedTypeIds& cached_tids() {
  static CachedTypeIds ids;
  return ids;
}

inline void initialize_cached_type_ids(gxf_context_t context, const std::string& name) {
  auto& ids = cached_tids();

  ids.init_result = GxfComponentTypeId(
      context, nvidia::TypenameAsString<nvidia::gxf::Timestamp>(), &ids.timestamp_tid);
  if (ids.init_result != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR(
        "{}: failed to get type ID for Timestamp: {}", name, GxfResultStr(ids.init_result));
    return;
  }
  ids.init_result =
      GxfComponentTypeId(context, nvidia::TypenameAsString<nvidia::gxf::Tensor>(), &ids.tensor_tid);
  if (ids.init_result != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR(
        "{}: failed to get type ID for Tensor: {}", name, GxfResultStr(ids.init_result));
    return;
  }
  ids.init_result =
      GxfComponentTypeId(context, nvidia::TypenameAsString<Message>(), &ids.message_tid);
  if (ids.init_result != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR(
        "{}: failed to get type ID for Message: {}", name, GxfResultStr(ids.init_result));
    return;
  }
  ids.init_result = GxfComponentTypeId(
      context, nvidia::TypenameAsString<MetadataDictionary>(), &ids.metadata_tid);
  if (ids.init_result != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR("{}: failed to get type ID for MetadataDictionary: {}",
                       name,
                       GxfResultStr(ids.init_result));
    return;
  }
  ids.init_result =
      GxfComponentTypeId(context, nvidia::TypenameAsString<MessageLabel>(), &ids.message_label_tid);
  if (ids.init_result != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR(
        "{}: failed to get type ID for MessageLabel: {}", name, GxfResultStr(ids.init_result));
    return;
  }
  ids.initialized = true;
  HOLOSCAN_LOG_DEBUG("{}: cached type IDs initialized", name);
}

inline bool tid_equals(const gxf_tid_t& a, const gxf_tid_t& b) {
  return a.hash1 == b.hash1 && a.hash2 == b.hash2;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// NDPL helpers (used by export/import_native_descriptor)
// ---------------------------------------------------------------------------
namespace detail {

constexpr uint32_t kNdplMagic = 0x4C50444E;  // "NDPL"
constexpr uint8_t kNdplVersion = 1;

/// Compute product of shape dimensions * bytes_per_element with overflow detection.
/// Returns 0 on overflow or if bytes_per_element is 0.
inline uint64_t safe_tensor_element_bytes(const uint64_t* shape, uint32_t rank,
                                          uint64_t bytes_per_element) {
  if (bytes_per_element == 0 || rank == 0)
    return 0;
  uint64_t count = 1;
  for (uint32_t d = 0; d < rank && d < 8; ++d) {
    if (shape[d] == 0)
      return 0;
    if (count > std::numeric_limits<uint64_t>::max() / shape[d])
      return 0;  // overflow
    count *= shape[d];
  }
  if (count > std::numeric_limits<uint64_t>::max() / bytes_per_element)
    return 0;  // overflow
  return count * bytes_per_element;
}

inline void ndpl_append_bytes(std::vector<uint8_t>& out, const void* data, size_t size) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  out.insert(out.end(), bytes, bytes + size);
}

inline bool ndpl_append_string(std::vector<uint8_t>& out, const std::string& value) {
  if (value.size() > std::numeric_limits<uint16_t>::max())
    return false;
  const uint16_t length = static_cast<uint16_t>(value.size());
  ndpl_append_bytes(out, &length, sizeof(length));
  ndpl_append_bytes(out, value.data(), value.size());
  return true;
}

inline bool ndpl_read_string(const std::vector<uint8_t>& data, size_t& offset, std::string& out) {
  uint16_t length = 0;
  if (sizeof(length) > data.size() || offset > data.size() - sizeof(length))
    return false;
  std::memcpy(&length, data.data() + offset, sizeof(length));
  offset += sizeof(length);
  if (length > data.size() || offset > data.size() - length)
    return false;
  out.assign(reinterpret_cast<const char*>(data.data() + offset), length);
  offset += length;
  return true;
}

inline std::string primitive_type_to_dtype_string(nvidia::gxf::PrimitiveType type) {
  switch (type) {
    case nvidia::gxf::PrimitiveType::kFloat32:
      return "float32";
    case nvidia::gxf::PrimitiveType::kFloat64:
      return "float64";
    case nvidia::gxf::PrimitiveType::kInt8:
      return "int8";
    case nvidia::gxf::PrimitiveType::kInt16:
      return "int16";
    case nvidia::gxf::PrimitiveType::kInt32:
      return "int32";
    case nvidia::gxf::PrimitiveType::kInt64:
      return "int64";
    case nvidia::gxf::PrimitiveType::kUnsigned8:
      return "uint8";
    case nvidia::gxf::PrimitiveType::kUnsigned16:
      return "uint16";
    case nvidia::gxf::PrimitiveType::kUnsigned32:
      return "uint32";
    case nvidia::gxf::PrimitiveType::kUnsigned64:
      return "uint64";
    case nvidia::gxf::PrimitiveType::kComplex64:
      return "complex64";
    case nvidia::gxf::PrimitiveType::kComplex128:
      return "complex128";
    case nvidia::gxf::PrimitiveType::kFloat16:
      return "float16";
    default:
      return "unknown";
  }
}

inline nvidia::gxf::PrimitiveType dtype_string_to_primitive_type(const std::string& dtype) {
  if (dtype == "float32")
    return nvidia::gxf::PrimitiveType::kFloat32;
  if (dtype == "float64")
    return nvidia::gxf::PrimitiveType::kFloat64;
  if (dtype == "int8")
    return nvidia::gxf::PrimitiveType::kInt8;
  if (dtype == "int16")
    return nvidia::gxf::PrimitiveType::kInt16;
  if (dtype == "int32")
    return nvidia::gxf::PrimitiveType::kInt32;
  if (dtype == "int64")
    return nvidia::gxf::PrimitiveType::kInt64;
  if (dtype == "uint8")
    return nvidia::gxf::PrimitiveType::kUnsigned8;
  if (dtype == "uint16")
    return nvidia::gxf::PrimitiveType::kUnsigned16;
  if (dtype == "uint32")
    return nvidia::gxf::PrimitiveType::kUnsigned32;
  if (dtype == "uint64")
    return nvidia::gxf::PrimitiveType::kUnsigned64;
  if (dtype == "complex64")
    return nvidia::gxf::PrimitiveType::kComplex64;
  if (dtype == "complex128")
    return nvidia::gxf::PrimitiveType::kComplex128;
  if (dtype == "float16")
    return nvidia::gxf::PrimitiveType::kFloat16;
  return nvidia::gxf::PrimitiveType::kCustom;
}

inline NativeTensorMetadata tensor_metadata_from_tensor(const nvidia::gxf::Tensor& tensor) {
  NativeTensorMetadata metadata;
  for (uint32_t d = 0; d < tensor.rank(); ++d) {
    metadata.shape.push_back(tensor.shape().dimension(d));
    metadata.strides.push_back(tensor.stride(d));
  }
  metadata.dtype = primitive_type_to_dtype_string(tensor.element_type());
  metadata.storage_type = tensor.storage_type();
  metadata.bytes_per_element = tensor.bytes_per_element();
  return metadata;
}

inline nvidia::gxf::Expected<nvidia::gxf::Shape> shape_from_metadata(
    const NativeTensorMetadata& metadata) {
  if (metadata.shape.size() > nvidia::gxf::Shape::kMaxRank) {
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }
  std::array<int32_t, nvidia::gxf::Shape::kMaxRank> dims = {};
  for (size_t d = 0; d < metadata.shape.size(); ++d) {
    if (metadata.shape[d] < 0 ||
        metadata.shape[d] > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
    }
    dims[d] = static_cast<int32_t>(metadata.shape[d]);
  }
  return nvidia::gxf::Shape(dims, static_cast<uint32_t>(metadata.shape.size()));
}

inline nvidia::gxf::Expected<nvidia::gxf::Tensor::stride_array_t> strides_from_metadata(
    const NativeTensorMetadata& metadata) {
  if (!metadata.strides.empty() && metadata.strides.size() != metadata.shape.size()) {
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }
  if (metadata.strides.size() > nvidia::gxf::Shape::kMaxRank) {
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }
  nvidia::gxf::Tensor::stride_array_t strides = {};
  for (size_t d = 0; d < metadata.strides.size(); ++d) {
    if (metadata.strides[d] < 0) {
      return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
    }
    strides[d] = static_cast<uint64_t>(metadata.strides[d]);
  }
  return strides;
}

}  // namespace detail

/// Shared HOLO/NDPL entity serializer core, parameterized on the endpoint type.
///
/// Implements the full PubSubEntitySerializer interface: entity serialize/deserialize
/// using the "HOLO" wire format, and native-descriptor export/import using the NDPL
/// envelope. Subclasses only provide the constructor (passing endpoint type and name).
///
/// EndpointT must provide: write(), read(), write_trivial_type(), read_trivial_type(),
/// bytes_written(), bytes_remaining(), and constructors from vector<uint8_t>* (write),
/// const vector<uint8_t>* (read), and (uint8_t*, size_t) (write into buffer).
template <typename EndpointT>
class HoloEntitySerializerBase : public nvidia::gxf::PubSubEntitySerializer {
 public:
  // Bring base class overload (move semantics) into scope
  using PubSubEntitySerializer::deserialize;

  // --- Wire format types (public for testing) ---

  struct SerializationHeader {
    uint32_t magic = 0x484F4C4F;  // "HOLO"
    uint32_t version = 1;
    uint32_t num_components = 0;
    uint32_t flags = 0;
    uint64_t total_size = 0;
  };

  enum class ComponentType : uint32_t {
    kUnknown = 0,
    kTensor = 1,
    kTimestamp = 2,
    kMessage = 3,
    kMetadataDictionary = 4,
    kMessageLabel = 5,
  };

  struct ComponentHeader {
    uint32_t type_id = 0;
    uint32_t name_length = 0;
    uint32_t data_size = 0;
    uint32_t flags = 0;
  };

  static constexpr uint32_t kFlagIsGpuTensor = 0x01;

  struct TensorMetadata {
    uint32_t rank = 0;
    uint32_t element_type = 0;
    uint32_t storage_type = 0;
    uint64_t bytes_per_element = 0;
    uint64_t total_bytes = 0;
    uint64_t shape[8] = {0};
    uint64_t strides[8] = {0};
  };

  ~HoloEntitySerializerBase() override = default;

  HoloEntitySerializerBase(const HoloEntitySerializerBase&) = delete;
  HoloEntitySerializerBase& operator=(const HoloEntitySerializerBase&) = delete;

  // --- PubSubEntitySerializer interface ---

  nvidia::gxf::Expected<std::vector<uint8_t>> serialize(
      nvidia::gxf::Entity entity, nvidia::gxf::Handle<nvidia::gxf::Allocator> allocator =
                                      nvidia::gxf::Handle<nvidia::gxf::Allocator>()) override;

  nvidia::gxf::Expected<size_t> serialize_into(
      nvidia::gxf::Entity entity, uint8_t* buffer, size_t buffer_size,
      nvidia::gxf::Handle<nvidia::gxf::Allocator> allocator =
          nvidia::gxf::Handle<nvidia::gxf::Allocator>()) override;

  nvidia::gxf::Expected<nvidia::gxf::Entity> deserialize(
      const std::vector<uint8_t>& data, gxf_context_t context,
      nvidia::gxf::Handle<nvidia::gxf::Allocator> allocator) override;

  size_t estimate_size(nvidia::gxf::Entity entity) override;

  bool supports_gpu_tensors() const override { return true; }
  bool supports_zero_copy() const override { return false; }
  bool supports_direct_buffer_serialization() const override { return true; }
  bool supports_native_descriptors() const override { return native_adapter_ != nullptr; }

  nvidia::gxf::Expected<void> export_native_descriptor(
      gxf_uid_t entity_uid, gxf_context_t context,
      nvidia::gxf::Handle<nvidia::gxf::Allocator> allocator,
      nvidia::gxf::NativeDescriptorPayload& out_payload,
      const std::string& protocol_name = "") override;

  nvidia::gxf::Expected<nvidia::gxf::Entity> import_native_descriptor(
      const nvidia::gxf::NativeDescriptorPayload& payload, gxf_context_t context,
      nvidia::gxf::Handle<nvidia::gxf::Allocator> allocator) override;

  const char* name() const override { return serializer_name_.c_str(); }

  // --- Configuration ---

  void set_native_buffer_adapter(std::shared_ptr<NativeBufferProtocolAdapter> adapter) {
    native_adapter_ = std::move(adapter);
  }
  void set_cuda_stream(cudaStream_t stream) { cuda_stream_ = stream; }
  cudaStream_t cuda_stream() const { return cuda_stream_; }

 protected:
  explicit HoloEntitySerializerBase(std::string serializer_name, cudaStream_t cuda_stream = nullptr)
      : serializer_name_(std::move(serializer_name)), cuda_stream_(cuda_stream) {}

  std::string serializer_name_;

 private:
  // --- Bounds constants ---
  static constexpr size_t kMaxComponentNameLength = kMaxComponentNameSize;
  static constexpr size_t kMaxComponentDataSize = 1024 * 1024 * 1024;
  static constexpr size_t kMaxTensorBytes = 1024 * 1024 * 1024;
  static constexpr size_t kMaxMetadataItems = 10000;
  static constexpr size_t kMaxMessageLabelPaths = 10000;
  static constexpr size_t kMaxOperatorsPerPath = 10000;

  // --- Type conversion helpers ---
  static uint32_t primitive_type_to_uint(nvidia::gxf::PrimitiveType type) {
    return static_cast<uint32_t>(type);
  }
  static nvidia::gxf::PrimitiveType uint_to_primitive_type(uint32_t value) {
    return static_cast<nvidia::gxf::PrimitiveType>(value);
  }
  static uint32_t storage_type_to_uint(nvidia::gxf::MemoryStorageType type) {
    return static_cast<uint32_t>(type);
  }
  static nvidia::gxf::MemoryStorageType uint_to_storage_type(uint32_t value) {
    return static_cast<nvidia::gxf::MemoryStorageType>(value);
  }

  // --- String serialization (ContiguousDataHeader format, compatible across backends) ---
  static expected<size_t, RuntimeError> serialize_string(const std::string& data,
                                                         EndpointT* endpoint) {
    ContiguousDataHeader header;
    header.size = data.size();
    header.bytes_per_element = header.size > 0 ? sizeof(data[0]) : 1;
    auto size = endpoint->write_trivial_type(&header);
    if (!size)
      return forward_error(size);
    if (header.size > 0) {
      auto size2 = endpoint->write(data.data(), header.size * header.bytes_per_element);
      if (!size2)
        return forward_error(size2);
      return size.value() + size2.value();
    }
    return size.value();
  }

  static expected<std::string, RuntimeError> deserialize_string(EndpointT* endpoint) {
    ContiguousDataHeader header;
    auto header_size = endpoint->read_trivial_type(&header);
    if (!header_size)
      return forward_error(header_size);
    std::string data;
    if (header.size > 0) {
      data.resize(header.size);
      auto result = endpoint->read(data.data(), header.size * header.bytes_per_element);
      if (!result)
        return forward_error(result);
    }
    return data;
  }

  // --- Ensure type IDs are cached ---
  bool ensure_type_ids(gxf_context_t context) {
    std::call_once(detail::cached_tid_init_flag(),
                   [&]() { detail::initialize_cached_type_ids(context, serializer_name_); });
    return detail::cached_tids().initialized;
  }

  // --- Core implementation methods ---
  nvidia::gxf::Expected<size_t> serialize_to_endpoint(
      nvidia::gxf::Entity entity, EndpointT& endpoint,
      nvidia::gxf::Handle<nvidia::gxf::Allocator> allocator);

  nvidia::gxf::Expected<std::vector<uint8_t>> serialize_non_gpu_sideband(
      nvidia::gxf::Entity entity);

  nvidia::gxf::Expected<void> deserialize_non_gpu_sideband(
      const uint8_t* data, size_t size, nvidia::gxf::Entity& entity,
      nvidia::gxf::Handle<nvidia::gxf::Allocator> allocator =
          nvidia::gxf::Handle<nvidia::gxf::Allocator>());

  nvidia::gxf::Expected<std::vector<uint8_t>> stage_tensor_to_host(const void* gpu_ptr,
                                                                   size_t size);
  nvidia::gxf::Expected<void> stage_host_to_device(void* gpu_ptr, const void* host_ptr,
                                                   size_t size);

  // --- Member data ---
  cudaStream_t cuda_stream_ = nullptr;
  std::shared_ptr<NativeBufferProtocolAdapter> native_adapter_;
  std::atomic<uint32_t> sequence_counter_{0};
};

// ===========================================================================
// Template method implementations
// ===========================================================================

template <typename EndpointT>
nvidia::gxf::Expected<std::vector<uint8_t>> HoloEntitySerializerBase<EndpointT>::serialize(
    nvidia::gxf::Entity entity, nvidia::gxf::Handle<nvidia::gxf::Allocator> allocator) {
  std::vector<uint8_t> buffer;
  buffer.reserve(4096);
  EndpointT endpoint(&buffer);
  auto result = serialize_to_endpoint(entity, endpoint, allocator);
  if (!result)
    return nvidia::gxf::ForwardError(result);
  SerializationHeader header;
  std::memcpy(&header, buffer.data(), sizeof(header));
  header.total_size = buffer.size();
  std::memcpy(buffer.data(), &header, sizeof(header));
  return buffer;
}

template <typename EndpointT>
nvidia::gxf::Expected<size_t> HoloEntitySerializerBase<EndpointT>::serialize_into(
    nvidia::gxf::Entity entity, uint8_t* buffer, size_t buffer_size,
    nvidia::gxf::Handle<nvidia::gxf::Allocator> allocator) {
  EndpointT endpoint(buffer, buffer_size);
  auto result = serialize_to_endpoint(entity, endpoint, allocator);
  if (!result)
    return nvidia::gxf::ForwardError(result);
  size_t total_bytes = result.value();
  SerializationHeader header;
  std::memcpy(&header, buffer, sizeof(header));
  header.total_size = total_bytes;
  std::memcpy(buffer, &header, sizeof(header));
  return total_bytes;
}

template <typename EndpointT>
nvidia::gxf::Expected<size_t> HoloEntitySerializerBase<EndpointT>::serialize_to_endpoint(
    nvidia::gxf::Entity entity, EndpointT& endpoint,
    nvidia::gxf::Handle<nvidia::gxf::Allocator> allocator) {
  (void)allocator;

  if (!ensure_type_ids(entity.context())) {
    HOLOSCAN_LOG_ERROR("{}::serialize: failed to initialize type IDs [{}]",
                       serializer_name_,
                       GxfResultStr(detail::cached_tids().init_result));
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }
  const auto& tids = detail::cached_tids();

  auto maybe_components = entity.findAll();
  if (!maybe_components) {
    HOLOSCAN_LOG_ERROR("{}::serialize: failed to get entity components [{}]",
                       serializer_name_,
                       GxfResultStr(maybe_components.error()));
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }
  const auto& components = maybe_components.value();

  // First pass: count components and check for GPU tensors
  uint32_t num_components = 0;
  bool has_gpu_tensors = false;
  for (size_t i = 0; i < components.size(); ++i) {
    const auto& mc = components[i];
    if (!mc)
      continue;
    const auto& component = mc.value();
    const gxf_tid_t tid = component.tid();
    if (detail::tid_equals(tid, tids.timestamp_tid) || detail::tid_equals(tid, tids.tensor_tid) ||
        detail::tid_equals(tid, tids.message_tid) || detail::tid_equals(tid, tids.metadata_tid) ||
        detail::tid_equals(tid, tids.message_label_tid)) {
      num_components++;
      if (detail::tid_equals(tid, tids.tensor_tid)) {
        auto th =
            nvidia::gxf::Handle<nvidia::gxf::Tensor>::Create(entity.context(), component.cid());
        if (th && th.value()->storage_type() == nvidia::gxf::MemoryStorageType::kDevice) {
          has_gpu_tensors = true;
        }
      }
    }
  }

  // Write header
  SerializationHeader header;
  header.num_components = num_components;
  header.flags = has_gpu_tensors ? 0x01 : 0;
  header.total_size = 0;
  HOLO_CHECK_WRITE(endpoint.write_trivial_type(&header), "failed to write header");

  // Second pass: serialize components
  for (size_t i = 0; i < components.size(); ++i) {
    const auto& mc = components[i];
    if (!mc)
      continue;
    const auto& component = mc.value();
    const gxf_tid_t tid = component.tid();

    // --- Timestamp ---
    if (detail::tid_equals(tid, tids.timestamp_tid)) {
      auto ts_handle =
          nvidia::gxf::Handle<nvidia::gxf::Timestamp>::Create(entity.context(), component.cid());
      if (!ts_handle)
        continue;
      auto timestamp = ts_handle.value();
      ComponentHeader ch;
      ch.type_id = static_cast<uint32_t>(ComponentType::kTimestamp);
      ch.name_length = std::strlen(component.name());
      ch.data_size = sizeof(int64_t) * 2;
      ch.flags = 0;
      HOLO_CHECK_WRITE(endpoint.write_trivial_type(&ch), "failed to write Timestamp header");
      HOLO_CHECK_WRITE(endpoint.write(component.name(), ch.name_length),
                       "failed to write Timestamp name");
      HOLO_CHECK_WRITE(endpoint.write_trivial_type(&timestamp->acqtime),
                       "failed to write Timestamp acqtime");
      HOLO_CHECK_WRITE(endpoint.write_trivial_type(&timestamp->pubtime),
                       "failed to write Timestamp pubtime");
    } else if (detail::tid_equals(tid, tids.tensor_tid)) {
      // --- Tensor ---
      auto tensor_handle =
          nvidia::gxf::Handle<nvidia::gxf::Tensor>::Create(entity.context(), component.cid());
      if (!tensor_handle)
        continue;
      auto tensor = tensor_handle.value();
      const size_t tensor_size = tensor->size();
      const void* tensor_ptr = tensor->pointer();
      const auto st = tensor->storage_type();
      bool is_gpu = (st == nvidia::gxf::MemoryStorageType::kDevice);
      if (st == nvidia::gxf::MemoryStorageType::kCudaManaged) {
        cudaStreamSynchronize(cuda_stream_);
      }
      std::vector<uint8_t> staged_data;
      const void* data_ptr = tensor_ptr;
      if (is_gpu) {
        auto staged = stage_tensor_to_host(tensor_ptr, tensor_size);
        if (!staged) {
          HOLOSCAN_LOG_ERROR("{}::serialize: failed to stage GPU tensor", serializer_name_);
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        }
        staged_data = std::move(staged.value());
        data_ptr = staged_data.data();
      }
      TensorMetadata meta;
      meta.rank = tensor->rank();
      meta.element_type = primitive_type_to_uint(tensor->element_type());
      meta.storage_type = storage_type_to_uint(st);
      meta.bytes_per_element = tensor->bytes_per_element();
      meta.total_bytes = tensor_size;
      auto shape = tensor->shape();
      for (uint32_t d = 0; d < meta.rank && d < 8; ++d) {
        meta.shape[d] = shape.dimension(d);
        meta.strides[d] = tensor->stride(d);
      }
      ComponentHeader ch;
      ch.type_id = static_cast<uint32_t>(ComponentType::kTensor);
      ch.name_length = std::strlen(component.name());
      ch.data_size = sizeof(TensorMetadata) + tensor_size;
      ch.flags = is_gpu ? kFlagIsGpuTensor : 0;
      HOLO_CHECK_WRITE(endpoint.write_trivial_type(&ch), "failed to write Tensor header");
      HOLO_CHECK_WRITE(endpoint.write(component.name(), ch.name_length),
                       "failed to write Tensor name");
      HOLO_CHECK_WRITE(endpoint.write_trivial_type(&meta), "failed to write TensorMetadata");
      HOLO_CHECK_WRITE(endpoint.write(data_ptr, tensor_size), "failed to write tensor data");
    } else if (detail::tid_equals(tid, tids.message_tid)) {
      // --- Message ---
      auto msg_handle = nvidia::gxf::Handle<Message>::Create(entity.context(), component.cid());
      if (!msg_handle)
        continue;
      const auto& message = *msg_handle.value();
      auto encoded = encode_message_payload(message);
      if (!encoded) {
        HOLOSCAN_LOG_ERROR("{}::serialize: encode_message_payload failed", serializer_name_);
        return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
      }
      std::vector<uint8_t> msg_buffer;
      EndpointT msg_endpoint(&msg_buffer);
      auto ns = serialize_string(encoded.value().codec_name, &msg_endpoint);
      if (!ns) {
        HOLOSCAN_LOG_ERROR("{}::serialize: failed to serialize codec name", serializer_name_);
        return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
      }
      msg_endpoint.write(encoded.value().payload.data(), encoded.value().payload.size());
      ComponentHeader ch;
      ch.type_id = static_cast<uint32_t>(ComponentType::kMessage);
      ch.name_length = std::strlen(component.name());
      ch.data_size = static_cast<uint32_t>(msg_buffer.size());
      ch.flags = 0;
      HOLO_CHECK_WRITE(endpoint.write_trivial_type(&ch), "failed to write Message header");
      HOLO_CHECK_WRITE(endpoint.write(component.name(), ch.name_length),
                       "failed to write Message name");
      HOLO_CHECK_WRITE(endpoint.write(msg_buffer.data(), msg_buffer.size()),
                       "failed to write Message data");
    } else if (detail::tid_equals(tid, tids.metadata_tid)) {
      // --- MetadataDictionary ---
      auto mh = nvidia::gxf::Handle<MetadataDictionary>::Create(entity.context(), component.cid());
      if (!mh)
        continue;
      auto meta_result = encode_metadata_dictionary_payload(*mh.value());
      if (!meta_result) {
        HOLOSCAN_LOG_ERROR("{}::serialize: encode_metadata_dictionary_payload failed",
                           serializer_name_);
        return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
      }
      auto& meta_buffer = meta_result.value();
      ComponentHeader ch;
      ch.type_id = static_cast<uint32_t>(ComponentType::kMetadataDictionary);
      ch.name_length = std::strlen(component.name());
      ch.data_size = static_cast<uint32_t>(meta_buffer.size());
      ch.flags = 0;
      HOLO_CHECK_WRITE(endpoint.write_trivial_type(&ch),
                       "failed to write MetadataDictionary header");
      HOLO_CHECK_WRITE(endpoint.write(component.name(), ch.name_length),
                       "failed to write MetadataDictionary name");
      HOLO_CHECK_WRITE(endpoint.write(meta_buffer.data(), meta_buffer.size()),
                       "failed to write MetadataDictionary data");
    } else if (detail::tid_equals(tid, tids.message_label_tid)) {
      // --- MessageLabel ---
      auto lh = nvidia::gxf::Handle<MessageLabel>::Create(entity.context(), component.cid());
      if (!lh)
        continue;
      auto label_result = encode_message_label_payload(*lh.value());
      if (!label_result) {
        HOLOSCAN_LOG_ERROR("{}::serialize: encode_message_label_payload failed", serializer_name_);
        return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
      }
      auto& label_buffer = label_result.value();
      ComponentHeader ch;
      ch.type_id = static_cast<uint32_t>(ComponentType::kMessageLabel);
      ch.name_length = std::strlen(component.name());
      ch.data_size = static_cast<uint32_t>(label_buffer.size());
      ch.flags = 0;
      HOLO_CHECK_WRITE(endpoint.write_trivial_type(&ch), "failed to write MessageLabel header");
      HOLO_CHECK_WRITE(endpoint.write(component.name(), ch.name_length),
                       "failed to write MessageLabel name");
      HOLO_CHECK_WRITE(endpoint.write(label_buffer.data(), label_buffer.size()),
                       "failed to write MessageLabel data");
    }
  }

  size_t total_bytes = endpoint.bytes_written();
  HOLOSCAN_LOG_DEBUG("{}::serialize: serialized {} components, {} bytes",
                     serializer_name_,
                     num_components,
                     total_bytes);
  return total_bytes;
}

template <typename EndpointT>
nvidia::gxf::Expected<nvidia::gxf::Entity> HoloEntitySerializerBase<EndpointT>::deserialize(
    const std::vector<uint8_t>& data, gxf_context_t context,
    nvidia::gxf::Handle<nvidia::gxf::Allocator> allocator) {
  if (data.size() < sizeof(SerializationHeader)) {
    HOLOSCAN_LOG_ERROR("{}::deserialize: data too small for header", serializer_name_);
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }

  EndpointT endpoint(&data);
  SerializationHeader header;
  HOLO_CHECK_READ(endpoint.read_trivial_type(&header), "failed to read header");
  if (header.magic != 0x484F4C4F) {
    HOLOSCAN_LOG_ERROR("{}::deserialize: invalid magic number", serializer_name_);
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }
  if (header.version != 1) {
    HOLOSCAN_LOG_ERROR("{}::deserialize: unsupported version {}", serializer_name_, header.version);
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }

  auto maybe_entity = nvidia::gxf::Entity::New(context);
  if (!maybe_entity) {
    HOLOSCAN_LOG_ERROR("{}::deserialize: failed to create entity [{}]",
                       serializer_name_,
                       GxfResultStr(maybe_entity.error()));
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }
  nvidia::gxf::Entity entity = maybe_entity.value();
  auto& registry = CodecRegistry::get_instance();

  for (uint32_t i = 0; i < header.num_components; ++i) {
    ComponentHeader ch;
    HOLO_CHECK_READ(endpoint.read_trivial_type(&ch), "failed to read component header");
    if (ch.name_length > kMaxComponentNameLength) {
      HOLOSCAN_LOG_ERROR(
          "{}::deserialize: name_length {} exceeds max", serializer_name_, ch.name_length);
      return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
    }
    if (ch.data_size > kMaxComponentDataSize) {
      HOLOSCAN_LOG_ERROR(
          "{}::deserialize: data_size {} exceeds max", serializer_name_, ch.data_size);
      return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
    }
    const size_t remaining = endpoint.bytes_remaining();
    if (ch.name_length > remaining || ch.data_size > remaining - ch.name_length) {
      HOLOSCAN_LOG_ERROR(
          "{}::deserialize: insufficient data for component {}", serializer_name_, i);
      return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
    }
    std::string comp_name(ch.name_length, '\0');
    if (ch.name_length > 0) {
      HOLO_CHECK_READ(endpoint.read(comp_name.data(), ch.name_length),
                      "failed to read component name");
    }

    ComponentType type = static_cast<ComponentType>(ch.type_id);
    switch (type) {
      case ComponentType::kTimestamp: {
        auto mt = entity.add<nvidia::gxf::Timestamp>(comp_name.c_str());
        if (!mt) {
          HOLOSCAN_LOG_ERROR("{}::deserialize: failed to add Timestamp", serializer_name_);
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        }
        auto ts = mt.value();
        HOLO_CHECK_READ(endpoint.read_trivial_type(&ts->acqtime), "Timestamp acqtime");
        HOLO_CHECK_READ(endpoint.read_trivial_type(&ts->pubtime), "Timestamp pubtime");
        break;
      }
      case ComponentType::kTensor: {
        TensorMetadata meta;
        HOLO_CHECK_READ(endpoint.read_trivial_type(&meta), "failed to read TensorMetadata");
        if (meta.total_bytes > kMaxTensorBytes || meta.rank > 8) {
          HOLOSCAN_LOG_ERROR("{}::deserialize: invalid tensor metadata", serializer_name_);
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        }
        {
          const uint64_t min_bytes =
              detail::safe_tensor_element_bytes(meta.shape, meta.rank, meta.bytes_per_element);
          if (min_bytes == 0 || meta.total_bytes < min_bytes) {
            HOLOSCAN_LOG_ERROR(
                "{}::deserialize: tensor shape/bytes_per_element inconsistent with total_bytes "
                "(min_bytes={}, total_bytes={})",
                serializer_name_,
                min_bytes,
                meta.total_bytes);
            return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
          }
        }
        auto original_storage = uint_to_storage_type(meta.storage_type);
        bool needs_h2d = (ch.flags & kFlagIsGpuTensor) != 0;
        nvidia::gxf::MemoryStorageType target_storage;
        if (needs_h2d) {
          target_storage = nvidia::gxf::MemoryStorageType::kDevice;
        } else if (original_storage == nvidia::gxf::MemoryStorageType::kCudaManaged) {
          target_storage = nvidia::gxf::MemoryStorageType::kCudaManaged;
        } else if (original_storage == nvidia::gxf::MemoryStorageType::kSystem) {
          target_storage = nvidia::gxf::MemoryStorageType::kSystem;
        } else {
          target_storage = nvidia::gxf::MemoryStorageType::kHost;
        }
        std::array<int32_t, nvidia::gxf::Shape::kMaxRank> dims = {};
        nvidia::gxf::Tensor::stride_array_t strides = {};
        for (uint32_t d = 0; d < meta.rank && d < 8; ++d) {
          dims[d] = static_cast<int32_t>(meta.shape[d]);
          strides[d] = meta.strides[d];
        }
        nvidia::gxf::Shape shape(dims, meta.rank);
        auto mt = entity.add<nvidia::gxf::Tensor>(comp_name.c_str());
        if (!mt) {
          HOLOSCAN_LOG_ERROR("{}::deserialize: failed to add Tensor", serializer_name_);
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        }
        auto tensor = mt.value();
        auto reshape_result = tensor->reshapeCustom(shape,
                                                    uint_to_primitive_type(meta.element_type),
                                                    meta.bytes_per_element,
                                                    strides,
                                                    target_storage,
                                                    allocator);
        if (!reshape_result) {
          HOLOSCAN_LOG_ERROR("{}::deserialize: reshapeCustom failed [{}]",
                             serializer_name_,
                             GxfResultStr(reshape_result.error()));
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        }
        if (meta.total_bytes > tensor->size()) {
          HOLOSCAN_LOG_ERROR("{}::deserialize: total_bytes ({}) exceeds allocated tensor size ({})",
                             serializer_name_,
                             meta.total_bytes,
                             tensor->size());
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        }
        void* dest_ptr = tensor->pointer();
        if (needs_h2d) {
          std::vector<uint8_t> temp(meta.total_bytes);
          HOLO_CHECK_READ(endpoint.read(temp.data(), meta.total_bytes),
                          "failed to read tensor data");
          auto sr = stage_host_to_device(dest_ptr, temp.data(), meta.total_bytes);
          if (!sr) {
            HOLOSCAN_LOG_ERROR("{}::deserialize: H2D staging failed", serializer_name_);
            return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
          }
        } else {
          HOLO_CHECK_READ(endpoint.read(dest_ptr, meta.total_bytes), "failed to read tensor data");
          if (target_storage == nvidia::gxf::MemoryStorageType::kCudaManaged) {
            cudaStreamSynchronize(cuda_stream_);
          }
        }
        break;
      }
      case ComponentType::kMessage: {
        std::vector<uint8_t> msg_data(ch.data_size);
        HOLO_CHECK_READ(endpoint.read(msg_data.data(), ch.data_size),
                        "failed to read Message data");
        const std::vector<uint8_t>* msg_ptr = &msg_data;
        EndpointT msg_ep(msg_ptr);
        auto maybe_cn = deserialize_string(&msg_ep);
        if (!maybe_cn) {
          HOLOSCAN_LOG_ERROR("{}::deserialize: failed to read Message codec name",
                             serializer_name_);
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        }
        auto dfunc = registry.get_deserializer(maybe_cn.value());
        auto maybe_msg = dfunc(&msg_ep);
        if (!maybe_msg) {
          HOLOSCAN_LOG_ERROR("{}::deserialize: failed to deserialize Message payload",
                             serializer_name_);
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        }
        auto mm = entity.add<Message>(comp_name.c_str());
        if (!mm) {
          HOLOSCAN_LOG_ERROR("{}::deserialize: failed to add Message", serializer_name_);
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        }
        *mm.value() = maybe_msg.value();
        break;
      }
      case ComponentType::kMetadataDictionary: {
        std::vector<uint8_t> md(ch.data_size);
        HOLO_CHECK_READ(endpoint.read(md.data(), ch.data_size),
                        "failed to read MetadataDictionary data");
        auto mr = decode_metadata_dictionary_payload(md);
        if (!mr) {
          HOLOSCAN_LOG_ERROR("{}::deserialize: decode_metadata_dictionary_payload failed",
                             serializer_name_);
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        }
        auto mm = entity.add<MetadataDictionary>(comp_name.c_str());
        if (!mm) {
          HOLOSCAN_LOG_ERROR("{}::deserialize: failed to add MetadataDictionary", serializer_name_);
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        }
        *mm.value() = std::move(mr.value());
        break;
      }
      case ComponentType::kMessageLabel: {
        std::vector<uint8_t> ld(ch.data_size);
        HOLO_CHECK_READ(endpoint.read(ld.data(), ch.data_size), "failed to read MessageLabel data");
        auto lr = decode_message_label_payload(ld);
        if (!lr) {
          HOLOSCAN_LOG_ERROR("{}::deserialize: decode_message_label_payload failed",
                             serializer_name_);
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        }
        auto ml = entity.add<MessageLabel>(comp_name.c_str());
        if (!ml) {
          HOLOSCAN_LOG_ERROR("{}::deserialize: failed to add MessageLabel", serializer_name_);
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        }
        *ml.value() = std::move(lr.value());
        break;
      }
      default: {
        HOLOSCAN_LOG_WARN(
            "{}::deserialize: skipping unknown component type {}", serializer_name_, ch.type_id);
        std::vector<uint8_t> skip(ch.data_size);
        HOLO_CHECK_READ(endpoint.read(skip.data(), ch.data_size), "failed to skip unknown data");
        break;
      }
    }
  }

  HOLOSCAN_LOG_DEBUG(
      "{}::deserialize: deserialized {} components", serializer_name_, header.num_components);
  return entity;
}

template <typename EndpointT>
size_t HoloEntitySerializerBase<EndpointT>::estimate_size(nvidia::gxf::Entity entity) {
  if (!ensure_type_ids(entity.context()))
    return 0;
  const auto& tids = detail::cached_tids();

  size_t estimate = sizeof(SerializationHeader);
  auto mc = entity.findAll();
  if (!mc)
    return estimate;
  const auto& components = mc.value();

  for (size_t i = 0; i < components.size(); ++i) {
    const auto& m = components[i];
    if (!m)
      continue;
    const auto& c = m.value();
    const gxf_tid_t tid = c.tid();
    if (detail::tid_equals(tid, tids.timestamp_tid)) {
      estimate += sizeof(ComponentHeader) + 64 + sizeof(int64_t) * 2;
    } else if (detail::tid_equals(tid, tids.tensor_tid)) {
      auto th = nvidia::gxf::Handle<nvidia::gxf::Tensor>::Create(entity.context(), c.cid());
      if (th) {
        estimate += sizeof(ComponentHeader) + 64 + sizeof(TensorMetadata) + th.value()->size();
      }
    } else if (detail::tid_equals(tid, tids.message_tid)) {
      estimate += sizeof(ComponentHeader) + 64 + 4096;
    } else if (detail::tid_equals(tid, tids.metadata_tid)) {
      auto mh = nvidia::gxf::Handle<MetadataDictionary>::Create(entity.context(), c.cid());
      if (mh)
        estimate += sizeof(ComponentHeader) + 64 + mh.value()->size() * 1024;
    } else if (detail::tid_equals(tid, tids.message_label_tid)) {
      estimate += sizeof(ComponentHeader) + 64 + 4096;
    }
  }
  return estimate + (estimate / 10);
}

// --- GPU staging ---

template <typename EndpointT>
nvidia::gxf::Expected<std::vector<uint8_t>>
HoloEntitySerializerBase<EndpointT>::stage_tensor_to_host(const void* gpu_ptr, size_t size) {
  std::vector<uint8_t> buf(size);
  cudaError_t err =
      cudaMemcpyAsync(buf.data(), gpu_ptr, size, cudaMemcpyDeviceToHost, cuda_stream_);
  if (err != cudaSuccess) {
    HOLOSCAN_LOG_ERROR(
        "{}: cudaMemcpyAsync D2H failed: {}", serializer_name_, cudaGetErrorString(err));
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }
  err = cudaStreamSynchronize(cuda_stream_);
  if (err != cudaSuccess) {
    HOLOSCAN_LOG_ERROR(
        "{}: cudaStreamSynchronize failed: {}", serializer_name_, cudaGetErrorString(err));
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }
  return buf;
}

template <typename EndpointT>
nvidia::gxf::Expected<void> HoloEntitySerializerBase<EndpointT>::stage_host_to_device(
    void* gpu_ptr, const void* host_ptr, size_t size) {
  cudaError_t err = cudaMemcpyAsync(gpu_ptr, host_ptr, size, cudaMemcpyHostToDevice, cuda_stream_);
  if (err != cudaSuccess) {
    HOLOSCAN_LOG_ERROR(
        "{}: cudaMemcpyAsync H2D failed: {}", serializer_name_, cudaGetErrorString(err));
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }
  err = cudaStreamSynchronize(cuda_stream_);
  if (err != cudaSuccess) {
    HOLOSCAN_LOG_ERROR(
        "{}: cudaStreamSynchronize failed: {}", serializer_name_, cudaGetErrorString(err));
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }
  return nvidia::gxf::Expected<void>();
}

// --- Native descriptor export/import ---

template <typename EndpointT>
nvidia::gxf::Expected<void> HoloEntitySerializerBase<EndpointT>::export_native_descriptor(
    gxf_uid_t entity_uid, gxf_context_t context,
    nvidia::gxf::Handle<nvidia::gxf::Allocator> allocator,
    nvidia::gxf::NativeDescriptorPayload& out_payload, const std::string& protocol_name) {
  (void)allocator;
  if (!native_adapter_ || !native_adapter_->is_initialized()) {
    HOLOSCAN_LOG_ERROR("{}::export_native_descriptor: adapter not initialized", serializer_name_);
    return nvidia::gxf::Unexpected(GXF_NOT_IMPLEMENTED);
  }
  const std::string selected_protocol =
      protocol_name.empty() ? native_adapter_->default_protocol_name() : protocol_name;
  if (!native_adapter_->supports_protocol(selected_protocol)) {
    return nvidia::gxf::Unexpected(GXF_NOT_IMPLEMENTED);
  }
  if (!ensure_type_ids(context)) {
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }
  const auto& tids = detail::cached_tids();

  auto maybe_entity = nvidia::gxf::Entity::Shared(context, entity_uid);
  if (!maybe_entity)
    return nvidia::gxf::ForwardError(maybe_entity);
  nvidia::gxf::Entity entity = maybe_entity.value();

  auto maybe_components = entity.findAll();
  if (!maybe_components)
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  const auto& components = maybe_components.value();

  struct NativeTensorInfo {
    nvidia::gxf::Handle<nvidia::gxf::Tensor> tensor;
    std::string name;
  };
  std::vector<NativeTensorInfo> native_tensors;

  for (size_t i = 0; i < components.size(); ++i) {
    const auto& mc = components[i];
    if (!mc)
      continue;
    const auto& comp = mc.value();
    if (detail::tid_equals(comp.tid(), tids.tensor_tid)) {
      auto th = nvidia::gxf::Handle<nvidia::gxf::Tensor>::Create(context, comp.cid());
      if (th && native_adapter_->can_export_tensor(*th.value())) {
        native_tensors.push_back({th.value(), comp.name()});
      }
    }
  }
  if (native_tensors.empty()) {
    return nvidia::gxf::Unexpected(GXF_NOT_IMPLEMENTED);
  }

  HOLOSCAN_LOG_TRACE(
      "{}::export_native_descriptor: exporting {} native tensor(s) with protocol '{}'",
      serializer_name_,
      native_tensors.size(),
      selected_protocol);

  auto maybe_sideband = serialize_non_gpu_sideband(entity);
  if (!maybe_sideband)
    return nvidia::gxf::ForwardError(maybe_sideband);
  auto& sideband_bytes = maybe_sideband.value();

  if (native_tensors.size() > std::numeric_limits<uint16_t>::max()) {
    HOLOSCAN_LOG_ERROR("{}::export_native_descriptor: too many native tensors ({})",
                       serializer_name_,
                       native_tensors.size());
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }
  if (sideband_bytes.size() > std::numeric_limits<uint32_t>::max()) {
    HOLOSCAN_LOG_ERROR("{}::export_native_descriptor: sideband too large ({} bytes)",
                       serializer_name_,
                       sideband_bytes.size());
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }
  std::vector<uint8_t> payload_bytes;
  payload_bytes.reserve(1024 + sideband_bytes.size());
  uint16_t num_gpu_tensors = static_cast<uint16_t>(native_tensors.size());
  uint32_t sideband_size = static_cast<uint32_t>(sideband_bytes.size());
  detail::ndpl_append_bytes(payload_bytes, &detail::kNdplMagic, sizeof(detail::kNdplMagic));
  detail::ndpl_append_bytes(payload_bytes, &detail::kNdplVersion, sizeof(detail::kNdplVersion));
  if (!detail::ndpl_append_string(payload_bytes, selected_protocol)) {
    HOLOSCAN_LOG_ERROR("{}::export_native_descriptor: protocol name too long", serializer_name_);
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }
  detail::ndpl_append_bytes(payload_bytes, &num_gpu_tensors, sizeof(num_gpu_tensors));
  detail::ndpl_append_bytes(payload_bytes, &sideband_size, sizeof(sideband_size));
  detail::ndpl_append_bytes(payload_bytes, sideband_bytes.data(), sideband_bytes.size());

  for (auto& ni : native_tensors) {
    uint32_t seq = sequence_counter_.fetch_add(1);
    void* raw_ptr = static_cast<void*>(ni.tensor->pointer());
    auto entity_holder = std::shared_ptr<void>(raw_ptr, [entity_copy = entity](void*) mutable {});
    auto maybe_desc =
        native_adapter_->export_tensor(raw_ptr,
                                       entity_holder,
                                       detail::tensor_metadata_from_tensor(*ni.tensor),
                                       selected_protocol,
                                       seq);
    if (!maybe_desc) {
      HOLOSCAN_LOG_ERROR(
          "{}::export_native_descriptor: failed to export tensor '{}'", serializer_name_, ni.name);
      return nvidia::gxf::ForwardError(maybe_desc);
    }
    auto& desc = maybe_desc.value();
    if (desc.size() > std::numeric_limits<uint32_t>::max()) {
      HOLOSCAN_LOG_ERROR("{}::export_native_descriptor: descriptor too large for tensor '{}'",
                         serializer_name_,
                         ni.name);
      return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
    }
    uint32_t desc_size = static_cast<uint32_t>(desc.size());
    detail::ndpl_append_bytes(payload_bytes, &desc_size, sizeof(desc_size));
    detail::ndpl_append_bytes(payload_bytes, desc.data(), desc.size());
    if (!detail::ndpl_append_string(payload_bytes, ni.name)) {
      HOLOSCAN_LOG_ERROR(
          "{}::export_native_descriptor: tensor name too long '{}'", serializer_name_, ni.name);
      return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
    }
  }

  out_payload.descriptor_bytes = std::move(payload_bytes);
  out_payload.source_entity_uid = entity_uid;
  out_payload.descriptor_format_version = native_adapter_->descriptor_format_version();
  out_payload.protocol_name = selected_protocol;
  HOLOSCAN_LOG_TRACE(
      "{}::export_native_descriptor: emitted native descriptor payload (protocol='{}', "
      "tensor_count={}, sideband_bytes={})",
      serializer_name_,
      selected_protocol,
      native_tensors.size(),
      sideband_bytes.size());
  return nvidia::gxf::Expected<void>();
}

template <typename EndpointT>
nvidia::gxf::Expected<nvidia::gxf::Entity>
HoloEntitySerializerBase<EndpointT>::import_native_descriptor(
    const nvidia::gxf::NativeDescriptorPayload& payload, gxf_context_t context,
    nvidia::gxf::Handle<nvidia::gxf::Allocator> allocator) {
  if (!native_adapter_ || !native_adapter_->is_initialized()) {
    HOLOSCAN_LOG_ERROR("{}::import_native_descriptor: adapter not initialized", serializer_name_);
    return nvidia::gxf::Unexpected(GXF_NOT_IMPLEMENTED);
  }
  const auto& data = payload.descriptor_bytes;
  size_t offset = 0;
  auto read_field = [&data, &offset](void* dest, size_t sz) -> bool {
    if (offset + sz > data.size())
      return false;
    std::memcpy(dest, data.data() + offset, sz);
    offset += sz;
    return true;
  };

  uint32_t magic = 0;
  uint8_t version = 0;
  std::string ndpl_protocol;
  uint16_t num_gpu_tensors = 0;
  uint32_t sideband_size = 0;

  if (!read_field(&magic, sizeof(magic)) || magic != detail::kNdplMagic) {
    HOLOSCAN_LOG_ERROR("{}::import_native_descriptor: invalid NDPL magic", serializer_name_);
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }
  if (!read_field(&version, sizeof(version)) || version != detail::kNdplVersion) {
    HOLOSCAN_LOG_ERROR("{}::import_native_descriptor: unsupported NDPL version", serializer_name_);
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }
  if (!detail::ndpl_read_string(data, offset, ndpl_protocol) || ndpl_protocol.empty()) {
    HOLOSCAN_LOG_ERROR("{}::import_native_descriptor: missing protocol name", serializer_name_);
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }
  if (!read_field(&num_gpu_tensors, sizeof(num_gpu_tensors)) ||
      !read_field(&sideband_size, sizeof(sideband_size))) {
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }
  if (offset + sideband_size > data.size()) {
    HOLOSCAN_LOG_ERROR("{}::import_native_descriptor: truncated sideband", serializer_name_);
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }
  if (payload.protocol_name.empty() || payload.protocol_name != ndpl_protocol) {
    HOLOSCAN_LOG_ERROR("{}::import_native_descriptor: protocol mismatch ('{}' vs NDPL '{}')",
                       serializer_name_,
                       payload.protocol_name,
                       ndpl_protocol);
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }
  if (!native_adapter_->supports_protocol(payload.protocol_name)) {
    HOLOSCAN_LOG_ERROR("{}::import_native_descriptor: unsupported protocol '{}'",
                       serializer_name_,
                       payload.protocol_name);
    return nvidia::gxf::Unexpected(GXF_NOT_IMPLEMENTED);
  }

  const uint8_t* sideband_data = data.data() + offset;
  offset += sideband_size;

  auto maybe_entity = nvidia::gxf::Entity::New(context);
  if (!maybe_entity)
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  nvidia::gxf::Entity entity = maybe_entity.value();

  if (sideband_size > 0) {
    auto result = deserialize_non_gpu_sideband(sideband_data, sideband_size, entity, allocator);
    if (!result)
      return nvidia::gxf::ForwardError(result);
  }

  for (uint16_t t = 0; t < num_gpu_tensors; ++t) {
    uint32_t desc_size = 0;
    if (!read_field(&desc_size, sizeof(desc_size)) || offset + desc_size > data.size()) {
      HOLOSCAN_LOG_ERROR("{}::import_native_descriptor: truncated descriptor", serializer_name_);
      return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
    }
    std::vector<uint8_t> desc_bytes(data.data() + offset, data.data() + offset + desc_size);
    offset += desc_size;

    uint16_t name_len = 0;
    if (!read_field(&name_len, sizeof(name_len)) || offset + name_len > data.size()) {
      return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
    }
    std::string tensor_name(reinterpret_cast<const char*>(data.data() + offset), name_len);
    offset += name_len;

    auto maybe_imported = native_adapter_->import_tensor_generic(desc_bytes, payload.protocol_name);
    if (!maybe_imported) {
      if (maybe_imported.error() != GXF_PUBSUB_DESCRIPTOR_EXPIRED) {
        HOLOSCAN_LOG_ERROR("{}::import_native_descriptor: failed to import tensor '{}'",
                           serializer_name_,
                           tensor_name);
      }
      return nvidia::gxf::ForwardError(maybe_imported);
    }
    auto imported = std::make_shared<ImportedNativeTensor>(std::move(maybe_imported.value()));

    auto mt = entity.add<nvidia::gxf::Tensor>(tensor_name.c_str());
    if (!mt)
      return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
    auto tensor = mt.value();

    auto ms = detail::shape_from_metadata(imported->metadata);
    if (!ms)
      return nvidia::gxf::ForwardError(ms);
    const auto mst = imported->metadata.strides.empty()
                         ? nvidia::gxf::Expected<nvidia::gxf::Tensor::stride_array_t>(
                               nvidia::gxf::ComputeTrivialStrides(
                                   ms.value(), imported->metadata.bytes_per_element))
                         : detail::strides_from_metadata(imported->metadata);
    if (!mst)
      return nvidia::gxf::ForwardError(mst);

    auto wr = tensor->wrapMemory(
        ms.value(),
        detail::dtype_string_to_primitive_type(imported->metadata.dtype),
        imported->metadata.bytes_per_element,
        mst,
        imported->metadata.storage_type,
        imported->data(),
        [imported](void*) -> nvidia::gxf::Expected<void> { return nvidia::gxf::Expected<void>(); });
    if (!wr) {
      HOLOSCAN_LOG_ERROR("{}::import_native_descriptor: wrapMemory failed for '{}'",
                         serializer_name_,
                         tensor_name);
      return nvidia::gxf::ForwardError(wr);
    }
  }
  return entity;
}

// --- Non-GPU sideband serialization ---

template <typename EndpointT>
nvidia::gxf::Expected<std::vector<uint8_t>>
HoloEntitySerializerBase<EndpointT>::serialize_non_gpu_sideband(nvidia::gxf::Entity entity) {
  if (!ensure_type_ids(entity.context())) {
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  }
  const auto& tids = detail::cached_tids();

  std::vector<uint8_t> buffer;
  buffer.reserve(1024);
  EndpointT endpoint(&buffer);

  auto mc = entity.findAll();
  if (!mc)
    return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
  const auto& components = mc.value();

  // Count non-GPU components
  uint32_t count = 0;
  for (size_t i = 0; i < components.size(); ++i) {
    const auto& m = components[i];
    if (!m)
      continue;
    const gxf_tid_t tid = m.value().tid();
    if (detail::tid_equals(tid, tids.timestamp_tid) || detail::tid_equals(tid, tids.message_tid) ||
        detail::tid_equals(tid, tids.metadata_tid) ||
        detail::tid_equals(tid, tids.message_label_tid)) {
      count++;
    } else if (detail::tid_equals(tid, tids.tensor_tid)) {
      auto th = nvidia::gxf::Handle<nvidia::gxf::Tensor>::Create(entity.context(), m.value().cid());
      if (th && th.value()->storage_type() != nvidia::gxf::MemoryStorageType::kDevice) {
        count++;
      }
    }
  }
  endpoint.write_trivial_type(&count);

  for (size_t i = 0; i < components.size(); ++i) {
    const auto& m = components[i];
    if (!m)
      continue;
    const auto& comp = m.value();
    const gxf_tid_t tid = comp.tid();

    if (detail::tid_equals(tid, tids.timestamp_tid)) {
      auto th = nvidia::gxf::Handle<nvidia::gxf::Timestamp>::Create(entity.context(), comp.cid());
      if (!th)
        continue;
      auto ts = th.value();
      ComponentHeader ch;
      ch.type_id = static_cast<uint32_t>(ComponentType::kTimestamp);
      ch.name_length = std::strlen(comp.name());
      ch.data_size = sizeof(int64_t) * 2;
      ch.flags = 0;
      endpoint.write_trivial_type(&ch);
      endpoint.write(comp.name(), ch.name_length);
      endpoint.write_trivial_type(&ts->acqtime);
      endpoint.write_trivial_type(&ts->pubtime);

    } else if (detail::tid_equals(tid, tids.message_tid)) {
      auto mh = nvidia::gxf::Handle<Message>::Create(entity.context(), comp.cid());
      if (!mh)
        continue;
      auto encoded = encode_message_payload(*mh.value());
      if (!encoded)
        continue;
      std::vector<uint8_t> msg_buf;
      EndpointT msg_ep(&msg_buf);
      serialize_string(encoded.value().codec_name, &msg_ep);
      msg_ep.write(encoded.value().payload.data(), encoded.value().payload.size());
      ComponentHeader ch;
      ch.type_id = static_cast<uint32_t>(ComponentType::kMessage);
      ch.name_length = std::strlen(comp.name());
      ch.data_size = static_cast<uint32_t>(msg_buf.size());
      ch.flags = 0;
      endpoint.write_trivial_type(&ch);
      endpoint.write(comp.name(), ch.name_length);
      endpoint.write(msg_buf.data(), msg_buf.size());

    } else if (detail::tid_equals(tid, tids.metadata_tid)) {
      auto mh = nvidia::gxf::Handle<MetadataDictionary>::Create(entity.context(), comp.cid());
      if (!mh)
        continue;
      auto mr = encode_metadata_dictionary_payload(*mh.value());
      if (!mr)
        continue;
      auto& meta_buf = mr.value();
      ComponentHeader ch;
      ch.type_id = static_cast<uint32_t>(ComponentType::kMetadataDictionary);
      ch.name_length = std::strlen(comp.name());
      ch.data_size = static_cast<uint32_t>(meta_buf.size());
      ch.flags = 0;
      endpoint.write_trivial_type(&ch);
      endpoint.write(comp.name(), ch.name_length);
      endpoint.write(meta_buf.data(), meta_buf.size());

    } else if (detail::tid_equals(tid, tids.message_label_tid)) {
      auto lh = nvidia::gxf::Handle<MessageLabel>::Create(entity.context(), comp.cid());
      if (!lh)
        continue;
      auto lr = encode_message_label_payload(*lh.value());
      if (!lr)
        continue;
      auto& label_buf = lr.value();
      ComponentHeader ch;
      ch.type_id = static_cast<uint32_t>(ComponentType::kMessageLabel);
      ch.name_length = std::strlen(comp.name());
      ch.data_size = static_cast<uint32_t>(label_buf.size());
      ch.flags = 0;
      endpoint.write_trivial_type(&ch);
      endpoint.write(comp.name(), ch.name_length);
      endpoint.write(label_buf.data(), label_buf.size());

    } else if (detail::tid_equals(tid, tids.tensor_tid)) {
      auto th = nvidia::gxf::Handle<nvidia::gxf::Tensor>::Create(entity.context(), comp.cid());
      if (!th)
        continue;
      auto tensor = th.value();
      if (tensor->storage_type() == nvidia::gxf::MemoryStorageType::kDevice)
        continue;
      const size_t tsz = tensor->size();
      if (tensor->storage_type() == nvidia::gxf::MemoryStorageType::kCudaManaged) {
        cudaStreamSynchronize(cuda_stream_);
      }
      TensorMetadata meta;
      meta.rank = tensor->rank();
      meta.element_type = primitive_type_to_uint(tensor->element_type());
      meta.storage_type = storage_type_to_uint(tensor->storage_type());
      meta.bytes_per_element = tensor->bytes_per_element();
      meta.total_bytes = tsz;
      auto shape = tensor->shape();
      for (uint32_t d = 0; d < meta.rank && d < 8; ++d) {
        meta.shape[d] = shape.dimension(d);
      }
      ComponentHeader ch;
      ch.type_id = static_cast<uint32_t>(ComponentType::kTensor);
      ch.name_length = std::strlen(comp.name());
      ch.data_size = sizeof(TensorMetadata) + tsz;
      ch.flags = 0;
      endpoint.write_trivial_type(&ch);
      endpoint.write(comp.name(), ch.name_length);
      endpoint.write_trivial_type(&meta);
      endpoint.write(tensor->pointer(), tsz);
    }
  }
  return buffer;
}

template <typename EndpointT>
nvidia::gxf::Expected<void> HoloEntitySerializerBase<EndpointT>::deserialize_non_gpu_sideband(
    const uint8_t* data, size_t size, nvidia::gxf::Entity& entity,
    nvidia::gxf::Handle<nvidia::gxf::Allocator> allocator) {
  const std::vector<uint8_t> data_vec(data, data + size);
  EndpointT endpoint(&data_vec);

  uint32_t count = 0;
  HOLO_CHECK_READ(endpoint.read_trivial_type(&count), "sideband: failed to read component count");
  auto& registry = CodecRegistry::get_instance();

  for (uint32_t i = 0; i < count; ++i) {
    ComponentHeader ch;
    HOLO_CHECK_READ(endpoint.read_trivial_type(&ch), "sideband: failed to read header");
    std::string cn(ch.name_length, '\0');
    if (ch.name_length > 0) {
      HOLO_CHECK_READ(endpoint.read(cn.data(), ch.name_length), "sideband: failed to read name");
    }
    ComponentType type = static_cast<ComponentType>(ch.type_id);
    switch (type) {
      case ComponentType::kTimestamp: {
        auto mt = entity.add<nvidia::gxf::Timestamp>(cn.c_str());
        if (!mt)
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        auto ts = mt.value();
        HOLO_CHECK_READ(endpoint.read_trivial_type(&ts->acqtime), "sideband: ts acqtime");
        HOLO_CHECK_READ(endpoint.read_trivial_type(&ts->pubtime), "sideband: ts pubtime");
        break;
      }
      case ComponentType::kMessage: {
        std::vector<uint8_t> msg_data(ch.data_size);
        HOLO_CHECK_READ(endpoint.read(msg_data.data(), ch.data_size), "sideband: msg data");
        const std::vector<uint8_t>* mp = &msg_data;
        EndpointT msg_ep(mp);
        auto mcn = deserialize_string(&msg_ep);
        if (!mcn)
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        auto df = registry.get_deserializer(mcn.value());
        auto mm = df(&msg_ep);
        if (!mm)
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        auto me = entity.add<Message>(cn.c_str());
        if (!me)
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        *me.value() = mm.value();
        break;
      }
      case ComponentType::kMetadataDictionary: {
        std::vector<uint8_t> md(ch.data_size);
        HOLO_CHECK_READ(endpoint.read(md.data(), ch.data_size), "sideband: meta data");
        auto mr = decode_metadata_dictionary_payload(md);
        if (!mr)
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        auto me = entity.add<MetadataDictionary>(cn.c_str());
        if (!me)
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        *me.value() = std::move(mr.value());
        break;
      }
      case ComponentType::kMessageLabel: {
        std::vector<uint8_t> ld(ch.data_size);
        HOLO_CHECK_READ(endpoint.read(ld.data(), ch.data_size), "sideband: label data");
        auto lr = decode_message_label_payload(ld);
        if (!lr)
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        auto me = entity.add<MessageLabel>(cn.c_str());
        if (!me)
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        *me.value() = std::move(lr.value());
        break;
      }
      case ComponentType::kTensor: {
        TensorMetadata meta;
        HOLO_CHECK_READ(endpoint.read_trivial_type(&meta), "sideband: TensorMetadata");
        if (meta.total_bytes > kMaxTensorBytes || meta.rank > 8) {
          HOLOSCAN_LOG_ERROR("{}::deserialize_non_gpu_sideband: invalid tensor metadata",
                             serializer_name_);
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        }
        {
          const uint64_t min_bytes =
              detail::safe_tensor_element_bytes(meta.shape, meta.rank, meta.bytes_per_element);
          if (min_bytes == 0 || meta.total_bytes < min_bytes) {
            HOLOSCAN_LOG_ERROR(
                "{}::deserialize_non_gpu_sideband: tensor shape/bytes_per_element inconsistent "
                "with total_bytes (min_bytes={}, total_bytes={})",
                serializer_name_,
                min_bytes,
                meta.total_bytes);
            return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
          }
        }
        std::vector<uint8_t> td(meta.total_bytes);
        HOLO_CHECK_READ(endpoint.read(td.data(), meta.total_bytes), "sideband: tensor data");
        std::array<int32_t, nvidia::gxf::Shape::kMaxRank> dims = {};
        for (uint32_t d = 0; d < meta.rank && d < 8; ++d) {
          dims[d] = static_cast<int32_t>(meta.shape[d]);
        }
        nvidia::gxf::Shape shape(dims, meta.rank);
        auto mt = entity.add<nvidia::gxf::Tensor>(cn.c_str());
        if (!mt)
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        auto tensor = mt.value();
        auto rr =
            tensor->reshapeCustom(shape,
                                  uint_to_primitive_type(meta.element_type),
                                  meta.bytes_per_element,
                                  nvidia::gxf::ComputeTrivialStrides(shape, meta.bytes_per_element),
                                  uint_to_storage_type(meta.storage_type),
                                  allocator);
        if (!rr) {
          HOLOSCAN_LOG_ERROR("{}::deserialize_non_gpu_sideband: reshapeCustom failed",
                             serializer_name_);
          return nvidia::gxf::ForwardError(rr);
        }
        if (meta.total_bytes > tensor->size()) {
          HOLOSCAN_LOG_ERROR(
              "{}::deserialize_non_gpu_sideband: total_bytes ({}) exceeds "
              "allocated tensor size ({})",
              serializer_name_,
              meta.total_bytes,
              tensor->size());
          return nvidia::gxf::Unexpected(GXF_PUBSUB_SERIALIZATION_FAILED);
        }
        std::memcpy(tensor->pointer(), td.data(), meta.total_bytes);
        break;
      }
      default: {
        std::vector<uint8_t> skip(ch.data_size);
        HOLO_CHECK_READ(endpoint.read(skip.data(), ch.data_size), "sideband: skip unknown");
        break;
      }
    }
  }
  return nvidia::gxf::Expected<void>();
}

}  // namespace holoscan

#undef HOLO_CHECK_READ
#undef HOLO_CHECK_WRITE

#endif /* PUBSUB_COMMON_INCLUDE_PUBSUB_HOLO_ENTITY_SERIALIZER_BASE_HPP */
