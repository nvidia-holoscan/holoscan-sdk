/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

#include <flatbuffers/flatbuffers.h>

#include "holoscan/core/port.hpp"
#include "holoscan/core/predicate_input_view.hpp"
#include "holoscan/holoscan_schema_export.h"
#include "holoscan/schema/descriptor_flatbuffer_payload.hpp"
#include "holoscan/schema/flatbuffer_schema_traits.hpp"
#include "holoscan/schema/schema_package.hpp"
#include "holoscan/schema/tensor_materialization.hpp"
#include "holoscan/schema/tensor_packet_metadata.hpp"

#include "image_bfbs_generated.h"
#include "image_generated.h"
#include "imu_bfbs_generated.h"
#include "imu_generated.h"
#include "string_bfbs_generated.h"
#include "string_generated.h"
#include "tensor_bfbs_generated.h"
#include "tensor_generated.h"
#include "tensor_packet_bfbs_generated.h"
#include "tensor_packet_generated.h"
#include "time_bfbs_generated.h"
#include "time_generated.h"

namespace holoscan::schema {

namespace detail {

inline constexpr std::size_t kMaxDescriptorTensorMetadataCapacity = 64U * 1024U;

template <typename T>
[[nodiscard]] bool read_descriptor_wire_value(const std::span<const std::byte> source,
                                              const std::size_t offset, T& value) noexcept {
  static_assert(std::is_trivially_copyable_v<T>);
  if (offset > source.size() || sizeof(T) > source.size() - offset) {
    return false;
  }
  std::memcpy(&value, source.data() + offset, sizeof(T));
  return true;
}

template <typename T>
[[nodiscard]] bool write_descriptor_wire_value(const std::span<std::byte> destination,
                                               const std::size_t offset, const T& value) noexcept {
  static_assert(std::is_trivially_copyable_v<T>);
  if (offset > destination.size() || sizeof(T) > destination.size() - offset) {
    return false;
  }
  std::memcpy(destination.data() + offset, &value, sizeof(T));
  return true;
}

// FlatBuffers allocator that admits exactly one caller-owned bounded span.
class BoundedSpanAllocator final : public flatbuffers::Allocator {
 public:
  explicit BoundedSpanAllocator(std::span<std::byte> storage) noexcept : storage_(storage) {}

  std::uint8_t* allocate(std::size_t size) override {
    if (allocated_ || size == 0U || size > storage_.size()) {
      throw std::bad_alloc{};
    }
    allocated_ = true;
    return reinterpret_cast<std::uint8_t*>(storage_.data());
  }

  void deallocate(std::uint8_t* pointer, std::size_t size) override {
    static_cast<void>(pointer);
    static_cast<void>(size);
    allocated_ = false;
  }

  std::uint8_t* reallocate_downward(std::uint8_t* old_pointer, std::size_t old_size,
                                    std::size_t new_size, std::size_t in_use_back,
                                    std::size_t in_use_front) override {
    static_cast<void>(old_pointer);
    static_cast<void>(old_size);
    static_cast<void>(new_size);
    static_cast<void>(in_use_back);
    static_cast<void>(in_use_front);
    throw std::bad_alloc{};
  }

 private:
  std::span<std::byte> storage_;
  bool allocated_{};
};

inline constexpr std::array<std::string_view, 13> kSdkSchemaFlatcFlags{
    "--bfbs-gen-embed",
    "--cpp",
    "--cpp-ptr-type",
    "std::shared_ptr",
    "--gen-object-api",
    "--reflect-names",
    "--reflect-types",
    "--schema",
    "-o",
    "<generated-schema-dir>",
    "-I",
    "<schema-root-0>",
    "<ordered-root-schema-files>",
};

inline constexpr SchemaPackageSource kTimeSource{
    .logical_path = "time.fbs",
    .content_sha256 =
        schema_hex_bytes<32>("51d9cc42ca6d3ca4589eac955767c9ea772512128c72a2ec299254bc87f5855a"),
};
inline constexpr SchemaPackageSource kHeaderSource{
    .logical_path = "header.fbs",
    .content_sha256 =
        schema_hex_bytes<32>("30355c7ec1853367a9fce23a983ce45509fa8bd3da23411fc0628f960c351ca2"),
};
inline constexpr SchemaPackageSource kTensorSource{
    .logical_path = "tensor.fbs",
    .content_sha256 =
        schema_hex_bytes<32>("1f6a3fb98deab8fe9d3d2c47f4e8c44e3b1b4b2c011f5472cecaf88df607bf76"),
};

inline constexpr std::array<SchemaPackageSource, 3> kImageIncludes{
    kTensorSource,
    kHeaderSource,
    kTimeSource,
};
inline constexpr std::array<SchemaPackageSource, 1> kTensorPacketIncludes{kTensorSource};
inline constexpr std::array<SchemaPackageSource, 2> kImuIncludes{
    kHeaderSource,
    kTimeSource,
};
inline constexpr std::array<SchemaPackageSource, 0> kNoIncludes{};

template <typename Root>
[[nodiscard]] bool verify_flatbuffer_root(std::span<const std::byte> bytes) noexcept {
  if (bytes.empty()) {
    return false;
  }
  flatbuffers::Verifier verifier{reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()};
  return verifier.VerifyBuffer<Root>(nullptr);
}

template <typename Root>
[[nodiscard]] const Root* parse_flatbuffer_root(std::span<const std::byte> bytes) noexcept {
  if (bytes.empty()) {
    return nullptr;
  }
  return flatbuffers::GetRoot<Root>(reinterpret_cast<const std::uint8_t*>(bytes.data()));
}

template <typename Native, typename Root>
[[nodiscard]] expected<std::size_t, Error> encode_flatbuffer_native(
    const Native& value, std::span<std::byte> destination) noexcept {
  if (destination.empty() ||
      reinterpret_cast<std::uintptr_t>(destination.data()) % alignof(std::max_align_t) != 0U) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "schema serialization storage is empty or unaligned"});
  }
  try {
    BoundedSpanAllocator allocator{destination};
    flatbuffers::FlatBufferBuilder builder{destination.size(), &allocator, false};
    const auto root = Root::Pack(builder, &value);
    if (root.IsNull()) {
      return make_unexpected(Error{ErrorCode::kCodecError, "generated schema pack failed"});
    }
    builder.Finish(root);
    const std::size_t size = builder.GetSize();
    if (size == 0U || size > destination.size()) {
      return make_unexpected(
          Error{ErrorCode::kResourceExhausted, "serialized schema exceeds its package bound"});
    }
    std::memmove(destination.data(), builder.GetBufferPointer(), size);
    return size;
  } catch (const std::bad_alloc&) {
    return make_unexpected(
        Error{ErrorCode::kResourceExhausted, "serialized schema exceeds bounded storage"});
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kCodecError, "generated schema pack failed"});
  }
}

template <typename Native, typename Root>
[[nodiscard]] expected<Native, Error> materialize_flatbuffer_native(
    std::span<const std::byte> bytes) noexcept {
  try {
    const Root* root = parse_flatbuffer_root<Root>(bytes);
    if (root == nullptr) {
      return make_unexpected(Error{ErrorCode::kCodecError, "generated schema root is absent"});
    }
    Native value{};
    root->UnPackTo(&value);
    return value;
  } catch (const std::bad_alloc&) {
    return make_unexpected(
        Error{ErrorCode::kResourceExhausted, "generated schema materialization allocation failed"});
  } catch (...) {
    return make_unexpected(
        Error{ErrorCode::kCodecError, "generated schema materialization failed"});
  }
}

template <typename BinarySchema>
[[nodiscard]] std::span<const std::byte> embedded_binary_schema() noexcept {
  return {reinterpret_cast<const std::byte*>(BinarySchema::data()), BinarySchema::size()};
}

template <typename Root, typename BinarySchema>
[[nodiscard]] SchemaPackageDescriptor<Root> make_flatbuffer_package_from_static_evidence(
    SchemaIdentity identity, SchemaPackageEvidence evidence) noexcept {
  return SchemaPackageDescriptor<Root>{identity,
                                       embedded_binary_schema<BinarySchema>(),
                                       evidence,
                                       &verify_flatbuffer_root<Root>,
                                       &parse_flatbuffer_root<Root>};
}

template <typename Native, typename Root, typename BinarySchema>
  requires holoscan::detail::HasGeneratedSchemaNativeTraits<Native> &&
           std::same_as<typename GeneratedSchemaNativeTraits<Native>::root_type, Root>
[[nodiscard]] SchemaPackageDescriptor<Native> make_flatbuffer_native_package_from_static_evidence(
    SchemaIdentity identity, SchemaPackageEvidence evidence) noexcept {
  return SchemaPackageDescriptor<Native>{identity,
                                         embedded_binary_schema<BinarySchema>(),
                                         evidence,
                                         &verify_flatbuffer_root<Root>,
                                         generated_schema_max_serialized_size_v<Native>,
                                         &encode_flatbuffer_native<Native, Root>,
                                         &materialize_flatbuffer_native<Native, Root>};
}

template <typename Root, typename BinarySchema, std::size_t IncludeCount>
[[nodiscard]] SchemaPackageDescriptor<Root> make_sdk_flatbuffer_package(
    std::string_view root_path, std::span<const SchemaPackageSource, IncludeCount> includes) {
  return SchemaPackageDescriptor<Root>{
      schema_identity<Root>(),
      embedded_binary_schema<BinarySchema>(),
      SchemaPackageEvidence{
          .flatc_build = "flatc 25.12.19",
          .ordered_flags = kSdkSchemaFlatcFlags,
          .root_logical_path = root_path,
          .includes = includes,
          .include_root_mapping = "schema-root-0=.",
          .host_paths_stripped = true,
          .generator_format_version = 1U,
          .package_format_version = 1U,
          .runtime_abi = FlatBufferRuntimeABI{.major = 25U, .minor = 12U, .revision = 19U},
      },
      &verify_flatbuffer_root<Root>,
      &parse_flatbuffer_root<Root>,
  };
}

template <typename Native, typename Root, typename BinarySchema, std::size_t IncludeCount>
[[nodiscard]] SchemaPackageDescriptor<Native> make_sdk_flatbuffer_native_package(
    std::string_view root_path, std::span<const SchemaPackageSource, IncludeCount> includes) {
  return SchemaPackageDescriptor<Native>{
      schema_identity<Native>(),
      embedded_binary_schema<BinarySchema>(),
      SchemaPackageEvidence{
          .flatc_build = "flatc 25.12.19",
          .ordered_flags = kSdkSchemaFlatcFlags,
          .root_logical_path = root_path,
          .includes = includes,
          .include_root_mapping = "schema-root-0=.",
          .host_paths_stripped = true,
          .generator_format_version = 1U,
          .package_format_version = 1U,
          .runtime_abi = FlatBufferRuntimeABI{.major = 25U, .minor = 12U, .revision = 19U},
      },
      &verify_flatbuffer_root<Root>,
      generated_schema_max_serialized_size_v<Native>,
      &encode_flatbuffer_native<Native, Root>,
      &materialize_flatbuffer_native<Native, Root>,
  };
}

}  // namespace detail
}  // namespace holoscan::schema

namespace holoscan::schema {

/// @brief Build a generated FlatBuffers root package from caller-supplied evidence.
///
/// This is the supported package-construction surface for a generated table root used as a
/// borrowed verified view. The returned descriptor deep-copies and owns every string and span in
/// \p evidence; the embedded binary schema remains generated package-static storage.
///
/// @tparam Root Generated FlatBuffers table root.
/// @tparam BinarySchema Generated embedded `.bfbs` provider.
/// @param identity Complete identity returned by the root's `schema_identity` specialization.
/// @param evidence Reproducible caller-owned generator and Runtime evidence.
/// @return Package descriptor using the generated verifier and borrowed-root parser.
template <typename Root, typename BinarySchema>
[[nodiscard]] SchemaPackageDescriptor<Root> make_flatbuffer_package(
    SchemaIdentity identity, const SchemaPackageEvidence& evidence) {
  return SchemaPackageDescriptor<Root>::with_owned_evidence(
      identity,
      detail::embedded_binary_schema<BinarySchema>(),
      evidence,
      &detail::verify_flatbuffer_root<Root>,
      &detail::parse_flatbuffer_root<Root>);
}

/// @brief Build an owning object-API FlatBuffers package from caller-supplied evidence.
///
/// This is the supported package-construction surface for an application-defined generated
/// object-API native type. Serialization is bounded by
/// `GeneratedSchemaNativeTraits<Native>::max_serialized_size`; verification always precedes
/// owning materialization. The returned descriptor deep-copies and owns every string and span in
/// \p evidence; the embedded binary schema remains generated package-static storage.
///
/// @tparam Native Generated object-API owning type.
/// @tparam Root Generated FlatBuffers table root named by the native traits.
/// @tparam BinarySchema Generated embedded `.bfbs` provider.
/// @param identity Complete identity returned by the native type's `schema_identity`
/// specialization.
/// @param evidence Reproducible caller-owned generator and Runtime evidence.
/// @return Package descriptor using the generic bounded owning-value codec.
template <typename Native, typename Root, typename BinarySchema>
  requires holoscan::detail::HasGeneratedSchemaNativeTraits<Native> &&
           std::same_as<typename GeneratedSchemaNativeTraits<Native>::root_type, Root>
[[nodiscard]] SchemaPackageDescriptor<Native> make_flatbuffer_native_package(
    SchemaIdentity identity, const SchemaPackageEvidence& evidence) {
  return SchemaPackageDescriptor<Native>::with_owned_evidence(
      identity,
      detail::embedded_binary_schema<BinarySchema>(),
      evidence,
      &detail::verify_flatbuffer_root<Root>,
      generated_schema_max_serialized_size_v<Native>,
      &detail::encode_flatbuffer_native<Native, Root>,
      &detail::materialize_flatbuffer_native<Native, Root>);
}

/// @brief Borrow an indexed collection of authoritative descriptor-schema Tensors.
///
/// The view and every referenced Tensor remain owned by the caller. A schema metadata encoder may
/// inspect the Tensors only for the duration of its callback.
class DescriptorTensorReferences {
 public:
  /// @brief Construct a borrowed view over non-null Tensor pointers.
  /// @param tensors Ordered Tensor pointers whose pointees outlive this view.
  explicit DescriptorTensorReferences(std::span<const holoscan::Tensor* const> tensors) noexcept
      : tensors_(tensors) {}

  /// @brief Return the number of indexed Tensor fields.
  [[nodiscard]] std::size_t size() const noexcept { return tensors_.size(); }

  /// @brief Return whether the collection is empty.
  [[nodiscard]] bool empty() const noexcept { return tensors_.empty(); }

  /// @brief Return one Tensor pointer, or nullptr for an invalid index.
  [[nodiscard]] const holoscan::Tensor* at(std::size_t index) const noexcept {
    return index < tensors_.size() ? tensors_[index] : nullptr;
  }

 private:
  std::span<const holoscan::Tensor* const> tensors_;
};

namespace detail {

template <typename Root, std::size_t MetadataCapacity, std::size_t MaxTensors>
using DescriptorTensorRuntimePayload =
    DescriptorFlatBufferPayload<Root, MetadataCapacity, MaxTensors>;

template <typename Root, std::size_t MetadataCapacity, std::size_t MaxTensors>
struct DescriptorTensorRuntimeEnvelope {
  SampleMetadata metadata{};
  DescriptorTensorRuntimePayload<Root, MetadataCapacity, MaxTensors> payload{};
};

[[nodiscard]] constexpr bool sample_metadata_equal(const SampleMetadata& lhs,
                                                   const SampleMetadata& rhs) noexcept {
  return lhs.version.major == rhs.version.major && lhs.version.minor == rhs.version.minor &&
         lhs.sequence == rhs.sequence && lhs.capture_timestamp_ns == rhs.capture_timestamp_ns &&
         lhs.source_clock_id == rhs.source_clock_id &&
         lhs.publish_timestamp_ns == rhs.publish_timestamp_ns &&
         lhs.publish_clock_id == rhs.publish_clock_id && lhs.trace_id == rhs.trace_id &&
         lhs.frame_id == rhs.frame_id && lhs.byte_size == rhs.byte_size &&
         lhs.payload_kind == rhs.payload_kind && lhs.flags == rhs.flags;
}

template <typename Native>
[[nodiscard]] constexpr std::size_t descriptor_tensor_count(const Native&) noexcept {
  return holoscan::detail::descriptor_tensor_schema_tensor_count_v<Native>;
}

template <typename Native>
[[nodiscard]] const holoscan::Tensor* descriptor_tensor_at(const Native& value,
                                                           std::size_t index) noexcept {
  return holoscan::detail::descriptor_tensor_schema_tensor_at(value, index);
}

template <typename Root>
[[nodiscard]] expected<TensorViewMetadata, Error> descriptor_tensor_view_metadata_at_from_root(
    const Root& root, const std::span<const std::byte> metadata_bytes, const std::size_t index) {
  if constexpr (requires {
                  {
                    DescriptorFlatBufferTensorViewTraits<Root>::tensor_view_metadata_at(root, index)
                  } -> std::same_as<expected<TensorViewMetadata, Error>>;
                }) {
    return DescriptorFlatBufferTensorViewTraits<Root>::tensor_view_metadata_at(root, index);
  } else {
    if (index != 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if constexpr (requires {
                    {
                      DescriptorFlatBufferTensorViewTraits<Root>::tensor_view_metadata(root)
                    } -> std::same_as<expected<TensorViewMetadata, Error>>;
                  }) {
      return DescriptorFlatBufferTensorViewTraits<Root>::tensor_view_metadata(root);
    } else {
      return DescriptorFlatBufferTensorViewTraits<Root>::tensor_view_metadata(metadata_bytes);
    }
  }
}

template <typename Native, typename Root, std::size_t MetadataCapacity, typename Prepare,
          typename Reset, typename Encode, typename Assign>
struct DescriptorTensorPortTypeOpsFactory {
  static constexpr std::size_t kTensorCount =
      holoscan::detail::descriptor_tensor_schema_tensor_count_v<Native>;
  static constexpr std::uint64_t kAllTensorsMask = kTensorCount == 64U
                                                       ? std::numeric_limits<std::uint64_t>::max()
                                                       : (std::uint64_t{1U} << kTensorCount) - 1U;
  using Payload = DescriptorTensorRuntimePayload<Root, MetadataCapacity, kTensorCount>;
  using Envelope = DescriptorTensorRuntimeEnvelope<Root, MetadataCapacity, kTensorCount>;

  static_assert(kTensorCount > 0U && kTensorCount <= 64U);
  static_assert(std::is_standard_layout_v<Envelope>);
  static_assert(std::is_standard_layout_v<Payload>);
  static_assert(std::is_trivially_copyable_v<Envelope>);
  static_assert(MetadataCapacity <= kMaxDescriptorTensorMetadataCapacity,
                "descriptor Tensor metadata capacity exceeds the 64 KiB Runtime wire bound");
  static_assert(sizeof(Envelope) <= std::numeric_limits<std::uint32_t>::max());
  static_assert(std::is_empty_v<Prepare> && std::default_initializable<Prepare>);
  static_assert(std::is_empty_v<Reset> && std::default_initializable<Reset>);
  static_assert(std::is_empty_v<Encode> && std::default_initializable<Encode>);
  static_assert(std::is_empty_v<Assign> && std::default_initializable<Assign>);
  static_assert(std::is_nothrow_invocable_r_v<expected<void, Error>, Prepare, void*>);
  static_assert(std::is_nothrow_invocable_r_v<void, Reset, void*>);
  static_assert(std::is_nothrow_invocable_r_v<expected<std::size_t, Error>, Encode, const Native&,
                                              DescriptorTensorReferences, std::span<std::byte>>);
  static_assert(std::is_nothrow_invocable_r_v<expected<void, Error>, Assign, void*, const Root&,
                                              std::span<holoscan::Tensor>, SampleMetadata>);

  struct DecodedEnvelope {
    SampleMetadata metadata{};
    std::array<PayloadDescriptor, kTensorCount> descriptors{};
    std::span<const std::byte> metadata_bytes{};
  };

  [[nodiscard]] static bool decode_sample_metadata(const std::span<const std::byte> bytes,
                                                   SampleMetadata& metadata) noexcept {
    constexpr std::size_t kBase = offsetof(Envelope, metadata);
    return read_descriptor_wire_value(
               bytes,
               kBase + offsetof(SampleMetadata, version) + offsetof(SampleMetadataVersion, major),
               metadata.version.major) &&
           read_descriptor_wire_value(
               bytes,
               kBase + offsetof(SampleMetadata, version) + offsetof(SampleMetadataVersion, minor),
               metadata.version.minor) &&
           read_descriptor_wire_value(
               bytes, kBase + offsetof(SampleMetadata, sequence), metadata.sequence) &&
           read_descriptor_wire_value(bytes,
                                      kBase + offsetof(SampleMetadata, capture_timestamp_ns),
                                      metadata.capture_timestamp_ns) &&
           read_descriptor_wire_value(
               bytes,
               kBase + offsetof(SampleMetadata, source_clock_id) + offsetof(ClockId, value),
               metadata.source_clock_id.value) &&
           read_descriptor_wire_value(bytes,
                                      kBase + offsetof(SampleMetadata, publish_timestamp_ns),
                                      metadata.publish_timestamp_ns) &&
           read_descriptor_wire_value(
               bytes,
               kBase + offsetof(SampleMetadata, publish_clock_id) + offsetof(ClockId, value),
               metadata.publish_clock_id.value) &&
           read_descriptor_wire_value(
               bytes, kBase + offsetof(SampleMetadata, trace_id), metadata.trace_id) &&
           read_descriptor_wire_value(
               bytes, kBase + offsetof(SampleMetadata, frame_id), metadata.frame_id) &&
           read_descriptor_wire_value(
               bytes, kBase + offsetof(SampleMetadata, byte_size), metadata.byte_size) &&
           read_descriptor_wire_value(
               bytes, kBase + offsetof(SampleMetadata, payload_kind), metadata.payload_kind) &&
           read_descriptor_wire_value(
               bytes, kBase + offsetof(SampleMetadata, flags), metadata.flags);
  }

  [[nodiscard]] static bool encode_sample_metadata(const SampleMetadata& metadata,
                                                   const std::span<std::byte> bytes) noexcept {
    constexpr std::size_t kBase = offsetof(Envelope, metadata);
    return write_descriptor_wire_value(
               bytes,
               kBase + offsetof(SampleMetadata, version) + offsetof(SampleMetadataVersion, major),
               metadata.version.major) &&
           write_descriptor_wire_value(
               bytes,
               kBase + offsetof(SampleMetadata, version) + offsetof(SampleMetadataVersion, minor),
               metadata.version.minor) &&
           write_descriptor_wire_value(
               bytes, kBase + offsetof(SampleMetadata, sequence), metadata.sequence) &&
           write_descriptor_wire_value(bytes,
                                       kBase + offsetof(SampleMetadata, capture_timestamp_ns),
                                       metadata.capture_timestamp_ns) &&
           write_descriptor_wire_value(
               bytes,
               kBase + offsetof(SampleMetadata, source_clock_id) + offsetof(ClockId, value),
               metadata.source_clock_id.value) &&
           write_descriptor_wire_value(bytes,
                                       kBase + offsetof(SampleMetadata, publish_timestamp_ns),
                                       metadata.publish_timestamp_ns) &&
           write_descriptor_wire_value(
               bytes,
               kBase + offsetof(SampleMetadata, publish_clock_id) + offsetof(ClockId, value),
               metadata.publish_clock_id.value) &&
           write_descriptor_wire_value(
               bytes, kBase + offsetof(SampleMetadata, trace_id), metadata.trace_id) &&
           write_descriptor_wire_value(
               bytes, kBase + offsetof(SampleMetadata, frame_id), metadata.frame_id) &&
           write_descriptor_wire_value(
               bytes, kBase + offsetof(SampleMetadata, byte_size), metadata.byte_size) &&
           write_descriptor_wire_value(
               bytes, kBase + offsetof(SampleMetadata, payload_kind), metadata.payload_kind) &&
           write_descriptor_wire_value(
               bytes, kBase + offsetof(SampleMetadata, flags), metadata.flags);
  }

  [[nodiscard]] static expected<DecodedEnvelope, Error> decode_envelope(
      const std::span<const std::byte> bytes, const std::uint64_t descriptor_carriage_mask,
      const std::uint64_t expected_generation) noexcept {
    constexpr std::size_t kPayloadBase = offsetof(Envelope, payload);
    using Carriage = typename Payload::descriptor_carriage_type;
    constexpr std::size_t kCarriageOffset = kPayloadBase + offsetof(Payload, descriptor);
    constexpr std::size_t kDescriptorsOffset = kCarriageOffset + offsetof(Carriage, descriptors);
    constexpr std::size_t kMetadataOffset = kPayloadBase + offsetof(Payload, metadata);
    if (bytes.size() < kMetadataOffset || bytes.size() > sizeof(Envelope) ||
        expected_generation == 0U) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (descriptor_carriage_mask != 0U && descriptor_carriage_mask != kAllTensorsMask) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const bool descriptors_required = descriptor_carriage_mask != 0U;
    DecodedEnvelope envelope;
    std::uint32_t envelope_abi_version = 0U;
    std::uint32_t envelope_header_size = 0U;
    SchemaId schema_id{};
    std::uint32_t schema_version = 0U;
    std::uint32_t metadata_size = 0U;
    std::uint64_t endpoint_generation = 0U;
    std::uint32_t tensor_count = 0U;
    std::uint32_t reserved = 0U;
    std::uint64_t presence_mask = 0U;
    std::uint64_t descriptor_mask = 0U;
    const bool fixed_fields_valid =
        decode_sample_metadata(bytes, envelope.metadata) &&
        read_descriptor_wire_value(
            bytes, kPayloadBase + offsetof(Payload, envelope_abi_version), envelope_abi_version) &&
        read_descriptor_wire_value(
            bytes, kPayloadBase + offsetof(Payload, envelope_header_size), envelope_header_size) &&
        read_descriptor_wire_value(bytes, kPayloadBase + offsetof(Payload, schema_id), schema_id) &&
        read_descriptor_wire_value(
            bytes, kPayloadBase + offsetof(Payload, schema_version), schema_version) &&
        read_descriptor_wire_value(
            bytes, kPayloadBase + offsetof(Payload, metadata_size), metadata_size) &&
        read_descriptor_wire_value(
            bytes, kPayloadBase + offsetof(Payload, endpoint_generation), endpoint_generation) &&
        read_descriptor_wire_value(
            bytes, kCarriageOffset + offsetof(Carriage, tensor_count), tensor_count) &&
        read_descriptor_wire_value(
            bytes, kCarriageOffset + offsetof(Carriage, reserved), reserved) &&
        read_descriptor_wire_value(
            bytes, kCarriageOffset + offsetof(Carriage, presence_mask), presence_mask) &&
        read_descriptor_wire_value(
            bytes, kCarriageOffset + offsetof(Carriage, descriptor_mask), descriptor_mask);
    const bool header_valid =
        fixed_fields_valid && envelope_abi_version == Payload::expected_envelope_abi_version &&
        envelope_header_size == kSchemaEnvelopeHeaderSize &&
        schema_id == Payload::expected_schema_id &&
        schema_version == Payload::expected_schema_version &&
        endpoint_generation == expected_generation && metadata_size != 0U &&
        metadata_size <= Payload::metadata_capacity && tensor_count == kTensorCount &&
        reserved == 0U && presence_mask == kAllTensorsMask && descriptor_mask == kAllTensorsMask;
    if (!header_valid || kMetadataOffset > bytes.size() ||
        metadata_size > bytes.size() - kMetadataOffset) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    envelope.metadata_bytes = bytes.subspan(kMetadataOffset, metadata_size);
    auto metadata_valid = Payload::verify_metadata(envelope.metadata_bytes);
    if (!metadata_valid) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (descriptors_required) {
      for (std::size_t index = 0U; index < kTensorCount; ++index) {
        if (!read_descriptor_wire_value(bytes,
                                        kDescriptorsOffset + index * sizeof(PayloadDescriptor),
                                        envelope.descriptors[index])) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        auto valid = Payload::validate_descriptor(envelope.descriptors[index]);
        if (!valid || envelope.descriptors[index].schema_id != Payload::expected_schema_id) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
      }
    }
    return envelope;
  }

  [[nodiscard]] static expected<std::size_t, Error> encode_envelope(
      const Native& value, const DescriptorTensorReferences authoritative_tensors,
      const SampleMetadata metadata, const std::uint64_t endpoint_generation,
      const std::span<std::byte> destination) noexcept {
    if (endpoint_generation == 0U || destination.size() != sizeof(Envelope) ||
        authoritative_tensors.size() != kTensorCount ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % alignof(std::max_align_t) != 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    constexpr std::size_t kPayloadBase = offsetof(Envelope, payload);
    using Carriage = typename Payload::descriptor_carriage_type;
    constexpr std::size_t kCarriageOffset = kPayloadBase + offsetof(Payload, descriptor);
    constexpr std::size_t kMetadataOffset = kPayloadBase + offsetof(Payload, metadata);
    std::memset(destination.data(), 0, kMetadataOffset);
    const bool header_encoded =
        encode_sample_metadata(metadata, destination) &&
        write_descriptor_wire_value(destination,
                                    kPayloadBase + offsetof(Payload, envelope_abi_version),
                                    Payload::expected_envelope_abi_version) &&
        write_descriptor_wire_value(destination,
                                    kPayloadBase + offsetof(Payload, envelope_header_size),
                                    kSchemaEnvelopeHeaderSize) &&
        write_descriptor_wire_value(destination,
                                    kPayloadBase + offsetof(Payload, schema_id),
                                    Payload::expected_schema_id) &&
        write_descriptor_wire_value(destination,
                                    kPayloadBase + offsetof(Payload, schema_version),
                                    Payload::expected_schema_version) &&
        write_descriptor_wire_value(destination,
                                    kPayloadBase + offsetof(Payload, endpoint_generation),
                                    endpoint_generation) &&
        write_descriptor_wire_value(destination,
                                    kCarriageOffset + offsetof(Carriage, tensor_count),
                                    static_cast<std::uint32_t>(kTensorCount)) &&
        write_descriptor_wire_value(
            destination, kCarriageOffset + offsetof(Carriage, reserved), std::uint32_t{0U}) &&
        write_descriptor_wire_value(
            destination, kCarriageOffset + offsetof(Carriage, presence_mask), kAllTensorsMask) &&
        write_descriptor_wire_value(
            destination, kCarriageOffset + offsetof(Carriage, descriptor_mask), kAllTensorsMask);
    if (!header_encoded || kMetadataOffset > destination.size() ||
        Payload::metadata_capacity > destination.size() - kMetadataOffset) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto encoded = Encode{}(value,
                            authoritative_tensors,
                            destination.subspan(kMetadataOffset, Payload::metadata_capacity));
    if (!encoded) {
      return make_unexpected(std::move(encoded).error());
    }
    if (*encoded == 0U || *encoded > Payload::metadata_capacity ||
        *encoded > std::numeric_limits<std::uint32_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    const auto metadata_size = static_cast<std::uint32_t>(*encoded);
    if (!write_descriptor_wire_value(
            destination, kPayloadBase + offsetof(Payload, metadata_size), metadata_size)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return kMetadataOffset + *encoded;
  }

  [[nodiscard]] static expected<std::size_t, Error> encode_sample(
      const void* payload, const SampleMetadata metadata, const std::uint64_t endpoint_generation,
      const std::span<std::byte> destination) noexcept {
    if (payload == nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const auto& value = *static_cast<const Native*>(payload);
    if (descriptor_tensor_count(value) != kTensorCount) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    std::array<const holoscan::Tensor*, kTensorCount> tensors{};
    for (std::size_t index = 0U; index < kTensorCount; ++index) {
      tensors[index] = descriptor_tensor_at(value, index);
      if (tensors[index] == nullptr) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
    }
    return encode_envelope(
        value,
        DescriptorTensorReferences{std::span<const holoscan::Tensor* const>{tensors}},
        metadata,
        endpoint_generation,
        destination);
  }

  [[nodiscard]] static expected<std::size_t, Error> encode_loan(
      const void* payload, const holoscan::Tensor& tensor, const SampleMetadata metadata,
      const std::uint64_t endpoint_generation, const std::span<std::byte> destination) noexcept {
    if (payload == nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if constexpr (kTensorCount == 1U) {
      const std::array<const holoscan::Tensor*, 1U> tensors{&tensor};
      return encode_envelope(
          *static_cast<const Native*>(payload),
          DescriptorTensorReferences{std::span<const holoscan::Tensor* const>{tensors}},
          metadata,
          endpoint_generation,
          destination);
    } else {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
  }

  [[nodiscard]] static expected<std::size_t, Error> encode_loans(
      const void* payload, const std::span<const holoscan::Tensor> authoritative_tensors,
      const SampleMetadata metadata, const std::uint64_t endpoint_generation,
      const std::span<std::byte> destination) noexcept {
    if (payload == nullptr || authoritative_tensors.size() != kTensorCount) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    std::array<const holoscan::Tensor*, kTensorCount> tensors{};
    for (std::size_t index = 0U; index < kTensorCount; ++index) {
      tensors[index] = &authoritative_tensors[index];
    }
    return encode_envelope(
        *static_cast<const Native*>(payload),
        DescriptorTensorReferences{std::span<const holoscan::Tensor* const>{tensors}},
        metadata,
        endpoint_generation,
        destination);
  }

  [[nodiscard]] static expected<SampleMetadata, Error> sample_metadata(
      const std::span<const std::byte> bytes, const std::uint64_t descriptor_carriage_mask,
      const std::uint64_t expected_generation) noexcept {
    auto envelope = decode_envelope(bytes, descriptor_carriage_mask, expected_generation);
    if (!envelope) {
      return make_unexpected(std::move(envelope).error());
    }
    return envelope->metadata;
  }

  [[nodiscard]] static expected<TensorViewMetadata, Error> tensor_view_metadata_at(
      const std::span<const std::byte> bytes, const std::uint32_t index,
      const std::uint64_t expected_generation) noexcept {
    auto envelope = decode_envelope(bytes, kAllTensorsMask, expected_generation);
    if (!envelope) {
      return make_unexpected(std::move(envelope).error());
    }
    const auto metadata_bytes = envelope->metadata_bytes;
    const Root* root = Payload::root_from_metadata(metadata_bytes);
    if (root == nullptr || index >= kTensorCount) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    try {
      return descriptor_tensor_view_metadata_at_from_root(*root, metadata_bytes, index);
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
  }

  [[nodiscard]] static expected<TensorViewMetadata, Error> tensor_view_metadata(
      const std::span<const std::byte> bytes, const std::uint64_t expected_generation) noexcept {
    if constexpr (kTensorCount == 1U) {
      return tensor_view_metadata_at(bytes, 0U, expected_generation);
    } else {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
  }

  [[nodiscard]] static expected<std::span<const std::byte>, Error> validated_metadata_bytes(
      const std::span<const std::byte> bytes, const SampleMetadata metadata) noexcept {
    constexpr std::size_t kMetadataSizeOffset =
        offsetof(Envelope, payload) + offsetof(Payload, metadata_size);
    constexpr std::size_t kMetadataOffset =
        offsetof(Envelope, payload) + offsetof(Payload, metadata);
    if (bytes.size() < kMetadataOffset || bytes.size() > sizeof(Envelope)) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    SampleMetadata encoded_metadata{};
    std::uint32_t metadata_size = 0U;
    if (!decode_sample_metadata(bytes, encoded_metadata)) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    std::memcpy(&metadata_size, bytes.data() + kMetadataSizeOffset, sizeof(metadata_size));
    if (!sample_metadata_equal(encoded_metadata, metadata) || metadata_size == 0U ||
        metadata_size > Payload::metadata_capacity ||
        kMetadataOffset + metadata_size > bytes.size()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    return bytes.subspan(kMetadataOffset, metadata_size);
  }

  [[nodiscard]] static expected<void, Error> assign_tensors_sample(
      void* storage, const std::span<const std::byte> bytes,
      const std::span<holoscan::Tensor> tensors, const SampleMetadata metadata) noexcept {
    if (storage == nullptr || tensors.size() != kTensorCount) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto metadata_bytes = validated_metadata_bytes(bytes, metadata);
    if (!metadata_bytes) {
      return make_unexpected(std::move(metadata_bytes).error());
    }
    const Root* root = Payload::root_from_metadata(*metadata_bytes);
    if (root == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    for (std::size_t index = 0U; index < kTensorCount; ++index) {
      expected<TensorViewMetadata, Error> tensor_metadata =
          make_unexpected(Error{ErrorCode::kProtocolError});
      try {
        tensor_metadata =
            descriptor_tensor_view_metadata_at_from_root(*root, *metadata_bytes, index);
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      if (!tensor_metadata || !holoscan::tensor_matches(tensors[index], *tensor_metadata)) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
    }
    return Assign{}(storage, *root, tensors, metadata);
  }

  [[nodiscard]] static expected<void, Error> assign_sample(void* storage,
                                                           const std::span<const std::byte> bytes,
                                                           holoscan::Tensor tensor,
                                                           const SampleMetadata metadata) noexcept {
    if constexpr (kTensorCount == 1U) {
      std::array<holoscan::Tensor, 1U> tensors{std::move(tensor)};
      return assign_tensors_sample(storage, bytes, tensors, metadata);
    } else {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
  }

  [[nodiscard]] static expected<void, Error> construct_predicate_view(
      void* storage, const void* sample, const InputSelection selection,
      const std::uint64_t* active_generation, const std::uint64_t expected_generation) noexcept {
    if (storage == nullptr || sample == nullptr || active_generation == nullptr ||
        reinterpret_cast<std::uintptr_t>(storage) % alignof(PredicateInputView<Native>) != 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto view = holoscan::detail::make_runtime_predicate_input_view<Native>(
        holoscan::detail::PredicateInputBinding<Native>{
            .sample = static_cast<const Sample<Native>*>(sample),
            .selection = selection,
            .active_generation = active_generation,
            .expected_generation = expected_generation,
        });
    if (!view) {
      return make_unexpected(std::move(view).error());
    }
    std::construct_at(static_cast<PredicateInputView<Native>*>(storage), std::move(*view));
    return {};
  }

  [[nodiscard]] static holoscan::detail::RuntimePortTypeOps make() noexcept {
    return holoscan::detail::RuntimePortTypeOps{
        .payload_size = sizeof(Native),
        .payload_alignment = alignof(Native),
        .wire_capacity = sizeof(Envelope),
        .sample_size = sizeof(Sample<Native>),
        .sample_alignment = alignof(Sample<Native>),
        .predicate_view_size = sizeof(PredicateInputView<Native>),
        .predicate_view_alignment = alignof(PredicateInputView<Native>),
        .codec_family = holoscan::detail::RuntimePayloadCodecFamily::kDescriptorTensorSchema,
        .payload_kind = PayloadKind::kDescriptorBacked,
        .inline_pod = false,
        .generic_loan_supported = false,
        .persistent_sample_storage = true,
        .generic_loan_slot_size_bytes = 0U,
        .descriptor_offset = offsetof(Envelope, payload) + offsetof(Payload, descriptor) +
                             offsetof(typename Payload::descriptor_carriage_type, descriptors),
        .descriptor_tensor_static_count = static_cast<std::uint32_t>(kTensorCount),
        .encode_payload = nullptr,
        .construct_sample = nullptr,
        .destroy_sample =
            [](void* storage) noexcept { std::destroy_at(static_cast<Sample<Native>*>(storage)); },
        .prepare_sample_storage = [](void* storage) noexcept { return Prepare{}(storage); },
        .reset_sample_storage = [](void* storage) noexcept { Reset{}(storage); },
        .sample_payload = [](const void* sample) noexcept -> const void* {
          return &static_cast<const Sample<Native>*>(sample)->data;
        },
        .sample_metadata = [](const void* sample) noexcept -> const SampleMetadata* {
          return &static_cast<const Sample<Native>*>(sample)->metadata;
        },
        .descriptor_tensor = [](const void* payload) noexcept -> const holoscan::Tensor* {
          if (payload == nullptr) {
            return nullptr;
          }
          const auto& value = *static_cast<const Native*>(payload);
          return descriptor_tensor_count(value) == 1U ? descriptor_tensor_at(value, 0U) : nullptr;
        },
        .descriptor_tensor_count = [](const void* payload) noexcept -> std::uint32_t {
          return payload == nullptr ? 0U
                                    : static_cast<std::uint32_t>(descriptor_tensor_count(
                                          *static_cast<const Native*>(payload)));
        },
        .descriptor_tensor_at = [](const void* payload,
                                   const std::uint32_t index) noexcept -> const holoscan::Tensor* {
          return payload == nullptr
                     ? nullptr
                     : descriptor_tensor_at(*static_cast<const Native*>(payload), index);
        },
        .encode_descriptor_tensor_sample = &encode_sample,
        .encode_descriptor_tensor_loan = &encode_loan,
        .encode_descriptor_tensor_loans = &encode_loans,
        .descriptor_tensor_sample_metadata = &sample_metadata,
        .descriptor_tensor_view_metadata = &tensor_view_metadata,
        .descriptor_tensor_view_metadata_at = &tensor_view_metadata_at,
        .assign_descriptor_tensor_sample = &assign_sample,
        .assign_descriptor_tensors_sample = &assign_tensors_sample,
        .construct_predicate_view = &construct_predicate_view,
        .destroy_predicate_view =
            [](void* storage) noexcept {
              std::destroy_at(static_cast<PredicateInputView<Native>*>(storage));
            },
        .construct_value = nullptr,
        .destroy_value = nullptr,
        .bind_input =
            [](void* declaration, holoscan::detail::RuntimeInputBinding* binding) noexcept {
              holoscan::detail::RuntimePortBindingAccess::bind_input(
                  *static_cast<Input<Native>*>(declaration), binding);
            },
        .unbind_input =
            [](void* declaration, holoscan::detail::RuntimeInputBinding* binding) noexcept {
              holoscan::detail::RuntimePortBindingAccess::unbind_input(
                  *static_cast<Input<Native>*>(declaration), binding);
            },
        .bind_output =
            [](void* declaration, holoscan::detail::RuntimeOutputBinding* binding) noexcept {
              holoscan::detail::RuntimePortBindingAccess::bind_output(
                  *static_cast<Output<Native>*>(declaration), binding);
            },
        .unbind_output =
            [](void* declaration, holoscan::detail::RuntimeOutputBinding* binding) noexcept {
              holoscan::detail::RuntimePortBindingAccess::unbind_output(
                  *static_cast<Output<Native>*>(declaration), binding);
            },
        .publish_tensor_loan = [](void* declaration,
                                  const void* descriptor,
                                  holoscan::Tensor tensor,
                                  EmitOptions options) noexcept -> expected<void, Error> {
          return holoscan::detail::RuntimePortBindingAccess::emit_descriptor_tensor(
              *static_cast<Output<Native>*>(declaration), descriptor, tensor, options);
        },
        .publish_tensor_loans = [](void* declaration,
                                   const void* descriptor,
                                   const std::span<const holoscan::Tensor>
                                       tensors,
                                   EmitOptions options) noexcept -> expected<void, Error> {
          return holoscan::detail::RuntimePortBindingAccess::emit_descriptor_tensors(
              *static_cast<Output<Native>*>(declaration), descriptor, tensors, options);
        },
        .bind_tensor_output =
            [](void* declaration, holoscan::detail::TensorOutputBinding binding) noexcept {
              holoscan::detail::TensorOutputPortBindingAccess::bind(
                  *static_cast<Output<Native>*>(declaration), binding);
            },
        .unbind_tensor_output =
            [](void* declaration) noexcept {
              holoscan::detail::TensorOutputPortBindingAccess::unbind(
                  *static_cast<Output<Native>*>(declaration));
            },
    };
  }
};

template <typename Codec, typename Native, typename Root>
concept StandardDescriptorTensorCodec =
    std::is_empty_v<Codec> && std::default_initializable<Codec> &&
    requires(Codec codec, flatbuffers::FlatBufferBuilder& builder, const Native& source,
             Native& destination,
             std::span<const flatbuffers::Offset<schema::Tensor>> tensor_offsets,
             const Root& root) {
      {
        codec.encode_root(builder, source, tensor_offsets)
      } -> std::same_as<expected<flatbuffers::Offset<Root>, Error>>;
      { codec.assign_root(destination, root) } -> std::same_as<expected<void, Error>>;
    };

template <typename Native, auto TensorMember>
concept StandardDescriptorTensorMember =
    std::same_as<decltype(TensorMember), std::shared_ptr<holoscan::Tensor> Native::*>;

template <typename Codec, typename Native>
concept HasStandardDescriptorTensorPrepare =
    requires(Codec codec, Native& value) { codec.prepare(value); };

template <typename Codec, typename Native>
concept ValidStandardDescriptorTensorPrepare =
    !HasStandardDescriptorTensorPrepare<Codec, Native> || requires(Codec codec, Native& value) {
      { codec.prepare(value) } -> std::same_as<expected<void, Error>>;
    };

template <typename Codec, typename Native>
concept HasStandardDescriptorTensorReset =
    requires(Codec codec, Native& value) { codec.reset(value); };

template <typename Codec, typename Native>
concept ValidStandardDescriptorTensorReset =
    !HasStandardDescriptorTensorReset<Codec, Native> || requires(Codec codec, Native& value) {
      { codec.reset(value) } noexcept -> std::same_as<void>;
    };

template <typename Native, typename Root, typename Codec, auto... TensorMembers>
struct StandardDescriptorTensorCallbacks {
  using TensorOffset = flatbuffers::Offset<schema::Tensor>;

  static constexpr std::size_t kTensorCount = sizeof...(TensorMembers);
  static_assert(kTensorCount > 0U,
                "standard descriptor Tensor adapter requires at least one Tensor member");
  static_assert((StandardDescriptorTensorMember<Native, TensorMembers> && ...),
                "standard descriptor Tensor members must be std::shared_ptr<Tensor> fields");
  static_assert(StandardDescriptorTensorCodec<Codec, Native, Root>,
                "standard descriptor Tensor codec must be empty and provide encode_root and "
                "assign_root with the documented signatures");
  static_assert(ValidStandardDescriptorTensorPrepare<Codec, Native>,
                "standard descriptor Tensor prepare hook must return expected<void, Error>");
  static_assert(ValidStandardDescriptorTensorReset<Codec, Native>,
                "standard descriptor Tensor reset hook must return void and be noexcept");

  struct Prepare {
    [[nodiscard]] expected<void, Error> operator()(void* storage) const noexcept {
      if (storage == nullptr ||
          reinterpret_cast<std::uintptr_t>(storage) % alignof(Sample<Native>) != 0U) {
        return make_unexpected(
            Error{ErrorCode::kInvalidArgument, "descriptor sample storage is null or unaligned"});
      }
      try {
        Native value{};
        ((value.*TensorMembers = std::make_shared<holoscan::Tensor>()), ...);
        if constexpr (HasStandardDescriptorTensorPrepare<Codec, Native>) {
          auto prepared = Codec{}.prepare(value);
          if (!prepared) {
            return make_unexpected(std::move(prepared).error());
          }
        }
        std::construct_at(
            static_cast<Sample<Native>*>(storage), std::move(value), SampleMetadata{});
        return {};
      } catch (const std::bad_alloc&) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted,
                                     "descriptor sample preparation allocation failed"});
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure, "descriptor sample preparation failed"});
      }
    }
  };

  struct Reset {
    void operator()(void* storage) const noexcept {
      auto& sample = *static_cast<Sample<Native>*>(storage);
      sample.metadata = {};
      if constexpr (HasStandardDescriptorTensorReset<Codec, Native>) {
        Codec{}.reset(sample.data);
      }
      assert(((sample.data.*TensorMembers != nullptr) && ...));
      if (((sample.data.*TensorMembers != nullptr) && ...)) {
        ((*(sample.data.*TensorMembers) = holoscan::Tensor{}), ...);
      }
    }
  };

  struct Encode {
    [[nodiscard]] expected<std::size_t, Error> operator()(
        const Native& value, const DescriptorTensorReferences tensors,
        const std::span<std::byte> destination) const noexcept {
      if (tensors.size() != kTensorCount || destination.empty() ||
          reinterpret_cast<std::uintptr_t>(destination.data()) %
                  alignof(flatbuffers::largest_scalar_t) !=
              0U) {
        return make_unexpected(
            Error{ErrorCode::kInvalidArgument, "descriptor metadata inputs are invalid"});
      }
      try {
        BoundedSpanAllocator allocator{destination};
        flatbuffers::FlatBufferBuilder builder{destination.size(), &allocator, false};
        std::array<TensorOffset, kTensorCount> tensor_offsets{};
        for (std::size_t index = 0U; index < kTensorCount; ++index) {
          const holoscan::Tensor* tensor = tensors.at(index);
          if (tensor == nullptr) {
            return make_unexpected(
                Error{ErrorCode::kInvalidArgument, "descriptor Tensor reference is null"});
          }
          auto encoded = create_tensor_metadata(builder, *tensor);
          if (!encoded) {
            return make_unexpected(std::move(encoded).error());
          }
          tensor_offsets[index] = *encoded;
        }
        auto root =
            Codec{}.encode_root(builder, value, std::span<const TensorOffset>{tensor_offsets});
        if (!root) {
          return make_unexpected(std::move(root).error());
        }
        if (root->IsNull()) {
          return make_unexpected(
              Error{ErrorCode::kCodecError, "descriptor metadata root creation failed"});
        }
        builder.Finish(*root);
        const std::size_t size = builder.GetSize();
        if (size == 0U || size > destination.size()) {
          return make_unexpected(
              Error{ErrorCode::kResourceExhausted, "descriptor metadata exceeds bounded storage"});
        }
        std::memmove(destination.data(), builder.GetBufferPointer(), size);
        return size;
      } catch (const std::bad_alloc&) {
        return make_unexpected(
            Error{ErrorCode::kResourceExhausted, "descriptor metadata exceeds bounded storage"});
      } catch (...) {
        return make_unexpected(
            Error{ErrorCode::kCodecError, "descriptor metadata encoding failed"});
      }
    }
  };

  struct Assign {
    [[nodiscard]] expected<void, Error> operator()(void* storage, const Root& root,
                                                   const std::span<holoscan::Tensor> tensors,
                                                   const SampleMetadata metadata) const noexcept {
      if (storage == nullptr || tensors.size() != kTensorCount) {
        return make_unexpected(
            Error{ErrorCode::kInvalidArgument, "descriptor assignment inputs are invalid"});
      }
      auto& sample = *static_cast<Sample<Native>*>(storage);
      if (!((sample.data.*TensorMembers != nullptr) && ...)) {
        return make_unexpected(
            Error{ErrorCode::kProtocolError, "descriptor Tensor sample storage is missing"});
      }
      try {
        auto assigned = Codec{}.assign_root(sample.data, root);
        if (!assigned) {
          return make_unexpected(std::move(assigned).error());
        }
        std::size_t index = 0U;
        ((*(sample.data.*TensorMembers) = std::move(tensors[index++])), ...);
        sample.metadata = metadata;
        return {};
      } catch (const std::bad_alloc&) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted,
                                     "descriptor metadata assignment allocation failed"});
      } catch (...) {
        return make_unexpected(
            Error{ErrorCode::kCodecError, "descriptor metadata assignment failed"});
      }
    }
  };
};

}  // namespace detail

/// @brief Build Runtime operations for one descriptor-backed FlatBuffers Tensor schema.
///
/// The four callbacks must be empty, default-constructible function objects so the returned raw
/// operation table needs no mutable registry or captured state. Non-capturing lambdas are the
/// intended spelling. Their fixed call signatures are:
///
/// - `prepare(void*) -> expected<void, Error>`;
/// - `reset(void*) -> void`;
/// - `encode(const Native&, DescriptorTensorReferences, span<byte>) -> expected<size_t, Error>`;
/// - `assign(void*, const Root&, span<Tensor>, SampleMetadata) -> expected<void, Error>`.
///
/// The encoder's indexed view and the assigner's Tensor span contain the trait-declared static
/// Tensor count in canonical field order. A singular schema receives a one-element collection and
/// retains its dedicated field-zero hooks. Descriptor carriage requirements remain private to the
/// generic envelope validator.
///
/// @tparam Native Owning generated object-API type exposed on ports.
/// @tparam Root Generated FlatBuffers metadata root.
/// @tparam MetadataCapacity Exact positive bounded metadata capacity.
/// @param prepare Stateless callback that cold-constructs persistent `Sample<Native>` storage.
/// @param reset Stateless callback that clears one persistent sample for reuse.
/// @param encode Stateless callback that encodes metadata from authoritative indexed Tensors.
/// @param assign Stateless callback that transactionally assigns a validated root and Tensors.
/// @return Complete process-local descriptor-backed Runtime operations table.
template <typename Native, typename Root, std::size_t MetadataCapacity>
  requires PublishedPayload<Native> &&
           holoscan::detail::HasDescriptorTensorSchemaNativeTraits<Native> &&
           std::same_as<typename DescriptorTensorSchemaNativeTraits<Native>::root_type, Root>
[[nodiscard]] holoscan::detail::RuntimePortTypeOps make_descriptor_tensor_port_type_ops(
    auto prepare, auto reset, auto encode, auto assign) noexcept {
  static_assert(MetadataCapacity == DescriptorTensorSchemaNativeTraits<Native>::metadata_capacity);
  static_assert(holoscan::detail::schema_identity_equal(schema_identity<Native>(),
                                                        flatbuffer_schema_identity_v<Root>),
                "descriptor native and FlatBuffers root identities must match");
  using Factory = detail::DescriptorTensorPortTypeOpsFactory<Native,
                                                             Root,
                                                             MetadataCapacity,
                                                             std::remove_cvref_t<decltype(prepare)>,
                                                             std::remove_cvref_t<decltype(reset)>,
                                                             std::remove_cvref_t<decltype(encode)>,
                                                             std::remove_cvref_t<decltype(assign)>>;
  return Factory::make();
}

/// @brief Build descriptor-backed Runtime operations with SDK-owned standard defaults.
///
/// The adapter cold-constructs persistent sample storage, retains one Tensor holder per named
/// native member, builds metadata-only Tensor tables in bounded storage, validates received Tensor
/// metadata against authoritative handles, and commits Tensor handles plus \ref SampleMetadata.
/// The stateless codec supplies only schema-specific ordinary-field work:
///
/// - optional `prepare(Native&) -> expected<void, Error>`;
/// - optional `reset(Native&) -> void noexcept`;
/// - `encode_root(FlatBufferBuilder&, const Native&, span<const Offset<schema::Tensor>>)
///   -> expected<Offset<Root>, Error>`; and
/// - `assign_root(Native&, const Root&) -> expected<void, Error>`.
///
/// `prepare` and `reset` may initialize or clear ordinary fields but must not replace the named
/// Tensor holders. `assign_root` must validate before mutation, assign every ordinary field
/// including absent optionals, and leave the Tensor members untouched; after it succeeds, the
/// adapter commits all authoritative Tensor handles and sample metadata.
///
/// `TensorMembers` are ordered in canonical FlatBuffers field-id order and must exactly match the
/// trait-declared static Tensor count.
///
/// @tparam Native Owning generated object-API type exposed on ports.
/// @tparam Root Generated FlatBuffers metadata root.
/// @tparam MetadataCapacity Exact positive bounded metadata capacity.
/// @tparam TensorMembers Ordered `std::shared_ptr<Tensor>` members of `Native`.
/// @tparam Codec Deduced empty/default-constructible schema-specific codec.
/// @param codec Stateless codec value used only to deduce \p Codec.
/// @return Complete process-local descriptor-backed Runtime operations table.
template <typename Native, typename Root, std::size_t MetadataCapacity, auto... TensorMembers,
          typename Codec>
  requires PublishedPayload<Native> &&
           holoscan::detail::HasDescriptorTensorSchemaNativeTraits<Native> &&
           std::same_as<typename DescriptorTensorSchemaNativeTraits<Native>::root_type, Root>
[[nodiscard]] holoscan::detail::RuntimePortTypeOps make_standard_descriptor_tensor_port_type_ops(
    [[maybe_unused]] Codec codec) noexcept {
  static_assert(
      sizeof...(TensorMembers) == holoscan::detail::descriptor_tensor_schema_tensor_count_v<Native>,
      "standard descriptor Tensor member count must match the native trait");
  using Callbacks =
      detail::StandardDescriptorTensorCallbacks<Native, Root, Codec, TensorMembers...>;
  return make_descriptor_tensor_port_type_ops<Native, Root, MetadataCapacity>(
      typename Callbacks::Prepare{},
      typename Callbacks::Reset{},
      typename Callbacks::Encode{},
      typename Callbacks::Assign{});
}

}  // namespace holoscan::schema

namespace holoscan {

/// @brief Describe the SDK Image owning type as descriptor metadata plus one Core Tensor.
///
/// Image metadata is bounded independently from Tensor storage. This specialization deliberately
/// does not opt `ImageT` into `GeneratedSchemaNativeTraits`: ordinary generated-schema encoding
/// would serialize or materialize Tensor bytes and bypass the plan-owned Tensor lifetime.
template <>
struct DescriptorTensorSchemaNativeTraits<schema::ImageT> {
  using root_type = schema::Image;  ///< Generated metadata table root.
  static constexpr std::size_t metadata_capacity =
      schema::flatbuffer_default_descriptor_metadata_capacity_v<root_type>;
  static constexpr std::size_t tensor_count = 1U;  ///< Number of direct Tensor fields.
  static constexpr std::array<std::uint16_t, 1U> tensor_field_ids{4U};

  /// @brief Return the embedded Tensor without extending its lifetime.
  /// @param value Owning generated Image value.
  /// @return Embedded Tensor, or `nullptr` when the Image has no Tensor member.
  [[nodiscard]] static const Tensor* tensor(const schema::ImageT& value) noexcept {
    return value.data.get();
  }

  /// @brief Return the indexed embedded Tensor without extending its lifetime.
  [[nodiscard]] static const Tensor* tensor_at(const schema::ImageT& value,
                                               const std::size_t index) noexcept {
    return index == 0U ? tensor(value) : nullptr;
  }
};

/// @brief Describe TensorPacket as bounded descriptor metadata plus one Core Tensor.
template <>
struct DescriptorTensorSchemaNativeTraits<schema::TensorPacketT> {
  using root_type = schema::TensorPacket;  ///< Generated metadata table root.
  static constexpr std::size_t metadata_capacity =
      schema::flatbuffer_default_descriptor_metadata_capacity_v<root_type>;
  static constexpr std::size_t tensor_count = 1U;  ///< Number of direct Tensor fields.
  static constexpr std::array<std::uint16_t, 1U> tensor_field_ids{3U};

  /// @brief Return the embedded Tensor without extending its lifetime.
  /// @param value Owning generated TensorPacket value.
  /// @return Embedded Tensor, or `nullptr` when the packet has no Tensor member.
  [[nodiscard]] static const Tensor* tensor(const schema::TensorPacketT& value) noexcept {
    return value.data.get();
  }

  /// @brief Return the indexed embedded Tensor without extending its lifetime.
  [[nodiscard]] static const Tensor* tensor_at(const schema::TensorPacketT& value,
                                               const std::size_t index) noexcept {
    return index == 0U ? tensor(value) : nullptr;
  }
};

/// @brief Describe the SDK Imu owning generated type and finite serialized capacity.
template <>
struct GeneratedSchemaNativeTraits<schema::ImuT> {
  using root_type = schema::Imu;  ///< Generated table root used only for package codec evidence.
  static constexpr std::size_t max_serialized_size =
      schema::flatbuffer_default_inline_capacity_v<root_type>;  ///< Maximum serialized bytes.
};

/// @brief Return the SDK Image semantic identity for its descriptor-backed owning type.
template <>
consteval SchemaIdentity schema_identity<schema::ImageT>() {
  return schema::FlatBufferSchemaTraits<schema::Image>::identity;
}

/// @brief Return the SDK TensorPacket identity for its descriptor-backed owning type.
template <>
consteval SchemaIdentity schema_identity<schema::TensorPacketT>() {
  return schema::FlatBufferSchemaTraits<schema::TensorPacket>::identity;
}

namespace detail {

/// @brief Return the descriptor-backed Runtime operations for generated Image values.
/// @return Process-local operations whose address never enters a plan or transport record.
template <>
[[nodiscard]] HOLOSCAN_SCHEMA_EXPORT const RuntimePortTypeOps*
runtime_port_type_ops<schema::ImageT>() noexcept;

/// @brief Return descriptor-backed Runtime operations for generated TensorPacket values.
/// @return Process-local operations whose address never enters a plan or transport record.
template <>
[[nodiscard]] HOLOSCAN_SCHEMA_EXPORT const RuntimePortTypeOps*
runtime_port_type_ops<schema::TensorPacketT>() noexcept;

}  // namespace detail

/// @brief Describe the SDK String owning generated type and finite serialized capacity.
template <>
struct GeneratedSchemaNativeTraits<schema::StringT> {
  using root_type = schema::String;  ///< Generated table root used only for package codec evidence.
  static constexpr std::size_t max_serialized_size =
      schema::flatbuffer_default_inline_capacity_v<root_type>;  ///< Maximum serialized bytes.
};

/// @brief Describe the SDK Time owning generated type and finite serialized capacity.
template <>
struct GeneratedSchemaNativeTraits<schema::TimeT> {
  using root_type = schema::Time;  ///< Generated table root used only for package codec evidence.
  static constexpr std::size_t max_serialized_size =
      schema::flatbuffer_default_inline_capacity_v<root_type>;  ///< Maximum serialized bytes.
};

/// @brief Return the SDK Imu semantic identity for its owning generated type.
/// @return Complete stable identity shared with the generated Imu root package.
template <>
consteval SchemaIdentity schema_identity<schema::ImuT>() {
  return schema::FlatBufferSchemaTraits<schema::Imu>::identity;
}

/// @brief Return the SDK String semantic identity for its owning generated type.
/// @return Complete stable identity shared with the generated String root package.
template <>
consteval SchemaIdentity schema_identity<schema::StringT>() {
  return schema::FlatBufferSchemaTraits<schema::String>::identity;
}

/// @brief Return the SDK Time semantic identity for its owning generated type.
/// @return Complete stable identity shared with the generated Time root package.
template <>
consteval SchemaIdentity schema_identity<schema::TimeT>() {
  return schema::FlatBufferSchemaTraits<schema::Time>::identity;
}

/// @brief Return the generated SDK Image package and verifier evidence.
/// @return Process-lifetime immutable package; concurrent callers may borrow it safely.
template <>
inline const SchemaPackageDescriptor<schema::Image>& schema_package<schema::Image>() noexcept {
  static const SchemaPackageDescriptor<schema::Image> package =
      schema::detail::make_sdk_flatbuffer_package<schema::Image, schema::ImageBinarySchema>(
          "image.fbs", std::span{schema::detail::kImageIncludes});
  return package;
}

/// @brief Return the generated SDK Imu package and verifier evidence.
/// @return Process-lifetime immutable package; concurrent callers may borrow it safely.
template <>
inline const SchemaPackageDescriptor<schema::Imu>& schema_package<schema::Imu>() noexcept {
  static const SchemaPackageDescriptor<schema::Imu> package =
      schema::detail::make_sdk_flatbuffer_package<schema::Imu, schema::ImuBinarySchema>(
          "imu.fbs", std::span{schema::detail::kImuIncludes});
  return package;
}

/// @brief Return the generated SDK String package and verifier evidence.
/// @return Process-lifetime immutable package; concurrent callers may borrow it safely.
template <>
inline const SchemaPackageDescriptor<schema::String>& schema_package<schema::String>() noexcept {
  static const SchemaPackageDescriptor<schema::String> package =
      schema::detail::make_sdk_flatbuffer_package<schema::String, schema::StringBinarySchema>(
          "string.fbs", std::span{schema::detail::kNoIncludes});
  return package;
}

/// @brief Return the generated SDK Tensor package and verifier evidence.
/// @return Process-lifetime immutable package; concurrent callers may borrow it safely.
template <>
inline const SchemaPackageDescriptor<schema::Tensor>& schema_package<schema::Tensor>() noexcept {
  static const SchemaPackageDescriptor<schema::Tensor> package =
      schema::detail::make_sdk_flatbuffer_package<schema::Tensor, schema::TensorBinarySchema>(
          "tensor.fbs", std::span{schema::detail::kNoIncludes});
  return package;
}

/// @brief Return the generated SDK TensorPacket package and verifier evidence.
/// @return Process-lifetime immutable package; concurrent callers may borrow it safely.
template <>
inline const SchemaPackageDescriptor<schema::TensorPacket>&
schema_package<schema::TensorPacket>() noexcept {
  static const SchemaPackageDescriptor<schema::TensorPacket> package =
      schema::detail::make_sdk_flatbuffer_package<schema::TensorPacket,
                                                  schema::TensorPacketBinarySchema>(
          "tensor_packet.fbs", std::span{schema::detail::kTensorPacketIncludes});
  return package;
}

/// @brief Return the generated SDK Time package and verifier evidence.
/// @return Process-lifetime immutable package; concurrent callers may borrow it safely.
template <>
inline const SchemaPackageDescriptor<schema::Time>& schema_package<schema::Time>() noexcept {
  static const SchemaPackageDescriptor<schema::Time> package =
      schema::detail::make_sdk_flatbuffer_package<schema::Time, schema::TimeBinarySchema>(
          "time.fbs", std::span{schema::detail::kNoIncludes});
  return package;
}

/// @brief Return the generated SDK Imu package for its owning native type.
/// @return Process-lifetime immutable package; concurrent callers may borrow it safely.
template <>
inline const SchemaPackageDescriptor<schema::ImuT>& schema_package<schema::ImuT>() noexcept {
  static const SchemaPackageDescriptor<schema::ImuT> package = schema::detail::
      make_sdk_flatbuffer_native_package<schema::ImuT, schema::Imu, schema::ImuBinarySchema>(
          "imu.fbs", std::span{schema::detail::kImuIncludes});
  return package;
}

/// @brief Return the generated SDK String package for its owning native type.
/// @return Process-lifetime immutable package; concurrent callers may borrow it safely.
template <>
inline const SchemaPackageDescriptor<schema::StringT>& schema_package<schema::StringT>() noexcept {
  static const SchemaPackageDescriptor<schema::StringT> package =
      schema::detail::make_sdk_flatbuffer_native_package<schema::StringT,
                                                         schema::String,
                                                         schema::StringBinarySchema>(
          "string.fbs", std::span{schema::detail::kNoIncludes});
  return package;
}

/// @brief Return the generated SDK Time package for its owning native type.
/// @return Process-lifetime immutable package; concurrent callers may borrow it safely.
template <>
inline const SchemaPackageDescriptor<schema::TimeT>& schema_package<schema::TimeT>() noexcept {
  static const SchemaPackageDescriptor<schema::TimeT> package = schema::detail::
      make_sdk_flatbuffer_native_package<schema::TimeT, schema::Time, schema::TimeBinarySchema>(
          "time.fbs", std::span{schema::detail::kNoIncludes});
  return package;
}

}  // namespace holoscan
