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
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

#include <flatbuffers/flatbuffers.h>

#include <holoscan/holoscan_schema_export.h>
#include <holoscan/core/port.hpp>
#include <holoscan/core/predicate_input_view.hpp>
#include <holoscan/schema/bounded_span_allocator.hpp>
#include <holoscan/schema/descriptor_flatbuffer_payload.hpp>
#include <holoscan/schema/flatbuffer_root_access.hpp>
#include <holoscan/schema/flatbuffer_schema_traits.hpp>
#include <holoscan/schema/schema_package.hpp>
#include <holoscan/schema/tensor_materialization.hpp>

namespace holoscan::schema {

/// @brief Bind the SDK Tensor custom native type to its generated FlatBuffers root.
template <>
struct FlatBufferSchemaBinding<Tensor> {
  using root_type = Tensor;
  using native_type = Tensor;
  static constexpr std::uint32_t compatibility_epoch = 1U;
  static constexpr FlatBufferFileIdentifier file_identifier =
      make_flatbuffer_file_identifier("HSTN");
  static constexpr SchemaFamilyIdentity family_identity =
      make_schema_family_identity<Tensor>(compatibility_epoch);
  static constexpr SchemaIdentity identity = make_schema_identity(family_identity, file_identifier);
  static constexpr bool descriptor_tensor = false;
};

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

template <typename Root>
[[nodiscard]] bool verify_flatbuffer_root(std::span<const std::byte> bytes) noexcept {
  return !bytes.empty() && verify_flatbuffer_root_bytes<Root>(bytes);
}

template <typename Root>
[[nodiscard]] const Root* parse_flatbuffer_root(std::span<const std::byte> bytes) noexcept {
  return parse_flatbuffer_root_bytes<Root>(bytes);
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
    builder.Finish(root, flatbuffer_file_identifier_c_string_v<Root>.data());
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

}  // namespace detail
}  // namespace holoscan::schema

namespace holoscan {

/// @brief Delegate ordinary owning-schema traits to the native type's generated root binding.
template <typename Native>
  requires detail::HasFlatBufferSchemaNativeBinding<Native> &&
           (!schema::FlatBufferSchemaBinding<typename Native::TableType>::descriptor_tensor)
struct GeneratedSchemaNativeTraits<Native> {
  using root_type = typename Native::TableType;
  static constexpr std::size_t max_serialized_size =
      schema::FlatBufferSchemaBinding<root_type>::max_serialized_size;
};

/// @brief Delegate descriptor-Tensor traits to the native type's generated root binding.
template <typename Native>
  requires detail::HasFlatBufferSchemaNativeBinding<Native> &&
           schema::FlatBufferSchemaBinding<typename Native::TableType>::descriptor_tensor
struct DescriptorTensorSchemaNativeTraits<Native> {
  using root_type = typename Native::TableType;
  using Binding = schema::FlatBufferSchemaBinding<root_type>;
  static constexpr std::size_t metadata_capacity = Binding::metadata_capacity;
  static constexpr std::size_t tensor_count = Binding::tensor_count;
  static constexpr auto tensor_field_ids = Binding::tensor_field_ids;
  static constexpr auto tensor_field_names = [] {
    static_assert(
        requires { Binding::tensor_field_names; },
        "registered descriptor Tensor schemas require generated field names");
    if constexpr (requires { Binding::tensor_field_names; }) {
      return Binding::tensor_field_names;
    } else {
      return std::array<std::string_view, 0U>{};
    }
  }();

  /// @brief Copy independently owned non-Tensor fields when the provider qualifies that operation.
  /// @param value Original schema; no backing Tensor is retained in the result.
  /// @return Owning metadata with Tensor members unset, or the provider's failure.
  /// @note This optional capability does not change ordinary descriptor-schema admission.
  [[nodiscard]] static expected<Native, Error> clone_metadata(const Native& value)
    requires requires {
      { Binding::clone_metadata(value) } -> std::same_as<expected<Native, Error>>;
    }
  {
    return Binding::clone_metadata(value);
  }

  [[nodiscard]] static const Tensor* tensor_at(const Native& value,
                                               const std::size_t index) noexcept {
    return Binding::tensor_at(value, index);
  }

  [[nodiscard]] static const Tensor* tensor(const Native& value) noexcept
    requires(Binding::tensor_count == 1U)
  {  // NOLINT(whitespace/braces)
    return Binding::tensor_at(value, 0U);
  }
};

}  // namespace holoscan

namespace holoscan::schema::detail {

template <typename Root>
[[nodiscard]] SchemaPackageDescriptor<Root> make_flatbuffer_package(
    SchemaTransportProfile transport_profile = {}) noexcept {
  return SchemaPackageDescriptor<Root>{
      schema_identity<Root>(),
      SchemaCodecDescriptor<Root>{
          .verify = &verify_flatbuffer_root<Root>,
          .parse = &parse_flatbuffer_root<Root>,
      },
      SchemaCodecCapacity{},
      transport_profile,
  };
}

template <typename Native, typename Root>
  requires holoscan::detail::HasGeneratedSchemaNativeTraits<Native> &&
           std::same_as<typename GeneratedSchemaNativeTraits<Native>::root_type, Root>
[[nodiscard]] SchemaPackageDescriptor<Native> make_flatbuffer_native_package() noexcept {
  return SchemaPackageDescriptor<Native>{
      schema_identity<Native>(),
      SchemaCodecDescriptor<Native>{
          .verify = &verify_flatbuffer_root<Root>,
          .encode = &encode_flatbuffer_native<Native, Root>,
          .materialize = &materialize_flatbuffer_native<Native, Root>,
      },
      SchemaCodecCapacity{.max_serialized_size = generated_schema_max_serialized_size_v<Native>},
      SchemaTransportProfile{},
  };
}

template <typename Native, typename Root>
  requires holoscan::detail::HasDescriptorTensorSchemaNativeTraits<Native> &&
           std::same_as<typename DescriptorTensorSchemaNativeTraits<Native>::root_type, Root>
[[nodiscard]] SchemaPackageDescriptor<Native> make_flatbuffer_descriptor_native_package(
    const SchemaTransportProfile transport_profile) noexcept {
  return SchemaPackageDescriptor<Native>{
      schema_identity<Native>(),
      SchemaCodecDescriptor<Native>{
          .verify = &verify_flatbuffer_root<Root>,
      },
      SchemaCodecCapacity{},
      transport_profile,
  };
}

}  // namespace holoscan::schema::detail

namespace holoscan {

template <typename Root>
  requires detail::HasFlatBufferSchemaRootBinding<Root>
inline const SchemaPackageDescriptor<Root>& schema_package() noexcept {
  using Binding = schema::FlatBufferSchemaBinding<Root>;
  static const SchemaPackageDescriptor<Root> package = [] {
    if constexpr (Binding::descriptor_tensor) {
      return schema::detail::make_flatbuffer_package<Root>(
          detail::descriptor_tensor_schema_transport_profile_v<typename Binding::native_type>);
    } else {
      return schema::detail::make_flatbuffer_package<Root>();
    }
  }();
  return package;
}

template <typename Native>
  requires detail::HasFlatBufferSchemaNativeBinding<Native>
inline const SchemaPackageDescriptor<Native>& schema_package() noexcept {
  using Root = typename Native::TableType;
  using Binding = schema::FlatBufferSchemaBinding<Root>;
  static const SchemaPackageDescriptor<Native> package = [] {
    if constexpr (Binding::descriptor_tensor) {
      return schema::detail::make_flatbuffer_descriptor_native_package<Native, Root>(
          detail::descriptor_tensor_schema_transport_profile_v<Native>);
    } else {
      return schema::detail::make_flatbuffer_native_package<Native, Root>();
    }
  }();
  return package;
}

}  // namespace holoscan

namespace holoscan::schema {

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
    const Root& root, const std::size_t index) {
  return DescriptorFlatBufferTensorViewTraits<Root>::tensor_view_metadata_at(root, index);
}

template <typename Native, typename Root, std::size_t MetadataCapacity, typename Prepare,
          typename Reset, typename Encode, typename Assign>
struct DescriptorTensorPortTypeOpsFactory {
  static constexpr std::size_t kTensorCount =
      holoscan::detail::descriptor_tensor_schema_tensor_count_v<Native>;
  static constexpr std::uint64_t kAllTensorsMask = descriptor_tensor_all_fields_mask(kTensorCount);
  static constexpr SchemaTransportProfile kTransportProfile =
      holoscan::detail::descriptor_tensor_schema_transport_profile_v<Native>;
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
    SchemaId schema_family_id{};
    std::uint32_t metadata_size = 0U;
    std::uint32_t transport_profile_tag = 0U;
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
        read_descriptor_wire_value(
            bytes, kPayloadBase + offsetof(Payload, schema_family_id), schema_family_id) &&
        read_descriptor_wire_value(
            bytes, kPayloadBase + offsetof(Payload, metadata_size), metadata_size) &&
        read_descriptor_wire_value(bytes,
                                   kPayloadBase + offsetof(Payload, transport_profile_tag),
                                   transport_profile_tag) &&
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
    if (!fixed_fields_valid) {
      return make_unexpected(
          Error{ErrorCode::kProtocolError, "descriptor Tensor envelope fields are truncated"});
    }
    if (envelope_abi_version != Payload::expected_envelope_abi_version ||
        envelope_header_size != kSchemaEnvelopeHeaderSize ||
        schema_family_id != Payload::expected_schema_family_id ||
        transport_profile_tag != schema_transport_profile_tag(kTransportProfile)) {
      return make_unexpected(
          Error{ErrorCode::kProtocolError, "descriptor Tensor envelope identity is invalid"});
    }
    if (endpoint_generation != expected_generation) {
      return make_unexpected(
          Error{ErrorCode::kProtocolError, "descriptor Tensor endpoint generation does not match"});
    }
    if (metadata_size == 0U || metadata_size > Payload::metadata_capacity ||
        kMetadataOffset > bytes.size() || metadata_size > bytes.size() - kMetadataOffset) {
      return make_unexpected(
          Error{ErrorCode::kProtocolError, "descriptor Tensor metadata size is invalid"});
    }
    if (tensor_count != kTensorCount || reserved != 0U || presence_mask != kAllTensorsMask ||
        descriptor_mask != kAllTensorsMask) {
      return make_unexpected(
          Error{ErrorCode::kProtocolError, "descriptor Tensor carriage inventory is invalid"});
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
        if (!valid || envelope.descriptors[index].schema_id != Payload::expected_schema_family_id) {
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
                                    kPayloadBase + offsetof(Payload, schema_family_id),
                                    Payload::expected_schema_family_id) &&
        write_descriptor_wire_value(destination,
                                    kPayloadBase + offsetof(Payload, transport_profile_tag),
                                    schema_transport_profile_tag(kTransportProfile)) &&
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
      return descriptor_tensor_view_metadata_at_from_root(*root, index);
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
      const std::span<holoscan::Tensor> tensors,
      const holoscan::detail::DescriptorTensorWireMetadata wire_metadata,
      const holoscan::detail::DescriptorTensorSampleMetadata sample_metadata) noexcept {
    if (storage == nullptr || tensors.size() != kTensorCount) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto metadata_bytes = validated_metadata_bytes(bytes, wire_metadata.metadata);
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
        tensor_metadata = descriptor_tensor_view_metadata_at_from_root(*root, index);
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      if (!tensor_metadata || !holoscan::tensor_matches(tensors[index], *tensor_metadata)) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
    }
    return Assign{}(storage, *root, tensors, sample_metadata.metadata);
  }

  [[nodiscard]] static expected<void, Error> assign_sample(
      void* storage, const std::span<const std::byte> bytes, holoscan::Tensor tensor,
      const holoscan::detail::DescriptorTensorWireMetadata wire_metadata,
      const holoscan::detail::DescriptorTensorSampleMetadata sample_metadata) noexcept {
    if constexpr (kTensorCount == 1U) {
      std::array<holoscan::Tensor, 1U> tensors{std::move(tensor)};
      return assign_tensors_sample(storage, bytes, tensors, wire_metadata, sample_metadata);
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
        .transport_profile_tag = schema_transport_profile_tag(kTransportProfile),
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
        builder.Finish(*root, flatbuffer_file_identifier_c_string_v<Root>.data());
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
  static_assert(
      schema_identity_equal(schema_identity<Native>(), flatbuffer_schema_identity_v<Root>),
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
