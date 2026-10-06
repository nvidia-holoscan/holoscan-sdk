/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <span>
#include <type_traits>
#include <utility>

#include <flatbuffers/flatbuffer_builder.h>
#include <flatbuffers/flatbuffers.h>

#include <holoscan/channel/types.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/schema/flatbuffer_root_access.hpp>
#include <holoscan/schema/flatbuffer_schema_traits.hpp>
#include <holoscan/schema/schema_package.hpp>
#include <holoscan/schema/tensor_materialization.hpp>

namespace holoscan::schema {

/** @brief Extract Tensor view metadata from descriptor-backed schema metadata.
 * @tparam Root Generated FlatBuffer metadata root type.
 * @details Schemas that support opaque external-buffer Tensor materialization specialize this
 * trait. */
template <typename Root>
struct DescriptorFlatBufferTensorViewTraits {
  /// @brief Report that the schema does not define an indexed Tensor inventory.
  static constexpr std::size_t tensor_count = 0U;

  /** @brief Report that the schema does not define indexed Tensor view metadata.
   * @param metadata_bytes Serialized schema metadata bytes.
   * @param index Tensor accessor index.
   * @return ErrorCode::kNotImplemented for the unspecialized trait. */
  [[nodiscard]] static expected<TensorViewMetadata, Error> tensor_view_metadata_at(
      [[maybe_unused]] std::span<const std::byte> metadata_bytes,
      [[maybe_unused]] std::size_t index) {
    return make_unexpected(
        Error{ErrorCode::kNotImplemented, "schema does not expose Tensor view metadata"});
  }
};

/// @brief Supply the common Tensor-view trait implementation for a generated FlatBuffers root.
///
/// The adapter verifies the enclosing root, selects a generated Tensor field by index, and decodes
/// that field through \ref tensor_view_metadata. A one-field specialization inherits from this
/// adapter with one generated accessor:
///
/// @code{.cpp}
/// template <>
/// struct DescriptorFlatBufferTensorViewTraits<MyRoot>
///     : DescriptorFlatBufferTensorViewAdapter<MyRoot, &MyRoot::tensor> {};
/// @endcode
///
/// The accessor parameter pack is ordered so multi-Tensor schemas can retain the same
/// specialization shape when indexed Runtime carriage is available.
///
/// @tparam Root Generated FlatBuffers table root.
/// @tparam TensorAccessors Ordered generated member functions returning Tensor table pointers.
template <typename Root, auto... TensorAccessors>
struct DescriptorFlatBufferTensorViewAdapter {
  static_assert(sizeof...(TensorAccessors) > 0U,
                "descriptor Tensor view adapter requires at least one Tensor accessor");
  static_assert((std::same_as<decltype(TensorAccessors), const schema::Tensor* (Root::*)() const> &&
                 ...),
                "descriptor Tensor view accessors must be generated const Tensor member functions");

  /// @brief Return the number of indexed generated Tensor fields.
  static constexpr std::size_t tensor_count = sizeof...(TensorAccessors);

  /// @brief Verify a generated root and decode one indexed metadata-only Tensor member.
  /// @param root Already-verified generated root.
  /// @param index Tensor accessor index in canonical field order.
  /// @return Tensor view metadata, or Error for an invalid index, a missing Tensor field, or
  /// embedded Tensor data.
  [[nodiscard]] static expected<TensorViewMetadata, Error> tensor_view_metadata_at(
      const Root& root, const std::size_t index) {
    if (index >= tensor_count) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "descriptor Tensor index is invalid"});
    }
    constexpr std::array accessors{TensorAccessors...};
    const schema::Tensor* tensor = std::invoke(accessors[index], root);
    if (tensor == nullptr) {
      return make_unexpected(Error{ErrorCode::kCodecError, "descriptor Tensor field is missing"});
    }
    return holoscan::schema::tensor_view_metadata(*tensor);
  }

  /// @brief Decode the Tensor member of an already-verified singular generated root.
  /// @param root Already-verified generated root.
  /// @return Tensor view metadata, or Error for missing or embedded Tensor data.
  [[nodiscard]] static expected<TensorViewMetadata, Error> tensor_view_metadata(const Root& root) {
    static_assert(tensor_count == 1U,
                  "singular descriptor Tensor view requires exactly one generated accessor");
    return tensor_view_metadata_at(root, 0U);
  }

  /// @brief Verify serialized metadata and decode one indexed Tensor member.
  /// @param metadata_bytes Serialized enclosing-root metadata.
  /// @param index Tensor accessor index in canonical field order.
  /// @return Tensor view metadata, or Error for malformed bytes, an invalid index, a missing
  /// Tensor field, or embedded Tensor data.
  [[nodiscard]] static expected<TensorViewMetadata, Error> tensor_view_metadata_at(
      const std::span<const std::byte> metadata_bytes, const std::size_t index) {
    if (metadata_bytes.empty()) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "descriptor Tensor metadata is empty"});
    }
    if (!detail::verify_flatbuffer_root_bytes<Root>(metadata_bytes)) {
      return make_unexpected(
          Error{ErrorCode::kCodecError, "descriptor Tensor root verification failed"});
    }
    const Root* root = detail::parse_flatbuffer_root_bytes<Root>(metadata_bytes);
    if (root == nullptr) {
      return make_unexpected(Error{ErrorCode::kCodecError, "descriptor Tensor root is missing"});
    }
    return tensor_view_metadata_at(*root, index);
  }

  /// @brief Verify serialized metadata for a singular root and decode its Tensor member.
  /// @param metadata_bytes Serialized enclosing-root metadata.
  /// @return Tensor view metadata, or Error for malformed or nonsingular metadata.
  [[nodiscard]] static expected<TensorViewMetadata, Error> tensor_view_metadata(
      const std::span<const std::byte> metadata_bytes) {
    static_assert(tensor_count == 1U,
                  "singular descriptor Tensor view requires exactly one generated accessor");
    return tensor_view_metadata_at(metadata_bytes, 0U);
  }
};

/// @brief Carry the descriptor-array envelope's explicit per-field carriage state.
/// @tparam MaxTensors Positive static Tensor-field count, at most 64.
template <std::size_t MaxTensors>
struct DescriptorTensorCarriage {
  static_assert(MaxTensors > 0U && MaxTensors <= 64U,
                "descriptor Tensor carriage supports 1 through 64 fields");

  // The current ABI requires every declared Tensor field to be present and descriptor-carried.
  // Optional presence or alternate carriage requires explicit protocol evolution before either
  // mask can admit a proper subset.
  static constexpr std::uint64_t all_fields_mask = descriptor_tensor_all_fields_mask(MaxTensors);

  std::uint32_t tensor_count{MaxTensors};                   ///< Exact occupied descriptor count.
  std::uint32_t reserved{};                                 ///< Must remain zero for this ABI.
  std::uint64_t presence_mask{all_fields_mask};             ///< Present generated Tensor members.
  std::uint64_t descriptor_mask{all_fields_mask};           ///< Members using descriptor carriage.
  std::array<PayloadDescriptor, MaxTensors> descriptors{};  ///< Canonically indexed references.
};

/**
 * @brief Fixed-size schema metadata plus a payload descriptor for external payload data.
 *
 * This wrapper is intended for schema messages such as Image/Tensor metadata whose large or
 * device-backed data should not be materialized into the FlatBuffer for same-process delivery. The
 * FlatBuffer metadata remains inline and self-contained, while descriptor identifies the associated
 * payload reference carrying the actual buffer. Same-process paths use a local sidecar reference;
 * cross-process paths preserve a bounded transport-opaque external-buffer reference without
 * exposing or narrowing its source-private carrier to the temporary same-process descriptor
 * format.
 *
 * @warning This POD wrapper only stores and validates the metadata-plus-descriptor envelope. When
 * sent through generic ChannelPublisher<DescriptorFlatBufferPayload<...>>, the descriptor is still
 * just copied as inline metadata and is not resolved. Same-process Tensor materialization requires
 * the dedicated
 * ChannelRuntime::create_tensor_schema_publisher() and create_tensor_schema_subscriber() endpoints,
 * which mint a descriptor from TensorSidecarStore and resolve it back to a Tensor on receive.
 *
 * @tparam Root Generated FlatBuffer metadata root type.
 * @tparam MetadataCapacity Maximum serialized FlatBuffer metadata bytes stored inline.
 * @tparam MaxTensors Static generated Tensor-field count.
 */
template <typename Root, std::size_t MetadataCapacity, std::size_t MaxTensors>
struct DescriptorFlatBufferPayload {
  static_assert(MaxTensors > 0U && MaxTensors <= 64U,
                "descriptor FlatBuffer payload supports 1 through 64 Tensor fields");

  using root_type = Root;
  using descriptor_carriage_type = DescriptorTensorCarriage<MaxTensors>;
  static constexpr std::size_t metadata_capacity =
      MetadataCapacity;  ///< Maximum serialized metadata byte count.
  static constexpr std::size_t max_tensors = MaxTensors;  ///< Static Tensor-field count.
  static constexpr std::uint32_t expected_envelope_abi_version =
      kDescriptorTensorSchemaEnvelopeABIVersion;  ///< Exact descriptor-envelope ABI.
  static constexpr SchemaIdentity expected_schema_identity =
      flatbuffer_schema_identity_v<Root>;  ///< Exact generated-root semantic identity.
  static constexpr SchemaId expected_schema_family_id =
      expected_schema_identity.family_id;  ///< Stable compact compatibility-family identity.

  /// @brief Return the generated descriptor-Tensor carriage profile for this root package.
  [[nodiscard]] static std::uint32_t expected_transport_profile_tag() noexcept {
    return schema_transport_profile_tag(schema_package<Root>().transport_profile());
  }

  /// @brief Return whether the root package exactly describes this descriptor wrapper geometry.
  [[nodiscard]] static bool has_compatible_transport_profile() noexcept {
    const SchemaTransportProfile profile = schema_package<Root>().transport_profile();
    return profile.kind == SchemaTransportProfileKind::kDescriptorTensor &&
           schema_transport_profile_tag(profile) != 0U &&
           profile.tensor_field_ids.size() == MaxTensors &&
           schema_transport_profile_field_mask(profile) ==
               descriptor_carriage_type::all_fields_mask &&
           profile.metadata_capacity_bytes == MetadataCapacity;
  }

  std::uint32_t envelope_abi_version{expected_envelope_abi_version};  ///< Exact wrapper ABI.
  std::uint32_t envelope_header_size{kSchemaEnvelopeHeaderSize};      ///< Exact prefix byte size.
  SchemaId schema_family_id{expected_schema_family_id};  ///< Stable compatibility family.
  std::uint32_t metadata_size{};                         ///< Valid bytes in `metadata`.
  std::uint32_t transport_profile_tag{};  ///< Exact generated descriptor carriage profile.
  std::uint64_t endpoint_generation{};    ///< Attached endpoint generation, or zero before binding.
  descriptor_carriage_type descriptor{};  ///< Canonically indexed descriptor carriage.
  std::array<std::byte, MetadataCapacity> metadata{};  ///< Fixed-capacity metadata storage.

  /// @brief Return the versioned schema envelope prefix carried by this wrapper.
  /// @return Value prefix containing ABI, complete identity, size, and endpoint generation.
  [[nodiscard]] SchemaEnvelopeHeader envelope_header() const noexcept {
    return SchemaEnvelopeHeader{
        .abi_version = envelope_abi_version,
        .header_size = envelope_header_size,
        .schema_family_id = schema_family_id,
        .payload_size = metadata_size,
        .transport_profile_tag = transport_profile_tag,
        .endpoint_generation = endpoint_generation,
    };
  }

  /**
   * @brief Return a view over the serialized FlatBuffer metadata bytes.
   */
  [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
    return {metadata.data(), metadata_size <= MetadataCapacity ? metadata_size : MetadataCapacity};
  }

  /**
   * @brief Return a byte view suitable for FlatBuffers generated APIs.
   */
  [[nodiscard]] std::span<const std::uint8_t> uint8_bytes() const noexcept {
    const auto view = bytes();
    return {reinterpret_cast<const std::uint8_t*>(view.data()), view.size()};
  }

  /**
   * @brief Return a pointer to the FlatBuffer root, or nullptr when metadata is empty.
   *
   * @warning Call this only after bytes were produced by copy_from(), copy_from_builder(), or after
   * validate() succeeds. It intentionally does not re-run the FlatBuffers verifier.
   */
  [[nodiscard]] const Root* root() const noexcept { return root_from_metadata(bytes()); }

  /**
   * @brief Return a pointer to a FlatBuffer root in verified metadata bytes.
   *
   * @warning Call this only after verify_metadata() succeeds. The returned pointer is valid only
   * while the supplied metadata storage remains alive and unmodified.
   */
  [[nodiscard]] static const Root* root_from_metadata(
      const std::span<const std::byte> metadata_bytes) noexcept {
    return detail::parse_flatbuffer_root_bytes<Root>(metadata_bytes);
  }

  /// @brief Return the encoded descriptor count bounded by this static envelope.
  /// @return Encoded canonical descriptor count clamped to the static field count.
  /// @note Call validate() before treating this count as authoritative peer input.
  [[nodiscard]] constexpr std::size_t descriptor_count() const noexcept {
    return std::min<std::size_t>(descriptor.tensor_count, MaxTensors);
  }

  /// @brief Return one canonical descriptor, or null for an out-of-range index.
  /// @param index Zero-based canonical Tensor-field index.
  /// @return Corresponding descriptor, or nullptr when \p index is out of range.
  [[nodiscard]] constexpr const PayloadDescriptor* descriptor_at(
      const std::size_t index) const noexcept {
    return index < descriptor.tensor_count && index < MaxTensors ? &descriptor.descriptors[index]
                                                                 : nullptr;
  }

  /// @brief Return one mutable canonical descriptor, or null for an out-of-range index.
  /// @param index Zero-based canonical Tensor-field index.
  /// @return Corresponding mutable descriptor, or nullptr when \p index is out of range.
  [[nodiscard]] constexpr PayloadDescriptor* descriptor_at(const std::size_t index) noexcept {
    return index < descriptor.tensor_count && index < MaxTensors ? &descriptor.descriptors[index]
                                                                 : nullptr;
  }

  /// @brief Return the generated-field presence mask.
  /// @return Bit mask naming generated Tensor fields present in the schema metadata.
  [[nodiscard]] constexpr std::uint64_t tensor_presence_mask() const noexcept {
    return descriptor.presence_mask;
  }

  /// @brief Return the fields carried by external descriptors.
  /// @return Bit mask naming Tensor fields whose storage uses descriptor carriage.
  [[nodiscard]] constexpr std::uint64_t tensor_descriptor_mask() const noexcept {
    return descriptor.descriptor_mask;
  }

  /**
   * @brief Validate exact schema identity, metadata bytes, and same-process descriptor shape.
   *
   * @note Family identity admits compatible FlatBuffers revisions. The descriptor transport
   * profile remains exact.
   */
  [[nodiscard]] expected<void, Error> validate() const {
    if (!has_compatible_transport_profile()) {
      return make_unexpected(
          Error{ErrorCode::kCodecError, "descriptor Tensor transport profile mismatch"});
    }
    if (envelope_abi_version != expected_envelope_abi_version) {
      return make_unexpected(
          Error{ErrorCode::kCodecError, "descriptor Tensor envelope ABI version mismatch"});
    }
    if (envelope_header_size != kSchemaEnvelopeHeaderSize) {
      return make_unexpected(
          Error{ErrorCode::kCodecError, "descriptor Tensor envelope header size mismatch"});
    }
    if (schema_family_id != expected_schema_family_id) {
      return make_unexpected(Error{ErrorCode::kCodecError, "FlatBuffer schema family mismatch"});
    }
    if (transport_profile_tag != expected_transport_profile_tag()) {
      return make_unexpected(
          Error{ErrorCode::kCodecError, "FlatBuffer transport profile tag mismatch"});
    }
    if (metadata_size == 0U) {
      return make_unexpected(Error{ErrorCode::kCodecError, "FlatBuffer metadata bytes are empty"});
    }
    if (metadata_size > MetadataCapacity) {
      return make_unexpected(
          Error{ErrorCode::kCodecError, "FlatBuffer metadata size exceeds inline capacity"});
    }
    if (descriptor.tensor_count != MaxTensors || descriptor.reserved != 0U ||
        descriptor.presence_mask != descriptor.all_fields_mask ||
        descriptor.descriptor_mask != descriptor.all_fields_mask) {
      return make_unexpected(Error{ErrorCode::kCodecError, "descriptor Tensor carriage mismatch"});
    }
    for (const auto& field_descriptor : descriptor.descriptors) {
      auto descriptor_status = validate_descriptor(field_descriptor);
      if (!descriptor_status) {
        return make_unexpected(std::move(descriptor_status).error());
      }
    }
    return verify_metadata(bytes());
  }

  /// @brief Verify attached peer metadata before exposing a generated root view.
  /// @param package Exact generated package validated and bound during attachment.
  /// @param expected_generation Exact nonzero endpoint generation bound during attachment.
  /// @return Borrowed verified generated root, or protocol/local error with no typed exposure.
  /// @warning A successful pointer remains valid only while this wrapper is alive and unmodified.
  [[nodiscard]] expected<const Root*, Error> verified_root(
      const SchemaPackageDescriptor<Root>& package, std::uint64_t expected_generation) const {
    return verify_and_parse_received_schema(package,
                                            envelope_header(),
                                            expected_schema_identity,
                                            expected_generation,
                                            bytes(),
                                            expected_envelope_abi_version);
  }

  /**
   * @brief Build a descriptor-backed payload from a FlatBufferBuilder and public payload reference.
   * @param builder Builder containing serialized metadata.
   * @param payload_descriptor Public transport reference for the associated payload.
   * @param endpoint_generation Attached endpoint generation, or zero for an unbound local value.
   * @return Payload or Error for invalid metadata or descriptor.
   */
  [[nodiscard]] static expected<DescriptorFlatBufferPayload, Error> copy_from_builder(
      const flatbuffers::FlatBufferBuilder& builder, const PayloadDescriptor& payload_descriptor,
      std::uint64_t endpoint_generation = 0U)
    requires(MaxTensors == 1U)
  {  // NOLINT(whitespace/braces)
    return copy_from_builder(
        builder, std::span<const PayloadDescriptor>{&payload_descriptor, 1U}, endpoint_generation);
  }

  /**
   * @brief Build a descriptor-backed payload from serialized metadata bytes and public payload
   * reference after verification.
   * @param metadata_bytes Serialized FlatBuffer metadata.
   * @param payload_descriptor Public transport reference for the associated payload.
   * @param endpoint_generation Attached endpoint generation, or zero for an unbound local value.
   * @return Payload or Error for empty, oversized, invalid, or mismatched metadata.
   */
  [[nodiscard]] static expected<DescriptorFlatBufferPayload, Error> copy_from(
      const std::span<const std::byte> metadata_bytes, const PayloadDescriptor& payload_descriptor,
      std::uint64_t endpoint_generation = 0U)
    requires(MaxTensors == 1U)
  {  // NOLINT(whitespace/braces)
    return copy_from(metadata_bytes,
                     std::span<const PayloadDescriptor>{&payload_descriptor, 1U},
                     endpoint_generation);
  }

  /**
   * @brief Build a descriptor-backed payload from already-verified metadata bytes and public
   * payload reference.
   * @param metadata_bytes Serialized metadata previously verified by the exact generated verifier.
   * @param payload_descriptor Public transport reference for the associated payload.
   * @param endpoint_generation Attached endpoint generation, or zero for an unbound local value.
   * @return Payload or Error for empty, oversized, or invalid descriptor state.
   */
  [[nodiscard]] static expected<DescriptorFlatBufferPayload, Error> copy_from_trusted_metadata(
      const std::span<const std::byte> metadata_bytes, const PayloadDescriptor& payload_descriptor,
      std::uint64_t endpoint_generation = 0U)
    requires(MaxTensors == 1U)
  {  // NOLINT(whitespace/braces)
    return copy_from_trusted_metadata(metadata_bytes,
                                      std::span<const PayloadDescriptor>{&payload_descriptor, 1U},
                                      endpoint_generation);
  }

  /// @brief Build a descriptor Tensor payload from a metadata builder and descriptor array.
  [[nodiscard]] static expected<DescriptorFlatBufferPayload, Error> copy_from_builder(
      const flatbuffers::FlatBufferBuilder& builder,
      const std::span<const PayloadDescriptor> payload_descriptors,
      std::uint64_t endpoint_generation = 0U)
    requires(MaxTensors >= 1U)
  {  // NOLINT(whitespace/braces)
    const auto* data = reinterpret_cast<const std::byte*>(builder.GetBufferPointer());
    return copy_from(std::span<const std::byte>(data, builder.GetSize()),
                     payload_descriptors,
                     endpoint_generation);
  }

  /// @brief Build a verified descriptor Tensor payload from canonical descriptors.
  [[nodiscard]] static expected<DescriptorFlatBufferPayload, Error> copy_from(
      const std::span<const std::byte> metadata_bytes,
      const std::span<const PayloadDescriptor> payload_descriptors,
      std::uint64_t endpoint_generation = 0U)
    requires(MaxTensors >= 1U)
  {  // NOLINT(whitespace/braces)
    auto descriptors_status = validate_descriptors(payload_descriptors);
    if (!descriptors_status) {
      return make_unexpected(std::move(descriptors_status).error());
    }
    auto profile_status = validate_transport_profile();
    if (!profile_status) {
      return make_unexpected(std::move(profile_status).error());
    }
    auto metadata_status = verify_metadata(metadata_bytes);
    if (!metadata_status) {
      return make_unexpected(std::move(metadata_status).error());
    }
    return copy_from_verified_metadata(metadata_bytes, payload_descriptors, endpoint_generation);
  }

  /// @brief Build a descriptor Tensor payload from already verified metadata.
  [[nodiscard]] static expected<DescriptorFlatBufferPayload, Error> copy_from_trusted_metadata(
      const std::span<const std::byte> metadata_bytes,
      const std::span<const PayloadDescriptor> payload_descriptors,
      std::uint64_t endpoint_generation = 0U)
    requires(MaxTensors >= 1U)
  {  // NOLINT(whitespace/braces)
    auto descriptors_status = validate_descriptors(payload_descriptors);
    if (!descriptors_status) {
      return make_unexpected(std::move(descriptors_status).error());
    }
    auto profile_status = validate_transport_profile();
    if (!profile_status) {
      return make_unexpected(std::move(profile_status).error());
    }
    if (metadata_bytes.empty()) {
      return make_unexpected(Error{ErrorCode::kCodecError, "FlatBuffer metadata bytes are empty"});
    }
    if (metadata_bytes.size() > MetadataCapacity) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "FlatBuffer metadata exceeds inline capacity"});
    }
    return copy_from_verified_metadata(metadata_bytes, payload_descriptors, endpoint_generation);
  }

  /**
   * @brief Verify serialized metadata bytes as a FlatBuffer for Root.
   * @param metadata_bytes Serialized FlatBuffer metadata.
   * @return Empty success or Error for empty, oversized, or invalid metadata.
   */
  [[nodiscard]] static expected<void, Error> verify_metadata(
      const std::span<const std::byte> metadata_bytes) {
    auto profile_status = validate_transport_profile();
    if (!profile_status) {
      return profile_status;
    }
    if (metadata_bytes.empty()) {
      return make_unexpected(Error{ErrorCode::kCodecError, "FlatBuffer metadata bytes are empty"});
    }
    if (metadata_bytes.size() > MetadataCapacity) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "FlatBuffer metadata exceeds inline capacity"});
    }
    if (!detail::verify_flatbuffer_root_bytes<Root>(metadata_bytes)) {
      return make_unexpected(
          Error{ErrorCode::kCodecError, "FlatBuffer metadata verification failed"});
    }
    return {};
  }

  /**
   * @brief Extract Tensor view metadata for buffer-handle materialization.
   * @param payload Descriptor-backed payload containing serialized schema metadata.
   * @return Tensor view metadata on success, or Error when the schema does not expose a compatible
   * Tensor view.
   */
  [[nodiscard]] static expected<TensorViewMetadata, Error> tensor_view_metadata(
      const DescriptorFlatBufferPayload& payload) {
    if constexpr (MaxTensors == 1U) {
      return DescriptorFlatBufferTensorViewTraits<Root>::tensor_view_metadata_at(payload.bytes(),
                                                                                 0U);
    } else {
      return make_unexpected(
          Error{ErrorCode::kNotSupported, "Tensor view requires an explicit field index"});
    }
  }

  /**
   * @brief Extract one indexed Tensor view from already-verified multi-Tensor metadata.
   * @param payload Descriptor-backed payload whose metadata was verified and remains unmodified.
   * @param index Canonical Tensor-field index.
   * @return Tensor view metadata on success, or Error when the root, index, or field is invalid.
   * @warning The caller must first validate the same immutable payload bytes.
   */
  [[nodiscard]] static expected<TensorViewMetadata, Error> tensor_view_metadata_at_from_verified(
      const DescriptorFlatBufferPayload& payload, const std::size_t index) {
    return tensor_view_metadata_at_from_verified(payload.bytes(), index);
  }

  /**
   * @brief Extract one indexed Tensor view from already-verified metadata bytes.
   * @param metadata_bytes Serialized metadata previously verified and kept immutable.
   * @param index Canonical Tensor-field index.
   * @return Tensor view metadata on success, or Error when the root, index, or field is invalid.
   * @warning The caller must first verify the same immutable metadata bytes.
   */
  [[nodiscard]] static expected<TensorViewMetadata, Error> tensor_view_metadata_at_from_verified(
      const std::span<const std::byte> metadata_bytes, const std::size_t index) {
    const Root* root = root_from_metadata(metadata_bytes);
    if (root == nullptr) {
      return make_unexpected(Error{ErrorCode::kCodecError, "descriptor Tensor root is missing"});
    }
    return DescriptorFlatBufferTensorViewTraits<Root>::tensor_view_metadata_at(*root, index);
  }

  /**
   * @brief Validate that a public payload reference has a supported structural shape.
   * @param payload_descriptor Local or opaque external-buffer payload reference to validate.
   * @return Empty success or Error when the descriptor kind or local identity is invalid.
   */
  [[nodiscard]] static expected<void, Error> validate_descriptor(
      const PayloadDescriptor& payload_descriptor) {
    if (payload_descriptor.payload_kind == ChannelPayloadKind::kLocalBufferDescriptor) {
      if (payload_descriptor.local.token == 0U) {
        return make_unexpected(Error{ErrorCode::kCodecError, "Payload descriptor token is empty"});
      }
      return {};
    }
    if (payload_descriptor.payload_kind == ChannelPayloadKind::kBufferHandle) {
      // External-buffer import is validated by the cross-process materialization path. Keep this
      // header-only schema wrapper structural so generic schema payload users do not need to link
      // a backend implementation just to validate copied metadata.
      if (!payload_descriptor.is_buffer_handle()) {
        return make_unexpected(
            Error{ErrorCode::kCodecError, "Opaque payload reference is empty or malformed"});
      }
      return {};
    }
    return make_unexpected(
        Error{ErrorCode::kCodecError, "Payload descriptor kind is not supported"});
  }

 private:
  [[nodiscard]] static expected<void, Error> validate_transport_profile() {
    if (!has_compatible_transport_profile()) {
      return make_unexpected(
          Error{ErrorCode::kCodecError, "descriptor Tensor transport profile mismatch"});
    }
    return {};
  }

  [[nodiscard]] static expected<void, Error> validate_descriptors(
      const std::span<const PayloadDescriptor> payload_descriptors)
    requires(MaxTensors >= 1U)
  {  // NOLINT(whitespace/braces)
    if (payload_descriptors.size() != MaxTensors) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "descriptor Tensor count mismatch"});
    }
    for (const auto& payload_descriptor : payload_descriptors) {
      auto descriptor_status = validate_descriptor(payload_descriptor);
      if (!descriptor_status) {
        return make_unexpected(std::move(descriptor_status).error());
      }
    }
    return {};
  }

  [[nodiscard]] static DescriptorFlatBufferPayload copy_from_verified_metadata(
      const std::span<const std::byte> metadata_bytes,
      const std::span<const PayloadDescriptor> payload_descriptors,
      std::uint64_t endpoint_generation) {
    DescriptorFlatBufferPayload wrapper;
    wrapper.metadata_size = static_cast<std::uint32_t>(metadata_bytes.size());
    wrapper.transport_profile_tag = expected_transport_profile_tag();
    wrapper.endpoint_generation = endpoint_generation;
    std::copy(payload_descriptors.begin(),
              payload_descriptors.end(),
              wrapper.descriptor.descriptors.begin());
    std::memcpy(wrapper.metadata.data(), metadata_bytes.data(), metadata_bytes.size());
    return wrapper;
  }
};

/**
 * @brief Descriptor-backed FlatBuffer payload using the root type's trait capacity.
 *
 * Application-defined descriptor schemas use their generated `metadata_capacity` bound for
 * metadata plus descriptor transport.
 */
namespace detail {

template <typename Root>
[[nodiscard]] consteval std::size_t descriptor_flatbuffer_tensor_count() {
  constexpr std::size_t kTensorCount =
      static_cast<std::size_t>(DescriptorFlatBufferTensorViewTraits<Root>::tensor_count);
  static_assert(kTensorCount > 0U,
                "DescriptorFlatBufferTensorViewTraits<Root>::tensor_count must be positive");
  static_assert(kTensorCount <= 64U,
                "DescriptorFlatBufferTensorViewTraits<Root>::tensor_count must not exceed 64");
  return kTensorCount;
}

}  // namespace detail

template <typename Root>
using DescriptorFlatBufferPayloadFor =
    DescriptorFlatBufferPayload<Root, flatbuffer_descriptor_metadata_capacity_v<Root>,
                                detail::descriptor_flatbuffer_tensor_count<Root>()>;

namespace detail {

/// @brief Stand in for any metadata root when asserting the payload's own layout.
///
/// These two properties belong to the template and not to any schema: the payload stores
/// descriptors and inline metadata bytes, never a Root value, so Root may be incomplete. Asserting
/// against a tag declared here keeps the check from depending on which domain schema Core happens
/// to declare -- the original spelling used Image, which stopped compiling the moment that schema
/// moved to holoscan::sensor_io, for reasons that had nothing to do with layout.
struct DescriptorPayloadLayoutProbe;

using DescriptorPayloadLayoutProbePayload =
    DescriptorFlatBufferPayload<DescriptorPayloadLayoutProbe, 2048>;

}  // namespace detail

static_assert(std::is_standard_layout_v<detail::DescriptorPayloadLayoutProbePayload>);
static_assert(std::is_trivially_copyable_v<detail::DescriptorPayloadLayoutProbePayload>);

}  // namespace holoscan::schema
