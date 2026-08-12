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
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

#include <flatbuffers/flatbuffer_builder.h>
#include <flatbuffers/flatbuffers.h>

#include "holoscan/channel/types.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"
#include "holoscan/schema/flatbuffer_schema_traits.hpp"
#include "holoscan/schema/schema_package.hpp"
#include "holoscan/schema/tensor_materialization.hpp"

namespace holoscan::schema {

/** @brief Extract Tensor view metadata from descriptor-backed schema metadata.
 * @tparam Root Generated FlatBuffer metadata root type.
 * @details Schemas that support opaque external-buffer Tensor materialization specialize this
 * trait. */
template <typename Root>
struct DescriptorFlatBufferTensorViewTraits {
  /** @brief Report that the schema does not define Tensor view metadata.
   * @param metadata_bytes Serialized schema metadata bytes.
   * @return ErrorCode::kNotImplemented for the unspecialized trait. */
  [[nodiscard]] static expected<TensorViewMetadata, Error> tensor_view_metadata(
      [[maybe_unused]] std::span<const std::byte> metadata_bytes) {
    return make_unexpected(
        Error{ErrorCode::kNotImplemented, "schema does not expose Tensor view metadata"});
  }
};

/// @brief Supply the common Tensor-view trait implementation for a generated FlatBuffers root.
///
/// The adapter verifies the enclosing root, selects a generated Tensor field by index, and decodes
/// that field through \ref tensor_view_metadata. A singular specialization may inherit from this
/// adapter and name its generated accessor:
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
    flatbuffers::Verifier verifier{reinterpret_cast<const std::uint8_t*>(metadata_bytes.data()),
                                   metadata_bytes.size()};
    if (!verifier.VerifyBuffer<Root>(nullptr)) {
      return make_unexpected(
          Error{ErrorCode::kCodecError, "descriptor Tensor root verification failed"});
    }
    const Root* root =
        flatbuffers::GetRoot<Root>(reinterpret_cast<const std::uint8_t*>(metadata_bytes.data()));
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

  // ABI version 3 requires every declared Tensor field to be present and descriptor-carried.
  // Optional presence or alternate carriage requires explicit protocol evolution before either
  // mask can admit a proper subset.
  static constexpr std::uint64_t all_fields_mask = MaxTensors == 64U
                                                       ? std::numeric_limits<std::uint64_t>::max()
                                                       : (std::uint64_t{1U} << MaxTensors) - 1U;

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
  static constexpr SchemaId expected_schema_id =
      expected_schema_identity.id;  ///< Complete compact semantic identity.
  static constexpr std::uint32_t expected_schema_version =
      flatbuffer_schema_version_v<Root>;  ///< Exact nonzero semantic version.

  std::uint32_t envelope_abi_version{expected_envelope_abi_version};  ///< Exact wrapper ABI.
  std::uint32_t envelope_header_size{kSchemaEnvelopeHeaderSize};      ///< Exact prefix byte size.
  SchemaId schema_id{expected_schema_id};                 ///< Complete compact identity.
  std::uint32_t schema_version{expected_schema_version};  ///< Exact semantic version.
  std::uint32_t metadata_size{};                          ///< Valid bytes in `metadata`.
  std::uint64_t endpoint_generation{};    ///< Attached endpoint generation, or zero before binding.
  descriptor_carriage_type descriptor{};  ///< Canonically indexed descriptor carriage.
  std::array<std::byte, MetadataCapacity> metadata{};  ///< Fixed-capacity metadata storage.

  /// @brief Return the versioned schema envelope prefix carried by this wrapper.
  /// @return Value prefix containing ABI, complete identity, size, and endpoint generation.
  [[nodiscard]] SchemaEnvelopeHeader envelope_header() const noexcept {
    return SchemaEnvelopeHeader{
        .abi_version = envelope_abi_version,
        .header_size = envelope_header_size,
        .schema_id = schema_id,
        .schema_version = schema_version,
        .payload_size = metadata_size,
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
    if (metadata_bytes.empty()) {
      return nullptr;
    }
    return flatbuffers::GetRoot<Root>(reinterpret_cast<const std::uint8_t*>(metadata_bytes.data()));
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
   * @note Schema version matching is exact in this first slice. Additive FlatBuffers evolution can
   * be relaxed later once graph setup has an explicit compatibility policy.
   */
  [[nodiscard]] expected<void, Error> validate() const {
    if (envelope_abi_version != expected_envelope_abi_version) {
      return make_unexpected(
          Error{ErrorCode::kCodecError,
                "descriptor Tensor envelope ABI version mismatch (version 3 required)"});
    }
    if (envelope_header_size != kSchemaEnvelopeHeaderSize) {
      return make_unexpected(
          Error{ErrorCode::kCodecError, "descriptor Tensor envelope header size mismatch"});
    }
    if (schema_id != expected_schema_id || schema_version != expected_schema_version) {
      return make_unexpected(Error{ErrorCode::kCodecError, "FlatBuffer schema identity mismatch"});
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
   * @brief Build a descriptor-backed payload from a FlatBufferBuilder after verification.
   * @param builder Builder containing serialized metadata.
   * @param payload_descriptor Transport descriptor for the payload.
   * @param endpoint_generation Attached endpoint generation, or zero for an unbound local value.
   * @return Payload or Error for invalid metadata or descriptor.
   */
  [[nodiscard]] static expected<DescriptorFlatBufferPayload, Error> copy_from_builder(
      const flatbuffers::FlatBufferBuilder& builder,
      const ChannelSampleDescriptor& payload_descriptor, std::uint64_t endpoint_generation = 0U)
    requires(MaxTensors == 1U)
  {  // NOLINT(whitespace/braces)
    const auto* data = reinterpret_cast<const std::byte*>(builder.GetBufferPointer());
    return copy_from(std::span<const std::byte>(data, builder.GetSize()),
                     payload_descriptor,
                     endpoint_generation);
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
    const auto* data = reinterpret_cast<const std::byte*>(builder.GetBufferPointer());
    return copy_from(std::span<const std::byte>(data, builder.GetSize()),
                     payload_descriptor,
                     endpoint_generation);
  }

  /**
   * @brief Build a descriptor-backed payload from serialized metadata bytes after verification.
   * @param metadata_bytes Serialized FlatBuffer metadata.
   * @param payload_descriptor Transport descriptor for the payload.
   * @param endpoint_generation Attached endpoint generation, or zero for an unbound local value.
   * @return Payload or Error for empty, oversized, invalid, or mismatched metadata.
   */
  [[nodiscard]] static expected<DescriptorFlatBufferPayload, Error> copy_from(
      const std::span<const std::byte> metadata_bytes,
      const ChannelSampleDescriptor& payload_descriptor, std::uint64_t endpoint_generation = 0U)
    requires(MaxTensors == 1U)
  {  // NOLINT(whitespace/braces)
    auto descriptor_status = validate_descriptor(payload_descriptor);
    if (!descriptor_status) {
      return make_unexpected(std::move(descriptor_status).error());
    }
    auto metadata_status = verify_metadata(metadata_bytes);
    if (!metadata_status) {
      return make_unexpected(std::move(metadata_status).error());
    }
    return copy_from_verified_metadata(metadata_bytes,
                                       payload_descriptor_from_descriptor(payload_descriptor),
                                       endpoint_generation);
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
    auto descriptor_status = validate_descriptor(payload_descriptor);
    if (!descriptor_status) {
      return make_unexpected(std::move(descriptor_status).error());
    }
    auto metadata_status = verify_metadata(metadata_bytes);
    if (!metadata_status) {
      return make_unexpected(std::move(metadata_status).error());
    }
    return copy_from_verified_metadata(metadata_bytes, payload_descriptor, endpoint_generation);
  }

  /**
   * @brief Build a descriptor-backed payload from already-verified metadata bytes.
   *
   * External or untrusted metadata should use copy_from() so the generated FlatBuffers verifier
   * runs at the boundary.
   * @param metadata_bytes Serialized metadata previously verified by the exact generated verifier.
   * @param payload_descriptor Transport descriptor for the associated payload.
   * @param endpoint_generation Attached endpoint generation, or zero for an unbound local value.
   * @return Payload or Error for empty, oversized, or invalid descriptor state.
   */
  [[nodiscard]] static expected<DescriptorFlatBufferPayload, Error> copy_from_trusted_metadata(
      const std::span<const std::byte> metadata_bytes,
      const ChannelSampleDescriptor& payload_descriptor, std::uint64_t endpoint_generation = 0U)
    requires(MaxTensors == 1U)
  {  // NOLINT(whitespace/braces)
    auto descriptor_status = validate_descriptor(payload_descriptor);
    if (!descriptor_status) {
      return make_unexpected(std::move(descriptor_status).error());
    }
    if (metadata_bytes.empty()) {
      return make_unexpected(Error{ErrorCode::kCodecError, "FlatBuffer metadata bytes are empty"});
    }
    if (metadata_bytes.size() > MetadataCapacity) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "FlatBuffer metadata exceeds inline capacity"});
    }
    return copy_from_verified_metadata(metadata_bytes,
                                       payload_descriptor_from_descriptor(payload_descriptor),
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
    auto descriptor_status = validate_descriptor(payload_descriptor);
    if (!descriptor_status) {
      return make_unexpected(std::move(descriptor_status).error());
    }
    if (metadata_bytes.empty()) {
      return make_unexpected(Error{ErrorCode::kCodecError, "FlatBuffer metadata bytes are empty"});
    }
    if (metadata_bytes.size() > MetadataCapacity) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "FlatBuffer metadata exceeds inline capacity"});
    }
    return copy_from_verified_metadata(metadata_bytes, payload_descriptor, endpoint_generation);
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
    if (metadata_bytes.empty()) {
      return make_unexpected(Error{ErrorCode::kCodecError, "FlatBuffer metadata bytes are empty"});
    }
    if (metadata_bytes.size() > MetadataCapacity) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "FlatBuffer metadata exceeds inline capacity"});
    }
    const auto* data = reinterpret_cast<const std::uint8_t*>(metadata_bytes.data());
    auto verifier = flatbuffers::Verifier(data, metadata_bytes.size());
    if (!verifier.VerifyBuffer<Root>(nullptr)) {
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
    return DescriptorFlatBufferTensorViewTraits<Root>::tensor_view_metadata(payload.bytes());
  }

  /**
   * @brief Extract Tensor view metadata from an already-verified payload.
   * @param payload Descriptor-backed payload whose metadata was verified and remains unmodified.
   * @return Tensor view metadata on success, or Error when the root is missing or incompatible.
   * @warning The caller must first validate the same immutable payload bytes.
   */
  [[nodiscard]] static expected<TensorViewMetadata, Error> tensor_view_metadata_from_verified(
      const DescriptorFlatBufferPayload& payload) {
    const Root* root = payload.root();
    if (root == nullptr) {
      return make_unexpected(Error{ErrorCode::kCodecError, "descriptor Tensor root is missing"});
    }
    if constexpr (requires {
                    {
                      DescriptorFlatBufferTensorViewTraits<Root>::tensor_view_metadata(*root)
                    } -> std::same_as<expected<TensorViewMetadata, Error>>;
                  }) {
      return DescriptorFlatBufferTensorViewTraits<Root>::tensor_view_metadata(*root);
    } else {
      // Preserve compatibility with manual traits that only implement the verified-byte API.
      return DescriptorFlatBufferTensorViewTraits<Root>::tensor_view_metadata(payload.bytes());
    }
  }

  /**
   * @brief Validate that a descriptor has the local-buffer shape supported today.
   *
   * @note This is a structural pre-check shared by generic descriptor-backed payloads. The
   * materializing receive path still performs the authoritative process-identity and
   * TensorSidecarStore lookup before returning a Tensor handle. Keeping that operating-system
   * check in the private transport implementation makes this installed header portable.
   */
  [[nodiscard]] static expected<void, Error> validate_descriptor(
      const ChannelSampleDescriptor& payload_descriptor) {
    if (!payload_descriptor.has_supported_abi()) {
      return make_unexpected(Error{ErrorCode::kCodecError, "Channel descriptor ABI mismatch"});
    }
    if (payload_descriptor.sidecar_token == 0U) {
      return make_unexpected(Error{ErrorCode::kCodecError, "Channel descriptor token is empty"});
    }
    if (payload_descriptor.payload_kind != ChannelPayloadKind::kLocalBufferDescriptor) {
      return make_unexpected(
          Error{ErrorCode::kCodecError, "Channel descriptor payload kind is not a local buffer"});
    }
    return {};
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
      const std::span<const std::byte> metadata_bytes, const PayloadDescriptor& payload_descriptor,
      std::uint64_t endpoint_generation)
    requires(MaxTensors == 1U)
  {  // NOLINT(whitespace/braces)
    DescriptorFlatBufferPayload wrapper;
    wrapper.metadata_size = static_cast<std::uint32_t>(metadata_bytes.size());
    wrapper.endpoint_generation = endpoint_generation;
    wrapper.descriptor.descriptors[0] = payload_descriptor;
    std::memcpy(wrapper.metadata.data(), metadata_bytes.data(), metadata_bytes.size());
    return wrapper;
  }

  [[nodiscard]] static DescriptorFlatBufferPayload copy_from_verified_metadata(
      const std::span<const std::byte> metadata_bytes,
      const ChannelSampleDescriptor& payload_descriptor, std::uint64_t endpoint_generation)
    requires(MaxTensors == 1U)
  {  // NOLINT(whitespace/braces)
    return copy_from_verified_metadata(metadata_bytes,
                                       payload_descriptor_from_descriptor(payload_descriptor),
                                       endpoint_generation);
  }

  [[nodiscard]] static DescriptorFlatBufferPayload copy_from_verified_metadata(
      const std::span<const std::byte> metadata_bytes,
      const std::span<const PayloadDescriptor> payload_descriptors,
      std::uint64_t endpoint_generation) {
    DescriptorFlatBufferPayload wrapper;
    wrapper.metadata_size = static_cast<std::uint32_t>(metadata_bytes.size());
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
 * Application-defined schemas can specialize `FlatBufferSchemaTraits<Root>` with
 * `default_descriptor_metadata_capacity` and then use this alias for metadata
 * plus descriptor transport.
 */
namespace detail {

template <typename Root>
[[nodiscard]] consteval std::size_t descriptor_flatbuffer_tensor_count() {
  if constexpr (requires {
                  {
                    DescriptorFlatBufferTensorViewTraits<Root>::tensor_count
                  } -> std::convertible_to<std::size_t>;
                }) {
    constexpr std::size_t kTensorCount =
        static_cast<std::size_t>(DescriptorFlatBufferTensorViewTraits<Root>::tensor_count);
    static_assert(kTensorCount > 0U,
                  "DescriptorFlatBufferTensorViewTraits<Root>::tensor_count must be positive");
    return kTensorCount;
  } else if constexpr (requires(const Root& root) {
                         DescriptorFlatBufferTensorViewTraits<Root>::tensor_view_metadata_at(
                             root, std::size_t{});
                       }) {
    static_assert(!std::same_as<Root, Root>,
                  "indexed DescriptorFlatBufferTensorViewTraits<Root> must declare tensor_count");
  }
  // Preserve the original singular-trait compatibility path. Indexed traits must declare their
  // complete generated field count above so a stale multi-Tensor specialization cannot truncate.
  return 1U;
}

}  // namespace detail

template <typename Root>
using DescriptorFlatBufferPayloadFor =
    DescriptorFlatBufferPayload<Root, flatbuffer_default_descriptor_metadata_capacity_v<Root>,
                                detail::descriptor_flatbuffer_tensor_count<Root>()>;

static_assert(std::is_standard_layout_v<DescriptorFlatBufferPayload<Image, 2048>>);
static_assert(std::is_trivially_copyable_v<DescriptorFlatBufferPayload<Image, 2048>>);

}  // namespace holoscan::schema
