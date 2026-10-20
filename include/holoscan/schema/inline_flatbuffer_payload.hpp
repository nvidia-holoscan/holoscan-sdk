/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>
#include <utility>

#include <flatbuffers/flatbuffer_builder.h>
#include <flatbuffers/flatbuffers.h>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/schema/flatbuffer_root_access.hpp>
#include <holoscan/schema/flatbuffer_schema_traits.hpp>
#include <holoscan/schema/schema_package.hpp>

namespace holoscan::schema {

/**
 * @brief Fixed-capacity inline payload for self-contained FlatBuffer messages.
 *
 * This type is intentionally standard-layout and trivially copyable so it can
 * be sent through `ChannelPublisher<T>` / `ChannelSubscriber<T>` as an inline
 * typed payload. It is intended for small schemas whose complete data fits in
 * the inline payload. Large or device-backed Tensor/Image schemas should use a
 * descriptor-backed payload instead of materializing bytes into this wrapper.
 *
 * @tparam Root Generated FlatBuffer root type.
 * @tparam Capacity Maximum serialized FlatBuffer bytes stored inline.
 */
template <OrdinaryFlatBufferSchemaRoot Root, std::size_t Capacity>
struct InlineFlatBufferPayload {
  using root_type = Root;
  static constexpr std::size_t inline_capacity = Capacity;  ///< Maximum serialized byte count.
  static constexpr SchemaIdentity expected_schema_identity =
      flatbuffer_schema_identity_v<Root>;  ///< Exact generated-root semantic identity.
  static constexpr SchemaId expected_schema_family_id =
      expected_schema_identity.family_id;  ///< Stable compact compatibility-family identity.

  std::uint32_t envelope_abi_version{kSchemaEnvelopeABIVersion};  ///< Exact wrapper ABI version.
  std::uint32_t envelope_header_size{kSchemaEnvelopeHeaderSize};  ///< Exact prefix byte size.
  SchemaId schema_family_id{expected_schema_family_id};           ///< Stable compatibility family.
  std::uint32_t payload_size{};                                   ///< Valid bytes in `payload`.
  std::uint32_t transport_profile_tag{};  ///< Reserved zero in the inline envelope ABI.
  std::uint64_t endpoint_generation{};    ///< Attached endpoint generation, or zero before binding.
  std::array<std::byte, Capacity> payload{};  ///< Fixed-capacity serialized storage.

  /// @brief Return the versioned schema envelope prefix carried by this wrapper.
  /// @return Value prefix containing ABI, complete identity, size, and endpoint generation.
  [[nodiscard]] SchemaEnvelopeHeader envelope_header() const noexcept {
    return SchemaEnvelopeHeader{
        .abi_version = envelope_abi_version,
        .header_size = envelope_header_size,
        .schema_family_id = schema_family_id,
        .payload_size = payload_size,
        .transport_profile_tag = transport_profile_tag,
        .endpoint_generation = endpoint_generation,
    };
  }

  /**
   * @brief Return a view over the serialized FlatBuffer bytes.
   */
  [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
    return {payload.data(), payload_size <= Capacity ? payload_size : Capacity};
  }

  /**
   * @brief Return a byte view suitable for FlatBuffers generated APIs.
   */
  [[nodiscard]] std::span<const std::uint8_t> uint8_bytes() const noexcept {
    const auto view = bytes();
    return {reinterpret_cast<const std::uint8_t*>(view.data()), view.size()};
  }

  /**
   * @brief Return whether the wrapper carries no serialized bytes.
   */
  [[nodiscard]] bool empty() const noexcept { return payload_size == 0U; }

  /**
   * @brief Return a pointer to the FlatBuffer root, or nullptr when empty.
   *
   * @warning Call this only after bytes were produced by copy_from(), copy_from_builder(), or after
   * validate() succeeds. It intentionally does not re-run the FlatBuffers verifier.
   */
  [[nodiscard]] const Root* root() const noexcept {
    return detail::parse_flatbuffer_root_bytes<Root>(bytes());
  }

  /**
   * @brief Validate exact schema identity, size bounds, and FlatBuffer structure.
   */
  [[nodiscard]] expected<void, Error> validate() const {
    if (envelope_abi_version != kSchemaEnvelopeABIVersion ||
        envelope_header_size != kSchemaEnvelopeHeaderSize) {
      return make_unexpected(Error{ErrorCode::kCodecError, "FlatBuffer envelope ABI mismatch"});
    }
    if (schema_family_id != expected_schema_family_id || transport_profile_tag != 0U) {
      return make_unexpected(Error{ErrorCode::kCodecError, "FlatBuffer schema identity mismatch"});
    }
    if (payload_size == 0U) {
      return make_unexpected(Error{ErrorCode::kCodecError, "FlatBuffer payload bytes are empty"});
    }
    if (payload_size > Capacity) {
      return make_unexpected(
          Error{ErrorCode::kCodecError, "FlatBuffer payload size exceeds inline capacity"});
    }
    return verify_bytes(bytes());
  }

  /// @brief Verify an attached peer message before exposing a generated root view.
  /// @param package Exact generated package validated and bound during attachment.
  /// @param expected_generation Exact nonzero endpoint generation bound during attachment.
  /// @return Borrowed verified generated root, or protocol/local error with no typed exposure.
  /// @warning A successful pointer remains valid only while this wrapper is alive and unmodified.
  [[nodiscard]] expected<const Root*, Error> verified_root(
      const SchemaPackageDescriptor<Root>& package, std::uint64_t expected_generation) const {
    return verify_and_parse_received_schema(
        package, envelope_header(), expected_schema_identity, expected_generation, bytes());
  }

  /**
   * @brief Build an inline payload from a FlatBufferBuilder after verification.
   * @param builder Builder containing the serialized root.
   * @param endpoint_generation Attached endpoint generation, or zero for an unbound local value.
   * @return Payload or error for empty, oversized, or malformed bytes.
   */
  [[nodiscard]] static expected<InlineFlatBufferPayload, Error> copy_from_builder(
      const flatbuffers::FlatBufferBuilder& builder, std::uint64_t endpoint_generation = 0U) {
    const auto* data = reinterpret_cast<const std::byte*>(builder.GetBufferPointer());
    return copy_from(std::span<const std::byte>(data, builder.GetSize()), endpoint_generation);
  }

  /**
   * @brief Build an inline payload from serialized bytes after verification.
   * @param bytes Serialized FlatBuffer bytes.
   * @param endpoint_generation Attached endpoint generation, or zero for an unbound local value.
   * @return Payload or error for empty, oversized, or malformed bytes.
   */
  [[nodiscard]] static expected<InlineFlatBufferPayload, Error> copy_from(
      const std::span<const std::byte> bytes, std::uint64_t endpoint_generation = 0U) {
    auto verified = verify_bytes(bytes);
    if (!verified) {
      return make_unexpected(std::move(verified).error());
    }
    return copy_from_verified_bytes(bytes, endpoint_generation);
  }

  /**
   * @brief Build an inline payload from already-verified local transport bytes.
   *
   * External or untrusted bytes should use copy_from() so the generated
   * FlatBuffers verifier runs at the boundary.
   * @param bytes Serialized bytes previously verified by the exact generated verifier.
   * @param endpoint_generation Attached endpoint generation, or zero for an unbound local value.
   * @return Payload or error for empty or oversized bytes.
   */
  [[nodiscard]] static expected<InlineFlatBufferPayload, Error> copy_from_trusted_bytes(
      const std::span<const std::byte> bytes, std::uint64_t endpoint_generation = 0U) {
    if (bytes.empty()) {
      return make_unexpected(Error{ErrorCode::kCodecError, "FlatBuffer payload bytes are empty"});
    }
    if (bytes.size() > Capacity) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "FlatBuffer payload exceeds inline capacity"});
    }
    return copy_from_verified_bytes(bytes, endpoint_generation);
  }

  /**
   * @brief Verify serialized bytes as a FlatBuffer for Root.
   */
  [[nodiscard]] static expected<void, Error> verify_bytes(const std::span<const std::byte> bytes) {
    if (bytes.empty()) {
      return make_unexpected(Error{ErrorCode::kCodecError, "FlatBuffer payload bytes are empty"});
    }
    if (bytes.size() > Capacity) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "FlatBuffer payload exceeds inline capacity"});
    }
    if (!detail::verify_flatbuffer_root_bytes<Root>(bytes)) {
      return make_unexpected(
          Error{ErrorCode::kCodecError, "FlatBuffer payload verification failed"});
    }
    return {};
  }

 private:
  [[nodiscard]] static InlineFlatBufferPayload copy_from_verified_bytes(
      const std::span<const std::byte> bytes, std::uint64_t endpoint_generation) {
    InlineFlatBufferPayload wrapper;
    wrapper.payload_size = static_cast<std::uint32_t>(bytes.size());
    wrapper.endpoint_generation = endpoint_generation;
    std::memcpy(wrapper.payload.data(), bytes.data(), bytes.size());
    return wrapper;
  }
};

/**
 * @brief Fixed-capacity inline FlatBuffer payload using the root type's trait capacity.
 *
 * Application-defined ordinary schemas use their complete serialized-payload bound. Descriptor-
 * Tensor roots require \c DescriptorFlatBufferPayloadFor<Root> so their metadata and Tensor
 * descriptors retain one exact carriage profile.
 */
template <OrdinaryFlatBufferSchemaRoot Root>
using InlineFlatBufferPayloadFor =
    InlineFlatBufferPayload<Root, flatbuffer_schema_max_serialized_size_v<Root>>;

}  // namespace holoscan::schema
