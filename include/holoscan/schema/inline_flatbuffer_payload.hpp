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

#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"
#include "holoscan/schema/flatbuffer_schema_traits.hpp"
#include "holoscan/schema/schema_package.hpp"

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
template <typename Root, std::size_t Capacity>
struct InlineFlatBufferPayload {
  using root_type = Root;
  static constexpr std::size_t inline_capacity = Capacity;  ///< Maximum serialized byte count.
  static constexpr SchemaIdentity expected_schema_identity =
      flatbuffer_schema_identity_v<Root>;  ///< Exact generated-root semantic identity.
  static constexpr SchemaId expected_schema_id =
      expected_schema_identity.id;  ///< Complete compact semantic identity.
  static constexpr std::uint32_t expected_schema_version =
      flatbuffer_schema_version_v<Root>;  ///< Exact nonzero semantic version.

  std::uint32_t envelope_abi_version{kSchemaEnvelopeABIVersion};  ///< Exact wrapper ABI version.
  std::uint32_t envelope_header_size{kSchemaEnvelopeHeaderSize};  ///< Exact v2 prefix byte size.
  SchemaId schema_id{expected_schema_id};                         ///< Complete compact identity.
  std::uint32_t schema_version{expected_schema_version};          ///< Exact semantic version.
  std::uint32_t payload_size{};                                   ///< Valid bytes in `payload`.
  std::uint64_t endpoint_generation{};  ///< Attached endpoint generation, or zero before binding.
  std::array<std::byte, Capacity> payload{};  ///< Fixed-capacity serialized storage.

  /// @brief Return the versioned schema envelope prefix carried by this wrapper.
  /// @return Value prefix containing ABI, complete identity, size, and endpoint generation.
  [[nodiscard]] SchemaEnvelopeHeader envelope_header() const noexcept {
    return SchemaEnvelopeHeader{
        .abi_version = envelope_abi_version,
        .header_size = envelope_header_size,
        .schema_id = schema_id,
        .schema_version = schema_version,
        .payload_size = payload_size,
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
    const auto view = uint8_bytes();
    if (view.empty()) {
      return nullptr;
    }
    return flatbuffers::GetRoot<Root>(view.data());
  }

  /**
   * @brief Validate exact schema identity, size bounds, and FlatBuffer structure.
   *
   * @note Schema version matching is exact in this first slice. Additive FlatBuffers evolution can
   * be relaxed later once graph setup has an explicit compatibility policy.
   */
  [[nodiscard]] expected<void, Error> validate() const {
    if (envelope_abi_version != kSchemaEnvelopeABIVersion ||
        envelope_header_size != kSchemaEnvelopeHeaderSize) {
      return make_unexpected(Error{ErrorCode::kCodecError, "FlatBuffer envelope ABI mismatch"});
    }
    if (schema_id != expected_schema_id || schema_version != expected_schema_version) {
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
    const auto* data = reinterpret_cast<const std::uint8_t*>(bytes.data());
    auto verifier = flatbuffers::Verifier(data, bytes.size());
    if (!verifier.VerifyBuffer<Root>(nullptr)) {
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
 * Application-defined schemas can specialize `FlatBufferSchemaTraits<Root>` with
 * `default_inline_capacity` and then use this alias instead of repeating the
 * capacity at each port declaration.
 */
template <typename Root>
using InlineFlatBufferPayloadFor =
    InlineFlatBufferPayload<Root, flatbuffer_default_inline_capacity_v<Root>>;

static_assert(std::is_standard_layout_v<InlineFlatBufferPayload<Time, 512>>);
static_assert(std::is_trivially_copyable_v<InlineFlatBufferPayload<Time, 512>>);

}  // namespace holoscan::schema
