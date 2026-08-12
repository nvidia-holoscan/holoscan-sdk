/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "channel_catalog_binding.hpp"

namespace holoscan::channel_internal {

namespace pod_codec_detail {

[[nodiscard]] constexpr ScalarIdentityKind scalar_identity(PodScalarKind kind) noexcept {
  switch (kind) {
    case PodScalarKind::kUInt8:
      return ScalarIdentityKind::kUInt8;
    case PodScalarKind::kInt8:
      return ScalarIdentityKind::kInt8;
    case PodScalarKind::kUInt16:
      return ScalarIdentityKind::kUInt16;
    case PodScalarKind::kInt16:
      return ScalarIdentityKind::kInt16;
    case PodScalarKind::kUInt32:
      return ScalarIdentityKind::kUInt32;
    case PodScalarKind::kInt32:
      return ScalarIdentityKind::kInt32;
    case PodScalarKind::kUInt64:
      return ScalarIdentityKind::kUInt64;
    case PodScalarKind::kInt64:
      return ScalarIdentityKind::kInt64;
    case PodScalarKind::kFloat32:
      return ScalarIdentityKind::kFloat32;
    case PodScalarKind::kFloat64:
      return ScalarIdentityKind::kFloat64;
  }
  return ScalarIdentityKind::kLocalEnum;
}

}  // namespace pod_codec_detail

/// @brief Carry the derived, immutable digest evidence for one validated POD identity.
struct PodContractEvidence {
  std::array<std::uint8_t, 32> identity_digest{};  ///< Complete type-identity SHA-256.
  TypeKey key{};                                   ///< First 16 digest bytes in digest order.
  std::array<std::uint8_t, 32> layout_digest{};    ///< Complete layout SHA-256.
};

/// @brief Validate and derive canonical digest evidence for one owned POD identity.
/// @param identity Complete pointer-free identity copied from a specialization or catalog.
/// @return Derived evidence, or a checked invalid-identity error.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<PodContractEvidence, Error>
derive_pod_contract_evidence(const PodIdentityRecord& identity);

/// @brief Compare two already validated POD attachment authorities exactly.
/// @param expected_identity Complete frozen endpoint identity.
/// @param expected_evidence Frozen digest and compact-key evidence.
/// @param actual_identity Complete attaching-client identity.
/// @param actual_evidence Attaching-client digest and compact-key evidence.
/// @return Success only for identical contracts, including distinct C++ alias types.
/// @retval ErrorCode::kInvalidArgument Either compact key is inconsistent with its full digest.
/// @retval ErrorCode::kIncompatible A type mismatch or authority/key collision was detected.
///
/// The caller validates each digest against its canonical preimage when freezing the catalog or
/// constructing the local identity. Keeping this comparison separate provides a deterministic
/// compact-key collision-injection seam without weakening artifact validation.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> validate_pod_attachment_contract(
    const PodIdentityRecord& expected_identity, const PodContractEvidence& expected_evidence,
    const PodIdentityRecord& actual_identity, const PodContractEvidence& actual_evidence);

/// @brief Encode and decode one frozen POD layout without copying whole-object representation.
///
/// The attaching template materializes `ChannelPodLayout<T>` into the pointer-free identity once.
/// This source-private codec then converts only the declared scalar fields to the canonical
/// little-endian representation. Decode writes every byte of the admitted padding-free storage, so
/// the public type facade may start `T` with `std::bit_cast` without default construction.
class HOLOSCAN_CORE_EXPORT PodFieldwiseCodec {
 public:
  /// @brief Validate immutable geometry before binding a codec to an endpoint.
  /// @param identity Complete pointer-free POD identity and layout.
  /// @return Success, or `ErrorCode::kInvalidArgument` for invalid geometry.
  [[nodiscard]] static expected<void, Error> validate(const PodIdentityRecord& identity);

  /// @brief Encode one native object representation field by field.
  /// @param identity Previously validated complete POD identity and layout.
  /// @param native Exact-size native object view. Only declared field subspans are read.
  /// @param wire Exact-size destination.
  /// @return Success or `ErrorCode::kInvalidArgument` for an incorrect destination size.
  [[nodiscard]] static expected<void, Error> encode(const PodIdentityRecord& identity,
                                                    std::span<const std::byte> native,
                                                    std::span<std::byte> wire);

  /// @brief Encode using geometry already validated during endpoint binding.
  /// @param identity Frozen POD identity accepted by `PodFieldwiseCodec::validate()`.
  /// @param native Exact-size native object view. Only declared field subspans are read.
  /// @param wire Exact-size destination.
  /// @return Success or `ErrorCode::kInvalidArgument` for an incorrect span size.
  [[nodiscard]] static expected<void, Error> encode_prevalidated(const PodIdentityRecord& identity,
                                                                 std::span<const std::byte> native,
                                                                 std::span<std::byte> wire);

  /// @brief Decode canonical bytes into fully initialized native object storage.
  /// @param identity Previously validated complete POD identity and layout.
  /// @param wire Exact-size canonical little-endian/IEEE representation.
  /// @param native Exact-size destination. Every byte is initialized on success.
  /// @return Success or a checked local-geometry/protocol error without partial success.
  [[nodiscard]] static expected<void, Error> decode(const PodIdentityRecord& identity,
                                                    std::span<const std::byte> wire,
                                                    std::span<std::byte> native);

  /// @brief Decode using geometry already validated during endpoint binding.
  /// @param identity Frozen POD identity accepted by `PodFieldwiseCodec::validate()`.
  /// @param wire Exact-size canonical little-endian/IEEE representation.
  /// @param native Exact-size destination. Every byte is initialized on success.
  /// @return Success or a checked span-size error without partial success.
  [[nodiscard]] static expected<void, Error> decode_prevalidated(const PodIdentityRecord& identity,
                                                                 std::span<const std::byte> wire,
                                                                 std::span<std::byte> native);

 private:
  template <std::endian NativeEndian>
  [[nodiscard]] static expected<void, Error> encode_prevalidated_for_native_endian(
      const PodIdentityRecord& identity, std::span<const std::byte> native,
      std::span<std::byte> wire) {
    static_assert(NativeEndian == std::endian::little || NativeEndian == std::endian::big);
    if (native.size() != identity.object_size || wire.size() != identity.object_size) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "ENDPOINT_POD_CODEC_SIZE_INVALID"});
    }
    for (const auto& field : identity.fields) {
      const std::size_t element_size = detail::pod_scalar_size(field.scalar_kind);
      if constexpr (NativeEndian == std::endian::little) {
        const std::size_t byte_count = element_size * field.element_count;
        std::copy_n(
            native.begin() + field.byte_offset, byte_count, wire.begin() + field.byte_offset);
      } else {
        for (std::size_t index = 0U; index < field.element_count; ++index) {
          const std::size_t offset = field.byte_offset + (index * element_size);
          auto encoded = CatalogScalarCodec::encode_for_native_endian<NativeEndian>(
              pod_codec_detail::scalar_identity(field.scalar_kind),
              native.subspan(offset, element_size),
              wire.subspan(offset, element_size));
          if (!encoded) {
            return make_unexpected(Error{ErrorCode::kCodecError, "ENDPOINT_POD_CODEC_INVALID"});
          }
        }
      }
    }
    return {};
  }

  template <std::endian NativeEndian>
  [[nodiscard]] static expected<void, Error> decode_prevalidated_for_native_endian(
      const PodIdentityRecord& identity, std::span<const std::byte> wire,
      std::span<std::byte> native) {
    static_assert(NativeEndian == std::endian::little || NativeEndian == std::endian::big);
    if (native.size() != identity.object_size) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "ENDPOINT_POD_CODEC_SIZE_INVALID"});
    }
    if (wire.size() != identity.object_size) {
      return make_unexpected(Error{ErrorCode::kProtocolError, "ENDPOINT_POD_CODEC_SIZE_INVALID"});
    }
    for (const auto& field : identity.fields) {
      const std::size_t element_size = detail::pod_scalar_size(field.scalar_kind);
      if constexpr (NativeEndian == std::endian::little) {
        const std::size_t byte_count = element_size * field.element_count;
        std::copy_n(
            wire.begin() + field.byte_offset, byte_count, native.begin() + field.byte_offset);
      } else {
        for (std::size_t index = 0U; index < field.element_count; ++index) {
          const std::size_t offset = field.byte_offset + (index * element_size);
          auto decoded = CatalogScalarCodec::decode_for_native_endian<NativeEndian>(
              pod_codec_detail::scalar_identity(field.scalar_kind),
              wire.subspan(offset, element_size),
              native.subspan(offset, element_size));
          if (!decoded) {
            return make_unexpected(Error{ErrorCode::kProtocolError, "ENDPOINT_POD_CODEC_INVALID"});
          }
        }
      }
    }
    return {};
  }

  friend class PodFieldwiseCodecTestAccess;
};

}  // namespace holoscan::channel_internal
