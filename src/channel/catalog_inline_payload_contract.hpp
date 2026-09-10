/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/schema/schema_package.hpp>

#include "channel_payload_limits.hpp"

namespace holoscan::channel_internal {

/// Fixed data-attribution prefix carried by every ordinary catalog data frame.
inline constexpr std::size_t kCatalogDataAttributionBytes = 24U;

/// Maximum complete ordinary catalog frame, derived from the common inline-payload limit.
inline constexpr std::size_t kMaximumCatalogDataFrameBytes = kMaximumInlinePayloadBytes;

/// Return the canonical catalog wire width for one scalar identity.
[[nodiscard]] constexpr std::size_t catalog_scalar_wire_size(
    const ScalarIdentityKind kind) noexcept {
  switch (kind) {
    case ScalarIdentityKind::kBoolean:
    case ScalarIdentityKind::kUInt8:
    case ScalarIdentityKind::kInt8:
      return 1U;
    case ScalarIdentityKind::kUInt16:
    case ScalarIdentityKind::kInt16:
      return 2U;
    case ScalarIdentityKind::kUInt32:
    case ScalarIdentityKind::kInt32:
    case ScalarIdentityKind::kFloat32:
      return 4U;
    case ScalarIdentityKind::kUInt64:
    case ScalarIdentityKind::kInt64:
    case ScalarIdentityKind::kFloat64:
      return 8U;
    case ScalarIdentityKind::kLocalEnum:
      return 0U;
  }
  return 0U;
}

/// Return the native object width accepted for one catalog scalar identity.
[[nodiscard]] constexpr std::size_t catalog_scalar_native_size(
    const ScalarIdentityKind kind) noexcept {
  return kind == ScalarIdentityKind::kBoolean ? sizeof(bool) : catalog_scalar_wire_size(kind);
}

/// Carry one ordinary catalog endpoint's bounded application, framing, and bucket geometry.
struct CatalogInlinePayloadContract {
  std::uint64_t native_payload_capacity_bytes{};  ///< Application materialization bound.
  std::uint64_t wire_payload_capacity_bytes{};    ///< Encoded payload including schema envelope.
  std::uint64_t framing_overhead_bytes{};         ///< Attribution plus optional schema envelope.
  std::uint64_t required_frame_bytes{};           ///< Complete catalog data-frame bound.
  bool frame_fits_inline_transport{};  ///< Whether a fixed-inline HoloMQ bucket admits the frame.
};

/// Derive one ordinary scalar, POD, or serialized-schema catalog frame contract.
[[nodiscard]] constexpr std::optional<CatalogInlinePayloadContract> catalog_inline_payload_contract(
    const PayloadIdentityRecord& identity,
    const std::uint64_t serialized_payload_capacity_bytes = 0U) noexcept {
  std::uint64_t native_payload_capacity_bytes{};
  std::uint64_t wire_payload_capacity_bytes{};
  std::uint64_t framing_overhead_bytes{kCatalogDataAttributionBytes};
  if (identity.kind == PayloadIdentityKind::kScalar && identity.scalar) {
    native_payload_capacity_bytes = catalog_scalar_native_size(identity.scalar->kind);
    wire_payload_capacity_bytes = catalog_scalar_wire_size(identity.scalar->kind);
  } else if (identity.kind == PayloadIdentityKind::kPod && identity.pod) {
    native_payload_capacity_bytes = identity.pod->object_size;
    wire_payload_capacity_bytes = identity.pod->object_size;
  } else if (identity.kind == PayloadIdentityKind::kSchema && identity.schema &&
             identity.payload_kind == PayloadKind::kSerializedBytes &&
             serialized_payload_capacity_bytes != 0U) {
    native_payload_capacity_bytes = serialized_payload_capacity_bytes;
    if (serialized_payload_capacity_bytes >
        std::numeric_limits<std::uint64_t>::max() - kSchemaEnvelopeHeaderSize) {
      return std::nullopt;
    }
    wire_payload_capacity_bytes = serialized_payload_capacity_bytes + kSchemaEnvelopeHeaderSize;
    framing_overhead_bytes += kSchemaEnvelopeHeaderSize;
  } else {
    return std::nullopt;
  }
  if (native_payload_capacity_bytes == 0U || wire_payload_capacity_bytes == 0U ||
      wire_payload_capacity_bytes >
          std::numeric_limits<std::uint64_t>::max() - kCatalogDataAttributionBytes) {
    return std::nullopt;
  }
  const std::uint64_t required_frame_bytes =
      wire_payload_capacity_bytes + kCatalogDataAttributionBytes;
  const bool frame_fits_inline_transport =
      required_frame_bytes <= std::numeric_limits<std::size_t>::max()
          ? inline_payload_bucket_size(static_cast<std::size_t>(required_frame_bytes)).has_value()
          : false;
  return CatalogInlinePayloadContract{
      .native_payload_capacity_bytes = native_payload_capacity_bytes,
      .wire_payload_capacity_bytes = wire_payload_capacity_bytes,
      .framing_overhead_bytes = framing_overhead_bytes,
      .required_frame_bytes = required_frame_bytes,
      .frame_fits_inline_transport = frame_fits_inline_transport,
  };
}

}  // namespace holoscan::channel_internal
