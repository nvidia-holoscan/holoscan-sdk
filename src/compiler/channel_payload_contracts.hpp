/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <limits>
#include <optional>

#include <holoscan/core/execution_plan.hpp>
#include <holoscan/schema/schema_package.hpp>

namespace holoscan::detail {

/// Return the fixed native object width for a supported scalar payload identity.
[[nodiscard]] inline std::optional<std::uint64_t> scalar_payload_size_bytes(
    const PayloadIdentityRecord& identity) noexcept {
  if (identity.kind != PayloadIdentityKind::kScalar || !identity.scalar) {
    return std::nullopt;
  }
  switch (identity.scalar->kind) {
    case ScalarIdentityKind::kBoolean:
      // Process-local typed storage contains a native C++ bool. Its canonical catalog wire
      // representation remains independently fixed at one byte by the catalog codec.
      return sizeof(bool);
    case ScalarIdentityKind::kInt8:
    case ScalarIdentityKind::kUInt8:
      return 1U;
    case ScalarIdentityKind::kInt16:
    case ScalarIdentityKind::kUInt16:
      return 2U;
    case ScalarIdentityKind::kInt32:
    case ScalarIdentityKind::kUInt32:
    case ScalarIdentityKind::kFloat32:
      return 4U;
    case ScalarIdentityKind::kInt64:
    case ScalarIdentityKind::kUInt64:
    case ScalarIdentityKind::kFloat64:
      return 8U;
    case ScalarIdentityKind::kLocalEnum:
      switch (identity.scalar->native_object_size) {
        case 1U:
        case 2U:
        case 4U:
        case 8U:
          return identity.scalar->native_object_size;
        default:
          return std::nullopt;
      }
  }
  return std::nullopt;
}

/// Return the bounded Channel payload size implied by one frozen payload identity.
[[nodiscard]] inline std::optional<std::uint64_t> channel_payload_size_bytes(
    const PayloadIdentityRecord& identity, std::uint64_t serialized_payload_capacity_bytes,
    bool process_local_schema_envelope) noexcept {
  if (identity.kind == PayloadIdentityKind::kSchema && identity.schema &&
      serialized_payload_capacity_bytes != 0U) {
    if (identity.payload_kind == PayloadKind::kDescriptorBacked) {
      return serialized_payload_capacity_bytes;
    }
    if (identity.payload_kind != PayloadKind::kSerializedBytes) {
      return std::nullopt;
    }
    if (!process_local_schema_envelope) {
      return serialized_payload_capacity_bytes;
    }
    if (serialized_payload_capacity_bytes >
        std::numeric_limits<std::uint64_t>::max() - kSchemaEnvelopeHeaderSize) {
      return std::nullopt;
    }
    return serialized_payload_capacity_bytes + kSchemaEnvelopeHeaderSize;
  }
  if (identity.kind == PayloadIdentityKind::kPod && identity.pod &&
      identity.pod->object_size != 0U) {
    return identity.pod->object_size;
  }
  return scalar_payload_size_bytes(identity);
}

}  // namespace holoscan::detail
