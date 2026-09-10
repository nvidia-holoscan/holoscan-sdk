/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>

#include <holoscan/core/payload_identity.hpp>
#include <holoscan/core/schema_identity.hpp>
#include <holoscan/core/tensor_contract.hpp>

namespace holoscan {

/// @brief Carry producer-side schema and interface Tensor materialization intent.
///
/// Omitting this carrier resolves to
/// \ref InterfaceTensorEmitPolicy::kRequireExportable with no inline cap. When
/// producer and boundary carriers both apply, one explicit side supplies the
/// effective policy, identical sides agree, and a policy/cap mismatch rejects
/// the edge. \ref max_inline_bytes is required and positive only for
/// \ref InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded;
/// every non-inline policy rejects an inline cap. \ref max_tensor_byte_span
/// bounds staging for a Tensor-bearing schema whose foreign emit policy permits
/// copying.
struct PayloadOptions {
  std::optional<SchemaIdentity> schema;  ///< Complete semantic identity, never a compact id alone.
  std::optional<std::uint32_t> max_inline_bytes;  ///< Positive logical-byte cap for inline policy.
  std::optional<std::uint64_t> max_tensor_byte_span;  ///< Positive embedded-Tensor staging bound.
  InterfaceTensorEmitPolicy interface_tensor_emit_policy{
      InterfaceTensorEmitPolicy::kRequireExportable};  ///< Omitted baseline policy is fail-closed.
};

/// @brief Preserve producer payload options without retaining caller views.
struct PayloadOptionsRecord {
  std::optional<SchemaIdentityRecord> schema;         ///< Owned exact schema override.
  std::optional<std::uint32_t> max_inline_bytes;      ///< Positive admitted inline cap.
  std::optional<std::uint64_t> max_tensor_byte_span;  ///< Positive embedded-Tensor staging bound.
  InterfaceTensorEmitPolicy interface_tensor_emit_policy{
      InterfaceTensorEmitPolicy::kRequireExportable};  ///< Closed interface emission policy.
};

}  // namespace holoscan
