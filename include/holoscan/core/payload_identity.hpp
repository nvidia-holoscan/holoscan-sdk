/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <holoscan/channel/pod_type.hpp>
#include <holoscan/core/payload_kind.hpp>
#include <holoscan/core/schema_identity.hpp>

namespace holoscan {

/// @brief Classify the exact stable representation of an admitted scalar port.
enum class ScalarIdentityKind : std::uint8_t {
  kBoolean = 0,     ///< C++ \c bool with logical false/true semantics.
  kUInt8 = 1,       ///< Exact unsigned 8-bit integer representation.
  kInt8 = 2,        ///< Exact signed 8-bit integer representation.
  kUInt16 = 3,      ///< Exact unsigned 16-bit integer representation.
  kInt16 = 4,       ///< Exact signed 16-bit integer representation.
  kUInt32 = 5,      ///< Exact unsigned 32-bit integer representation.
  kInt32 = 6,       ///< Exact signed 32-bit integer representation.
  kUInt64 = 7,      ///< Exact unsigned 64-bit integer representation.
  kInt64 = 8,       ///< Exact signed 64-bit integer representation.
  kFloat32 = 9,     ///< IEC 60559 binary32 representation.
  kFloat64 = 10,    ///< IEC 60559 binary64 representation.
  kLocalEnum = 11,  ///< Process-local enum, admitted only on local typed Graph edges.
};

/// @brief Preserve pointer-free scalar classification for Graph inspection.
struct ScalarIdentityRecord {
  ScalarIdentityKind kind{ScalarIdentityKind::kBoolean};  ///< Exact closed representation.
};

/// @brief Preserve a complete owned schema identity in cold Graph metadata.
struct SchemaIdentityRecord {
  std::string canonical_name;                 ///< Owned exact semantic name.
  std::uint32_t version{};                    ///< Nonzero exact semantic version.
  std::array<std::byte, 32> schema_sha256{};  ///< Complete canonical schema digest.
  SchemaId id{};  ///< Derived compact reference that is never sufficient as identity alone.
};

/// @brief Preserve one owned raw-POD field declaration before codec lowering.
struct PodFieldIdentityRecord {
  std::string name;                                  ///< Owned exact field name.
  std::uint32_t byte_offset{};                       ///< Actual object byte offset.
  PodScalarKind scalar_kind{PodScalarKind::kUInt8};  ///< Closed fixed-width scalar kind.
  std::uint32_t element_count{};                     ///< One or a positive fixed-array length.
};

/// @brief Preserve complete pointer-free raw-POD authoring identity.
///
/// EA1-006 requires ordered fields to cover every object byte exactly once,
/// with no gap, overlap, or tail padding. Within one Graph, every use of the
/// same canonical name and version must carry this exact complete layout.
/// Canonical layout digests and the fieldwise transport codec are deferred to
/// EA1-013.
struct PodIdentityRecord {
  std::string canonical_name;        ///< Owned exact type name.
  std::uint32_t version{};           ///< Nonzero exact type version.
  std::uint32_t object_size{};       ///< Host object size validated during authoring.
  std::uint32_t object_alignment{};  ///< Host object alignment validated during authoring.
  std::vector<PodFieldIdentityRecord> fields;  ///< Ordered copied layout fields.
};

/// @brief Classify one pointer-free semantic payload identity record.
enum class PayloadIdentityKind : std::uint8_t {
  kScalar = 0,  ///< Built-in arithmetic or local-only enum classification.
  kTensor = 1,  ///< The canonical Holoscan \c Tensor payload.
  kSchema = 2,  ///< Generated payload with complete schema identity.
  kPod = 3,     ///< Raw POD payload with explicit type and layout specialization.
};

/// @brief Carry pointer-free payload identity copied from Operator setup.
///
/// Default construction produces an unvalidated placeholder. A validated scalar, schema, or POD
/// identity populates exactly its matching optional record, while a validated Tensor identity
/// populates none. No validated identity populates more than one auxiliary record. The record
/// contains no RTTI token, compiler type name, view, or application pointer and is therefore safe
/// to retain in an immutable Graph snapshot.
struct PayloadIdentityRecord {
  PayloadIdentityKind kind{PayloadIdentityKind::kScalar};  ///< Active identity family.
  PayloadKind payload_kind{PayloadKind::kInlineTyped};     ///< Declared representation family.
  std::optional<ScalarIdentityRecord> scalar;  ///< Scalar classification when applicable.
  std::optional<SchemaIdentityRecord> schema;  ///< Owned generated schema identity.
  std::optional<PodIdentityRecord> pod;        ///< Owned raw-POD identity and structural layout.
};

}  // namespace holoscan
