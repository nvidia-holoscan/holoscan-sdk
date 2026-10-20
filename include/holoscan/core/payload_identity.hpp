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
  /// Native object size for \ref ScalarIdentityKind::kLocalEnum; zero for portable scalars.
  std::uint8_t native_object_size{};
};

/// @brief Preserve a complete owned schema identity in cold Graph metadata.
struct SchemaIdentityRecord {
  std::string canonical_name;                  ///< Owned stable compatibility-family name.
  std::uint32_t compatibility_epoch{};         ///< Positive intentional wire-break epoch.
  SchemaId family_id{};                        ///< Stable compact compatibility-family identity.
  FlatBufferFileIdentifier file_identifier{};  ///< Exact independent in-buffer type check.
};

/// @brief Deep-copy a borrowed schema identity into cold Graph/catalog storage.
[[nodiscard]] inline SchemaIdentityRecord make_schema_identity_record(
    const SchemaIdentity& identity) {
  return SchemaIdentityRecord{
      .canonical_name = std::string(identity.canonical_name),
      .compatibility_epoch = identity.compatibility_epoch,
      .family_id = identity.family_id,
      .file_identifier = identity.file_identifier,
  };
}

/// @brief Borrow a complete identity view from an owned identity record.
/// @warning The returned name view remains valid only while the record's string storage is stable.
[[nodiscard]] inline SchemaIdentity schema_identity_view(
    const SchemaIdentityRecord& identity) noexcept {
  return SchemaIdentity{
      .canonical_name = identity.canonical_name,
      .compatibility_epoch = identity.compatibility_epoch,
      .family_id = identity.family_id,
      .file_identifier = identity.file_identifier,
  };
}

/// @brief Compare two complete owned schema identities exactly.
[[nodiscard]] inline bool schema_identity_equal(const SchemaIdentityRecord& left,
                                                const SchemaIdentityRecord& right) noexcept {
  return left.canonical_name == right.canonical_name &&
         left.compatibility_epoch == right.compatibility_epoch &&
         left.family_id == right.family_id && left.file_identifier == right.file_identifier;
}

/// @brief Compare an owned schema identity with a borrowed identity exactly.
[[nodiscard]] inline bool schema_identity_equal(const SchemaIdentityRecord& left,
                                                const SchemaIdentity& right) noexcept {
  return schema_identity_equal(schema_identity_view(left), right);
}

/// @brief Compare a borrowed schema identity with an owned identity exactly.
[[nodiscard]] inline bool schema_identity_equal(const SchemaIdentity& left,
                                                const SchemaIdentityRecord& right) noexcept {
  return schema_identity_equal(right, left);
}

/// @brief Validate one complete owned schema identity through the borrowed core contract.
[[nodiscard]] inline bool valid_schema_identity(const SchemaIdentityRecord& identity) noexcept {
  return valid_schema_identity(schema_identity_view(identity));
}

/// @brief Preserve one owned raw-POD field declaration before codec lowering.
struct PodFieldIdentityRecord {
  std::string name;                                  ///< Owned exact field name.
  std::uint32_t byte_offset{};                       ///< Actual object byte offset.
  PodScalarKind scalar_kind{PodScalarKind::kUInt8};  ///< Closed fixed-width scalar kind.
  std::uint32_t element_count{};                     ///< One or a positive fixed-array length.
};

/// @brief Preserve complete pointer-free raw-POD authoring identity.
///
/// HSDK5-006 requires ordered fields to cover every object byte exactly once,
/// with no gap, overlap, or tail padding. Within one Graph, every use of the
/// same canonical name and version must carry this exact complete layout.
/// Canonical layout digests and the fieldwise transport codec are deferred to
/// HSDK5-013.
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
  /// Exact generated Holoscan carriage profile; zero for non-descriptor payloads.
  std::uint32_t transport_profile_tag{};
  std::optional<ScalarIdentityRecord> scalar;  ///< Scalar classification when applicable.
  std::optional<SchemaIdentityRecord> schema;  ///< Owned generated schema identity.
  std::optional<PodIdentityRecord> pod;        ///< Owned raw-POD identity and structural layout.
};

namespace detail {

/// @brief Return the fixed identity reserved for process-local Python object payloads.
[[nodiscard]] consteval SchemaIdentity py_object_payload_schema_identity() noexcept {
  return make_schema_identity(make_schema_family_identity("holoscan.PyObjectPayload"),
                              make_flatbuffer_file_identifier("PYOB"));
}

/// @brief Return whether an owned payload record is the fixed Python object identity.
[[nodiscard]] inline bool is_py_object_payload_identity(
    const PayloadIdentityRecord& identity) noexcept {
  return identity.kind == PayloadIdentityKind::kSchema && identity.schema.has_value() &&
         schema_identity_equal(*identity.schema, py_object_payload_schema_identity());
}

/// @brief Return whether a payload type carries the fixed process-local Python object identity.
template <typename T>
inline constexpr bool kPyObjectPayload = [] {
  if constexpr (HasSchemaIdentity<T>) {
    return schema_identity_equal(schema_identity<T>(), py_object_payload_schema_identity());
  }
  return false;
}();

}  // namespace detail

}  // namespace holoscan
