/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <dlpack/dlpack.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/memory_kind.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_schema_types.hpp>
#include <holoscan/memory/tensor_shape.hpp>

namespace holoscan {

/// Maximum number of nodes in one derived Tensor byte-span expression.
inline constexpr std::size_t kMaximumTensorByteSpanExpressionNodes = 32U;

/// Maximum depth of one derived Tensor byte-span expression.
inline constexpr std::size_t kMaximumTensorByteSpanExpressionDepth = 16U;

/// Maximum distinct Parameter and input dependencies in one byte-span expression.
inline constexpr std::size_t kMaximumTensorByteSpanExpressionDependencies = 8U;

/// Maximum bounded shape length admitted by a Phase-1 Parameter leaf.
inline constexpr std::size_t kMaximumTensorByteSpanParameterExtents = 8U;

/// Maximum UTF-8 bytes in one stable Parameter byte-span projector identity.
inline constexpr std::size_t kMaximumTensorByteSpanProjectionIdBytes = 255U;

/// @brief Classify one derived Tensor byte-span operation.
enum class TensorByteSpanOperation : std::uint8_t {
  kPositiveConstant = 0,          ///< Positive literal byte count or scalar operand.
  kParameterByteSpan = 1,         ///< Positive integral Parameter byte count.
  kParameterContiguousShape = 2,  ///< Bounded integral shape reduced with the output dtype.
  kInputMaxByteSpan = 3,          ///< Finite connected producer or imported-boundary capacity.
  kMultiply = 4,                  ///< Checked multiplication of two positive operands.
  kCeilingDivide = 5,             ///< Checked positive ceiling division.
  kMaximum = 6,                   ///< Maximum of two positive operands.
  kAlignUp = 7,                   ///< Checked alignment to a positive power of two.
  kParameterProjection = 8,       ///< Named C++ projection over finalized owning Parameters.
};

/// @brief Explain why one Tensor byte-span expression is structurally invalid.
enum class TensorByteSpanExpressionInvalidReason : std::uint8_t {
  kNone = 0,                 ///< The expression is structurally valid.
  kEmpty = 1,                ///< No expression node was authored.
  kNonPositiveConstant = 2,  ///< A literal operand was zero.
  kInvalidOperand = 3,       ///< An operand did not form a valid closed expression.
  kNodeLimitExceeded = 4,    ///< The expression exceeded its finite node capacity.
  kDepthLimitExceeded = 5,   ///< The expression exceeded its finite depth capacity.
  kAlignmentInvalid = 6,     ///< Alignment was zero, non-power-of-two, or too large.
  kProjectionInvalid = 7,    ///< Projector identity, callable, or dependencies are invalid.
};

/// @brief Select how a Tensor crossing an interface handles non-exportable storage.
enum class InterfaceTensorEmitPolicy : std::uint8_t {
  kRequireExportable = 0,  ///< Require an already exportable, layout-compatible Tensor.
  kCopyIfNeeded = 1,       ///< Preserve zero-copy when possible and otherwise use an admitted copy.
  kInlineSmallHostThenCopyIfNeeded = 2,  ///< Inline eligible small host data, otherwise copy.
};
/// @brief Classify admitted logical Tensor strides.
enum class TensorStridePolicy : std::uint8_t {
  kContiguousOnly = 0,  ///< Admit only canonical contiguous logical layouts.
  kBoundedStrided = 1,  ///< Admit noncontiguous layouts within explicit per-axis bounds.
};
/// @brief Bound one logical Tensor axis.
struct TensorAxisBounds {
  std::uint64_t max_extent{};            ///< Maximum logical extent; zero permits zero extent.
  std::uint64_t max_abs_stride_bytes{};  ///< Maximum absolute byte stride.

  friend constexpr bool operator==(const TensorAxisBounds&,
                                   const TensorAxisBounds&) noexcept = default;
};

/// @brief Preserve one declared projector dependency without process-local reader state.
struct TensorByteSpanProjectionDependencyRecord {
  ParamFieldId parameter_field{};                          ///< Stable terminal field identity.
  ParamSchemaHash parameter_full_id{};                     ///< Complete terminal field identity.
  std::vector<ParamSchemaHash> parameter_member_full_ids;  ///< Ordered typed path identities.
  std::string parameter_path;                              ///< Stable nested display path.

  friend bool operator==(const TensorByteSpanProjectionDependencyRecord&,
                         const TensorByteSpanProjectionDependencyRecord&) noexcept = default;
};

/// @brief Preserve one pointer-free node in a normalized Graph byte-span expression.
struct TensorByteSpanExpressionNodeRecord {
  TensorByteSpanOperation operation{TensorByteSpanOperation::kPositiveConstant};  ///< Node kind.
  std::uint32_t left{};                           ///< Left child node index for binary operations.
  std::uint32_t right{};                          ///< Right child node index for binary operations.
  std::uint64_t value{};                          ///< Positive constant or alignment operand.
  std::uint16_t depth{};                          ///< Canonical one-based expression depth.
  ParamFieldId parameter_field{};                 ///< Stable Parameter field identity, or zero.
  ParamSchemaHash parameter_full_id{};            ///< Complete Parameter field identity.
  std::string parameter_path;                     ///< Stable nested field path, or empty.
  std::string parameter_schema_id;                ///< Stable owning schema identity, or empty.
  ParamSchemaVersion parameter_schema_version{};  ///< Owning schema version, or zero.
  std::string input_name;                         ///< Owning-Operator input name, or empty.
  std::string projection_id;                      ///< Stable projector identity, or empty.
  /// Canonically ordered explicit Parameter dependencies for a projection leaf.
  std::vector<TensorByteSpanProjectionDependencyRecord> projection_dependencies;

  friend bool operator==(const TensorByteSpanExpressionNodeRecord&,
                         const TensorByteSpanExpressionNodeRecord&) noexcept = default;
};

/// @brief Own one normalized pointer-free Tensor byte-span expression in Graph state.
struct TensorByteSpanExpressionRecord {
  std::vector<TensorByteSpanExpressionNodeRecord> nodes;  ///< Canonical postorder nodes.

  friend bool operator==(const TensorByteSpanExpressionRecord&,
                         const TensorByteSpanExpressionRecord&) noexcept = default;
};

/// Maximum number of source values in one Parameter-derived Tensor memory-kind mapping.
inline constexpr std::size_t kMaximumTensorMemoryKindMappingCases = kParamAllowedValueCapacity;

/// @brief Classify the closed form of one Parameter-derived Tensor memory-kind expression.
enum class TensorMemoryKindExpressionForm : std::uint8_t {
  kDirect = 0,  ///< Read an exact `MemoryKind` Parameter value.
  kMapped = 1,  ///< Map one Boolean or schema-declared enum value through an owned table.
};

/// @brief Explain why one Tensor memory-kind expression is structurally invalid.
enum class TensorMemoryKindExpressionInvalidReason : std::uint8_t {
  kNone = 0,                  ///< The expression is structurally valid.
  kEmpty = 1,                 ///< No Parameter dependency was authored.
  kMappingEmpty = 2,          ///< A mapped expression supplied no cases.
  kMappingLimitExceeded = 3,  ///< The mapping exceeded its bounded public capacity.
  kMappingDuplicate = 4,      ///< Two cases supplied the same source value.
  kMappingTargetInvalid = 5,  ///< A case selected unknown or invalid memory.
  kUnexpectedMapping = 6,     ///< A direct expression unexpectedly supplied mapping cases.
};

/// @brief Preserve one canonical source-to-memory mapping row in normalized Graph state.
struct TensorMemoryKindMappingRecord {
  ParamScalarValue source;                       ///< Exact stable scalar source value.
  MemoryKind memory_kind{MemoryKind::kUnknown};  ///< Concrete mapped result.

  friend constexpr bool operator==(TensorMemoryKindMappingRecord,
                                   TensorMemoryKindMappingRecord) noexcept = default;
};

/// @brief Own one normalized pointer-free Tensor memory-kind expression in Graph state.
struct TensorMemoryKindExpressionRecord {
  TensorMemoryKindExpressionForm form{TensorMemoryKindExpressionForm::kDirect};  ///< Closed form.
  ParamFieldId parameter_field{};                          ///< Stable Parameter field identity.
  ParamSchemaHash parameter_full_id{};                     ///< Complete Parameter field identity.
  std::vector<ParamSchemaHash> parameter_member_full_ids;  ///< Ordered typed path identities.
  std::string parameter_path;                              ///< Canonical display path.
  std::string parameter_schema_id;                         ///< Stable owning schema identity.
  ParamSchemaVersion parameter_schema_version{};           ///< Owning schema version.
  std::vector<TensorMemoryKindMappingRecord> mapping;      ///< Canonical source-ordered mapping.

  friend bool operator==(const TensorMemoryKindExpressionRecord&,
                         const TensorMemoryKindExpressionRecord&) noexcept = default;
};

/// @brief Describe Tensor facts that can be compared with a concrete value.
///
/// Each absent member is unconstrained. A direction-specific declaration that is
/// present must contain at least one meaningful fact across its representation,
/// bounds, storage, transfer, or conversion members; omit the spec entirely
/// to express an unconstrained pass-through port.
struct TensorRepresentation {
  std::optional<MemoryKind> memory_kind;  ///< Exact physical storage class when constrained.
  std::optional<DLDataType> dtype;        ///< Exact DLPack element type when constrained.
  std::optional<std::uint32_t> rank;      ///< Exact logical rank when constrained; zero is scalar.
  /// Admitted stride family when constrained; two connected ports with explicit policies require
  /// an exact policy match rather than treating contiguous as a subset of bounded-strided.
  std::optional<TensorStridePolicy> strides;
};

/// @brief Borrow maximum absolute byte-stride bounds for one authoring call.
using TensorMaxAbsByteStridesView = std::span<const std::uint64_t>;

/// @brief Carry finite logical and resource bounds for a Tensor contract.
///
/// A present maximum is positive. An exact representation rank supplies the
/// effective maximum rank when max_rank is absent; logical rank zero uses one
/// physical metadata slot. Bounded-strided interoperability requires one axis
/// record per effective maximum rank and a positive stride bound for each
/// record. Borrowed shape and stride views are copied during contract
/// normalization before the authoring call returns.
struct TensorBounds {
  std::optional<std::uint32_t> max_rank;       ///< Positive dynamic-rank upper bound.
  std::optional<std::uint64_t> max_byte_span;  ///< Positive logical byte-span upper bound.
  std::optional<TensorShapeView> max_shape;    ///< Optional call-lifetime per-axis extents.
  /// Optional call-lifetime maximum absolute byte strides, one per maximum-shape axis.
  std::optional<TensorMaxAbsByteStridesView> max_abs_byte_strides;
  /// Optional call-lifetime pre-paired axis bounds for direct normalized authoring.
  std::span<const TensorAxisBounds> axes;
};

/// @brief Refine maximum-shape Tensor-bounds authoring.
struct TensorBoundsOptions {
  std::optional<std::uint32_t> max_rank;       ///< Optional wider dynamic-rank upper bound.
  std::optional<std::uint64_t> max_byte_span;  ///< Optional explicit authoritative byte cap.
  /// Optional positive maximum absolute byte strides, one per maximum-shape axis.
  std::optional<TensorMaxAbsByteStridesView> max_abs_byte_strides;
};

/// @brief Create a maximum-byte-span-only Tensor bound.
/// @param max_byte_span Positive maximum addressed byte span.
/// @return Bounds that do not constrain shape or per-axis strides.
[[nodiscard]] constexpr TensorBounds tensor_bounds(std::uint64_t max_byte_span) noexcept {
  return {.max_byte_span = max_byte_span};
}

/// @brief Create maximum-shape Tensor bounds with optional stride and byte-cap refinements.
/// @param max_shape Non-owning maximum logical extents copied by the receiving authoring API.
/// @param options Optional maximum rank, byte span, and absolute byte-stride bounds.
/// @return Call-lifetime view whose backing arrays remain owned by the caller.
/// @note Omitting TensorBoundsOptions::max_abs_byte_strides requests C-contiguous bounds and sets
/// the enclosing representation to kContiguousOnly. That explicit policy must match another
/// connected port's explicit stride policy. The exact dtype supplies the element size during
/// normalization.
[[nodiscard]] constexpr TensorBounds tensor_bounds(TensorShapeView max_shape,
                                                   TensorBoundsOptions options = {}) noexcept {
  return {
      .max_rank = options.max_rank,
      .max_byte_span = options.max_byte_span,
      .max_shape = max_shape,
      .max_abs_byte_strides = options.max_abs_byte_strides,
  };
}

/// @brief Carry Tensor copy and inline materialization policy.
///
/// The max_inline_bytes member is present and positive exactly for the
/// kInlineSmallHostThenCopyIfNeeded policy. The max_staging_byte_span member is
/// valid only for a copy-capable policy and is a positive optional override;
/// when copying is admitted and it is absent, normalization may use the
/// contract's maximum byte span. This policy does not select memory conversion
/// or a CUDA device.
struct TensorTransferPolicy {
  InterfaceTensorEmitPolicy foreign_emit_policy{
      InterfaceTensorEmitPolicy::kRequireExportable};  ///< Foreign storage handling.
  std::optional<std::uint64_t> max_inline_bytes;       ///< Positive inline byte cap.
  std::optional<std::uint64_t> max_staging_byte_span;  ///< Positive staging-bound override.
};

/// @brief Declare consumer-side Tensor requirements.
struct TensorInputSpec {
  TensorRepresentation representation;  ///< Facts required from every delivered Tensor.
  std::optional<TensorBounds> bounds;   ///< Optional finite logical constraints.
  std::optional<MemoryConversionPolicy> memory_conversion;  ///< Optional conversion request.
};

/// @brief Select whether an output passes through storage or requests a Runtime pool.
enum class TensorOutputStorage : std::uint8_t {
  /// Validate declared promises without requesting new backing storage. When used as a Graph
  /// override, this does not remove a Runtime pool already declared during Operator setup.
  kPassThrough = 0,
  kRuntimePool = 1,  ///< Provision finite Runtime-owned backing storage from the contract.
};

/// @brief Classify one direction-specific normalized Tensor contract.
enum class TensorContractRole : std::uint8_t {
  kInput = 0,             ///< Consumer requirements.
  kOutput = 1,            ///< Producer promises and storage intent.
  kExternalBoundary = 2,  ///< Independently comparable external interoperability contract.
};

/// @brief Classify how one effective normalized Tensor fact was obtained.
enum class TensorContractValueOrigin : std::uint8_t {
  kAbsent = 0,                     ///< No effective value exists.
  kAuthored = 1,                   ///< The authoring contract supplied the fact.
  kDerivedExactRank = 2,           ///< Metadata capacity was derived from exact rank.
  kDerivedMaxByteSpan = 3,         ///< Staging capacity was derived from the overall byte bound.
  kDefaultedBoundary = 4,          ///< Boundary normalization supplied contiguous layout.
  kDerivedShapeBounds = 5,         ///< Maximum shape/stride bounds supplied the effective fact.
  kCompilerDerivedByteSpan = 6,    ///< Compiler resolved a declared byte-span expression.
  kCompilerDerivedMemoryKind = 7,  ///< Compiler resolved a declared memory-kind expression.
};

/// @brief Preserve cold-path provenance for effective normalized Tensor facts.
struct TensorContractOrigins {
  TensorContractValueOrigin memory_kind{TensorContractValueOrigin::kAbsent};  ///< Storage fact.
  TensorContractValueOrigin dtype{TensorContractValueOrigin::kAbsent};        ///< Element type.
  TensorContractValueOrigin rank{TensorContractValueOrigin::kAbsent};         ///< Exact rank.
  TensorContractValueOrigin strides{TensorContractValueOrigin::kAbsent};      ///< Stride family.
  TensorContractValueOrigin max_rank{TensorContractValueOrigin::kAbsent};     ///< Rank capacity.
  TensorContractValueOrigin max_byte_span{
      TensorContractValueOrigin::kAbsent};  ///< Overall byte capacity.
  TensorContractValueOrigin max_staging_byte_span{
      TensorContractValueOrigin::kAbsent};  ///< Copy-staging capacity.
};

/// @brief Declare producer-side Tensor promises and optional storage intent.
///
/// Runtime-owned storage requires a concrete non-unknown memory kind, a
/// positive maximum byte span, and either a positive maximum rank or an exact
/// rank from which metadata capacity can be derived.
struct TensorOutputSpec {
  TensorRepresentation representation;  ///< Facts promised for every emitted Tensor.
  std::optional<TensorBounds> bounds;   ///< Optional finite logical/resource bounds.
  TensorOutputStorage storage{TensorOutputStorage::kPassThrough};  ///< Backing ownership intent.
  std::optional<TensorTransferPolicy> transfer;  ///< Optional interface materialization policy.
};

/// @brief Declare a finite Tensor interoperability contract at an external boundary.
///
/// A valid boundary has a concrete non-unknown memory kind, a positive maximum
/// byte span, an effective positive maximum rank, and a concrete stride policy.
/// Omitting the stride policy during authoring may normalize to contiguous only
/// as an external-boundary convenience; ordinary ports do not make that
/// inference.
struct TensorBoundaryContract {
  TensorRepresentation representation;           ///< Independently comparable representation facts.
  TensorBounds bounds;                           ///< Required finite boundary bounds.
  std::optional<TensorTransferPolicy> transfer;  ///< Optional materialization policy.
  std::optional<MemoryConversionPolicy> memory_conversion;  ///< Optional conversion request.
};

/// @brief Hold one deep-owned normalized Tensor role contract on the cold path.
struct TensorContractRecord {
  TensorContractRole role{TensorContractRole::kInput};  ///< Direction-specific contract role.
  TensorRepresentation representation;                  ///< Validated representation facts.
  std::optional<std::uint32_t> max_rank;                ///< Effective positive metadata capacity.
  std::optional<std::uint64_t> max_byte_span;           ///< Effective positive byte-span bound.
  std::vector<TensorAxisBounds> axes;                   ///< Owned maximum-shape/stride evidence.
  std::optional<TensorTransferPolicy> transfer;  ///< Validated and defaulted transfer policy.
  std::optional<MemoryConversionPolicy> memory_conversion;  ///< Consumer-side conversion request.
  TensorOutputStorage storage{TensorOutputStorage::kPassThrough};  ///< Output storage intent.
  TensorContractOrigins origins;  ///< Diagnostic authored-versus-derived provenance.
};

}  // namespace holoscan
