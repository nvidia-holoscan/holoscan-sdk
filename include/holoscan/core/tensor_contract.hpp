/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <concepts>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

namespace holoscan {

class TensorOutputSpecBuilder;
class TensorFieldOutputSpecBuilder;

namespace detail {
class OperatorSpecCompilerAccess;
}  // namespace detail

/// Maximum number of nodes in one derived Tensor byte-span expression.
inline constexpr std::size_t kMaximumTensorByteSpanExpressionNodes = 32U;

/// Maximum depth of one derived Tensor byte-span expression.
inline constexpr std::size_t kMaximumTensorByteSpanExpressionDepth = 16U;

/// Maximum distinct Parameter and input dependencies in one byte-span expression.
inline constexpr std::size_t kMaximumTensorByteSpanExpressionDependencies = 8U;

/// Maximum bounded shape length admitted by a Phase-1 Parameter leaf.
inline constexpr std::size_t kMaximumTensorByteSpanParameterExtents = 8U;

/// @brief Classify one closed derived Tensor byte-span operation.
enum class TensorByteSpanOperation : std::uint8_t {
  kPositiveConstant = 0,          ///< Positive literal byte count or scalar operand.
  kParameterByteSpan = 1,         ///< Positive integral Parameter byte count.
  kParameterContiguousShape = 2,  ///< Bounded integral shape reduced with the output dtype.
  kInputMaxByteSpan = 3,          ///< Finite connected producer or imported-boundary capacity.
  kMultiply = 4,                  ///< Checked multiplication of two positive operands.
  kCeilingDivide = 5,             ///< Checked positive ceiling division.
  kMaximum = 6,                   ///< Maximum of two positive operands.
  kAlignUp = 7,                   ///< Checked alignment to a positive power of two.
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
};

namespace detail {

enum class TensorByteSpanParameterReadStatus : std::uint8_t {
  kSuccess,
  kTypeInvalid,
  kValueInvalid,
};

struct TensorByteSpanParameterValue {
  std::array<std::uint64_t, kMaximumTensorByteSpanParameterExtents> values{};
  std::size_t count{};
};

/// Process-local Parameter dependency consumed only while replaying an Operator factory.
struct TensorByteSpanParameterDependency {
  const void* root_type_token{};
  const ParamSchemaDescriptor* (*root_schema)() noexcept {};
  const ParamFieldDescriptor* (*final_field)() noexcept {};
  std::string (*canonical_path)() = nullptr;
  TensorByteSpanParameterReadStatus (*read_initial)(const void*,
                                                    TensorByteSpanParameterValue&) noexcept {};
  bool shape{};

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return root_type_token != nullptr && root_schema != nullptr && final_field != nullptr &&
           canonical_path != nullptr && read_initial != nullptr;
  }
};

struct TensorByteSpanAuthoringNode {
  TensorByteSpanOperation operation{TensorByteSpanOperation::kPositiveConstant};
  std::uint32_t left{};
  std::uint32_t right{};
  std::uint64_t value{};
  std::uint16_t depth{1U};
  TensorByteSpanParameterDependency parameter;
  const void* input_declaration{};
};

template <typename Value>
[[nodiscard]] TensorByteSpanParameterReadStatus read_positive_parameter_value(
    const Value& value, TensorByteSpanParameterValue& result) noexcept {
  using Exact = std::remove_cv_t<Value>;
  if constexpr (std::is_integral_v<Exact> && !std::same_as<Exact, bool>) {
    if constexpr (std::is_signed_v<Exact>) {
      if (value <= 0) {
        return TensorByteSpanParameterReadStatus::kValueInvalid;
      }
    } else if (value == 0U) {
      return TensorByteSpanParameterReadStatus::kValueInvalid;
    }
    result.values[0] = static_cast<std::uint64_t>(value);
    result.count = 1U;
    return TensorByteSpanParameterReadStatus::kSuccess;
  } else {
    return TensorByteSpanParameterReadStatus::kTypeInvalid;
  }
}

template <typename Value>
[[nodiscard]] TensorByteSpanParameterReadStatus read_positive_parameter_shape(
    const Value& value, TensorByteSpanParameterValue& result) noexcept {
  using Exact = std::remove_cv_t<Value>;
  if constexpr (IsStdVector<Exact>::value) {
    using Element = std::remove_cv_t<typename StdVectorElement<Exact>::Type>;
    if constexpr (std::is_integral_v<Element> && !std::same_as<Element, bool>) {
      if (value.empty() || value.size() > result.values.size()) {
        return TensorByteSpanParameterReadStatus::kValueInvalid;
      }
      for (std::size_t index = 0U; index < value.size(); ++index) {
        if constexpr (std::is_signed_v<Element>) {
          if (value[index] <= 0) {
            return TensorByteSpanParameterReadStatus::kValueInvalid;
          }
        } else if (value[index] == 0U) {
          return TensorByteSpanParameterReadStatus::kValueInvalid;
        }
        result.values[index] = static_cast<std::uint64_t>(value[index]);
      }
      result.count = value.size();
      return TensorByteSpanParameterReadStatus::kSuccess;
    }
  }
  return TensorByteSpanParameterReadStatus::kTypeInvalid;
}

template <auto First, auto... Rest>
struct TensorByteSpanParameterDependencyModel {
  using Root = std::remove_cv_t<typename MemberPointerTraits<decltype(First)>::Owner>;
  using Value = std::remove_cv_t<typename ParamPathValue<First, Rest...>::Type>;

  [[nodiscard]] static const ParamSchemaDescriptor* root_schema() noexcept {
    return param_schema_descriptor<Root>();
  }

  template <bool Shape>
  [[nodiscard]] static TensorByteSpanParameterReadStatus read_initial(
      const void* aggregate, TensorByteSpanParameterValue& result) noexcept {
    if (aggregate == nullptr) {
      return TensorByteSpanParameterReadStatus::kValueInvalid;
    }
    const auto& value = param_path_value<First, Rest...>(*static_cast<const Root*>(aggregate));
    if constexpr (Shape) {
      return read_positive_parameter_shape(value, result);
    } else {
      return read_positive_parameter_value(value, result);
    }
  }

  template <bool Shape>
  [[nodiscard]] static TensorByteSpanParameterDependency make() noexcept {
    return TensorByteSpanParameterDependency{
        .root_type_token = &kParameterTypeToken<Root>,
        .root_schema = &root_schema,
        .final_field = &final_param_field<First, Rest...>,
        .canonical_path = &canonical_param_path<First, Rest...>,
        .read_initial = &read_initial<Shape>,
        .shape = Shape,
    };
  }
};

}  // namespace detail

/// @brief Build one finite Tensor slot byte-capacity expression for compile-time resolution.
///
/// Expressions are immutable authoring values. Compilation resolves them from generation-zero
/// Parameters and finite connected-input capacity, then publishes an ordinary concrete pool plan.
/// Changing a recompile-required dependency requires compiling a new execution plan; it does not
/// require rebuilding or relinking the application binary. Runtime never evaluates this value.
class HOLOSCAN_CORE_EXPORT TensorByteSpanExpression {
 public:
  TensorByteSpanExpression() = default;

  /// @brief Return the first structural authoring failure retained by this expression.
  /// @return `TensorByteSpanExpressionInvalidReason::kNone` for a valid expression.
  [[nodiscard]] TensorByteSpanExpressionInvalidReason invalid_reason() const noexcept {
    return invalid_reason_;
  }

  /// @brief Multiply this expression by another positive expression.
  /// @param factor Positive finite expression.
  /// @return Combined checked expression, or a structurally invalid value when limits are exceeded.
  [[nodiscard]] TensorByteSpanExpression multiplied_by(
      const TensorByteSpanExpression& factor) const;

  /// @brief Divide this expression by another positive expression with ceiling rounding.
  /// @param divisor Positive finite expression.
  /// @return Combined checked expression, or a structurally invalid value when limits are exceeded.
  [[nodiscard]] TensorByteSpanExpression ceiling_divided_by(
      const TensorByteSpanExpression& divisor) const;

  /// @brief Select the maximum of this expression and another positive expression.
  /// @param other Positive finite expression.
  /// @return Combined checked expression, or a structurally invalid value when limits are exceeded.
  [[nodiscard]] TensorByteSpanExpression maximum_with(const TensorByteSpanExpression& other) const;

  /// @brief Align this expression upward to a positive power-of-two byte boundary.
  /// @param alignment Positive power-of-two alignment no larger than 2^30 bytes.
  /// @return Combined checked expression, or a structurally invalid value for bad alignment.
  [[nodiscard]] TensorByteSpanExpression aligned_to(std::uint64_t alignment) const;

 private:
  explicit TensorByteSpanExpression(detail::TensorByteSpanAuthoringNode node);
  [[nodiscard]] TensorByteSpanExpression combine(const TensorByteSpanExpression& right,
                                                 TensorByteSpanOperation operation) const;

  std::vector<detail::TensorByteSpanAuthoringNode> nodes_;
  TensorByteSpanExpressionInvalidReason invalid_reason_{
      TensorByteSpanExpressionInvalidReason::kEmpty};

  friend class detail::OperatorSpecCompilerAccess;
  friend class TensorFieldOutputSpecBuilder;
  friend class TensorOutputSpecBuilder;
  friend TensorByteSpanExpression tensor_byte_span(std::uint64_t);
  template <auto... MemberPath>
  friend TensorByteSpanExpression parameter_byte_span(ParamRef<MemberPath...>);
  template <auto... MemberPath>
  friend TensorByteSpanExpression parameter_contiguous_byte_span(ParamRef<MemberPath...>);
  friend TensorByteSpanExpression detail_make_input_max_byte_span(const void*);
};

/// @internal Build an input dependency after the complete port type is available.
[[nodiscard]] HOLOSCAN_CORE_EXPORT TensorByteSpanExpression
detail_make_input_max_byte_span(const void* input);

/// @brief Create one positive literal Tensor byte-span expression operand.
/// @param value Positive byte count.
/// @return Literal expression; zero is retained as invalid authoring evidence.
[[nodiscard]] HOLOSCAN_CORE_EXPORT TensorByteSpanExpression tensor_byte_span(std::uint64_t value);

/// @brief Derive Tensor slot bytes from one positive integral Parameter field.
/// @tparam MemberPath Declared owning-Operator Parameter field path.
/// @param reference Empty compile-time dependency token.
/// @return Process-local authoring expression resolved during compilation.
template <auto... MemberPath>
[[nodiscard]] TensorByteSpanExpression parameter_byte_span(ParamRef<MemberPath...> reference) {
  (void)reference;
  using Model = detail::TensorByteSpanParameterDependencyModel<MemberPath...>;
  return TensorByteSpanExpression{detail::TensorByteSpanAuthoringNode{
      .operation = TensorByteSpanOperation::kParameterByteSpan,
      .parameter = Model::template make<false>(),
  }};
}

/// @brief Derive contiguous Tensor slot bytes from one bounded integral shape Parameter.
/// @tparam MemberPath Declared owning-Operator Parameter field path.
/// @param reference Empty compile-time dependency token.
/// @return Process-local authoring expression resolved with the output dtype and exact rank.
template <auto... MemberPath>
[[nodiscard]] TensorByteSpanExpression parameter_contiguous_byte_span(
    ParamRef<MemberPath...> reference) {
  (void)reference;
  using Model = detail::TensorByteSpanParameterDependencyModel<MemberPath...>;
  return TensorByteSpanExpression{detail::TensorByteSpanAuthoringNode{
      .operation = TensorByteSpanOperation::kParameterContiguousShape,
      .parameter = Model::template make<true>(),
  }};
}

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

  friend bool operator==(const TensorByteSpanExpressionNodeRecord&,
                         const TensorByteSpanExpressionNodeRecord&) noexcept = default;
};

/// @brief Own one normalized pointer-free Tensor byte-span expression in Graph state.
struct TensorByteSpanExpressionRecord {
  std::vector<TensorByteSpanExpressionNodeRecord> nodes;  ///< Canonical postorder nodes.

  friend bool operator==(const TensorByteSpanExpressionRecord&,
                         const TensorByteSpanExpressionRecord&) noexcept = default;
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
  kAbsent = 0,                   ///< No effective value exists.
  kAuthored = 1,                 ///< The authoring contract supplied the fact.
  kDerivedExactRank = 2,         ///< Metadata capacity was derived from exact rank.
  kDerivedMaxByteSpan = 3,       ///< Staging capacity was derived from the overall byte bound.
  kDefaultedBoundary = 4,        ///< Boundary normalization supplied contiguous layout.
  kDerivedShapeBounds = 5,       ///< Maximum shape/stride bounds supplied the effective fact.
  kCompilerDerivedByteSpan = 6,  ///< Compiler resolved a declared byte-span expression.
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

namespace detail {

/// @brief Normalize and deep-copy one consumer Tensor spec.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<TensorContractRecord, Error>
normalize_tensor_input_spec(const TensorInputSpec& spec);

/// @brief Normalize and deep-copy one producer Tensor spec.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<TensorContractRecord, Error>
normalize_tensor_output_spec(const TensorOutputSpec& spec);

/// @brief Normalize a runtime-pool output whose positive byte span is compile-derived.
/// @param spec Producer facts with runtime-pool storage and no literal maximum byte span.
/// @return Owned contract retaining unresolved byte/staging capacity for compiler resolution.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<TensorContractRecord, Error>
normalize_tensor_output_spec_with_derived_byte_span(const TensorOutputSpec& spec);

/// @brief Normalize a partial pass-through refinement of a compile-derived output contract.
/// @param spec Producer facts with pass-through storage and no literal maximum byte span.
/// @return Owned partial contract retaining unresolved byte/staging capacity for later merging.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<TensorContractRecord, Error>
normalize_tensor_output_spec_refinement_with_derived_byte_span(const TensorOutputSpec& spec);

/// @brief Normalize and deep-copy one external-boundary Tensor contract.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<TensorContractRecord, Error>
normalize_tensor_boundary_contract(const TensorBoundaryContract& contract);

/// @brief Project normalized representation facts into the legacy compiler/runtime view.
[[nodiscard]] HOLOSCAN_CORE_EXPORT TensorRepresentation
tensor_representation(const TensorContractRecord& contract) noexcept;

/// @brief Compare two owned normalized Tensor contracts.
[[nodiscard]] HOLOSCAN_CORE_EXPORT bool tensor_contract_equal(
    const TensorContractRecord& left, const TensorContractRecord& right) noexcept;

}  // namespace detail

}  // namespace holoscan
