/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <concepts>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <utility>
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

template <auto Member>
[[nodiscard]] bool append_param_member_full_id(std::vector<ParamSchemaHash>& result) {
  const ParamFieldDescriptor* field = declared_member_descriptor<Member>();
  if (field == nullptr) {
    return false;
  }
  result.push_back(field->full_id);
  return true;
}

template <auto First, auto... Rest>
[[nodiscard]] std::vector<ParamSchemaHash> param_member_full_ids() {
  std::vector<ParamSchemaHash> result;
  result.reserve(1U + sizeof...(Rest));
  if (!append_param_member_full_id<First>(result) ||
      !(append_param_member_full_id<Rest>(result) && ...)) {
    result.clear();
  }
  return result;
}

/// Process-local Parameter dependency consumed only while replaying an Operator factory.
struct TensorByteSpanParameterDependency {
  const void* root_type_token{};
  const ParamSchemaDescriptor* (*root_schema)() noexcept {};
  const ParamFieldDescriptor* (*final_field)() noexcept {};
  std::vector<ParamSchemaHash> (*member_full_ids)() = nullptr;
  std::string (*canonical_path)() = nullptr;
  TensorByteSpanParameterReadStatus (*read_initial)(const void*,
                                                    TensorByteSpanParameterValue&) noexcept {};
  bool shape{};

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return root_type_token != nullptr && root_schema != nullptr && final_field != nullptr &&
           member_full_ids != nullptr && canonical_path != nullptr && read_initial != nullptr;
  }
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
    const Value* value = param_path_value<First, Rest...>(*static_cast<const Root*>(aggregate));
    if (value == nullptr) {
      return TensorByteSpanParameterReadStatus::kValueInvalid;
    }
    if constexpr (Shape) {
      return read_positive_parameter_shape(*value, result);
    } else {
      return read_positive_parameter_value(*value, result);
    }
  }

  template <bool Shape>
  [[nodiscard]] static TensorByteSpanParameterDependency make() noexcept {
    return TensorByteSpanParameterDependency{
        .root_type_token = &kParameterTypeToken<Root>,
        .root_schema = &root_schema,
        .final_field = &final_param_field<First, Rest...>,
        .member_full_ids = &param_member_full_ids<First, Rest...>,
        .canonical_path = &canonical_param_path<First, Rest...>,
        .read_initial = &read_initial<Shape>,
        .shape = Shape,
    };
  }
};

/// Process-local stable-field identity for one declared projector dependency.
struct TensorByteSpanProjectionDependency {
  const void* root_type_token{};
  const ParamSchemaDescriptor* (*root_schema)() noexcept {};
  const ParamFieldDescriptor* (*root_field)() noexcept {};
  const ParamFieldDescriptor* (*final_field)() noexcept {};
  std::vector<ParamSchemaHash> (*member_full_ids)() = nullptr;
  bool (*has_sensitive_member)() noexcept {};
  std::string (*canonical_path)() = nullptr;

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return root_type_token != nullptr && root_schema != nullptr && root_field != nullptr &&
           final_field != nullptr && member_full_ids != nullptr &&
           has_sensitive_member != nullptr && canonical_path != nullptr;
  }
};

template <auto First, auto... Rest>
struct TensorByteSpanProjectionDependencyModel {
  using Root = std::remove_cv_t<typename MemberPointerTraits<decltype(First)>::Owner>;

  [[nodiscard]] static const ParamSchemaDescriptor* root_schema() noexcept {
    return param_schema_descriptor<Root>();
  }

  [[nodiscard]] static TensorByteSpanProjectionDependency make() noexcept {
    return TensorByteSpanProjectionDependency{
        .root_type_token = &kParameterTypeToken<Root>,
        .root_schema = &root_schema,
        .root_field = &declared_member_descriptor<First>,
        .final_field = &final_param_field<First, Rest...>,
        .member_full_ids = &param_member_full_ids<First, Rest...>,
        .has_sensitive_member = &param_path_has_sensitive_member<First, Rest...>,
        .canonical_path = &canonical_param_path<First, Rest...>,
    };
  }
};

template <typename T>
struct TensorByteSpanProjectionReferenceTraits {
  static constexpr bool kValid = false;
  using Root = void;
};

template <auto... MemberPath>
struct TensorByteSpanProjectionReferenceTraits<ParamRef<MemberPath...>> {
  static constexpr bool kValid = true;
  using Model = TensorByteSpanProjectionDependencyModel<MemberPath...>;
  using Root = typename Model::Root;

  [[nodiscard]] static TensorByteSpanProjectionDependency make(ParamRef<MemberPath...>) noexcept {
    return Model::make();
  }
};

template <typename T>
struct TensorByteSpanProjectorTraits {
  static constexpr bool kFunction = false;
  static constexpr bool kNoexcept = false;
  static constexpr bool kExactResult = false;
  using Params = void;
};

template <typename Result, typename ParamsT>
struct TensorByteSpanProjectorTraits<Result (*)(const ParamsT&)> {
  static constexpr bool kFunction = !std::is_volatile_v<ParamsT>;
  static constexpr bool kNoexcept = false;
  static constexpr bool kExactResult = std::same_as<Result, expected<std::uint64_t, Error>>;
  using Params = std::remove_cv_t<ParamsT>;
};

template <typename Result, typename ParamsT>
struct TensorByteSpanProjectorTraits<Result (*)(const ParamsT&) noexcept> {
  static constexpr bool kFunction = !std::is_volatile_v<ParamsT>;
  static constexpr bool kNoexcept = true;
  static constexpr bool kExactResult = std::same_as<Result, expected<std::uint64_t, Error>>;
  using Params = std::remove_cv_t<ParamsT>;
};

/// Process-local exact callable identity for one named byte-span projector.
struct TensorByteSpanProjectorIdentity {
  const std::type_info* (*type)() noexcept {};
  const void* (*value)() noexcept {};
  bool (*matches)(const std::type_info&, const void*) noexcept {};

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return type != nullptr && value != nullptr && matches != nullptr;
  }
};

[[nodiscard]] inline bool same_tensor_byte_span_projector(
    const TensorByteSpanProjectorIdentity& left,
    const TensorByteSpanProjectorIdentity& right) noexcept {
  if (!left || !right) {
    return false;
  }
  const std::type_info* const left_type = left.type();
  const std::type_info* const right_type = right.type();
  const void* const left_value = left.value();
  const void* const right_value = right.value();
  return left_type != nullptr && right_type != nullptr && left_value != nullptr &&
         right_value != nullptr && left.matches(*right_type, right_value) &&
         right.matches(*left_type, left_value);
}

/// Process-local executable state for one named byte-span projection leaf.
struct TensorByteSpanParameterProjection {
  const void* root_type_token{};
  const ParamSchemaDescriptor* (*root_schema)() noexcept {};
  expected<std::uint64_t, Error> (*evaluate)(const void*) noexcept {};
  TensorByteSpanProjectorIdentity implementation;
  std::string stable_id;
  std::vector<TensorByteSpanProjectionDependency> dependencies;

  [[nodiscard]] explicit operator bool() const noexcept {
    return root_type_token != nullptr && root_schema != nullptr && evaluate != nullptr &&
           implementation && !stable_id.empty() && !dependencies.empty();
  }
};

template <auto Projector, typename ParamsT>
struct TensorByteSpanProjectorModel {
  using ProjectorType = decltype(Projector);
  inline static constexpr ProjectorType kProjector = Projector;

  [[nodiscard]] static const ParamSchemaDescriptor* root_schema() noexcept {
    return param_schema_descriptor<ParamsT>();
  }

  [[nodiscard]] static expected<std::uint64_t, Error> evaluate(const void* aggregate) noexcept {
    if (aggregate == nullptr) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "PHYSICAL_TENSOR_BYTE_SPAN_DEPENDENCY_INVALID"});
    }
    return Projector(*static_cast<const ParamsT*>(aggregate));
  }

  [[nodiscard]] static const std::type_info* implementation_type() noexcept {
    return &typeid(ProjectorType);
  }

  [[nodiscard]] static const void* implementation_value() noexcept { return &kProjector; }

  [[nodiscard]] static bool implementation_matches(const std::type_info& type,
                                                   const void* value) noexcept {
    return value != nullptr && type == typeid(ProjectorType) &&
           *static_cast<const ProjectorType*>(value) == Projector;
  }

  [[nodiscard]] static TensorByteSpanProjectorIdentity implementation_identity() noexcept {
    return TensorByteSpanProjectorIdentity{
        .type = &implementation_type,
        .value = &implementation_value,
        .matches = &implementation_matches,
    };
  }
};

struct TensorByteSpanAuthoringNode {
  TensorByteSpanOperation operation{TensorByteSpanOperation::kPositiveConstant};
  std::uint32_t left{};
  std::uint32_t right{};
  std::uint64_t value{};
  std::uint16_t depth{1U};
  TensorByteSpanParameterDependency parameter;
  TensorByteSpanParameterProjection projection;
  const void* input_declaration{};
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
  template <auto Projector, typename... ParamReferences>
  friend TensorByteSpanExpression parameter_projected_byte_span(std::string_view,
                                                                ParamReferences...);
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

/// @brief Derive Tensor slot bytes with a named compile-only C++ Parameter projector.
/// @tparam Projector Free or static non-throwing function with exact
/// `expected<uint64_t, Error>(const Params&) noexcept` signature.
/// @tparam ParamReferences One or more declared owning-Params dependency token types.
/// @param stable_projection_id Nonempty bounded UTF-8 semantic identity copied by this call.
/// @param dependencies Every Parameter field read by the projector, including nested paths.
/// @return Output-local expression leaf that may use the ordinary checked expression operations.
/// @retval ErrorCode::kArithmeticOverflow The projector's checked arithmetic overflowed. Returning
/// this code preserves the existing `PHYSICAL_TENSOR_BYTE_SPAN_OVERFLOW` compile diagnostic;
/// other errors produce `PHYSICAL_TENSOR_BYTE_SPAN_PROJECTION_FAILED` without exposing messages.
///
/// The projector must be deterministic, side-effect-free, reentrant, and read only the declared
/// dependencies. Compilation invokes it once against finalized generation-zero Parameters. Runtime
/// retains neither the callable nor its dependency readers.
template <auto Projector, typename... ParamReferences>
[[nodiscard]] TensorByteSpanExpression parameter_projected_byte_span(
    std::string_view stable_projection_id, ParamReferences... dependencies) {
  using ProjectorTraits = detail::TensorByteSpanProjectorTraits<decltype(Projector)>;
  if constexpr (!ProjectorTraits::kFunction) {
    static_assert(ProjectorTraits::kFunction,
                  "parameter byte-span projector must be a free or static function taking const "
                  "Params&");
    return {};
  } else if constexpr (!ProjectorTraits::kNoexcept) {
    static_assert(ProjectorTraits::kNoexcept,
                  "parameter byte-span projector must be declared noexcept");
    return {};
  } else if constexpr (!ProjectorTraits::kExactResult) {
    static_assert(ProjectorTraits::kExactResult,
                  "parameter byte-span projector must return expected<uint64_t, Error>");
    return {};
  } else {
    static_assert(Projector != nullptr, "parameter byte-span projector must not be null");
    using Params = typename ProjectorTraits::Params;
    constexpr bool kReferencesValid = (detail::TensorByteSpanProjectionReferenceTraits<
                                           std::remove_cvref_t<ParamReferences>>::kValid &&
                                       ...);
    static_assert(sizeof...(ParamReferences) != 0U,
                  "parameter byte-span projector requires at least one Parameter dependency");
    static_assert(sizeof...(ParamReferences) <= kMaximumTensorByteSpanExpressionDependencies,
                  "parameter byte-span projector exceeds the dependency limit");
    if constexpr (!kReferencesValid) {
      static_assert(kReferencesValid,
                    "parameter byte-span projector dependencies must be param_ref values");
      return {};
    } else {
      constexpr bool kRootsMatch =
          (std::same_as<typename detail::TensorByteSpanProjectionReferenceTraits<
                            std::remove_cvref_t<ParamReferences>>::Root,
                        Params> &&
           ...);
      static_assert(kRootsMatch,
                    "parameter byte-span projector dependencies must belong to its exact Params "
                    "type");
      using Model = detail::TensorByteSpanProjectorModel<Projector, Params>;
      std::vector<detail::TensorByteSpanProjectionDependency> copied;
      copied.reserve(sizeof...(ParamReferences));
      (copied.push_back(detail::TensorByteSpanProjectionReferenceTraits<
                        std::remove_cvref_t<ParamReferences>>::make(dependencies)),
       ...);
      return TensorByteSpanExpression{detail::TensorByteSpanAuthoringNode{
          .operation = TensorByteSpanOperation::kParameterProjection,
          .projection =
              detail::TensorByteSpanParameterProjection{
                  .root_type_token = &detail::kParameterTypeToken<Params>,
                  .root_schema = &Model::root_schema,
                  .evaluate = &Model::evaluate,
                  .implementation = Model::implementation_identity(),
                  .stable_id = std::string{stable_projection_id},
                  .dependencies = std::move(copied),
              },
      }};
    }
  }
}

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

namespace detail {

enum class TensorMemoryKindParameterSource : std::uint8_t {
  kMemoryKind,
  kBoolean,
  kEnum,
};

/// Process-local Parameter dependency consumed only while replaying an Operator factory.
struct TensorMemoryKindParameterDependency {
  const void* root_type_token{};
  const ParamSchemaDescriptor* (*root_schema)() noexcept {};
  const ParamFieldDescriptor* (*root_field)() noexcept {};
  const ParamFieldDescriptor* (*final_field)() noexcept {};
  std::vector<ParamSchemaHash> (*member_full_ids)() = nullptr;
  std::string (*canonical_path)() = nullptr;
  bool (*has_sensitive_member)() noexcept {};
  bool (*read_initial)(const void*, ParamScalarValue&) noexcept {};
  TensorMemoryKindParameterSource source{TensorMemoryKindParameterSource::kMemoryKind};

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return root_type_token != nullptr && root_schema != nullptr && root_field != nullptr &&
           final_field != nullptr && member_full_ids != nullptr && canonical_path != nullptr &&
           has_sensitive_member != nullptr && read_initial != nullptr;
  }
};

struct TensorMemoryKindAuthoringCase {
  ParamScalarValue source;
  MemoryKind memory_kind{MemoryKind::kUnknown};
};

template <auto First, auto... Rest>
struct TensorMemoryKindParameterDependencyModel {
  using Root = std::remove_cv_t<typename MemberPointerTraits<decltype(First)>::Owner>;
  using Value = std::remove_cv_t<typename ParamPathValue<First, Rest...>::Type>;

  [[nodiscard]] static const ParamSchemaDescriptor* root_schema() noexcept {
    return param_schema_descriptor<Root>();
  }

  [[nodiscard]] static bool read_initial(const void* aggregate, ParamScalarValue& result) noexcept {
    if (aggregate == nullptr) {
      return false;
    }
    const Value* value = param_path_value<First, Rest...>(*static_cast<const Root*>(aggregate));
    if (value == nullptr) {
      return false;
    }
    result = ParamScalarValue::from(*value);
    return result.kind != ParamScalarKind::kNone;
  }

  template <TensorMemoryKindParameterSource Source>
  [[nodiscard]] static TensorMemoryKindParameterDependency make() noexcept {
    return TensorMemoryKindParameterDependency{
        .root_type_token = &kParameterTypeToken<Root>,
        .root_schema = &root_schema,
        .root_field = &declared_member_descriptor<First>,
        .final_field = &final_param_field<First, Rest...>,
        .member_full_ids = &param_member_full_ids<First, Rest...>,
        .canonical_path = &canonical_param_path<First, Rest...>,
        .has_sensitive_member = &param_path_has_sensitive_member<First, Rest...>,
        .read_initial = &read_initial,
        .source = Source,
    };
  }
};

}  // namespace detail

/// @brief Carry one typed source-to-memory mapping case into closed expression authoring.
/// @tparam Source Exact Boolean or enum Parameter field type.
template <typename Source>
  requires(std::same_as<std::remove_cv_t<Source>, bool> || std::is_enum_v<std::remove_cv_t<Source>>)
struct TensorMemoryKindCase {
  Source source;           ///< Exact source value.
  MemoryKind memory_kind;  ///< Concrete non-unknown destination kind.
};

/// @brief Create one typed source-to-memory mapping case.
/// @tparam Source Exact Boolean or enum type inferred from \p source.
/// @param source Parameter value selected by this case.
/// @param memory_kind Concrete destination memory kind.
/// @return Typed case copied by `parameter_memory_kind`.
template <typename Source>
  requires(std::same_as<std::remove_cv_t<Source>, bool> || std::is_enum_v<std::remove_cv_t<Source>>)
[[nodiscard]] constexpr TensorMemoryKindCase<std::remove_cv_t<Source>> memory_kind_case(
    Source source, MemoryKind memory_kind) noexcept {
  return TensorMemoryKindCase<std::remove_cv_t<Source>>{
      .source = source,
      .memory_kind = memory_kind,
  };
}

/// @brief Build one closed compile-time Tensor output memory-kind expression.
///
/// Compilation resolves the expression from finalized generation-zero Parameters and freezes the
/// concrete result into the execution plan. Changing a recompile-required dependency requires a
/// new plan. Runtime and lifecycle transitions never evaluate this value or migrate a pool.
class HOLOSCAN_CORE_EXPORT TensorMemoryKindExpression {
 public:
  TensorMemoryKindExpression() = default;

  /// @brief Return the first structural authoring failure retained by this expression.
  /// @return `TensorMemoryKindExpressionInvalidReason::kNone` when locally inspectable structure is
  /// valid.
  /// @note Parameter schema type, field, domain, and mapping agreement are validated when the
  /// declaration is normalized into a Graph.
  [[nodiscard]] TensorMemoryKindExpressionInvalidReason invalid_reason() const noexcept {
    return invalid_reason_;
  }

 private:
  TensorMemoryKindExpression(TensorMemoryKindExpressionForm form,
                             detail::TensorMemoryKindParameterDependency parameter,
                             std::vector<detail::TensorMemoryKindAuthoringCase> cases);

  TensorMemoryKindExpressionForm form_{TensorMemoryKindExpressionForm::kDirect};
  detail::TensorMemoryKindParameterDependency parameter_;
  std::vector<detail::TensorMemoryKindAuthoringCase> cases_;
  TensorMemoryKindExpressionInvalidReason invalid_reason_{
      TensorMemoryKindExpressionInvalidReason::kEmpty};

  friend class detail::OperatorSpecCompilerAccess;
  friend class TensorFieldOutputSpecBuilder;
  friend class TensorOutputSpecBuilder;
  template <auto... MemberPath>
  friend TensorMemoryKindExpression parameter_memory_kind(ParamRef<MemberPath...>);
  template <auto... MemberPath, typename Source>
  friend TensorMemoryKindExpression parameter_memory_kind(
      ParamRef<MemberPath...>, std::initializer_list<TensorMemoryKindCase<Source>>);
};

/// @brief Derive Tensor output memory directly from one exact `MemoryKind` Parameter field.
/// @tparam MemberPath Declared owning-Operator Parameter field path.
/// @param reference Empty compile-time dependency token.
/// @return Process-local authoring expression resolved during compilation.
/// @note The field schema must declare a nonempty `.enum_values(...)` domain containing exactly the
/// concrete memory kinds supported by the Operator. `kUnknown` is not an admissible choice.
template <auto... MemberPath>
[[nodiscard]] TensorMemoryKindExpression parameter_memory_kind(ParamRef<MemberPath...> reference) {
  (void)reference;
  using Model = detail::TensorMemoryKindParameterDependencyModel<MemberPath...>;
  static_assert(std::same_as<typename Model::Value, MemoryKind>,
                "direct parameter_memory_kind requires an exact MemoryKind field");
  return TensorMemoryKindExpression{
      TensorMemoryKindExpressionForm::kDirect,
      Model::template make<detail::TensorMemoryKindParameterSource::kMemoryKind>(),
      {}};
}

/// @brief Derive Tensor output memory through an explicit finite source-value mapping.
/// @tparam MemberPath Declared owning-Operator Parameter field path.
/// @tparam Source Exact Boolean or schema-declared enum field type.
/// @param reference Empty compile-time dependency token.
/// @param cases Complete mapping copied during this call.
/// @return Process-local authoring expression resolved during compilation.
template <auto... MemberPath, typename Source>
[[nodiscard]] TensorMemoryKindExpression parameter_memory_kind(
    ParamRef<MemberPath...> reference, std::initializer_list<TensorMemoryKindCase<Source>> cases) {
  (void)reference;
  using Model = detail::TensorMemoryKindParameterDependencyModel<MemberPath...>;
  using ExactSource = std::remove_cv_t<Source>;
  static_assert(std::same_as<typename Model::Value, ExactSource>,
                "memory-kind mapping source must exactly match the Parameter field type");
  static_assert(std::same_as<ExactSource, bool> || std::is_enum_v<ExactSource>,
                "memory-kind mapping requires a Boolean or enum Parameter field");
  constexpr auto kSource = std::same_as<ExactSource, bool>
                               ? detail::TensorMemoryKindParameterSource::kBoolean
                               : detail::TensorMemoryKindParameterSource::kEnum;
  std::vector<detail::TensorMemoryKindAuthoringCase> copied;
  copied.reserve(cases.size());
  for (const auto& item : cases) {
    copied.push_back(detail::TensorMemoryKindAuthoringCase{
        .source = ParamScalarValue::from(item.source),
        .memory_kind = item.memory_kind,
    });
  }
  return TensorMemoryKindExpression{
      TensorMemoryKindExpressionForm::kMapped, Model::template make<kSource>(), std::move(copied)};
}

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

/// @brief Carry output-specific facts resolved by the compiler before physical Tensor lowering.
///
/// Each present member is the sole authoring source for that fact. The legacy byte-span overload
/// remains available and forwards to this aggregate. Future derivations should add named members
/// here instead of multiplying positional overloads.
struct TensorOutputDerivations {
  std::optional<TensorByteSpanExpression> max_byte_span;  ///< Per-slot capacity expression.
  std::optional<TensorMemoryKindExpression> memory_kind;  ///< Physical storage expression.
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

/// @brief Normalize a runtime-pool output with selected compile-derived allocation facts.
/// @param spec Producer facts omitting every derived literal source.
/// @param derive_byte_span Whether slot byte capacity is supplied by a derivation.
/// @param derive_memory_kind Whether physical memory kind is supplied by a derivation.
/// @return Owned contract retaining unresolved selected facts for compiler resolution.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<TensorContractRecord, Error>
normalize_tensor_output_spec_with_derivations(const TensorOutputSpec& spec, bool derive_byte_span,
                                              bool derive_memory_kind);

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
