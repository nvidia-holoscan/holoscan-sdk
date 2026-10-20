/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <concepts>
#include <cstddef>
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
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/tensor_contract_types.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

namespace holoscan {

class TensorOutputSpecBuilder;
class TensorFieldOutputSpecBuilder;

namespace detail {
class OperatorSpecCompilerAccess;
}  // namespace detail

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
  const ParamSchemaDescriptor* hosted_schema{};
  const ParamFieldDescriptor* hosted_field{};
  std::string_view hosted_path;
  bool shape{};

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    const bool static_dependency = root_schema != nullptr && final_field != nullptr &&
                                   member_full_ids != nullptr && canonical_path != nullptr &&
                                   read_initial != nullptr;
    const bool hosted_dependency =
        hosted_schema != nullptr && hosted_field != nullptr && !hosted_path.empty();
    return root_type_token != nullptr && (static_dependency || hosted_dependency);
  }

  [[nodiscard]] const ParamSchemaDescriptor* schema() const noexcept {
    return hosted_schema != nullptr ? hosted_schema
                                    : (root_schema != nullptr ? root_schema() : nullptr);
  }
  [[nodiscard]] const ParamFieldDescriptor* field() const noexcept {
    return hosted_field != nullptr ? hosted_field
                                   : (final_field != nullptr ? final_field() : nullptr);
  }
  [[nodiscard]] std::string path() const {
    return !hosted_path.empty() ? std::string{hosted_path}
                                : (canonical_path != nullptr ? canonical_path() : std::string{});
  }
  [[nodiscard]] std::vector<ParamSchemaHash> path_full_ids() const {
    return hosted_field != nullptr
               ? std::vector<ParamSchemaHash>{hosted_field->full_id}
               : (member_full_ids != nullptr ? member_full_ids() : std::vector<ParamSchemaHash>{});
  }
  [[nodiscard]] TensorByteSpanParameterReadStatus read(
      const void* aggregate, TensorByteSpanParameterValue& result) const noexcept;
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
  static constexpr ProjectorType kProjector = Projector;

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
  friend TensorByteSpanExpression detail_make_hosted_parameter_byte_span(
      const void*, const ParamSchemaDescriptor*, const ParamFieldDescriptor*, bool);
};

/// @internal Build an input dependency after the complete port type is available.
[[nodiscard]] HOLOSCAN_CORE_EXPORT TensorByteSpanExpression
detail_make_input_max_byte_span(const void* input);

/// @internal Build a hosted-parameter dependency without synthesizing a C++ member pointer.
[[nodiscard]] HOLOSCAN_CORE_EXPORT TensorByteSpanExpression detail_make_hosted_parameter_byte_span(
    const void* root_type_token, const ParamSchemaDescriptor* schema,
    const ParamFieldDescriptor* field, bool shape);

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
  const ParamSchemaDescriptor* hosted_schema{};
  const ParamFieldDescriptor* hosted_field{};
  std::string_view hosted_path;
  TensorMemoryKindParameterSource source{TensorMemoryKindParameterSource::kMemoryKind};

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    const bool static_dependency = root_schema != nullptr && root_field != nullptr &&
                                   final_field != nullptr && member_full_ids != nullptr &&
                                   canonical_path != nullptr && has_sensitive_member != nullptr &&
                                   read_initial != nullptr;
    const bool hosted_dependency =
        hosted_schema != nullptr && hosted_field != nullptr && !hosted_path.empty();
    return root_type_token != nullptr && (static_dependency || hosted_dependency);
  }

  [[nodiscard]] const ParamSchemaDescriptor* schema() const noexcept {
    return hosted_schema != nullptr ? hosted_schema
                                    : (root_schema != nullptr ? root_schema() : nullptr);
  }
  [[nodiscard]] const ParamFieldDescriptor* field() const noexcept {
    return hosted_field != nullptr ? hosted_field
                                   : (final_field != nullptr ? final_field() : nullptr);
  }
  [[nodiscard]] const ParamFieldDescriptor* root() const noexcept {
    return hosted_field != nullptr ? hosted_field
                                   : (root_field != nullptr ? root_field() : nullptr);
  }
  [[nodiscard]] std::string path() const {
    return !hosted_path.empty() ? std::string{hosted_path}
                                : (canonical_path != nullptr ? canonical_path() : std::string{});
  }
  [[nodiscard]] std::vector<ParamSchemaHash> path_full_ids() const {
    return hosted_field != nullptr
               ? std::vector<ParamSchemaHash>{hosted_field->full_id}
               : (member_full_ids != nullptr ? member_full_ids() : std::vector<ParamSchemaHash>{});
  }
  [[nodiscard]] bool sensitive() const noexcept {
    return hosted_field != nullptr ? hosted_field->sensitivity == ParamSensitivity::kSensitive
                                   : (has_sensitive_member == nullptr || has_sensitive_member());
  }
  [[nodiscard]] bool read(const void* aggregate, ParamScalarValue& result) const noexcept {
    if (hosted_field != nullptr) {
      return read_param_field_scalar(*hosted_field, aggregate, result);
    }
    return read_initial != nullptr && read_initial(aggregate, result);
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
  friend TensorMemoryKindExpression detail_make_hosted_parameter_memory_kind(
      const void*, const ParamSchemaDescriptor*, const ParamFieldDescriptor*,
      std::span<const detail::TensorMemoryKindAuthoringCase>);
};

/// @internal Build a hosted Boolean-to-memory-kind dependency without a C++ member pointer.
[[nodiscard]] HOLOSCAN_CORE_EXPORT TensorMemoryKindExpression
detail_make_hosted_parameter_memory_kind(
    const void* root_type_token, const ParamSchemaDescriptor* schema,
    const ParamFieldDescriptor* field,
    std::span<const detail::TensorMemoryKindAuthoringCase> mapping);

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

/// @brief Carry output-specific facts resolved by the compiler before physical Tensor lowering.
///
/// Each present member is the sole authoring source for that fact. The legacy byte-span overload
/// remains available and forwards to this aggregate. Future derivations should add named members
/// here instead of multiplying positional overloads.
struct TensorOutputDerivations {
  std::optional<TensorByteSpanExpression> max_byte_span;  ///< Per-slot capacity expression.
  std::optional<TensorMemoryKindExpression> memory_kind;  ///< Physical storage expression.
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
