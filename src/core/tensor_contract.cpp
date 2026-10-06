/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/tensor_contract.hpp>

#include "memory_kind_validation.hpp"

namespace holoscan {
namespace {

constexpr std::uint64_t kMaximumTensorByteSpanAlignment = 1ULL << 30U;

[[nodiscard]] bool valid_expression_shape(
    const std::vector<detail::TensorByteSpanAuthoringNode>& nodes) noexcept {
  if (nodes.empty() || nodes.size() > kMaximumTensorByteSpanExpressionNodes) {
    return false;
  }
  for (std::size_t index = 0U; index < nodes.size(); ++index) {
    const auto& node = nodes[index];
    if (node.depth == 0U || node.depth > kMaximumTensorByteSpanExpressionDepth) {
      return false;
    }
    const bool binary = node.operation == TensorByteSpanOperation::kMultiply ||
                        node.operation == TensorByteSpanOperation::kCeilingDivide ||
                        node.operation == TensorByteSpanOperation::kMaximum;
    if (binary && (node.left >= index || node.right >= index)) {
      return false;
    }
    if (node.operation == TensorByteSpanOperation::kAlignUp && node.left >= index) {
      return false;
    }
  }
  return true;
}

}  // namespace

TensorByteSpanExpression::TensorByteSpanExpression(detail::TensorByteSpanAuthoringNode node)
    : nodes_{std::move(node)} {
  if (nodes_.front().operation == TensorByteSpanOperation::kPositiveConstant &&
      nodes_.front().value == 0U) {
    invalid_reason_ = TensorByteSpanExpressionInvalidReason::kNonPositiveConstant;
    return;
  }
  if (nodes_.front().operation == TensorByteSpanOperation::kParameterProjection &&
      (!nodes_.front().projection ||
       nodes_.front().projection.stable_id.size() > kMaximumTensorByteSpanProjectionIdBytes ||
       nodes_.front().projection.dependencies.size() >
           kMaximumTensorByteSpanExpressionDependencies)) {
    invalid_reason_ = TensorByteSpanExpressionInvalidReason::kProjectionInvalid;
    return;
  }
  invalid_reason_ = TensorByteSpanExpressionInvalidReason::kNone;
}

TensorByteSpanExpression TensorByteSpanExpression::combine(
    const TensorByteSpanExpression& right, TensorByteSpanOperation operation) const {
  TensorByteSpanExpression result;
  if (invalid_reason_ != TensorByteSpanExpressionInvalidReason::kNone) {
    result.invalid_reason_ = invalid_reason_;
    return result;
  }
  if (right.invalid_reason_ != TensorByteSpanExpressionInvalidReason::kNone) {
    result.invalid_reason_ = right.invalid_reason_;
    return result;
  }
  if (nodes_.empty() || right.nodes_.empty()) {
    result.invalid_reason_ = TensorByteSpanExpressionInvalidReason::kEmpty;
    return result;
  }
  if (nodes_.size() + right.nodes_.size() + 1U > kMaximumTensorByteSpanExpressionNodes) {
    result.invalid_reason_ = TensorByteSpanExpressionInvalidReason::kNodeLimitExceeded;
    return result;
  }
  result.nodes_ = nodes_;
  const std::uint32_t offset = static_cast<std::uint32_t>(result.nodes_.size());
  result.nodes_.reserve(nodes_.size() + right.nodes_.size() + 1U);
  for (auto node : right.nodes_) {
    const bool binary = node.operation == TensorByteSpanOperation::kMultiply ||
                        node.operation == TensorByteSpanOperation::kCeilingDivide ||
                        node.operation == TensorByteSpanOperation::kMaximum;
    if (binary) {
      node.left += offset;
      node.right += offset;
    } else if (node.operation == TensorByteSpanOperation::kAlignUp) {
      node.left += offset;
    }
    result.nodes_.push_back(std::move(node));
  }
  const std::uint16_t depth =
      static_cast<std::uint16_t>(std::max(nodes_.back().depth, right.nodes_.back().depth) + 1U);
  if (depth > kMaximumTensorByteSpanExpressionDepth) {
    result.nodes_.clear();
    result.invalid_reason_ = TensorByteSpanExpressionInvalidReason::kDepthLimitExceeded;
    return result;
  }
  result.nodes_.push_back(detail::TensorByteSpanAuthoringNode{
      .operation = operation,
      .left = static_cast<std::uint32_t>(nodes_.size() - 1U),
      .right = static_cast<std::uint32_t>(result.nodes_.size() - 1U),
      .depth = depth,
  });
  result.invalid_reason_ = valid_expression_shape(result.nodes_)
                               ? TensorByteSpanExpressionInvalidReason::kNone
                               : TensorByteSpanExpressionInvalidReason::kInvalidOperand;
  return result;
}

TensorByteSpanExpression TensorByteSpanExpression::multiplied_by(
    const TensorByteSpanExpression& factor) const {
  return combine(factor, TensorByteSpanOperation::kMultiply);
}

TensorByteSpanExpression TensorByteSpanExpression::ceiling_divided_by(
    const TensorByteSpanExpression& divisor) const {
  return combine(divisor, TensorByteSpanOperation::kCeilingDivide);
}

TensorByteSpanExpression TensorByteSpanExpression::maximum_with(
    const TensorByteSpanExpression& other) const {
  return combine(other, TensorByteSpanOperation::kMaximum);
}

TensorByteSpanExpression TensorByteSpanExpression::aligned_to(std::uint64_t alignment) const {
  TensorByteSpanExpression result = *this;
  if (invalid_reason_ != TensorByteSpanExpressionInvalidReason::kNone) {
    return result;
  }
  if (nodes_.empty()) {
    result.invalid_reason_ = TensorByteSpanExpressionInvalidReason::kEmpty;
    return result;
  }
  if (alignment == 0U || alignment > kMaximumTensorByteSpanAlignment ||
      (alignment & (alignment - 1U)) != 0U) {
    result.nodes_.clear();
    result.invalid_reason_ = TensorByteSpanExpressionInvalidReason::kAlignmentInvalid;
    return result;
  }
  if (nodes_.size() == kMaximumTensorByteSpanExpressionNodes) {
    result.nodes_.clear();
    result.invalid_reason_ = TensorByteSpanExpressionInvalidReason::kNodeLimitExceeded;
    return result;
  }
  if (nodes_.back().depth == kMaximumTensorByteSpanExpressionDepth) {
    result.nodes_.clear();
    result.invalid_reason_ = TensorByteSpanExpressionInvalidReason::kDepthLimitExceeded;
    return result;
  }
  result.nodes_.push_back(detail::TensorByteSpanAuthoringNode{
      .operation = TensorByteSpanOperation::kAlignUp,
      .left = static_cast<std::uint32_t>(nodes_.size() - 1U),
      .value = alignment,
      .depth = static_cast<std::uint16_t>(nodes_.back().depth + 1U),
  });
  result.invalid_reason_ = valid_expression_shape(result.nodes_)
                               ? TensorByteSpanExpressionInvalidReason::kNone
                               : TensorByteSpanExpressionInvalidReason::kInvalidOperand;
  return result;
}

TensorByteSpanExpression tensor_byte_span(std::uint64_t value) {
  return TensorByteSpanExpression{detail::TensorByteSpanAuthoringNode{
      .operation = TensorByteSpanOperation::kPositiveConstant,
      .value = value,
  }};
}

TensorByteSpanExpression detail_make_input_max_byte_span(const void* input) {
  return TensorByteSpanExpression{detail::TensorByteSpanAuthoringNode{
      .operation = TensorByteSpanOperation::kInputMaxByteSpan,
      .input_declaration = input,
  }};
}

TensorByteSpanExpression detail_make_hosted_parameter_byte_span(const void* root_type_token,
                                                                const ParamSchemaDescriptor* schema,
                                                                const ParamFieldDescriptor* field,
                                                                bool shape) {
  return TensorByteSpanExpression{detail::TensorByteSpanAuthoringNode{
      .operation = shape ? TensorByteSpanOperation::kParameterContiguousShape
                         : TensorByteSpanOperation::kParameterByteSpan,
      .parameter =
          detail::TensorByteSpanParameterDependency{
              .root_type_token = root_type_token,
              .hosted_schema = schema,
              .hosted_field = field,
              .hosted_path = field == nullptr ? std::string_view{} : field->canonical_name,
              .shape = shape,
          },
  }};
}

TensorMemoryKindExpression::TensorMemoryKindExpression(
    TensorMemoryKindExpressionForm form, detail::TensorMemoryKindParameterDependency parameter,
    std::vector<detail::TensorMemoryKindAuthoringCase> cases)
    : form_(form), parameter_(parameter), cases_(std::move(cases)) {
  if (!parameter_) {
    invalid_reason_ = TensorMemoryKindExpressionInvalidReason::kEmpty;
    return;
  }
  if (form_ == TensorMemoryKindExpressionForm::kDirect) {
    invalid_reason_ = cases_.empty() ? TensorMemoryKindExpressionInvalidReason::kNone
                                     : TensorMemoryKindExpressionInvalidReason::kUnexpectedMapping;
    return;
  }
  if (cases_.empty()) {
    invalid_reason_ = TensorMemoryKindExpressionInvalidReason::kMappingEmpty;
    return;
  }
  if (cases_.size() > kMaximumTensorMemoryKindMappingCases) {
    invalid_reason_ = TensorMemoryKindExpressionInvalidReason::kMappingLimitExceeded;
    return;
  }
  for (std::size_t index = 0U; index < cases_.size(); ++index) {
    if (!detail::valid_concrete_memory_kind(cases_[index].memory_kind)) {
      invalid_reason_ = TensorMemoryKindExpressionInvalidReason::kMappingTargetInvalid;
      return;
    }
    const auto duplicate = std::find_if(
        cases_.begin(), cases_.begin() + static_cast<std::ptrdiff_t>(index), [&](const auto& item) {
          return item.source == cases_[index].source;
        });
    if (duplicate != cases_.begin() + static_cast<std::ptrdiff_t>(index)) {
      invalid_reason_ = TensorMemoryKindExpressionInvalidReason::kMappingDuplicate;
      return;
    }
  }
  invalid_reason_ = TensorMemoryKindExpressionInvalidReason::kNone;
}

TensorMemoryKindExpression detail_make_hosted_parameter_memory_kind(
    const void* root_type_token, const ParamSchemaDescriptor* schema,
    const ParamFieldDescriptor* field,
    std::span<const detail::TensorMemoryKindAuthoringCase> mapping) {
  return TensorMemoryKindExpression{
      TensorMemoryKindExpressionForm::kMapped,
      detail::TensorMemoryKindParameterDependency{
          .root_type_token = root_type_token,
          .hosted_schema = schema,
          .hosted_field = field,
          .hosted_path = field == nullptr ? std::string_view{} : field->canonical_name,
          .source = detail::TensorMemoryKindParameterSource::kBoolean,
      },
      {mapping.begin(), mapping.end()}};
}

}  // namespace holoscan

namespace holoscan::detail {
namespace {

template <typename Enum>
[[nodiscard]] constexpr bool enum_at_most(Enum value, Enum last) noexcept {
  using Underlying = std::underlying_type_t<Enum>;
  static_assert(std::is_unsigned_v<Underlying>);
  return static_cast<Underlying>(value) <= static_cast<Underlying>(last);
}

[[nodiscard]] bool valid_dtype(const DLDataType dtype) noexcept {
  if (dtype.bits == 0U || dtype.lanes == 0U) {
    return false;
  }
  switch (dtype.code) {
    case kDLInt:
    case kDLUInt:
    case kDLFloat:
    case kDLOpaqueHandle:
    case kDLBfloat:
    case kDLComplex:
    case kDLBool:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] Error contract_error(std::string_view token) {
  return Error{ErrorCode::kInvalidArgument, std::string(token)};
}

[[nodiscard]] bool checked_add(const std::uint64_t left, const std::uint64_t right,
                               std::uint64_t& result) noexcept {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    return false;
  }
  result = left + right;
  return true;
}

[[nodiscard]] bool checked_multiply(const std::uint64_t left, const std::uint64_t right,
                                    std::uint64_t& result) noexcept {
  if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  result = left * right;
  return true;
}

[[nodiscard]] std::optional<std::uint64_t> dtype_item_size(
    const std::optional<DLDataType>& dtype) noexcept {
  if (!dtype || !valid_dtype(*dtype)) {
    return std::nullopt;
  }
  std::uint64_t bits = 0U;
  if (!checked_multiply(dtype->bits, dtype->lanes, bits)) {
    return std::nullopt;
  }
  return (bits + 7U) / 8U;
}

// Logical shape, stride, and byte-span invariants must be normalized as one ordered contract.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> normalize_bounds(const TensorBounds& bounds,
                                                     TensorContractRecord& result) {
  if ((bounds.max_rank && *bounds.max_rank == 0U) ||
      (bounds.max_byte_span && *bounds.max_byte_span == 0U)) {
    return make_unexpected(contract_error("GRAPH_TENSOR_LOGICAL_BOUNDS_INVALID"));
  }

  result.max_rank = bounds.max_rank;
  result.max_byte_span = bounds.max_byte_span;
  result.origins.max_rank =
      bounds.max_rank ? TensorContractValueOrigin::kAuthored : TensorContractValueOrigin::kAbsent;
  result.origins.max_byte_span = bounds.max_byte_span ? TensorContractValueOrigin::kAuthored
                                                      : TensorContractValueOrigin::kAbsent;
  if (!bounds.max_shape) {
    if (bounds.max_abs_byte_strides) {
      return make_unexpected(contract_error("GRAPH_TENSOR_LOGICAL_BOUNDS_INVALID"));
    }
    if (!bounds.axes.empty()) {
      std::optional<std::uint32_t> effective_rank = result.max_rank;
      if (!effective_rank && result.representation.rank) {
        effective_rank = *result.representation.rank == 0U ? 1U : *result.representation.rank;
      }
      if (!effective_rank || bounds.axes.size() > *effective_rank) {
        return make_unexpected(contract_error("GRAPH_TENSOR_LOGICAL_BOUNDS_INVALID"));
      }
      result.axes.assign(bounds.axes.begin(), bounds.axes.end());
    }
    return {};
  }
  if (!bounds.axes.empty()) {
    return make_unexpected(contract_error("GRAPH_TENSOR_LOGICAL_BOUNDS_INVALID"));
  }

  const TensorShapeView shape = *bounds.max_shape;
  if (shape.size() > std::numeric_limits<std::uint32_t>::max() ||
      (bounds.max_rank && shape.size() > *bounds.max_rank)) {
    return make_unexpected(contract_error("GRAPH_TENSOR_LOGICAL_BOUNDS_INVALID"));
  }
  const std::uint32_t shape_rank = static_cast<std::uint32_t>(shape.size());
  if (!result.max_rank) {
    result.max_rank = shape_rank == 0U ? 1U : shape_rank;
    result.origins.max_rank = TensorContractValueOrigin::kDerivedShapeBounds;
  }
  if (result.representation.rank && *result.representation.rank != shape_rank) {
    return make_unexpected(contract_error("GRAPH_TENSOR_LOGICAL_BOUNDS_INVALID"));
  }

  const auto item_size = dtype_item_size(result.representation.dtype);
  if (!item_size) {
    return make_unexpected(contract_error("GRAPH_TENSOR_BOUNDS_DTYPE_REQUIRED"));
  }
  for (const std::int64_t extent : shape) {
    if (extent < 0) {
      return make_unexpected(contract_error("GRAPH_TENSOR_LOGICAL_BOUNDS_INVALID"));
    }
  }

  result.axes.reserve(shape.size());
  std::uint64_t derived_span = 0U;
  if (bounds.max_abs_byte_strides) {
    const auto strides = *bounds.max_abs_byte_strides;
    if (strides.size() != shape.size() ||
        std::any_of(strides.begin(), strides.end(), [](const std::uint64_t stride) {
          return stride == 0U;
        })) {
      return make_unexpected(contract_error("GRAPH_TENSOR_STRIDE_BOUNDS_INCOMPLETE"));
    }
    if (result.representation.strides &&
        *result.representation.strides != TensorStridePolicy::kBoundedStrided) {
      return make_unexpected(contract_error("GRAPH_TENSOR_STRIDE_POLICY_CONFLICT"));
    }
    result.representation.strides = TensorStridePolicy::kBoundedStrided;
    result.origins.strides = TensorContractValueOrigin::kDerivedShapeBounds;
    const bool zero_extent = std::any_of(
        shape.begin(), shape.end(), [](const std::int64_t extent) { return extent == 0; });
    if (!zero_extent) {
      derived_span = *item_size;
    }
    for (std::size_t axis = 0U; axis < shape.size(); ++axis) {
      result.axes.push_back(TensorAxisBounds{
          .max_extent = static_cast<std::uint64_t>(shape[axis]),
          .max_abs_stride_bytes = strides[axis],
      });
      if (!zero_extent) {
        std::uint64_t contribution = 0U;
        std::uint64_t next = 0U;
        if (!checked_multiply(
                static_cast<std::uint64_t>(shape[axis] - 1), strides[axis], contribution) ||
            !checked_add(derived_span, contribution, next)) {
          return make_unexpected(contract_error("GRAPH_TENSOR_LOGICAL_BOUNDS_INVALID"));
        }
        derived_span = next;
      }
    }
  } else {
    if (result.representation.strides &&
        *result.representation.strides != TensorStridePolicy::kContiguousOnly) {
      return make_unexpected(contract_error("GRAPH_TENSOR_STRIDE_BOUNDS_INCOMPLETE"));
    }
    result.representation.strides = TensorStridePolicy::kContiguousOnly;
    result.origins.strides = TensorContractValueOrigin::kDerivedShapeBounds;
    std::uint64_t stride = *item_size;
    std::uint64_t element_count = 1U;
    result.axes.resize(shape.size());
    for (std::size_t reverse = shape.size(); reverse > 0U; --reverse) {
      const std::size_t axis = reverse - 1U;
      result.axes[axis] = TensorAxisBounds{
          .max_extent = static_cast<std::uint64_t>(shape[axis]),
          .max_abs_stride_bytes = stride,
      };
      const std::uint64_t extent = static_cast<std::uint64_t>(shape[axis]);
      if (!checked_multiply(element_count, extent, element_count) ||
          !checked_multiply(stride, std::max<std::uint64_t>(extent, 1U), stride)) {
        return make_unexpected(contract_error("GRAPH_TENSOR_LOGICAL_BOUNDS_INVALID"));
      }
    }
    if (!checked_multiply(element_count, *item_size, derived_span)) {
      return make_unexpected(contract_error("GRAPH_TENSOR_LOGICAL_BOUNDS_INVALID"));
    }
  }

  if (result.max_byte_span && derived_span > *result.max_byte_span) {
    return make_unexpected(contract_error("GRAPH_TENSOR_LOGICAL_BOUNDS_INVALID"));
  }
  if (!result.max_byte_span && derived_span != 0U) {
    result.max_byte_span = derived_span;
    result.origins.max_byte_span = TensorContractValueOrigin::kDerivedShapeBounds;
  }
  return {};
}

[[nodiscard]] expected<void, Error> normalize_transfer(
    const std::optional<TensorTransferPolicy>& transfer, TensorContractRecord& result) {
  if (!transfer) {
    return {};
  }
  const bool inline_policy =
      transfer->foreign_emit_policy == InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded;
  const bool copy_capable =
      transfer->foreign_emit_policy == InterfaceTensorEmitPolicy::kCopyIfNeeded || inline_policy;
  if (!enum_at_most(transfer->foreign_emit_policy,
                    InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded)) {
    return make_unexpected(contract_error("GRAPH_TENSOR_TRANSFER_POLICY_INVALID"));
  }
  if (inline_policy != transfer->max_inline_bytes.has_value() ||
      transfer->max_inline_bytes.value_or(1U) == 0U ||
      transfer->max_inline_bytes.value_or(0U) > std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(contract_error("GRAPH_TENSOR_INLINE_POLICY_INVALID"));
  }
  if (transfer->max_staging_byte_span.value_or(1U) == 0U ||
      (!copy_capable && transfer->max_staging_byte_span)) {
    return make_unexpected(contract_error("GRAPH_TENSOR_STAGING_POLICY_INVALID"));
  }
  result.transfer = *transfer;
  result.origins.max_staging_byte_span = transfer->max_staging_byte_span
                                             ? TensorContractValueOrigin::kAuthored
                                             : TensorContractValueOrigin::kAbsent;
  if (copy_capable && !result.transfer->max_staging_byte_span) {
    result.transfer->max_staging_byte_span = result.max_byte_span;
    result.origins.max_staging_byte_span = TensorContractValueOrigin::kDerivedMaxByteSpan;
  }
  return {};
}

// Representation, transfer, and allocation facts share one first-error normalization path.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<TensorContractRecord, Error> normalize_contract(
    const TensorRepresentation& representation, const std::optional<TensorBounds>& bounds,
    const std::optional<TensorTransferPolicy>& transfer,
    const std::optional<MemoryConversionPolicy>& memory_conversion,
    const TensorOutputStorage storage, const TensorContractRole role,
    const bool derived_byte_span = false, const bool derived_memory_kind = false) {
  if (representation.dtype && !valid_dtype(*representation.dtype)) {
    return make_unexpected(contract_error("GRAPH_TENSOR_DTYPE_INVALID"));
  }
  if ((representation.memory_kind && !valid_concrete_memory_kind(*representation.memory_kind)) ||
      (representation.strides &&
       !enum_at_most(*representation.strides, TensorStridePolicy::kBoundedStrided)) ||
      (memory_conversion && !enum_at_most(*memory_conversion, MemoryConversionPolicy::kRequire)) ||
      !enum_at_most(storage, TensorOutputStorage::kRuntimePool)) {
    return make_unexpected(contract_error("GRAPH_TENSOR_ENUM_INVALID"));
  }

  const bool authored_bounds =
      bounds && (bounds->max_rank || bounds->max_byte_span || bounds->max_shape ||
                 bounds->max_abs_byte_strides || !bounds->axes.empty());
  const bool authored_meaningful = representation.memory_kind || representation.dtype ||
                                   representation.rank || representation.strides ||
                                   authored_bounds || transfer || memory_conversion ||
                                   storage == TensorOutputStorage::kRuntimePool;
  if (!authored_meaningful) {
    return make_unexpected(contract_error("GRAPH_TENSOR_CONTRACT_EMPTY"));
  }

  TensorContractOrigins origins;
  origins.memory_kind = representation.memory_kind ? TensorContractValueOrigin::kAuthored
                                                   : TensorContractValueOrigin::kAbsent;
  origins.dtype = representation.dtype ? TensorContractValueOrigin::kAuthored
                                       : TensorContractValueOrigin::kAbsent;
  origins.rank = representation.rank ? TensorContractValueOrigin::kAuthored
                                     : TensorContractValueOrigin::kAbsent;
  origins.strides = representation.strides ? TensorContractValueOrigin::kAuthored
                                           : TensorContractValueOrigin::kAbsent;
  TensorContractRecord result{
      .role = role,
      .representation = representation,
      .memory_conversion = memory_conversion,
      .storage = storage,
      .origins = origins,
  };
  if (bounds) {
    auto normalized = normalize_bounds(*bounds, result);
    if (!normalized) {
      return make_unexpected(std::move(normalized.error()));
    }
  }
  if (result.representation.rank) {
    const std::uint32_t effective_rank =
        *result.representation.rank == 0U ? 1U : *result.representation.rank;
    if (result.max_rank && *result.representation.rank > *result.max_rank) {
      return make_unexpected(contract_error("GRAPH_TENSOR_LOGICAL_BOUNDS_INVALID"));
    }
    if (!result.max_rank) {
      result.max_rank = effective_rank;
      result.origins.max_rank = TensorContractValueOrigin::kDerivedExactRank;
    }
  }
  auto normalized_transfer = normalize_transfer(transfer, result);
  if (!normalized_transfer) {
    return make_unexpected(std::move(normalized_transfer.error()));
  }
  if (role == TensorContractRole::kExternalBoundary && !result.representation.strides) {
    result.representation.strides = TensorStridePolicy::kContiguousOnly;
    result.origins.strides = TensorContractValueOrigin::kDefaultedBoundary;
  }
  if (role == TensorContractRole::kExternalBoundary &&
      (!result.representation.memory_kind || !result.max_rank || !result.max_byte_span ||
       !result.representation.strides)) {
    return make_unexpected(contract_error("GRAPH_TENSOR_BOUNDARY_INVALID"));
  }
  if (storage == TensorOutputStorage::kRuntimePool &&
      ((!derived_memory_kind && !result.representation.memory_kind) || !result.max_rank ||
       (!derived_byte_span && !result.max_byte_span))) {
    return make_unexpected(contract_error("GRAPH_TENSOR_ALLOCATION_INVALID"));
  }
  const bool copy_capable = result.transfer && result.transfer->foreign_emit_policy !=
                                                   InterfaceTensorEmitPolicy::kRequireExportable;
  if (copy_capable && !derived_byte_span &&
      (!result.max_byte_span || !result.transfer->max_staging_byte_span)) {
    return make_unexpected(contract_error("GRAPH_TENSOR_STAGING_INVALID"));
  }
  if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t)) {
    if (storage == TensorOutputStorage::kRuntimePool && result.max_byte_span &&
        *result.max_byte_span > std::numeric_limits<std::size_t>::max()) {
      return make_unexpected(contract_error("GRAPH_TENSOR_ALLOCATION_INVALID"));
    }
  }
  if (result.representation.strides == TensorStridePolicy::kBoundedStrided &&
      (!result.max_rank || result.axes.size() != *result.max_rank ||
       std::any_of(result.axes.begin(), result.axes.end(), [](const auto& axis) {
         return axis.max_abs_stride_bytes == 0U;
       }))) {
    return make_unexpected(contract_error("GRAPH_TENSOR_STRIDE_BOUNDS_INCOMPLETE"));
  }
  return result;
}

}  // namespace

expected<TensorContractRecord, Error> normalize_tensor_input_spec(const TensorInputSpec& spec) {
  return normalize_contract(spec.representation,
                            spec.bounds,
                            std::nullopt,
                            spec.memory_conversion,
                            TensorOutputStorage::kPassThrough,
                            TensorContractRole::kInput);
}

expected<TensorContractRecord, Error> normalize_tensor_output_spec(const TensorOutputSpec& spec) {
  return normalize_contract(spec.representation,
                            spec.bounds,
                            spec.transfer,
                            std::nullopt,
                            spec.storage,
                            TensorContractRole::kOutput);
}

expected<TensorContractRecord, Error> normalize_tensor_output_spec_with_derived_byte_span(
    const TensorOutputSpec& spec) {
  return normalize_tensor_output_spec_with_derivations(spec, true, false);
}

expected<TensorContractRecord, Error> normalize_tensor_output_spec_with_derivations(
    const TensorOutputSpec& spec, const bool derive_byte_span, const bool derive_memory_kind) {
  if (!derive_byte_span && !derive_memory_kind) {
    return make_unexpected(contract_error("GRAPH_TENSOR_DERIVATION_INVALID"));
  }
  if (spec.storage != TensorOutputStorage::kRuntimePool) {
    return make_unexpected(contract_error(derive_memory_kind
                                              ? "GRAPH_TENSOR_MEMORY_KIND_DERIVATION_INVALID"
                                              : "GRAPH_TENSOR_BYTE_SPAN_SOURCE_CONFLICT"));
  }
  if (derive_byte_span && (spec.bounds && (spec.bounds->max_byte_span || spec.bounds->max_shape))) {
    return make_unexpected(contract_error("GRAPH_TENSOR_BYTE_SPAN_SOURCE_CONFLICT"));
  }
  if (derive_memory_kind && spec.representation.memory_kind) {
    return make_unexpected(contract_error("GRAPH_TENSOR_MEMORY_KIND_SOURCE_CONFLICT"));
  }
  return normalize_contract(spec.representation,
                            spec.bounds,
                            spec.transfer,
                            std::nullopt,
                            spec.storage,
                            TensorContractRole::kOutput,
                            derive_byte_span,
                            derive_memory_kind);
}

expected<TensorContractRecord, Error>
normalize_tensor_output_spec_refinement_with_derived_byte_span(const TensorOutputSpec& spec) {
  if (spec.storage != TensorOutputStorage::kPassThrough ||
      (spec.bounds && (spec.bounds->max_byte_span || spec.bounds->max_shape))) {
    return make_unexpected(contract_error("GRAPH_TENSOR_BYTE_SPAN_SOURCE_CONFLICT"));
  }
  return normalize_contract(spec.representation,
                            spec.bounds,
                            spec.transfer,
                            std::nullopt,
                            spec.storage,
                            TensorContractRole::kOutput,
                            true);
}

expected<TensorContractRecord, Error> normalize_tensor_boundary_contract(
    const TensorBoundaryContract& contract) {
  return normalize_contract(contract.representation,
                            std::optional{contract.bounds},
                            contract.transfer,
                            contract.memory_conversion,
                            TensorOutputStorage::kPassThrough,
                            TensorContractRole::kExternalBoundary);
}

TensorRepresentation tensor_representation(const TensorContractRecord& contract) noexcept {
  return TensorRepresentation{
      .memory_kind = contract.representation.memory_kind,
      .dtype = contract.representation.dtype,
      .rank = contract.representation.rank,
      .strides = contract.representation.strides,
  };
}

bool tensor_contract_equal(const TensorContractRecord& left,
                           const TensorContractRecord& right) noexcept {
  const bool dtype_equal =
      left.representation.dtype.has_value() == right.representation.dtype.has_value() &&
      (!left.representation.dtype ||
       (left.representation.dtype->code == right.representation.dtype->code &&
        left.representation.dtype->bits == right.representation.dtype->bits &&
        left.representation.dtype->lanes == right.representation.dtype->lanes));
  const bool transfer_equal =
      left.transfer.has_value() == right.transfer.has_value() &&
      (!left.transfer ||
       (left.transfer->foreign_emit_policy == right.transfer->foreign_emit_policy &&
        left.transfer->max_inline_bytes == right.transfer->max_inline_bytes &&
        left.transfer->max_staging_byte_span == right.transfer->max_staging_byte_span));
  return left.role == right.role &&
         left.representation.memory_kind == right.representation.memory_kind && dtype_equal &&
         left.representation.rank == right.representation.rank &&
         left.representation.strides == right.representation.strides &&
         left.max_rank == right.max_rank && left.max_byte_span == right.max_byte_span &&
         left.axes == right.axes && transfer_equal &&
         left.memory_conversion == right.memory_conversion && left.storage == right.storage;
}

}  // namespace holoscan::detail
