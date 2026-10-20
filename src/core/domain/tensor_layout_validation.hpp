/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <dlpack/dlpack.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>

#include <holoscan/core/domain/dlpack_utils.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan::detail {

enum class TensorStrideUnit : std::uint8_t {
  kBytes,
  kElements,
};

struct ValidatedTensorLayout {
  std::uint64_t item_bytes{};
  std::uint64_t element_count{};
  std::uint64_t byte_size{};
  std::uint64_t byte_span{};
};

[[nodiscard]] constexpr bool valid_tensor_dtype(const DLDataType dtype) noexcept {
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

// Layout validation is one bounded arithmetic proof whose ordered checks prevent intermediate
// overflow and partial acceptance; splitting it would duplicate that state.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] inline expected<ValidatedTensorLayout, Error> validate_tensor_layout(
    const std::span<const std::int64_t> shape, const DLDataType dtype,
    const std::optional<std::span<const std::int64_t>> strides,
    const TensorStrideUnit stride_unit) noexcept {
  ValidatedTensorLayout result;
  result.item_bytes = dlpack_dtype_item_size_bytes(dtype);
  if (!valid_tensor_dtype(dtype) || result.item_bytes == 0U ||
      (strides && strides->size() != shape.size())) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  auto has_zero_extent = false;
  for (const std::int64_t extent : shape) {
    if (extent < 0) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    has_zero_extent = has_zero_extent || extent == 0;
  }

  result.element_count = has_zero_extent ? 0U : 1U;
  if (!has_zero_extent) {
    for (const std::int64_t extent : shape) {
      const auto normalized_extent = static_cast<std::uint64_t>(extent);
      if (normalized_extent != 0U &&
          result.element_count > std::numeric_limits<std::uint64_t>::max() / normalized_extent) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      result.element_count *= normalized_extent;
    }
    if (result.element_count > std::numeric_limits<std::uint64_t>::max() / result.item_bytes) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    result.byte_size = result.element_count * result.item_bytes;
  }

  if (!strides) {
    result.byte_span = result.byte_size;
    return result;
  }

  std::uint64_t maximum_offset = 0U;
  for (std::size_t index = 0U; index < shape.size(); ++index) {
    const std::int64_t stride = (*strides)[index];
    if (stride < 0) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    std::uint64_t byte_stride = static_cast<std::uint64_t>(stride);
    if (stride_unit == TensorStrideUnit::kElements) {
      if (byte_stride > std::numeric_limits<std::uint64_t>::max() / result.item_bytes) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      byte_stride *= result.item_bytes;
    } else if (byte_stride % result.item_bytes != 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }

    const auto extent_minus_one =
        shape[index] == 0 ? 0U : static_cast<std::uint64_t>(shape[index] - 1);
    if (byte_stride != 0U &&
        extent_minus_one > std::numeric_limits<std::uint64_t>::max() / byte_stride) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const std::uint64_t offset = extent_minus_one * byte_stride;
    if (maximum_offset > std::numeric_limits<std::uint64_t>::max() - offset) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    maximum_offset += offset;
  }

  if (has_zero_extent) {
    return result;
  }
  if (maximum_offset > std::numeric_limits<std::uint64_t>::max() - result.item_bytes) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  result.byte_span = maximum_offset + result.item_bytes;

  // Prove non-overlap without allocating or enumerating elements. Sorting active axes by stride
  // and requiring each stride to start beyond the span covered by all smaller-stride axes accepts
  // ordinary contiguous, padded, and axis-permuted layouts and rejects ambiguous interleaving.
  std::uint64_t covered_span = result.item_bytes;
  std::uint64_t previous_stride = 0U;
  std::size_t previous_index = 0U;
  bool have_previous = false;
  for (std::size_t selected_count = 0U; selected_count < shape.size(); ++selected_count) {
    std::size_t selected_index = shape.size();
    std::uint64_t selected_stride = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t index = 0U; index < shape.size(); ++index) {
      if (shape[index] <= 1) {
        continue;
      }
      std::uint64_t byte_stride = static_cast<std::uint64_t>((*strides)[index]);
      if (stride_unit == TensorStrideUnit::kElements) {
        byte_stride *= result.item_bytes;
      }
      const bool after_previous = !have_previous || byte_stride > previous_stride ||
                                  (byte_stride == previous_stride && index > previous_index);
      if (after_previous && (byte_stride < selected_stride ||
                             (byte_stride == selected_stride && index < selected_index))) {
        selected_index = index;
        selected_stride = byte_stride;
      }
    }
    if (selected_index == shape.size()) {
      break;
    }
    if (selected_stride < covered_span) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const auto extent_minus_one = static_cast<std::uint64_t>(shape[selected_index] - 1);
    const std::uint64_t additional_span = extent_minus_one * selected_stride;
    if (covered_span > std::numeric_limits<std::uint64_t>::max() - additional_span) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    covered_span += additional_span;
    previous_stride = selected_stride;
    previous_index = selected_index;
    have_previous = true;
  }
  return result;
}

}  // namespace holoscan::detail
