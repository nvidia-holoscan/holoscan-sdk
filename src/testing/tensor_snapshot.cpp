/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include <holoscan/testing/tensor_snapshot.hpp>

#include "core/domain/tensor_access.hpp"

namespace holoscan::testing::detail {
namespace {
expected<std::vector<std::int64_t>, Error> snapshot_byte_strides(const Tensor& tensor) {
  // The caller has validated the full layout, including positive item size and nonnegative strides.
  // Core's layout proof uses uint64_t spans. Snapshot byte strides have an int64_t public
  // representation; an empty logical shape can still overflow that narrower metadata.
  constexpr auto maximum_stride =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  const auto shape = tensor.shape_span();
  const auto source_strides = tensor.dlpack_strides_span();
  const auto item_bytes = tensor.itemsize();
  std::vector<std::int64_t> result(shape.size());
  if (!source_strides.empty()) {
    for (std::size_t axis = 0U; axis < shape.size(); ++axis) {
      const auto stride = static_cast<std::uint64_t>(source_strides[axis]);
      if (stride > maximum_stride / item_bytes) {
        return make_unexpected(
            Error{ErrorCode::kNotSupported, "operator test Tensor snapshot byte stride overflows"});
      }
      result[axis] = static_cast<std::int64_t>(stride * item_bytes);
    }
  } else {
    auto stride = item_bytes;
    for (std::size_t axis = shape.size(); axis > 0U; --axis) {
      result[axis - 1U] = static_cast<std::int64_t>(stride);
      // The product after axis zero is not a stride and need not be materialized.
      if (axis > 1U) {
        const auto extent = static_cast<std::uint64_t>(shape[axis - 1U]);
        if (extent != 0U && stride > maximum_stride / extent) {
          return make_unexpected(Error{ErrorCode::kNotSupported,
                                       "operator test Tensor snapshot byte stride overflows"});
        }
        stride *= extent;
      }
    }
  }
  return result;
}
}  // namespace
struct TensorSnapshotAccess {
  static expected<TensorSnapshot, Error> copy(const Tensor& tensor, SampleMetadata metadata,
                                              std::size_t maximum_bytes, cudaStream_t stream) {
    const auto layout = holoscan::detail::TensorAccess::snapshot(tensor);
    if (!layout.valid || (layout.byte_size != 0U &&
                          (!tensor.is_contiguous() || tensor.nbytes() <= 0 ||
                           static_cast<std::uint64_t>(tensor.nbytes()) != layout.byte_size ||
                           tensor.data() == nullptr))) {
      return make_unexpected(
          Error{ErrorCode::kNotSupported,
                "operator test snapshot requires a valid contiguous Tensor layout"});
    }
    const auto bytes = layout.byte_size;
    if (bytes > maximum_bytes) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted,
                                   "operator test Tensor snapshot byte bound exceeded"});
    }
    auto strides = snapshot_byte_strides(tensor);
    if (!strides)
      return make_unexpected(std::move(strides).error());
    TensorSnapshot result;
    result.bytes_.resize(static_cast<std::size_t>(bytes));
    result.shape_ = tensor.shape();
    result.strides_ = std::move(*strides);
    result.offset_ = tensor.byte_offset();
    result.device_ = tensor.device();
    result.dtype_ = tensor.dtype();
    result.metadata_ = metadata;
    if (bytes == 0U)
      return result;
    auto copied = tensor.copy_to_host(result.bytes_.data(), result.bytes_.size(), stream);
    if (!copied)
      return make_unexpected(std::move(copied).error());
    return result;
  }
};
expected<TensorSnapshot, Error> capture_tensor_snapshot(const Tensor& value,
                                                        SampleMetadata metadata,
                                                        std::size_t maximum_bytes,
                                                        cudaStream_t stream) {
  return TensorSnapshotAccess::copy(value, metadata, maximum_bytes, stream);
}
}  // namespace holoscan::testing::detail
