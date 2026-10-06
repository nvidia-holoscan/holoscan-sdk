// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include <dlpack/dlpack.h>
#include <nvcv/Tensor.hpp>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "image_metadata.hpp"

namespace holoscan::vision::detail {

enum class TensorOperation : std::uint8_t {
  kResize,
  kCustomCrop,
  kCopyMakeBorder,
  kRemap,
  kRemapMap,
  kNormalize,
  kConvertTo,
  kReformat,
  kBayerDemosaic,
};

/// Fixed-capacity byte-stride storage for Vision tensors (rank at most four).
struct FixedTensorByteStrides {
  std::array<std::int64_t, ImageTensorView::kMaxRank> values{};
  std::uint32_t rank{};

  [[nodiscard]] std::span<const std::int64_t> span() const noexcept {
    return {values.data(), rank};
  }
};

/// Convert explicit DLPack element strides, or derive implicit contiguous strides, into bytes.
[[nodiscard]] expected<FixedTensorByteStrides, Error> fixed_tensor_byte_strides(
    const Tensor& tensor);

[[nodiscard]] expected<ImageTensorView, Error> describe_image_tensor(
    const void* data, std::span<const std::int64_t> shape,
    std::span<const std::int64_t> byte_strides, DLDataType dtype, DLDevice device,
    TensorOperation operation, std::uint32_t max_batch_size,
    ImageTensorLayout image_layout = ImageTensorLayout::kInterleaved);

[[nodiscard]] expected<ImageTensorView, Error> describe_image_tensor(
    const Tensor& tensor, TensorOperation operation, std::uint32_t max_batch_size,
    ImageTensorLayout image_layout = ImageTensorLayout::kInterleaved);

[[nodiscard]] expected<std::array<std::int64_t, ImageTensorView::kMaxRank>, Error>
contiguous_byte_strides(std::span<const std::int64_t> shape, DLDataType dtype);

/// Resolve the conservative four-channel Vision output bound used by compile-time projectors.
/// Returns `kInvalidArgument` for malformed geometry, `kNotSupported` for a CV-CUDA span limit,
/// and `kArithmeticOverflow` when the intermediate byte count is not representable.
[[nodiscard]] expected<std::uint64_t, Error> projected_image_output_byte_span(
    std::int32_t width, std::int32_t height, std::uint32_t max_batch_size,
    std::size_t max_element_size) noexcept;

[[nodiscard]] expected<void, Error> validate_crop_rectangle(std::int64_t input_width,
                                                            std::int64_t input_height,
                                                            std::int32_t x, std::int32_t y,
                                                            std::int32_t width,
                                                            std::int32_t height);

[[nodiscard]] expected<nvcv::Tensor, Error> wrap_nvcv_tensor(const ImageTensorView& view);

[[nodiscard]] expected<nvcv::Tensor, Error> wrap_nvcv_tensor(
    const void* data, std::span<const std::int64_t> shape,
    std::span<const std::int64_t> byte_strides, DLDataType dtype, const char* layout);

// Cold-only NVCV 0.18 metadata-handle warmup; does not submit work or touch pixel storage.
// Create count simultaneous handles (1..3), then release them without changing the global
// pool policy. This is not a reservation against concurrent users or later pool growth.
[[nodiscard]] expected<void, Error> prepare_nvcv_tensor_handles(std::size_t count) noexcept;

}  // namespace holoscan::vision::detail
