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
#include <holoscan/core/sample.hpp>

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
};

enum class ImageTensorLayout : std::uint8_t {
  kInterleaved,
  kPlanar,
};

struct ImageTensorView {
  static constexpr std::size_t kMaxRank = 4U;

  const void* data{};
  std::array<std::int64_t, kMaxRank> shape{};
  std::array<std::int64_t, kMaxRank> byte_strides{};
  DLDataType dtype{};
  DLDevice device{};
  std::uint32_t rank{};
  std::int64_t batch{};
  std::int64_t height{};
  std::int64_t width{};
  std::int64_t channels{};
  ImageTensorLayout image_layout{ImageTensorLayout::kInterleaved};

  [[nodiscard]] const char* layout() const noexcept {
    if (image_layout == ImageTensorLayout::kPlanar) {
      return rank == 3U ? "CHW" : "NCHW";
    }
    return rank == 3U ? "HWC" : "NHWC";
  }
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

[[nodiscard]] inline EmitOptions forward_metadata(const SampleMetadata& metadata) {
  EmitOptions options{.flags = metadata.flags};
  if (metadata.source_clock_id.valid()) {
    options.capture_time = metadata.capture_time();
  }
  if (metadata.frame_id != 0U) {
    options.frame_id = metadata.frame_id;
  }
  if (metadata.trace_id != 0U) {
    options.trace = TraceContext::root(metadata.trace_id);
  }
  return options;
}

}  // namespace holoscan::vision::detail
