// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include "holoviz_op/backends/image_layout.hpp"

namespace holoscan::visualization {

/// Element type and channel count a single-plane ImageFormat expects from its HWC tensor.
struct FormatElement {
  std::uint8_t bits = 0;
  std::int64_t channels = 0;
};

inline std::optional<FormatElement> format_element(ImageFormat format) noexcept {
  switch (format) {
    case ImageFormat::kR8Unorm:
      return FormatElement{8U, 1};
    case ImageFormat::kR8G8B8Unorm:
      return FormatElement{8U, 3};
    case ImageFormat::kR8G8B8A8Unorm:
    case ImageFormat::kB8G8R8A8Unorm:
      return FormatElement{8U, 4};
    case ImageFormat::kR16Unorm:
      return FormatElement{16U, 1};
    case ImageFormat::kR16G16B16A16Unorm:
      return FormatElement{16U, 4};
    case ImageFormat::kNV12:
      return FormatElement{8U, 1};
    case ImageFormat::kYUYV422:
      return FormatElement{8U, 2};
    case ImageFormat::kUnknown:
    // sRGB formats are output-only; sources declare ColorSpace separately.
    case ImageFormat::kR8G8B8A8Srgb:
    case ImageFormat::kB8G8R8A8Srgb:
      return std::nullopt;
  }
  return std::nullopt;
}

/// True when a tensor's dtype and trailing channel dimension agree with the declared format.
/// Accepts the natural element type with `{H, W}` (single channel only) or `{H, W, channels}`,
/// and a packed byte view `uint8 {H, W, bytes_per_pixel}` for producers that emit raw bytes.
/// Without this check a mismatched sample (e.g. uint16 on a kR8Unorm layer) uploads with the
/// wrong pitch and renders silently garbled.
inline bool tensor_matches_format(const holoscan::Tensor& tensor,
                                  const std::vector<std::int64_t>& shape,
                                  ImageFormat format) noexcept {
  const auto element = format_element(format);
  if (!element || shape.size() < 2U || shape.size() > 3U) {
    return false;
  }
  const DLDataType dtype = tensor.dtype();
  if (dtype.code != kDLUInt || dtype.lanes != 1U) {
    return false;
  }
  const std::int64_t channels = shape.size() == 3U ? shape[2] : 1;
  if (dtype.bits == element->bits) {
    return channels == element->channels;
  }
  return dtype.bits == 8U &&
         channels == static_cast<std::int64_t>(backends::source_pixel_bytes(format));
}

struct TensorImageLayout {
  backends::ImageLayout layout;
  uint32_t width;
  uint32_t height;
  size_t pitch;
  size_t storage_rows;
};

inline holoscan::expected<TensorImageLayout, holoscan::Error> image_input_layout(
    const Tensor& tensor, const ImageLayerDesc& desc) {
  const auto invalid = [](const char* message) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument, message});
  };
  const auto shape = tensor.shape();
  const auto strides = tensor.strides();
  if (!tensor_matches_format(tensor, shape, desc.format) || shape[0] <= 0 || shape[1] <= 0) {
    return invalid("dtype, rank, or channel count does not match the declared format");
  }
  const bool nv12 = desc.format == ImageFormat::kNV12;
  if (nv12 && (shape.size() != 2 || shape[0] % 3 != 0)) {
    return invalid("NV12 requires uint8 {H + H/2, W} storage");
  }
  const auto rows = static_cast<size_t>(shape[0]);
  const auto height = nv12 ? rows / 3 * 2 : rows;
  const auto width = static_cast<size_t>(shape[1]);
  if (width > desc.max_width || height > desc.max_height) {
    return invalid("visible dimensions exceed the declared maximum");
  }
  const auto layout = backends::image_layout(
      desc.format, static_cast<uint32_t>(width), static_cast<uint32_t>(height));
  if (!layout) {
    return invalid("invalid extent or subsampling alignment");
  }
  size_t pitch = layout->row_bytes[0];
  if (!strides.empty()) {
    // Inner strides must be packed, but only where they can reach an address: an extent-1 axis
    // is always indexed at 0, and producers (NumPy, CuPy, Torch) routinely leave a 0 or stale
    // stride there. The row stride is consumed as the pitch below, so it is always checked.
    const auto bpp = static_cast<std::int64_t>(backends::source_pixel_bytes(desc.format));
    const bool width_packed = shape[1] == 1 || strides[1] == bpp;
    const bool channels_packed =
        shape.size() != 3 || shape[2] == 1 || strides[2] == tensor.dtype().bits / 8;
    if (strides.size() != shape.size() || strides[0] <= 0 || !width_packed || !channels_packed) {
      return invalid("image requires positive row pitch and packed pixels/channels");
    }
    pitch = static_cast<size_t>(strides[0]);
  }
  const auto span = backends::image_plane_span(layout->row_bytes[0], rows, pitch);
  const auto address = reinterpret_cast<std::uintptr_t>(tensor.data());
  if (!span || address == 0 || address > std::numeric_limits<std::uintptr_t>::max() - (*span - 1)) {
    return invalid("invalid data pointer or row span");
  }
  return TensorImageLayout{
      *layout, static_cast<uint32_t>(width), static_cast<uint32_t>(height), pitch, rows};
}

}  // namespace holoscan::visualization
