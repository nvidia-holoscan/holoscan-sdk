// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

#include <holoscan/visualization/image_format.hpp>

namespace holoscan::visualization::backends {

inline bool is_yuv(ImageFormat format) {
  return format == ImageFormat::kNV12 || format == ImageFormat::kYUYV422;
}

inline uint32_t source_pixel_bytes(ImageFormat format) {
  switch (format) {
    case ImageFormat::kR8Unorm:
    case ImageFormat::kNV12:
      return 1;
    case ImageFormat::kR16Unorm:
    case ImageFormat::kYUYV422:
      return 2;
    case ImageFormat::kR8G8B8Unorm:
      return 3;
    case ImageFormat::kR8G8B8A8Unorm:
    case ImageFormat::kB8G8R8A8Unorm:
      return 4;
    case ImageFormat::kR16G16B16A16Unorm:
      return 8;
    default:
      return 0;
  }
}

struct ImageLayout {
  uint32_t plane_count = 1;
  std::array<size_t, 2> row_bytes{};
  std::array<size_t, 2> rows{};
  std::array<size_t, 2> offsets{};
  size_t bytes = 0;
};

// Tight source storage. Check before multiplying, including the extra NV12 chroma rows.
inline std::optional<ImageLayout> image_layout(ImageFormat format, uint32_t width,
                                               uint32_t height) {
  const size_t bpp = source_pixel_bytes(format);
  if (!bpp || !width || !height || (is_yuv(format) && width % 2 != 0) ||
      (format == ImageFormat::kNV12 && height % 2 != 0)) {
    return std::nullopt;
  }
  const size_t rows = static_cast<size_t>(height) + (format == ImageFormat::kNV12 ? height / 2 : 0);
  if (width > std::numeric_limits<size_t>::max() / bpp / rows) {
    return std::nullopt;
  }
  ImageLayout result;
  result.row_bytes[0] = static_cast<size_t>(width) * bpp;
  result.rows[0] = height;
  result.bytes = result.row_bytes[0] * rows;
  if (format == ImageFormat::kNV12) {
    result.plane_count = 2;
    result.row_bytes[1] = width;
    result.rows[1] = height / 2;
    result.offsets[1] = result.row_bytes[0] * height;
  }
  return result;
}

// The last row need only own its active pixels, not its trailing padding.
inline std::optional<size_t> image_plane_span(size_t row_bytes, size_t rows, size_t pitch) {
  if (!rows || pitch < row_bytes ||
      (rows > 1 && pitch > (std::numeric_limits<size_t>::max() - row_bytes) / (rows - 1))) {
    return std::nullopt;
  }
  return (rows - 1) * pitch + row_bytes;
}

}  // namespace holoscan::visualization::backends
