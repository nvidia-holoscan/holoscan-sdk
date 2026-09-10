// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "backends/format_conversions.hpp"

#include <string>

#include <magic_enum/magic_enum.hpp>

namespace holoviz::backends {

expected<rhi::Format> to_rhi_format(ImageFormat format) {
  // No default: a new ImageFormat enumerator must be mapped (or rejected) here explicitly.
  switch (format) {
    case ImageFormat::kR8Unorm:
      return rhi::Format::kR8Unorm;
    case ImageFormat::kR8G8B8A8Unorm:
      return rhi::Format::kR8G8B8A8Unorm;
    case ImageFormat::kB8G8R8A8Unorm:
      return rhi::Format::kB8G8R8A8Unorm;
    case ImageFormat::kUnknown:
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] ImageFormat::kUnknown has no RHI format; it is the "
                           "uninitialized sentinel."});
    case ImageFormat::kR8G8B8Unorm:
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] kR8G8B8Unorm has no single RHI format (it needs the "
                           "buffer-fetch decode path)."});
    case ImageFormat::kNV12:
    case ImageFormat::kYUYV422:
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] YUV ImageFormats lower to multiple RHI planes inside "
                           "the render backend; they have no single RHI format."});
  }
  return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                               .message = std::string("[holoviz] Unhandled ImageFormat value ") +
                                          std::to_string(static_cast<uint32_t>(format)) + "."});
}

expected<ImageFormat> to_image_format(rhi::Format format) {
  switch (format) {
    case rhi::Format::kR8Unorm:
      return ImageFormat::kR8Unorm;
    case rhi::Format::kR8G8B8A8Unorm:
      return ImageFormat::kR8G8B8A8Unorm;
    case rhi::Format::kB8G8R8A8Unorm:
      return ImageFormat::kB8G8R8A8Unorm;
    default:
      return make_unexpected(Error{.code = ErrorCode::kFailure,
                                   .message = std::string("[holoviz] RHI format ") +
                                              std::string(magic_enum::enum_name(format)) +
                                              " is not representable as a public ImageFormat."});
  }
}

}  // namespace holoviz::backends
