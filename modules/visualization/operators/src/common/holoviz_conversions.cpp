// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "common/holoviz_conversions.hpp"

#include <cstdint>
#include <string>
#include <string_view>

#include <magic_enum/magic_enum.hpp>

namespace holoscan::visualization::common {

expected<holoviz::Format> to_holoviz_format(ImageFormat format) {
  // No default: a new ImageFormat enumerator must be mapped (or rejected) here explicitly.
  switch (format) {
    case ImageFormat::kR8Unorm:
      return holoviz::Format::kR8Unorm;
    case ImageFormat::kR8G8B8A8Unorm:
      return holoviz::Format::kR8G8B8A8Unorm;
    case ImageFormat::kB8G8R8A8Unorm:
      return holoviz::Format::kB8G8R8A8Unorm;
    case ImageFormat::kR16Unorm:
      return holoviz::Format::kR16Unorm;
    case ImageFormat::kR16G16B16A16Unorm:
      return holoviz::Format::kR16G16B16A16Unorm;
    case ImageFormat::kUnknown:
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] ImageFormat::kUnknown has no HoloViz format; it is the "
                           "uninitialized sentinel."});
    case ImageFormat::kR8G8B8Unorm:
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] kR8G8B8Unorm has no single HoloViz format (it needs the "
                           "buffer-fetch decode path)."});
    case ImageFormat::kNV12:
    case ImageFormat::kYUYV422:
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] YUV ImageFormats lower to multiple HoloViz planes inside "
                           "the render backend; they have no single HoloViz format."});
  }
  return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                               .message = std::string("[holoviz] Unhandled ImageFormat value ") +
                                          std::to_string(static_cast<uint32_t>(format)) + "."});
}

expected<ImageFormat> to_image_format(holoviz::Format format) {
  switch (format) {
    case holoviz::Format::kR8Unorm:
      return ImageFormat::kR8Unorm;
    case holoviz::Format::kR8G8B8A8Unorm:
      return ImageFormat::kR8G8B8A8Unorm;
    case holoviz::Format::kB8G8R8A8Unorm:
      return ImageFormat::kB8G8R8A8Unorm;
    case holoviz::Format::kR16Unorm:
      return ImageFormat::kR16Unorm;
    case holoviz::Format::kR16G16B16A16Unorm:
      return ImageFormat::kR16G16B16A16Unorm;
    default:
      return make_unexpected(Error{.code = ErrorCode::kFailure,
                                   .message = std::string("[holoviz] HoloViz format ") +
                                              std::string(magic_enum::enum_name(format)) +
                                              " is not representable as a public ImageFormat."});
  }
}

std::string_view image_format_name(ImageFormat format) {
  return magic_enum::enum_name(format);
}

holoviz::PresentMode to_holoviz_present_mode(PresentMode mode) {
  switch (mode) {
    case PresentMode::kAuto:
      return holoviz::PresentMode::kAuto;
    case PresentMode::kFIFO:
      return holoviz::PresentMode::kFIFO;
    case PresentMode::kFIFOLatestReady:
      return holoviz::PresentMode::kFIFOLatestReady;
    case PresentMode::kMailbox:
      return holoviz::PresentMode::kMailbox;
    case PresentMode::kImmediate:
      return holoviz::PresentMode::kImmediate;
  }
  return holoviz::PresentMode::kAuto;  // out-of-range input; unreachable for valid enumerators
}

PresentMode to_present_mode(holoviz::PresentMode mode) {
  switch (mode) {
    case holoviz::PresentMode::kAuto:
      return PresentMode::kAuto;
    case holoviz::PresentMode::kFIFO:
      return PresentMode::kFIFO;
    case holoviz::PresentMode::kFIFOLatestReady:
      return PresentMode::kFIFOLatestReady;
    case holoviz::PresentMode::kMailbox:
      return PresentMode::kMailbox;
    case holoviz::PresentMode::kImmediate:
      return PresentMode::kImmediate;
  }
  return PresentMode::kAuto;  // out-of-range input; unreachable for valid enumerators
}

}  // namespace holoscan::visualization::common
