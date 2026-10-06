// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <holoscan/sensor_io/image_metadata.hpp>

#include "image_metadata.hpp"

namespace holoscan::vision::detail {
namespace {

using namespace holoscan::schema;

bool dtype_is(::DLDataType dtype, std::uint8_t code, std::uint8_t bits) {
  return dtype.code == code && dtype.bits == bits && dtype.lanes == 1;
}

bool is_bayer(ImageEncoding encoding) {
  return encoding >= ImageEncoding_BAYER_RGGB && encoding <= ImageEncoding_BAYER_GBRG;
}

bool lossless_cast(::DLDataType input, ::DLDataType output) {
  if (input.code == kDLFloat) {
    return output.code == kDLFloat && output.bits >= input.bits;
  }
  if (input.code != kDLUInt && input.code != kDLInt) {
    return false;
  }
  if (input.code == output.code) {
    return output.bits >= input.bits;
  }
  if (input.code == kDLUInt && output.code == kDLInt) {
    return output.bits > input.bits;
  }
  if (output.code == kDLFloat) {
    const auto digits = output.bits == 32   ? std::numeric_limits<float>::digits
                        : output.bits == 64 ? std::numeric_limits<double>::digits
                                            : 0;
    return input.bits - (input.code == kDLInt ? 1 : 0) <= digits;
  }
  return false;
}

ImageEncoding bayer_phase(ImageEncoding encoding, std::int32_t x, std::int32_t y) {
  // Phase indices encode the red site's x/y parity, independently of enum numbering.
  constexpr ImageEncoding phases[]{ImageEncoding_BAYER_RGGB,
                                   ImageEncoding_BAYER_GRBG,
                                   ImageEncoding_BAYER_GBRG,
                                   ImageEncoding_BAYER_BGGR};
  for (std::size_t i = 0; i < 4; ++i) {
    if (phases[i] == encoding) {
      return phases[i ^ (static_cast<unsigned>(x) & 1U) ^ ((static_cast<unsigned>(y) & 1U) << 1U)];
    }
  }
  return encoding;
}

ImageEncoding converted_encoding(ImageEncoding encoding, ::DLDataType dtype) {
  const bool u8 = dtype_is(dtype, kDLUInt, 8);
  const bool u16 = dtype_is(dtype, kDLUInt, 16);
  const bool f32 = dtype_is(dtype, kDLFloat, 32);
  switch (encoding) {
    case ImageEncoding_MONO8:
    case ImageEncoding_MONO16:
    case ImageEncoding_MONO32:
    case ImageEncoding_MONO32F:
      if (u8) {
        return ImageEncoding_MONO8;
      }
      if (u16) {
        return ImageEncoding_MONO16;
      }
      if (dtype_is(dtype, kDLUInt, 32)) {
        return ImageEncoding_MONO32;
      }
      if (f32) {
        return ImageEncoding_MONO32F;
      }
      break;
    case ImageEncoding_RGB8:
    case ImageEncoding_RGB16:
    case ImageEncoding_RGB32F:
      if (u8) {
        return ImageEncoding_RGB8;
      }
      if (u16) {
        return ImageEncoding_RGB16;
      }
      if (f32) {
        return ImageEncoding_RGB32F;
      }
      break;
    case ImageEncoding_RGBA8:
    case ImageEncoding_RGBA16:
    case ImageEncoding_RGBA32F:
      if (u8) {
        return ImageEncoding_RGBA8;
      }
      if (u16) {
        return ImageEncoding_RGBA16;
      }
      if (f32) {
        return ImageEncoding_RGBA32F;
      }
      break;
    case ImageEncoding_BGR8:
    case ImageEncoding_BGRA8:
      if (u8) {
        return encoding;
      }
      break;
    case ImageEncoding_YUV32F:
      if (f32) {
        return encoding;
      }
      break;
    default:
      if (is_bayer(encoding)) {
        return encoding;
      }
      break;
  }
  return ImageEncoding_CUSTOM;
}

}  // namespace

expected<void, Error> validate_image_payload(const ImageT& image, const ImageTensorView& view) {
  if (image.width != view.width) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Vision Image width must match Tensor width"});
  }
  if (image.height != view.height) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Vision Image height must match Tensor height"});
  }
  const bool planar = view.image_layout == ImageTensorLayout::kPlanar;
  std::int64_t channels = 0;
  std::uint8_t code = kDLUInt;
  std::uint8_t bits = 8;
  bool encoding_planar = false;
  switch (image.encoding) {
    case ImageEncoding_CUSTOM:
      break;
    case ImageEncoding_R8_G8_B8:
    case ImageEncoding_B8_G8_R8:
      encoding_planar = true;
      channels = 3;
      break;
    case ImageEncoding_RGB8:
    case ImageEncoding_BGR8:
      channels = 3;
      break;
    case ImageEncoding_RGBA8:
    case ImageEncoding_BGRA8:
      channels = 4;
      break;
    case ImageEncoding_RGB16:
      channels = 3;
      bits = 16;
      break;
    case ImageEncoding_RGBA16:
      channels = 4;
      bits = 16;
      break;
    case ImageEncoding_MONO8:
      channels = 1;
      break;
    case ImageEncoding_MONO16:
      channels = 1;
      bits = 16;
      break;
    case ImageEncoding_MONO32:
      channels = 1;
      bits = 32;
      break;
    case ImageEncoding_MONO32F:
      channels = 1;
      bits = 32;
      code = kDLFloat;
      break;
    case ImageEncoding_RGB32F:
    case ImageEncoding_YUV32F:
      channels = 3;
      bits = 32;
      code = kDLFloat;
      break;
    case ImageEncoding_RGBA32F:
      channels = 4;
      bits = 32;
      code = kDLFloat;
      break;
    default:
      if (!is_bayer(image.encoding)) {
        // Packed words and chroma planes are not independently addressable scalar pixels.
        return make_unexpected(
            Error{ErrorCode::kNotSupported,
                  "Vision Image encoding must describe scalar Tensor pixels; packed and "
                  "chroma-subsampled encodings are unsupported"});
      }
      channels = 1;
      bits = view.dtype.bits;
      code = view.dtype.code;
      break;
  }
  if (channels != 0) {
    if (channels != view.channels) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                   "Vision Image encoding channels must match Tensor channels"});
    }
    if (!dtype_is(view.dtype, code, bits)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                   "Vision Image encoding dtype must match Tensor dtype"});
    }
    if (channels > 1 && planar != encoding_planar) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                   "Vision Image encoding layout must match Tensor layout"});
    }
  }
  if (!image.plane_layouts.empty() || encoding_planar) {
    const auto plane_count = planar ? static_cast<std::size_t>(view.channels) : 1U;
    if (image.plane_layouts.size() != plane_count) {
      return make_unexpected(Error{
          ErrorCode::kInvalidArgument,
          "Vision Image plane_layouts count must match Tensor layout: one per planar channel, "
          "otherwise one"});
    }
    const auto channel_axis = view.rank == 3 ? 0U : 1U;
    const auto row_axis = planar ? channel_axis + 1 : channel_axis;
    const auto row_stride = static_cast<std::uint64_t>(view.byte_strides[row_axis]);
    const auto plane_stride = planar ? static_cast<std::uint64_t>(view.byte_strides[channel_axis])
                                     : row_stride * view.height;
    const auto pixel_bytes = (planar ? 1 : view.channels) * (view.dtype.bits / 8U);
    const auto active_span = row_stride * (view.height - 1) + view.width * pixel_bytes;
    for (std::size_t i = 0; i < image.plane_layouts.size(); ++i) {
      const auto& plane = image.plane_layouts[i];
      if (plane.offset_bytes() != i * plane_stride) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                     "Vision Image plane_layouts[" + std::to_string(i) +
                                         "].offset_bytes must match Tensor plane offset"});
      }
      if (plane.row_stride_bytes() != row_stride) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                     "Vision Image plane_layouts[" + std::to_string(i) +
                                         "].row_stride_bytes must match Tensor row stride"});
      }
      if (plane.size_bytes() < active_span || plane.size_bytes() > plane_stride) {
        return make_unexpected(Error{
            ErrorCode::kInvalidArgument,
            "Vision Image plane_layouts[" + std::to_string(i) +
                "].size_bytes must cover the active Tensor pixels without exceeding plane stride"});
      }
    }
  }
  return {};
}

expected<void, Error> update_image_metadata(ImageT& image, const ImageTensorView& input,
                                            const ImageTensorView& output,
                                            ImageMetadataTransform transform) {
  const bool dtype_changed =
      input.dtype.code != output.dtype.code || input.dtype.bits != output.dtype.bits;
  switch (transform.geometry) {
    case ImageGeometryChange::kCrop: {
      const auto x = static_cast<std::int64_t>(image.roi_offset_x) + transform.x;
      const auto y = static_cast<std::int64_t>(image.roi_offset_y) + transform.y;
      if (x > std::numeric_limits<std::int32_t>::max() ||
          y > std::numeric_limits<std::int32_t>::max()) {
        return make_unexpected(Error{ErrorCode::kArithmeticOverflow});
      }
      image.roi_offset_x = static_cast<std::int32_t>(x);
      image.roi_offset_y = static_cast<std::int32_t>(y);
      image.encoding = bayer_phase(image.encoding, transform.x, transform.y);
      break;
    }
    case ImageGeometryChange::kPad:
      // Synthetic padding outside the sensor cannot be represented by nonnegative sensor offsets.
      if (image.roi_offset_x < transform.x || image.roi_offset_y < transform.y) {
        image.roi_offset_x = 0;
        image.roi_offset_y = 0;
      } else {
        image.roi_offset_x -= transform.x;
        image.roi_offset_y -= transform.y;
      }
      if (input.width != output.width || input.height != output.height) {
        if (is_bayer(image.encoding)) {
          image.encoding = ImageEncoding_CUSTOM;
        }
      }
      break;
    case ImageGeometryChange::kResize:
    case ImageGeometryChange::kWarp:
      if (transform.geometry == ImageGeometryChange::kWarp || input.width != output.width ||
          input.height != output.height) {
        image.roi_offset_x = 0;
        image.roi_offset_y = 0;
        if (is_bayer(image.encoding)) {
          image.encoding = ImageEncoding_CUSTOM;
        }
      }
      break;
    default:
      break;
  }
  if (dtype_changed) {
    image.encoding = converted_encoding(image.encoding, output.dtype);
  }
  const bool unchanged_geometry = (transform.geometry == ImageGeometryChange::kResize ||
                                   transform.geometry == ImageGeometryChange::kPad) &&
                                  input.width == output.width && input.height == output.height;
  if (transform.values_changed || (!transform.preserves_precision && !unchanged_geometry) ||
      (dtype_changed && !lossless_cast(input.dtype, output.dtype))) {
    image.significant_bits = kUnknownImageSignificantBits;
  } else if (dtype_changed && image.significant_bits == 0) {
    // Zero describes the input container. A wider output does not add source precision.
    image.significant_bits = input.dtype.bits;
  }
  if (dtype_changed || transform.values_changed) {
    // Precision alone does not establish a pixel code range or valid-bit alignment.
    // For example, U8 -> F32 does not normalize to [0, 1].
    image.color_space = ColorSpace_UNSPECIFIED;
  }
  const bool planar = output.image_layout == ImageTensorLayout::kPlanar && output.channels > 1;
  if (input.image_layout != output.image_layout && output.channels > 1) {
    switch (image.encoding) {
      case ImageEncoding_RGB8:
        image.encoding = ImageEncoding_R8_G8_B8;
        break;
      case ImageEncoding_BGR8:
        image.encoding = ImageEncoding_B8_G8_R8;
        break;
      case ImageEncoding_R8_G8_B8:
        image.encoding = ImageEncoding_RGB8;
        break;
      case ImageEncoding_B8_G8_R8:
        image.encoding = ImageEncoding_BGR8;
        break;
      default:
        image.encoding = ImageEncoding_CUSTOM;
        break;
    }
  }
  image.width = static_cast<std::int32_t>(output.width);
  image.height = static_cast<std::int32_t>(output.height);
  image.plane_layouts.clear();
  if (planar) {
    image.plane_layouts.reserve(static_cast<std::size_t>(output.channels));
    const auto channel_axis = output.rank == 3 ? 0U : 1U;
    const auto plane_stride = static_cast<std::uint64_t>(output.byte_strides[channel_axis]);
    const auto row_stride = static_cast<std::uint32_t>(output.byte_strides[channel_axis + 1]);
    // A tensor span ends at its last active element, not at the end of the final pitch.
    const auto plane_span = static_cast<std::uint64_t>(row_stride) * (output.height - 1) +
                            output.width * (output.dtype.bits / 8U);
    for (std::int64_t i = 0; i < output.channels; ++i) {
      image.plane_layouts.emplace_back(i * plane_stride, row_stride, plane_span);
    }
  }
  return validate_image_payload(image, output);
}

}  // namespace holoscan::vision::detail
