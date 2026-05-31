/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HOLOSCAN_OPERATORS_FORMAT_CONVERTER_FORMAT_CONVERTER_COMMON_HPP
#define HOLOSCAN_OPERATORS_FORMAT_CONVERTER_FORMAT_CONVERTER_COMMON_HPP

#include <cstdint>
#include <stdexcept>
#include <string>

#include "gxf/std/tensor.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan::ops {

enum class FormatDType {
  kUnknown,
  kRGB888,
  kRGBA8888,
  kUnsigned8,
  kFloat32,
  kYUV420,
  kNV12BT709HDTV,
  kNV12BT709CSC,
  kNV12BT601Full,
  kYUYV,
  kRGB161616,
  kRGBA16161616,
  kUnsigned16
};

enum class FormatConversionType {
  kUnknown,
  kNone,
  kUnsigned8ToFloat32,
  kFloat32ToUnsigned8,
  kRGB888ToRGBA8888,
  kRGBA8888ToRGB888,
  kRGBA8888ToFloat32,
  kRGB888ToYUV420,
  kYUV420ToRGBA8888,
  kYUV420ToRGB888,
  kNV12BT709HDTVToRGB888,
  kNV12BT709CSCToRGB888,
  kNV12BT601FullToRGB888,
  kYUYVToRGB888,
  kRGB161616ToRGB888,
  kRGBA16161616ToRGB888,
};

/// Parses dtype string labels used by FormatConverterOp and FormatConverterGpuResidentOp.
inline FormatDType toFormatDType(const std::string& str) {
  if (str == "rgb888") {
    return FormatDType::kRGB888;
  }
  if (str == "uint8") {
    return FormatDType::kUnsigned8;
  }
  if (str == "float32") {
    return FormatDType::kFloat32;
  }
  if (str == "rgba8888") {
    return FormatDType::kRGBA8888;
  }
  if (str == "yuv420") {
    return FormatDType::kYUV420;
  }
  if (str == "nv12") {  // alias for "nv12_bt709_hdtv" to retain backwards compatibility
    return FormatDType::kNV12BT709HDTV;
  }
  if (str == "nv12_bt709_hdtv") {
    return FormatDType::kNV12BT709HDTV;
  }
  if (str == "nv12_bt709_csc") {
    return FormatDType::kNV12BT709CSC;
  }
  if (str == "nv12_bt601_full") {
    return FormatDType::kNV12BT601Full;
  }
  if (str == "yuyv") {
    return FormatDType::kYUYV;
  }
  if (str == "rgb161616") {
    return FormatDType::kRGB161616;
  }
  if (str == "rgba16161616") {
    return FormatDType::kRGBA16161616;
  }
  return FormatDType::kUnknown;
}

inline constexpr FormatConversionType getFormatConversionType(FormatDType from, FormatDType to) {
  if (from != FormatDType::kUnknown && to != FormatDType::kUnknown && from == to) {
    return FormatConversionType::kNone;
  }
  if (from == FormatDType::kUnsigned8 && to == FormatDType::kFloat32) {
    return FormatConversionType::kUnsigned8ToFloat32;
  }
  if (from == FormatDType::kFloat32 && to == FormatDType::kUnsigned8) {
    return FormatConversionType::kFloat32ToUnsigned8;
  }
  if (from == FormatDType::kUnsigned8 && to == FormatDType::kRGBA8888) {
    return FormatConversionType::kRGB888ToRGBA8888;
  }
  if (from == FormatDType::kRGBA8888 && to == FormatDType::kUnsigned8) {
    return FormatConversionType::kRGBA8888ToRGB888;
  }
  if (from == FormatDType::kRGBA8888 && to == FormatDType::kFloat32) {
    return FormatConversionType::kRGBA8888ToFloat32;
  }
  if (from == FormatDType::kUnsigned8 && to == FormatDType::kYUV420) {
    return FormatConversionType::kRGB888ToYUV420;
  }
  if (from == FormatDType::kYUV420 && to == FormatDType::kRGBA8888) {
    return FormatConversionType::kYUV420ToRGBA8888;
  }
  if (from == FormatDType::kYUV420 && to == FormatDType::kUnsigned8) {
    return FormatConversionType::kYUV420ToRGB888;
  }
  if (from == FormatDType::kNV12BT601Full && to == FormatDType::kUnsigned8) {
    return FormatConversionType::kNV12BT601FullToRGB888;
  }
  if (from == FormatDType::kNV12BT709HDTV && to == FormatDType::kUnsigned8) {
    return FormatConversionType::kNV12BT709HDTVToRGB888;
  }
  if (from == FormatDType::kNV12BT709CSC && to == FormatDType::kUnsigned8) {
    return FormatConversionType::kNV12BT709CSCToRGB888;
  }
  if (from == FormatDType::kYUYV && to == FormatDType::kUnsigned8) {
    return FormatConversionType::kYUYVToRGB888;
  }
  if (from == FormatDType::kRGBA16161616 &&
      (to == FormatDType::kUnsigned8 || to == FormatDType::kRGB888)) {
    return FormatConversionType::kRGBA16161616ToRGB888;
  }
  if (from == FormatDType::kRGB161616 &&
      (to == FormatDType::kUnsigned8 || to == FormatDType::kRGB888)) {
    return FormatConversionType::kRGB161616ToRGB888;
  }
  return FormatConversionType::kUnknown;
}

inline constexpr FormatDType normalizeFormatDType(FormatDType dtype) {
  switch (dtype) {
    case FormatDType::kRGB888:
      return FormatDType::kUnsigned8;
    default:
      return dtype;
  }
}

inline constexpr uint32_t elementSizeFromFormatDType(FormatDType dtype) {
  switch (dtype) {
    case FormatDType::kRGB888:
    case FormatDType::kRGBA8888:
    case FormatDType::kUnsigned8:
    case FormatDType::kYUV420:
    case FormatDType::kNV12BT709HDTV:
    case FormatDType::kNV12BT709CSC:
    case FormatDType::kNV12BT601Full:
    case FormatDType::kYUYV:
      return sizeof(uint8_t);
    case FormatDType::kRGB161616:
    case FormatDType::kRGBA16161616:
    case FormatDType::kUnsigned16:
      return sizeof(uint16_t);
    case FormatDType::kFloat32:
      return sizeof(float);
    default:
      return 0;
  }
}

inline constexpr nvidia::gxf::PrimitiveType primitiveTypeFromFormatDType(FormatDType dtype) {
  switch (dtype) {
    case FormatDType::kRGB888:
    case FormatDType::kRGBA8888:
    case FormatDType::kUnsigned8:
    case FormatDType::kYUV420:
    case FormatDType::kNV12BT601Full:
    case FormatDType::kNV12BT709HDTV:
    case FormatDType::kNV12BT709CSC:
    case FormatDType::kYUYV:
      return nvidia::gxf::PrimitiveType::kUnsigned8;
    case FormatDType::kRGB161616:
    case FormatDType::kRGBA16161616:
      return nvidia::gxf::PrimitiveType::kUnsigned16;
    case FormatDType::kFloat32:
      return nvidia::gxf::PrimitiveType::kFloat32;
    default:
      return nvidia::gxf::PrimitiveType::kCustom;
  }
}

inline constexpr FormatDType FormatDTypeFromPrimitiveType(nvidia::gxf::PrimitiveType type) {
  switch (type) {
    case nvidia::gxf::PrimitiveType::kUnsigned8:
      return FormatDType::kUnsigned8;
    case nvidia::gxf::PrimitiveType::kUnsigned16:
      return FormatDType::kUnsigned16;
    case nvidia::gxf::PrimitiveType::kFloat32:
      return FormatDType::kFloat32;
    default:
      return FormatDType::kUnknown;
  }
}

inline gxf_result_t verifyFormatDTypeChannels(FormatDType dtype, int channel_count) {
  switch (dtype) {
    case FormatDType::kRGB161616:
      if (channel_count != 3) {
        HOLOSCAN_LOG_ERROR("Invalid channel count for RGB161616 {} != 3\n", channel_count);
        return GXF_FAILURE;
      }
      break;
    case FormatDType::kRGB888:
      if (channel_count != 3) {
        HOLOSCAN_LOG_ERROR("Invalid channel count for RGB888 {} != 3\n", channel_count);
        return GXF_FAILURE;
      }
      break;
    case FormatDType::kRGBA16161616:
      if (channel_count != 4) {
        HOLOSCAN_LOG_ERROR("Invalid channel count for RGBA16161616 {} != 4\n", channel_count);
        return GXF_FAILURE;
      }
      break;
    case FormatDType::kRGBA8888:
      if (channel_count != 4) {
        HOLOSCAN_LOG_ERROR("Invalid channel count for RGBA8888 {} != 4\n", channel_count);
        return GXF_FAILURE;
      }
      break;
    default:
      break;
  }
  return GXF_SUCCESS;
}

inline void verifyFormatDTypeChannelsOrThrow(FormatDType dtype, int channel_count) {
  switch (dtype) {
    case FormatDType::kRGB161616:
      if (channel_count != 3) {
        throw std::runtime_error(
            fmt::format("Invalid channel count for RGB161616 {} != 3", channel_count));
      }
      break;
    case FormatDType::kRGB888:
      if (channel_count != 3) {
        throw std::runtime_error(
            fmt::format("Invalid channel count for RGB888 {} != 3", channel_count));
      }
      break;
    case FormatDType::kRGBA16161616:
      if (channel_count != 4) {
        throw std::runtime_error(
            fmt::format("Invalid channel count for RGBA16161616 {} != 4", channel_count));
      }
      break;
    case FormatDType::kRGBA8888:
      if (channel_count != 4) {
        throw std::runtime_error(
            fmt::format("Invalid channel count for RGBA8888 {} != 4", channel_count));
      }
      break;
    default:
      break;
  }
}

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_FORMAT_CONVERTER_FORMAT_CONVERTER_COMMON_HPP */
