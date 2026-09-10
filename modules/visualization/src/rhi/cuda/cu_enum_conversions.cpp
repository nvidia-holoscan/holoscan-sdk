/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cu_enum_conversions.hpp"

namespace holoviz::rhi::cuda {

expected<CuArrayFormat> from_holoviz_format(Format format) {
  switch (format) {
    case Format::kR8Unorm:
      return CuArrayFormat{CU_AD_FORMAT_UNSIGNED_INT8, 1};
    case Format::kR8G8B8A8Unorm:
    case Format::kR8G8B8A8Srgb:
      return CuArrayFormat{CU_AD_FORMAT_UNSIGNED_INT8, 4};
    case Format::kR16G16B16A16Sfloat:
      return CuArrayFormat{CU_AD_FORMAT_HALF, 4};
    case Format::kR32Uint:
      return CuArrayFormat{CU_AD_FORMAT_UNSIGNED_INT32, 1};
    case Format::kR32Sfloat:
      return CuArrayFormat{CU_AD_FORMAT_FLOAT, 1};
    case Format::kR32G32Sfloat:
      return CuArrayFormat{CU_AD_FORMAT_FLOAT, 2};
    case Format::kR32G32B32A32Sfloat:
      return CuArrayFormat{CU_AD_FORMAT_FLOAT, 4};

    // No CUDA array equivalent: BGRA, 10-bit packed, 3-channel float, depth.
    case Format::kB8G8R8A8Unorm:
    case Format::kA2B10G10R10Unorm:
    case Format::kR32G32B32Sfloat:
    case Format::kD32Sfloat:
    default:
      return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                   .message = "[holoviz] Format has no CUDA array equivalent."});
  }
}

}  // namespace holoviz::rhi::cuda
