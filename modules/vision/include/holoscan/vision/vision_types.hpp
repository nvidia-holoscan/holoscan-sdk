// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace holoscan::vision {

/// @brief Maximum batch dimension supported by Vision operator kernels.
inline constexpr std::uint32_t kMaximumBatchSize = 65'535U;

/// @brief Select an image interpolation algorithm.
enum class InterpolationMode : std::uint8_t {
  kNearest = 0,  ///< Nearest-neighbor interpolation.
  kLinear = 1,   ///< Bilinear interpolation.
  kCubic = 2,    ///< Bicubic interpolation.
  kArea = 3,     ///< Area interpolation.
};

/// @brief Select how samples outside an image are generated.
enum class BorderMode : std::uint8_t {
  kConstant = 0,    ///< Fill samples with a configured constant value.
  kReplicate = 1,   ///< Replicate the nearest edge sample.
  kReflect = 2,     ///< Reflect samples while repeating the edge sample.
  kWrap = 3,        ///< Wrap samples from the opposite image edge.
  kReflect101 = 4,  ///< Reflect samples without repeating the edge sample.
};

/// @brief Select how a remap tensor encodes source coordinates.
enum class RemapMapValueMode : std::uint8_t {
  kAbsolute = 0,            ///< Store absolute source pixel coordinates.
  kAbsoluteNormalized = 1,  ///< Store normalized absolute source coordinates.
  kRelativeNormalized = 2,  ///< Store normalized offsets from destination coordinates.
};

/// @brief Select how image frames are synchronized with coordinate maps.
enum class RemapMapSynchronizationMode : std::uint8_t {
  kLatest = 0,  ///< Reuse the newest causal map within the configured maximum age.
  kExact = 1,   ///< Require image and map tensors with identical capture times.
};

/// @brief Select a scalar element type for an operator output.
///
/// Each enumerator maps to a scalar DLPack dtype with one lane: \c kU8 to
/// `(kDLUInt, 8, 1)`, \c kS8 to `(kDLInt, 8, 1)`, \c kU16 to
/// `(kDLUInt, 16, 1)`, \c kS16 to `(kDLInt, 16, 1)`, \c kS32 to
/// `(kDLInt, 32, 1)`, \c kF32 to `(kDLFloat, 32, 1)`, and \c kF64 to
/// `(kDLFloat, 64, 1)`.
enum class VisionElementType : std::uint8_t {
  kU8 = 0,   ///< DLPack `(kDLUInt, 8, 1)` unsigned 8-bit integer.
  kS8 = 1,   ///< DLPack `(kDLInt, 8, 1)` signed 8-bit integer.
  kU16 = 2,  ///< DLPack `(kDLUInt, 16, 1)` unsigned 16-bit integer.
  kS16 = 3,  ///< DLPack `(kDLInt, 16, 1)` signed 16-bit integer.
  kS32 = 4,  ///< DLPack `(kDLInt, 32, 1)` signed 32-bit integer.
  kF32 = 5,  ///< DLPack `(kDLFloat, 32, 1)` 32-bit floating point.
  kF64 = 6,  ///< DLPack `(kDLFloat, 64, 1)` 64-bit floating point.
};

/// @brief Select floating-point to integer conversion rounding.
enum class RoundingMode : std::uint8_t {
  kNearest = 0,   ///< Round to the nearest representable integer.
  kTruncate = 1,  ///< Round toward zero.
};

/// @brief Select the image layout conversion performed by ReformatOp.
enum class ReformatMode : std::uint8_t {
  kInterleavedToPlanar = 0,  ///< Convert HWC/NHWC to CHW/NCHW.
  kPlanarToInterleaved = 1,  ///< Convert CHW/NCHW to HWC/NHWC.
};

namespace detail {

[[nodiscard]] inline bool finite_float(const float& value) noexcept {
  return std::isfinite(value);
}

[[nodiscard]] inline bool finite_double(const double& value) noexcept {
  return std::isfinite(value);
}

[[nodiscard]] inline bool nonempty_finite_float_vector(const std::vector<float>& values) noexcept {
  if (values.empty()) {
    return false;
  }
  for (const float value : values) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

}  // namespace detail

}  // namespace holoscan::vision
