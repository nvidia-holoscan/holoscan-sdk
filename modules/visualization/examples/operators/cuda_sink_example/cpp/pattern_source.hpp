/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstdint>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace cuda_sink {

/// Element type of the emitted frames: RGBA, one byte per channel.
inline constexpr DLDataType kRgbaDtype{kDLUInt, 8U, 1U};
/// Element type of the emitted overlay: float32 rectangle corners.
inline constexpr DLDataType kOverlayDtype{kDLFloat, 32U, 1U};
/// Rectangles per overlay sample; each is four floats (x0, y0, x1, y1), normalized [0,1].
inline constexpr std::uint32_t kOverlayRectangles = 3U;

/// @brief Clock-driven source of an animated RGBA8 image plus a matching rectangle overlay.
///
/// Every tick emits two samples that describe the same frame: a `schema::ImageT` (RGBA8-encoded,
/// `{height, width, 4}` uint8 CUDA tensor data) rendered by a kernel (a slowly rotating colour
/// field with a bright moving disc), matching HolovizOp's image-kind ports, and a
/// `{kOverlayRectangles * 4}` float32 host tensor with rectangles that track the disc and the
/// frame border, in HolovizOp's geometry wire format (kRectangleList: x0, y0, x1, y1 per
/// rectangle, normalized to the frame, origin top-left).
class PatternSource final : public holoscan::Operator<> {
 public:
  /// @brief Construct a source.
  /// @param width Frame width in pixels.
  /// @param height Frame height in pixels.
  /// @param period Emission period; the animation advances one step per emission.
  PatternSource(std::uint32_t width, std::uint32_t height, std::chrono::nanoseconds period);

  /// @brief Declare the pooled image and overlay outputs.
  void setup(holoscan::OperatorSpec& spec) override;

  /// @brief Tick at the configured period.
  [[nodiscard]] holoscan::Contract contract() const override;

  /// @brief Render one frame and its overlay into pooled tensors and publish both.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  /// Rendered image: schema::ImageT (encoding RGBA8) wrapping a `{height, width, 4}` uint8
  /// CUDA-device Tensor.
  holoscan::Output<holoscan::schema::ImageT> frame;
  /// Rectangle overlay for the same frame, `{kOverlayRectangles * 4}` float32 on the host.
  holoscan::Output<holoscan::Tensor> overlay;

 private:
  std::uint32_t width_;
  std::uint32_t height_;
  std::chrono::nanoseconds period_;
  std::uint32_t frame_index_{};
};

}  // namespace cuda_sink
