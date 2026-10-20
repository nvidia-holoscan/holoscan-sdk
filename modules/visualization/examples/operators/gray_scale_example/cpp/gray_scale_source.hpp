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

namespace gray_scale {

/// Element type of the emitted frames: one unsigned 16-bit sample per pixel.
inline constexpr DLDataType kGrayDtype{kDLUInt, 16U, 1U};

/// @brief Clock-driven source of animated 16-bit grayscale frames rendered by a CUDA kernel.
///
/// Emits `schema::ImageT` frames (MONO16-encoded, `{height, width}` uint16 CUDA tensor data) at
/// a fixed period, matching HolovizOp's image-kind ports (see the ImageT port alignment design).
/// The pattern is a smooth diagonal ramp over the full 0..65535 range with a sweeping bright bar,
/// so that a consumer misreading the samples as 8-bit would show visible banding or garbage. The
/// display path itself presents through an 8-bit swapchain; the 16-bit precision is preserved
/// through upload and sampling, not on screen.
class GrayScaleSource final : public holoscan::Operator<> {
 public:
  /// @brief Construct a source.
  /// @param width Frame width in pixels.
  /// @param height Frame height in pixels.
  /// @param period Emission period; the pattern advances one step per emission.
  /// @param frame_limit Compute calls after which the source completes naturally (the run then
  /// drains to \c kCompleted); zero leaves the source unbounded.
  GrayScaleSource(std::uint32_t width, std::uint32_t height, std::chrono::nanoseconds period,
                  std::uint64_t frame_limit = 0U);

  /// @brief Declare the pooled frame output.
  void setup(holoscan::OperatorSpec& spec) override;

  /// @brief Tick at the configured period and, with a frame limit, complete after it.
  [[nodiscard]] holoscan::Contract contract() const override;

  /// @brief Render one frame into a pooled tensor and publish it.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  /// Rendered grayscale frame: schema::ImageT (encoding MONO16) wrapping a `{height, width}`
  /// uint16 CUDA-device Tensor.
  holoscan::Output<holoscan::schema::ImageT> frame;

 private:
  std::uint32_t width_;
  std::uint32_t height_;
  std::chrono::nanoseconds period_;
  std::uint64_t frame_limit_;
  std::uint32_t frame_index_{};
};

}  // namespace gray_scale
