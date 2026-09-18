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

namespace gray_scale {

/// Element type of the emitted frames: one unsigned 16-bit sample per pixel.
inline constexpr DLDataType kGrayDtype{kDLUInt, 16U, 1U};

/// @brief Clock-driven source of animated 16-bit grayscale frames rendered by a CUDA kernel.
///
/// Emits `{height, width}` uint16 CUDA tensors at a fixed period. The pattern is a smooth
/// diagonal ramp over the full 0..65535 range with a sweeping bright bar, so that a consumer
/// misreading the samples as 8-bit would show visible banding or garbage. The display path
/// itself presents through an 8-bit swapchain; the 16-bit precision is preserved through upload
/// and sampling, not on screen.
class GrayScaleSource final : public holoscan::Operator<> {
 public:
  /// @brief Construct a source.
  /// @param width Frame width in pixels.
  /// @param height Frame height in pixels.
  /// @param period Emission period; the pattern advances one step per emission.
  GrayScaleSource(std::uint32_t width, std::uint32_t height, std::chrono::nanoseconds period);

  /// @brief Declare the pooled frame output.
  void setup(holoscan::OperatorSpec& spec) override;

  /// @brief Tick at the configured period.
  [[nodiscard]] holoscan::Contract contract() const override;

  /// @brief Render one frame into a pooled tensor and publish it.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  /// Rendered grayscale frame, `{height, width}` uint16 on the CUDA device.
  holoscan::Output<holoscan::Tensor> frame;

 private:
  std::uint32_t width_;
  std::uint32_t height_;
  std::chrono::nanoseconds period_;
  std::uint32_t frame_index_{};
};

}  // namespace gray_scale
