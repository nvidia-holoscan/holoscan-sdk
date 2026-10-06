/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstddef>
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
#include <holoscan/visualization/holoviz_op.hpp>

namespace text_layer {

/// Element type of the emitted frames: RGBA, one byte per channel.
inline constexpr DLDataType kRgbaDtype{kDLUInt, 8U, 1U};
/// Element type of the emitted text records: raw bytes (HolovizOp's text wire format).
inline constexpr DLDataType kRecordDtype{kDLUInt, 8U, 1U};
/// Rows in every text sample (N). Unused rows are left blank and HolovizOp skips them.
inline constexpr std::uint32_t kMaxRuns = 6U;
/// Text bytes per row (L): a run is at most this many characters.
inline constexpr std::uint32_t kMaxChars = 40U;
/// Bytes per row: the 16-byte TextRunHeader followed by kMaxChars of UTF-8.
inline constexpr std::size_t kRowBytes = holoscan::visualization::kTextRunHeaderBytes + kMaxChars;
/// The text layer's per-frame glyph budget: every row full.
inline constexpr std::uint32_t kMaxGlyphs = kMaxRuns * kMaxChars;

/// @brief Clock-driven source of an animated RGBA8 image plus text labels describing it.
///
/// Every tick emits two samples for the same frame: a `schema::ImageT` frame (RGBA8-encoded,
/// `{height, width, 4}` uint8 CUDA tensor data, matching HolovizOp's image-kind ports) rendered
/// by a kernel (a slowly drifting colour field with a bright orbiting disc), and a
/// `{kMaxRuns, 16 + kMaxChars}` uint8 host tensor of fixed-width text records in HolovizOp's
/// text wire format (`TextRunHeader` + NUL-padded UTF-8 per row): a title, the frame counter,
/// the elapsed time, a legend, and a label that follows the disc.
class TextDemoSource final : public holoscan::Operator<> {
 public:
  /// @brief Construct a source.
  /// @param width Frame width in pixels.
  /// @param height Frame height in pixels.
  /// @param period Emission period; the animation advances one step per emission.
  /// @param frame_limit Compute calls after which the source completes naturally (the run then
  /// drains to \c kCompleted); zero leaves the source unbounded.
  TextDemoSource(std::uint32_t width, std::uint32_t height, std::chrono::nanoseconds period,
                 std::uint64_t frame_limit = 0U);

  /// @brief Declare the pooled image and text record outputs.
  void setup(holoscan::OperatorSpec& spec) override;

  /// @brief Tick at the configured period and, with a frame limit, complete after it.
  [[nodiscard]] holoscan::Contract contract() const override;

  /// @brief Render one frame and its labels into pooled tensors and publish both.
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  /// Rendered image: schema::ImageT (encoding RGBA8) wrapping a `{height, width, 4}` uint8
  /// CUDA-device Tensor.
  holoscan::Output<holoscan::schema::ImageT> frame;
  /// Text records for the same frame, `{kMaxRuns, 16 + kMaxChars}` uint8 on the host.
  holoscan::Output<holoscan::Tensor> labels;

 private:
  std::uint32_t width_;
  std::uint32_t height_;
  std::chrono::nanoseconds period_;
  std::uint64_t frame_limit_;
  std::uint32_t frame_index_{};
};

}  // namespace text_layer
