/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pattern_source.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include <cuda_runtime_api.h>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/schema_identity.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

namespace cuda_sink {

namespace {

constexpr dim3 kBlock{16U, 16U, 1U};
constexpr float kTwoPi = 6.283185307F;
/// Animation period in frames: the disc completes one orbit and the hue field one rotation.
constexpr std::uint32_t kCycleFrames = 600U;

/// Position of the moving disc, normalized to the frame, for animation step @p frame_index.
/// Shared between the kernel (which draws it) and the overlay (which frames it).
__host__ __device__ inline void disc_center(std::uint32_t frame_index, float& cx, float& cy) {
  const float angle =
      kTwoPi * static_cast<float>(frame_index % kCycleFrames) / static_cast<float>(kCycleFrames);
  cx = 0.5F + 0.3F * cosf(angle);
  cy = 0.5F + 0.3F * sinf(angle);
}

/// Render one RGBA8 frame: a colour field whose hue rotates with time and varies across the
/// frame, plus a bright disc on the orbit disc_center() describes.
__global__ void render_pattern(std::uint8_t* dst, std::uint32_t width, std::uint32_t height,
                               std::uint32_t frame_index) {
  const std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const float u = (static_cast<float>(x) + 0.5F) / static_cast<float>(width);
  const float v = (static_cast<float>(y) + 0.5F) / static_cast<float>(height);
  const float phase =
      static_cast<float>(frame_index % kCycleFrames) / static_cast<float>(kCycleFrames);

  // Three phase-shifted sinusoids give a smooth, obviously animated colour field.
  const float t = kTwoPi * (u * 0.5F + v * 0.25F + phase);
  float r = 0.5F + 0.5F * sinf(t);
  float g = 0.5F + 0.5F * sinf(t + kTwoPi / 3.0F);
  float b = 0.5F + 0.5F * sinf(t + 2.0F * kTwoPi / 3.0F);

  float cx = 0.0F;
  float cy = 0.0F;
  disc_center(frame_index, cx, cy);
  const float aspect = static_cast<float>(width) / static_cast<float>(height);
  const float dx = (u - cx) * aspect;
  const float dy = v - cy;
  if (dx * dx + dy * dy < 0.06F * 0.06F) {
    r = g = b = 1.0F;  // the disc the overlay rectangle tracks
  }

  std::uint8_t* pixel = dst + (static_cast<std::size_t>(y) * width + x) * 4U;
  pixel[0] = static_cast<std::uint8_t>(r * 255.0F);
  pixel[1] = static_cast<std::uint8_t>(g * 255.0F);
  pixel[2] = static_cast<std::uint8_t>(b * 255.0F);
  pixel[3] = 255U;
}

[[nodiscard]] holoscan::expected<void, holoscan::Error> cuda_launch_result(
    cudaError_t result, std::string_view operation) {
  if (result == cudaSuccess) {
    return {};
  }
  std::string message{operation};
  message.append(": ");
  message.append(cudaGetErrorString(result));
  return holoscan::make_unexpected(
      holoscan::Error{holoscan::ErrorCode::kFailure, std::move(message)});
}

}  // namespace

PatternSource::PatternSource(std::uint32_t width, std::uint32_t height,
                             std::chrono::nanoseconds period)
    : width_(width), height_(height), period_(period) {}

void PatternSource::setup(holoscan::OperatorSpec& spec) {
  const std::uint64_t frame_bytes = static_cast<std::uint64_t>(width_) * height_ * 4U;
  spec.output(frame, "frame")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation = {.memory_kind = holoscan::MemoryKind::kCudaDevice,
                             .dtype = kRgbaDtype,
                             .rank = 3U},
          .bounds = holoscan::tensor_bounds(frame_bytes),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      })
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
  // HolovizOp's geometry ports expect a rank-1 float32 host tensor.
  spec.output(overlay, "overlay")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                             .dtype = kOverlayDtype,
                             .rank = 1U},
          .bounds = holoscan::tensor_bounds(kOverlayRectangles * 4U * sizeof(float)),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      });
}

holoscan::Contract PatternSource::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = period_});
  return result;
}

holoscan::expected<void, holoscan::Error> PatternSource::compute(
    holoscan::ExecutionContext& context) {
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kNotSupported,
                        "PatternSource: missing Runtime CUDA compute stream resource"});
  }
  const std::uint32_t frame_index = frame_index_++;

  // Image: a kernel writes straight into the pooled device tensor on the activation stream.
  const std::array<std::int64_t, 3> shape{
      static_cast<std::int64_t>(height_), static_cast<std::int64_t>(width_), 4};
  auto image =
      frame.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kRgbaDtype});
  if (!image) {
    return holoscan::make_unexpected(std::move(image).error());
  }
  auto writer = image->write(context.cuda_stream());
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  const dim3 grid{(width_ + kBlock.x - 1U) / kBlock.x, (height_ + kBlock.y - 1U) / kBlock.y, 1U};
  render_pattern<<<grid, kBlock, 0, stream>>>(
      static_cast<std::uint8_t*>(writer->data()), width_, height_, frame_index);
  if (auto launched = cuda_launch_result(cudaGetLastError(), "render_pattern launch"); !launched) {
    return launched;
  }
  if (auto committed = std::move(*writer).commit(); !committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }

  // Overlay: rectangles in HolovizOp's kRectangleList wire format (x0, y0, x1, y1 each,
  // normalized to the frame, origin top-left) — a box around the disc, a thin frame border, and a
  // bar that sweeps with the animation phase.
  const std::array<std::int64_t, 1> overlay_shape{kOverlayRectangles * 4};
  auto rects = overlay.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = overlay_shape, .dtype = kOverlayDtype});
  if (!rects) {
    return holoscan::make_unexpected(std::move(rects).error());
  }
  auto filled = rects->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& guard)
                                                 -> holoscan::expected<void, holoscan::Error> {
    auto* out = guard.data_as<float>();
    if (out == nullptr || guard.byte_size() < kOverlayRectangles * 4U * sizeof(float)) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure,
                                                       "PatternSource: overlay loan is too small"});
    }
    float cx = 0.0F;
    float cy = 0.0F;
    disc_center(frame_index, cx, cy);
    const float half_w = 0.06F * static_cast<float>(height_) / static_cast<float>(width_);
    const float half_h = 0.06F;
    // Box around the disc.
    out[0] = cx - half_w * 1.5F;
    out[1] = cy - half_h * 1.5F;
    out[2] = cx + half_w * 1.5F;
    out[3] = cy + half_h * 1.5F;
    // Frame border, inset a little so the line stays inside the render target.
    out[4] = 0.02F;
    out[5] = 0.02F;
    out[6] = 0.98F;
    out[7] = 0.98F;
    // Sweeping vertical bar.
    const float sweep =
        static_cast<float>(frame_index % kCycleFrames) / static_cast<float>(kCycleFrames);
    out[8] = sweep * 0.96F + 0.02F;
    out[9] = 0.05F;
    out[10] = out[8] + 0.02F;
    out[11] = 0.95F;
    return {};
  });
  if (!filled) {
    return holoscan::make_unexpected(std::move(filled).error());
  }

  // Publish the overlay first. The image port is the render operator's trigger: a frame is
  // rendered as soon as an image sample is admitted, and a geometry layer whose sample has not
  // arrived by then is skipped for that frame (only image layers hold their last sample). Emitting
  // the rectangles before the image guarantees they are already there when the trigger fires.
  const holoscan::EmitOptions options{.capture_time = context.activation_time()};
  if (auto emitted = overlay.emit(std::move(*rects), options); !emitted) {
    return emitted;
  }
  // emit_tensor() ignores the descriptor's own `data` field in favor of the loan, so only the
  // semantic fields need to be set here; HolovizOp validates them against the declared layer.
  holoscan::schema::ImageT descriptor{};
  descriptor.height = static_cast<std::int32_t>(height_);
  descriptor.width = static_cast<std::int32_t>(width_);
  descriptor.encoding = holoscan::schema::ImageEncoding_RGBA8;
  descriptor.color_space = holoscan::schema::ColorSpace_UNSPECIFIED;
  return frame.emit_tensor(std::move(*image), descriptor, options);
}

}  // namespace cuda_sink
