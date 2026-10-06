/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "text_source.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

#include <cuda_runtime_api.h>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/schema_identity.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

namespace text_layer {

namespace {

constexpr dim3 kBlock{16U, 16U, 1U};
/// Frames per animation cycle: the colour field drifts once and the disc orbits once.
constexpr std::uint32_t kCycleFrames = 600U;
constexpr float kPi = 3.14159265358979F;

/// Where the bright disc is on frame @p frame_index, normalized to the frame, origin top-left:
/// an ellipse around the centre. Shared by the kernel and the label that follows the disc.
__host__ __device__ inline void disc_center(std::uint32_t frame_index, float& cx, float& cy) {
  const float phase = static_cast<float>(frame_index % kCycleFrames) / kCycleFrames;
  const float angle = phase * 2.0F * kPi;
  cx = 0.5F + 0.32F * cosf(angle);
  cy = 0.5F + 0.30F * sinf(angle);
}

/// Render one animated RGBA8 frame: a colour field whose hue drifts along the diagonal, dimmed
/// so text stays legible on top of it, with a bright disc orbiting the centre.
__global__ void render_pattern(std::uint8_t* dst, std::uint32_t width, std::uint32_t height,
                               std::uint32_t frame_index) {
  const std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }
  const float u = static_cast<float>(x) / static_cast<float>(width);
  const float v = static_cast<float>(y) / static_cast<float>(height);
  const float phase = static_cast<float>(frame_index % kCycleFrames) / kCycleFrames;
  const float t = (u + v) * 0.5F + phase;
  float r = 0.5F + 0.5F * cosf(2.0F * kPi * t);
  float g = 0.5F + 0.5F * cosf(2.0F * kPi * (t + 1.0F / 3.0F));
  float b = 0.5F + 0.5F * cosf(2.0F * kPi * (t + 2.0F / 3.0F));
  r *= 0.45F;
  g *= 0.45F;
  b *= 0.45F;

  float cx = 0.0F;
  float cy = 0.0F;
  disc_center(frame_index, cx, cy);
  const float aspect = static_cast<float>(width) / static_cast<float>(height);
  const float dx = (u - cx) * aspect;
  const float dy = v - cy;
  if (dx * dx + dy * dy < 0.06F * 0.06F) {
    r = g = b = 1.0F;
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

/// Write one text record row: the header, then the text truncated to kMaxChars, NUL-padded.
void write_run(std::uint8_t* row, const holoscan::visualization::TextRunHeader& header,
               std::string_view text) {
  std::memcpy(row, &header, sizeof(header));
  std::uint8_t* chars = row + holoscan::visualization::kTextRunHeaderBytes;
  std::memset(chars, 0, kMaxChars);
  std::memcpy(chars, text.data(), std::min<std::size_t>(text.size(), kMaxChars));
}

/// Packed 0xAABBGGRR from 8-bit channels (the text layer's colour convention).
constexpr std::uint32_t rgba(std::uint32_t r, std::uint32_t g, std::uint32_t b,
                             std::uint32_t a = 255U) {
  return (a << 24U) | (b << 16U) | (g << 8U) | r;
}

}  // namespace

TextDemoSource::TextDemoSource(std::uint32_t width, std::uint32_t height,
                               std::chrono::nanoseconds period, std::uint64_t frame_limit)
    : width_(width), height_(height), period_(period), frame_limit_(frame_limit) {}

void TextDemoSource::setup(holoscan::OperatorSpec& spec) {
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
  // HolovizOp's text ports expect a rank-2 uint8 host tensor of fixed-width records.
  spec.output(labels, "labels")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                             .dtype = kRecordDtype,
                             .rank = 2U},
          .bounds = holoscan::tensor_bounds(static_cast<std::uint64_t>(kMaxRuns) * kRowBytes),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      });
}

holoscan::Contract TextDemoSource::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = period_});
  if (frame_limit_ != 0U) {
    result.complete_after(holoscan::ComputeCalls{.limit = frame_limit_});
  }
  return result;
}

holoscan::expected<void, holoscan::Error> TextDemoSource::compute(
    holoscan::ExecutionContext& context) {
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kNotSupported,
                        "TextDemoSource: missing Runtime CUDA compute stream resource"});
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

  // Labels: one TextRunHeader + NUL-padded text per row (see TextLayerDesc). The layer is
  // declared with normalized_coordinates, so positions are fractions of the frame with the origin
  // top-left (the default unit would be framebuffer pixels); sizes are in framebuffer pixels.
  const std::array<std::int64_t, 2> record_shape{kMaxRuns, static_cast<std::int64_t>(kRowBytes)};
  auto records = labels.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = record_shape, .dtype = kRecordDtype});
  if (!records) {
    return holoscan::make_unexpected(std::move(records).error());
  }
  auto filled = records->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& guard)
                                                   -> holoscan::expected<void, holoscan::Error> {
    auto* rows = guard.data_as<std::uint8_t>();
    if (rows == nullptr || guard.byte_size() < kMaxRuns * kRowBytes) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure, "TextDemoSource: text record loan is too small"});
    }
    using holoscan::visualization::TextRunHeader;
    std::array<char, kMaxChars + 1U> text{};

    write_run(rows + 0U * kRowBytes,
              TextRunHeader{.x = 0.03F, .y = 0.04F, .size_px = 30.0F, .color = rgba(255, 255, 255)},
              "HolovizOp text layer");

    std::snprintf(text.data(), text.size(), "frame %u", frame_index);
    write_run(rows + 1U * kRowBytes,
              TextRunHeader{.x = 0.03F, .y = 0.12F, .size_px = 18.0F, .color = rgba(212, 255, 0)},
              text.data());

    const double elapsed_s =
        static_cast<double>(frame_index) * std::chrono::duration<double>(period_).count();
    std::snprintf(text.data(), text.size(), "t = %7.2f s", elapsed_s);
    write_run(rows + 2U * kRowBytes,
              TextRunHeader{.x = 0.03F, .y = 0.17F, .size_px = 18.0F, .color = rgba(212, 255, 0)},
              text.data());

    write_run(
        rows + 3U * kRowBytes,
        TextRunHeader{.x = 0.03F, .y = 0.92F, .size_px = 14.0F, .color = rgba(255, 255, 255, 160)},
        "uint8 {N, 16 + L} records, host memory");

    // The label follows the disc: just to its right, vertically centred on it.
    float cx = 0.0F;
    float cy = 0.0F;
    disc_center(frame_index, cx, cy);
    write_run(rows + 4U * kRowBytes,
              TextRunHeader{
                  .x = cx + 0.07F, .y = cy - 0.03F, .size_px = 22.0F, .color = rgba(255, 96, 0)},
              "<- disc");

    // Row 5 stays blank: HolovizOp skips rows whose text is empty.
    std::memset(rows + 5U * kRowBytes, 0, kRowBytes);
    return {};
  });
  if (!filled) {
    return holoscan::make_unexpected(std::move(filled).error());
  }

  // Publish the labels first. The image port is HolovizOp's trigger: a frame is rendered as soon
  // as an image sample is admitted, and a text layer whose sample has not arrived by then is
  // skipped for that frame (only image layers hold their last sample). Emitting the records
  // before the image guarantees they are already there when the trigger fires.
  const holoscan::EmitOptions options{.capture_time = context.activation_time()};
  if (auto emitted = labels.emit(std::move(*records), options); !emitted) {
    return emitted;
  }
  // emit_tensor() ignores the descriptor's own `data` field in favor of the loan, so only the
  // semantic fields need to be set; HolovizOp validates them against the declared image layer.
  holoscan::schema::ImageT descriptor{};
  descriptor.height = static_cast<std::int32_t>(height_);
  descriptor.width = static_cast<std::int32_t>(width_);
  descriptor.encoding = holoscan::schema::ImageEncoding_RGBA8;
  descriptor.color_space = holoscan::schema::ColorSpace_UNSPECIFIED;
  return frame.emit_tensor(std::move(*image), descriptor, options);
}

}  // namespace text_layer
