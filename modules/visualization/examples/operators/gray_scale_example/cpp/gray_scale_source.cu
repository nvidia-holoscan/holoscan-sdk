/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "gray_scale_source.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include <cuda_runtime_api.h>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/schema_identity.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

namespace gray_scale {

namespace {

constexpr dim3 kBlock{16U, 16U, 1U};

/// Render one animated 16-bit grayscale frame.
///
/// The base is a diagonal ramp that spans the whole 16-bit range across the frame diagonal and
/// scrolls slowly with time; on a 640-wide frame neighbouring pixels differ by ~60 codes, far
/// below one 8-bit step. The full precision survives upload and sampling; the window's BGRA8
/// swapchain quantizes the final displayed color, so this exercises 16-bit input interpretation
/// and normalization, not 16-bit display. A bright vertical bar sweeps left to right on top of it.
__global__ void render_gray(std::uint16_t* dst, std::uint32_t width, std::uint32_t height,
                            std::uint32_t frame_index) {
  const std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const float span = static_cast<float>(width + height - 2U);
  const float phase = static_cast<float>(frame_index % 1024U) / 1024.0F;
  float t = (static_cast<float>(x + y) / span) + phase;
  t -= floorf(t);  // wrap into [0, 1)
  float value = t * 65535.0F;

  const std::uint32_t bar_x = (frame_index * 2U) % width;
  const std::uint32_t bar_half_width = width / 40U + 1U;
  const std::int32_t distance = static_cast<std::int32_t>(x) - static_cast<std::int32_t>(bar_x);
  if (distance >= -static_cast<std::int32_t>(bar_half_width) &&
      distance <= static_cast<std::int32_t>(bar_half_width)) {
    value = 65535.0F;
  }

  dst[static_cast<std::size_t>(y) * width + x] = static_cast<std::uint16_t>(value);
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

GrayScaleSource::GrayScaleSource(std::uint32_t width, std::uint32_t height,
                                 std::chrono::nanoseconds period, std::uint64_t frame_limit)
    : width_(width), height_(height), period_(period), frame_limit_(frame_limit) {}

void GrayScaleSource::setup(holoscan::OperatorSpec& spec) {
  const std::uint64_t frame_bytes =
      static_cast<std::uint64_t>(width_) * height_ * sizeof(std::uint16_t);
  spec.output(frame, "frame")
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation = {.memory_kind = holoscan::MemoryKind::kCudaDevice,
                             .dtype = kGrayDtype,
                             .rank = 2U},
          .bounds = holoscan::tensor_bounds(frame_bytes),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      })
      .payload_options(holoscan::PayloadOptions{
          .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
      });
}

holoscan::Contract GrayScaleSource::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = period_});
  if (frame_limit_ != 0U) {
    result.complete_after(holoscan::ComputeCalls{.limit = frame_limit_});
  }
  return result;
}

holoscan::expected<void, holoscan::Error> GrayScaleSource::compute(
    holoscan::ExecutionContext& context) {
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kNotSupported,
                        "GrayScaleSource: missing Runtime CUDA compute stream resource"});
  }

  const std::array<std::int64_t, 2> shape{static_cast<std::int64_t>(height_),
                                          static_cast<std::int64_t>(width_)};
  auto loan =
      frame.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kGrayDtype});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }

  const dim3 grid{(width_ + kBlock.x - 1U) / kBlock.x, (height_ + kBlock.y - 1U) / kBlock.y, 1U};
  render_gray<<<grid, kBlock, 0, stream>>>(
      static_cast<std::uint16_t*>(writer->data()), width_, height_, frame_index_++);
  if (auto launched = cuda_launch_result(cudaGetLastError(), "render_gray launch"); !launched) {
    return launched;
  }

  auto committed = std::move(*writer).commit();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }

  // emit_tensor() ignores the descriptor's own `data` field in favor of the loan, so only the
  // semantic fields need to be set here.
  holoscan::schema::ImageT descriptor{};
  descriptor.height = static_cast<std::int32_t>(height_);
  descriptor.width = static_cast<std::int32_t>(width_);
  descriptor.encoding = holoscan::schema::ImageEncoding_MONO16;
  descriptor.color_space = holoscan::schema::ColorSpace_UNSPECIFIED;
  return frame.emit_tensor(std::move(*loan),
                           descriptor,
                           holoscan::EmitOptions{.capture_time = context.activation_time()});
}

}  // namespace gray_scale
