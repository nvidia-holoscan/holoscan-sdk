/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "./video_buffer_mock.hpp"

#include <cuda_runtime.h>
#include <npp.h>

#include <utility>

#include "gxf/core/entity.hpp"
#include "gxf/core/expected.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/tensor.hpp"


#define CUDA_TRY(stmt)                                                                            \
  ({                                                                                              \
    cudaError_t _holoscan_cuda_err = stmt;                                                        \
    if (cudaSuccess != _holoscan_cuda_err) {                                                      \
      GXF_LOG_ERROR("CUDA Runtime call %s in line %d of file %s failed with '%s' (%d).\n", #stmt, \
                    __LINE__, __FILE__, cudaGetErrorString(_holoscan_cuda_err),                   \
                    _holoscan_cuda_err);                                                          \
    }                                                                                             \
    _holoscan_cuda_err;                                                                           \
  })

namespace nvidia {
namespace holoscan {
namespace mocks {

constexpr int32_t DEFAULT_SRC_WIDTH = 640;
constexpr int32_t DEFAULT_SRC_HEIGHT = 480;
constexpr int16_t DEFAULT_SRC_CHANNELS = 3;
constexpr uint8_t DEFAULT_SRC_BYTES_PER_PIXEL = 1;

gxf_result_t VideoBufferMock::start() {
  return GXF_SUCCESS;
}

gxf_result_t VideoBufferMock::stop() {
  buffer_.freeBuffer();
  return GXF_SUCCESS;
}

gxf_result_t VideoBufferMock::tick() {
  const int32_t rows = in_height_;
  const int32_t columns = in_width_;
  const int16_t channels = in_channels_;
  const uint8_t bytes_per_pixel = in_bytes_per_pixel_;

  if (buffer_.size() == 0) {
    buffer_.resize(pool_, columns * rows * sizeof(float), gxf::MemoryStorageType::kDevice);
  }

  // Pass the frame downstream.
  auto message = gxf::Entity::New(context());
  if (!message) {
    GXF_LOG_ERROR("Failed to allocate message");
    return GXF_FAILURE;
  }

  auto buffer = message.value().add<gxf::VideoBuffer>();
  if (!buffer) {
    GXF_LOG_ERROR("Failed to allocate video buffer");
    return GXF_FAILURE;
  }

  gxf::VideoTypeTraits<gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA> video_type;
  gxf::VideoFormatSize<gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA> color_format;
  auto color_planes = color_format.getDefaultColorPlanes(columns, rows);
  gxf::VideoBufferInfo info{static_cast<uint32_t>(columns), static_cast<uint32_t>(rows),
                            video_type.value, color_planes,
                            gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR};

  const auto buffer_ptr = buffer_.pointer();
  const int32_t row_step = channels * bytes_per_pixel * columns;

  nppiSet_8u_C4R(std::array<uint8_t, 4>{255, 0, 0, 255}.data(), buffer_ptr, row_step,
                 NppiSize{static_cast<int>(columns), static_cast<int>(rows / 3)});
  nppiSet_8u_C4R(std::array<uint8_t, 4>{0, 255, 0, 255}.data(), buffer_ptr + (row_step * rows / 3),
                 row_step, NppiSize{static_cast<int>(columns), static_cast<int>(rows / 3)});
  nppiSet_8u_C4R(std::array<uint8_t, 4>{0, 0, 255, 255}.data(),
                 buffer_ptr + (row_step * rows * 2 / 3), row_step,
                 NppiSize{static_cast<int>(columns), static_cast<int>(rows / 3)});

  auto storage_type = gxf::MemoryStorageType::kDevice;
  buffer.value()->wrapMemory(info, buffer_.size(), storage_type, buffer_.pointer(), nullptr);

  const auto result = out_->publish(std::move(message.value()));

  printf("count: %" PRIu64 "\n", ++count_);

  return gxf::ToResultCode(message);
}

gxf_result_t VideoBufferMock::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(in_width_, "in_width", "SourceWidth", "Width of the image.",
                                 DEFAULT_SRC_WIDTH);
  result &= registrar->parameter(in_height_, "in_height", "SourceHeight", "Height of the image.",
                                 DEFAULT_SRC_HEIGHT);
  result &= registrar->parameter(in_channels_, "in_channels", "SourceChannels",
                                 "Number of channels.", DEFAULT_SRC_CHANNELS);
  result &= registrar->parameter(in_bytes_per_pixel_,
                                 "in_bytes_per_pixel",
                                 "InputBytesPerPixel",
                                 "Number of bytes per pixel of the image.",
                                 DEFAULT_SRC_BYTES_PER_PIXEL);
  result &= registrar->parameter(out_tensor_name_, "out_tensor_name", "OutputTensorName",
                                 "Name of the output tensor.", std::string(""));

  result &= registrar->parameter(out_, "out", "Output", "Output channel.");
  result &= registrar->parameter(pool_, "pool", "Pool", "Pool to allocate the output message.");
  return gxf::ToResultCode(result);
}

}  // namespace mocks
}  // namespace holoscan
}  // namespace nvidia
