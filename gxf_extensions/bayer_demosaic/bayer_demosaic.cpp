/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <cuda_runtime.h>

#include <iostream>

#include "bayer_demosaic.hpp"


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

namespace nvidia::holoscan {

gxf_result_t BayerDemosaic::registerInterface(nvidia::gxf::Registrar* registrar) {
  nvidia::gxf::Expected<void> result;
  result &= registrar->parameter(
    receiver_, "receiver", "Entity receiver",
    "Receiver channel");
  result &= registrar->parameter(
    transmitter_, "transmitter", "Entity transmitter",
    "Transmitter channel");
  result &= registrar->parameter(in_tensor_name_, "in_tensor_name", "InputTensorName",
    "Name of the input tensor.", std::string(""));
  result &= registrar->parameter(out_tensor_name_, "out_tensor_name", "OutputTensorName",
    "Name of the output tensor.", std::string(""));
  result &= registrar->parameter(pool_, "pool", "Pool", "Pool to allocate the output message.");
  result &= registrar->parameter(cuda_stream_pool_, "cuda_stream_pool", "CUDA Stream Pool",
                                                    "CUDA Stream pool to create CUDA streams.");
  result &= registrar->parameter(bayer_interp_mode_, "interpolation_mode",
    "Interpolation used for demosaicing",
    "The interpolation model to be used for demosaicing (default UNDEFINED). Values available at:"
    "https://docs.nvidia.com/cuda/npp/group__typedefs__npp.html#ga2b58ebd329141d560aa4367f1708f191",
     0);
  result &= registrar->parameter(bayer_grid_pos_, "bayer_grid_pos", "Bayer grid position",
    "The Bayer grid position (default GBRG). Values available at:"
    "https://docs.nvidia.com/cuda/npp/group__typedefs__npp.html#ga5597309d6766fb2dffe155990d915ecb",
     2);
  result &= registrar->parameter(generate_alpha_, "generate_alpha", "Generate alpha channel",
    "Generate alpha channel.", false);
  result &= registrar->parameter(alpha_value_, "alpha_value", "Alpha value to be generated",
    "Alpha value to be generated if `generate_alpha` is set to `true` (default `255`).", 255);
  return nvidia::gxf::ToResultCode(result);
}

gxf_result_t BayerDemosaic::initialize() {
  // create a CUDA stream from the CUDA stream pool
  if (!cuda_stream_) {
    auto maybe_stream = cuda_stream_pool_.get()->allocateStream();
    if (!maybe_stream) {
      GXF_LOG_ERROR("Failed to allocate CUDA stream");
      return gxf::ToResultCode(maybe_stream);
    }
    cuda_stream_ = std::move(maybe_stream.value());
  }

  // assign the CUDA stream to the NPP stream context
  auto nppStatus = nppGetStreamContext(&npp_stream_ctx_);
  if (NPP_SUCCESS != nppStatus) {
    GXF_LOG_ERROR("Failed to get NPP cuda stream context");
    return GXF_FAILURE;
  }
  npp_stream_ctx_.hStream = cuda_stream_->stream().value();

  npp_bayer_interp_mode_ = static_cast<NppiInterpolationMode>(bayer_interp_mode_.get());
  npp_bayer_grid_pos_ = static_cast<NppiBayerGridPosition>(bayer_grid_pos_.get());

  return GXF_SUCCESS;
}

gxf_result_t BayerDemosaic::deinitialize() {
  return GXF_SUCCESS;
}

gxf_result_t BayerDemosaic::start() {
  return GXF_SUCCESS;
}

gxf_result_t BayerDemosaic::stop() {
  device_scratch_buffer_.freeBuffer();
  return GXF_SUCCESS;
}

gxf_result_t BayerDemosaic::tick() {
  // Process input message
  const auto in_message = receiver_->receive();
  if (!in_message || in_message.value().is_null()) {
    return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE;
  }

  void * input_data_ptr = nullptr;
  gxf::Shape in_shape{0, 0, 0};
  int32_t rows = 0;
  int32_t columns = 0;
  int16_t in_channels = 0;
  auto input_memory_type = gxf::MemoryStorageType::kHost;
  gxf::PrimitiveType element_type = gxf::PrimitiveType::kCustom;
  uint32_t element_size = gxf::PrimitiveTypeSize(element_type);

  // get tensor attached to message by the name defined in the parameters
  auto maybe_video = in_message.value().get<gxf::VideoBuffer>();
  if (maybe_video) {
    // Convert VideoBuffer to Tensor
    auto frame = maybe_video.value();

    // NOTE: VideoBuffer::moveToTensor() converts VideoBuffer instance to the Tensor instance
    // with an unexpected shape:  [width, height] or [width, height, num_planes].
    // And, if we use moveToTensor() to convert VideoBuffer to Tensor, we may lose the original
    // video buffer when the VideoBuffer instance is used in other places. For that reason, we
    // directly access internal data of VideoBuffer instance to access Tensor data.
    const auto& buffer_info = frame->video_frame_info();
    switch (buffer_info.color_format) {
      case gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY:
        element_type = gxf::PrimitiveType::kUnsigned8;
        break;
      default:
        GXF_LOG_ERROR("Unsupported input format: %d\n", buffer_info.color_format);
        return GXF_FAILURE;
    }

    // get input image metadata
    in_shape = gxf::Shape{static_cast<int32_t>(buffer_info.height),
                          static_cast<int32_t>(buffer_info.width), 1};

    rows = buffer_info.height;
    columns = buffer_info.width;
    in_channels = 1;
    element_size = gxf::PrimitiveTypeSize(element_type);
    input_memory_type = frame->storage_type();
    input_data_ptr = frame->pointer();

    // if the input tensor is not coming from device then move it to device
    if (input_memory_type != gxf::MemoryStorageType::kDevice) {
      size_t buffer_size = rows * columns * in_channels * element_size;

      if (buffer_size > device_scratch_buffer_.size()) {
        device_scratch_buffer_.resize(pool_, buffer_size, gxf::MemoryStorageType::kDevice);
        if (!device_scratch_buffer_.pointer()) {
          GXF_LOG_ERROR("Failed to allocate device scratch buffer (%d bytes)", buffer_size);
          return GXF_FAILURE;
        }
      }

      CUDA_TRY(cudaMemcpy(
        static_cast<void *>(device_scratch_buffer_.pointer()),
        static_cast<const void *>(frame->pointer()),
        buffer_size,
        cudaMemcpyHostToDevice));
      input_data_ptr = device_scratch_buffer_.pointer();
    }
  } else {
    const auto maybe_tensor = in_message.value().get<gxf::Tensor>(in_tensor_name_.get().c_str());
    if (!maybe_tensor) {
      GXF_LOG_ERROR("Tensor '%s' not found in message.\n", in_tensor_name_.get().c_str());
      return GXF_FAILURE;
    }

    gxf::Handle<gxf::Tensor> in_tensor = maybe_tensor.value();

    input_data_ptr = in_tensor->pointer();
    in_shape  = in_tensor->shape();
    rows = in_shape.dimension(0);
    columns = in_shape.dimension(1);
    in_channels = in_shape.dimension(2);
    element_type = in_tensor->element_type();
    element_size = gxf::PrimitiveTypeSize(element_type);
    input_memory_type = in_tensor->storage_type();
  }

  int16_t out_channels = 3 + generate_alpha_;

  if (element_type != gxf::PrimitiveType::kUnsigned8 &&
      element_type != gxf::PrimitiveType::kUnsigned16) {
    GXF_LOG_ERROR("Unexpected bytes in element representation %d (size %d)", element_type,
                                                                            element_size);
    return GXF_FAILURE;
  }

  // allocate tensors with names in the output message
  gxf::Expected<gxf::Entity> out_message = CreateTensorMap(
    context(),
    pool_,
    {{
      out_tensor_name_.get(),
      gxf::MemoryStorageType::kDevice,
      gxf::Shape{rows, columns, out_channels},
      element_type,
      0,
      gxf::ComputeTrivialStrides(gxf::Shape{rows, columns, out_channels}, element_size)
    }});

  if (!out_message) {
    GXF_LOG_ERROR("Failed to allocate tensors in output message");
    return gxf::ToResultCode(out_message);
  }

  // get the tensor of interest
  const auto maybe_output_tensor = out_message.value().get<gxf::Tensor>
                                                       (out_tensor_name_.get().c_str());
  if (!maybe_output_tensor) {
    GXF_LOG_ERROR("Failed to access output tensor with name `%s`", out_tensor_name_.get().c_str());
    return gxf::ToResultCode(maybe_output_tensor);
  }

  void* output_data_ptr = maybe_output_tensor.value()->pointer();

  // NPP to demosaic
  if (element_type == gxf::PrimitiveType::kUnsigned8) {
    if (generate_alpha_) {
      nppiCFAToRGBA_8u_C1AC4R_Ctx(
        static_cast<const Npp8u*>(input_data_ptr),
        columns * in_channels * element_size,
        {static_cast<int>(columns), static_cast<int>(rows)},
        {0, 0, static_cast<int>(columns), static_cast<int>(rows)},
        static_cast<Npp8u*>(output_data_ptr),
        columns * out_channels * element_size,
        npp_bayer_grid_pos_,
        npp_bayer_interp_mode_,
        alpha_value_,
        npp_stream_ctx_);
    } else {
      nppiCFAToRGB_8u_C1C3R_Ctx(
        static_cast<const Npp8u*>(input_data_ptr),
        columns * in_channels * element_size,
        {static_cast<int>(columns), static_cast<int>(rows)},
        {0, 0, static_cast<int>(columns), static_cast<int>(rows)},
        static_cast<Npp8u*>(output_data_ptr),
        columns * out_channels * element_size,
        npp_bayer_grid_pos_,
        npp_bayer_interp_mode_,
        npp_stream_ctx_);
    }
  } else {
    if (generate_alpha_) {
      nppiCFAToRGBA_16u_C1AC4R_Ctx(
        static_cast<const Npp16u*>(input_data_ptr),
        columns * in_channels * element_size,
        {static_cast<int>(columns), static_cast<int>(rows)},
        {0, 0, static_cast<int>(columns), static_cast<int>(rows)},
        static_cast<Npp16u*>(output_data_ptr),
        columns * out_channels * element_size,
        npp_bayer_grid_pos_,
        npp_bayer_interp_mode_,
        alpha_value_,
        npp_stream_ctx_);
    } else {
      nppiCFAToRGB_16u_C1C3R_Ctx(
        static_cast<const Npp16u*>(input_data_ptr),
        columns * in_channels * element_size,
        {static_cast<int>(columns), static_cast<int>(rows)},
        {0, 0, static_cast<int>(columns), static_cast<int>(rows)},
        static_cast<Npp16u*>(output_data_ptr),
        columns * out_channels * element_size,
        npp_bayer_grid_pos_,
        npp_bayer_interp_mode_,
        npp_stream_ctx_);
    }
  }

  const auto result = transmitter_->publish(out_message.value());
  return gxf::ToResultCode(result);
}

}  // namespace nvidia::holoscan
