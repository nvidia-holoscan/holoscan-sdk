/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/operators/bayer_demosaic/bayer_demosaic.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <utility>

#include "gxf/std/tensor.hpp"
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

#define CUDA_TRY(stmt)                                                                     \
  ({                                                                                       \
    cudaError_t _holoscan_cuda_err = stmt;                                                 \
    if (cudaSuccess != _holoscan_cuda_err) {                                               \
      GXF_LOG_ERROR("CUDA Runtime call %s in line %d of file %s failed with '%s' (%d).\n", \
                    #stmt,                                                                 \
                    __LINE__,                                                              \
                    __FILE__,                                                              \
                    cudaGetErrorString(_holoscan_cuda_err),                                \
                    _holoscan_cuda_err);                                                   \
    }                                                                                      \
    _holoscan_cuda_err;                                                                    \
  })

namespace holoscan::ops {

void BayerDemosaicOp::setup(OperatorSpec& spec) {
  auto& receiver = spec.input<gxf::Entity>("receiver");
  auto& transmitter = spec.output<gxf::Entity>("transmitter");

  spec.param(receiver_, "receiver", "Entity receiver", "Receiver channel", &receiver);
  spec.param(
      transmitter_, "transmitter", "Entity transmitter", "Transmitter channel", &transmitter);
  spec.param(in_tensor_name_,
             "in_tensor_name",
             "InputTensorName",
             "Name of the input tensor",
             std::string(""));
  spec.param(out_tensor_name_,
             "out_tensor_name",
             "OutputTensorName",
             "Name of the output tensor",
             std::string(""));
  spec.param(pool_, "pool", "Pool", "Pool to allocate the output message.");
  spec.param(
      bayer_interp_mode_,
      "interpolation_mode",
      "Interpolation used for demosaicing",
      "The interpolation model to be used for demosaicing (default: NPPI_INTER_UNDEFINED). Values "
      "available at: "
      "https://docs.nvidia.com/cuda/npp/nppdefs.html?highlight="
      "Two%20parameter%20cubic%20filter#c.NppiInterpolationMode",
      0);
  spec.param(bayer_grid_pos_,
             "bayer_grid_pos",
             "Bayer grid position",
             "The Bayer grid position (default: NPPI_BAYER_GBRG). Values available at: "
             "https://docs.nvidia.com/cuda/npp/nppdefs.html?highlight="
             "Two%20parameter%20cubic%20filter#c.NppiBayerGridPosition",
             2);
  spec.param(generate_alpha_,
             "generate_alpha",
             "Generate alpha channel",
             "Generate alpha channel.",
             false);
  spec.param(alpha_value_,
             "alpha_value",
             "Alpha value to be generated",
             "Alpha value to be generated if `generate_alpha` is set to `true` (default `255`).",
             255);
  cuda_stream_handler_.define_params(spec);
}

void BayerDemosaicOp::initialize() {
  Operator::initialize();

  npp_bayer_interp_mode_ = static_cast<NppiInterpolationMode>(bayer_interp_mode_.get());
  if (npp_bayer_interp_mode_ != NPPI_INTER_UNDEFINED) {
    // according to NPP docs only NPPI_INTER_UNDEFINED is supported for Bayer demosaic
    // https://docs.nvidia.com/cuda/archive/12.2.0/npp/group__image__color__debayer.html
    throw std::runtime_error(fmt::format("Unsupported bayer_interp_mode: {}. Must be 0",
                                         static_cast<int>(npp_bayer_interp_mode_)));
  }

  npp_bayer_grid_pos_ = static_cast<NppiBayerGridPosition>(bayer_grid_pos_.get());
}

void BayerDemosaicOp::stop() {
  device_scratch_buffer_.freeBuffer();
}

void BayerDemosaicOp::compute(InputContext& op_input, OutputContext& op_output,
                              ExecutionContext& context) {
  // Process input message
  auto maybe_message = op_input.receive<gxf::Entity>("receiver");
  if (!maybe_message || maybe_message.value().is_null()) {
    throw std::runtime_error("No message available");
  }
  auto in_message = maybe_message.value();

  // get the CUDA stream from the input message
  gxf_result_t stream_handler_result =
      cuda_stream_handler_.from_message(context.context(), in_message);
  if (stream_handler_result != GXF_SUCCESS) {
    throw std::runtime_error("Failed to get the CUDA stream from incoming messages");
  }

  // assign the CUDA stream to the NPP stream context
  npp_stream_ctx_.hStream = cuda_stream_handler_.get_cuda_stream(context.context());

  void* input_data_ptr = nullptr;
  nvidia::gxf::Shape in_shape{0, 0, 0};
  int32_t rows = 0;
  int32_t columns = 0;
  int16_t in_channels = 0;
  auto input_memory_type = nvidia::gxf::MemoryStorageType::kHost;
  nvidia::gxf::PrimitiveType element_type = nvidia::gxf::PrimitiveType::kCustom;
  uint32_t element_size = nvidia::gxf::PrimitiveTypeSize(element_type);

  // get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto pool =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), pool_->gxf_cid());

  // Get either the Tensor or VideoBuffer attached to the message
  bool is_video_buffer;
  nvidia::gxf::Handle<nvidia::gxf::VideoBuffer> video_buffer;
  try {
    video_buffer = holoscan::gxf::get_videobuffer(in_message);
    is_video_buffer = true;
  } catch (const std::runtime_error& r_) {
    HOLOSCAN_LOG_TRACE("Failed to read VideoBuffer with error: {}", std::string(r_.what()));
    is_video_buffer = false;
  }

  if (is_video_buffer) {
    // Convert VideoBuffer to Tensor
    auto frame = video_buffer.get();

    // NOTE: VideoBuffer::moveToTensor() converts VideoBuffer instance to the Tensor instance
    // with an unexpected shape:  [width, height] or [width, height, num_planes].
    // And, if we use moveToTensor() to convert VideoBuffer to Tensor, we may lose the original
    // video buffer when the VideoBuffer instance is used in other places. For that reason, we
    // directly access internal data of VideoBuffer instance to access Tensor data.
    const auto& buffer_info = frame->video_frame_info();
    switch (buffer_info.color_format) {
      case nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY:
        element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
        break;
      default:
        throw std::runtime_error(fmt::format("Unsupported input format: {}\n",
                                             static_cast<int>(buffer_info.color_format)));
    }

    // get input image metadata
    in_shape = nvidia::gxf::Shape{
        static_cast<int32_t>(buffer_info.height), static_cast<int32_t>(buffer_info.width), 1};

    rows = buffer_info.height;
    columns = buffer_info.width;
    in_channels = 1;
    element_size = nvidia::gxf::PrimitiveTypeSize(element_type);
    input_memory_type = frame->storage_type();
    input_data_ptr = frame->pointer();

    // if the input tensor is not coming from device then move it to device
    if (input_memory_type == nvidia::gxf::MemoryStorageType::kSystem) {
      size_t buffer_size = rows * columns * in_channels * element_size;

      if (buffer_size > device_scratch_buffer_.size()) {
        device_scratch_buffer_.resize(
            pool.value(), buffer_size, nvidia::gxf::MemoryStorageType::kDevice);
        if (!device_scratch_buffer_.pointer()) {
          throw std::runtime_error(
              fmt::format("Failed to allocate device scratch buffer ({} bytes)", buffer_size));
        }
      }

      CUDA_TRY(cudaMemcpy(static_cast<void*>(device_scratch_buffer_.pointer()),
                          static_cast<const void*>(frame->pointer()),
                          buffer_size,
                          cudaMemcpyHostToDevice));
      input_data_ptr = device_scratch_buffer_.pointer();
    }
  } else {
    // get tensor attached to message by the name defined in the parameters
    const auto maybe_tensor = in_message.get<Tensor>(in_tensor_name_.get().c_str());
    if (!maybe_tensor) {
      throw std::runtime_error(
          fmt::format("Tensor '{}' not found in message.\n", in_tensor_name_.get()));
    }
    // Tensor in_tensor;
    auto in_tensor = maybe_tensor;

    // Get needed information from the tensor
    // cast Holoscan::Tensor to nvidia::gxf::Tensor so attribute access code can remain as-is
    nvidia::gxf::Tensor in_tensor_gxf{in_tensor->dl_ctx()};
    auto in_rank = in_tensor_gxf.rank();
    if (in_rank != 3) {
      throw std::runtime_error(
          fmt::format("Input tensor has {} dimensions. Expected a tensor with 3 dimensions "
                      "(corresponding to an RGB or RGBA image).",
                      in_rank));
    }

    DLDevice dev = in_tensor->device();
    if ((dev.device_type != kDLCUDA) && (dev.device_type != kDLCPU) &&
        (dev.device_type != kDLCUDAHost)) {
      throw std::runtime_error(
          "Input tensor must be in CUDA device memory, CUDA pinned memory or on the CPU.");
    }

    // Originally had:
    //   auto is_contiguous = in_tensor_gxf.isContiguous().value();
    // but there was a bug in GXF 4.0's isContiguous(), so added is_contiguous to holoscan::Tensor
    // instead.
    if (!in_tensor->is_contiguous()) {
      throw std::runtime_error(
          fmt::format("Tensor must have a row-major memory layout (values along the last axis "
                      " are adjacent in memory). Detected shape:({}, {}, {}), "
                      "strides: ({}, {}, {})",
                      in_tensor_gxf.shape().dimension(0),
                      in_tensor_gxf.shape().dimension(1),
                      in_tensor_gxf.shape().dimension(2),
                      in_tensor_gxf.stride(0),
                      in_tensor_gxf.stride(1),
                      in_tensor_gxf.stride(2)));
    }

    if (dev.device_type == kDLCPU) {
      size_t buffer_size = rows * columns * in_channels * element_size;

      if (buffer_size > device_scratch_buffer_.size()) {
        device_scratch_buffer_.resize(
            pool.value(), buffer_size, nvidia::gxf::MemoryStorageType::kDevice);
        if (!device_scratch_buffer_.pointer()) {
          throw std::runtime_error(
              fmt::format("Failed to allocate device scratch buffer ({} bytes)", buffer_size));
        }
      }

      CUDA_TRY(cudaMemcpy(static_cast<void*>(device_scratch_buffer_.pointer()),
                          static_cast<const void*>(in_tensor_gxf.pointer()),
                          buffer_size,
                          cudaMemcpyHostToDevice));
      input_data_ptr = device_scratch_buffer_.pointer();
    } else {
      input_data_ptr = in_tensor_gxf.pointer();
    }

    if (input_data_ptr == nullptr) {
      // This should never happen, but just in case...
      HOLOSCAN_LOG_ERROR("Unable to get tensor data pointer. nullptr returned.");
    }
    in_shape = in_tensor_gxf.shape();
    rows = in_shape.dimension(0);
    columns = in_shape.dimension(1);
    in_channels = in_shape.dimension(2);
    element_type = in_tensor_gxf.element_type();
    element_size = nvidia::gxf::PrimitiveTypeSize(element_type);
    input_memory_type = in_tensor_gxf.storage_type();
  }

  int16_t out_channels = 3 + generate_alpha_;

  if (element_type != nvidia::gxf::PrimitiveType::kUnsigned8 &&
      element_type != nvidia::gxf::PrimitiveType::kUnsigned16) {
    throw std::runtime_error(fmt::format("Unexpected bytes in element representation {} (size {})",
                                         static_cast<int32_t>(element_type),
                                         element_size));
  }

  // allocate tensors with names in the output message
  nvidia::gxf::Expected<nvidia::gxf::Entity> out_message =
      CreateTensorMap(context.context(),
                      pool.value(),
                      {{out_tensor_name_.get(),
                        nvidia::gxf::MemoryStorageType::kDevice,
                        nvidia::gxf::Shape{rows, columns, out_channels},
                        element_type,
                        0,
                        nvidia::gxf::ComputeTrivialStrides(
                            nvidia::gxf::Shape{rows, columns, out_channels}, element_size)}},
                      false);

  if (!out_message) {
    throw std::runtime_error(fmt::format("Failed to allocate tensors in output message: {}",
                                         GxfResultStr(out_message.error())));
  }

  // get the tensor of interest
  const auto maybe_output_tensor =
      out_message.value().get<nvidia::gxf::Tensor>(out_tensor_name_.get().c_str());
  if (!maybe_output_tensor) {
    throw std::runtime_error(
        fmt::format("Failed to access output tensor with name `{}`", out_tensor_name_.get()));
  }

  void* output_data_ptr = maybe_output_tensor.value()->pointer();

  // NPP to demosaic
  if (element_type == nvidia::gxf::PrimitiveType::kUnsigned8) {
    if (generate_alpha_) {
      nppiCFAToRGBA_8u_C1AC4R_Ctx(static_cast<const Npp8u*>(input_data_ptr),
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
      nppiCFAToRGB_8u_C1C3R_Ctx(static_cast<const Npp8u*>(input_data_ptr),
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
      nppiCFAToRGBA_16u_C1AC4R_Ctx(static_cast<const Npp16u*>(input_data_ptr),
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
      nppiCFAToRGB_16u_C1C3R_Ctx(static_cast<const Npp16u*>(input_data_ptr),
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

  // pass the CUDA stream to the output message
  stream_handler_result = cuda_stream_handler_.to_message(out_message);
  if (stream_handler_result != GXF_SUCCESS) {
    throw std::runtime_error("Failed to add the CUDA stream to the outgoing messages");
  }

  // Emit the tensor
  auto result = gxf::Entity(std::move(out_message.value()));
  op_output.emit(result, "transmitter");
}

}  // namespace holoscan::ops
