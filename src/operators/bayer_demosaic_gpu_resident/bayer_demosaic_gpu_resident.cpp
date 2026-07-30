/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/operators/bayer_demosaic_gpu_resident/bayer_demosaic_gpu_resident.hpp>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/io_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/logger/logger.hpp>

namespace holoscan::ops {

void BayerDemosaicGpuResidentOp::setup(OperatorSpec& spec) {
  // For GPU-resident operators, we use device_input/device_output with fixed memory sizes.
  // The memory size is calculated based on configuration parameters:
  // - Input: width * height * 1 channel * element_size
  // - Output: width * height * (3 or 4 channels) * element_size
  //
  // Since the memory sizes need to be known at setup time for GPU-resident graph execution,
  // the user must specify width, height, pixel_type, and generate_alpha parameters.

  // Note: The actual device_input/device_output ports are created in initialize()
  // after parameters are available. Here we just define the parameters.

  spec.device_input(
      "in", 0);  // default memory block size is set to 0 which will be overwritten in initialize()
  spec.device_output(
      "out", 0);  // default memory block size is set to 0 which will be overwritten in initialize()

  spec.param(width_,
             "width",
             "Image Width",
             "Width of the input image in pixels. Must be an even number.",
             0);  // Will be validated in initialize

  spec.param(height_,
             "height",
             "Image Height",
             "Height of the input image in pixels. Must be an even number.",
             0);  // Will be validated in initialize

  spec.param(pixel_type_,
             "pixel_type",
             "Pixel Type",
             "Pixel type: 0 = uint8 (1 byte), 1 = uint16 (2 bytes). Default: 0.",
             0);

  spec.param(
      bayer_interp_mode_,
      "interpolation_mode",
      "Interpolation used for demosaicing",
      "The interpolation model to be used for demosaicing (default: NPPI_INTER_UNDEFINED). "
      "Note: Only NPPI_INTER_UNDEFINED (0) is supported for Bayer demosaic according to NPP docs.",
      0);

  spec.param(bayer_grid_pos_,
             "bayer_grid_pos",
             "Bayer grid position",
             "The Bayer grid position (default: NPPI_BAYER_GBRG). Values: "
             "0=BGGR, 1=RGGB, 2=GBRG, 3=GRBG.",
             2);

  spec.param(generate_alpha_,
             "generate_alpha",
             "Generate alpha channel",
             "Generate alpha channel (4-channel RGBA output). Default: false (3-channel RGB).",
             false);

  spec.param(alpha_value_,
             "alpha_value",
             "Alpha value to be generated",
             "Alpha value to be generated if `generate_alpha` is true. Default: 255.",
             255);
}

void BayerDemosaicGpuResidentOp::initialize() {
  // First call base class initialize
  GPUResidentOperator::initialize();

  // Validate width and height
  int32_t width = width_.get();
  int32_t height = height_.get();

  if (width <= 0 || height <= 0) {
    throw std::runtime_error(fmt::format(
        "BayerDemosaicGpuResidentOp: width ({}) and height ({}) must be positive", width, height));
  }

  if (width % 2 != 0) {
    throw std::runtime_error(
        fmt::format("BayerDemosaicGpuResidentOp: width ({}) must be an even number", width));
  }

  if (height % 2 != 0) {
    throw std::runtime_error(
        fmt::format("BayerDemosaicGpuResidentOp: height ({}) must be an even number", height));
  }

  // Validate and set interpolation mode
  npp_bayer_interp_mode_ = static_cast<NppiInterpolationMode>(bayer_interp_mode_.get());
  if (npp_bayer_interp_mode_ != NPPI_INTER_UNDEFINED) {
    throw std::runtime_error(
        fmt::format("BayerDemosaicGpuResidentOp: Unsupported interpolation_mode: {}. "
                    "Only NPPI_INTER_UNDEFINED (0) is supported for Bayer demosaic.",
                    static_cast<int>(npp_bayer_interp_mode_)));
  }

  // Set Bayer grid position
  npp_bayer_grid_pos_ = static_cast<NppiBayerGridPosition>(bayer_grid_pos_.get());

  // Determine element size based on pixel type
  int32_t pixel_type = pixel_type_.get();
  if (pixel_type == 0) {
    element_size_ = 1;  // uint8
  } else if (pixel_type == 1) {
    element_size_ = 2;  // uint16
  } else {
    throw std::runtime_error(fmt::format(
        "BayerDemosaicGpuResidentOp: Invalid pixel_type: {}. Must be 0 (uint8) or 1 (uint16).",
        pixel_type));
  }

  // Calculate output channels
  out_channels_ = generate_alpha_.get() ? 4 : 3;

  // Pre-compute values for compute function
  in_line_step_ = width * 1 * static_cast<int32_t>(element_size_);  // 1 channel input
  out_line_step_ = width * out_channels_ * static_cast<int32_t>(element_size_);

  roi_size_.width = width;
  roi_size_.height = height;

  roi_rect_.x = 0;
  roi_rect_.y = 0;
  roi_rect_.width = width;
  roi_rect_.height = height;

  // Calculate memory sizes for device ports
  size_t input_memory_size =
      static_cast<size_t>(width) * static_cast<size_t>(height) * element_size_;
  size_t output_memory_size = static_cast<size_t>(width) * static_cast<size_t>(height) *
                              static_cast<size_t>(out_channels_) * element_size_;

  // Register device input and output ports with the operator spec
  // These are used by the GPU-resident executor to allocate device memory
  auto& spec = *this->spec();

  // change the memory block size of the device input and output ports
  spec.device_input("in", input_memory_size);
  spec.device_output("out", output_memory_size);

  HOLOSCAN_LOG_INFO(
      "BayerDemosaicGpuResidentOp initialized: {}x{}, pixel_type={}, "
      "generate_alpha={}, bayer_grid_pos={}, in_size={}, out_size={}",
      width,
      height,
      pixel_type,
      generate_alpha_.get(),
      bayer_grid_pos_.get(),
      input_memory_size,
      output_memory_size);
}

void BayerDemosaicGpuResidentOp::compute([[maybe_unused]] InputContext& op_input,
                                         [[maybe_unused]] OutputContext& op_output,
                                         [[maybe_unused]] ExecutionContext& context) {
  // In GPU-resident mode:
  // - This function is called once from the CPU during graph capture
  // - All CUDA operations launched here are captured into a CUDA graph
  // - Subsequent iterations execute the captured graph directly on the GPU
  //
  // Important considerations:
  // - Do NOT use any host-side conditionals that depend on runtime data
  // - Do NOT perform any host-device synchronization
  // - Only launch CUDA kernels/operations that will be captured

  // Get device memory addresses for input and output
  void* input_ptr = device_memory("in");
  void* output_ptr = device_memory("out");

  if (input_ptr == nullptr || output_ptr == nullptr) {
    throw std::runtime_error(fmt::format(
        "BayerDemosaicGpuResidentOp::compute() - Device memory not available: in={}, out={}",
        static_cast<void*>(input_ptr),
        static_cast<void*>(output_ptr)));
  }

  // Get the CUDA stream for this operator
  auto stream_ptr = cuda_stream();
  if (!stream_ptr) {
    throw std::runtime_error("BayerDemosaicGpuResidentOp::compute() - CUDA stream not available");
  }
  cudaStream_t stream = *stream_ptr;

  // Assign the CUDA stream to the NPP stream context
  npp_stream_ctx_.hStream = stream;

  HOLOSCAN_LOG_INFO("BayerDemosaicGpuResidentOp::compute() -- {} -- in: {} out: {}, stream: {}",
                    name(),
                    input_ptr,
                    output_ptr,
                    static_cast<void*>(stream));

  // Call NPP demosaic function based on pixel type and alpha generation setting
  // Note: These NPP functions launch CUDA kernels that will be captured by the GPU-resident
  // executor
  NppStatus npp_status = NPP_SUCCESS;

  if (element_size_ == 1) {
    // 8-bit unsigned integer
    if (generate_alpha_.get()) {
      npp_status = nppiCFAToRGBA_8u_C1AC4R_Ctx(static_cast<const Npp8u*>(input_ptr),
                                               in_line_step_,
                                               roi_size_,
                                               roi_rect_,
                                               static_cast<Npp8u*>(output_ptr),
                                               out_line_step_,
                                               npp_bayer_grid_pos_,
                                               npp_bayer_interp_mode_,
                                               static_cast<Npp8u>(alpha_value_.get()),
                                               npp_stream_ctx_);
    } else {
      npp_status = nppiCFAToRGB_8u_C1C3R_Ctx(static_cast<const Npp8u*>(input_ptr),
                                             in_line_step_,
                                             roi_size_,
                                             roi_rect_,
                                             static_cast<Npp8u*>(output_ptr),
                                             out_line_step_,
                                             npp_bayer_grid_pos_,
                                             npp_bayer_interp_mode_,
                                             npp_stream_ctx_);
    }
  } else {
    // 16-bit unsigned integer
    if (generate_alpha_.get()) {
      npp_status = nppiCFAToRGBA_16u_C1AC4R_Ctx(static_cast<const Npp16u*>(input_ptr),
                                                in_line_step_,
                                                roi_size_,
                                                roi_rect_,
                                                static_cast<Npp16u*>(output_ptr),
                                                out_line_step_,
                                                npp_bayer_grid_pos_,
                                                npp_bayer_interp_mode_,
                                                static_cast<Npp16u>(alpha_value_.get()),
                                                npp_stream_ctx_);
    } else {
      npp_status = nppiCFAToRGB_16u_C1C3R_Ctx(static_cast<const Npp16u*>(input_ptr),
                                              in_line_step_,
                                              roi_size_,
                                              roi_rect_,
                                              static_cast<Npp16u*>(output_ptr),
                                              out_line_step_,
                                              npp_bayer_grid_pos_,
                                              npp_bayer_interp_mode_,
                                              npp_stream_ctx_);
    }
  }

  if (npp_status != NPP_SUCCESS) {
    HOLOSCAN_LOG_ERROR(
        "BayerDemosaicGpuResidentOp::compute() -- NPP demosaic failed with status: {}",
        static_cast<int>(npp_status));
  } else {
    HOLOSCAN_LOG_INFO("BayerDemosaicGpuResidentOp::compute() -- Demosaic operation launched");
  }
}

}  // namespace holoscan::ops
