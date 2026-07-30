/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_OPERATORS_RAW_IMAGE_PROCESSOR_RAW_IMAGE_PROCESSOR_HPP
#define HOLOSCAN_OPERATORS_RAW_IMAGE_PROCESSOR_RAW_IMAGE_PROCESSOR_HPP

#include <memory>

#include "holoscan/utils/cuda/cuda_rtc.hpp"

#include <holoscan/core/operator.hpp>
#include <holoscan/core/parameter.hpp>

namespace holoscan::ops {

class RawImageProcessorOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RawImageProcessorOp);

  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(holoscan::InputContext&, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext&) override;

 private:
  holoscan::Parameter<int> pixel_format_;
  holoscan::Parameter<int> bayer_format_;
  holoscan::Parameter<int32_t> optical_black_;
  holoscan::Parameter<int> cuda_device_ordinal_;

  CUcontext cuda_context_ = nullptr;
  CUdevice cuda_device_ = 0;
  bool is_integrated_ = false;
  bool host_memory_warning_ = false;

  std::shared_ptr<holoscan::CudaFunctionLauncher> cuda_function_launcher_;

  holoscan::UniqueCUdeviceptr histogram_memory_;
  holoscan::UniqueCUdeviceptr white_balance_gains_memory_;

  uint32_t histogram_threadblock_size_ = 0;
  int64_t expected_frame_number_ = 0;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_RAW_IMAGE_PROCESSOR_RAW_IMAGE_PROCESSOR_HPP */
