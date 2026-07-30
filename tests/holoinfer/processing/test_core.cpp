/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_core.hpp"

#include <memory>
#include <string>

void HoloInferProcessingTests::clear_processor() {
  holoscan_processor_context_.reset();
}

HoloInfer::InferStatus HoloInferProcessingTests::call_parameter_check_processing() {
  return HoloInfer::processor_validity_check(
      processed_map, in_tensor_processing, out_tensor_processing);
}

HoloInfer::InferStatus HoloInferProcessingTests::setup_processor(bool use_cuda_graphs) {
  holoscan_processor_context_ = std::make_unique<HoloInfer::ProcessorContext>();
  auto status = holoscan_processor_context_->initialize(
      process_operations, custom_kernels, use_cuda_graphs, config_path);
  return status;
}

HoloInfer::InferStatus HoloInferProcessingTests::execute_processor() {
  auto status = holoscan_processor_context_->process(process_operations,
                                                     processed_map,
                                                     data_per_tensor,
                                                     dims_per_tensor,
                                                     process_with_cuda,
                                                     cuda_stream);
  return status;
}
