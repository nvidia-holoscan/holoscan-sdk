/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
