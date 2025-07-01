/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef HOLOINFER_PROCESSING_TEST_CORE_HPP
#define HOLOINFER_PROCESSING_TEST_CORE_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <holoinfer.hpp>
#include <holoinfer_utils.hpp>

namespace HoloInfer = holoscan::inference;

class ProcessingTests {
 public:
  ProcessingTests() {}
  void processing_assert(const HoloInfer::InferStatus& status, const std::string& module,
                         unsigned int current_test, const std::string& test_name,
                         HoloInfer::holoinfer_code assert_type);

  HoloInfer::InferStatus call_parameter_check_processing();
  HoloInfer::InferStatus setup_processor(bool use_cuda_graphs = false);
  HoloInfer::InferStatus execute_processor();
  void clear_processor();

  void parameter_test();
  void parameter_setup_test();
  void print_summary();
  int get_status();

 private:
  /// Default parameters for inference
  unsigned int pass_test_count = 0, fail_test_count = 0, total_test_count = 0;

  std::map<std::string, std::vector<std::string>> process_operations = {
      {"plax_cham_infer", {"max_per_channel_scaled"}}};

  std::map<std::string, std::vector<std::string>> processed_map = {
      {"plax_cham_infer", {"plax_chamber_processed"}}};
  std::vector<std::string> out_tensor_processing = {"plax_chamber_processed"};
  std::vector<std::string> in_tensor_processing = {
      "plax_cham_infer", "aortic_infer", "bmode_infer"};

  std::unique_ptr<HoloInfer::ProcessorContext> holoscan_processor_context_;
  HoloInfer::DataMap data_per_tensor;
  std::map<std::string, std::vector<int>> dims_per_tensor;
  bool process_with_cuda = false;
  cudaStream_t cuda_stream = 0;
  std::string config_path = "";
  std::map<std::string, std::string> custom_kernels;

  const std::map<unsigned int, std::string> test_identifier_process = {
      {1, "Processing Params, input tensors are empty"},
      {2, "Processing Params, processed_map empty value vector check"},
      {3, "Processing Params, processed_map empty tensor name check"},
      {4, "Processing Params, processed_map duplicate tensor name check"},
      {5, "Processing Params, output_tensor exist in processed_map"},
      {6, "Processing Params, output_tensor is unique"},
      {7, "Processing Params, empty operation vector in process_operation"},
      {8, "Processing Params, dummy operation in vector"},
      {9, "Processing Params, print operation not supported"},
      {10, "Processing Params, incorrect config path"},
      {11, "Processing Params, incorrect tensor in result map"},
      {12, "Processing Params, empty dimension map"},
      {13, "Processing Params, Mismatch tensor in processed_map"},
      {14, "Processing Params, Empty data buffer"},
      {15, "Processing Params, Empty config for generate boxes"},
      {16, "Processing Params, Incorrect config path for generate boxes"},
      {17, "Processing Params, incorrect tensor for generate boxes"},
      {18, "Processing Params, Custom CUDA kernel: empty cuda kernels map"},
      {19, "Processing Params, Custom CUDA kernel: Incorrect naming"},
      {20, "Processing Params, Custom CUDA kernel: Incorrect key in custom kernel map"},
      {21, "Processing Params, Custom CUDA kernel: Empty kernel in custom kernel map"},
      {22, "Processing Params, Custom CUDA kernel: Incorrect kernel in custom kernel map"},
      {23, "Processing Params, Custom CUDA kernel: CUDA Graphs true"}};
};

#endif /* HOLOINFER_PROCESSING_TEST_CORE_HPP */
