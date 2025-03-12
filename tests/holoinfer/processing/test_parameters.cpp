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

#include "test_core.hpp"

#include <memory>
#include <string>
#include <utility>

void ProcessingTests::parameter_test() {
  std::string test_module = "Parameter test processing";

  // input tensors to processor
  // Test: Processing Params, input tensors are empty
  auto in_names = std::move(in_tensor_processing);
  auto status = call_parameter_check_processing();
  processing_assert(
      status, test_module, 1, test_identifier_process.at(1), HoloInfer::holoinfer_code::H_ERROR);
  in_tensor_processing = std::move(in_names);

  // processed_map to processor
  // Test: Processing Params, processed_map empty value vector check
  auto str_value = processed_map.at("plax_cham_infer")[0];
  processed_map.at("plax_cham_infer").pop_back();
  status = call_parameter_check_processing();
  processing_assert(
      status, test_module, 2, test_identifier_process.at(2), HoloInfer::holoinfer_code::H_ERROR);
  processed_map.at("plax_cham_infer").push_back(str_value);

  // Test: Processing Params, processed_map empty tensor name check
  processed_map.at("plax_cham_infer").push_back("");
  status = call_parameter_check_processing();
  processing_assert(
      status, test_module, 3, test_identifier_process.at(3), HoloInfer::holoinfer_code::H_ERROR);
  processed_map.at("plax_cham_infer").pop_back();

  // Test: Processing Params, processed_map duplicate tensor name check
  processed_map.at("plax_cham_infer").push_back(str_value);
  status = call_parameter_check_processing();
  processing_assert(
      status, test_module, 4, test_identifier_process.at(4), HoloInfer::holoinfer_code::H_ERROR);
  processed_map.at("plax_cham_infer").pop_back();

  // output tensor names test
  // Test: Processing Params, output_tensor exist in processed_map
  out_tensor_processing.push_back("dummy-output");
  status = call_parameter_check_processing();
  processing_assert(
      status, test_module, 5, test_identifier_process.at(5), HoloInfer::holoinfer_code::H_ERROR);
  out_tensor_processing.pop_back();

  // Test: Processing Params, output_tensor is unique
  out_tensor_processing.push_back(out_tensor_processing[0]);
  status = call_parameter_check_processing();
  processing_assert(
      status, test_module, 6, test_identifier_process.at(6), HoloInfer::holoinfer_code::H_ERROR);
  out_tensor_processing.pop_back();
}

void ProcessingTests::parameter_setup_test() {
  std::string test_module = "Parameter test processing";
  // add tests for process_operations
  // Test: Processing Params, empty operation vector in process_operation
  process_operations.insert({"dummy-input", {}});
  auto status = setup_processor();
  processing_assert(
      status, test_module, 7, test_identifier_process.at(7), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
  process_operations.erase("dummy-input");

  // Test: Processing Params, dummy operation in vector
  process_operations.at("plax_cham_infer").push_back("dummy-operation");
  status = setup_processor();
  processing_assert(
      status, test_module, 8, test_identifier_process.at(8), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
  process_operations.at("plax_cham_infer").pop_back();

  // add tests for operations (print)
  // Test: Processing Params, print operation not supported
  auto oper = process_operations.at("plax_cham_infer");
  process_operations.at("plax_cham_infer") = {"print_all"};
  status = setup_processor();
  processing_assert(
      status, test_module, 9, test_identifier_process.at(9), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
  process_operations.at("plax_cham_infer") = std::move(oper);

  // add tests for config
  // Test: Processing Params, incorrect config path
  config_path = "dummy-path.txt";
  status = setup_processor();
  processing_assert(
      status, test_module, 10, test_identifier_process.at(10), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
  config_path = "";

  // incorrect tensor map
  // Test: Processing Params, incorrect tensor in result map
  status = setup_processor();
  auto db = std::make_shared<HoloInfer::DataBuffer>();
  data_per_tensor.insert({"dummy-tensor", db});
  status = execute_processor();
  processing_assert(
      status, test_module, 11, test_identifier_process.at(11), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
  data_per_tensor.erase("dummy-tensor");

  // empty dimension map
  // Test: Processing Params, empty dimension map
  status = setup_processor();
  data_per_tensor.insert({"plax_cham_infer", db});
  status = execute_processor();
  processing_assert(
      status, test_module, 12, test_identifier_process.at(12), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();

  // missing entry in processed_map
  // Test: Processing Params, Mismatch tensor in processed_map
  status = setup_processor();
  auto pm_data = processed_map.at("plax_cham_infer");
  processed_map.erase("plax_cham_infer");
  data_per_tensor.insert({"plax_cham_infer", db});
  dims_per_tensor.insert({"plax_cham_infer", {}});
  status = execute_processor();
  processing_assert(
      status, test_module, 13, test_identifier_process.at(13), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
  processed_map["plax_cham_infer"] = pm_data;

  // empty input buffer
  // Test: Processing Params, Empty data buffer
  status = setup_processor();
  status = execute_processor();
  processing_assert(
      status, test_module, 14, test_identifier_process.at(14), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();

  // add tests for generate_boxes
  // Test: Processing Params, Empty config for generate boxes
  process_operations.insert({"plax_cham_infer:tensor2", {"generate_boxes"}});
  status = setup_processor();
  processing_assert(
      status, test_module, 15, test_identifier_process.at(15), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();

  // Test: Processing Params, Incorrect config path for generate boxes
  config_path = "postprocessing.yaml";
  status = setup_processor();
  processing_assert(
      status, test_module, 16, test_identifier_process.at(16), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();

  // Test: Processing Params, incorrect tensor for generate boxes
  data_per_tensor.clear();
  status = setup_processor();
  status = execute_processor();
  processing_assert(
      status, test_module, 17, test_identifier_process.at(17), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
  process_operations.erase("plax_cham_infer:tensor2");

  // Test: Processing Params, Empty custom_kernels map with custom_cuda kernel operation
  config_path = "";
  process_operations.at("plax_cham_infer").push_back("custom_cuda_kernel-1");
  status = setup_processor();
  processing_assert(
      status, test_module, 18, test_identifier_process.at(18), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
  process_operations.at("plax_cham_infer").pop_back();

  // Test: Processing Params, custom cuda kernel incorrect name
  process_operations.at("plax_cham_infer").push_back("custom_cuda_kernel-1-2");
  status = setup_processor();
  processing_assert(
      status, test_module, 19, test_identifier_process.at(19), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
  process_operations.at("plax_cham_infer").pop_back();

  // Test: Processing Params, custom cuda kernel key not present in custom_kernels map
  process_operations.at("plax_cham_infer").push_back("custom_cuda_kernel-1");
  custom_kernels["cuda_kernel"] = "";
  status = setup_processor();
  processing_assert(
      status, test_module, 20, test_identifier_process.at(20), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
  custom_kernels.erase("cuda_kernel");

  // Test: Processing Params, custom cuda kernel empty in the custom_kernels map
  custom_kernels["cuda_kernel-1"] = "";
  status = setup_processor();
  processing_assert(
      status, test_module, 21, test_identifier_process.at(21), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();

  // Test: Processing Params, Incorrect custom cuda kernel definition
  custom_kernels["cuda_kernel-1"] = "Cuda Kernel";
  status = setup_processor();
  processing_assert(
      status, test_module, 22, test_identifier_process.at(22), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();

  process_operations.at("plax_cham_infer").pop_back();
}
