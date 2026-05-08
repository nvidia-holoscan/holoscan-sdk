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
#include <utility>

// =============================================================================
// Parameter validation tests (originally parameter_test)
// =============================================================================

TEST_F(HoloInferProcessingTests, Params_InputTensorsEmpty) {
  auto in_names = std::move(in_tensor_processing);
  auto status = call_parameter_check_processing();
  EXPECT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_ERROR);
  in_tensor_processing = std::move(in_names);
}

TEST_F(HoloInferProcessingTests, Params_ProcessedMapEmptyValueVector) {
  auto str_value = processed_map.at("plax_cham_infer")[0];
  processed_map.at("plax_cham_infer").pop_back();
  auto status = call_parameter_check_processing();
  EXPECT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferProcessingTests, Params_ProcessedMapEmptyTensorName) {
  processed_map.at("plax_cham_infer").push_back("");
  auto status = call_parameter_check_processing();
  EXPECT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferProcessingTests, Params_ProcessedMapDuplicateTensorName) {
  auto str_value = processed_map.at("plax_cham_infer")[0];
  processed_map.at("plax_cham_infer").push_back(str_value);
  auto status = call_parameter_check_processing();
  EXPECT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferProcessingTests, Params_OutputTensorExistInProcessedMap) {
  out_tensor_processing.push_back("dummy-output");
  auto status = call_parameter_check_processing();
  EXPECT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferProcessingTests, Params_OutputTensorIsUnique) {
  out_tensor_processing.push_back(out_tensor_processing[0]);
  auto status = call_parameter_check_processing();
  EXPECT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_ERROR);
}

// =============================================================================
// Parameter setup tests (originally parameter_setup_test)
// =============================================================================

TEST_F(HoloInferProcessingTests, Setup_EmptyOperationVector) {
  process_operations.insert({"dummy-input", {}});
  auto status = setup_processor();
  EXPECT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
}

TEST_F(HoloInferProcessingTests, Setup_DummyOperationInVector) {
  process_operations.at("plax_cham_infer").push_back("dummy-operation");
  auto status = setup_processor();
  EXPECT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
}

TEST_F(HoloInferProcessingTests, Setup_PrintOperationNotSupported) {
  process_operations.at("plax_cham_infer") = {"print_all"};
  auto status = setup_processor();
  EXPECT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
}

TEST_F(HoloInferProcessingTests, Setup_IncorrectConfigPath) {
  config_path = "dummy-path.txt";
  auto status = setup_processor();
  EXPECT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
}

TEST_F(HoloInferProcessingTests, Setup_IncorrectTensorInResultMap) {
  auto status = setup_processor();
  auto db = std::make_shared<HoloInfer::DataBuffer>();
  data_per_tensor.insert({"dummy-tensor", db});
  status = execute_processor();
  EXPECT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
}

TEST_F(HoloInferProcessingTests, Setup_EmptyDimensionMap) {
  auto status = setup_processor();
  auto db = std::make_shared<HoloInfer::DataBuffer>();
  data_per_tensor.insert({"plax_cham_infer", db});
  status = execute_processor();
  EXPECT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
}

TEST_F(HoloInferProcessingTests, Setup_MismatchTensorInProcessedMap) {
  auto status = setup_processor();
  auto pm_data = processed_map.at("plax_cham_infer");
  processed_map.erase("plax_cham_infer");
  auto db = std::make_shared<HoloInfer::DataBuffer>();
  data_per_tensor.insert({"plax_cham_infer", db});
  dims_per_tensor.insert({"plax_cham_infer", {}});
  status = execute_processor();
  EXPECT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
}

TEST_F(HoloInferProcessingTests, Setup_EmptyDataBuffer) {
  auto db = std::make_shared<HoloInfer::DataBuffer>();
  data_per_tensor.insert({"plax_cham_infer", db});
  auto status = setup_processor();
  status = execute_processor();
  EXPECT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
}

TEST_F(HoloInferProcessingTests, Setup_EmptyConfigForGenerateBoxes) {
  process_operations.insert({"plax_cham_infer:tensor2", {"generate_boxes"}});
  auto status = setup_processor();
  EXPECT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
}

TEST_F(HoloInferProcessingTests, Setup_IncorrectConfigPathForGenerateBoxes) {
  process_operations.insert({"plax_cham_infer:tensor2", {"generate_boxes"}});
  config_path = "postprocessing.yaml";
  auto status = setup_processor();
  EXPECT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
}

TEST_F(HoloInferProcessingTests, Setup_IncorrectTensorForGenerateBoxes) {
  process_operations.insert({"plax_cham_infer:tensor2", {"generate_boxes"}});
  config_path = "postprocessing.yaml";
  auto status = setup_processor();
  status = execute_processor();
  EXPECT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
}

TEST_F(HoloInferProcessingTests, Setup_CustomCudaKernel_EmptyKernelsMap) {
  process_operations.at("plax_cham_infer").push_back("custom_cuda_kernel-1");
  auto status = setup_processor();
  EXPECT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
}

TEST_F(HoloInferProcessingTests, Setup_CustomCudaKernel_IncorrectNaming) {
  process_operations.at("plax_cham_infer").push_back("custom_cuda_kernel-1-2");
  auto status = setup_processor();
  EXPECT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
}

TEST_F(HoloInferProcessingTests, Setup_CustomCudaKernel_IncorrectKeyInMap) {
  process_operations.at("plax_cham_infer").push_back("custom_cuda_kernel-1");
  custom_kernels["cuda_kernel"] = "";
  auto status = setup_processor();
  EXPECT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
}

TEST_F(HoloInferProcessingTests, Setup_CustomCudaKernel_EmptyKernelInMap) {
  process_operations.at("plax_cham_infer").push_back("custom_cuda_kernel-1");
  custom_kernels["cuda_kernel-1"] = "";
  auto status = setup_processor();
  EXPECT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
}

TEST_F(HoloInferProcessingTests, Setup_CustomCudaKernel_IncorrectKernelInMap) {
  process_operations.at("plax_cham_infer").push_back("custom_cuda_kernel-1");
  custom_kernels["cuda_kernel-1"] = "Cuda Kernel";
  auto status = setup_processor();
  EXPECT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_ERROR);
  clear_processor();
}

TEST_F(HoloInferProcessingTests, Setup_CustomCudaKernel_CudaGraphsTrue) {
  process_operations.at("plax_cham_infer").push_back("custom_cuda_kernel-1");
  custom_kernels["cuda_kernel-1"] = R"(
      extern "C" __global__ void
      customKernel1(const unsigned char* input, unsigned char* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) { output[idx] = input[idx]; }
  }
  )";

  auto status = setup_processor(true);
  EXPECT_EQ(status.get_code(), HoloInfer::holoinfer_code::H_SUCCESS);
  clear_processor();
}
