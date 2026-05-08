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

#include <yaml-cpp/yaml.h>

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// =============================================================================
// TRT backend — basic error conditions
// =============================================================================

TEST_F(HoloInferTests, TRT_EmptyInputData) {
  backend = "trt";
  auto status = prepare_for_inference();
  auto dmap = std::move(inference_specs_->data_per_tensor_);
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
  inference_specs_->data_per_tensor_ = std::move(dmap);
}

TEST_F(HoloInferTests, TRT_EmptyInferenceParameters) {
  backend = "trt";
  clear_specs();
  setup_specifications();
  // holoscan_infer_context_ is null — no set_inference_params called
  auto status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, TRT_MissingInputTensor) {
  backend = "trt";
  auto status = prepare_for_inference();
  auto dm = std::move(inference_specs_->data_per_tensor_.at("m1_pre_proc"));
  inference_specs_->data_per_tensor_.erase("m1_pre_proc");
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, TRT_MissingOutputTensor) {
  backend = "trt";
  auto status = prepare_for_inference();
  auto dm = std::move(inference_specs_->output_per_model_.at("m2_infer"));
  inference_specs_->output_per_model_.erase("m2_infer");
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, TRT_EmptyInputCudaBuffer1) {
  backend = "trt";
  auto status = prepare_for_inference();
  inference_specs_->data_per_tensor_.at("m1_pre_proc")->device_buffer_->resize(0);
  inference_specs_->data_per_tensor_.at("m1_pre_proc")->device_buffer_ = nullptr;
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, TRT_EmptyInputCudaBuffer2) {
  backend = "trt";
  auto status = prepare_for_inference();
  inference_specs_->data_per_tensor_.at("m1_pre_proc")->device_buffer_ =
      std::make_shared<HoloInfer::DeviceBuffer>();
  // device_buffer_ exists but has size 0
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, TRT_EmptyOutputCudaBuffer1) {
  backend = "trt";
  auto status = prepare_for_inference();
  inference_specs_->output_per_model_.at("m2_infer")->device_buffer_->resize(0);
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, TRT_EmptyOutputCudaBuffer2) {
  backend = "trt";
  auto status = prepare_for_inference();
  inference_specs_->output_per_model_.at("m2_infer")->device_buffer_ = nullptr;
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, TRT_EmptyOutputCudaBuffer3) {
  backend = "trt";
  auto status = prepare_for_inference();
  inference_specs_->output_per_model_.at("m2_infer")->device_buffer_ =
      std::make_shared<HoloInfer::DeviceBuffer>();
  // device_buffer_ exists but has size 0
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

// =============================================================================
// TRT backend — successful inference
// =============================================================================

TEST_F(HoloInferTests, TRT_BasicEndToEndCudaInference) {
  backend = "trt";
  auto status = prepare_for_inference();
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, TRT_BasicSequentialEndToEndCudaInference) {
  backend = "trt";
  parallel_inference = false;
  auto status = prepare_for_inference();
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, TRT_InputOnHostInference) {
  backend = "trt";
  input_on_cuda = false;
  auto status = prepare_for_inference();
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, TRT_OutputOnHostInference) {
  backend = "trt";
  output_on_cuda = false;
  auto status = prepare_for_inference();
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, TRT_InputOutputOnHostInference) {
  backend = "trt";
  input_on_cuda = false;
  output_on_cuda = false;
  auto status = prepare_for_inference();
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, TRT_EmptyHostInput) {
  backend = "trt";
  input_on_cuda = false;
  output_on_cuda = false;
  auto status = prepare_for_inference();
  inference_specs_->data_per_tensor_.at("m1_pre_proc")->host_buffer_->resize(0);
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, TRT_EmptyHostOutput) {
  backend = "trt";
  input_on_cuda = false;
  output_on_cuda = false;
  auto status = prepare_for_inference();
  inference_specs_->output_per_model_.at("m2_infer")->host_buffer_->resize(0);
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

// =============================================================================
// TRT backend — multi-rank tests
// =============================================================================

TEST_F(HoloInferTests, TRT_MultiRankRank5) {
  backend = "trt";
  model_path_map["model_1"] = model_folder + "identity_model_5r.onnx";
  model_path_map["model_2"] = model_folder + "identity_model_5r.onnx";
  in_tensor_dimensions["m1_pre_proc"] = {1, 1, 1, 1, 1};
  in_tensor_dimensions["m2_pre_proc"] = {1, 1, 1, 1, 1};
  auto status = prepare_for_inference();
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, TRT_MultiRankRank9) {
  backend = "trt";
  model_path_map["model_1"] = model_folder + "identity_model_9r.onnx";
  model_path_map["model_2"] = model_folder + "identity_model_9r.onnx";
  in_tensor_dimensions["m1_pre_proc"] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  in_tensor_dimensions["m2_pre_proc"] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  auto status = prepare_for_inference();
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

// =============================================================================
// TRT backend — multi-GPU tests (skipped if second GPU is unavailable)
// =============================================================================

TEST_F(HoloInferTests, TRT_BasicSequentialInferenceMultiGPU) {
  cudaDeviceProp device_prop;
  if (cudaGetDeviceProperties(&device_prop, 1) != cudaSuccess) {
    cudaGetLastError();
    GTEST_SKIP() << "Second GPU not available";
  }
  backend = "trt";
  input_on_cuda = true;
  output_on_cuda = true;
  parallel_inference = false;
  device_map.at("model_1") = "1";
  auto status = prepare_for_inference();
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, TRT_BasicParallelInferenceMultiGPU) {
  cudaDeviceProp device_prop;
  if (cudaGetDeviceProperties(&device_prop, 1) != cudaSuccess) {
    cudaGetLastError();
    GTEST_SKIP() << "Second GPU not available";
  }
  backend = "trt";
  input_on_cuda = true;
  output_on_cuda = true;
  parallel_inference = true;
  device_map.at("model_1") = "1";
  auto status = prepare_for_inference();
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, TRT_ParallelInferenceMultiGPU_IOOnHost) {
  cudaDeviceProp device_prop;
  if (cudaGetDeviceProperties(&device_prop, 1) != cudaSuccess) {
    cudaGetLastError();
    GTEST_SKIP() << "Second GPU not available";
  }
  backend = "trt";
  input_on_cuda = false;
  output_on_cuda = false;
  parallel_inference = true;
  device_map.at("model_1") = "1";
  auto status = prepare_for_inference();
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, TRT_ParallelInferenceMultiGPU_InputOnHost) {
  cudaDeviceProp device_prop;
  if (cudaGetDeviceProperties(&device_prop, 1) != cudaSuccess) {
    cudaGetLastError();
    GTEST_SKIP() << "Second GPU not available";
  }
  backend = "trt";
  input_on_cuda = false;
  output_on_cuda = true;
  parallel_inference = true;
  device_map.at("model_1") = "1";
  auto status = prepare_for_inference();
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, TRT_ParallelInferenceMultiGPU_OutputOnHost) {
  cudaDeviceProp device_prop;
  if (cudaGetDeviceProperties(&device_prop, 1) != cudaSuccess) {
    cudaGetLastError();
    GTEST_SKIP() << "Second GPU not available";
  }
  backend = "trt";
  input_on_cuda = true;
  output_on_cuda = false;
  parallel_inference = true;
  device_map.at("model_1") = "1";
  auto status = prepare_for_inference();
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

// =============================================================================
// TRT backend — dynamic input tests
// =============================================================================

TEST_F(HoloInferTests, TRT_DynamicInput_EmptyTrtOptProfile) {
  backend = "trt";
  model_path_map["model_1"] = model_folder + "identity_model_dynamic.onnx";
  model_path_map["model_2"] = model_folder + "identity_model_dynamic.onnx";
  in_tensor_dimensions["m1_pre_proc"] = {2, 256, 256};
  in_tensor_dimensions["m2_pre_proc"] = {8, 256, 256};
  // batch_sizes not updated — empty trt_opt_profile
  cleanup_engines();
  dynamic_inputs = true;
  auto status = prepare_for_inference();
  for (const auto& td : in_tensor_dimensions) {
    inference_specs_->dims_per_tensor_.at(td.first) = td.second;
  }
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
  cleanup_engines();
}

TEST_F(HoloInferTests, TRT_DynamicInput_WrongTrtOptProfile) {
  backend = "trt";
  model_path_map["model_1"] = model_folder + "identity_model_dynamic.onnx";
  model_path_map["model_2"] = model_folder + "identity_model_dynamic.onnx";
  in_tensor_dimensions["m1_pre_proc"] = {2, 256, 256};
  in_tensor_dimensions["m2_pre_proc"] = {8, 256, 256};
  batch_sizes["model_1"] = {"1, 2"};  // wrong — missing max value
  cleanup_engines();
  dynamic_inputs = true;
  auto status = prepare_for_inference();
  for (const auto& td : in_tensor_dimensions) {
    inference_specs_->dims_per_tensor_.at(td.first) = td.second;
  }
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
  cleanup_engines();
}

TEST_F(HoloInferTests, TRT_DynamicInput_CorrectTrtOptProfile) {
  backend = "trt";
  model_path_map["model_1"] = model_folder + "identity_model_dynamic.onnx";
  model_path_map["model_2"] = model_folder + "identity_model_dynamic.onnx";
  in_tensor_dimensions["m1_pre_proc"] = {2, 256, 256};
  in_tensor_dimensions["m2_pre_proc"] = {8, 256, 256};
  batch_sizes["model_1"] = {"1, 2, 8"};
  cleanup_engines();
  dynamic_inputs = true;
  auto status = prepare_for_inference();
  for (const auto& td : in_tensor_dimensions) {
    inference_specs_->dims_per_tensor_.at(td.first) = td.second;
  }
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
  cleanup_engines();
}

TEST_F(HoloInferTests, TRT_MultiDynamicInput_IncorrectTrtOptProfile) {
  backend = "trt";
  model_path_map["model_1"] = model_folder + "identity_dynamic_multi.onnx";
  model_path_map["model_2"] = model_folder + "identity_dynamic_multi.onnx";
  in_tensor_dimensions["m1_pre_proc"] = {2, 256, 256};
  in_tensor_dimensions["m2_pre_proc"] = {8, 32, 256};
  batch_sizes["model_1"] = {"1, 2, 8"};  // incorrect for multi-input model
  cleanup_engines();
  dynamic_inputs = true;
  auto status = prepare_for_inference();
  for (const auto& td : in_tensor_dimensions) {
    inference_specs_->dims_per_tensor_.at(td.first) = td.second;
  }
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
  cleanup_engines();
}

TEST_F(HoloInferTests, TRT_MultiDynamicInput_CorrectTrtOptProfile) {
  backend = "trt";
  model_path_map["model_1"] = model_folder + "identity_dynamic_multi.onnx";
  model_path_map["model_2"] = model_folder + "identity_dynamic_multi.onnx";
  in_tensor_dimensions["m1_pre_proc"] = {2, 256, 256};
  in_tensor_dimensions["m2_pre_proc"] = {8, 32, 256};
  batch_sizes["model_1"] = {"1, 2, 8, 32, 128, 256"};
  cleanup_engines();
  dynamic_inputs = true;
  auto status = prepare_for_inference();
  for (const auto& td : in_tensor_dimensions) {
    inference_specs_->dims_per_tensor_.at(td.first) = td.second;
  }
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
  cleanup_engines();
}

// =============================================================================
// ONNX backend tests
// =============================================================================

#if defined(HOLOINFER_ORT_ENABLED)

TEST_F(HoloInferTests, ONNX_BasicParallelEndToEndCudaInference) {
  backend = "onnxrt";
  input_on_cuda = true;
  output_on_cuda = true;
  infer_on_cpu = false;
  auto status = prepare_for_inference();
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, ONNX_InputOnHostCudaInference) {
  backend = "onnxrt";
  input_on_cuda = false;
  output_on_cuda = true;
  infer_on_cpu = false;
  auto status = prepare_for_inference();
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, ONNX_OutputOnHostCudaInference) {
  backend = "onnxrt";
  input_on_cuda = true;
  output_on_cuda = false;
  infer_on_cpu = false;
  auto status = prepare_for_inference();
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, ONNX_BasicParallelInferenceOnCPU) {
  backend = "onnxrt";
  input_on_cuda = false;
  output_on_cuda = false;
  infer_on_cpu = true;
  auto status = prepare_for_inference();
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, ONNX_InputOutputOnDeviceCPUInference) {
  backend = "onnxrt";
  input_on_cuda = true;
  output_on_cuda = true;
  infer_on_cpu = true;
  auto status = prepare_for_inference();
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, ONNX_BasicSequentialInferenceOnCPU) {
  backend = "onnxrt";
  input_on_cuda = false;
  output_on_cuda = false;
  infer_on_cpu = true;
  parallel_inference = false;
  auto status = prepare_for_inference();
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, ONNX_BasicSequentialInferenceOnGPU) {
  backend = "onnxrt";
  input_on_cuda = false;
  output_on_cuda = false;
  infer_on_cpu = false;
  parallel_inference = false;
  auto status = prepare_for_inference();
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, ONNX_BasicParallelInferenceOnGPU) {
  backend = "onnxrt";
  input_on_cuda = false;
  output_on_cuda = false;
  infer_on_cpu = false;
  parallel_inference = true;
  auto status = prepare_for_inference();
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, ONNX_EmptyHostInput) {
  backend = "onnxrt";
  input_on_cuda = false;
  output_on_cuda = false;
  infer_on_cpu = false;
  auto status = prepare_for_inference();
  inference_specs_->data_per_tensor_.at("m1_pre_proc")->host_buffer_->resize(0);
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, ONNX_EmptyHostOutput) {
  backend = "onnxrt";
  input_on_cuda = false;
  output_on_cuda = false;
  infer_on_cpu = false;
  auto status = prepare_for_inference();
  inference_specs_->output_per_model_.at("m2_infer")->host_buffer_->resize(0);
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

// ONNX multi-GPU: run if second GPU available, otherwise test single-GPU error
TEST_F(HoloInferTests, ONNX_BasicSequentialInferenceMultiGPU) {
  cudaDeviceProp device_prop;
  if (cudaGetDeviceProperties(&device_prop, 1) != cudaSuccess) {
    cudaGetLastError();
    GTEST_SKIP() << "Second GPU not available";
  }
  backend = "onnxrt";
  parallel_inference = false;
  device_map.at("model_2") = "1";
  auto status = prepare_for_inference();
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, ONNX_BasicParallelInferenceMultiGPU) {
  cudaDeviceProp device_prop;
  if (cudaGetDeviceProperties(&device_prop, 1) != cudaSuccess) {
    cudaGetLastError();
    GTEST_SKIP() << "Second GPU not available";
  }
  backend = "onnxrt";
  parallel_inference = true;
  device_map.at("model_2") = "1";
  auto status = prepare_for_inference();
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, ONNX_InferenceSingleGPUWithMultiGPUSettings) {
  cudaDeviceProp device_prop;
  if (cudaGetDeviceProperties(&device_prop, 1) == cudaSuccess) {
    GTEST_SKIP() << "This test only runs on single-GPU systems";
  }
  cudaGetLastError();
  backend = "onnxrt";
  device_map.at("model_2") = "1";
  auto status = prepare_for_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

// ONNX dynamic input tests
TEST_F(HoloInferTests, ONNX_DynamicInput_WithIncorrectFlag) {
  backend = "onnxrt";
  model_path_map["model_1"] = model_folder + "identity_model_dynamic.onnx";
  model_path_map["model_2"] = model_folder + "identity_model_dynamic.onnx";
  in_tensor_dimensions["m1_pre_proc"] = {2, 256, 256};
  in_tensor_dimensions["m2_pre_proc"] = {8, 256, 256};
  dynamic_inputs = false;  // incorrect — model is dynamic but flag is off
  auto status = prepare_for_inference();
  for (const auto& td : in_tensor_dimensions) {
    inference_specs_->dims_per_tensor_.at(td.first) = td.second;
  }
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, ONNX_DynamicInput_WithCorrectFlag) {
  backend = "onnxrt";
  model_path_map["model_1"] = model_folder + "identity_model_dynamic.onnx";
  model_path_map["model_2"] = model_folder + "identity_model_dynamic.onnx";
  in_tensor_dimensions["m1_pre_proc"] = {2, 256, 256};
  in_tensor_dimensions["m2_pre_proc"] = {8, 256, 256};
  dynamic_inputs = true;
  auto status = prepare_for_inference();
  for (const auto& td : in_tensor_dimensions) {
    inference_specs_->dims_per_tensor_.at(td.first) = td.second;
  }
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

#endif  // HOLOINFER_ORT_ENABLED

// =============================================================================
// Torch backend tests
// =============================================================================

#if defined(HOLOINFER_TORCH_ENABLED)

TEST_F(HoloInferTests, Torch_DynamicInput_WithIncorrectFlag) {
  if (!HoloInfer::is_torch_cuda_sm_compatible()) {
    GTEST_SKIP() << "Torch CUDA unavailable or SM incompatible";
  }
  model_path_map["model_1"] = model_folder + "torch_dynamic_test.pt";
  model_path_map["model_2"] = model_folder + "torch_dynamic_test.pt";
  in_tensor_dimensions["m1_pre_proc"] = {2, 256, 256};
  in_tensor_dimensions["m2_pre_proc"] = {256, 256};
  dynamic_inputs = false;  // incorrect flag
  backend = "torch";
  auto status = prepare_for_inference();
  for (const auto& td : in_tensor_dimensions) {
    inference_specs_->dims_per_tensor_.at(td.first) = td.second;
  }
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Torch_DynamicInput_WithCorrectFlag) {
  if (!HoloInfer::is_torch_cuda_sm_compatible()) {
    GTEST_SKIP() << "Torch CUDA unavailable or SM incompatible";
  }
  model_path_map["model_1"] = model_folder + "torch_dynamic_test.pt";
  model_path_map["model_2"] = model_folder + "torch_dynamic_test.pt";
  in_tensor_dimensions["m1_pre_proc"] = {2, 256, 256};
  in_tensor_dimensions["m2_pre_proc"] = {256, 256};
  dynamic_inputs = true;
  backend = "torch";
  auto status = prepare_for_inference();
  for (const auto& td : in_tensor_dimensions) {
    inference_specs_->dims_per_tensor_.at(td.first) = td.second;
  }
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

// Parameterized Torch policy tests
struct TorchPolicyParam {
  int test_id;
  const char* policy_name;
};

class HoloInferTorchPolicyTests : public HoloInferTests,
                                  public ::testing::WithParamInterface<TorchPolicyParam> {
 protected:
  void SetUp() override {
    HoloInferTests::SetUp();
    if (!HoloInfer::is_torch_cuda_sm_compatible()) {
      GTEST_SKIP() << "Torch CUDA unavailable or SM incompatible";
    }
  }
};

TEST_P(HoloInferTorchPolicyTests, RunPolicy) {
  const auto& param = GetParam();
  backend = "torch";

  std::string model_path = model_folder + "test_torch_backend/" + param.policy_name + ".pt";
  std::string policy_yaml_path = model_folder + "test_torch_backend/" + param.policy_name + ".yaml";

  ASSERT_TRUE(std::filesystem::exists(model_path)) << "Model file not found: " << model_path;
  ASSERT_TRUE(std::filesystem::exists(policy_yaml_path))
      << "Policy YAML not found: " << policy_yaml_path;

  YAML::Node policy_yaml = YAML::LoadFile(policy_yaml_path);

  in_tensor_names.clear();
  out_tensor_names.clear();
  in_tensor_dimensions.clear();

  for (const auto& input_node : policy_yaml["inference"]["input_nodes"]) {
    std::string node_name = input_node.first.as<std::string>();
    in_tensor_names.push_back(node_name);
    std::string dim_str = input_node.second["dim"].as<std::string>();
    std::vector<int> dimensions;
    if (!dim_str.empty()) {
      std::istringstream iss(dim_str);
      std::string token;
      while (iss >> token) {
        dimensions.push_back(std::stoi(token));
      }
    }
    in_tensor_dimensions[node_name] = dimensions;
  }

  for (const auto& output_node : policy_yaml["inference"]["output_nodes"]) {
    out_tensor_names.push_back(output_node.first.as<std::string>());
  }

  model_path_map = {{param.policy_name, model_path}};
  inference_map = {{param.policy_name, out_tensor_names}};
  pre_processor_map = {{param.policy_name, in_tensor_names}};
  device_map = {};

  auto status = prepare_for_inference();
  status = do_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(TorchPolicyTests, HoloInferTorchPolicyTests,
                         ::testing::Values(TorchPolicyParam{38, "simple_policy"},
                                           TorchPolicyParam{39, "dict_input_policy"},
                                           TorchPolicyParam{40, "list_input_policy"},
                                           TorchPolicyParam{41, "tuple_output_policy"},
                                           TorchPolicyParam{42, "nested_list_policy"},
                                           TorchPolicyParam{43, "nested_dict_policy"},
                                           TorchPolicyParam{44, "nested_list_and_dict_policy"},
                                           TorchPolicyParam{45, "heterogeneous_io_policy"}),
                         [](const ::testing::TestParamInfo<TorchPolicyParam>& info) {
                           return info.param.policy_name;
                         });

#endif  // HOLOINFER_TORCH_ENABLED

// =============================================================================
// Dependency map tests
// =============================================================================

TEST_F(HoloInferTests, DependencyMap_LinearPlanOrder) {
  auto pre_backup = pre_processor_map;
  auto inf_backup = inference_map;

  pre_processor_map["model_1"] = {"m1_pre_proc"};
  pre_processor_map["model_2"] = {"m1_infer"};  // depends on model_1 output
  inference_map["model_1"] = {"m1_infer"};
  inference_map["model_2"] = {"m2_infer"};

  std::vector<std::vector<std::string>> plan;
  auto dep_status = HoloInfer::build_execution_plan(pre_processor_map, inference_map, plan);

  if (dep_status.get_code() == HoloInfer::holoinfer_code::H_SUCCESS) {
    bool ok = (plan.size() == 2 && plan[0].size() == 1 && plan[1].size() == 1 &&
               plan[0][0] == "model_1" && plan[1][0] == "model_2");
    if (!ok) {
      dep_status.set_code(HoloInfer::holoinfer_code::H_ERROR);
      dep_status.set_message("Unexpected execution plan ordering");
    }
  }

  HOLOINFER_EXPECT_STATUS(dep_status, HoloInfer::holoinfer_code::H_SUCCESS);

  pre_processor_map = std::move(pre_backup);
  inference_map = std::move(inf_backup);
}

TEST_F(HoloInferTests, DependencyMap_CycleDetection) {
  auto pre_backup = pre_processor_map;
  auto inf_backup = inference_map;

  pre_processor_map["model_1"] = {"m2_infer"};  // circular: model_1 depends on model_2
  pre_processor_map["model_2"] = {"m1_infer"};  // circular: model_2 depends on model_1
  inference_map["model_1"] = {"m1_infer"};
  inference_map["model_2"] = {"m2_infer"};

  std::vector<std::vector<std::string>> plan;
  auto dep_status = HoloInfer::build_execution_plan(pre_processor_map, inference_map, plan);
  HOLOINFER_EXPECT_STATUS(dep_status, HoloInfer::holoinfer_code::H_ERROR);

  pre_processor_map = std::move(pre_backup);
  inference_map = std::move(inf_backup);
}
