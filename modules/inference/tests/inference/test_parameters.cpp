/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_core.hpp"

#include <utility>

// =============================================================================
// Parameter validity check tests (originally parameter_test_inference)
// =============================================================================

TEST_F(HoloInferTests, Params_ModelPathMap_DummyPath) {
  model_path_map.insert({"test-dummy", "path-dummy"});
  auto status = call_parameter_check_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Params_ModelPathMap_KeyMismatchWithInputMap) {
  model_path_map.insert({"test-dummy", model_path_map.at("model_1")});
  auto status = call_parameter_check_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Params_InputMap_MismatchWithModelPathMap) {
  input_map.insert({"test-dummy", {"data-dummy"}});
  auto status = call_parameter_check_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Params_InputMap_EmptyValueVector) {
  input_map.at("model_1").pop_back();
  auto status = call_parameter_check_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Params_InputMap_EmptyTensorName) {
  input_map.at("model_1").push_back("");
  auto status = call_parameter_check_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Params_InputMap_DuplicateTensorName) {
  auto str_value = input_map.at("model_1")[0];
  input_map.at("model_1").push_back(str_value);
  auto status = call_parameter_check_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Params_InputTensor_ExistInInputMap) {
  in_tensor_names.push_back("dummy-input");
  auto status = call_parameter_check_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Params_InputTensor_IsUnique) {
  in_tensor_names.push_back(in_tensor_names[0]);
  auto status = call_parameter_check_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Params_OutputMap_MismatchCheck) {
  output_map.insert({"test-dummy", {"data-dummy"}});
  auto status = call_parameter_check_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Params_OutputMap_DuplicateEntry) {
  auto str_value = output_map.at("model_1")[0];
  output_map.at("model_1").push_back(str_value);
  auto status = call_parameter_check_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Params_OutputTensor_ExistInOutputMap) {
  out_tensor_names.push_back("dummy-output");
  auto status = call_parameter_check_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Params_OutputTensor_IsUnique) {
  out_tensor_names.push_back(out_tensor_names[0]);
  auto status = call_parameter_check_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Params_InputParameterSetCheck) {
  auto status = call_parameter_check_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

// =============================================================================
// Inference specification setup tests (originally parameter_setup_test)
// =============================================================================

TEST_F(HoloInferTests, Setup_TRT_EmptyModelPath) {
  auto mmm = std::move(model_path_map);
  auto status = create_specifications();
  clear_specs();
  model_path_map = std::move(mmm);
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Setup_TRT_OutputMapKeyMismatchWithModelPathMap) {
  model_path_map["test-dummy"] = model_path_map.at("model_1");
  auto status = create_specifications();
  clear_specs();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Setup_TRT_OutputMapMissingEntry) {
  auto imap = std::move(output_map);
  auto status = create_specifications();
  clear_specs();
  output_map = std::move(imap);
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Setup_TRT_BackendType) {
  backend = "tensorflow";
  auto status = create_specifications();
  clear_specs();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Setup_TRT_Default) {
  backend = "trt";
  auto status = create_specifications();
  clear_specs();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferTests, Setup_TRT_DisableCudaGraphs) {
  backend = "trt";
  use_cuda_graphs = false;
  auto status = create_specifications();
  clear_specs();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}
