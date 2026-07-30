/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_core.hpp"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <string>
#include <utility>

// =============================================================================
// Parameter validity check tests (originally parameter_test_inference)
// =============================================================================

TEST_F(HoloInferTests, Params_ModelPathMap_DummyPath) {
  model_path_map.insert({"test-dummy", "path-dummy"});
  auto status = call_parameter_check_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Params_ModelPathMap_KeyMismatchWithPreProcessorMap) {
  model_path_map.insert({"test-dummy", model_path_map.at("model_1")});
  auto status = call_parameter_check_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Params_PreProcessorMap_MismatchWithModelPathMap) {
  pre_processor_map.insert({"test-dummy", {"data-dummy"}});
  auto status = call_parameter_check_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Params_PreProcessorMap_EmptyValueVector) {
  pre_processor_map.at("model_1").pop_back();
  auto status = call_parameter_check_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Params_PreProcessorMap_EmptyTensorName) {
  pre_processor_map.at("model_1").push_back("");
  auto status = call_parameter_check_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Params_PreProcessorMap_DuplicateTensorName) {
  auto str_value = pre_processor_map.at("model_1")[0];
  pre_processor_map.at("model_1").push_back(str_value);
  auto status = call_parameter_check_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Params_InputTensor_ExistInPreProcessorMap) {
  in_tensor_names.push_back("dummy-input");
  auto status = call_parameter_check_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Params_InputTensor_IsUnique) {
  in_tensor_names.push_back(in_tensor_names[0]);
  auto status = call_parameter_check_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Params_InferenceMap_MismatchCheck) {
  inference_map.insert({"test-dummy", {"data-dummy"}});
  auto status = call_parameter_check_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Params_InferenceMap_DuplicateEntry) {
  auto str_value = inference_map.at("model_1")[0];
  inference_map.at("model_1").push_back(str_value);
  auto status = call_parameter_check_inference();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Params_OutputTensor_ExistInInferenceMap) {
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

TEST_F(HoloInferTests, Setup_TRT_InferenceMapKeyMismatchWithModelPathMap) {
  model_path_map["test-dummy"] = model_path_map.at("model_1");
  auto status = create_specifications();
  clear_specs();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Setup_TRT_InferenceMapMissingEntry) {
  auto imap = std::move(inference_map);
  auto status = create_specifications();
  clear_specs();
  inference_map = std::move(imap);
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTests, Setup_TRT_CpuBasedInference) {
  infer_on_cpu = true;
  auto status = create_specifications();
  clear_specs();
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

#if defined(HOLOINFER_ORT_ENABLED)
TEST_F(HoloInferOnnxRuntimeTests, Setup_ONNX_InputOutputCudaBuffer) {
  backend = "onnxrt";
  auto status = create_specifications();
  clear_specs();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}

TEST_F(HoloInferOnnxRuntimeTests, Setup_ONNX_IncorrectModelFileFormat) {
  backend = "onnxrt";
  auto pc_path = model_path_map.at("model_1");
  model_path_map.at("model_1") = "model.engine";
  input_on_cuda = false;
  output_on_cuda = false;
  auto status = create_specifications();
  clear_specs();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferOnnxRuntimeTests, Setup_ONNX_EnginePathTrue) {
  backend = "onnxrt";
  is_engine_path = true;
  auto status = create_specifications();
  clear_specs();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferOnnxRuntimeTests, Setup_ONNX_Default) {
  backend = "onnxrt";
  input_on_cuda = false;
  output_on_cuda = false;
  auto status = create_specifications();
  clear_specs();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_SUCCESS);
}
#endif  // HOLOINFER_ORT_ENABLED

#if defined(HOLOINFER_TORCH_ENABLED)
TEST_F(HoloInferTests, Setup_Torch_IncorrectModelFileFormat) {
  backend = "torch";
  auto status = create_specifications();
  clear_specs();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

// Fixture for Torch config-file validation tests.
// Sets up a single-model torch configuration and cleans up temp files.
class HoloInferTorchConfigTests : public HoloInferTests {
 protected:
  void SetUp() override {
    HoloInferTests::SetUp();
    backend = "torch";
    model_path_map = {{"test_model", model_folder + "model.pt"}};
    pre_processor_map = {{"test_model", {"input_"}}};
    inference_map = {{"test_model", {"output_"}}};
    device_map = {};
  }

  void TearDown() override {
    std::filesystem::remove(model_folder + "model.pt");
    std::filesystem::remove(model_folder + "model.yaml");
    HoloInferTests::TearDown();
  }
};

TEST_F(HoloInferTorchConfigTests, Setup_Torch_ModelFileMissing) {
  // model.pt does not exist
  auto status = create_specifications();
  clear_specs();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTorchConfigTests, Setup_Torch_ConfigFileMissing) {
  std::filesystem::copy_file(model_folder + "test_torch_backend/simple_policy.pt",
                             model_folder + "model.pt",
                             std::filesystem::copy_options::overwrite_existing);
  // model.yaml does not exist
  auto status = create_specifications();
  clear_specs();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTorchConfigTests, Setup_Torch_InferenceMissingInConfigFile) {
  std::filesystem::copy_file(model_folder + "test_torch_backend/simple_policy.pt",
                             model_folder + "model.pt",
                             std::filesystem::copy_options::overwrite_existing);
  // Empty YAML — no inference node
  std::ofstream yaml_file(model_folder + "model.yaml");
  yaml_file.close();
  auto status = create_specifications();
  clear_specs();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTorchConfigTests, Setup_Torch_InputNodeMissingInConfigFile) {
  std::filesystem::copy_file(model_folder + "test_torch_backend/simple_policy.pt",
                             model_folder + "model.pt",
                             std::filesystem::copy_options::overwrite_existing);
  YAML::Node config;
  config["inference"]["test_node"] = "test";
  std::ofstream yaml_file(model_folder + "model.yaml");
  yaml_file << config;
  yaml_file.close();
  auto status = create_specifications();
  clear_specs();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTorchConfigTests, Setup_Torch_DtypeMissingInInputNode) {
  std::filesystem::copy_file(model_folder + "test_torch_backend/simple_policy.pt",
                             model_folder + "model.pt",
                             std::filesystem::copy_options::overwrite_existing);
  YAML::Node config;
  config["inference"]["input_nodes"]["0"]["dim"] = "3";
  std::ofstream yaml_file(model_folder + "model.yaml");
  yaml_file << config;
  yaml_file.close();
  auto status = create_specifications();
  clear_specs();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTorchConfigTests, Setup_Torch_IncorrectDtypeInConfigFile) {
  std::filesystem::copy_file(model_folder + "test_torch_backend/simple_policy.pt",
                             model_folder + "model.pt",
                             std::filesystem::copy_options::overwrite_existing);
  YAML::Node config;
  config["inference"]["input_nodes"]["0"]["dim"] = "3";
  config["inference"]["input_nodes"]["0"]["dtype"] = "float";  // invalid dtype
  std::ofstream yaml_file(model_folder + "model.yaml");
  yaml_file << config;
  yaml_file.close();
  auto status = create_specifications();
  clear_specs();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTorchConfigTests, Setup_Torch_OutputNodeMissingInConfigFile) {
  std::filesystem::copy_file(model_folder + "test_torch_backend/simple_policy.pt",
                             model_folder + "model.pt",
                             std::filesystem::copy_options::overwrite_existing);
  YAML::Node config;
  config["inference"]["input_nodes"]["0"]["dim"] = "3";
  config["inference"]["input_nodes"]["0"]["dtype"] = "kFloat32";
  // No output_nodes
  std::ofstream yaml_file(model_folder + "model.yaml");
  yaml_file << config;
  yaml_file.close();
  auto status = create_specifications();
  clear_specs();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTorchConfigTests, Setup_Torch_UnsupportedInputFormatInConfigFile) {
  std::filesystem::copy_file(model_folder + "test_torch_backend/simple_policy.pt",
                             model_folder + "model.pt",
                             std::filesystem::copy_options::overwrite_existing);
  YAML::Node config;
  config["inference"]["input_nodes"]["0"]["dim"] = "3";
  config["inference"]["input_nodes"]["0"]["dtype"] = "kFloat32";
  config["inference"]["output_nodes"]["tensor_0"]["dim"] = "1";
  config["inference"]["output_nodes"]["tensor_0"]["dtype"] = "kFloat32";
  // 5-level nested list — unsupported format
  YAML::Node nested;
  nested[0][0][0][0][0] = "0";
  config["inference"]["input_format"] = nested;
  std::ofstream yaml_file(model_folder + "model.yaml");
  yaml_file << config;
  yaml_file.close();
  auto status = create_specifications();
  clear_specs();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTorchConfigTests, Setup_Torch_OutermostInputFormatNodeIsNotAList) {
  std::filesystem::copy_file(model_folder + "test_torch_backend/simple_policy.pt",
                             model_folder + "model.pt",
                             std::filesystem::copy_options::overwrite_existing);
  YAML::Node config;
  config["inference"]["input_nodes"]["0"]["dim"] = "3";
  config["inference"]["input_nodes"]["0"]["dtype"] = "kFloat32";
  config["inference"]["output_nodes"]["tensor_0"]["dim"] = "1";
  config["inference"]["output_nodes"]["tensor_0"]["dtype"] = "kFloat32";
  // Dict node as input_format — outermost must be a list
  YAML::Node dict_node;
  dict_node["key"] = "0";
  config["inference"]["input_format"] = dict_node;
  std::ofstream yaml_file(model_folder + "model.yaml");
  yaml_file << config;
  yaml_file.close();
  auto status = create_specifications();
  clear_specs();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}

TEST_F(HoloInferTorchConfigTests, Setup_Torch_InputFormatMismatchInConfigFile) {
  std::filesystem::copy_file(model_folder + "test_torch_backend/simple_policy.pt",
                             model_folder + "model.pt",
                             std::filesystem::copy_options::overwrite_existing);
  YAML::Node config;
  config["inference"]["input_nodes"]["0"]["dim"] = "3";
  config["inference"]["input_nodes"]["0"]["dtype"] = "kFloat32";
  config["inference"]["output_nodes"]["tensor_0"]["dim"] = "1";
  config["inference"]["output_nodes"]["tensor_0"]["dtype"] = "kFloat32";
  // Assign a dict directly as input_format — input_format must be a list, not a map
  YAML::Node dict_node;
  dict_node["key"] = "0";
  config["inference"]["input_format"] = dict_node;
  std::ofstream yaml_file(model_folder + "model.yaml");
  yaml_file << config;
  yaml_file.close();
  auto status = create_specifications();
  clear_specs();
  HOLOINFER_EXPECT_STATUS(status, HoloInfer::holoinfer_code::H_ERROR);
}
#endif  // HOLOINFER_TORCH_ENABLED
