/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>
#include <utility>

// Checks parameter setup and values before creating inference specifications
void HoloInferTests::parameter_test_inference() {
  std::string test_module = "Parameter test inference";
  auto status = HoloInfer::InferStatus();
  // Check for the validity of parameters from configuration

  // model_path_map tests
  // Test: Parameters, model_path_map: dummy path test
  model_path_map.insert({"test-dummy", "path-dummy"});
  status = call_parameter_check_inference();
  holoinfer_assert(
      status, test_module, 1, test_identifier_params.at(1), HoloInfer::holoinfer_code::H_ERROR);

  // Test: Parameters, model_path_map: key mismatch with pre_processor_map
  model_path_map.at("test-dummy") = model_path_map.at("model_1");
  status = call_parameter_check_inference();
  holoinfer_assert(
      status, test_module, 2, test_identifier_params.at(2), HoloInfer::holoinfer_code::H_ERROR);
  model_path_map.erase("test-dummy");

  // pre_processor_map tests
  // Test: Parameters, pre_processor_map mismatch check with model_path_map
  pre_processor_map.insert({"test-dummy", {"data-dummy"}});
  status = call_parameter_check_inference();
  holoinfer_assert(
      status, test_module, 3, test_identifier_params.at(3), HoloInfer::holoinfer_code::H_ERROR);

  // Test: Parameters, pre_processor_map empty value vector check
  pre_processor_map.erase("test-dummy");
  auto str_value = pre_processor_map.at("model_1")[0];
  pre_processor_map.at("model_1").pop_back();
  status = call_parameter_check_inference();
  holoinfer_assert(
      status, test_module, 4, test_identifier_params.at(4), HoloInfer::holoinfer_code::H_ERROR);
  pre_processor_map.at("model_1").push_back(str_value);

  // Test: Parameters, pre_processor_map empty tensor name check
  pre_processor_map.at("model_1").push_back("");
  status = call_parameter_check_inference();
  holoinfer_assert(
      status, test_module, 5, test_identifier_params.at(5), HoloInfer::holoinfer_code::H_ERROR);
  pre_processor_map.at("model_1").pop_back();

  // Test: Parameters, pre_processor_map duplicate tensor name check
  pre_processor_map.at("model_1").push_back(str_value);
  status = call_parameter_check_inference();
  holoinfer_assert(
      status, test_module, 6, test_identifier_params.at(6), HoloInfer::holoinfer_code::H_ERROR);
  pre_processor_map.at("model_1").pop_back();

  // input tensor names test
  // Test: Parameters, input_tensor exist in pre_processor_map
  in_tensor_names.push_back("dummy-input");
  status = call_parameter_check_inference();
  holoinfer_assert(
      status, test_module, 7, test_identifier_params.at(7), HoloInfer::holoinfer_code::H_ERROR);
  in_tensor_names.pop_back();

  // Test: Parameters, input_tensor is unique
  in_tensor_names.push_back(in_tensor_names[0]);
  status = call_parameter_check_inference();
  holoinfer_assert(
      status, test_module, 8, test_identifier_params.at(8), HoloInfer::holoinfer_code::H_ERROR);
  in_tensor_names.pop_back();

  // inference_map tests
  // Test: Parameters, inference_map mismatch check
  inference_map.insert({"test-dummy", {"data-dummy"}});
  status = call_parameter_check_inference();
  holoinfer_assert(
      status, test_module, 9, test_identifier_params.at(9), HoloInfer::holoinfer_code::H_ERROR);
  inference_map.erase("test-dummy");

  // Test: Parameters, inference_map duplicate entry check
  str_value = inference_map.at("model_1")[0];
  inference_map.at("model_1").push_back(str_value);
  status = call_parameter_check_inference();
  holoinfer_assert(
      status, test_module, 10, test_identifier_params.at(10), HoloInfer::holoinfer_code::H_ERROR);
  inference_map.at("model_1").pop_back();

  // output tensor names test
  // Test: Parameters, output_tensor exist in inference_map
  out_tensor_names.push_back("dummy-output");
  status = call_parameter_check_inference();
  holoinfer_assert(
      status, test_module, 11, test_identifier_params.at(11), HoloInfer::holoinfer_code::H_ERROR);
  out_tensor_names.pop_back();

  // Test: Parameters, output_tensor is unique
  out_tensor_names.push_back(out_tensor_names[0]);
  status = call_parameter_check_inference();
  holoinfer_assert(
      status, test_module, 12, test_identifier_params.at(12), HoloInfer::holoinfer_code::H_ERROR);
  out_tensor_names.pop_back();

  // Test: Parameters, Input parameter set check
  status = call_parameter_check_inference();
  holoinfer_assert(
      status, test_module, 13, test_identifier_params.at(13), HoloInfer::holoinfer_code::H_SUCCESS);
}

// Tests during inference specification creation and parameters setup
void HoloInferTests::parameter_setup_test() {
  std::string test_module = "Inference setup";

  // model_path_map and inference_map tests start
  // Test: TRT backend, Empty model path check
  auto mmm = std::move(model_path_map);
  auto status = create_specifications();
  clear_specs();
  model_path_map = std::move(mmm);
  holoinfer_assert(
      status, test_module, 14, test_identifier_params.at(14), HoloInfer::holoinfer_code::H_ERROR);

  // Test: TRT backend, Inference map key mismatch with model path map
  model_path_map["test-dummy"] = model_path_map.at("model_1");
  status = create_specifications();
  clear_specs();
  model_path_map.erase("test-dummy");
  holoinfer_assert(
      status, test_module, 15, test_identifier_params.at(15), HoloInfer::holoinfer_code::H_ERROR);

  // Test: TRT backend, Inference map, missing entry
  auto imap = std::move(inference_map);
  status = create_specifications();
  clear_specs();
  inference_map = std::move(imap);
  holoinfer_assert(
      status, test_module, 16, test_identifier_params.at(16), HoloInfer::holoinfer_code::H_ERROR);
  // model_path_map and inference_map tests end

  // Test: TRT backend, CPU based inference
  infer_on_cpu = true;
  status = create_specifications();
  clear_specs();
  infer_on_cpu = false;
  holoinfer_assert(
      status, test_module, 17, test_identifier_params.at(17), HoloInfer::holoinfer_code::H_ERROR);

  // backend tests start
  // Test: TRT backend, Backend type
  backend = "tensorflow";
  status = create_specifications();
  clear_specs();
  holoinfer_assert(
      status, test_module, 18, test_identifier_params.at(18), HoloInfer::holoinfer_code::H_ERROR);

#if use_torch
  // Test: Torch backend, incorrect model file format
  backend = "torch";
  status = create_specifications();
  clear_specs();
  holoinfer_assert(
      status, test_module, 19, test_identifier_params.at(19), HoloInfer::holoinfer_code::H_ERROR);

  auto backup_path_map = std::move(model_path_map);
  auto backup_pre_map = std::move(pre_processor_map);
  auto backup_infer_map = std::move(inference_map);
  auto backup_device_map = std::move(device_map);

  model_path_map = {{"test_model", model_folder + "model.pt"}};
  pre_processor_map = {{"test_model", {"input_"}}};
  inference_map = {{"test_model", {"output_"}}};
  device_map = {};

  // Test: Torch backend, Model file missing
  status = create_specifications();
  clear_specs();
  holoinfer_assert(
      status, test_module, 25, test_identifier_params.at(25), HoloInfer::holoinfer_code::H_ERROR);

  // Test: Torch backend, Config file missing
  std::filesystem::rename(model_folder + "identity_model.pt", model_folder + "model.pt");
  status = create_specifications();
  clear_specs();
  holoinfer_assert(
      status, test_module, 26, test_identifier_params.at(26), HoloInfer::holoinfer_code::H_ERROR);

  // Test: Torch backend, Inference node missing in Config file
  std::ofstream torch_config_file(model_folder + "model.yaml");
  status = create_specifications();
  clear_specs();
  holoinfer_assert(
      status, test_module, 27, test_identifier_params.at(27), HoloInfer::holoinfer_code::H_ERROR);

  YAML::Node torch_inference;

  // Test: Torch backend, Input node missing in Config file
  torch_inference["inference"]["test_node"] = "test";
  torch_config_file << torch_inference;
  std::cout << torch_inference << std::endl;
  torch_config_file.close();
  status = create_specifications();
  clear_specs();
  holoinfer_assert(
      status, test_module, 28, test_identifier_params.at(28), HoloInfer::holoinfer_code::H_ERROR);

  // Test: Torch backend, dtype missing in input node in Config file
  torch_config_file.open(model_folder + "model.yaml", std::ofstream::trunc);
  torch_inference["inference"]["input_nodes"]["input"]["id"] = "1";
  std::cout << torch_inference << std::endl;
  torch_config_file << torch_inference;
  torch_config_file.close();
  status = create_specifications();
  clear_specs();
  holoinfer_assert(
      status, test_module, 29, test_identifier_params.at(29), HoloInfer::holoinfer_code::H_ERROR);

  // Test: Torch backend, Incorrect dtype in config file
  torch_config_file.open(model_folder + "model.yaml", std::ofstream::trunc);
  torch_inference["inference"]["input_nodes"]["input"]["dtype"] = "float";
  torch_config_file << torch_inference;
  torch_config_file.close();
  status = create_specifications();
  clear_specs();
  holoinfer_assert(
      status, test_module, 30, test_identifier_params.at(30), HoloInfer::holoinfer_code::H_ERROR);

  // Test: Torch backend, Output node missing in config file correct
  torch_config_file.open(model_folder + "model.yaml", std::ofstream::trunc);
  torch_inference["inference"]["input_nodes"]["input"]["dtype"] = "kFloat32";
  torch_config_file << torch_inference;
  torch_config_file.close();
  status = create_specifications();
  clear_specs();
  holoinfer_assert(
      status, test_module, 31, test_identifier_params.at(31), HoloInfer::holoinfer_code::H_ERROR);

  // Restore all changes to previous state
  std::filesystem::rename(model_folder + "model.pt", model_folder + "identity_model.pt");
  std::filesystem::remove(model_folder + "model.yaml");
  model_path_map = std::move(backup_path_map);
  pre_processor_map = std::move(backup_pre_map);
  inference_map = std::move(backup_infer_map);
  device_map = std::move(backup_device_map);
#endif

  if (use_onnxruntime) {
    // Test: ONNX backend, Input/Output cuda buffer test
    backend = "onnxrt";
    status = create_specifications();
    clear_specs();
    holoinfer_assert(status,
                     test_module,
                     20,
                     test_identifier_params.at(20),
                     HoloInfer::holoinfer_code::H_SUCCESS);

    // Test: ONNX backend, incorrect model file format
    backend = "onnxrt";
    auto pc_path = model_path_map.at("model_1");
    model_path_map.at("model_1") = "model.engine";
    input_on_cuda = false;
    output_on_cuda = false;
    status = create_specifications();
    clear_specs();
    model_path_map.at("model_1") = pc_path;
    holoinfer_assert(
        status, test_module, 21, test_identifier_params.at(21), HoloInfer::holoinfer_code::H_ERROR);

    // Test: ONNX backend, Engine path true test
    is_engine_path = true;
    status = create_specifications();
    clear_specs();
    holoinfer_assert(
        status, test_module, 22, test_identifier_params.at(22), HoloInfer::holoinfer_code::H_ERROR);

    // Test: ONNX backend, Default
    is_engine_path = false;
    input_on_cuda = false;
    output_on_cuda = false;
    status = create_specifications();
    clear_specs();
    holoinfer_assert(status,
                     test_module,
                     23,
                     test_identifier_params.at(23),
                     HoloInfer::holoinfer_code::H_SUCCESS);
  }

  // Test: TRT backend, Default check 1
  backend = "trt";
  infer_on_cpu = false;
  input_on_cuda = true;
  output_on_cuda = true;
  status = create_specifications();
  clear_specs();
  holoinfer_assert(
      status, test_module, 24, test_identifier_params.at(24), HoloInfer::holoinfer_code::H_SUCCESS);

  // Test: TRT backend, disable CUDA Graphs
  backend = "trt";
  use_cuda_graphs = false;
  status = create_specifications();
  clear_specs();
  holoinfer_assert(
      status, test_module, 32, test_identifier_params.at(32), HoloInfer::holoinfer_code::H_SUCCESS);
  use_cuda_graphs = true;
}
