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

#include <memory>
#include <string>
#include <utility>

void HoloInferTests::inference_tests() {
  std::string test_module = "Inference tests";
  backend = "trt";

  // Test: TRT backend, Empty input data
  auto status = prepare_for_inference();
  auto dmap = std::move(inference_specs_->data_per_tensor_);
  status = do_inference();
  holoinfer_assert(
      status, test_module, 1, test_identifier_infer.at(1), HoloInfer::holoinfer_code::H_ERROR);
  inference_specs_->data_per_tensor_ = std::move(dmap);

  // Test: TRT backend, Empty inference parameters
  clear_specs();
  setup_specifications();
  status = do_inference();
  holoinfer_assert(
      status, test_module, 2, test_identifier_infer.at(2), HoloInfer::holoinfer_code::H_ERROR);

  // Test: TRT backend, Missing input tensor
  status = prepare_for_inference();
  auto dm = std::move(inference_specs_->data_per_tensor_.at("m1_pre_proc"));
  inference_specs_->data_per_tensor_.erase("m1_pre_proc");
  status = do_inference();
  holoinfer_assert(
      status, test_module, 3, test_identifier_infer.at(3), HoloInfer::holoinfer_code::H_ERROR);
  inference_specs_->data_per_tensor_.insert({"m1_pre_proc", dm});

  // Test: TRT backend, Missing output tensor
  status = prepare_for_inference();
  dm = std::move(inference_specs_->output_per_model_.at("m2_infer"));
  inference_specs_->output_per_model_.erase("m2_infer");
  status = do_inference();
  holoinfer_assert(
      status, test_module, 4, test_identifier_infer.at(4), HoloInfer::holoinfer_code::H_ERROR);
  inference_specs_->output_per_model_.insert({"m2_infer", dm});

  // Test: TRT backend, Empty input cuda buffer 1
  auto dbs = inference_specs_->data_per_tensor_.at("m1_pre_proc")->device_buffer_->size();
  inference_specs_->data_per_tensor_.at("m1_pre_proc")->device_buffer_->resize(0);
  inference_specs_->data_per_tensor_.at("m1_pre_proc")->device_buffer_ = nullptr;
  status = do_inference();
  holoinfer_assert(
      status, test_module, 5, test_identifier_infer.at(5), HoloInfer::holoinfer_code::H_ERROR);
  inference_specs_->data_per_tensor_.at("m1_pre_proc")->device_buffer_ =
      std::make_shared<HoloInfer::DeviceBuffer>();

  // Test: TRT backend, Empty input cuda buffer 2
  status = do_inference();
  holoinfer_assert(
      status, test_module, 6, test_identifier_infer.at(6), HoloInfer::holoinfer_code::H_ERROR);
  inference_specs_->data_per_tensor_.at("m1_pre_proc")->device_buffer_->resize(dbs);

  // Test: TRT backend, Empty output cuda buffer 1
  dbs = inference_specs_->output_per_model_.at("m2_infer")->device_buffer_->size();
  inference_specs_->output_per_model_.at("m2_infer")->device_buffer_->resize(0);
  status = do_inference();
  holoinfer_assert(
      status, test_module, 7, test_identifier_infer.at(7), HoloInfer::holoinfer_code::H_ERROR);

  // Test: TRT backend, Empty output cuda buffer 2
  inference_specs_->output_per_model_.at("m2_infer")->device_buffer_ = nullptr;
  status = do_inference();
  holoinfer_assert(
      status, test_module, 8, test_identifier_infer.at(8), HoloInfer::holoinfer_code::H_ERROR);
  inference_specs_->output_per_model_.at("m2_infer")->device_buffer_ =
      std::make_shared<HoloInfer::DeviceBuffer>();

  // Test: TRT backend, Empty output cuda buffer 3
  status = do_inference();
  holoinfer_assert(
      status, test_module, 9, test_identifier_infer.at(9), HoloInfer::holoinfer_code::H_ERROR);
  inference_specs_->output_per_model_.at("m2_infer")->device_buffer_->resize(dbs);

  // Test: TRT backend, Basic end-to-end cuda inference
  status = do_inference();
  holoinfer_assert(
      status, test_module, 10, test_identifier_infer.at(10), HoloInfer::holoinfer_code::H_SUCCESS);

  // Test: TRT backend, Basic sequential end-to-end cuda inference
  parallel_inference = false;
  status = prepare_for_inference();
  status = do_inference();
  holoinfer_assert(
      status, test_module, 11, test_identifier_infer.at(11), HoloInfer::holoinfer_code::H_SUCCESS);

  // Test: TRT backend, Input on host inference
  input_on_cuda = false;
  parallel_inference = true;
  status = prepare_for_inference();
  status = do_inference();
  holoinfer_assert(
      status, test_module, 12, test_identifier_infer.at(12), HoloInfer::holoinfer_code::H_SUCCESS);

  // Test: TRT backend, Output on host inference
  input_on_cuda = true;
  output_on_cuda = false;
  status = prepare_for_inference();
  status = do_inference();
  holoinfer_assert(
      status, test_module, 13, test_identifier_infer.at(13), HoloInfer::holoinfer_code::H_SUCCESS);

  // Test: TRT backend, Input/Output on host inference
  input_on_cuda = false;
  output_on_cuda = false;
  status = prepare_for_inference();
  status = do_inference();
  holoinfer_assert(
      status, test_module, 14, test_identifier_infer.at(14), HoloInfer::holoinfer_code::H_SUCCESS);

  // Test: TRT backend, Empty host input
  size_t re_dbs = 0;
  dbs = inference_specs_->data_per_tensor_.at("m1_pre_proc")->host_buffer_->size();
  inference_specs_->data_per_tensor_.at("m1_pre_proc")->host_buffer_->resize(re_dbs);
  status = do_inference();
  holoinfer_assert(
      status, test_module, 15, test_identifier_infer.at(15), HoloInfer::holoinfer_code::H_ERROR);
  inference_specs_->data_per_tensor_.at("m1_pre_proc")->host_buffer_->resize(dbs);

  // Test: TRT backend, Empty host output
  dbs = inference_specs_->output_per_model_.at("m2_infer")->host_buffer_->size();
  inference_specs_->output_per_model_.at("m2_infer")->host_buffer_->resize(re_dbs);
  status = do_inference();
  holoinfer_assert(
      status, test_module, 16, test_identifier_infer.at(16), HoloInfer::holoinfer_code::H_ERROR);
  inference_specs_->output_per_model_.at("m2_infer")->host_buffer_->resize(dbs);

  if (use_onnxruntime) {
    backend = "onnxrt";

    // Test: ONNX backend, Basic parallel end-to-end cuda inference
    input_on_cuda = true;
    output_on_cuda = true;
    infer_on_cpu = false;
    status = prepare_for_inference();
    status = do_inference();
    holoinfer_assert(status,
                     test_module,
                     35,
                     test_identifier_infer.at(35),
                     HoloInfer::holoinfer_code::H_SUCCESS);

    // Test: ONNX backend, Input on host, cuda inference
    input_on_cuda = false;
    status = prepare_for_inference();
    status = do_inference();
    holoinfer_assert(status,
                     test_module,
                     36,
                     test_identifier_infer.at(36),
                     HoloInfer::holoinfer_code::H_SUCCESS);

    // Test: ONNX backend, Output on host, cuda inference
    input_on_cuda = true;
    output_on_cuda = false;
    status = prepare_for_inference();
    status = do_inference();
    holoinfer_assert(status,
                     test_module,
                     37,
                     test_identifier_infer.at(37),
                     HoloInfer::holoinfer_code::H_SUCCESS);

    // Test: ONNX backend, Basic parallel inference on CPU
    input_on_cuda = false;
    output_on_cuda = false;
    infer_on_cpu = true;
    status = prepare_for_inference();
    status = do_inference();
    holoinfer_assert(status,
                     test_module,
                     17,
                     test_identifier_infer.at(17),
                     HoloInfer::holoinfer_code::H_SUCCESS);

    // Test: ONNX backend, Input and output on device, CPU inference
    input_on_cuda = true;
    output_on_cuda = true;
    infer_on_cpu = true;
    status = prepare_for_inference();
    status = do_inference();
    holoinfer_assert(status,
                     test_module,
                     38,
                     test_identifier_infer.at(38),
                     HoloInfer::holoinfer_code::H_SUCCESS);

    // Test: ONNX backend, Basic sequential inference on CPU
    input_on_cuda = false;
    output_on_cuda = false;
    parallel_inference = false;
    status = prepare_for_inference();
    status = do_inference();
    holoinfer_assert(status,
                     test_module,
                     18,
                     test_identifier_infer.at(18),
                     HoloInfer::holoinfer_code::H_SUCCESS);

    // Test: ONNX backend, Basic sequential inference on GPU
    infer_on_cpu = false;
    status = prepare_for_inference();
    status = do_inference();
    holoinfer_assert(status,
                     test_module,
                     19,
                     test_identifier_infer.at(19),
                     HoloInfer::holoinfer_code::H_SUCCESS);

    // Test: ONNX backend, Basic parallel inference on GPU
    parallel_inference = true;
    status = prepare_for_inference();
    status = do_inference();
    holoinfer_assert(status,
                     test_module,
                     20,
                     test_identifier_infer.at(20),
                     HoloInfer::holoinfer_code::H_SUCCESS);

    // Test: ONNX backend, Empty host input
    dbs = inference_specs_->data_per_tensor_.at("m1_pre_proc")->host_buffer_->size();
    inference_specs_->data_per_tensor_.at("m1_pre_proc")->host_buffer_->resize(0);
    status = do_inference();
    holoinfer_assert(
        status, test_module, 21, test_identifier_infer.at(21), HoloInfer::holoinfer_code::H_ERROR);
    inference_specs_->data_per_tensor_.at("m1_pre_proc")->host_buffer_->resize(dbs);

    // Test: ONNX backend, Empty host output
    dbs = inference_specs_->output_per_model_.at("m2_infer")->host_buffer_->size();
    inference_specs_->output_per_model_.at("m2_infer")->host_buffer_->resize(0);
    status = do_inference();
    holoinfer_assert(
        status, test_module, 22, test_identifier_infer.at(22), HoloInfer::holoinfer_code::H_ERROR);
    inference_specs_->output_per_model_.at("m2_infer")->host_buffer_->resize(dbs);
  }

  // Multi-GPU tests
  cudaDeviceProp device_prop;
  auto dev_id = 1;
  backend = "trt";
  auto cstatus = cudaGetDeviceProperties(&device_prop, dev_id);
  device_map.at("model_1") = "1";

  if (cstatus == cudaSuccess) {
    // Test: TRT backend, Basic sequential inference on multi-GPU
    input_on_cuda = true;
    output_on_cuda = true;
    parallel_inference = false;
    status = prepare_for_inference();
    status = do_inference();
    holoinfer_assert(status,
                     test_module,
                     27,
                     test_identifier_infer.at(27),
                     HoloInfer::holoinfer_code::H_SUCCESS);

    // Test: TRT backend, Basic parallel inference on multi-GPU
    parallel_inference = true;
    status = prepare_for_inference();
    status = do_inference();
    holoinfer_assert(status,
                     test_module,
                     28,
                     test_identifier_infer.at(28),
                     HoloInfer::holoinfer_code::H_SUCCESS);

    // Test: TRT backend, Parallel inference on multi-GPU with I/O on host
    input_on_cuda = false;
    output_on_cuda = false;
    status = prepare_for_inference();
    status = do_inference();
    holoinfer_assert(status,
                     test_module,
                     29,
                     test_identifier_infer.at(29),
                     HoloInfer::holoinfer_code::H_SUCCESS);

    // Test: TRT backend, Parallel inference on multi-GPU with Input on host
    input_on_cuda = false;
    output_on_cuda = true;
    status = prepare_for_inference();
    status = do_inference();
    holoinfer_assert(status,
                     test_module,
                     30,
                     test_identifier_infer.at(30),
                     HoloInfer::holoinfer_code::H_SUCCESS);

    // Test: TRT backend, Parallel inference on multi-GPU with Output on host
    input_on_cuda = true;
    output_on_cuda = false;
    status = prepare_for_inference();
    status = do_inference();
    holoinfer_assert(status,
                     test_module,
                     31,
                     test_identifier_infer.at(31),
                     HoloInfer::holoinfer_code::H_SUCCESS);
  } else {
    // make sure the last error is reset, else Torch tests below will fail since they check for
    // the last error without doing a CUDA call before.
    cudaGetLastError();
  }
  device_map.at("model_1") = "0";

  device_map.at("model_2") = "1";
  if (cstatus == cudaSuccess) {
    // Test: ONNX backend, Basic sequential inference on multi-GPU
    status = prepare_for_inference();
    status = do_inference();
    holoinfer_assert(status,
                     test_module,
                     24,
                     test_identifier_infer.at(24),
                     HoloInfer::holoinfer_code::H_SUCCESS);

    // Test: ONNX backend, Basic parallel inference on multi-GPU
    parallel_inference = true;
    status = prepare_for_inference();
    status = do_inference();
    holoinfer_assert(status,
                     test_module,
                     26,
                     test_identifier_infer.at(26),
                     HoloInfer::holoinfer_code::H_SUCCESS);
  } else {
    // Test: ONNX backend, Inference single GPU with multi-GPU settings
    status = prepare_for_inference();
    holoinfer_assert(
        status, test_module, 25, test_identifier_infer.at(25), HoloInfer::holoinfer_code::H_ERROR);
  }
  device_map.at("model_2") = "0";

  // test multi-rank

  auto original_path = model_path_map["model_1"];
  auto original_dim = in_tensor_dimensions["m1_pre_proc"];

  model_path_map["model_1"] = model_folder + "identity_model_5r.onnx";
  model_path_map["model_2"] = model_folder + "identity_model_5r.onnx";

  in_tensor_dimensions["m1_pre_proc"] = {1, 1, 1, 1, 1};
  in_tensor_dimensions["m2_pre_proc"] = {1, 1, 1, 1, 1};

  status = prepare_for_inference();
  status = do_inference();
  holoinfer_assert(
      status, test_module, 32, test_identifier_infer.at(32), HoloInfer::holoinfer_code::H_SUCCESS);

  model_path_map["model_1"] = model_folder + "identity_model_9r.onnx";
  model_path_map["model_2"] = model_folder + "identity_model_9r.onnx";

  in_tensor_dimensions["m1_pre_proc"] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  in_tensor_dimensions["m2_pre_proc"] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

  status = prepare_for_inference();
  status = do_inference();
  holoinfer_assert(
      status, test_module, 33, test_identifier_infer.at(33), HoloInfer::holoinfer_code::H_ERROR);

  model_path_map["model_1"] = original_path;
  model_path_map["model_2"] = original_path;

  in_tensor_dimensions["m1_pre_proc"] = original_dim;
  in_tensor_dimensions["m2_pre_proc"] = original_dim;

  if (use_torch) {
    // Test: torch backend, Basic inference
    backend = "torch";

    auto backup_path_map = std::move(model_path_map);
    auto backup_pre_map = std::move(pre_processor_map);
    auto backup_infer_map = std::move(inference_map);
    auto backup_in_tensor_dimensions = std::move(in_tensor_dimensions);
    auto backup_device_map = std::move(device_map);

    model_path_map = {{"test_model", model_folder + "identity_model.pt"}};
    pre_processor_map = {{"test_model", {"input"}}};
    inference_map = {{"test_model", {"output"}}};
    in_tensor_dimensions = {{"input", {3, 10, 10}}};
    device_map = {};

    YAML::Node torch_inference;
    torch_inference["inference"]["input_nodes"]["input"]["dtype"] = "kFloat32";
    torch_inference["inference"]["input_nodes"]["input"]["dim"] = "3 10 10";
    torch_inference["inference"]["output_nodes"]["output"]["dtype"] = "kFloat32";

    std::ofstream torch_config_file(model_folder + "identity_model.yaml");
    torch_config_file << torch_inference;
    torch_config_file.close();

    status = prepare_for_inference();
    status = do_inference();
    holoinfer_assert(status,
                     test_module,
                     34,
                     test_identifier_infer.at(34),
                     HoloInfer::holoinfer_code::H_SUCCESS);

    // Restore all changes to previous state
    std::filesystem::remove(model_folder + "identity_model.yaml");
    model_path_map = std::move(backup_path_map);
    pre_processor_map = std::move(backup_pre_map);
    inference_map = std::move(backup_infer_map);
    in_tensor_dimensions = std::move(backup_in_tensor_dimensions);
    device_map = std::move(backup_device_map);
  }

  // cleaning engine files
  for (const auto& file : std::filesystem::directory_iterator(model_folder)) {
    if (file.is_regular_file()) {
      const auto filename = file.path().filename().string();
      if (filename.find(".engine.") != std::string::npos) {
        std::filesystem::remove(file.path());
        HOLOSCAN_LOG_INFO("Cleaning up engine file: {}", filename);
      }
    } else if (file.is_directory()) {
      const auto directory = file.path().string();
      if (directory.find("_onnx_cache_") != std::string::npos) {
        std::filesystem::remove_all(file.path());
        HOLOSCAN_LOG_INFO("Cleaning up onnx cache directory: {}", directory);
      }
    }
  }
}
