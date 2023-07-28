/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef HOLOINFER_INFERENCE_TESTS_HPP
#define HOLOINFER_INFERENCE_TESTS_HPP

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "test_infer_settings.hpp"

class HoloInferTests {
 public:
  HoloInferTests() {}
  void holoinfer_assert(const HoloInfer::InferStatus& status, const std::string& module,
                        unsigned int current_test, const std::string& test_name,
                        HoloInfer::holoinfer_code assert_type);

  void clear_specs();
  HoloInfer::InferStatus create_specifications();
  void setup_specifications();
  HoloInfer::InferStatus setup_inference();
  HoloInfer::InferStatus call_parameter_check_inference();
  void parameter_test_inference();

  void parameter_setup_test();
  HoloInfer::InferStatus prepare_for_inference();
  HoloInfer::InferStatus do_inference();
  void inference_tests();
  void print_summary();

 private:
  /// Default parameters for inference
  unsigned int pass_test_count = 0, fail_test_count = 0, total_test_count = 0;

  std::string backend = "trt";
  std::vector<std::string> in_tensor_names = {
      "plax_cham_pre_proc", "aortic_pre_proc", "bmode_pre_proc"};
  std::vector<std::string> out_tensor_names = {"plax_cham_infer", "aortic_infer", "bmode_infer"};

  std::map<std::string, std::string> model_path_map = {
      {"plax_chamber", "../data/multiai_ultrasound/models/plax_chamber.onnx"},
      {"aortic_stenosis", "../data/multiai_ultrasound/models/aortic_stenosis.onnx"},
      {"bmode_perspective", "../data/multiai_ultrasound/models/bmode_perspective.onnx"}};

  std::map<std::string, std::string> device_map = {
      {"plax_chamber", "0"}, {"aortic_stenosis", "0"}, {"bmode_perspective", "0"}};

  std::map<std::string, std::string> backend_map;

  std::map<std::string, std::vector<std::string>> pre_processor_map = {
      {"plax_chamber", {"plax_cham_pre_proc"}},
      {"aortic_stenosis", {"aortic_pre_proc"}},
      {"bmode_perspective", {"bmode_pre_proc"}}};

  std::map<std::string, std::vector<std::string>> inference_map = {
      {"plax_chamber", {"plax_cham_infer"}},
      {"aortic_stenosis", {"aortic_infer"}},
      {"bmode_perspective", {"bmode_infer"}}};

  bool parallel_inference = true;
  bool infer_on_cpu = false;
  bool enable_fp16 = false;
  bool input_on_cuda = true;
  bool output_on_cuda = true;
  bool is_engine_path = false;

  const std::map<std::string, std::vector<int>> in_tensor_dimensions = {
      {"plax_cham_pre_proc", {320, 320, 3}},
      {"aortic_pre_proc", {300, 300, 3}},
      {"bmode_pre_proc", {320, 240, 3}},
  };

  /// Pointer to inference context.
  std::unique_ptr<HoloInfer::InferContext> holoscan_infer_context_;

  /// Pointer to inference specifications
  std::shared_ptr<HoloInfer::InferenceSpecs> inference_specs_;

  const std::map<unsigned int, std::string> test_identifier_params = {
      {1, "Parameters, model_path_map: dummy path test"},
      {2, "Parameters, model_path_map: key mismatch with pre_processor_map"},
      {3, "Parameters, pre_processor_map mismatch check with model_path_map"},
      {4, "Parameters, pre_processor_map empty value vector check"},
      {5, "Parameters, pre_processor_map empty tensor name check"},
      {6, "Parameters, pre_processor_map duplicate tensor name check"},
      {7, "Parameters, input_tensor exist in pre_processor_map"},
      {8, "Parameters, input_tensor is unique"},
      {9, "Parameters, inference_map mismatch check"},
      {10, "Parameters, inference_map duplicate entry check"},
      {11, "Parameters, output_tensor exist in inference_map"},
      {12, "Parameters, output_tensor is unique"},
      {13, "Parameters, Input parameter set check"},
      {14, "TRT backend, Empty model path check"},
      {15, "TRT backend, Inference map key mismatch with model path map"},
      {16, "TRT backend, Inference map, missing entry"},
      {17, "TRT backend, CPU based inference"},
      {18, "TRT backend, Backend type"},
      {19, "Torch backend, incorrect model file format"},
      {20, "ONNX backend, Input/Output cuda buffer test"},
      {21, "ONNX backend, incorrect model file format"},
      {22, "ONNX backend, Engine path true test"},
      {23, "ONNX backend, Default"},
      {24, "TRT backend, Default check 1"}};

  const std::map<unsigned int, std::string> test_identifier_infer = {
      {1, "TRT backend, Empty input data"},
      {2, "TRT backend, Empty inference parameters"},
      {3, "TRT backend, Missing input tensor"},
      {4, "TRT backend, Missing output tensor"},
      {5, "TRT backend, Empty input cuda buffer 1"},
      {6, "TRT backend, Empty input cuda buffer 2"},
      {7, "TRT backend, Empty output cuda buffer 1"},
      {8, "TRT backend, Empty output cuda buffer 2"},
      {9, "TRT backend, Empty output cuda buffer 3"},
      {10, "TRT backend, Basic end-to-end cuda inference"},
      {11, "TRT backend, Basic sequential end-to-end cuda inference"},
      {12, "TRT backend, Input on host inference"},
      {13, "TRT backend, Output on host inference"},
      {14, "TRT backend, Input/Output on host inference"},
      {15, "TRT backend, Empty host input"},
      {16, "TRT backend, Empty host output"},
      {17, "ONNX backend, Basic parallel inference on CPU"},
      {18, "ONNX backend, Basic sequential inference on CPU"},
      {19, "ONNX backend, Basic sequential inference on GPU"},
      {20, "ONNX backend, Basic parallel inference on GPU"},
      {21, "ONNX backend, Empty host input"},
      {22, "ONNX backend, Empty host output"},
      {23, "ONNX backend on ARM, Basic sequential inference on GPU"},
      {24, "ONNX backend, Basic sequential inference on multi-GPU"},
      {25, "ONNX backend, Inference single GPU with multi-GPU settings"},
      {26, "ONNX backend, Basic parallel inference on multi-GPU"},
      {27, "TRT backend, Basic sequential inference on multi-GPU"},
      {28, "TRT backend, Basic parallel inference on multi-GPU"},
      {29, "TRT backend, Parallel inference on multi-GPU with I/O on host"},
      {30, "TRT backend, Parallel inference on multi-GPU with Input on host"},
      {31, "TRT backend, Parallel inference on multi-GPU with Output on host"}};
};

#endif /* HOLOINFER_INFERENCE_TESTS_HPP */
