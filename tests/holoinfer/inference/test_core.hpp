/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../../utils/holoinfer_backend_test_utils.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "test_infer_settings.hpp"

// Helper macro to assert HoloInfer status code with optional Torch context check
#if defined(HOLOINFER_TORCH_ENABLED)
#define HOLOINFER_EXPECT_STATUS(status, expected_code)                                   \
  do {                                                                                   \
    const auto& _s = (status);                                                           \
    EXPECT_EQ(_s.get_code(), (expected_code)) << "Status message: " << _s.get_message(); \
    EXPECT_EQ(_s.get_message().find("context setup failure"), std::string::npos)         \
        << "Unexpected context setup failure in status message";                         \
  } while (0)
#else
#define HOLOINFER_EXPECT_STATUS(status, expected_code) \
  EXPECT_EQ((status).get_code(), (expected_code)) << "Status message: " << (status).get_message()
#endif

class HoloInferTests : public ::testing::Test {
 protected:
  void clear_specs();
  HoloInfer::InferStatus create_specifications();
  void setup_specifications();
  HoloInfer::InferStatus setup_inference();
  HoloInfer::InferStatus call_parameter_check_inference();
  HoloInfer::InferStatus prepare_for_inference();
  HoloInfer::InferStatus do_inference();
  void cleanup_engines();

  /// Default parameters for inference
  std::string backend = "trt";

  std::vector<std::string> in_tensor_names = {"m1_pre_proc", "m2_pre_proc"};
  std::vector<std::string> out_tensor_names = {"m1_infer", "m2_infer"};

  std::string model_folder = "tests/holoinfer/test_models/";
  std::map<std::string, std::vector<std::string>> batch_sizes = {{"model_1", {"1, 1, 1"}}};
  bool dynamic_inputs = false;

  std::map<std::string, std::string> model_path_map = {
      {"model_1", model_folder + "identity_model.onnx"},
      {"model_2", model_folder + "identity_model.onnx"},
  };

  std::map<std::string, std::string> device_map = {{"model_1", "0"}, {"model_2", "0"}};
  std::map<std::string, std::string> dla_core_map = {{"model_1", "-1"}, {"model_2", "-1"}};

  std::map<std::string, std::string> temporal_map = {{"model_1", "1"}, {"model_2", "1"}};
  std::map<std::string, std::string> activation_map = {{"model_1", "1"}, {"model_2", "1"}};

  std::map<std::string, std::string> backend_map;

  std::map<std::string, std::vector<std::string>> pre_processor_map = {
      {"model_1", {"m1_pre_proc"}},
      {"model_2", {"m2_pre_proc"}},
  };

  std::map<std::string, std::vector<std::string>> inference_map = {
      {"model_1", {"m1_infer"}},
      {"model_2", {"m2_infer"}},
  };

  std::map<std::string, std::vector<int>> in_tensor_dimensions = {
      {"m1_pre_proc", {3, 256, 256}},
      {"m2_pre_proc", {3, 256, 256}},
  };

  bool parallel_inference = true;
  bool infer_on_cpu = false;
  bool enable_fp16 = false;
  bool input_on_cuda = true;
  bool output_on_cuda = true;
  bool is_engine_path = false;
  bool use_cuda_graphs = true;
  int32_t dla_core = -1;
  bool dla_gpu_fallback = true;

  /// Pointer to inference context.
  std::unique_ptr<HoloInfer::InferContext> holoscan_infer_context_;

  /// Pointer to inference specifications
  std::shared_ptr<HoloInfer::InferenceSpecs> inference_specs_;
};

#if defined(HOLOINFER_ORT_ENABLED)
class HoloInferOnnxRuntimeTests : public HoloInferTests {
 protected:
  void SetUp() override { HOLOSCAN_TEST_SKIP_IF_ONNX_RUNTIME_BACKEND_DISABLED(); }
};
#endif

#endif /* HOLOINFER_INFERENCE_TESTS_HPP */
