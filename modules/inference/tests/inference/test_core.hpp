/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "test_infer_settings.hpp"

// Helper macro to assert HoloInfer status code. (TRT-only build: no Torch context-setup
// nuance to special-case here.)
#define HOLOINFER_EXPECT_STATUS(status, expected_code) \
  EXPECT_EQ((status).get_code(), (expected_code)) << "Status message: " << (status).get_message()

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

  std::string model_folder = "modules/inference/tests/test_models/";
  std::map<std::string, std::vector<std::string>> batch_sizes = {{"model_1", {"1, 1, 1"}}};
  bool dynamic_inputs = false;

  std::map<std::string, std::string> model_path_map = {
      {"model_1", model_folder + "identity_model.onnx"},
      {"model_2", model_folder + "identity_model.onnx"},
  };

  std::map<std::string, std::string> dla_core_map = {{"model_1", "-1"}, {"model_2", "-1"}};

  std::map<std::string, std::string> backend_map;

  std::map<std::string, std::vector<std::string>> input_map = {
      {"model_1", {"m1_pre_proc"}},
      {"model_2", {"m2_pre_proc"}},
  };

  std::map<std::string, std::vector<std::string>> output_map = {
      {"model_1", {"m1_infer"}},
      {"model_2", {"m2_infer"}},
  };

  std::map<std::string, std::vector<int>> in_tensor_dimensions = {
      {"m1_pre_proc", {3, 256, 256}},
      {"m2_pre_proc", {3, 256, 256}},
  };

  bool parallel_inference = true;
  bool enable_fp16 = false;
  bool input_on_cuda = true;
  bool output_on_cuda = true;
  bool is_engine_path = false;
  bool use_cuda_graphs = true;
  int32_t dla_core = -1;
  bool dla_gpu_fallback = true;

  /// When true, setup_specifications() sets InferenceSpecs::fast_multi_model_ and wires
  /// InferenceSpecs::allocate_cuda_stream_ to a simple test stream-pool stand-in (see
  /// stream_pool_limit below). Used by the fast-path dispatch tests in
  /// test_fast_path_dispatch.cpp.
  bool fast_multi_model = false;

  /// When true, setup_specifications() wires the test stream-pool stand-in even though
  /// `fast_multi_model` is false — i.e. the per-backend TrtInfer contexts themselves lease a
  /// stream (see infer_manager_setup.cpp's allocate_cuda_stream_for_ctx). Used by the
  /// blocking-stream fallback test (case G) in test_fallback_chain.cpp, since the stand-in hands
  /// out plain (blocking) cudaStreamCreate() streams.
  bool provide_backend_stream_pool = false;

  /// Maximum number of distinct CUDA streams the test stream-pool stand-in will hand out
  /// before returning nullptr (simulating an exhausted / undersized CudaStreamPool). -1 means
  /// unlimited.
  int32_t stream_pool_limit = -1;

  /// Number of streams handed out so far by the test stream-pool stand-in in the current
  /// setup_specifications() call. Reset to zero every call.
  int32_t stream_pool_calls_ = 0;

  /// Streams created by the test stream-pool stand-in; destroyed in clear_specs().
  std::vector<cudaStream_t> allocated_test_streams_;

  /// Pointer to inference context.
  std::unique_ptr<HoloInfer::InferContext> holoscan_infer_context_;

  /// Pointer to inference specifications
  std::shared_ptr<HoloInfer::InferenceSpecs> inference_specs_;
};
