/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef HOLOINFER_PROCESSING_TEST_CORE_HPP
#define HOLOINFER_PROCESSING_TEST_CORE_HPP

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <holoinfer.hpp>
#include <holoinfer_utils.hpp>

namespace HoloInfer = holoscan::inference;

class HoloInferProcessingTests : public ::testing::Test {
 protected:
  HoloInfer::InferStatus call_parameter_check_processing();
  HoloInfer::InferStatus setup_processor(bool use_cuda_graphs = false);
  HoloInfer::InferStatus execute_processor();
  void clear_processor();

  /// Default parameters for processing
  std::map<std::string, std::vector<std::string>> process_operations = {
      {"plax_cham_infer", {"max_per_channel_scaled"}}};

  std::map<std::string, std::vector<std::string>> processed_map = {
      {"plax_cham_infer", {"plax_chamber_processed"}}};
  std::vector<std::string> out_tensor_processing = {"plax_chamber_processed"};
  std::vector<std::string> in_tensor_processing = {
      "plax_cham_infer", "aortic_infer", "bmode_infer"};

  std::unique_ptr<HoloInfer::ProcessorContext> holoscan_processor_context_;
  HoloInfer::DataMap data_per_tensor;
  std::map<std::string, std::vector<int>> dims_per_tensor;
  bool process_with_cuda = false;
  cudaStream_t cuda_stream = 0;
  std::string config_path = "";
  std::map<std::string, std::string> custom_kernels;
};

#endif /* HOLOINFER_PROCESSING_TEST_CORE_HPP */
