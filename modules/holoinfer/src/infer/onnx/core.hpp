/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef _HOLOSCAN_ONNX_INFER_CORE_H
#define _HOLOSCAN_ONNX_INFER_CORE_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <holoinfer_constants.hpp>
#include <infer/infer.hpp>

namespace holoscan {
namespace inference {
/**
 * Onnxruntime based inference class
 * */
class OnnxInfer : public InferBase {
 public:
  /**
   * @brief Constructor
   * @param model_file_path Path to onnx model file
   * @param cuda_flag Flag to show if inference will happen using CUDA
   * */
  OnnxInfer(const std::string& model_file_path, bool cuda_flag);

  /**
   * @brief Does the Core inference using Onnxruntime. Input and output buffer are supported on
   * Host. Inference is supported on host and device.
   * @param input_data Input DataBuffer
   * @param output_buffer Output DataBuffer, is populated with inferred results
   * @return InferStatus
   * */
  InferStatus do_inference(std::shared_ptr<DataBuffer>& input_data,
                           std::shared_ptr<DataBuffer>& output_buffer);

  /**
   * @brief Populate class parameters with model details and values
   * */
  void populate_model_details();

  /**
   * @brief Print model details
   * */
  void print_model_details();

  /**
   * @brief Create session options for inference
   * */
  int set_holoscan_inf_onnx_session_options();

  /**
   * @brief Get input data dimensions to the model
   * @return Vector of values as dimension
   * */
  std::vector<int64_t> get_input_dims() const;

  /**
   * @brief Get output data dimensions from the model
   * @return Vector of values as dimension
   * */
  std::vector<int64_t> get_output_dims() const;

  void cleanup() {
    session_.reset();
    env_.reset();
  }

 private:
  std::string model_path_{""};
  bool use_cuda_ = true;

  Ort::SessionOptions session_options_;
  OrtCUDAProviderOptions cuda_options_{};

  std::unique_ptr<Ort::Env> env_ = nullptr;
  std::unique_ptr<Ort::Session> session_ = nullptr;

  Ort::AllocatorWithDefaultOptions allocator_;

  size_t input_nodes_{0}, output_nodes_{0};

  std::vector<int64_t> input_dims_{};
  std::vector<int64_t> output_dims_{};

  ONNXTensorElementDataType input_type_, output_type_;

  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;

  std::vector<Ort::Value> input_tensors_;
  std::vector<Ort::Value> output_tensors_;

  std::vector<Ort::Value> input_tensors_gpu_;
  std::vector<Ort::Value> output_tensors_gpu_;

  Ort::MemoryInfo memory_info_ = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                            OrtMemType::OrtMemTypeDefault);

  std::vector<float> input_tensor_values_;
  std::vector<float> output_tensor_values_;

  Ort::MemoryInfo memory_info_cuda_ =
      Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
  std::unique_ptr<Ort::Allocator> memory_allocator_cuda_;
};

}  // namespace inference
}  // namespace holoscan

#endif
