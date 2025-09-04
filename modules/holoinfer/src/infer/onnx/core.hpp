/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <infer/infer.hpp>

namespace holoscan {
namespace inference {

class OnnxInferImpl;

/**
 * Onnxruntime based inference class
 * */
class OnnxInfer : public InferBase {
 public:
  /**
   * @brief Constructor
   * @param model_file_path Path to onnx model file
   * @param enable_fp16 Flag showing if trt engine file conversion will use FP16.
   * @param dla_core The DLA core index to execute the engine on, starts at 0. Set to -1 to disable
   * DLA.
   * @param dla_gpu_fallback If DLA is enabled, use the GPU if a layer cannot be executed on DLA. If
   * the fallback is disabled, engine creation will fail if a layer cannot executed on DLA.
   * @param cuda_flag Flag to show if inference will happen using CUDA
   * @param cuda_buf_in Flag to demonstrate if input data buffer is on cuda
   * @param cuda_buf_out Flag to demonstrate if output data buffer will be on cuda
   * @param allocate_cuda_stream Function to allocate a CUDA stream (optional)
   * */
  OnnxInfer(const std::string& model_file_path, bool enable_fp16, int32_t dla_core,
            bool dla_gpu_fallback, bool cuda_flag, bool cuda_buf_in, bool cuda_buf_out,
            std::function<cudaStream_t(int32_t device_id)> allocate_cuda_stream);

  /**
   * @brief Destructor
   * */
  ~OnnxInfer();

  /**
   * @brief Does the Core inference using Onnxruntime. Input and output buffer are supported on
   * Host. Inference is supported on host and device.
   * The provided CUDA data event is used to prepare the input data any execution of CUDA work
   * should be in sync with this event. If the inference is using CUDA it should record a CUDA
   * event and pass it back in `cuda_event_inference`.
   *
   * @param input_data Input DataBuffer
   * @param output_buffer Output DataBuffer, is populated with inferred results
   * @param cuda_event_data CUDA event to synchronize input data preparation
   * @param cuda_event_inference Pointer to CUDA event for inference synchronization
   * @return InferStatus
   * */
  InferStatus do_inference(const std::vector<std::shared_ptr<DataBuffer>>& input_data,
                           std::vector<std::shared_ptr<DataBuffer>>& output_buffer,
                           cudaEvent_t cuda_event_data, cudaEvent_t* cuda_event_inference);

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
   * @return Vector of input dimensions. Each dimension is a vector of int64_t corresponding to
   *         the shape of the input tensor.
   * */
  std::vector<std::vector<int64_t>> get_input_dims() const;

  /**
   * @brief Get output data dimensions from the model
   * @return Vector of input dimensions. Each dimension is a vector of int64_t corresponding to
   *         the shape of the input tensor.
   * */
  std::vector<std::vector<int64_t>> get_output_dims() const;

  /**
   * @brief Get input data types from the model
   * @return Vector of values as datatype per input tensor
   * */
  std::vector<holoinfer_datatype> get_input_datatype() const;

  /**
   * @brief Get output data types from the model
   * @return Vector of values as datatype per output tensor
   * */
  std::vector<holoinfer_datatype> get_output_datatype() const;

  void cleanup();

 private:
  OnnxInferImpl* impl_ = nullptr;
};

}  // namespace inference
}  // namespace holoscan

#endif
