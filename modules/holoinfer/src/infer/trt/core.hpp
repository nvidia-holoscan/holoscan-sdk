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
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>

#include <holoinfer_constants.hpp>
#include <infer/infer.hpp>
#include "utils.hpp"

namespace holoscan {
namespace inference {
/**
 * Class to execute TensorRT based inference
 * */
class TrtInfer : public InferBase {
 public:
  /**
   * @brief Constructor
   */
  TrtInfer(const std::string& model_path, const std::string& model_name, bool enable_fp16,
           bool is_engine_path, bool cuda_buf_in, bool cuda_buf_out);

  /**
   * @brief Destructor
   */
  ~TrtInfer();

  /**
   * @brief Does the Core inference with TRT backend
   * @param input_data Input DataBuffer
   * @param output_buffer Output DataBuffer, is populated with inferred results
   * @return InferStatus
   * */
  InferStatus do_inference(std::shared_ptr<DataBuffer>& input_data,
                           std::shared_ptr<DataBuffer>& output_buffer);

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

  void cleanup() {}

 private:
  /// @brief Path to onnx model file
  std::string model_path_;

  /// @brief A Unique identifier for the model
  std::string model_name_;

  /// @brief Dimensions of input buffer
  std::vector<int64_t> input_dims_;

  /// @brief Dimensions of output buffer
  std::vector<int64_t> output_dims_;

  /// @brief Data bindings for inference
  std::shared_ptr<std::vector<void*>> inference_bindings;

  /// @brief Use FP16 in TRT engine file generation
  bool enable_fp16_;

  /// @brief Flag showing if input buffer is on cuda
  bool cuda_buf_in_;

  /// @brief Flag showing if output buffer will be on cuda
  bool cuda_buf_out_;

  /// @brief Flag showing if input path map is path to engine file
  bool is_engine_path_;

  /**
   * @brief Parameter initialization
   */
  bool initialize_parameters();

  /**
   * @brief Load and prepare the (trt converted) network for inference
   */
  bool load_engine();

  /// @brief Pointer to TRT cuda engine
  std::unique_ptr<nvinfer1::ICudaEngine> engine_ = nullptr;

  /// @brief Pointer to TRT execution context
  std::unique_ptr<nvinfer1::IExecutionContext> context_ = nullptr;

  /// @brief Options to generate TRT engine file from onnx
  NetworkOptions network_options_;

  /// @brief Logger object
  Logger logger_;

  /// @brief Pointer to input DataBuffer. Holds both host and device buffers as specified
  std::shared_ptr<DataBuffer> input_buffer_;

  /// @brief Pointer to output DataBuffer. Holds both host and device buffers as specified
  std::shared_ptr<DataBuffer> output_buffer_;

  /// @brief Generated engine file path. The extension is unique per GPU model
  std::string engine_path_;

  /// @brief Cuda stream
  cudaStream_t cuda_stream_ = nullptr;
};

}  // namespace inference
}  // namespace holoscan
