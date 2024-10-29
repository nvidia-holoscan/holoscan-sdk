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
#ifndef _HOLOSCAN_TORCH_INFER_CORE_H
#define _HOLOSCAN_TORCH_INFER_CORE_H

#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <holoinfer_constants.hpp>
#include <holoinfer_utils.hpp>
#include <infer/infer.hpp>

namespace holoscan {
namespace inference {

class TorchInferImpl;

/**
 * Libtorch based inference class
 * */
class TorchInfer : public InferBase {
 public:
  /**
   * @brief Constructor
   * @param model_file_path Path to torch model file
   * @param cuda_flag Flag to show if inference will happen using CUDA
   * */
  TorchInfer(const std::string& model_file_path, bool cuda_flag, bool cuda_buf_in,
             bool cuda_buf_out);

  /**
   * @brief Destructor
   * */
  ~TorchInfer();

  /**
   * @brief Does the Core inference.
   * The provided CUDA data event is used to prepare the input data any execution of CUDA work
   * should be in sync with this event. If the inference is using CUDA it should record a CUDA
   * event and pass it back in `cuda_event_inference`.
   *
   * @param input_data Vector of Input DataBuffer
   * @param output_buffer Vector of Output DataBuffer, is populated with inferred results
   * @return InferStatus
   * */
  InferStatus do_inference(const std::vector<std::shared_ptr<DataBuffer>>& input_data,
                           std::vector<std::shared_ptr<DataBuffer>>& output_buffer,
                           cudaEvent_t cuda_event_data, cudaEvent_t *cuda_event_inference);

  /**
   * @brief Populate class parameters with model details and values
   * */
  InferStatus populate_model_details();

  /**
   * @brief Print model details
   * */
  void print_model_details();

  /**
   * @brief Get input data dimensions to the model
   * @return Vector of input dimensions. Each dimension is a vector of int64_t corresponding to
   *         the shape of the input tensor.
   * */
  std::vector<std::vector<int64_t>> get_input_dims() const;

  /**
   * @brief Get output data dimensions from the model
   * @return Vector of output dimensions. Each dimension is a vector of int64_t corresponding to
   *         the shape of the output tensor.
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

 private:
  TorchInferImpl* impl_ = nullptr;
};

}  // namespace inference
}  // namespace holoscan

#endif
