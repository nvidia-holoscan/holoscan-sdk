/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef MODULES_HOLOINFER_SRC_INFER_TORCH_CORE_HPP
#define MODULES_HOLOINFER_SRC_INFER_TORCH_CORE_HPP

#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <map>
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
   * @param cuda_buf_in Flag to demonstrate if input data buffer is on cuda
   * @param cuda_buf_out Flag to demonstrate if output data buffer will be on cuda
   * */
  TorchInfer(const std::string& model_file_path, bool cuda_flag, bool cuda_buf_in,
             bool cuda_buf_out, int device_id,
             std::function<cudaStream_t(int32_t device_id)> allocate_cuda_stream);

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
   * @param cuda_event_data CUDA event to synchronize input data preparation
   * @param cuda_event_inference Pointer to CUDA event for inference synchronization
   * @return InferStatus
   * */
  InferStatus do_inference(const std::vector<std::shared_ptr<DataBuffer>>& input_data,
                           std::vector<std::shared_ptr<DataBuffer>>& output_buffer,
                           cudaEvent_t cuda_event_data, cudaEvent_t* cuda_event_inference);

  /**
   * @brief Updates the dimensions per tensor in case of dynamic inputs.
   * Using the input Holoscan tensors and their dimension mapping, the internal input size vector is
   * updated
   *
   * @param input_tensors Vector of input Holoscan tensor names
   * @param dims_per_tensor Map storing the dimensions as values and Holoscan tensor names as keys.
   * @return true if the dynamic input dimensions were successfully updated, false otherwise
   */
  bool set_dynamic_input_dimension(const std::vector<std::string>& input_tensors,
                                   const std::map<std::string, std::vector<int>>& dims_per_tensor);

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

#endif /* MODULES_HOLOINFER_SRC_INFER_TORCH_CORE_HPP */
