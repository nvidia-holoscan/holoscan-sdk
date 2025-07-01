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
#ifndef _HOLOSCAN_INFER_CORE_H
#define _HOLOSCAN_INFER_CORE_H

#include <iostream>
#include <memory>
#include <vector>

#include <holoinfer_buffer.hpp>

namespace holoscan {
namespace inference {
/**
 * @brief Base Inference Class
 */
class InferBase {
 public:
  /**
   * @brief Default destructor
   * */
  virtual ~InferBase() = default;

  /**
   * @brief Does the Core inference
   * The provided CUDA data event is used to prepare the input data any execution of CUDA work
   * should be in sync with this event. If the inference is using CUDA it should record a CUDA
   * event and pass it back in `cuda_event_inference`.
   *
   * @param input_data Input DataBuffer
   * @param output_buffer Output DataBuffer, is populated with inferred results
   * @param cuda_event_data CUDA event recorded after data transfer
   * @param cuda_event_inference CUDA event recorded after inference
   * @return InferStatus
   * */
  virtual InferStatus do_inference(const std::vector<std::shared_ptr<DataBuffer>>& input_data,
                                   std::vector<std::shared_ptr<DataBuffer>>& output_buffer,
                                   cudaEvent_t cuda_event_data, cudaEvent_t* cuda_event_inference) {
    return InferStatus();
  }

  /**
   * @brief Get input data dimensions to the model
   * @return Vector of values as dimension
   * */
  virtual std::vector<std::vector<int64_t>> get_input_dims() const { return {}; }

  /**
   * @brief Get output data dimensions from the model
   * @return Vector of output dimensions. Each dimension is a vector of int64_t corresponding to
   *         the shape of the output tensor.
   * */
  virtual std::vector<std::vector<int64_t>> get_output_dims() const { return {}; }

  /**
   * @brief Get input data types from the model
   * @return Vector of input dimensions. Each dimension is a vector of int64_t corresponding to
   *         the shape of the input tensor.
   * */
  virtual std::vector<holoinfer_datatype> get_input_datatype() const { return {}; }

  /**
   * @brief Get output data types from the model
   * @return Vector of values as datatype per output tensor
   * */
  virtual std::vector<holoinfer_datatype> get_output_datatype() const { return {}; }

  virtual void cleanup() {}
};

}  // namespace inference
}  // namespace holoscan
#endif
