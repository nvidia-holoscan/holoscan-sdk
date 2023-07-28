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
#ifndef _HOLOSCAN_INFER_API_H
#define _HOLOSCAN_INFER_API_H

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "holoinfer_buffer.hpp"

namespace holoscan {
namespace inference {

/**
 * Inference Context class
 */
class _HOLOSCAN_EXTERNAL_API_ InferContext {
 public:
  InferContext();
  ~InferContext();
  /**
   * Set Inference parameters
   *
   * @param inference_specs   Pointer to inference specifications
   *
   * @returns InferStatus with appropriate holoinfer_code and message.
   */
  InferStatus set_inference_params(std::shared_ptr<InferenceSpecs>& inference_specs);

  /**
   * Executes the inference
   * Toolkit supports one input per model, in float32 type
   *
   * @param preprocess_data_map   Map of model names as key mapped to the preprocessed input data
   * @param output_data_map       Map of tensor names as key mapped to the inferred data
   *
   * @returns InferStatus with appropriate holoinfer_code and message.
   */
  InferStatus execute_inference(DataMap& preprocess_data_map, DataMap& output_data_map);

  /**
   * Gets output dimension per model
   *
   * @returns Map of model as key mapped to the output dimension (of inferred data)
   */
  DimType get_output_dimensions() const;
};

/**
 * Processor Context class
 */
class _HOLOSCAN_EXTERNAL_API_ ProcessorContext {
 public:
  ProcessorContext();
  /**
   * Initialize the preprocessor context
   *
   * @param process_operations   Map of tensor name as key, mapped to list of operations to be
   *                             applied in sequence on the tensor
   *
   * @returns InferStatus with appropriate holoinfer_code and message.
   */
  InferStatus initialize(const MultiMappings& process_operations, const std::string config_path);

  /**
   * Process the tensors with operations as initialized.
   * Toolkit supports one tensor input and output per model
   *
   * @param tensor_oper_map Map of tensor name as key, mapped to list of operations to be applied in
   * sequence on the tensor
   * @param in_out_tensor_map Map of input tensor name mapped to vector of output tensor names
   * after processing
   * @param processed_result_map Map is updated with output tensor name as key mapped to processed
   * output as a vector of float32 type
   * @param dimension_map Map is updated with model name as key mapped to dimension of processed
   * data as a vector
   *
   * @returns InferStatus with appropriate holoinfer_code and message.
   */
  InferStatus process(const MultiMappings& tensor_oper_map, const MultiMappings& in_out_tensor_map,
                      DataMap& processed_result_map,
                      const std::map<std::string, std::vector<int>>& dimension_map);

  /**
   * Get output data per Tensor
   * Toolkit supports one output per Tensor, in float32 type
   *
   * @returns Map of tensor name as key mapped to the output float32 type data as a vector
   */
  DataMap get_processed_data() const;

  /**
   * Get output dimension per model
   * Toolkit supports one output per model
   *
   * @returns Map of model as key mapped to the output dimension (of processed data) as a vector
   */
  DimType get_processed_data_dims() const;
};

}  // namespace inference
}  // namespace holoscan

#endif
