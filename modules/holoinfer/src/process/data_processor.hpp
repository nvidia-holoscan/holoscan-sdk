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
#ifndef _HOLOSCAN_DATA_PROCESSOR_H
#define _HOLOSCAN_DATA_PROCESSOR_H

#include <bits/stdc++.h>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <holoinfer.hpp>

namespace holoscan {
namespace inference {
/// Declaration of function callback used by DataProcessor
using processor_FP =
    std::function<void(const std::vector<int>&, const std::vector<float>&, std::vector<int64_t>&,
                       std::vector<float>&, const std::vector<std::string>&)>;

/**
 * @brief Data Processor class that processes one operations per tensor. Currently supports CPU
 * based single operation.
 */
class DataProcessor {
 public:
  /**
   * @brief Default Constructor
   */
  DataProcessor() {}

  /**
   * @brief Checks the validity of supported operations
   *
   * @param process_operations Map where tensor name is the key, and operations to perform on
   * the tensor as vector of strings. Each value in the vector of strings is the supported
   * operation.
   *
   * @returns InferStatus with appropriate code and message
   */
  InferStatus initialize(const MultiMappings& process_operations);

  /**
   * @brief Executes an operation via function callback. (Currently CPU based)
   *
   * @param operation Operation to perform. Refer to user docs for a list of supported operations
   * @param in_dims Dimension of the input tensor
   * @param in_data Input data buffer
   * @param out_dims Dimension of the output tensor
   * @param out_data Output data buffer
   * @param custom_strings Strings to display for custom print operations
   * @returns InferStatus with appropriate code and message
   */
  InferStatus process_operation(const std::string& operation, const std::vector<int>& in_dims,
                                const std::vector<float>& in_data, std::vector<int64_t>& out_dims,
                                std::vector<float>& out_data,
                                const std::vector<std::string>& custom_strings);

  /**
   * @brief Computes max per channel in input data and scales it to [0, 1]. (CPU based)
   *
   * @param operation Operation to perform. Refer to user docs for a list of supported operations
   * @param in_dims Dimension of the input tensor
   * @param in_data Input data buffer
   * @param out_dims Dimension of the output tensor
   * @param out_data Output data buffer
   */
  void compute_max_per_channel_cpu(const std::vector<int>& in_dims,
                                   const std::vector<float>& in_data,
                                   std::vector<int64_t>& out_dims, std::vector<float>& out_data);

  /**
   * @brief Print data in the input buffer. Ideally to be used by classification models.
   *
   * @param in_dims Dimension of the input tensor
   * @param in_data Input data buffer
   */
  void print_results(const std::vector<int>& in_dims, const std::vector<float>& in_data);

  /**
   * @brief Print custom text for binary classification results in the input buffer.
   *
   * @param in_dims Dimension of the input tensor
   * @param in_data Input data buffer
   * @param custom_strings Strings to display for custom print operations
   */
  void print_custom_binary_classification(const std::vector<int>& in_dims,
                                          const std::vector<float>& in_data,
                                          const std::vector<std::string>& custom_strings);

 private:
  /// Map defining supported operations by DataProcessor Class.
  /// Keyword in this map must be used exactly by the user in configuration.
  /// Operation is the key and its related implementation platform as the value.
  const std::map<std::string, holoinfer_data_processor> supported_operations_{
      {"max_per_channel_scaled", holoinfer_data_processor::HOST},
      {"print", holoinfer_data_processor::HOST},
      {"print_custom_binary_classification", holoinfer_data_processor::HOST}};

  /// Mapped function call for the function pointer of max_per_channel_scaled
  processor_FP max_per_channel_scaled_fp_ = [this](auto& in_dims, auto& in_data, auto& out_dims,
                                                   auto& out_data, auto& custom_strings) {
    compute_max_per_channel_cpu(in_dims, in_data, out_dims, out_data);
  };

  /// Mapped function call for the function pointer of print
  processor_FP print_results_fp_ = [this](auto& in_dims, auto& in_data, auto& out_dims,
                                          auto& out_data, auto& custom_strings) {
    print_results(in_dims, in_data);
  };

  /// Mapped function call for the function pointer of printing custom binary classification results
  processor_FP print_custom_binary_classification_fp_ =
      [this](auto& in_dims, auto& in_data, auto& out_dims, auto& out_data, auto& custom_strings) {
        print_custom_binary_classification(in_dims, in_data, custom_strings);
      };

  /// Map with supported operation as the key and related function pointer as value
  const std::map<std::string, processor_FP> oper_to_fp_{
      {"max_per_channel_scaled", max_per_channel_scaled_fp_},
      {"print", print_results_fp_},
      {"print_custom_binary_classification", print_custom_binary_classification_fp_}};
};

}  // namespace inference
}  // namespace holoscan

#endif
