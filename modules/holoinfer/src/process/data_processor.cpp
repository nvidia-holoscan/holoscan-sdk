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
#include "data_processor.hpp"

namespace holoscan {
namespace inference {

InferStatus DataProcessor::initialize(const MultiMappings& process_operations) {
  for (const auto& p_op : process_operations) {
    auto _operations = p_op.second;
    if (_operations.size() == 0) {
      return InferStatus(holoinfer_code::H_ERROR,
                         "Data processor, Empty operation list for tensor " + p_op.first);
    }

    for (const auto& _op : _operations) {
      if (_op.find("print") == std::string::npos) {
        if (supported_operations_.find(_op) == supported_operations_.end()) {
          return InferStatus(holoinfer_code::H_ERROR,
                             "Data processor, Operation " + _op + " not supported.");
        }
      } else {
        if ((_op.compare("print") != 0) &&
            (_op.find("print_custom_binary_classification") == std::string::npos)) {
          return InferStatus(holoinfer_code::H_ERROR,
                             "Data processor, Print operation: " + _op + " not supported.");
        }
      }
    }
  }
  return InferStatus();
}

void DataProcessor::print_results(const std::vector<int>& dimensions,
                                  const std::vector<float>& indata) {
  size_t dsize = indata.size();

  for (unsigned int i = 0; i < dsize - 1; i++) { std::cout << indata.at(i) << ", "; }
  std::cout << indata[dsize - 1] << "\n";
}

void DataProcessor::print_custom_binary_classification(
    const std::vector<int>& dimensions, const std::vector<float>& indata,
    const std::vector<std::string>& custom_strings) {
  size_t dsize = indata.size();

  if (dsize == 2) {
    auto first_value = 1.0 / (1 + exp(-indata.at(0)));
    auto second_value = 1.0 / (1 + exp(-indata.at(1)));

    if (first_value > second_value) {
      std::cout << custom_strings[0] << ". Confidence: " << first_value << "\n";
    } else {
      std::cout << custom_strings[1] << ". Confidence: " << second_value << "\n";
    }
  } else {
    HOLOSCAN_LOG_INFO("Input data size: {}", dsize);
    HOLOSCAN_LOG_INFO("This is binary classification custom print, size must be 2.");
  }
}

void DataProcessor::compute_max_per_channel_cpu(const std::vector<int>& dimensions,
                                                const std::vector<float>& outvector,
                                                std::vector<int64_t>& processed_dims,
                                                std::vector<float>& processed_data) {
  // Assuming NHWC format, TODO:: Generalize:: SD
  const size_t rows = dimensions[1];
  const size_t cols = dimensions[2];
  const size_t out_channels = dimensions[3];

  const float* out_result = outvector.data();
  std::vector<unsigned int> max_x_per_channel, max_y_per_channel;

  max_x_per_channel.resize(out_channels, 0);
  max_y_per_channel.resize(out_channels, 0);
  std::vector<float> maxV(out_channels, -1999);

  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      for (unsigned int c = 1; c < out_channels; c++) {
        unsigned int index = i * cols * out_channels + j * out_channels + c;
        float v1 = out_result[index];
        if (maxV[c] < v1) {
          maxV[c] = v1;
          max_x_per_channel[c] = i;
          max_y_per_channel[c] = j;
        }
      }
    }
  }

  for (unsigned int i = 0; i < out_channels; i++) {
    processed_data[2 * i] = static_cast<float>(max_x_per_channel[i]) / static_cast<float>(rows);
    processed_data[2 * i + 1] = static_cast<float>(max_y_per_channel[i]) / static_cast<float>(cols);
  }

  processed_dims.push_back(1);  // CHECK: if disabled, get_data_from_tensor fails.
  processed_dims.push_back(static_cast<int64_t>(2 * out_channels));
  max_x_per_channel.clear();
  max_y_per_channel.clear();
}

InferStatus DataProcessor::process_operation(const std::string& operation,
                                             const std::vector<int>& indims,
                                             const std::vector<float>& indata,
                                             std::vector<int64_t>& processed_dims,
                                             std::vector<float>& processed_data,
                                             const std::vector<std::string>& custom_strings) {
  if (oper_to_fp_.find(operation) == oper_to_fp_.end())
    return InferStatus(holoinfer_code::H_ERROR,
                       "Data processor, Operation " + operation + " not found in map");
  try {
    oper_to_fp_.at(operation)(indims, indata, processed_dims, processed_data, custom_strings);
  } catch (...) {
    return InferStatus(holoinfer_code::H_ERROR,
                       "Data processor, Exception in running " + operation);
  }
  if (operation.find("print") == std::string::npos && processed_data.size() == 0) {
    return InferStatus(holoinfer_code::H_ERROR, "Data processor, Processed data map empty");
  }
  return InferStatus();
}

}  // namespace inference
}  // namespace holoscan
