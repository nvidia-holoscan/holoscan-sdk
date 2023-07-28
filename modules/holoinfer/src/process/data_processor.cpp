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

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace holoscan {
namespace inference {

InferStatus DataProcessor::initialize(const MultiMappings& process_operations,
                                      const std::string config_path = {}) {
  if (config_path.length() > 0) {
    if (std::filesystem::exists(config_path)) {
      config_path_ = config_path;
      HOLOSCAN_LOG_INFO("Postprocessing config path: {}", config_path);
    } else {
      return InferStatus(holoinfer_code::H_ERROR, "Data processor, config path does not exist.");
    }
  }

  for (const auto& p_op : process_operations) {
    auto _operations = p_op.second;
    if (_operations.size() == 0) {
      return InferStatus(holoinfer_code::H_ERROR,
                         "Data processor, Empty operation list for tensor " + p_op.first);
    }

    for (const auto& _op : _operations) {
      if (_op.find("print") == std::string::npos) {
        if (supported_transforms_.find(_op) != supported_transforms_.end()) {
          // In future releases, this will be generalized with addition of more transforms.
          if (_op.compare("generate_boxes") == 0) {
            // unique key is created as a combination of input tensors and operation
            auto key = fmt::format("{}-{}", p_op.first, _op);
            HOLOSCAN_LOG_INFO("Transform map updated with key: {}", key);
            transforms_.insert({key, std::make_unique<GenerateBoxes>(config_path_)});

            std::vector<std::string> tensor_tokens;
            string_split(p_op.first, tensor_tokens, ':');

            auto status = transforms_.at(key)->initialize(tensor_tokens);
            if (status.get_code() != holoinfer_code::H_SUCCESS) {
              status.display_message();
              return status;
            }
          }
        } else if (supported_compute_operations_.find(_op) == supported_compute_operations_.end()) {
          return InferStatus(holoinfer_code::H_ERROR,
                             "Data processor, Operation " + _op + " not supported.");
        }
      } else {
        if (supported_print_operations_.find(_op) == supported_print_operations_.end() &&
            _op.find("print_custom_binary_classification") == std::string::npos) {
          return InferStatus(holoinfer_code::H_ERROR,
                             "Data processor, Print operation: " + _op + " not supported.");
        }
      }
    }
  }

  return InferStatus();
}

InferStatus DataProcessor::process_transform(const std::string& transform, const std::string& key,
                                             const std::map<std::string, void*>& indata,
                                             const std::map<std::string, std::vector<int>>& indim,
                                             DataMap& processed_data, DimType& processed_dims) {
  if (transform_to_fp_.find(transform) == transform_to_fp_.end()) {
    HOLOSCAN_LOG_ERROR("Transform {} not supported/declared.", transform);
    return InferStatus(holoinfer_code::H_ERROR,
                       "Data processor, Processing not supported for " + transform);
  }
  auto transform_key = fmt::format("{}-{}", key, transform);

  auto status =
      transform_to_fp_.at(transform)(transform_key, indata, indim, processed_data, processed_dims);
  return status;
}

InferStatus DataProcessor::print_results(const std::vector<int>& dimensions, const void* indata) {
  size_t dsize = accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<size_t>());
  if (dsize < 1) {
    HOLOSCAN_LOG_ERROR("Input data size must be at least 1.");
    return InferStatus(holoinfer_code::H_ERROR, "Data processor, Incorrect input data size");
  }
  auto indata_float = static_cast<const float*>(indata);
  for (unsigned int i = 0; i < dsize - 1; i++) { std::cout << indata_float[i] << ", "; }
  std::cout << indata_float[dsize - 1] << "\n";
  return InferStatus();
}

InferStatus DataProcessor::print_results_int32(const std::vector<int>& dimensions,
                                               const void* indata) {
  size_t dsize = accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<size_t>());
  if (dsize < 1) {
    HOLOSCAN_LOG_ERROR("Input data size must be at least 1.");
    return InferStatus(holoinfer_code::H_ERROR, "Data processor, Incorrect input data size");
  }
  auto indata_int32 = static_cast<const int32_t*>(indata);

  for (unsigned int i = 0; i < dsize - 1; i++) { std::cout << indata_int32[i] << ", "; }
  std::cout << indata_int32[dsize - 1] << "\n";
  return InferStatus();
}

InferStatus DataProcessor::print_custom_binary_classification(
    const std::vector<int>& dimensions, const void* indata,
    const std::vector<std::string>& custom_strings) {
  size_t dsize = accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<size_t>());
  if (dsize < 1) {
    HOLOSCAN_LOG_ERROR("Input data size must be at least 1.");
    return InferStatus(holoinfer_code::H_ERROR, "Data processor, Incorrect input data size");
  }
  auto indata_float = static_cast<const float*>(indata);

  if (dsize == 2) {
    auto first_value = 1.0 / (1 + exp(-indata_float[0]));
    auto second_value = 1.0 / (1 + exp(-indata_float[1]));

    if (first_value > second_value) {
      std::cout << custom_strings[0] << ". Confidence: " << first_value << "\n";
    } else {
      std::cout << custom_strings[1] << ". Confidence: " << second_value << "\n";
    }
  } else {
    HOLOSCAN_LOG_INFO("Input data size: {}", dsize);
    HOLOSCAN_LOG_INFO("This is binary classification custom print, size must be 2.");
  }
  return InferStatus();
}

InferStatus DataProcessor::scale_intensity_cpu(const std::vector<int>& dimensions,
                                               const void* indata,
                                               std::vector<int64_t>& processed_dims,
                                               DataMap& processed_data_map,
                                               const std::vector<std::string>& output_tensors) {
  if (output_tensors.size() == 0) {
    return InferStatus(holoinfer_code::H_ERROR,
                       "Data processor, Output tensor size 0 in scale_intensity_cpu.");
  }

  if (output_tensors.size() > 1) {
    HOLOSCAN_LOG_WARN(
        "Scale Intensity: Output tensor size greater than 1, only the first tensor is used.");
  }

  auto out_tensor_name = output_tensors[0];  // only one output tensor supported

  if (dimensions.size() != 3) {
    HOLOSCAN_LOG_ERROR("Input data dimension must be in CHW format.");
    return InferStatus(holoinfer_code::H_ERROR, "Data processor, Incorrect data format.");
  }

  if (dimensions[0] != 1) {
    HOLOSCAN_LOG_ERROR("Input data must be single channel.");
    return InferStatus(holoinfer_code::H_ERROR, "Data processor, Incorrect data channel.");
  }

  if (dimensions[1] < 1 || dimensions[2] < 1) {
    HOLOSCAN_LOG_ERROR("All elements in input dimension must be greater than 0");
    return InferStatus(holoinfer_code::H_ERROR, "Data processor, Incorrect input data size.");
  }

  size_t dsize = accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<size_t>());

  int channels = 3;
  if (processed_data_map.find(out_tensor_name) == processed_data_map.end()) {
    HOLOSCAN_LOG_INFO("Allocating memory for {} in scale_intensity_cpu", out_tensor_name);
    processed_data_map.insert(
        {out_tensor_name, std::make_shared<DataBuffer>(holoinfer_datatype::h_UInt8)});

    // allocate memory for the first time
    processed_data_map.at(out_tensor_name)->host_buffer.resize(dsize * channels);

    //  Data in HWC format for Holoviz, input is CHW format
    processed_dims.push_back(static_cast<int64_t>(dimensions[1]));
    processed_dims.push_back(static_cast<int64_t>(dimensions[2]));
    processed_dims.push_back(channels);
  }

  auto processed_data =
      static_cast<uint8_t*>(processed_data_map.at(out_tensor_name)->host_buffer.data());
  auto input_data = static_cast<const float*>(indata);
  float max = 0, min = 100000;

  for (auto index = 0; index < dsize; index++) {
    float v = input_data[index];
    if (max < v) { max = v; }
    if (min > v) { min = v; }
  }

  std::vector<uint32_t> histogram(256, 0);

  for (auto index = 0; index < dsize; index++) {
    auto value = uint8_t(255 * ((input_data[index] - min) / (max - min)));
    histogram[value]++;
  }

  std::vector<uint32_t> cdf_histogram(256);

  cdf_histogram[0] = histogram[0];
  for (int i = 1; i < 256; i++) { cdf_histogram[i] = histogram[i] + cdf_histogram[i - 1]; }

  uint32_t cdf_histogram_min = dsize, cdf_histogram_max = 0;
  for (int i = 0; i < 256; i++) {
    int32_t count = cdf_histogram[i];
    if (count < cdf_histogram_min) { cdf_histogram_min = count; }
    if (count > cdf_histogram_max) { cdf_histogram_max = count; }
  }

  std::vector<uint32_t> updated_histogram(256);
  for (int i = 0; i < 256; i++) {
    updated_histogram[i] =
        (uint32_t)(255.0 * (cdf_histogram[i] - cdf_histogram_min) / (dsize - cdf_histogram_min));
  }

  for (auto index = 0; index < dsize; index++) {
    auto dm_index = channels * index;
    auto fvalue = uint32_t(255 * ((input_data[index] - min) / (max - min)));
    auto value = updated_histogram[fvalue];
    for (int c = 0; c < channels; c++) { processed_data[dm_index + c] = value; }
  }

  return InferStatus();
}

InferStatus DataProcessor::compute_max_per_channel_cpu(
    const std::vector<int>& dimensions, const void* indata, std::vector<int64_t>& processed_dims,
    DataMap& processed_data_map, const std::vector<std::string>& output_tensors) {
  if (output_tensors.size() == 0) {
    return InferStatus(holoinfer_code::H_ERROR,
                       "Data processor, Output tensor size 0 in compute_max_per_channel.");
  }
  if (dimensions.size() != 4) {
    return InferStatus(
        holoinfer_code::H_ERROR,
        "Data processor, Input dimensions expected in NHWC format in compute_max_per_channel.");
  }
  auto out_tensor_name = output_tensors[0];  // only one output tensor supported
  // Assuming NHWC format
  const size_t rows = dimensions[1];
  const size_t cols = dimensions[2];
  const size_t out_channels = dimensions[dimensions.size() - 1];

  if (processed_data_map.find(out_tensor_name) == processed_data_map.end()) {
    // By default, create the float data type.
    HOLOSCAN_LOG_INFO("Allocating memory for {} in compute_max_per_channel", out_tensor_name);
    processed_data_map.insert({out_tensor_name, std::make_shared<DataBuffer>()});

    // allocate memory for the first time
    if (processed_data_map.at(out_tensor_name)->host_buffer.size() == 0) {
      // this is custom allocation for max per channel. (x, y)
      processed_data_map.at(out_tensor_name)->host_buffer.resize(2 * out_channels);
    }
    processed_dims.push_back(1);  // CHECK: if disabled, get_data_from_tensor fails.
    processed_dims.push_back(static_cast<int64_t>(2 * out_channels));
  }

  auto outdata = processed_data_map.at(out_tensor_name)->host_buffer.data();

  auto input_data = static_cast<const float*>(indata);
  auto processed_data = static_cast<float*>(outdata);
  std::vector<unsigned int> max_x_per_channel, max_y_per_channel;

  max_x_per_channel.resize(out_channels, 0);
  max_y_per_channel.resize(out_channels, 0);
  std::vector<float> maxV(out_channels, -1999);

  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      for (unsigned int c = 1; c < out_channels; c++) {
        unsigned int index = i * cols * out_channels + j * out_channels + c;
        float v1 = input_data[index];
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

  max_x_per_channel.clear();
  max_y_per_channel.clear();
  return InferStatus();
}

InferStatus DataProcessor::process_operation(const std::string& operation,
                                             const std::vector<int>& indims, const void* indata,
                                             std::vector<int64_t>& processed_dims,
                                             DataMap& processed_data_map,
                                             const std::vector<std::string>& output_tensors,
                                             const std::vector<std::string>& custom_strings) {
  if (indata == nullptr) {
    return InferStatus(holoinfer_code::H_ERROR,
                       "Data processor, Operation " + operation + ", Invalid input buffer");
  }

  if (oper_to_fp_.find(operation) == oper_to_fp_.end())
    return InferStatus(holoinfer_code::H_ERROR,
                       "Data processor, Operation " + operation + " not found in map");
  try {
    return oper_to_fp_.at(operation)(
        indims, indata, processed_dims, processed_data_map, output_tensors, custom_strings);
  } catch (...) {
    return InferStatus(holoinfer_code::H_ERROR,
                       "Data processor, Exception in running " + operation);
  }
  if (operation.find("print") == std::string::npos && processed_data_map.size() == 0) {
    return InferStatus(holoinfer_code::H_ERROR, "Data processor, Processed data map empty");
  }
  return InferStatus();
}

}  // namespace inference
}  // namespace holoscan
