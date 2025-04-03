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

#include "data_processor.hpp"

#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/analytics/csv_data_exporter.hpp>

namespace holoscan {
namespace inference {

InferStatus DataProcessor::initialize(const MultiMappings& process_operations,
                                      const Mappings& custom_kernels,
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
      std::vector<std::string> operation_strings;
      string_split(_op, operation_strings, ',');
      std::string operation = operation_strings[0];
      if (supported_transforms_.find(operation) != supported_transforms_.end()) {
        // In future releases, this will be generalized with addition of more transforms.
        if (operation == "generate_boxes") {
          // unique key is created as a combination of input tensors and operation
          auto key = fmt::format("{}-{}", p_op.first, operation);
          HOLOSCAN_LOG_INFO("Transform map updated with key: {}", key);
          transforms_.insert({key, std::make_unique<GenerateBoxes>(config_path_)});

          std::vector<std::string> tensor_tokens;
          string_split(p_op.first, tensor_tokens, ':');

          auto status = transforms_.at(key)->initialize(tensor_tokens);
          if (status.get_code() != holoinfer_code::H_SUCCESS) {
            status.display_message();
            return status;
          }
          continue;
        }
      }

      if (operation.find("custom_cuda_kernel") != std::string::npos) {
        //  Get the custom cuda kernel ID
        std::vector<std::string> oper_name_split;
        string_split(operation, oper_name_split, '-');
        operation = "custom_cuda_kernel";

        if (oper_name_split.size() != 2) {
          HOLOSCAN_LOG_ERROR("Operation name not as per specifications. {}", operation);
          HOLOSCAN_LOG_ERROR(
              "Custom cuda operation name must follow the following format: "
              "custom_cuda_kernel-<identifier>. A dash (-) must separate the operation "
              "(custom_cuda_kernel) and kernel identifier.");
          return InferStatus(
              holoinfer_code::H_ERROR,
              "Data processor, custom cuda kernel not defined as per specifications.");
        }
        auto kernel_identifier = oper_name_split[1];
        HOLOSCAN_LOG_INFO("Custom kernel Identifier: {}", kernel_identifier);

        // From all entries in custom_kernels, find multiple custom cuda kernels and store them in
        // a map
        if (custom_kernels.size() == 0) {
          HOLOSCAN_LOG_ERROR("Custom kernels not defined in the parameter set");
          return InferStatus(holoinfer_code::H_ERROR,
                             "Data processor, custom cuda kernel map does not exist.");
        }

        std::string current_kernel = "cuda_kernel-" + kernel_identifier;
        if (custom_kernels.find(current_kernel) == custom_kernels.end()) {
          HOLOSCAN_LOG_ERROR("Custom cuda kernel {} not defined in custom_kernels map.",
                             current_kernel);
          return InferStatus(holoinfer_code::H_ERROR,
                             "Data processor, cuda kernel not defined in custom cuda kernel map.");
        }

        if (custom_kernels.at(current_kernel).length() == 0) {
          HOLOSCAN_LOG_ERROR("Custom cuda kernel {} empty in custom_kernels map.", current_kernel);
          return InferStatus(holoinfer_code::H_ERROR,
                             "Data processor, cuda kernel length cannot be 0.");
        }

        // import from file if the input is a file path
        const std::string cuda_suffix{".cu"};
        std::string kernel_path = custom_kernels.at(current_kernel);
        bool has_cuda_suffix =
            kernel_path.compare(
                kernel_path.size() - cuda_suffix.size(), cuda_suffix.size(), cuda_suffix) == 0;

        if (has_cuda_suffix) {
          if (!std::filesystem::exists(kernel_path)) {
            HOLOSCAN_LOG_ERROR("Custom cuda kernel file not found: {}", kernel_path);
            return InferStatus(holoinfer_code::H_ERROR,
                               "Data processor, custom cuda kernel file not found.");
          }
          std::ifstream file(kernel_path);
          std::stringstream buffer;
          buffer << file.rdbuf();
          custom_cuda_src_ += buffer.str();
          HOLOSCAN_LOG_DEBUG("Cuda kernel source: {}", buffer.str());
        } else {
          custom_cuda_src_ += custom_kernels.at(current_kernel);
          const char* cuda_src = custom_kernels.at(current_kernel).c_str();
          HOLOSCAN_LOG_DEBUG("Cuda kernel source: {}", cuda_src);
        }

        // extract output datatype
        std::string current_out_dtype = "out_dtype-" + kernel_identifier;
        if (custom_kernels.find(current_out_dtype) != custom_kernels.end()) {
          auto output_dtype_string = custom_kernels.at(current_out_dtype);

          if (kHoloInferDataTypeMap.find(output_dtype_string) == kHoloInferDataTypeMap.end()) {
            HOLOSCAN_LOG_ERROR(
                "Data processor, Incorrect output data type {} in custom CUDA kernel.",
                output_dtype_string);
            HOLOSCAN_LOG_INFO(
                "Supported data type values are: kFloat32, kInt32, kInt8, kUInt8, KInt64, "
                "kFloat16");
            return InferStatus(holoinfer_code::H_ERROR,
                               "Data processor, Incorrect output data type in custom CUDA kernel.");
          }

          output_dtype_[kernel_identifier] = kHoloInferDataTypeMap.at(output_dtype_string);
        } else {
          output_dtype_[kernel_identifier] = holoinfer_datatype::h_Float32;
          HOLOSCAN_LOG_WARN(
              "Output datatype not specified in custom cuda map. Going with default datatype: "
              "float32");
        }

        // extract threads per block
        std::string current_threads_per_block = "thread_per_block-" + kernel_identifier;
        if (custom_kernels.find(current_threads_per_block) != custom_kernels.end()) {
          auto tpb_string = custom_kernels.at(current_threads_per_block);
          custom_kernel_thread_per_block_[kernel_identifier] = std::move(tpb_string);
        } else {
          custom_kernel_thread_per_block_[kernel_identifier] = "256";
          HOLOSCAN_LOG_WARN(
              "Threads per block not specific in custom kernels map with {}. Assuming it to be a "
              "1D kernel and setting thread_per_block as 256",
              current_threads_per_block);
        }
      }

      if (supported_compute_operations_.find(operation) == supported_compute_operations_.end() &&
          supported_print_operations_.find(operation) == supported_print_operations_.end() &&
          supported_export_operations_.find(operation) == supported_export_operations_.end()) {
        return InferStatus(holoinfer_code::H_ERROR,
                           "Data processor: Initializer, Operation " + _op + " not supported.");
      }
    }
  }

  if (custom_cuda_src_.length() != 0) {
    try {
      auto status = prepareCustomKernel();
      if (status.get_code() != holoinfer_code::H_SUCCESS) {
        HOLOSCAN_LOG_ERROR("Error in initializing custom cuda kernels");
        return status;
      }
    } catch (...) {
      return InferStatus(holoinfer_code::H_ERROR,
                         "Data processor, Exception in initializing custom cuda kernels.");
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

InferStatus DataProcessor::export_binary_classification_to_csv(
    const std::vector<int>& dimensions, const void* indata,
    const std::vector<std::string>& custom_strings) {
  size_t dsize = accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<size_t>());
  if (dsize < 1) {
    HOLOSCAN_LOG_ERROR("Input data size must be at least 1.");
    return InferStatus(holoinfer_code::H_ERROR, "Data processor, Incorrect input data size");
  }
  auto indata_float = static_cast<const float*>(indata);

  auto strings_count = custom_strings.size();
  if (strings_count != 4) {
    HOLOSCAN_LOG_INFO("The number of custom strings passed : {}", strings_count);
    HOLOSCAN_LOG_INFO(
        "This is export binary classification results to CSV file operation, size must be 4.");
    return InferStatus();
  }

  if (dsize != 2) {
    HOLOSCAN_LOG_INFO("Input data size: {}", dsize);
    HOLOSCAN_LOG_INFO(
        "This is export binary classification results to CSV file operation, size must be 2.");
    return InferStatus();
  }

  if (!data_exporter_) {
    const std::string app_name = custom_strings[0];
    const std::vector<std::string> columns = {
        custom_strings[1], custom_strings[2], custom_strings[3]};
    data_exporter_ = std::make_unique<CsvDataExporter>(app_name, columns);
  }

  auto first_value = 1.0 / (1 + exp(-indata_float[0]));
  auto second_value = 1.0 / (1 + exp(-indata_float[1]));
  std::vector<std::string> data;
  if (first_value > second_value) {
    std::ostringstream confidence_score_ss;
    confidence_score_ss << first_value;
    data = {"1", "0", confidence_score_ss.str()};
  } else {
    std::ostringstream confidence_score_ss;
    confidence_score_ss << second_value;
    data = {"0", "1", confidence_score_ss.str()};
  }
  data_exporter_->export_data(data);

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
    processed_data_map.at(out_tensor_name)->host_buffer_->resize(dsize * channels);

    //  Data in HWC format for Holoviz, input is CHW format
    processed_dims.push_back(static_cast<int64_t>(dimensions[1]));
    processed_dims.push_back(static_cast<int64_t>(dimensions[2]));
    processed_dims.push_back(channels);
  }

  auto processed_data =
      static_cast<uint8_t*>(processed_data_map.at(out_tensor_name)->host_buffer_->data());
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

InferStatus DataProcessor::compute_max_per_channel_scaled(
    const std::vector<int>& dimensions, const void* indata, std::vector<int64_t>& processed_dims,
    DataMap& processed_data_map, const std::vector<std::string>& output_tensors,
    bool process_with_cuda, cudaStream_t cuda_stream) {
  if (output_tensors.size() == 0) {
    return InferStatus(holoinfer_code::H_ERROR,
                       "Data processor, Output tensor size 0 in compute_max_per_channel_scaled.");
  }
  if (dimensions.size() != 4) {
    return InferStatus(holoinfer_code::H_ERROR,
                       "Data processor, Input dimensions expected in NHWC format in "
                       "compute_max_per_channel_scaled.");
  }
  auto out_tensor_name = output_tensors[0];  // only one output tensor supported
  // Assuming NHWC format
  const size_t rows = dimensions[1];
  const size_t cols = dimensions[2];
  const size_t out_channels = dimensions[dimensions.size() - 1];

  if (processed_data_map.find(out_tensor_name) == processed_data_map.end()) {
    // By default, create the float data type.
    HOLOSCAN_LOG_INFO("Allocating memory for {} in compute_max_per_channel_scaled",
                      out_tensor_name);
    const auto [db, success] =
        processed_data_map.insert({out_tensor_name, std::make_shared<DataBuffer>()});

    // this is custom allocation for max per channel. (x, y)
    if (process_with_cuda) {
      db->second->device_buffer_->resize(2 * out_channels);
    } else {
      db->second->host_buffer_->resize(2 * out_channels);
    }
    processed_dims.push_back(1);  // CHECK: if disabled, get_data_from_tensor fails.
    processed_dims.push_back(static_cast<int64_t>(2 * out_channels));
  }

  if (process_with_cuda) {
    void* outdata = processed_data_map.at(out_tensor_name)->device_buffer_->data();

    max_per_channel_scaled_cuda(rows,
                                cols,
                                out_channels,
                                reinterpret_cast<const float*>(indata),
                                reinterpret_cast<float*>(outdata),
                                cuda_stream);
  } else {
    void* outdata = processed_data_map.at(out_tensor_name)->host_buffer_->data();

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
      processed_data[2 * i + 1] =
          static_cast<float>(max_y_per_channel[i]) / static_cast<float>(cols);
    }

    max_x_per_channel.clear();
    max_y_per_channel.clear();
  }
  return InferStatus();
}

InferStatus DataProcessor::process_operation(const std::string& operation,
                                             const std::vector<int>& indims, const void* indata,
                                             std::vector<int64_t>& processed_dims,
                                             DataMap& processed_data_map,
                                             const std::vector<std::string>& output_tensors,
                                             const std::vector<std::string>& custom_strings,
                                             bool process_with_cuda, cudaStream_t cuda_stream) {
  if (indata == nullptr) {
    return InferStatus(
        holoinfer_code::H_ERROR,
        "Data processor: process operation, Operation " + operation + ", Invalid input buffer");
  }

  if (oper_to_fp_.find(operation) == oper_to_fp_.end()) {
    if (operation.find("custom_cuda_kernel") != std::string::npos) {
      //  Get the custom cuda kernel ID
      std::vector<std::string> oper_name_split;
      string_split(operation, oper_name_split, '-');
      if (oper_name_split.size() != 2) {
        HOLOSCAN_LOG_ERROR("Custom cuda kernel naming not as per specifications. {}", operation);
        HOLOSCAN_LOG_INFO(
            "Custom cuda kernel naming must be in following format: custom_cuda_kernel-<unique "
            "identifier>. There must be a single dash '-' in the name, used as a separator.");
        return InferStatus(holoinfer_code::H_ERROR,
                           "Data processor, Operation " + operation + " not as per specifications");
      }
      auto kernel_identifier = oper_name_split[1];
      auto cuda_operation = "custom_cuda_kernel";

      if (cuda_to_fp_.find(cuda_operation) != cuda_to_fp_.end()) {
        // custom_cuda_kernel is the generic function pointer and ID will define which custom cuda
        // kernel to call
        return cuda_to_fp_.at(cuda_operation)(kernel_identifier,
                                              indims,
                                              indata,
                                              processed_dims,
                                              processed_data_map,
                                              output_tensors,
                                              process_with_cuda,
                                              cuda_stream);
      }
    }
    return InferStatus(holoinfer_code::H_ERROR,
                       "Data processor, Operation " + operation + " not found in map");
  }
  try {
    return oper_to_fp_.at(operation)(indims,
                                     indata,
                                     processed_dims,
                                     processed_data_map,
                                     output_tensors,
                                     custom_strings,
                                     process_with_cuda,
                                     cuda_stream);
  } catch (...) {
    return InferStatus(holoinfer_code::H_ERROR,
                       "Data processor, Exception in running " + operation);
  }
}

DataProcessor::~DataProcessor() {
  const char* error_string;

  if (module_ != nullptr) {
    CUresult err = cuModuleUnload(module_);
    if (err != CUDA_SUCCESS) {
      cuGetErrorString(err, &error_string);
      HOLOSCAN_LOG_ERROR("Error unloading CUDA module: {}", error_string);
    }
  }

  if (context_ != nullptr) {
    CUresult err = cuCtxDestroy(context_);
    if (err != CUDA_SUCCESS) {
      cuGetErrorString(err, &error_string);
      HOLOSCAN_LOG_ERROR("Error destroying CUDA context: {}", error_string);
    }
  }
}

}  // namespace inference
}  // namespace holoscan
