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
#ifndef MODULES_HOLOINFER_SRC_PROCESS_DATA_PROCESSOR_HPP
#define MODULES_HOLOINFER_SRC_PROCESS_DATA_PROCESSOR_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <bits/stdc++.h>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <holoinfer.hpp>
#include <holoinfer_constants.hpp>
#include <holoinfer_utils.hpp>

#include <holoscan/core/analytics/data_exporter.hpp>
#include <process/transforms/generate_boxes.hpp>

namespace holoscan {
namespace inference {
/// Declaration of function callback used by DataProcessor. processor_FP is defined for operations
/// with fixed (currently one) input and output size, and for operations that do not need any
/// configuration
using processor_FP = std::function<InferStatus(
    const std::vector<int>&, const void*, std::vector<int64_t>&, DataMap&,
    const std::vector<std::string>& output_tensors, const std::vector<std::string>& custom_strings,
    bool process_with_cuda, cudaStream_t cuda_stream)>;

/// Declaration of function callback for custom cuda kernels used by DataProcessor.
using cuda_FP = std::function<InferStatus(const std::string&, const std::vector<int>&, const void*,
                                          std::vector<int64_t>&, DataMap&,
                                          const std::vector<std::string>& output_tensors,
                                          bool process_with_cuda, cudaStream_t cuda_stream)>;

// Declaration of function callback for transforms that need configuration (via a yaml file).
// Transforms additionally support multiple inputs and outputs from the processing.
using transforms_FP =
    std::function<InferStatus(const std::string&, const std::map<std::string, void*>&,
                              const std::map<std::string, std::vector<int>>&, DataMap&, DimType&)>;

/**
 * @brief Data Processor class that processes operations. Currently supports CPU based operations.
 */
class DataProcessor {
 public:
  /**
   * @brief Default Constructor
   */
  DataProcessor() {}

  /**
   * @brief Default Destructor
   */
  ~DataProcessor();

  /**
   * @brief Checks the validity of supported operations
   *
   * @param process_operations Map where tensor name is the key, and operations to perform on
   * the tensor as vector of strings. Each value in the vector of strings is the supported
   * operation.
   * @param custom_kernels Map of custom kernel identifier, mapped to related value as a string
   * @param config_path Path to the processing configuration settings
   *
   * @return InferStatus with appropriate code and message
   */
  InferStatus initialize(const MultiMappings& process_operations, const Mappings& custom_kernels,
                         const std::string config_path);

  /**
   * @brief Executes an operation via function callback.
   *
   * @param operation Operation to perform. Refer to user docs for a list of supported operations
   * @param in_dims Dimension of the input tensor
   * @param in_data Input data buffer
   * @param processed_dims Dimension of the output tensor, is populated during the processing
   * @param processed_data_map Output data map, that will be populated
   * @param output_tensors Tensor names to be populated in the out_data_map
   * @param custom_strings Strings to display for custom print operations
   * @param process_with_cuda Flag defining if processing should be done with CUDA
   * @param cuda_stream CUDA stream to use when procseeing is done with CUDA
   * @return InferStatus with appropriate code and message
   */
  InferStatus process_operation(const std::string& operation, const std::vector<int>& in_dims,
                                const void* in_data, std::vector<int64_t>& processed_dims,
                                DataMap& processed_data_map,
                                const std::vector<std::string>& output_tensors,
                                const std::vector<std::string>& custom_strings,
                                bool process_with_cuda, cudaStream_t cuda_stream);

  /**
   * @brief Executes a transform via function callback. (Currently CPU based)
   *
   * @param transform Data transform operation to perform.
   * @param key String identifier for the transform
   * @param indata Map with key as tensor name and value as data buffer
   * @param indims Map with key as tensor name and value as dimension of the input tensor
   * @param processed_data Output data map, that will be populated
   * @param processed_dims Dimension of the output tensor, is populated during the processing
   * @return InferStatus with appropriate code and message
   */
  InferStatus process_transform(const std::string& transform, const std::string& key,
                                const std::map<std::string, void*>& indata,
                                const std::map<std::string, std::vector<int>>& indim,
                                DataMap& processed_data, DimType& processed_dims);

  /**
   * @brief Computes max per channel in input data and scales it to [0, 1]. (supports both GPU and
   * CPU data)
   *
   * @param operation Operation to perform. Refer to user docs for a list of supported operations
   * @param in_dims Dimension of the input tensor
   * @param in_data Input data buffer
   * @param out_dims Dimension of the output tensor
   * @param out_data_map Output data buffer map
   * @param output_tensors Output tensor names, used to populate out_data_map
   * @param process_with_cuda Flag defining if processing should be done with CUDA
   * @param cuda_stream CUDA stream to use when procseeing is done with CUDA
   */
  InferStatus compute_max_per_channel_scaled(const std::vector<int>& in_dims, const void* in_data,
                                             std::vector<int64_t>& out_dims, DataMap& out_data_map,
                                             const std::vector<std::string>& output_tensors,
                                             bool process_with_cuda, cudaStream_t cuda_stream);

  /**
   * @brief Scales intensity using min-max values and histogram. (CPU based)
   *
   * @param operation Operation to perform. Refer to user docs for a list of supported operations
   * @param in_dims Dimension of the input tensor
   * @param in_data Input data buffer
   * @param out_dims Dimension of the output tensor
   * @param out_data_map Output data buffer map
   * @param output_tensors Output tensor names, used to populate out_data_map
   */
  InferStatus scale_intensity_cpu(const std::vector<int>& in_dims, const void* in_data,
                                  std::vector<int64_t>& out_dims, DataMap& out_data_map,
                                  const std::vector<std::string>& output_tensors);

  /**
   * @brief Print data in the input buffer in float32. Ideally to be used by classification models.
   *
   * @param in_dims Dimension of the input tensor
   * @param in_data Input data buffer
   */
  InferStatus print_results(const std::vector<int>& in_dims, const void* in_data);

  /**
   * @brief Print data in the input buffer in int32 form. Ideally to be used by classification
   * models.
   *
   * @param in_dims Dimension of the input tensor
   * @param in_data Input data buffer
   */
  InferStatus print_results_int32(const std::vector<int>& in_dims, const void* in_data);

  /**
   * @brief Print custom text for binary classification results in the input buffer.
   *
   * @param in_dims Dimension of the input tensor
   * @param in_data Input data buffer
   * @param custom_strings Strings to display for custom print operations
   */
  InferStatus print_custom_binary_classification(const std::vector<int>& in_dims,
                                                 const void* in_data,
                                                 const std::vector<std::string>& custom_strings);

  /**
   * @brief Export binary classification results in the input buffer to CSV file using Data Exporter
   *        API.
   *
   * @param in_dims Dimension of the input tensor
   * @param in_data Input data buffer
   * @param custom_strings The comma separated list of strings containing information for
   *                       the output CSV file. It should include application name as a first string
   *                       required for the Data Exporter API and column names.
   */
  InferStatus export_binary_classification_to_csv(const std::vector<int>& in_dims,
                                                  const void* in_data,
                                                  const std::vector<std::string>& custom_strings);
  /**
   * @brief Launches custom kernel at runtime.
   *
   * @param id Unique custom kernel id
   * @param dimensions Dimensions of input buffer
   * @param input Input data buffer
   * @param processed_dims Dimension of the output tensor, is populated during the processing
   * @param processed_data_map Output data map, that will be populated
   * @param output_tensors Output tensor names, used to populate out_data_map
   * @param process_with_cuda Flag defining if processing should be done with CUDA
   * @param cuda_stream CUDA stream to use when procseeing is done with CUDA
   */
  InferStatus launchCustomKernel(const std::string& id, const std::vector<int>& dimensions,
                                 const void* input, std::vector<int64_t>& processed_dims,
                                 DataMap& processed_data_map,
                                 const std::vector<std::string>& output_tensors,
                                 bool process_with_cuda, cudaStream_t cuda_stream);

  /**
   * @brief Initialization and preparation of all custom cuda kernels
   */
  InferStatus prepareCustomKernel();

 private:
  /// Map defining supported operations by DataProcessor Class.
  /// Keyword in this map must be used exactly by the user in configuration.
  /// Operation is the key and its related implementation platform as the value.
  /// Operations are defined with fixed number of input and outputs. Currently one for each.
  inline static const std::map<std::string, holoinfer_data_processor> supported_compute_operations_{
      {"max_per_channel_scaled", holoinfer_data_processor::h_CUDA_AND_HOST},
      {"custom_cuda_kernel", holoinfer_data_processor::h_CUDA_AND_HOST},
      {"scale_intensity_cpu", holoinfer_data_processor::h_HOST}};

  /// Map defining supported transforms by DataProcessor Class.
  /// Keyword in this map must be used exactly by the user in configuration.
  /// Transform is the key and its related implementation platform as the value.
  /// Transforms are defined with support for multiple input and outputs. Output tensors can be
  /// dynamic and are generated and populated at run time. Transforms need a configuration file for
  /// the setup.
  inline static const std::map<std::string, holoinfer_data_processor> supported_transforms_{
      {"generate_boxes", holoinfer_data_processor::h_HOST}};

  // Map with operation name as key, with pointer to its object
  std::map<std::string, std::unique_ptr<TransformBase>> transforms_;

  inline static const std::map<std::string, holoinfer_data_processor> supported_print_operations_{
      {"print", holoinfer_data_processor::h_HOST},
      {"print_int32", holoinfer_data_processor::h_HOST},
      {"print_custom_binary_classification", holoinfer_data_processor::h_HOST}};

  /// Map defining supported formats by DataProcessor Class to export results using Data Exporter
  /// API.
  inline static const std::map<std::string, holoinfer_data_processor> supported_export_operations_{
      {"export_binary_classification_to_csv", holoinfer_data_processor::h_HOST}};

  /// Mapped function call for the function pointer of max_per_channel_scaled
  processor_FP max_per_channel_scaled_fp_ =
      [this](auto& in_dims, const void* in_data, std::vector<int64_t>& out_dims, DataMap& out_data,
             auto& output_tensors, auto& custom_strings, bool process_with_cuda,
             cudaStream_t cuda_stream) {
        return compute_max_per_channel_scaled(
            in_dims, in_data, out_dims, out_data, output_tensors, process_with_cuda, cuda_stream);
      };

  /// Mapped function call for the function pointer of scale_intensity_cpu
  processor_FP scale_intensity_cpu_fp_ = [this](auto& in_dims, const void* in_data,
                                                std::vector<int64_t>& out_dims, DataMap& out_data,
                                                auto& output_tensors, auto& custom_strings,
                                                bool process_with_cuda, cudaStream_t cuda_stream) {
    return scale_intensity_cpu(in_dims, in_data, out_dims, out_data, output_tensors);
  };

  /// Mapped function call for the function pointer of print
  processor_FP print_results_fp_ =
      [this](auto& in_dims, const void* in_data, std::vector<int64_t>& out_dims, DataMap& out_data,
             auto& output_tensors, auto& custom_strings, bool process_with_cuda,
             cudaStream_t cuda_stream) { return print_results(in_dims, in_data); };

  /// Mapped function call for the function pointer of printing custom binary classification
  /// results
  processor_FP print_custom_binary_classification_fp_ =
      [this](auto& in_dims, const void* in_data, std::vector<int64_t>& out_dims, DataMap& out_data,
             auto& output_tensors, auto& custom_strings, bool process_with_cuda,
             cudaStream_t cuda_stream) {
        return print_custom_binary_classification(in_dims, in_data, custom_strings);
      };

  /// Mapped function call for the function pointer of exporting binary classification
  /// results to the CSV file using the Data Exporter API.
  processor_FP export_binary_classification_to_csv_fp_ =
      [this](auto& in_dims, const void* in_data, std::vector<int64_t>& out_dims, DataMap& out_data,
             auto& output_tensors, auto& custom_strings, bool process_with_cuda,
             cudaStream_t cuda_stream) {
        return export_binary_classification_to_csv(in_dims, in_data, custom_strings);
      };

  /// Mapped function call for the function pointer of print int32
  processor_FP print_results_i32_fp_ =
      [this](auto& in_dims, const void* in_data, std::vector<int64_t>& out_dims, DataMap& out_data,
             auto& output_tensors, auto& custom_strings, bool process_with_cuda,
             cudaStream_t cuda_stream) { return print_results_int32(in_dims, in_data); };

  /// Mapped function call for the function pointer of custom_cuda_kernel
  cuda_FP custom_cuda_kernel_fp_ = [this](const std::string& id, auto& in_dims, const void* in_data,
                                          std::vector<int64_t>& out_dims, DataMap& out_data,
                                          auto& output_tensors, bool process_with_cuda,
                                          cudaStream_t cuda_stream) {
    return launchCustomKernel(
        id, in_dims, in_data, out_dims, out_data, output_tensors, process_with_cuda, cuda_stream);
  };

  /// Map with supported operation as the key and related function pointer as value
  const std::map<std::string, processor_FP> oper_to_fp_{
      {"max_per_channel_scaled", max_per_channel_scaled_fp_},
      {"scale_intensity_cpu", scale_intensity_cpu_fp_},
      {"print", print_results_fp_},
      {"print_int32", print_results_i32_fp_},
      {"print_custom_binary_classification", print_custom_binary_classification_fp_},
      {"export_binary_classification_to_csv", export_binary_classification_to_csv_fp_}};

  /// Map with custom cuda operation as the key and related function pointer as value
  const std::map<std::string, cuda_FP> cuda_to_fp_{{"custom_cuda_kernel", custom_cuda_kernel_fp_}};

  /// Mapped function call for the function pointer of generate_boxes
  transforms_FP generate_boxes_fp_ = [this](const std::string& key,
                                            const std::map<std::string, void*>& indata,
                                            const std::map<std::string, std::vector<int>>& indim,
                                            DataMap& processed_data, DimType& processed_dims) {
    return transforms_.at(key)->execute(indata, indim, processed_data, processed_dims);
  };

  /// Map with supported transforms as the key and related function pointer as value
  const std::map<std::string, transforms_FP> transform_to_fp_{
      {"generate_boxes", generate_boxes_fp_}};

  /**
   * @brief Computes max per channel in input data and scales it to [0, 1], CUDA version
   *
   * @param rows rows
   * @param cols Dimension of the input tensor
   * @param channels Input data buffer
   * @param indata Input data
   * @param outdata Output data
   * @param cuda_stream CUDA stream to use when procseeing is done with CUDA
   */
  void max_per_channel_scaled_cuda(size_t rows, size_t cols, size_t channels, const float* indata,
                                   float* outdata, cudaStream_t cuda_stream);

  /// Configuration path
  std::string config_path_ = {};

  /// Data exporter
  std::unique_ptr<DataExporter> data_exporter_ = nullptr;

  // Custom CUDA kernel feature related parameters

  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
  CUmodule module_ = nullptr;
  std::string custom_cuda_src_;
  std::map<std::string, CUfunction> kernel_;
  std::map<std::string, holoinfer_datatype> output_dtype_;
  std::map<std::string, std::string> custom_kernel_thread_per_block_;
};

}  // namespace inference
}  // namespace holoscan

#endif /* MODULES_HOLOINFER_SRC_PROCESS_DATA_PROCESSOR_HPP */
