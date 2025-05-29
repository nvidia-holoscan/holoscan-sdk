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

#ifndef HOLOINFER_SRC_INCLUDE_HOLOINFER_UTILS_HPP
#define HOLOINFER_SRC_INCLUDE_HOLOINFER_UTILS_HPP

#include <sys/utsname.h>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "holoinfer_buffer.hpp"
#include "holoinfer_constants.hpp"

namespace holoscan {
namespace inference {

/**
 * @brief Checks Cuda result status
 * @param result Cuda result code
 */
cudaError_t check_cuda(cudaError_t result);

/**
 * Reports error with module, submodule and message, but does not throw an exception
 *
 * @param module    Module of error occurrence
 * @param submodule Submodule/Function of error occurrence with the error message (as string)
 */
int _HOLOSCAN_EXTERNAL_API_ report_error(const std::string& module, const std::string& submodule);

/**
 * Raise error with module, submodule and message
 *
 * @param module    Module of error occurrence
 * @param submodule Submodule/Function of error occurrence with the error message (as string)
 * @returns 1 (enum value of GXF_FAILURE)
 */
void _HOLOSCAN_EXTERNAL_API_ raise_error(const std::string& module, const std::string& submodule);

/**
 * @brief Checks for correctness of inference parameters from configuration.
 * @param model_path_map Map with model name as key, path to model as value
 * @param pre_processor_map Map of model name as key, mapped to vector of tensor names
 * @param inference_map Map with model name as key, mapped to vector of output tensor names
 * @param in_tensor_names Input tensor names
 * @param out_tensor_names Output tensor names
 * @return InferStatus with appropriate code and message
 */
InferStatus inference_validity_check(const Mappings& model_path_map,
                                     const MultiMappings& pre_processor_map,
                                     const MultiMappings& inference_map,
                                     std::vector<std::string>& in_tensor_names,
                                     std::vector<std::string>& out_tensor_names);

/**
 * @brief Checks for correctness of processing parameters from configuration.
 * @param processed_map Map with input tensor name as key, mapped to vector of output tensor names
 * @param in_tensor_names Input tensor names
 * @param out_tensor_names Output tensor names
 */
InferStatus processor_validity_check(const MultiMappings& processed_map,
                                     const std::vector<std::string>& in_tensor_names,
                                     const std::vector<std::string>& out_tensor_names);

/**
 * @brief Checks if the processor is arm based
 */
bool is_platform_aarch64();

void timer_init(TimePoint& _t);

/**
 * @brief Logs the module name and (end - start) time difference (at DEBUG level).
 *
 * @returns 0 if successful.
 */
int timer_check(TimePoint& start, TimePoint& end, const std::string& module);

void string_split(const std::string& line, std::vector<std::string>& tokens, char c);

/**
 * @brief Checks for correctness of input tensor dimensions.
 * @param pre_processor_map Map with model name as key, mapped to vector of input tensor names
 * @param model_input_dimensions Map with model name as key, mapped to input dimensions
 * @param dims_per_tensor Map with input tensor as key, mapped to its dimension
 */
InferStatus tensor_dimension_check(const MultiMappings& pre_processor_map,
                                   const DimType& model_input_dimensions,
                                   const std::map<std::string, std::vector<int>>& dims_per_tensor);

using node_type = std::map<std::string, std::map<std::string, std::string>>;
static const std::map<std::string, holoinfer_datatype> kHoloInferDataTypeMap = {
    {"kFloat32", holoinfer_datatype::h_Float32},
    {"kInt32", holoinfer_datatype::h_Int32},
    {"kInt8", holoinfer_datatype::h_Int8},
    {"kUInt8", holoinfer_datatype::h_UInt8},
    {"kInt64", holoinfer_datatype::h_Int64},
    {"kFloat16", holoinfer_datatype::h_Float16}};

InferStatus parse_yaml_node(const node_type& in_config, std::vector<std::string>& names,
                            std::vector<std::vector<int64_t>>& dims,
                            std::vector<holoinfer_datatype>& types);
}  // namespace inference
}  // namespace holoscan
#endif /* HOLOINFER_SRC_INCLUDE_HOLOINFER_UTILS_HPP */
