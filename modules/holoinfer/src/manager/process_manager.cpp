/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "process_manager.hpp"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace holoscan {
namespace inference {

InferStatus ManagerProcessor::initialize(const MultiMappings& process_operations,
                                         const std::string config_path = {}) {
  try {
    infer_data_ = std::make_unique<DataProcessor>();
  } catch (const std::bad_alloc&) {
    return InferStatus(holoinfer_code::H_ERROR,
                       "Process Manager, Holoscan out data core: Memory allocation error");
  }
  return infer_data_->initialize(process_operations, config_path);
}

InferStatus ManagerProcessor::process_multi_tensor_operation(
    const std::string tensor_name, const std::vector<std::string>& operation_names,
    DataMap& inferred_result_map, const std::map<std::string, std::vector<int>>& dimension_map) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);

  std::vector<std::string> tensor_tokens;

  if (tensor_name.find(":") != std::string::npos) { string_split(tensor_name, tensor_tokens, ':'); }

  // Single operation supported on multi-tensor data
  for (const auto& operation_name : operation_names) {
    std::map<std::string, void*> all_tensor_data;
    std::map<std::string, std::vector<int>> all_tensor_dims;
    for (const auto& tensor : tensor_tokens) {
      // All tensors must be in the inferred_result_map and dimension_map
      // For multi-operation on multi-tensor data (not supported yet)
      //  1- inferred_result_map is updated with results from the previous operation
      //  2- dimension_map is updated accordingly
      //  3- both changes are local (future release)
      if (inferred_result_map.find(tensor) == inferred_result_map.end()) {
        HOLOSCAN_LOG_ERROR("Tensor {} not found in result map.", tensor);
        return status;
      }
      if (dimension_map.find(tensor) == dimension_map.end()) {
        HOLOSCAN_LOG_ERROR("Tensor {} not found in dimension map.", tensor);
        return status;
      }
      void* input_data = inferred_result_map.at(tensor)->host_buffer_->data();
      const std::vector<int> dimensions = dimension_map.at(tensor);
      all_tensor_data[tensor] = input_data;
      all_tensor_dims[tensor] = dimensions;
    }

    return infer_data_->process_transform(operation_name,
                                          tensor_name,
                                          all_tensor_data,
                                          all_tensor_dims,
                                          processed_data_map_,
                                          processed_dims_map_);
  }
  return InferStatus();
}

InferStatus ManagerProcessor::process(const MultiMappings& tensor_oper_map,
                                      const MultiMappings& in_out_tensor_map,
                                      DataMap& inferred_result_map,
                                      const std::map<std::string, std::vector<int>>& dimension_map,
                                      bool process_with_cuda,
                                      cudaStream_t cuda_stream) {
  for (const auto& current_tensor_operation : tensor_oper_map) {
    auto& tensor_name = current_tensor_operation.first;
    auto operations = current_tensor_operation.second;

    // currently one operation is supported
    if (operations.size() != 1) {
      return InferStatus(
          holoinfer_code::H_ERROR,
          "Process manager, Single operation supported per entry in tensor_operations_map.");
    }
    // If the incoming tensor is not present in incoming result map
    // then either the tensor is absent (error) or its a multi_tensor_input case
    if (inferred_result_map.find(tensor_name) == inferred_result_map.end()) {
      if (tensor_name.find(":") != std::string::npos) {
        // multi-tensors are represented as a string, different tensor names are separated by ':'
        auto status = process_multi_tensor_operation(
            tensor_name, operations, inferred_result_map, dimension_map);
        if (status.get_code() != holoinfer_code::H_SUCCESS) {
          status.display_message();
          return InferStatus(holoinfer_code::H_ERROR,
                             "Process manager, Multi tensor operation failed for " + tensor_name);
        }
      } else {  // return an error as the single tensor is absent in result map
        return InferStatus(
            holoinfer_code::H_ERROR,
            "Process manager, Inference map does not contain results from " + tensor_name);
      }
    } else {
      // If tensor is present, means its a single tensor input and single tensor output use case
      if (dimension_map.find(tensor_name) == dimension_map.end()) {
        return InferStatus(
            holoinfer_code::H_ERROR,
            "Process manager, Dimension map does not contain results from " + tensor_name);
      }

      void* input_data;
      if (process_with_cuda) {
        input_data = inferred_result_map.at(tensor_name)->device_buffer_->data();
      } else {
        input_data = inferred_result_map.at(tensor_name)->host_buffer_->data();
      }

      const std::vector<int> dimensions = dimension_map.at(tensor_name);

      for (auto& operation : operations) {
        std::vector<int64_t> processed_dims;
        std::vector<std::string> out_tensor_names, custom_strings;

        // if operation is print or export, then no need to allocate output memory
        if (operation.find("print") != std::string::npos ||
            operation.find("export") != std::string::npos) {
          if (operation.compare("print") == 0 || operation.compare("print_int32") == 0) {
            std::cout << "Printing results from " << tensor_name << " -> ";
          } else {
            if (operation.find("custom") != std::string::npos) {
              std::istringstream cstrings(operation);

              std::string custom_string;
              while (std::getline(cstrings, custom_string, ',')) {
                custom_strings.push_back(custom_string);
              }
              if (custom_strings.size() != 3) {
                return InferStatus(
                    holoinfer_code::H_ERROR,
                    "Process manager, Custom binary print operation must generate 3 strings");
              }
              operation = custom_strings[0];
              custom_strings.erase(custom_strings.begin());
            } else if (operation.find("export_binary_classification_to_csv") != std::string::npos) {
              std::istringstream cstrings(operation);

              std::string custom_string;
              while (std::getline(cstrings, custom_string, ',')) {
                custom_strings.push_back(custom_string);
              }
              if (custom_strings.size() != 5) {
                return InferStatus(
                    holoinfer_code::H_ERROR,
                    "Process manager, Export output to CSV operation must generate 5 strings");
              }
              operation = custom_strings[0];
              custom_strings.erase(custom_strings.begin());
            }
          }
        } else {
          // Input tensor must have a mapped output tensor in in_out_tensor_map.
          // Mapped tensors are returned with populated databuffers which are then transmitted
          // to be ingested by next connecting operators.
          // Currently the design assumes, for one input tensor there will be one entry in
          // operations map. For e.g. A tensor "input" can only do one "operation", thus there
          // will be only one entry for tensor "input" in tensor_oper_map.
          // In upcoming releases
          //  1. Same tensor "input" can have multiple entries to execute multiple operations
          //  2. in_out_tensor_map will be updated to multiple outputs mapped to a single input
          if (in_out_tensor_map.find(tensor_name) == in_out_tensor_map.end()) {
            return InferStatus(
                holoinfer_code::H_ERROR,
                "Process manager, In tensor " + tensor_name + " has no out tensor mapping");
          }
          out_tensor_names = in_out_tensor_map.at(tensor_name);
        }
        InferStatus status = infer_data_->process_operation(operation,
                                                            dimensions,
                                                            input_data,
                                                            processed_dims,
                                                            processed_data_map_,
                                                            out_tensor_names,
                                                            custom_strings,
                                                            process_with_cuda,
                                                            cuda_stream);

        if (status.get_code() != holoinfer_code::H_SUCCESS) {
          status.display_message();
          return InferStatus(holoinfer_code::H_ERROR,
                             "Process manager, Error running operation " + operation);
        }
        if (processed_dims.size() != 0) {
          // Check for in_out_tensor_map for a valid entry
          // populated only once assuming processed data size is not changing.
          // key in proessed_dims_map_ is same as the input tensors, mapped to dimensions of each
          // mapped tensor against it in the in_out_tensor_map.
          // While transmitting, appropriate dimensions for tensors are extracted from
          // processed_dims_map_ after mapping it from in_out_tensor_map
          if (in_out_tensor_map.find(tensor_name) != in_out_tensor_map.end()) {
            processed_dims_map_.insert({tensor_name, {std::move(processed_dims)}});
          }
        }
      }
    }
  }
  return InferStatus();
}

DataMap ManagerProcessor::get_processed_data() const {
  return processed_data_map_;
}

DimType ManagerProcessor::get_processed_data_dims() const {
  return processed_dims_map_;
}

ProcessorContext::ProcessorContext() {
  try {
    process_manager_ = std::make_shared<ManagerProcessor>();
  } catch (const std::bad_alloc&) {
    HOLOSCAN_LOG_ERROR("Holoscan Outdata context: Memory allocation error.");
    throw;
  }
}

DimType ProcessorContext::get_processed_data_dims() const {
  return process_manager_->get_processed_data_dims();
}

DataMap ProcessorContext::get_processed_data() const {
  return process_manager_->get_processed_data();
}

InferStatus ProcessorContext::process(const MultiMappings& tensor_to_oper_map,
                                      const MultiMappings& in_out_tensor_map,
                                      DataMap& inferred_result_map,
                                      const std::map<std::string, std::vector<int>>& model_dims,
                                      bool process_with_cuda,
                                      cudaStream_t cuda_stream) {
  return process_manager_->process(tensor_to_oper_map,
                                  in_out_tensor_map,
                                  inferred_result_map,
                                  model_dims,
                                  process_with_cuda,
                                  cuda_stream);
}

InferStatus ProcessorContext::initialize(const MultiMappings& process_operations,
                                         const std::string config_path = {}) {
  return process_manager_->initialize(process_operations, config_path);
}

}  // namespace inference
}  // namespace holoscan
