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
#include "process_manager.hpp"

namespace holoscan {
namespace inference {

InferStatus ManagerProcessor::initialize(const MultiMappings& process_operations) {
  try {
    infer_data_ = std::make_unique<DataProcessor>();
  } catch (const std::bad_alloc&) {
    return InferStatus(holoinfer_code::H_ERROR,
                       "Process Manager, Holoscan out data core: Memory allocation error");
  }
  return infer_data_->initialize(process_operations);
}

InferStatus ManagerProcessor::process(
    const MultiMappings& tensor_oper_map, const Mappings& in_out_tensor_map,
    DataMap& inferred_result_map, const std::map<std::string, std::vector<int>>& dimension_map) {
  for (const auto& tensor_to_ops : tensor_oper_map) {
    auto tensor_name = tensor_to_ops.first;
    if (inferred_result_map.find(tensor_name) == inferred_result_map.end()) {
      return InferStatus(
          holoinfer_code::H_ERROR,
          "Process manager, Inference map does not contain results from " + tensor_name);
    }

    if (dimension_map.find(tensor_name) == dimension_map.end()) {
      return InferStatus(
          holoinfer_code::H_ERROR,
          "Process manager, Dimension map does not contain results from " + tensor_name);
    }

    std::vector<float> out_result = inferred_result_map.at(tensor_name)->host_buffer;
    const std::vector<int> dimensions = dimension_map.at(tensor_name);

    auto operations = tensor_to_ops.second;

    // If the input tensor has a mapped output tensor, then allocate memory
    if (in_out_tensor_map.find(tensor_name) != in_out_tensor_map.end()) {
      auto out_tensor_name = in_out_tensor_map.at(tensor_name);
      if (processed_data_map_.find(out_tensor_name) == processed_data_map_.end()) {
        processed_data_map_.insert({out_tensor_name, std::make_shared<DataBuffer>()});
      }
    }

    processed_dims_map_.clear();

    for (auto& operation : operations) {
      std::vector<int64_t> processed_dims;
      std::vector<float> process_vector;
      std::vector<std::string> custom_strings;

      // if operation is print, then no need to allocate output memory
      if (operation.find("print") != std::string::npos) {
        if (operation.compare("print") == 0) {
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
            operation = custom_strings.at(0);
            custom_strings.erase(custom_strings.begin());
          }
        }
      } else {
        if (in_out_tensor_map.find(tensor_name) == in_out_tensor_map.end()) {
          return InferStatus(
              holoinfer_code::H_ERROR,
              "Process manager, In tensor " + tensor_name + " has no out tensor mapping");
        }
        // allocate memory for the first time
        // TODO: every consecutive operation update the allocated buffer :: SD
        if (process_vector.size() == 0) {
          const size_t out_channels = dimensions[dimensions.size() - 1];
          process_vector.resize(2 * out_channels);
        }
      }

      InferStatus status = infer_data_->process_operation(
          operation, dimensions, out_result, processed_dims, process_vector, custom_strings);

      if (status.get_code() != holoinfer_code::H_SUCCESS) {
        return InferStatus(holoinfer_code::H_ERROR,
                           "Process manager, Error running operation " + operation);
      }
      // TODO: Update the input to process_operation with updated results if many operations in
      // sequence: SD
      // TODO: Generalize dims map to have mapping based on out_tensors

      if (in_out_tensor_map.find(tensor_name) != in_out_tensor_map.end()) {
        auto out_tensor_name = in_out_tensor_map.at(tensor_name);

        processed_data_map_.at(out_tensor_name)->host_buffer = std::move(process_vector);
        processed_dims_map_.insert({tensor_name, std::move(processed_dims)});
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
    process_manager = std::make_unique<ManagerProcessor>();
  } catch (const std::bad_alloc&) {
    HOLOSCAN_LOG_ERROR("Holoscan Outdata context: Memory allocation error.");
    throw;
  }
}

DimType ProcessorContext::get_processed_data_dims() const {
  return process_manager->get_processed_data_dims();
}

DataMap ProcessorContext::get_processed_data() const {
  return process_manager->get_processed_data();
}

InferStatus ProcessorContext::process(const MultiMappings& tensor_to_oper_map,
                                      const Mappings& in_out_tensor_map,
                                      DataMap& inferred_result_map,
                                      const std::map<std::string, std::vector<int>>& model_dims) {
  return process_manager->process(
      tensor_to_oper_map, in_out_tensor_map, inferred_result_map, model_dims);
}

InferStatus ProcessorContext::initialize(const MultiMappings& process_operations) {
  return process_manager->initialize(process_operations);
}

}  // namespace inference
}  // namespace holoscan
