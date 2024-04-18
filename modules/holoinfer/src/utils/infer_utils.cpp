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

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <holoinfer_utils.hpp>

namespace holoscan {
namespace inference {

cudaError_t check_cuda(cudaError_t result) {
  if (result != cudaSuccess) {
    HOLOSCAN_LOG_ERROR("Cuda runtime error, {}", cudaGetErrorString(result));
    std::stringstream error_string;
    error_string << "Cuda runtime error: " << cudaGetErrorName(result) << ", "
                 << cudaGetErrorString(result);
    throw std::runtime_error(error_string.str());
  }
  return result;
}

gxf_result_t report_error(const std::string& module, const std::string& submodule) {
  std::string error_string{"Error in " + module + ", Sub-module->" + submodule};
  HOLOSCAN_LOG_ERROR("{}", error_string);
  return GXF_FAILURE;
}

void raise_error(const std::string& module, const std::string& message) {
  std::string error_string{"Error in " + module + ", Sub-module->" + message};
  throw std::runtime_error(error_string);
}

void timer_init(TimePoint& _t) {
  _t = std::chrono::steady_clock::now();
}

gxf_result_t timer_check(TimePoint& start, TimePoint& end, const std::string& module) {
  timer_init(end);
  int64_t delta = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  HOLOSCAN_LOG_DEBUG("{} : {} microseconds", module.c_str(), delta);
  return GXF_SUCCESS;
}

bool is_platform_aarch64() {
  struct utsname buffer;

  if (uname(&buffer) == 0) {
    std::string machine(buffer.machine);

    if (machine.find("arm") != std::string::npos || machine.find("aarch64") != std::string::npos) {
      return true;
    }
  }
  // Return false in all other conditions.
  return false;
}

/// @brief Test equality of 2 input parameters
/// @param first First input to be compared
/// @param second Second input to be compared
/// @return True if parameters are equal else false
template <typename T>
bool check_equality(const T& first, const T& second) {
  return first == second;
}

/// @brief Test equality of a sequence of parameters
/// @param first First input to be compared
/// @param second Second input to be compared
/// @param args Sequence of inputs
/// @return True if all input parameters are equal else false
template <typename T, typename... Y>
bool check_equality(const T& first, const T& second, const Y&... args) {
  return (first == second) && check_equality(second, args...);
}

InferStatus check_multi_mappings_size_value(const MultiMappings& input_map,
                                            const std::string& type_of_map) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);

  if (input_map.empty()) {
    status.set_message(type_of_map + " is empty");
    return status;
  } else {
    for (const auto& map_data : input_map) {
      if (map_data.second.size() == 0) {
        status.set_message(type_of_map + ": empty vector for " + map_data.first);
        return status;
      } else {
        if (map_data.first.empty()) {
          status.set_message("Empty entry for key in " + type_of_map);
          return status;
        }
        for (const auto& tensor_name : map_data.second) {
          if (tensor_name.empty()) {
            status.set_message("Empty entry in the vector for key " + map_data.first);
            return status;
          }
        }
      }
    }
  }

  return InferStatus();
}

InferStatus check_mappings_size_value(const MultiMappings& input_map,
                                      const std::string& type_of_map) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);

  if (input_map.empty()) {
    status.set_message(type_of_map + " is empty");
    return status;
  } else {
    for (const auto& map_data : input_map) {
      if (map_data.first.empty() || map_data.second.empty()) {
        status.set_message("Empty entry for key or value in " + type_of_map);
        return status;
      }
    }
  }
  return InferStatus();
}

InferStatus inference_validity_check(const Mappings& model_path_map,
                                     const MultiMappings& pre_processor_map,
                                     const MultiMappings& inference_map,
                                     std::vector<std::string>& in_tensor_names,
                                     std::vector<std::string>& out_tensor_names) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);

  // check for model path map size
  if (model_path_map.empty()) {
    status.set_message("Model path map is empty");
    return status;
  } else {  // Check for valid model path file
    for (const auto& model_map : model_path_map) {
      if (model_map.first.empty()) {
        status.set_message("Empty key entry in model_path_map");
        return status;
      }
      if (!std::filesystem::exists(model_map.second)) {
        status.set_message("Invalid file path: " + model_map.second +
                           " for model: " + model_map.first);
        return status;
      }
    }
  }

  auto l_status = check_multi_mappings_size_value(pre_processor_map, "pre_processor_map");
  if (l_status.get_code() == holoinfer_code::H_ERROR) {
    l_status.display_message();
    return l_status;
  }

  l_status = check_multi_mappings_size_value(inference_map, "inference_map");
  if (l_status.get_code() == holoinfer_code::H_ERROR) { return l_status; }

  if (!check_equality(model_path_map.size(), pre_processor_map.size(), inference_map.size())) {
    status.set_message(
        "Size mismatch. model_path_map, pre_processor_map, "
        "inference_map, in_tensor_name, out_tensor_names must be of same size.");
    return status;
  }

  std::vector<std::string> input_tensors, output_tensors;

  for (const auto& model_path : model_path_map) {
    // Check if keys in model_path_map and pre_processor_map are identical
    if (pre_processor_map.find(model_path.first) == pre_processor_map.end()) {
      status.set_message("Model keyword: " + model_path.first + " not in pre_processor_map");
      return status;
    }

    if (inference_map.find(model_path.first) == inference_map.end()) {
      status.set_message("Model keyword: " + model_path.first + " not in inference_map");
      return status;
    }
  }

  // create a vector of input tensors and verify all tensors are unique
  for (const auto& infer_key : pre_processor_map) {
    std::vector<std::string> current_input_tensors;

    for (const auto& tensor_name : infer_key.second) {
      // check for duplicate in tensors in pre_processor map
      if (std::find(current_input_tensors.begin(), current_input_tensors.end(), tensor_name) !=
          current_input_tensors.end()) {
        status.set_message("Duplicate tensor name: " + tensor_name + " for key " + infer_key.first +
                           " in pre_processor map");
        return status;
      }
      current_input_tensors.push_back(tensor_name);
    }

    for (const auto& current_tensor : current_input_tensors) {
      if (std::find(input_tensors.begin(), input_tensors.end(), current_tensor) ==
          input_tensors.end()) {
        input_tensors.push_back(current_tensor);
      }
    }
  }

  if (in_tensor_names.empty()) {
    HOLOSCAN_LOG_INFO("Input tensor names empty from Config. Creating from pre_processor map.");
    in_tensor_names = std::move(input_tensors);
    HOLOSCAN_LOG_INFO("Input Tensor names: [{}]", fmt::join(in_tensor_names, ", "));
  } else {
    std::map<std::string, int> test_unique_input_map;
    for (const auto& in_tensor : in_tensor_names) {
      // check that each value in in_tensor_names exists in pre_processor map
      if (std::find(input_tensors.begin(), input_tensors.end(), in_tensor) == input_tensors.end()) {
        status.set_message("Tensor name: " + in_tensor + " absent in pre_processor map");
        return status;
      }

      // check each value in in_tensor_map is unique
      if (test_unique_input_map.find(in_tensor) == test_unique_input_map.end()) {
        test_unique_input_map.insert({in_tensor, 1});
      } else {
        status.set_message("Duplicate entry for: " + in_tensor + " in input tensor names");
        return status;
      }
    }
  }

  for (const auto& infer_key : inference_map) {
    for (const auto& tensor_name : infer_key.second) {
      // check for duplicate in tensors in inference map
      if (std::find(output_tensors.begin(), output_tensors.end(), tensor_name) !=
          output_tensors.end()) {
        status.set_message("Duplicate tensor name: " + tensor_name + " for key " + infer_key.first +
                           " in inference_map");
        return status;
      }
      output_tensors.push_back(tensor_name);
    }
  }

  if (out_tensor_names.empty()) {
    HOLOSCAN_LOG_INFO("Output tensor names empty from Config. Creating from inference map.");
    out_tensor_names = std::move(output_tensors);
    HOLOSCAN_LOG_INFO("Output Tensor names: [{}]", fmt::join(out_tensor_names, ", "));
  } else {
    std::map<std::string, int> test_unique_output_map;
    for (const auto& out_tensor : out_tensor_names) {
      // check that each value in out_tensor_names exists in inference map
      if (std::find(output_tensors.begin(), output_tensors.end(), out_tensor) ==
          output_tensors.end()) {
        status.set_message("Tensor name: " + out_tensor + " absent in inference map");
        return status;
      }

      // check each value in out_tensor_names is unique
      if (test_unique_output_map.find(out_tensor) == test_unique_output_map.end()) {
        test_unique_output_map.insert({out_tensor, 1});
      } else {
        status.set_message("Duplicate entry for: " + out_tensor + " in output tensor names");
        return status;
      }
    }
  }

  return InferStatus();
}

InferStatus processor_validity_check(const MultiMappings& processed_map,
                                     const std::vector<std::string>& in_tensor_names,
                                     const std::vector<std::string>& out_tensor_names) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);

  if (in_tensor_names.empty()) {
    status.set_message("Input tensor names cannot be empty");
    return status;
  }

  if (out_tensor_names.empty()) {
    status.set_message("WARNING: Output tensor names empty");
    // derive out_tensor_names from processed_map.
    // if processed map is absent, then its dynamic I/O or its print operation.
  } else {
    auto l_status = check_multi_mappings_size_value(processed_map, "processed_map");
    if (l_status.get_code() == holoinfer_code::H_ERROR) { return l_status; }

    std::vector<std::string> output_tensors;
    for (const auto& p_map : processed_map) {
      for (const auto& tensor_name : p_map.second) {
        // check for duplicate tensors in processed_map
        if (std::find(output_tensors.begin(), output_tensors.end(), tensor_name) !=
            output_tensors.end()) {
          status.set_message("Duplicate tensor name: " + tensor_name + " for key " + p_map.first +
                             " in processed_map.");
          return status;
        }
        output_tensors.push_back(tensor_name);
      }
    }

    std::map<std::string, int> test_unique_output_map;
    for (const auto& out_tensor : out_tensor_names) {
      // check that each value in out_tensor_names exists in processed_map
      if (std::find(output_tensors.begin(), output_tensors.end(), out_tensor) ==
          output_tensors.end()) {
        status.set_message("Tensor name: " + out_tensor + " absent in processed_map.");
        return status;
      }

      // check each value in out_tensor_names is unique
      if (test_unique_output_map.find(out_tensor) == test_unique_output_map.end()) {
        test_unique_output_map.insert({out_tensor, 1});
      } else {
        status.set_message("Duplicate entry for: " + out_tensor + " in output tensor names");
        return status;
      }
    }
  }
  return InferStatus();
}

void string_split(const std::string& line, std::vector<std::string>& tokens, char c) {
  std::string token;
  std::istringstream tokenStream(line);
  while (std::getline(tokenStream, token, c)) { tokens.push_back(token); }
}

InferStatus parse_yaml_node(const node_type& in_config, std::vector<std::string>& names,
                            std::vector<std::vector<int64_t>>& dims,
                            std::vector<holoinfer_datatype>& types) {
  for (const auto& [key, properties] : in_config) {
    if (key.length() == 0) {
      HOLOSCAN_LOG_ERROR("Key cannot be an empty string");
      return InferStatus(holoinfer_code::H_ERROR, "Error in yaml node parsing.");
    }
    names.push_back(key);

    if (properties.find("dim") != properties.end()) {
      std::vector<std::string> tokens;
      auto value = properties.at("dim");

      if (value.length() != 0) {
        string_split(value, tokens, ' ');
        if (tokens.size() > 0) {
          std::vector<int64_t> dim;
          for (const auto& t : tokens) {
            if (std::stoi(t) <= 0) {
              HOLOSCAN_LOG_ERROR("Entry in dimension must be greater than 0. Found: {}",
                                 std::stoi(t));
              return InferStatus(holoinfer_code::H_ERROR, "Error in yaml node parsing.");
            }
            dim.push_back(std::stoi(t));
          }
          dims.push_back(dim);
        }
      } else {
        HOLOSCAN_LOG_ERROR("Dimensions cannot be empty for {}", key);
        return InferStatus(holoinfer_code::H_ERROR, "Error in yaml node parsing.");
      }
    } else {
      // this is placeholder dimension, will be later populated after inference.
      dims.push_back({0});
    }

    if (properties.find("dtype") != properties.end()) {
      auto value = properties.at("dtype");
      if (kHoloInferDataTypeMap.find(value) != kHoloInferDataTypeMap.end()) {
        types.push_back(kHoloInferDataTypeMap.at(value));
      } else {
        HOLOSCAN_LOG_ERROR("Output datatype {} not supported", value);
        return InferStatus(holoinfer_code::H_ERROR, "Error in yaml node parsing.");
      }
    } else {
      HOLOSCAN_LOG_ERROR("dtype missing for {}", key);
      return InferStatus(holoinfer_code::H_ERROR, "Error in yaml node parsing.");
    }
  }
  return InferStatus();
}

}  // namespace inference
}  // namespace holoscan
