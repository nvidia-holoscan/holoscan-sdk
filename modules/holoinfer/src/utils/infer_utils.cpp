/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// need to add nolint flag because this file does not have an associated .h or .hpp file
#include <algorithm>   // NOLINT(build/include_order)
#include <filesystem>  // NOLINT(build/include_order)
#include <functional>  // NOLINT(build/include_order)
#include <map>         // NOLINT(build/include_order)
#include <queue>       // NOLINT(build/include_order)
#include <set>         // NOLINT(build/include_order)
#include <string>      // NOLINT(build/include_order)
#include <utility>     // NOLINT(build/include_order)
#include <vector>      // NOLINT(build/include_order)

#include <yaml-cpp/yaml.h>      // NOLINT(build/include_order)
#include <holoinfer_utils.hpp>  // NOLINT(build/include_order)

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

int report_error(const std::string& module, const std::string& submodule) {
  std::string error_string{"Error in " + module + ", Sub-module->" + submodule};
  HOLOSCAN_LOG_ERROR("{}", error_string);
  return 1;  // GXF_FAILURE
}

void raise_error(const std::string& module, const std::string& message) {
  std::string error_string{"Error in " + module + ", Sub-module->" + message};
  throw std::runtime_error(error_string);
}

void timer_init(TimePoint& _t) {
  _t = std::chrono::steady_clock::now();
}

int64_t timer_check(TimePoint& start, TimePoint& end, const std::string& module) {
  int64_t delta = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  HOLOSCAN_LOG_DEBUG("{} : {} microseconds", module.c_str(), delta);
  return delta;
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

void log_tensor_dimension(const std::vector<int>& tensor_dim,
                          const std::vector<int64_t>& input_model_dim,
                          const std::string& current_tensor) {
  HOLOSCAN_LOG_INFO("Input tensor {} dimensions: {}, Model input dimensions: {}",
                    current_tensor,
                    tensor_dim,
                    input_model_dim);
}

InferStatus tensor_dimension_check(const MultiMappings& pre_processor_map,
                                   const DimType& model_input_dimensions,
                                   const std::map<std::string, std::vector<int>>& dims_per_tensor,
                                   const std::vector<std::string>& all_input_tensors) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);

  for (const auto& model_dms : model_input_dimensions) {
    auto input_tensors = pre_processor_map.at(model_dms.first);
    auto input_dimensions = model_dms.second;

    for (size_t i = 0; i < input_tensors.size(); ++i) {
      auto current_tensor = input_tensors[i];
      if (std::find(all_input_tensors.begin(), all_input_tensors.end(), current_tensor) !=
          all_input_tensors.end()) {
        if (dims_per_tensor.find(current_tensor) == dims_per_tensor.end()) {
          auto status_msg =
              fmt::format("Input tensor {} not found in tensor dimension check for model {}.",
                          current_tensor,
                          model_dms.first);
          HOLOSCAN_LOG_ERROR(status_msg);
          status.set_message(status_msg);

          return status;
        }
        auto tensor_dim = dims_per_tensor.at(current_tensor);
        auto input_model_dim = input_dimensions[i];
        size_t batch_flag = 0;

        std::sort(tensor_dim.begin(), tensor_dim.end());
        std::sort(input_model_dim.begin(), input_model_dim.end());

        if (tensor_dim.size() != input_model_dim.size()) {
          // only case when input tensor and model input dimensions can be different is when the
          // model input is taking first dimension as a single batch and input tensor is ignoring
          // the batch dimension
          if (!(input_model_dim.size() - tensor_dim.size() == 1 && input_model_dim[0] == 1)) {
            log_tensor_dimension(tensor_dim, input_model_dim, current_tensor);
            auto status_msg =
                fmt::format("Input tensor {} has rank: {}, Model expects the rank to be {}.",
                            current_tensor,
                            tensor_dim.size(),
                            input_model_dim.size());
            HOLOSCAN_LOG_ERROR(status_msg);
            status.set_message(status_msg);
            return status;
          }
          batch_flag = 1;
        }

        for (size_t j = 0; j < tensor_dim.size(); ++j) {
          if (input_model_dim[j + batch_flag] > 0) {
            if (tensor_dim[j] != input_model_dim[j + batch_flag]) {
              log_tensor_dimension(tensor_dim, input_model_dim, current_tensor);
              auto status_msg = fmt::format(
                  "Input tensor {} dimension mismatch: Input tensor has value {}. Model expects it "
                  "to be {}.",
                  current_tensor,
                  tensor_dim[j],
                  input_model_dim[j + batch_flag]);
              HOLOSCAN_LOG_ERROR(status_msg);
              status.set_message(status_msg);
              return status;
            }
          }
        }
      }
    }
  }
  return InferStatus();
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
        auto status_msg = fmt::format("{}: empty vector for {}", type_of_map, map_data.first);
        HOLOSCAN_LOG_ERROR(status_msg);
        status.set_message(status_msg);
        return status;
      } else {
        if (map_data.first.empty()) {
          auto status_msg = fmt::format("Empty entry for key in {}", type_of_map);
          HOLOSCAN_LOG_ERROR(status_msg);
          status.set_message(status_msg);
          return status;
        }
        for (const auto& tensor_name : map_data.second) {
          if (tensor_name.empty()) {
            auto status_msg = fmt::format("Empty entry in the vector for key {}", map_data.first);
            HOLOSCAN_LOG_ERROR(status_msg);
            status.set_message(status_msg);
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
    auto status_msg = fmt::format("{} is empty", type_of_map);
    HOLOSCAN_LOG_ERROR(status_msg);
    status.set_message(status_msg);
    return status;
  } else {
    for (const auto& map_data : input_map) {
      if (map_data.first.empty() || map_data.second.empty()) {
        auto status_msg = fmt::format("Empty entry for key or value in {}", type_of_map);
        HOLOSCAN_LOG_ERROR(status_msg);
        status.set_message(status_msg);
        return status;
      }
    }
  }
  return InferStatus();
}

InferStatus setup_inference_io(const MultiMappings& pre_map, const MultiMappings& inf_map,
                               std::vector<std::string>& model_inputs,
                               std::vector<std::string>& model_outputs,
                               std::vector<std::string>& transmit_outputs,
                               const std::vector<std::string>& out_tensor_names) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);

  if (pre_map.empty()) {
    auto status_msg = fmt::format("Setup inference I/O, Pre-processor map is empty.");
    HOLOSCAN_LOG_ERROR(status_msg);
    status.set_message(status_msg);
    return status;
  }

  if (inf_map.empty()) {
    auto status_msg = fmt::format("Setup inference I/O, Inference map is empty.");
    HOLOSCAN_LOG_ERROR(status_msg);
    status.set_message(status_msg);
    return status;
  }

  // Derive external inputs/outputs
  std::set<std::string> produced;
  std::set<std::string> consumed;

  for (const auto& [_, outs] : inf_map) {
    produced.insert(outs.begin(), outs.end());
  }
  for (const auto& [_, ins] : pre_map) {
    consumed.insert(ins.begin(), ins.end());
  }
  for (const auto& t : consumed) {
    if (produced.find(t) == produced.end()) {
      model_inputs.push_back(t);
    }
  }

  // automatically compute tensors to be transmitted
  for (const auto& t : produced) {
    if (consumed.find(t) == consumed.end()) {
      transmit_outputs.push_back(t);
    }
    model_outputs.push_back(t);
  }

  // Ensure if user provided output tensors are present, merge them with transmit tensors
  if (!out_tensor_names.empty()) {
    for (const auto& t : out_tensor_names) {
      if (std::find(transmit_outputs.begin(), transmit_outputs.end(), t) ==
          transmit_outputs.end()) {
        if (std::find(model_outputs.begin(), model_outputs.end(), t) != model_outputs.end()) {
          transmit_outputs.push_back(t);
        } else {
          auto status_msg = fmt::format("Output tensor {} not found in model_outputs.", t);
          HOLOSCAN_LOG_ERROR(status_msg);
          status.set_message(status_msg);
          return status;
        }
      }
    }
  }

  return InferStatus();
}

InferStatus validate_dependency_map(const MultiMappings& pre_processor_map,
                                    const MultiMappings& dependency_map) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);

  // verify dependency map and construct graph to check for cycles
  if (!dependency_map.empty()) {
    for (const auto& [model_name, deps] : dependency_map) {
      if (pre_processor_map.find(model_name) == pre_processor_map.end()) {
        auto status_msg = fmt::format(
            "Model keyword: {} in dependency_map not found in pre_processor_map", model_name);
        HOLOSCAN_LOG_ERROR(status_msg);
        status.set_message(status_msg);
        return status;
      }
      for (const auto& dep_model : deps) {
        if (pre_processor_map.find(dep_model) == pre_processor_map.end()) {
          auto status_msg = fmt::format(
              "Dependency: {} for model: {} not found in pre_processor_map", dep_model, model_name);
          HOLOSCAN_LOG_ERROR(status_msg);
          status.set_message(status_msg);
          return status;
        }
        if (dep_model == model_name) {
          auto status_msg = fmt::format("Self dependency detected for model: {}", model_name);
          HOLOSCAN_LOG_ERROR(status_msg);
          status.set_message(status_msg);
          return status;
        }
      }
    }

    std::map<std::string, int> indegree;
    std::map<std::string, std::vector<std::string>> adj;
    for (const auto& [model_name, _] : pre_processor_map) {
      indegree[model_name] = 0;
    }
    for (const auto& [model_name, deps] : dependency_map) {
      for (const auto& dep_model : deps) {
        adj[dep_model].push_back(model_name);
        indegree[model_name]++;
      }
    }

    std::queue<std::string> ready;
    for (const auto& [model_name, degree] : indegree) {
      if (degree == 0)
        ready.push(model_name);
    }

    size_t visited = 0;
    while (!ready.empty()) {
      std::string current = std::move(ready.front());
      ready.pop();
      visited++;
      for (const auto& neighbor : adj[current]) {
        indegree[neighbor]--;
        if (indegree[neighbor] == 0)
          ready.push(neighbor);
      }
    }

    if (visited != pre_processor_map.size()) {
      auto status_msg = fmt::format("Cyclic dependency detected in dependency_map");
      HOLOSCAN_LOG_ERROR(status_msg);
      status.set_message(status_msg);
      return status;
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
    auto status_msg = fmt::format("Model path map is empty");
    HOLOSCAN_LOG_ERROR(status_msg);
    status.set_message(status_msg);
    return status;
  } else {  // Check for valid model path file
    for (const auto& model_map : model_path_map) {
      if (model_map.first.empty()) {
        auto status_msg = fmt::format("Empty key entry in model_path_map");
        HOLOSCAN_LOG_ERROR(status_msg);
        status.set_message(status_msg);
        return status;
      }
      if (!std::filesystem::exists(model_map.second)) {
        auto status_msg =
            fmt::format("Invalid file path: {} for model: {}", model_map.second, model_map.first);
        HOLOSCAN_LOG_ERROR(status_msg);
        status.set_message(status_msg);
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
  if (l_status.get_code() == holoinfer_code::H_ERROR) {
    return l_status;
  }

  if (!check_equality(model_path_map.size(), pre_processor_map.size(), inference_map.size())) {
    auto status_msg = fmt::format(
        "Size mismatch. model_path_map, pre_processor_map, "
        "inference_map, in_tensor_name, out_tensor_names must be of same size.");
    HOLOSCAN_LOG_ERROR(status_msg);
    status.set_message(status_msg);
    return status;
  }

  std::vector<std::string> input_tensors, output_tensors;

  for (const auto& model_path : model_path_map) {
    // Check if keys in model_path_map and pre_processor_map are identical
    if (pre_processor_map.find(model_path.first) == pre_processor_map.end()) {
      auto status_msg = fmt::format("Model keyword: {} not in pre_processor_map", model_path.first);
      HOLOSCAN_LOG_ERROR(status_msg);
      status.set_message(status_msg);
      return status;
    }

    if (inference_map.find(model_path.first) == inference_map.end()) {
      auto status_msg = fmt::format("Model keyword: {} not in inference_map", model_path.first);
      HOLOSCAN_LOG_ERROR(status_msg);
      status.set_message(status_msg);
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
        auto status_msg = fmt::format("Duplicate tensor name: {} for key {} in pre_processor map",
                                      tensor_name,
                                      infer_key.first);
        HOLOSCAN_LOG_ERROR(status_msg);
        status.set_message(status_msg);
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
        auto status_msg = fmt::format("Tensor name: {} absent in pre_processor map", in_tensor);
        HOLOSCAN_LOG_ERROR(status_msg);
        status.set_message(status_msg);
        return status;
      }

      // check each value in in_tensor_map is unique
      if (test_unique_input_map.find(in_tensor) == test_unique_input_map.end()) {
        test_unique_input_map.insert({in_tensor, 1});
      } else {
        auto status_msg = fmt::format("Duplicate entry for: {} in input tensor names", in_tensor);
        HOLOSCAN_LOG_ERROR(status_msg);
        status.set_message(status_msg);
        return status;
      }
    }
  }

  for (const auto& infer_key : inference_map) {
    for (const auto& tensor_name : infer_key.second) {
      // check for duplicate in tensors in inference map
      if (std::find(output_tensors.begin(), output_tensors.end(), tensor_name) !=
          output_tensors.end()) {
        auto status_msg = fmt::format(
            "Duplicate tensor name: {} for key {} in inference_map", tensor_name, infer_key.first);
        HOLOSCAN_LOG_ERROR(status_msg);
        status.set_message(status_msg);
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
        auto status_msg = fmt::format("Tensor name: {} absent in inference map", out_tensor);
        HOLOSCAN_LOG_ERROR(status_msg);
        status.set_message(status_msg);
        return status;
      }

      // check each value in out_tensor_names is unique
      if (test_unique_output_map.find(out_tensor) == test_unique_output_map.end()) {
        test_unique_output_map.insert({out_tensor, 1});
      } else {
        auto status_msg = fmt::format("Duplicate entry for: {} in output tensor names", out_tensor);
        HOLOSCAN_LOG_ERROR(status_msg);
        status.set_message(status_msg);
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
    auto status_msg = fmt::format("Input tensor names cannot be empty");
    HOLOSCAN_LOG_ERROR(status_msg);
    status.set_message(status_msg);
    return status;
  }

  if (out_tensor_names.empty()) {
    status.set_message("WARNING: Output tensor names empty");
    // derive out_tensor_names from processed_map.
    // if processed map is absent, then its dynamic I/O or its print operation.
  } else {
    auto l_status = check_multi_mappings_size_value(processed_map, "processed_map");
    if (l_status.get_code() == holoinfer_code::H_ERROR) {
      return l_status;
    }

    std::vector<std::string> output_tensors;
    for (const auto& p_map : processed_map) {
      for (const auto& tensor_name : p_map.second) {
        // check for duplicate tensors in processed_map
        if (std::find(output_tensors.begin(), output_tensors.end(), tensor_name) !=
            output_tensors.end()) {
          auto status_msg = fmt::format(
              "Duplicate tensor name: {} for key {} in processed_map.", tensor_name, p_map.first);
          HOLOSCAN_LOG_ERROR(status_msg);
          status.set_message(status_msg);
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
        auto status_msg = fmt::format("Tensor name: {} absent in processed_map.", out_tensor);
        HOLOSCAN_LOG_ERROR(status_msg);
        status.set_message(status_msg);
        return status;
      }

      // check each value in out_tensor_names is unique
      if (test_unique_output_map.find(out_tensor) == test_unique_output_map.end()) {
        test_unique_output_map.insert({out_tensor, 1});
      } else {
        auto status_msg = fmt::format("Duplicate entry for: {} in output tensor names", out_tensor);
        HOLOSCAN_LOG_ERROR(status_msg);
        status.set_message(status_msg);
        return status;
      }
    }
  }
  return InferStatus();
}

void string_split(const std::string& line, std::vector<std::string>& tokens, char c) {
  std::string token;
  std::istringstream tokenStream(line);
  while (std::getline(tokenStream, token, c)) {
    tokens.push_back(token);
  }
}

InferStatus parse_yaml_node(const YAML::Node& in_config, std::vector<std::string>& names,
                            std::vector<std::vector<int64_t>>& dims,
                            std::vector<std::string>& types) {
  // Iterate over YAML::Node directly to preserve insertion order
  for (auto it = in_config.begin(); it != in_config.end(); ++it) {
    std::string key = it->first.as<std::string>();
    auto properties_yaml = it->second;

    if (key.length() == 0) {
      HOLOSCAN_LOG_ERROR("Key cannot be an empty string");
      return InferStatus(holoinfer_code::H_ERROR, "Error in yaml node parsing.");
    }
    names.push_back(key);

    if (properties_yaml["dim"]) {
      std::vector<std::string> tokens;
      auto value = properties_yaml["dim"].as<std::string>();

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
          dims.push_back(std::move(dim));
        }
      } else {
        HOLOSCAN_LOG_ERROR("Dimensions cannot be empty for {}", key);
        return InferStatus(holoinfer_code::H_ERROR, "Error in yaml node parsing.");
      }
    } else {
      // this is placeholder dimension, will be later populated after inference.
      dims.push_back({0});
    }

    if (properties_yaml["dtype"]) {
      auto value = properties_yaml["dtype"].as<std::string>();
      if (kHoloInferDataTypeMap.find(value) != kHoloInferDataTypeMap.end()) {
        types.push_back(std::move(value));
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

InferStatus load_yaml(const std::string& yaml_file,
                      std::map<std::string, std::string>& model_path_map,
                      std::map<std::string, std::vector<std::string>>& pre_processor_map,
                      std::map<std::string, std::vector<std::string>>& inference_map,
                      std::map<std::string, std::vector<std::string>>& batch_sizes,
                      std::vector<std::string>& in_tensor_names,
                      std::vector<std::string>& out_tensor_names,
                      std::map<std::string, size_t>& tensor_to_buffersize,
                      std::map<std::string, holoinfer_datatype>& tensor_to_datatype) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);

  YAML::Node config = YAML::LoadFile(yaml_file);

  if (!config["inference_parameters"]) {
    HOLOSCAN_LOG_ERROR("Config error: inference_parameters key not present in input {} file.",
                       yaml_file);
    return status;
  }
  auto configuration = config["inference_parameters"].as<node_type>();

  if (configuration.find("model_path_map") == configuration.end()) {
    HOLOSCAN_LOG_ERROR("Config error: model_path_map key not present in input {} file.", yaml_file);
    return status;
  }

  auto model_map = configuration["model_path_map"];
  if (model_map.size() != 1) {
    HOLOSCAN_LOG_ERROR("Config error: only 1 model supported. Input: {}.", model_map.size());
    return status;
  }

  for (auto it = model_map.begin(); it != model_map.end(); ++it) {
    std::string key = it->first;
    std::string value = it->second;
    model_path_map[key] = std::move(value);
    batch_sizes[key] = {"1,1,1"};
  }

  if (configuration.find("pre_processor_map") == configuration.end()) {
    HOLOSCAN_LOG_ERROR("Config error: pre_processor_map key not present in input {} file.",
                       yaml_file);
    return status;
  }
  auto input_map = configuration["pre_processor_map"];

  if (input_map.size() != 1) {
    HOLOSCAN_LOG_ERROR("Config error: only 1 input per model supported. Inputs provided: {}.",
                       input_map.size());
    return status;
  }

  for (auto it = input_map.begin(); it != input_map.end(); ++it) {
    std::string key = it->first;
    std::string value = it->second;
    pre_processor_map[key].push_back(value);
    in_tensor_names.push_back(value);
    HOLOSCAN_LOG_INFO("Inserted Key/Value: {}, {}", key, value);
  }

  if (configuration.find("inference_map") == configuration.end()) {
    HOLOSCAN_LOG_ERROR("Config error: inference_map key not present in input {} file.", yaml_file);
    return status;
  }
  auto output_map = configuration["inference_map"];
  if (output_map.size() != 1) {
    HOLOSCAN_LOG_ERROR("Config error: only 1 output per model supported. Outputs provided: {}.",
                       output_map.size());
    return status;
  }

  for (auto it = output_map.begin(); it != output_map.end(); ++it) {
    std::string key = it->first;
    std::string value = it->second;
    inference_map[key].push_back(value);
    out_tensor_names.push_back(value);
  }

  if (!config["tensors"]) {
    HOLOSCAN_LOG_ERROR("Config error: tensors key not present in input {} file.", yaml_file);
    return status;
  }
  auto tensors = config["tensors"].as<node_type>();

  auto all_tensors = {in_tensor_names[0], out_tensor_names[0]};

  for (auto current_tensor : all_tensors) {
    if (tensors.find(current_tensor) == tensors.end()) {
      HOLOSCAN_LOG_ERROR("Config error: Details of tensor {} not present in input {} file.",
                         current_tensor,
                         yaml_file);
      return status;
    }
    auto tensor_info = tensors[current_tensor];
    if (tensor_info.find("dim") == tensor_info.end()) {
      HOLOSCAN_LOG_ERROR("Config error: dim key not present for tensor {}.", current_tensor);
      return status;
    }
    if (tensor_info.find("dtype") == tensor_info.end()) {
      HOLOSCAN_LOG_ERROR("Config error: dtype key not present for tensor {}.", current_tensor);
      return status;
    }
    std::string dims_string = tensor_info["dim"];
    std::string dtype = tensor_info["dtype"];

    if (kHoloInferDataTypeMap.find(dtype) == kHoloInferDataTypeMap.end()) {
      HOLOSCAN_LOG_ERROR(
          "Config error: dtype {} for tensor {}, not supported.", dtype, current_tensor);
      return status;
    }

    auto holoinfer_dtype = kHoloInferDataTypeMap.at(dtype);

    std::vector<std::string> dimensions;
    string_split(dims_string, dimensions, ',');

    if (dimensions.size() == 0) {
      HOLOSCAN_LOG_ERROR("Config error: dimension size is 0 for tensor {}.", current_tensor);
      return status;
    }
    std::vector<size_t> dimensions_int;
    for (auto& dim : dimensions) {
      const auto v = std::stoll(dim);
      if (v <= 0) {
        HOLOSCAN_LOG_ERROR(
            "Config error: dimension {} must be > 0 for tensor {}.", v, current_tensor);
        return status;
      }
      dimensions_int.push_back(static_cast<size_t>(v));
    }

    auto tensor_size =
        accumulate(dimensions_int.begin(), dimensions_int.end(), 1, std::multiplies<size_t>());

    tensor_to_buffersize.insert({current_tensor, tensor_size});
    tensor_to_datatype.insert({current_tensor, holoinfer_dtype});

    HOLOSCAN_LOG_DEBUG(
        "Inserted tensor {} with size {} and datatype {}", current_tensor, tensor_size, dtype);
  }
  HOLOSCAN_LOG_INFO("YAML configuration loaded");
  return InferStatus();
}

InferStatus build_execution_plan(const MultiMappings& pre_map, const MultiMappings& inf_map,
                                 std::vector<std::vector<std::string>>& execution_plan) {
  InferStatus status;

  if (pre_map.empty()) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message("Building execution plan, Pre-processor map is empty.");
    return status;
  }

  if (inf_map.empty()) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message("Building execution plan, Inference map is empty.");
    return status;
  }

  std::map<std::string, std::vector<std::string>> dep_map;

  // Build producer lookup
  std::map<std::string, std::string> producer_of;
  for (const auto& [model, outs] : inf_map) {
    for (const auto& tensor : outs) {
      producer_of[tensor] = model;
    }
  }

  // Derive dependency map
  for (const auto& [model, inputs] : pre_map) {
    std::set<std::string> deps;
    for (const auto& tensor : inputs) {
      auto it = producer_of.find(tensor);
      if (it != producer_of.end() && it->second != model) {
        deps.insert(it->second);
      }
    }
    if (!deps.empty()) {
      dep_map[model] = std::vector<std::string>(deps.begin(), deps.end());
    }
  }

  status = validate_dependency_map(pre_map, dep_map);
  if (status.get_code() != holoinfer_code::H_SUCCESS) {
    return status;
  }

  if (dep_map.empty()) {
    std::vector<std::string> level;
    level.reserve(pre_map.size());
    for (const auto& [model_instance, _] : pre_map) {
      level.push_back(model_instance);
    }
    execution_plan.push_back(std::move(level));
    return status;
  }

  std::map<std::string, int> indegree;
  std::map<std::string, std::vector<std::string>> adjacency;

  for (const auto& [model_instance, _] : pre_map) {
    indegree[model_instance] = 0;
  }

  for (const auto& [model_instance, deps] : dep_map) {
    if (indegree.find(model_instance) == indegree.end()) {
      status.set_code(holoinfer_code::H_ERROR);
      status.set_message("Building execution plan, Model " + model_instance +
                         " present in dep_map but not in inference parameters.");
      return status;
    }
    for (const auto& dep_model : deps) {
      if (indegree.find(dep_model) == indegree.end()) {
        status.set_code(holoinfer_code::H_ERROR);
        status.set_message("Building execution plan, Dependency " + dep_model + " for model " +
                           model_instance + " not found in inference parameters.");
        return status;
      }
      adjacency[dep_model].push_back(model_instance);
      indegree[model_instance]++;
    }
  }

  std::queue<std::string> ready;
  for (const auto& [model_instance, degree] : indegree) {
    if (degree == 0) {
      ready.push(model_instance);
    }
  }

  size_t visited = 0;
  while (!ready.empty()) {
    const auto level_size = ready.size();
    std::vector<std::string> level;
    level.reserve(level_size);

    for (size_t idx = 0; idx < level_size; ++idx) {
      std::string current_model = std::move(ready.front());
      ready.pop();
      visited++;

      for (const auto& dependent : adjacency[current_model]) {
        indegree[dependent]--;
        if (indegree[dependent] == 0) {
          ready.push(dependent);
        }
      }
      level.push_back(std::move(current_model));
    }
    execution_plan.push_back(std::move(level));
  }

  HOLOSCAN_LOG_DEBUG("Execution plan: ");
  for (const auto& level : execution_plan) {
    HOLOSCAN_LOG_DEBUG("Level: {}", level.size());
    std::stringstream ss;  // for logging the level
    for (const auto& model : level) {
      ss << model << " ";
    }
    HOLOSCAN_LOG_DEBUG("Models: {}", ss.str());
  }

  if (visited != pre_map.size()) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message("Building execution plan, Cyclic dependency detected in dependency_map.");
  }

  return status;
}

}  // namespace inference
}  // namespace holoscan
