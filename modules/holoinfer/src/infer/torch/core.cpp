/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "core.hpp"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <fmt/ranges.h>
#include <torch/script.h>

#include <yaml-cpp/yaml.h>

#include <functional>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace holoscan {
namespace inference {

// Input processor optimization
static const std::unordered_map<std::string, torch::ScalarType> kTorchTypeMap = {
    {"kFloat32", torch::kFloat32},
    {"kInt32", torch::kInt32},
    {"kInt8", torch::kInt8},
    {"kUInt8", torch::kUInt8},
    {"kInt64", torch::kInt64},
    {"kFloat16", torch::kFloat16},
    {"kBool", torch::kBool}};

// Pimpl
class TorchInferImpl {
 public:
  TorchInferImpl(const std::string& model_file_path, bool cuda_flag, bool cuda_buf_in,
                 bool cuda_buf_out);
  ~TorchInferImpl();

  std::string model_path_{""};
  size_t input_nodes_{0}, output_nodes_{0};

  std::vector<std::vector<int64_t>> input_dims_{};
  std::vector<std::vector<int64_t>> output_dims_{};
  YAML::Node output_format_{};

  std::vector<std::string> input_type_, output_type_;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;

  std::vector<torch::Tensor> output_tensors_;

  std::vector<torch::jit::IValue> inputs_;
  torch::jit::IValue outputs_;

  torch::jit::script::Module inference_module_;

  torch::DeviceType infer_device_;
  torch::DeviceType input_device_;
  torch::DeviceType output_device_;

  c10::cuda::CUDAStream infer_stream_ = c10::cuda::getStreamFromPool();
  std::unique_ptr<c10::cuda::CUDAStreamGuard> stream_guard_;
  cudaEvent_t cuda_event_ = nullptr;

  void print_model_details();

  InferStatus parse_node_details(const YAML::Node& config,
                                 const std::string& config_yaml_file_path);

  InferStatus populate_model_details();

  torch::Tensor create_tensor(const std::shared_ptr<DataBuffer>& input_buffer,
                              const std::vector<int64_t>& dims, const std::string& data_type_str,
                              InferStatus& status);

  InferStatus transfer_to_output(std::shared_ptr<DataBuffer>& output_buffer,
                                 torch::Tensor out_torch_tensor, std::vector<int64_t>& output_dim,
                                 const std::string& data_type_str);

  InferStatus torch_struct_to_map(const YAML::Node& node, const torch::IValue& torch_struct,
                                  std::unordered_map<std::string, torch::Tensor>& data_map);
  InferStatus get_io_schema(const YAML::Node& yaml_node, std::string& type_stream);

  struct InputProcessor {
    std::function<torch::IValue(const std::unordered_map<std::string, torch::Tensor>&)> processor;
    YAML::Node schema_node;
  };

  std::vector<InputProcessor> input_processors_;
  InferStatus set_input_processors(const std::vector<std::string>& input_type_structs,
                                   const std::vector<YAML::Node>& input_format);
  InferStatus create_torch_inputs(const std::unordered_map<std::string, torch::Tensor>& tensor_map,
                                  std::vector<torch::IValue>& torch_inputs);

 private:
  static inline torch::Tensor get_tensor_from_schema(
      const YAML::Node& schema, const std::unordered_map<std::string, torch::Tensor>& tensor_map) {
    try {
      return tensor_map.at(schema.as<std::string>());
    } catch (const std::out_of_range& e) {
      std::string requested_key = schema.as<std::string>();
      std::vector<std::string> available_keys;
      for (const auto& kv : tensor_map) {
        available_keys.push_back(kv.first);
      }
      HOLOSCAN_LOG_ERROR(
          "Torch core: Expected input tensor '{}' not found in tensor_map. Available tensors for "
          "this frame: {}",
          requested_key,
          fmt::join(available_keys, ", "));
      throw std::runtime_error("Torch core: Tensor key not found in tensor_map");
    }
  }

  static inline c10::List<torch::Tensor> get_list_from_schema(
      const YAML::Node& schema, const std::unordered_map<std::string, torch::Tensor>& tensor_map) {
    c10::List<torch::Tensor> list;
    for (const auto& element : schema) {
      list.push_back(get_tensor_from_schema(element, tensor_map));
    }
    return list;
  }

  static inline c10::Dict<std::string, torch::Tensor> get_dict_from_schema(
      const YAML::Node& schema, const std::unordered_map<std::string, torch::Tensor>& tensor_map) {
    c10::Dict<std::string, torch::Tensor> dict;
    for (const auto& element : schema) {
      dict.insert(element.first.as<std::string>(),
                  get_tensor_from_schema(element.second, tensor_map));
    }
    return dict;
  }

  static inline c10::Dict<std::string, c10::List<torch::Tensor>> get_dict_of_list_from_schema(
      const YAML::Node& schema, const std::unordered_map<std::string, torch::Tensor>& tensor_map) {
    c10::Dict<std::string, c10::List<torch::Tensor>> dict;
    for (const auto& element : schema) {
      dict.insert(element.first.as<std::string>(),
                  get_list_from_schema(element.second, tensor_map));
    }
    return dict;
  }

  static inline c10::Dict<std::string, c10::Dict<std::string, torch::Tensor>>
  get_dict_of_dict_from_schema(const YAML::Node& schema,
                               const std::unordered_map<std::string, torch::Tensor>& tensor_map) {
    c10::Dict<std::string, c10::Dict<std::string, torch::Tensor>> dict;
    for (const auto& element : schema) {
      dict.insert(element.first.as<std::string>(),
                  get_dict_from_schema(element.second, tensor_map));
    }
    return dict;
  }

  static inline c10::List<c10::Dict<std::string, torch::Tensor>> get_list_of_dict_from_schema(
      const YAML::Node& schema, const std::unordered_map<std::string, torch::Tensor>& tensor_map) {
    c10::List<c10::Dict<std::string, torch::Tensor>> list;
    for (const auto& element : schema) {
      list.push_back(get_dict_from_schema(element, tensor_map));
    }
    return list;
  }

  static inline c10::List<c10::List<torch::Tensor>> get_list_of_list_from_schema(
      const YAML::Node& schema, const std::unordered_map<std::string, torch::Tensor>& tensor_map) {
    c10::List<c10::List<torch::Tensor>> list;
    for (const auto& element : schema) {
      list.push_back(get_list_from_schema(element, tensor_map));
    }
    return list;
  }
};

torch::Tensor create_tensor_core(const std::shared_ptr<DataBuffer>& input_buffer,
                                 const std::vector<int64_t>& dims, torch::ScalarType data_type,
                                 torch::DeviceType infer_device, torch::DeviceType input_device,
                                 cudaStream_t cstream) {
  size_t input_tensor_size = accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());

  if (input_device == torch::kCPU) {
    if (input_buffer->host_buffer_->size() != input_tensor_size) {
      HOLOSCAN_LOG_ERROR("Torch: Input host buffer size mismatch.");
      return torch::empty({0});
    }
  } else if (input_buffer->device_buffer_->size() != input_tensor_size) {
    HOLOSCAN_LOG_ERROR("Torch: Input device buffer size mismatch.");
    return torch::empty({0});
  }

  torch::Tensor tensor;

  if (input_device == infer_device) {
    auto options = torch::TensorOptions().dtype(data_type).device(infer_device);
    if (input_device == torch::kCPU) {
      tensor = torch::from_blob(input_buffer->host_buffer_->data(), dims, options);
    } else {
      tensor = torch::from_blob(input_buffer->device_buffer_->data(), dims, options);
    }
  } else {
    // creates a new tensor on the inference device
    tensor = torch::zeros(dims, torch::TensorOptions().dtype(data_type).device(infer_device));
    auto element_size = c10::elementSize(data_type);
    if (infer_device == torch::kCPU) {
      // infer device is CPU and input device is GPU
      auto cstatus = cudaMemcpyAsync(tensor.data_ptr(),
                                     reinterpret_cast<void*>(input_buffer->device_buffer_->data()),
                                     input_tensor_size * element_size,
                                     cudaMemcpyDeviceToHost,
                                     cstream);
      if (cstatus != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Torch: DtoH transfer failed: {}", cudaGetErrorString(cstatus));
        return torch::empty({0});
      }
    } else if (infer_device == torch::kCUDA) {
      // infer device is GPU and input device is CPU
      auto cstatus = cudaMemcpyAsync(tensor.data_ptr(),
                                     reinterpret_cast<void*>(input_buffer->host_buffer_->data()),
                                     input_tensor_size * element_size,
                                     cudaMemcpyHostToDevice,
                                     cstream);
      if (cstatus != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Torch: HtoD transfer failed: {}", cudaGetErrorString(cstatus));
        return torch::empty({0});
      }
    }
  }

  return tensor;
}

torch::Tensor TorchInferImpl::create_tensor(const std::shared_ptr<DataBuffer>& input_buffer,
                                            const std::vector<int64_t>& dims,
                                            const std::string& data_type_str, InferStatus& status) {
  // Check for empty buffer first
  if (input_buffer->host_buffer_->size() == 0 && input_buffer->device_buffer_->size() == 0) {
    HOLOSCAN_LOG_ERROR("Torch core: Input buffer is empty");
    status = InferStatus(holoinfer_code::H_ERROR, "Torch core: Input buffer is empty");
    return torch::empty({0});
  }

  auto data_type = input_buffer->get_datatype();
  auto cstream = infer_stream_.stream();
  if (data_type != kHoloInferDataTypeMap.at(data_type_str)) {
    HOLOSCAN_LOG_ERROR("Torch core: Failed to create tensor. Data type mismatch, expected: {}",
                       data_type_str);
    status = InferStatus(holoinfer_code::H_ERROR, "Torch core: Data type mismatch");
    return torch::empty({0});
  }
  try {
    return create_tensor_core(
        input_buffer, dims, kTorchTypeMap.at(data_type_str), infer_device_, input_device_, cstream);
  } catch (const std::out_of_range& e) {
    std::vector<std::string> supported_types;
    for (const auto& [key, value] : kTorchTypeMap) {
      supported_types.push_back(key);
    }
    HOLOSCAN_LOG_ERROR(
        "Torch core: Unsupported data type when creating tensor: {}, supported types: {}",
        data_type_str,
        fmt::join(supported_types, ", "));
    status = InferStatus(holoinfer_code::H_ERROR, "Torch core: Unsupported data type");
    return torch::empty({0});
  } catch (const c10::Error& e) {
    HOLOSCAN_LOG_ERROR("Torch core: error in tensor creation: {}", e.what());
    status = InferStatus(holoinfer_code::H_ERROR, "Torch core: error in tensor creation");
    return torch::empty({0});
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Torch core: standard error in tensor creation: {}", e.what());
    status = InferStatus(holoinfer_code::H_ERROR, "Torch core: standard error in tensor creation");
    return torch::empty({0});
  } catch (...) {
    HOLOSCAN_LOG_ERROR("Torch core: unknown error in tensor creation");
    status = InferStatus(holoinfer_code::H_ERROR, "Torch core: unknown error in tensor creation");
    return torch::empty({0});
  }
}

InferStatus transfer_to_output_core(std::shared_ptr<DataBuffer>& output_buffer,
                                    torch::Tensor& output_tensor, std::vector<int64_t>& dims,
                                    const torch::DeviceType infer_device,
                                    const torch::DeviceType output_device,
                                    const torch::ScalarType data_type, cudaStream_t cstream) {
  auto element_size = c10::elementSize(data_type);
  size_t output_tensor_size = output_tensor.numel();
  if (output_device == torch::kCUDA) {
    output_buffer->device_buffer_->resize(output_tensor_size);
  } else {
    output_buffer->host_buffer_->resize(output_tensor_size);
  }

  // Populate dims for data transmission
  auto tensor_size = output_tensor.dim();
  dims.clear();
  for (size_t it = 0; it < tensor_size; it++) {
    int64_t s = output_tensor.sizes()[it];
    dims.push_back(s);
  }

  if (output_device == torch::kCPU) {
    if (infer_device == torch::kCPU) {
      memcpy(output_buffer->host_buffer_->data(),
             output_tensor.data_ptr(),
             output_tensor_size * element_size);
    } else {
      auto cstatus = cudaMemcpyAsync(output_buffer->host_buffer_->data(),
                                     output_tensor.data_ptr(),
                                     output_tensor_size * element_size,
                                     cudaMemcpyDeviceToHost,
                                     cstream);
      if (cstatus != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Torch: DtoH transfer failed: {}", cudaGetErrorString(cstatus));
        return InferStatus(holoinfer_code::H_ERROR, "Torch core, DtoH transfer.");
      }
      // When copying from device memory to pagable memory the call is synchronous with the host
      // execution. No need to synchronize here.
    }
  } else {
    if (infer_device == torch::kCPU) {
      auto cstatus = cudaMemcpyAsync(output_buffer->device_buffer_->data(),
                                     output_tensor.data_ptr(),
                                     output_tensor_size * element_size,
                                     cudaMemcpyHostToDevice,
                                     cstream);
      if (cstatus != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Torch: HtoD transfer failed: {}", cudaGetErrorString(cstatus));
        return InferStatus(holoinfer_code::H_ERROR, "Torch core, HtoD transfer.");
      }
      // When copying from pagable memory to device memory cudaMemcpyAsync() is copying the memory
      // to staging memory first and therefore is synchronous with the host execution. No need to
      // synchronize here.
    } else {
      auto cstatus = cudaMemcpyAsync(output_buffer->device_buffer_->data(),
                                     output_tensor.data_ptr(),
                                     output_tensor_size * element_size,
                                     cudaMemcpyDeviceToDevice,
                                     cstream);
      if (cstatus != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Torch: DtoD transfer failed: {}", cudaGetErrorString(cstatus));
        return InferStatus(holoinfer_code::H_ERROR, "Torch core, DtoD transfer.");
      }
    }
  }
  return InferStatus();
}

InferStatus TorchInferImpl::transfer_to_output(std::shared_ptr<DataBuffer>& output_buffer,
                                               torch::Tensor out_torch_tensor,
                                               std::vector<int64_t>& output_dim,
                                               const std::string& data_type_str) {
  auto data_type = output_buffer->get_datatype();
  out_torch_tensor = out_torch_tensor.contiguous();
  auto cstream = infer_stream_.stream();

  if (data_type != kHoloInferDataTypeMap.at(data_type_str)) {
    HOLOSCAN_LOG_ERROR("Torch core: Failed to transfer to output. Data type mismatch, expected: {}",
                       data_type_str);
    return InferStatus(holoinfer_code::H_ERROR, "Torch core, data type mismatch.");
  }
  try {
    return transfer_to_output_core(output_buffer,
                                   out_torch_tensor,
                                   output_dim,
                                   infer_device_,
                                   output_device_,
                                   kTorchTypeMap.at(data_type_str),
                                   cstream);
  } catch (const std::out_of_range& e) {
    std::vector<std::string> supported_types;
    for (const auto& [key, value] : kTorchTypeMap) {
      supported_types.push_back(key);
    }
    HOLOSCAN_LOG_ERROR(
        "Torch core: Unsupported data type when processing output: {}, supported types: {}",
        data_type_str,
        fmt::join(supported_types, ", "));
    return InferStatus(holoinfer_code::H_ERROR,
                       "Torch core: Failed to transfer torch tensor to output buffer.");
  } catch (const c10::Error& e) {
    HOLOSCAN_LOG_ERROR("Torch core: error transferring data from torch tensor to output buffer: {}",
                       e.what());
    return InferStatus(holoinfer_code::H_ERROR,
                       "Torch core: Failed to transfer torch tensor to output buffer.");
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR(
        "Torch core: standard error in transferring data from torch tensor to output buffer: {}",
        e.what());
    return InferStatus(holoinfer_code::H_ERROR,
                       "Torch core: Failed to transfer torch tensor to output buffer.");
  } catch (...) {
    HOLOSCAN_LOG_ERROR(
        "Torch core: unknown error in transferring data from torch tensor to output buffer");
    return InferStatus(holoinfer_code::H_ERROR,
                       "Torch core: Failed to transfer torch tensor to output buffer.");
  }
}

void TorchInfer::print_model_details() {
  impl_->print_model_details();
}

void TorchInferImpl::print_model_details() {
  HOLOSCAN_LOG_INFO("Input node count: {}", input_nodes_);
  HOLOSCAN_LOG_INFO("Output node count: {}", output_nodes_);

  HOLOSCAN_LOG_INFO("Input names: [{}]", fmt::join(input_names_, ", "));
  for (size_t a = 0; a < input_dims_.size(); a++) {
    HOLOSCAN_LOG_INFO(
        "Input Dimension for {}: [{}]", input_names_[a], fmt::join(input_dims_[a], ", "));
  }
  HOLOSCAN_LOG_INFO("Output names: [{}]", fmt::join(output_names_, ", "));
  for (size_t a = 0; a < output_dims_.size(); a++) {
    HOLOSCAN_LOG_INFO(
        "Output Dimension for {}: [{}]", output_names_[a], fmt::join(output_dims_[a], ", "));
  }
}

InferStatus TorchInferImpl::parse_node_details(const YAML::Node& config,
                                               const std::string& config_yaml_file_path) {
  // Read input nodes
  if (!config["inference"]["input_nodes"]) {
    HOLOSCAN_LOG_ERROR("Torch core: input_nodes key not present in model config file {}",
                       config_yaml_file_path);
    return InferStatus(holoinfer_code::H_ERROR, "Torch core, incorrect config file.");
  }
  auto infer_in_config = config["inference"]["input_nodes"];
  input_nodes_ = infer_in_config.size();

  auto status = parse_yaml_node(infer_in_config, input_names_, input_dims_, input_type_);
  if (status.get_code() != holoinfer_code::H_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Torch core: Node parsing failed for input.");
    return status;
  }

  if (!config["inference"]["output_nodes"]) {
    HOLOSCAN_LOG_ERROR("Torch core: output_nodes key not present in model config file {}",
                       config_yaml_file_path);
    return InferStatus(holoinfer_code::H_ERROR, "Torch core, incorrect config file.");
  }
  auto infer_out_config = config["inference"]["output_nodes"];
  output_nodes_ = infer_out_config.size();

  status = parse_yaml_node(infer_out_config, output_names_, output_dims_, output_type_);
  if (status.get_code() != holoinfer_code::H_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Torch core: Node parsing failed for output.");
    return status;
  }

  return InferStatus();
}

InferStatus TorchInferImpl::torch_struct_to_map(
    const YAML::Node& node, const torch::IValue& torch_struct,
    std::unordered_map<std::string, torch::Tensor>& data_map) {
  std::queue<std::pair<YAML::Node, torch::IValue>> node_torch_struct_pair;
  node_torch_struct_pair.push({node, torch_struct});

  while (!node_torch_struct_pair.empty()) {
    auto [current_node, current_torch_struct] = node_torch_struct_pair.front();
    node_torch_struct_pair.pop();

    // Skip null nodes
    if (current_node.IsNull()) {
      continue;
    }

    if (current_node.IsScalar() && current_torch_struct.isTensor()) {
      std::string key = current_node.as<std::string>();
      data_map.emplace(key, current_torch_struct.toTensor());
    } else if (current_node.IsSequence() && current_torch_struct.isList()) {
      auto list = current_torch_struct.toList();
      if (current_node.size() != list.size()) {
        HOLOSCAN_LOG_ERROR("Torch core: List size mismatch. Expected: {}, Got: {}",
                           current_node.size(),
                           list.size());
        return InferStatus(holoinfer_code::H_ERROR, "Torch core: List size mismatch.");
      }
      for (size_t i = 0; i < list.size(); ++i) {
        node_torch_struct_pair.push({current_node[i], list[i]});
      }
    } else if (current_node.IsSequence() && current_torch_struct.isTuple()) {
      auto tuple = current_torch_struct.toTuple();
      if (current_node.size() != tuple->elements().size()) {
        HOLOSCAN_LOG_ERROR("Torch core: Tuple size mismatch. Expected: {}, Got: {}",
                           current_node.size(),
                           tuple->elements().size());
        return InferStatus(holoinfer_code::H_ERROR, "Torch core: Tuple size mismatch.");
      }
      for (size_t i = 0; i < tuple->elements().size(); ++i) {
        node_torch_struct_pair.push({current_node[i], tuple->elements()[i]});
      }
    } else if (current_node.IsMap() && current_torch_struct.isGenericDict()) {
      auto dict = current_torch_struct.toGenericDict();
      for (const auto& item : dict) {
        node_torch_struct_pair.push({current_node[item.key().toString()->string()], item.value()});
      }
    } else {
      HOLOSCAN_LOG_ERROR(
          "Torch core: Error mapping from torch struct, unsupported node type or mismatch in type "
          "expected format");
      return InferStatus(holoinfer_code::H_ERROR, "Torch core: Error mapping from torch struct.");
    }
  }
  return InferStatus();
}

InferStatus TorchInferImpl::set_input_processors(const std::vector<std::string>& input_type_structs,
                                                 const std::vector<YAML::Node>& input_format) {
  input_processors_.clear();
  for (size_t i = 0; i < input_type_structs.size(); ++i) {
    const std::string& type_str = input_type_structs[i];
    const YAML::Node& schema_node = input_format[i];

    if (type_str == "Tensor") {
      input_processors_.push_back({[this, schema_node](const auto& tensor_map) -> torch::IValue {
                                     return torch::IValue(
                                         get_tensor_from_schema(schema_node, tensor_map));
                                   },
                                   schema_node});
    } else if (type_str == "Tensor[]") {
      input_processors_.push_back({[this, schema_node](const auto& tensor_map) -> torch::IValue {
                                     return torch::IValue(
                                         get_list_from_schema(schema_node, tensor_map));
                                   },
                                   schema_node});
    } else if (type_str == "Dict(str, Tensor)") {
      input_processors_.push_back({[this, schema_node](const auto& tensor_map) -> torch::IValue {
                                     return torch::IValue(
                                         get_dict_from_schema(schema_node, tensor_map));
                                   },
                                   schema_node});
    } else if (type_str == "Dict(str, Tensor[])") {
      input_processors_.push_back({[this, schema_node](const auto& tensor_map) -> torch::IValue {
                                     return torch::IValue(
                                         get_dict_of_list_from_schema(schema_node, tensor_map));
                                   },
                                   schema_node});
    } else if (type_str == "Dict(str, Dict(str, Tensor))") {
      input_processors_.push_back({[this, schema_node](const auto& tensor_map) -> torch::IValue {
                                     return torch::IValue(
                                         get_dict_of_dict_from_schema(schema_node, tensor_map));
                                   },
                                   schema_node});
    } else if (type_str == "Dict(str, Tensor)[]") {
      input_processors_.push_back({[this, schema_node](const auto& tensor_map) -> torch::IValue {
                                     return torch::IValue(
                                         get_list_of_dict_from_schema(schema_node, tensor_map));
                                   },
                                   schema_node});
    } else if (type_str == "Tensor[][]") {
      input_processors_.push_back({[this, schema_node](const auto& tensor_map) -> torch::IValue {
                                     return torch::IValue(
                                         get_list_of_list_from_schema(schema_node, tensor_map));
                                   },
                                   schema_node});
    } else {
      HOLOSCAN_LOG_ERROR("Torch core: Unknown input type stream: {}", type_str);
      return InferStatus(holoinfer_code::H_ERROR, "Provided input type stream is not supported.");
    }
  }
  return InferStatus();
}

InferStatus TorchInferImpl::get_io_schema(const YAML::Node& yaml_node, std::string& type_stream) {
  if (yaml_node.IsSequence()) {
    std::vector<std::string> child_type_streams;
    for (const auto& child : yaml_node) {
      std::string child_type_stream;
      auto status = get_io_schema(child, child_type_stream);
      if (status.get_code() != holoinfer_code::H_SUCCESS) {
        HOLOSCAN_LOG_ERROR("Torch core: Error getting input type stream");
        return status;
      }
      child_type_streams.push_back(child_type_stream);
    }
    if (!child_type_streams.empty()) {
      std::string first_type = child_type_streams[0];
      for (size_t i = 1; i < child_type_streams.size(); ++i) {
        if (child_type_streams[i] != first_type) {
          HOLOSCAN_LOG_ERROR("Torch core: Inconsistent types in schema - found {} and {}",
                             first_type,
                             child_type_streams[i]);
          return InferStatus(holoinfer_code::H_ERROR, "Inconsistent types in schema");
        }
      }
      type_stream = first_type;
    } else {
      HOLOSCAN_LOG_ERROR("Torch core: Empty list in schema");
      return InferStatus(holoinfer_code::H_ERROR, "Empty list in schema");
    }
    type_stream = fmt::format("{}[]", type_stream);
    return InferStatus();
  } else if (yaml_node.IsMap()) {
    std::vector<std::string> child_type_streams;
    for (const auto& child : yaml_node) {
      std::string child_type_stream;
      auto status = get_io_schema(child.second, child_type_stream);
      if (status.get_code() != holoinfer_code::H_SUCCESS) {
        return status;
      }
      child_type_streams.push_back(child_type_stream);
    }
    if (!child_type_streams.empty()) {
      std::string first_type = child_type_streams[0];
      for (size_t i = 1; i < child_type_streams.size(); ++i) {
        if (child_type_streams[i] != first_type) {
          HOLOSCAN_LOG_ERROR("Torch core: Inconsistent types in schema - found {} and {}",
                             first_type,
                             child_type_streams[i]);
          return InferStatus(holoinfer_code::H_ERROR, "Inconsistent types in schema");
        }
      }
      type_stream = first_type;
    } else {
      type_stream = "{}";
    }
    type_stream = fmt::format("Dict(str, {})", type_stream);
  } else if (yaml_node.IsScalar()) {
    type_stream = "Tensor";
  } else if (yaml_node.IsNull()) {
    HOLOSCAN_LOG_ERROR("Torch core: Null values are not allowed in input format");
    return InferStatus(holoinfer_code::H_ERROR, "Null values are not allowed in input format");
  } else {
    HOLOSCAN_LOG_ERROR("Torch core: Unsupported node type in schema");
    return InferStatus(holoinfer_code::H_ERROR, "Unsupported node type in schema");
  }
  return InferStatus();
}

InferStatus TorchInferImpl::populate_model_details() {
  std::string config_yaml_file_path;
  if (std::filesystem::exists(model_path_)) {
    config_yaml_file_path =
        std::filesystem::path(model_path_).replace_extension("").string() + ".yaml";
    if (!std::filesystem::exists(config_yaml_file_path)) {
      HOLOSCAN_LOG_ERROR("Inference config path not found: {}", config_yaml_file_path);
      return InferStatus(holoinfer_code::H_ERROR, "Torch core, invalid config path.");
    }
  } else {
    HOLOSCAN_LOG_ERROR("Inference config path not found: {}", model_path_);
    return InferStatus(holoinfer_code::H_ERROR, "Torch core, invalid model path.");
  }

  try {
    YAML::Node config = YAML::LoadFile(config_yaml_file_path);

    if (!config["inference"]) {
      HOLOSCAN_LOG_ERROR("Torch core: inference key not present in model config file {}",
                         config_yaml_file_path);
      return InferStatus(holoinfer_code::H_ERROR, "Torch core, incorrect config file.");
    }

    auto status = parse_node_details(config, config_yaml_file_path);
    if (status.get_code() != holoinfer_code::H_SUCCESS) {
      HOLOSCAN_LOG_ERROR("Torch core: Error parsing node details");
      return status;
    }

    YAML::Node input_format, output_format;
    // input format
    if (config["inference"]["input_format"]) {
      input_format = config["inference"]["input_format"];

      if (!input_format.IsSequence()) {
        HOLOSCAN_LOG_ERROR("Torch core: input format needs to be a list");
        return InferStatus(holoinfer_code::H_ERROR, "Torch core, input format needs to be a list.");
      }
    } else {
      // assume the input and output format are in the order of input_nodes and output_nodes
      HOLOSCAN_LOG_WARN("Torch core: no input format provided, using default format");
      input_format = YAML::Node(YAML::NodeType::Sequence);
      for (size_t i = 0; i < input_nodes_; i++) {
        input_format.push_back(input_names_[i]);
      }
    }
    HOLOSCAN_LOG_INFO("Torch core: input format: \n{}", YAML::Dump(input_format));

    std::vector<std::string> input_type_structs;
    for (const auto& input_argument : input_format) {
      std::string input_type_stream;
      status = get_io_schema(input_argument, input_type_stream);
      if (status.get_code() != holoinfer_code::H_SUCCESS) {
        HOLOSCAN_LOG_ERROR("Torch core: Error getting input type stream");
        return status;
      }
      input_type_structs.push_back(input_type_stream);
    }

    status = set_input_processors(input_type_structs, input_format.as<std::vector<YAML::Node>>());
    if (status.get_code() != holoinfer_code::H_SUCCESS) {
      HOLOSCAN_LOG_ERROR("Torch core: Error setting input processors");
      return status;
    }

    // output format
    if (config["inference"]["output_format"]) {
      output_format = config["inference"]["output_format"];
    } else {
      HOLOSCAN_LOG_WARN("Torch core: no output format provided, using default format");
      if (output_names_.size() == 1) {
        // for single output, we assume the output is single tensor instead of sequence of tensors
        output_format = YAML::Node(output_names_[0]);
      } else {
        output_format = YAML::Node(YAML::NodeType::Sequence);
        for (size_t i = 0; i < output_nodes_; i++) {
          output_format.push_back(output_names_[i]);
        }
      }
    }
    HOLOSCAN_LOG_INFO("Torch core: output format: \n {}", YAML::Dump(output_format));
    output_format_ = output_format;

    print_model_details();
  } catch (const YAML::ParserException& ex) {
    HOLOSCAN_LOG_ERROR("YAML error: {}", ex.what());
    return InferStatus(holoinfer_code::H_ERROR, "Torch core, YAML error.");
  }

  return InferStatus();
}

extern "C" TorchInfer* NewTorchInfer(const std::string& model_file_path, bool cuda_flag,
                                     bool cuda_buf_in, bool cuda_buf_out) {
  return new TorchInfer(model_file_path, cuda_flag, cuda_buf_in, cuda_buf_out);
}

TorchInfer::TorchInfer(const std::string& model_file_path, bool cuda_flag, bool cuda_buf_in,
                       bool cuda_buf_out)
    : impl_(new TorchInferImpl(model_file_path, cuda_flag, cuda_buf_in, cuda_buf_out)) {}

TorchInfer::~TorchInfer() {
  if (impl_) {
    delete impl_;
    impl_ = nullptr;
  }
}

TorchInferImpl::TorchInferImpl(const std::string& model_file_path, bool cuda_flag, bool cuda_buf_in,
                               bool cuda_buf_out)
    : model_path_(model_file_path) {
  try {
    infer_stream_ = c10::cuda::getStreamFromPool(true);
    stream_guard_ = std::make_unique<c10::cuda::CUDAStreamGuard>(infer_stream_);
    check_cuda(cudaEventCreateWithFlags(&cuda_event_, cudaEventDisableTiming));

    HOLOSCAN_LOG_INFO("Loading torchscript: {}", model_path_);
    inference_module_ = torch::jit::load(model_path_);
    inference_module_.eval();
    HOLOSCAN_LOG_INFO("Torchscript loaded");

    auto status = populate_model_details();
    if (status.get_code() != holoinfer_code::H_SUCCESS) {
      HOLOSCAN_LOG_ERROR(status.get_message());
      HOLOSCAN_LOG_ERROR("Torch core: Error populating model parameters");
      throw std::runtime_error("Torch core: constructor failed.");
    }

    infer_device_ = cuda_flag ? torch::kCUDA : torch::kCPU;
    input_device_ = cuda_buf_in ? torch::kCUDA : torch::kCPU;
    output_device_ = cuda_buf_out ? torch::kCUDA : torch::kCPU;

    torch::jit::getProfilingMode() = false;  // profiling mode on slows down things in start
    // May be exposed as parameter in future releases
    torch::jit::GraphOptimizerEnabledGuard guard{false};
    inference_module_.to(infer_device_);

    torch::NoGradGuard no_grad;
  } catch (const c10::Error& exception) {
    HOLOSCAN_LOG_ERROR(exception.what());
    throw;
  } catch (std::exception& e) {
    HOLOSCAN_LOG_ERROR(e.what());
    throw;
  } catch (...) {
    throw;
  }
}

TorchInferImpl::~TorchInferImpl() {
  if (cuda_event_) {
    cudaEventDestroy(cuda_event_);
  }
}

InferStatus TorchInfer::do_inference(const std::vector<std::shared_ptr<DataBuffer>>& input_buffer,
                                     std::vector<std::shared_ptr<DataBuffer>>& output_buffer,
                                     cudaEvent_t cuda_event_data,
                                     cudaEvent_t* cuda_event_inference) {
  InferStatus status = InferStatus(holoinfer_code::H_SUCCESS);

  // synchronize the CUDA stream used for inference with the CUDA event recorded when preparing
  // the input data
  check_cuda(cudaStreamWaitEvent(impl_->infer_stream_.stream(), cuda_event_data));

  impl_->stream_guard_->reset_stream(impl_->infer_stream_);

  if (impl_->input_nodes_ != input_buffer.size()) {
    status.set_message("Torch inference core: Input buffer size not equal to input nodes.");
    return status;
  }
  if (impl_->output_nodes_ != output_buffer.size()) {
    status.set_message("Torch inference core: Output buffer size not equal to output nodes.");
    return status;
  }

  impl_->inputs_.clear();
  torch::IValue output_struct;
  std::unordered_map<std::string, torch::Tensor> input_map;
  std::unordered_map<std::string, torch::Tensor> output_map;

  // transform input data to tensors
  for (size_t i = 0; i < impl_->input_nodes_; i++) {
    torch::Tensor input_tensor =
        impl_->create_tensor(input_buffer[i], impl_->input_dims_[i], impl_->input_type_[i], status);
    input_map.emplace(impl_->input_names_[i], input_tensor);
  }
  if (status.get_code() != holoinfer_code::H_SUCCESS) {
    return status;
  }

  // create torch inputs
  try {
    for (const auto& processor : impl_->input_processors_) {
      impl_->inputs_.push_back(processor.processor(input_map));
    }
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Torch core: Error in input creation: {}", e.what());
    return InferStatus(holoinfer_code::H_ERROR, "Torch core, input creation failed");
  }

  // compute module forward
  try {
    impl_->outputs_ = impl_->inference_module_.forward(impl_->inputs_);
  } catch (const c10::Error& exception) {
    HOLOSCAN_LOG_ERROR(exception.what());
    throw;
  } catch (std::exception& e) {
    HOLOSCAN_LOG_ERROR(e.what());
    throw;
  } catch (...) {
    throw;
  }
  if (impl_->infer_device_ == torch::kCUDA) {
    c10::cuda::getCurrentCUDAStream().synchronize();
  }

  try {
    status = impl_->torch_struct_to_map(impl_->output_format_, impl_->outputs_, output_map);
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR(e.what());
    throw;
  } catch (...) {
    throw;
  }
  if (status.get_code() != holoinfer_code::H_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Torch inference core: Error mapping output to structure");
    return status;
  }

  if (output_map.size() != output_buffer.size()) {
    HOLOSCAN_LOG_ERROR("Output buffer size ({}) not equal to number of tensors from the model ({})",
                       output_buffer.size(),
                       output_map.size());
    return InferStatus(
        holoinfer_code::H_ERROR,
        "Torch core: Output buffer size not equal to number of tensors from the model.");
  }
  for (unsigned int a = 0; a < output_buffer.size(); a++) {
    torch::Tensor current_tensor;
    try {
      current_tensor = output_map.at(impl_->output_names_[a]);
    } catch (const std::out_of_range& e) {
      HOLOSCAN_LOG_ERROR("Torch core: Output tensor name '{}' not found in model output",
                         impl_->output_names_[a]);
      return InferStatus(
          holoinfer_code::H_ERROR,
          "Torch core: Error in Torch core, output tensor name not found in output map.");
    }
    auto status = impl_->transfer_to_output(output_buffer[a],
                                            std::move(current_tensor),
                                            impl_->output_dims_[a],
                                            impl_->output_type_[a]);
    if (status.get_code() != holoinfer_code::H_SUCCESS) {
      HOLOSCAN_LOG_ERROR("Transfer of Tensor {} failed in inference core.",
                         impl_->output_names_[a]);
      return status;
    }
  }

  // record a CUDA event and pass it back to the caller
  check_cuda(cudaEventRecord(impl_->cuda_event_, impl_->infer_stream_.stream()));
  *cuda_event_inference = impl_->cuda_event_;
  return InferStatus();
}

std::vector<std::vector<int64_t>> TorchInfer::get_input_dims() const {
  return impl_->input_dims_;
}

std::vector<std::vector<int64_t>> TorchInfer::get_output_dims() const {
  return impl_->output_dims_;
}

std::vector<holoinfer_datatype> TorchInfer::get_input_datatype() const {
  std::vector<holoinfer_datatype> result;
  for (const auto& type_str : impl_->input_type_) {
    result.push_back(kHoloInferDataTypeMap.at(type_str));
  }
  return result;
}

std::vector<holoinfer_datatype> TorchInfer::get_output_datatype() const {
  std::vector<holoinfer_datatype> result;
  for (const auto& type_str : impl_->output_type_) {
    result.push_back(kHoloInferDataTypeMap.at(type_str));
  }
  return result;
}

}  // namespace inference

}  // namespace holoscan
