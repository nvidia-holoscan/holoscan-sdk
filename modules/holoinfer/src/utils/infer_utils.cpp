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

#include <holoinfer_utils.hpp>

namespace holoscan {
namespace inference {

gxf_result_t report_error(const std::string& module, const std::string& submodule) {
  std::string error_string{"Error in " + module + ", Sub-module->" + submodule};
  GXF_LOG_ERROR("%s\n", error_string.c_str());
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
  int64_t delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  GXF_LOG_DEBUG("%s : %d ms", module.c_str(), delta);
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

InferStatus map_data_to_model_from_tensor(const MultiMappings& model_data_mapping,
                                          DataMap& data_per_model, DataMap& data_per_input_tensor) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);

  if (model_data_mapping.size() == 0) {
    status.set_message("Model mapping cannot be empty.");
    return status;
  }
  for (auto& md_mapping : model_data_mapping) {
    if (md_mapping.second.size() != 1) {
      status.set_message("Entry in model data mapping must be of size 1.");
      return status;
    }
    auto tensor_name = md_mapping.second[0];

    if (data_per_input_tensor.find(tensor_name) == data_per_input_tensor.end()) {
      status.set_message("Tensor " + tensor_name + " missing in data_per_input_tensor.");
      return status;
    }

    if (data_per_model.find(md_mapping.first) == data_per_model.end()) {
      data_per_model.insert({md_mapping.first, data_per_input_tensor.at(tensor_name)});
    }
  }
  return InferStatus();
}

gxf_result_t multiai_get_data_per_model(const GXFReceivers& receivers,
                                        const std::vector<std::string>& in_tensors,
                                        DataMap& data_per_input_tensor,
                                        std::map<std::string, std::vector<int>>& dims_per_tensor,
                                        bool cuda_buffer_out, const std::string& module) {
  try {
    TimePoint s_time, e_time;
    timer_init(s_time);

    nvidia::gxf::MemoryStorageType to = nvidia::gxf::MemoryStorageType::kHost;

    if (cuda_buffer_out) { to = nvidia::gxf::MemoryStorageType::kDevice; }

    std::vector<nvidia::gxf::Entity> messages;
    messages.reserve(receivers.size());
    for (auto& receiver : receivers) {
      auto maybe_message = receiver->receive();
      if (maybe_message) { messages.push_back(std::move(maybe_message.value())); }
    }
    if (messages.empty()) {
      GXF_LOG_ERROR("No message available.");
      return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE;
    }
    // Process input message
    for (unsigned int i = 0; i < in_tensors.size(); ++i) {
      nvidia::gxf::Handle<nvidia::gxf::Tensor> in_tensor;

      for (unsigned int j = 0; j < messages.size(); ++j) {
        const auto& in_message = messages[j];

        const auto maybe_tensor = in_message.get<nvidia::gxf::Tensor>(in_tensors[i].c_str());
        if (maybe_tensor) {
          in_tensor = maybe_tensor.value();
          break;
        }
      }
      if (!in_tensor)
        return report_error("Inference toolkit",
                            "Data_per_tensor, Tensor " + in_tensors[i] + " not found");

      void* in_tensor_data = in_tensor->pointer();

      auto element_type = in_tensor->element_type();
      auto storage_type = in_tensor->storage_type();

      if (element_type != nvidia::gxf::PrimitiveType::kFloat32) {
        return report_error("Inference Toolkit", "Data extraction, element type not supported");
      }
      if (!(storage_type != nvidia::gxf::MemoryStorageType::kHost ||
            storage_type != nvidia::gxf::MemoryStorageType::kDevice)) {
        return report_error("Inference Toolkit",
                            "Data extraction, memory not resident on CPU or GPU");
      }
      if (to != nvidia::gxf::MemoryStorageType::kHost &&
          to != nvidia::gxf::MemoryStorageType::kDevice) {
        return report_error("Inference Toolkit",
                            "Input storage type in data extraction not supported");
      }

      std::vector<int> dims;
      for (unsigned int i = 0; i < in_tensor->shape().rank(); ++i)
        dims.push_back(in_tensor->shape().dimension(i));

      size_t buffer_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());

      if (data_per_input_tensor.find(in_tensors[i]) == data_per_input_tensor.end()) {
        auto db = std::make_shared<DataBuffer>();
        db->host_buffer.resize(buffer_size);
        db->device_buffer->resize(buffer_size);

        data_per_input_tensor.insert({in_tensors[i], std::move(db)});
      }

      dims_per_tensor[in_tensors[i]] = std::move(dims);

      if (to == nvidia::gxf::MemoryStorageType::kHost) {
        std::vector<float> in_tensor_ptr(buffer_size, 0);

        if (storage_type == nvidia::gxf::MemoryStorageType::kDevice) {
          cudaError_t cuda_result = cudaMemcpy(static_cast<void*>(in_tensor_ptr.data()),
                                               static_cast<const void*>(in_tensor_data),
                                               buffer_size * sizeof(float),
                                               cudaMemcpyDeviceToHost);
          if (cuda_result != cudaSuccess) {
            return report_error("Inference Toolkit",
                                "Data extraction, error in DtoH cudaMemcpy::" +
                                    std::string(cudaGetErrorString(cuda_result)));
          }

        } else if (storage_type == nvidia::gxf::MemoryStorageType::kHost) {
          memcpy(in_tensor_ptr.data(),
                 static_cast<float*>(in_tensor_data),
                 buffer_size * sizeof(float));
        }
        data_per_input_tensor.at(in_tensors[i])->host_buffer = std::move(in_tensor_ptr);

      } else {
        if (to == nvidia::gxf::MemoryStorageType::kDevice) {
          void* device_buff = data_per_input_tensor.at(in_tensors[i])->device_buffer->data();
          if (storage_type == nvidia::gxf::MemoryStorageType::kDevice) {
            cudaError_t cuda_result = cudaMemcpy(static_cast<void*>(device_buff),
                                                 static_cast<const void*>(in_tensor_data),
                                                 buffer_size * sizeof(float),
                                                 cudaMemcpyDeviceToDevice);

            if (cuda_result != cudaSuccess) {
              return report_error("Inference Toolkit",
                                  "Data extraction, error in DtoD cudaMemcpy::" +
                                      std::string(cudaGetErrorString(cuda_result)));
            }
          } else {
            return report_error("Inference toolkit", "Data extraction parameters not supported.");
          }
        }
      }
    }

    timer_init(e_time);
  } catch (std::exception& _ex) {
    return report_error("Inference toolkit",
                        "Data_per_tensor, Message: " + std::string(_ex.what()));
  } catch (...) { return report_error("Inference toolkit", "Data_per_tensor, Unknown exception"); }
  return GXF_SUCCESS;
}

gxf_result_t multiai_transmit_data_per_model(
    gxf_context_t& cont, const Mappings& model_to_tensor_map, DataMap& input_data_map,
    const GXFTransmitters& transmitters, const std::vector<std::string>& out_tensors,
    DimType& tensor_out_dims_map, bool cuda_buffer_in, bool cuda_buffer_out,
    const nvidia::gxf::PrimitiveType& element_type, const std::string& module,
    const nvidia::gxf::Handle<nvidia::gxf::Allocator>& allocator_) {
  try {
    if (element_type != nvidia::gxf::PrimitiveType::kFloat32) {
      return report_error("Inference toolkit", "Element type to be transmitted not supported");
    }

    nvidia::gxf::MemoryStorageType from = nvidia::gxf::MemoryStorageType::kHost;
    nvidia::gxf::MemoryStorageType to = nvidia::gxf::MemoryStorageType::kHost;

    if (cuda_buffer_in) { from = nvidia::gxf::MemoryStorageType::kDevice; }
    if (cuda_buffer_out) { to = nvidia::gxf::MemoryStorageType::kDevice; }

    TimePoint s_time, e_time;

    timer_init(s_time);

    // single transmitter used
    auto out_message = nvidia::gxf::Entity::New(cont);
    if (!out_message) { return report_error(module, "Inference Toolkit, Out message allocation"); }

    for (unsigned int i = 0; i < out_tensors.size(); ++i) {
      if (input_data_map.find(out_tensors[i]) == input_data_map.end())
        return report_error(module,
                            "Inference Toolkit, Mapped data not found for " + out_tensors[i]);

      std::string key_name{""};

      for (const auto& key_to_tensor : model_to_tensor_map) {
        if (key_to_tensor.second.compare(out_tensors[i]) == 0) key_name = key_to_tensor.first;
      }
      if (key_name.length() == 0)
        return report_error(module, "Tensor mapping not found in model to tensor map");

      if (tensor_out_dims_map.find(key_name) == tensor_out_dims_map.end())
        return report_error(module, "Tensor mapping not found in dimension map for " + key_name);

      auto out_tensor = out_message.value().add<nvidia::gxf::Tensor>(out_tensors[i].c_str());
      if (!out_tensor) return report_error(module, "Inference Toolkit, Out tensor allocation");

      std::vector<int64_t> dims = tensor_out_dims_map.at(key_name);

      nvidia::gxf::Shape output_shape;
      if (dims.size() == 4) {
        std::array<int, 4> dimarray;
        for (size_t i = 0; i < 4; ++i) { dimarray[i] = (static_cast<int>(dims[i])); }
        nvidia::gxf::Shape out_shape(dimarray);
        output_shape = std::move(out_shape);
      }
      if (dims.size() == 2) {
        nvidia::gxf::Shape out_shape{static_cast<int>(dims[0]), static_cast<int>(dims[1])};
        output_shape = std::move(out_shape);
      }

      size_t buffer_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());

      if (from == nvidia::gxf::MemoryStorageType::kHost) {
        if (to == nvidia::gxf::MemoryStorageType::kHost) {
          out_tensor.value()->reshape<float>(
              output_shape, nvidia::gxf::MemoryStorageType::kHost, allocator_);
          if (!out_tensor.value()->pointer())
            return report_error(module, "Inference Toolkit, Out tensor buffer allocation");

          nvidia::gxf::Expected<float*> out_tensor_data = out_tensor.value()->data<float>();
          if (!out_tensor_data)
            return report_error(module, "Inference Toolkit, Getting out tensor data");

          auto current_model_output = input_data_map.at(out_tensors[i]);
          memcpy(out_tensor_data.value(),
                 current_model_output->host_buffer.data(),
                 buffer_size * sizeof(float));
        } else {  // to is on device
          out_tensor.value()->reshape<float>(
              output_shape, nvidia::gxf::MemoryStorageType::kDevice, allocator_);
          if (!out_tensor.value()->pointer())
            return report_error(module, "Inference Toolkit, Out tensor buffer allocation");

          nvidia::gxf::Expected<float*> out_tensor_data = out_tensor.value()->data<float>();
          if (!out_tensor_data)
            return report_error(module, "Inference Toolkit, Getting out tensor data");

          auto current_model_dev_buff = input_data_map.at(out_tensors[i])->host_buffer.data();
          cudaError_t cuda_result = cudaMemcpy(static_cast<void*>(out_tensor_data.value()),
                                               static_cast<const void*>(current_model_dev_buff),
                                               buffer_size * sizeof(float),
                                               cudaMemcpyHostToDevice);
          if (cuda_result != cudaSuccess) {
            return report_error("Inference Toolkit",
                                "Data extraction, error in DtoD cudaMemcpy::" +
                                    std::string(cudaGetErrorString(cuda_result)));
          }
        }
      } else {
        if (from == nvidia::gxf::MemoryStorageType::kDevice) {
          if (to == nvidia::gxf::MemoryStorageType::kDevice) {
            out_tensor.value()->reshape<float>(
                output_shape, nvidia::gxf::MemoryStorageType::kDevice, allocator_);
            if (!out_tensor.value()->pointer())
              return report_error(module, "Inference Toolkit, Out tensor buffer allocation");

            nvidia::gxf::Expected<float*> out_tensor_data = out_tensor.value()->data<float>();
            if (!out_tensor_data)
              return report_error(module, "Inference Toolkit, Getting out tensor data");

            void* current_model_dev_buff = input_data_map.at(out_tensors[i])->device_buffer->data();
            cudaError_t cuda_result = cudaMemcpy(static_cast<void*>(out_tensor_data.value()),
                                                 static_cast<const void*>(current_model_dev_buff),
                                                 buffer_size * sizeof(float),
                                                 cudaMemcpyDeviceToDevice);
            if (cuda_result != cudaSuccess) {
              return report_error("Inference Toolkit",
                                  "Data extraction, error in DtoD cudaMemcpy::" +
                                      std::string(cudaGetErrorString(cuda_result)));
            }
          } else {  // to is on host
            out_tensor.value()->reshape<float>(
                output_shape, nvidia::gxf::MemoryStorageType::kHost, allocator_);
            if (!out_tensor.value()->pointer())
              return report_error(module, "Inference Toolkit, Out tensor buffer allocation");

            nvidia::gxf::Expected<float*> out_tensor_data = out_tensor.value()->data<float>();
            if (!out_tensor_data)
              return report_error(module, "Inference Toolkit, Getting out tensor data");

            void* current_model_dev_buff = input_data_map.at(out_tensors[i])->device_buffer->data();
            cudaError_t cuda_result = cudaMemcpy(static_cast<void*>(out_tensor_data.value()),
                                                 static_cast<const void*>(current_model_dev_buff),
                                                 buffer_size * sizeof(float),
                                                 cudaMemcpyDeviceToHost);
            if (cuda_result != cudaSuccess) {
              return report_error("Inference Toolkit",
                                  "Data extraction, error in DtoH cudaMemcpy::" +
                                      std::string(cudaGetErrorString(cuda_result)));
            }
          }
        }
      }
    }
    // single transmitter used
    const auto result = transmitters[0]->publish(std::move(out_message.value()));
    if (!result) return report_error(module, "Inference Toolkit, Publishing output");

    timer_init(e_time);
  } catch (std::exception& _ex) {
    return report_error("Inference toolkit",
                        "Data_transmission_tensor, Message: " + std::string(_ex.what()));
  } catch (...) {
    return report_error("Inference toolkit", "Data_transmission_tensor, Unknown exception");
  }
  return GXF_SUCCESS;
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
      // Only 1 Tensor supported per model as MultiMappings input
      if (map_data.second.size() != 1) {
        status.set_message(type_of_map + ": 1 tensor per model supported. Found: " +
                           std::to_string(map_data.second.size()));
        return status;
      } else {
        if (map_data.first.empty() || map_data.second[0].empty()) {
          status.set_message("Empty entry for key or value in " + type_of_map);
          return status;
        }
      }
    }
  }

  return InferStatus();
}

InferStatus check_mappings_size_value(const Mappings& input_map, const std::string& type_of_map) {
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

InferStatus multiai_inference_validity_check(const Mappings& model_path_map,
                                             const MultiMappings& pre_processor_map,
                                             const Mappings& inference_map,
                                             const std::vector<std::string>& in_tensor_names,
                                             const std::vector<std::string>& out_tensor_names) {
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

  l_status = check_mappings_size_value(inference_map, "inference_map");
  if (l_status.get_code() == holoinfer_code::H_ERROR) { return l_status; }

  if (in_tensor_names.empty()) {
    status.set_message("Input tensor names cannot be empty");
    return status;
  }

  if (out_tensor_names.empty()) {
    status.set_message("Output tensor names cannot be empty");
    return status;
  }

  if (!check_equality(model_path_map.size(),
                      pre_processor_map.size(),
                      inference_map.size(),
                      out_tensor_names.size())) {
    status.set_message(
        "Size mismatch. model_path_map, pre_processor_map, "
        "inference_map, in_tensor_name, out_tensor_names must be of same size.");
    return status;
  }

  for (const auto& model_path : model_path_map) {
    // Check if keys in model_path_map and pre_processor_map are identical
    if (pre_processor_map.find(model_path.first) == pre_processor_map.end()) {
      status.set_message("Model keyword: " + model_path.first + " not in pre_processor_map");
      return status;
    } else {  // check that values in pre_processor_map exist in in_tensor_names
      if (std::find(in_tensor_names.begin(),
                    in_tensor_names.end(),
                    pre_processor_map.at(model_path.first)[0]) == in_tensor_names.end()) {
        status.set_message("Input Tensor name does not contain: " +
                           pre_processor_map.at(model_path.first)[0]);

        return status;
      }
    }
    if (inference_map.find(model_path.first) == inference_map.end()) {
      status.set_message("Model keyword: " + model_path.first + " not in inference_map");
      return status;
    } else {  // check that values in inference_map exist in out_tensor_names
      if (std::find(out_tensor_names.begin(),
                    out_tensor_names.end(),
                    inference_map.at(model_path.first)) == out_tensor_names.end()) {
        status.set_message("Output Tensor name does not contain: " +
                           inference_map.at(model_path.first));

        return status;
      }
    }
  }

  return InferStatus();
}

InferStatus multiai_processor_validity_check(const Mappings& processed_map,
                                             const std::vector<std::string>& in_tensor_names,
                                             const std::vector<std::string>& out_tensor_names) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);

  if (in_tensor_names.empty()) {
    status.set_message("Input tensor names cannot be empty");
    return status;
  }

  if (out_tensor_names.empty()) {
    status.set_message("WARNING: Output tensor names empty");
  } else {
    auto l_status = check_mappings_size_value(processed_map, "processed_map");
    if (l_status.get_code() == holoinfer_code::H_ERROR) { return l_status; }
  }

  if (!check_equality(processed_map.size(), out_tensor_names.size())) {
    status.set_message("Size mismatch. processed_map, out_tensor_names must be of same size.");
    return status;
  }

  return InferStatus();
}

}  // namespace inference
}  // namespace holoscan
