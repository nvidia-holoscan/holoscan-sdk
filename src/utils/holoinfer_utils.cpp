/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_UTILS_HOLOINFER_HPP
#define HOLOSCAN_UTILS_HOLOINFER_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "holoscan/utils/holoinfer_utils.hpp"
#include "holoscan/core/io_context.hpp"

#include "gxf/std/tensor.hpp"  // TODO: remove?

#include <holoinfer.hpp>
#include <holoinfer_utils.hpp>
#include <holoinfer_buffer.hpp>

namespace HoloInfer = holoscan::inference;

namespace holoscan::utils {

gxf_result_t multiai_get_data_per_model(InputContext& op_input,
                                        const std::vector<std::string>& in_tensors,
                                        HoloInfer::DataMap& data_per_input_tensor,
                                        std::map<std::string, std::vector<int>>& dims_per_tensor,
                                        bool cuda_buffer_out, const std::string& module) {
  try {
    HoloInfer::TimePoint s_time, e_time;
    HoloInfer::timer_init(s_time);

    nvidia::gxf::MemoryStorageType to = nvidia::gxf::MemoryStorageType::kHost;

    if (cuda_buffer_out) { to = nvidia::gxf::MemoryStorageType::kDevice; }

    auto messages = op_input.receive<std::vector<holoscan::gxf::Entity>>("receivers");
    for (unsigned int i = 0; i < in_tensors.size(); ++i) {
      // nvidia::gxf::Handle<nvidia::gxf::Tensor> in_tensor;
      std::shared_ptr<holoscan::Tensor> in_tensor;
      for (unsigned int j = 0; j < messages.size(); ++j) {
        const auto& in_message = messages[j];
        const auto maybe_tensor = in_message.get<holoscan::Tensor>(in_tensors[i].c_str(), false);
        if (maybe_tensor) {
          // break out if the expected tensor name was found in this message
          in_tensor = maybe_tensor;
          break;
        }
      }
      if (!in_tensor)
        return HoloInfer::report_error(module,
                                       "Data_per_tensor, Tensor " + in_tensors[i] + " not found");

      // convert from Tensor to GXFTensor so code below can be re-used as-is.
      // (otherwise cannot easily get element_type, storage_type)
      holoscan::gxf::GXFTensor in_tensor_gxf{in_tensor.get()->dl_ctx()};
      void* in_tensor_data = in_tensor_gxf.pointer();

      auto element_type = in_tensor_gxf.element_type();
      auto storage_type = in_tensor_gxf.storage_type();

      if (element_type != nvidia::gxf::PrimitiveType::kFloat32) {
        return HoloInfer::report_error(module, "Data extraction, element type not supported");
      }
      if (!(storage_type != nvidia::gxf::MemoryStorageType::kHost ||
            storage_type != nvidia::gxf::MemoryStorageType::kDevice)) {
        return HoloInfer::report_error(module,
                                       "Data extraction, memory not resident on CPU or GPU");
      }
      if (to != nvidia::gxf::MemoryStorageType::kHost &&
          to != nvidia::gxf::MemoryStorageType::kDevice) {
        return HoloInfer::report_error(module,
                                       "Input storage type in data extraction not supported");
      }

      std::vector<int> dims;
      for (unsigned int i = 0; i < in_tensor_gxf.shape().rank(); ++i)
        dims.push_back(in_tensor_gxf.shape().dimension(i));

      size_t buffer_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());

      if (data_per_input_tensor.find(in_tensors[i]) == data_per_input_tensor.end()) {
        auto db = std::make_shared<HoloInfer::DataBuffer>();
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
            return HoloInfer::report_error(module,
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
              return HoloInfer::report_error(module,
                                             "Data extraction, error in DtoD cudaMemcpy::" +
                                                 std::string(cudaGetErrorString(cuda_result)));
            }
          } else {
            return HoloInfer::report_error(module, "Data extraction parameters not supported.");
          }
        }
      }
    }

    HoloInfer::timer_init(e_time);
  } catch (std::exception& _ex) {
    return HoloInfer::report_error(module, "Data_per_tensor, Message: " + std::string(_ex.what()));
  } catch (...) { return HoloInfer::report_error(module, "Data_per_tensor, Unknown exception"); }
  return GXF_SUCCESS;
}

gxf_result_t multiai_transmit_data_per_model(
    gxf_context_t& cont, const HoloInfer::Mappings& model_to_tensor_map,
    HoloInfer::DataMap& input_data_map, OutputContext& op_output,
    const std::vector<std::string>& out_tensors, HoloInfer::DimType& tensor_out_dims_map,
    bool cuda_buffer_in, bool cuda_buffer_out, const nvidia::gxf::PrimitiveType& element_type,
    const nvidia::gxf::Handle<nvidia::gxf::Allocator>& allocator_, const std::string& module) {
  try {
    if (element_type != nvidia::gxf::PrimitiveType::kFloat32) {
      return HoloInfer::report_error(module, "Element type to be transmitted not supported");
    }

    nvidia::gxf::MemoryStorageType from = nvidia::gxf::MemoryStorageType::kHost;
    nvidia::gxf::MemoryStorageType to = nvidia::gxf::MemoryStorageType::kHost;

    if (cuda_buffer_in) { from = nvidia::gxf::MemoryStorageType::kDevice; }
    if (cuda_buffer_out) { to = nvidia::gxf::MemoryStorageType::kDevice; }

    HoloInfer::TimePoint s_time, e_time;
    HoloInfer::timer_init(s_time);

    // single transmitter used
    auto out_message = nvidia::gxf::Entity::New(cont);
    if (!out_message) {
      return HoloInfer::report_error(module, "Inference Toolkit, Out message allocation");
    }

    for (unsigned int i = 0; i < out_tensors.size(); ++i) {
      if (input_data_map.find(out_tensors[i]) == input_data_map.end())
        return HoloInfer::report_error(
            module, "Inference Toolkit, Mapped data not found for " + out_tensors[i]);

      std::string key_name{""};

      for (const auto& key_to_tensor : model_to_tensor_map) {
        if (key_to_tensor.second.compare(out_tensors[i]) == 0) key_name = key_to_tensor.first;
      }
      if (key_name.length() == 0)
        return HoloInfer::report_error(module, "Tensor mapping not found in model to tensor map");

      if (tensor_out_dims_map.find(key_name) == tensor_out_dims_map.end())
        return HoloInfer::report_error(module,
                                       "Tensor mapping not found in dimension map for " + key_name);

      auto out_tensor = out_message.value().add<nvidia::gxf::Tensor>(out_tensors[i].c_str());
      if (!out_tensor)
        return HoloInfer::report_error(module, "Inference Toolkit, Out tensor allocation");

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
            return HoloInfer::report_error(module,
                                           "Inference Toolkit, Out tensor buffer allocation");

          nvidia::gxf::Expected<float*> out_tensor_data = out_tensor.value()->data<float>();
          if (!out_tensor_data)
            return HoloInfer::report_error(module, "Inference Toolkit, Getting out tensor data");

          auto current_model_output = input_data_map.at(out_tensors[i]);
          memcpy(out_tensor_data.value(),
                 current_model_output->host_buffer.data(),
                 buffer_size * sizeof(float));
        } else {  // to is on device
          out_tensor.value()->reshape<float>(
              output_shape, nvidia::gxf::MemoryStorageType::kDevice, allocator_);
          if (!out_tensor.value()->pointer())
            return HoloInfer::report_error(module,
                                           "Inference Toolkit, Out tensor buffer allocation");

          nvidia::gxf::Expected<float*> out_tensor_data = out_tensor.value()->data<float>();
          if (!out_tensor_data)
            return HoloInfer::report_error(module, "Inference Toolkit, Getting out tensor data");

          auto current_model_dev_buff = input_data_map.at(out_tensors[i])->host_buffer.data();
          cudaError_t cuda_result = cudaMemcpy(static_cast<void*>(out_tensor_data.value()),
                                               static_cast<const void*>(current_model_dev_buff),
                                               buffer_size * sizeof(float),
                                               cudaMemcpyHostToDevice);
          if (cuda_result != cudaSuccess) {
            return HoloInfer::report_error("Inference Toolkit",
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
              return HoloInfer::report_error(module,
                                             "Inference Toolkit, Out tensor buffer allocation");

            nvidia::gxf::Expected<float*> out_tensor_data = out_tensor.value()->data<float>();
            if (!out_tensor_data)
              return HoloInfer::report_error(module, "Inference Toolkit, Getting out tensor data");

            void* current_model_dev_buff = input_data_map.at(out_tensors[i])->device_buffer->data();
            cudaError_t cuda_result = cudaMemcpy(static_cast<void*>(out_tensor_data.value()),
                                                 static_cast<const void*>(current_model_dev_buff),
                                                 buffer_size * sizeof(float),
                                                 cudaMemcpyDeviceToDevice);
            if (cuda_result != cudaSuccess) {
              return HoloInfer::report_error("Inference Toolkit",
                                             "Data extraction, error in DtoD cudaMemcpy::" +
                                                 std::string(cudaGetErrorString(cuda_result)));
            }
          } else {  // to is on host
            out_tensor.value()->reshape<float>(
                output_shape, nvidia::gxf::MemoryStorageType::kHost, allocator_);
            if (!out_tensor.value()->pointer())
              return HoloInfer::report_error(module,
                                             "Inference Toolkit, Out tensor buffer allocation");

            nvidia::gxf::Expected<float*> out_tensor_data = out_tensor.value()->data<float>();
            if (!out_tensor_data)
              return HoloInfer::report_error(module, "Inference Toolkit, Getting out tensor data");

            void* current_model_dev_buff = input_data_map.at(out_tensors[i])->device_buffer->data();
            cudaError_t cuda_result = cudaMemcpy(static_cast<void*>(out_tensor_data.value()),
                                                 static_cast<const void*>(current_model_dev_buff),
                                                 buffer_size * sizeof(float),
                                                 cudaMemcpyDeviceToHost);
            if (cuda_result != cudaSuccess) {
              return HoloInfer::report_error("Inference Toolkit",
                                             "Data extraction, error in DtoH cudaMemcpy::" +
                                                 std::string(cudaGetErrorString(cuda_result)));
            }
          }
        }
      }
    }

    // single transmitter used
    auto result = gxf::Entity(std::move(out_message.value()));
    op_output.emit(result);
    HoloInfer::timer_init(e_time);
  } catch (std::exception& _ex) {
    return HoloInfer::report_error(module,
                                   "Data_transmission_tensor, Message: " + std::string(_ex.what()));
  } catch (...) {
    return HoloInfer::report_error(module, "Data_transmission_tensor, Unknown exception");
  }
  return GXF_SUCCESS;
}

}  // namespace holoscan::utils

#endif /* HOLOSCAN_UTILS_HOLOINFER_HPP */
