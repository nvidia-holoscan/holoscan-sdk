/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoinfer.hpp>
#include <holoinfer_buffer.hpp>
#include <holoinfer_utils.hpp>
#include "gxf/std/tensor.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/utils/holoinfer_utils.hpp"

namespace HoloInfer = holoscan::inference;

namespace holoscan::utils {

GxfTensorBuffer::GxfTensorBuffer(const holoscan::gxf::Entity& entity,
                                 const nvidia::gxf::Handle<nvidia::gxf::Tensor>& tensor)
    : entity_(entity), tensor_(tensor) {}

void* GxfTensorBuffer::data() {
  return reinterpret_cast<void*>(tensor_->pointer());
}

size_t GxfTensorBuffer::size() const {
  return tensor_->element_count();
}

size_t GxfTensorBuffer::get_bytes() const {
  return tensor_->bytes_size();
}

void GxfTensorBuffer::resize(size_t /*number_of_elements*/) {
  throw std::runtime_error("Resizing of GxfTensorBuffer is not supported");
}

gxf_result_t extract_data(const std::shared_ptr<HoloInfer::DataBuffer>& db,
                          nvidia::gxf::MemoryStorageType to,
                          nvidia::gxf::MemoryStorageType storage_type,
                          HoloInfer::holoinfer_datatype dtype, void* in_tensor_data,
                          size_t buffer_size, const std::string& module, cudaStream_t cstream) {
  if (to == nvidia::gxf::MemoryStorageType::kHost) {
    db->host_buffer_->resize(buffer_size);
  } else if (to == nvidia::gxf::MemoryStorageType::kDevice) {
    db->device_buffer_->resize(buffer_size);
  }

  if (to == nvidia::gxf::MemoryStorageType::kHost) {
    auto in_tensor_ptr = db->host_buffer_->data();

    if (storage_type == nvidia::gxf::MemoryStorageType::kDevice) {
      cudaError_t cuda_result = cudaMemcpyAsync(in_tensor_ptr,
                                                static_cast<const void*>(in_tensor_data),
                                                buffer_size * get_element_size(dtype),
                                                cudaMemcpyDeviceToHost,
                                                cstream);
      if (cuda_result != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Data extraction, error in DtoH cudaMemcpy: {}",
                           cudaGetErrorString(cuda_result));
        return HoloInfer::report_error(module, "Data extraction, DtoH cudaMemcpy.");
      }
      // When copying from device memory to pagable memory the call is synchronous with the host
      // execution. No need to synchronize here.
    } else if (storage_type == nvidia::gxf::MemoryStorageType::kHost) {
      memcpy(in_tensor_ptr, in_tensor_data, buffer_size * get_element_size(dtype));
    }
  } else {
    if (to == nvidia::gxf::MemoryStorageType::kDevice) {
      void* device_buff = db->device_buffer_->data();
      if (storage_type == nvidia::gxf::MemoryStorageType::kDevice) {
        cudaError_t cuda_result = cudaMemcpyAsync(static_cast<void*>(device_buff),
                                                  static_cast<const void*>(in_tensor_data),
                                                  buffer_size * get_element_size(dtype),
                                                  cudaMemcpyDeviceToDevice,
                                                  cstream);

        if (cuda_result != cudaSuccess) {
          HOLOSCAN_LOG_ERROR("Data transmission (DtoD) failed: {}",
                             cudaGetErrorString(cuda_result));
          return HoloInfer::report_error(module, "Data extraction, DtoD cudaMemcpy");
        }
      } else {
        return HoloInfer::report_error(module, "Data extraction parameters not supported.");
      }
    }
  }
  return GXF_SUCCESS;
}

gxf_result_t get_data_per_model(InputContext& op_input, const std::vector<std::string>& in_tensors,
                                HoloInfer::DataMap& data_per_input_tensor,
                                std::map<std::string, std::vector<int>>& dims_per_tensor,
                                bool cuda_buffer_out, const std::string& module,
                                gxf_context_t& context, CudaStreamHandler& cuda_stream_handler) {
  try {
    HoloInfer::TimePoint s_time, e_time;
    HoloInfer::timer_init(s_time);

    nvidia::gxf::MemoryStorageType to = nvidia::gxf::MemoryStorageType::kHost;

    if (cuda_buffer_out) { to = nvidia::gxf::MemoryStorageType::kDevice; }

    auto messages = op_input.receive<std::vector<holoscan::gxf::Entity>>("receivers").value();
    // get the CUDA stream from the input messages
    if (cuda_stream_handler.from_messages(context, messages) != GXF_SUCCESS) {
      throw std::runtime_error("Failed to get the CUDA stream from incoming messages");
    }
    const cudaStream_t cstream = cuda_stream_handler.get_cuda_stream(context);
    for (unsigned int i = 0; i < in_tensors.size(); ++i) {
      HOLOSCAN_LOG_DEBUG("Extracting data from tensor {}", in_tensors[i]);
      nvidia::gxf::Expected<nvidia::gxf::Handle<nvidia::gxf::Tensor>> maybe_in_tensor =
          nvidia::gxf::Unexpected{GXF_UNINITIALIZED_VALUE};
      size_t message_index;
      for (unsigned int j = 0; j < messages.size(); ++j) {
        maybe_in_tensor =
            messages[j].nvidia::gxf::Entity::get<nvidia::gxf::Tensor>(in_tensors[i].c_str());
        if (maybe_in_tensor) {
          // break out if the expected tensor name was found in this message
          message_index = j;
          break;
        }
      }
      if (!maybe_in_tensor) {
        return HoloInfer::report_error(module,
                                       "Data extraction, Tensor " + in_tensors[i] + " not found");
      }

      const auto& in_tensor = maybe_in_tensor.value();
      const auto storage_type = in_tensor->storage_type();

      if (storage_type != nvidia::gxf::MemoryStorageType::kHost &&
          storage_type != nvidia::gxf::MemoryStorageType::kDevice) {
        return HoloInfer::report_error(
            module, "Data extraction, memory not resident on CUDA pinned host memory or GPU.");
      }
      if (to != nvidia::gxf::MemoryStorageType::kHost &&
          to != nvidia::gxf::MemoryStorageType::kDevice) {
        return HoloInfer::report_error(module,
                                       "Input storage type in data extraction not supported.");
      }

      HoloInfer::holoinfer_datatype dtype;
      const auto element_type = in_tensor->element_type();
      switch (element_type) {
        case nvidia::gxf::PrimitiveType::kFloat32:
          dtype = HoloInfer::holoinfer_datatype::h_Float32;
          break;
        case nvidia::gxf::PrimitiveType::kInt32:
          dtype = HoloInfer::holoinfer_datatype::h_Int32;
          break;
        case nvidia::gxf::PrimitiveType::kInt8:
          dtype = HoloInfer::holoinfer_datatype::h_Int8;
          break;
        case nvidia::gxf::PrimitiveType::kInt64:
          dtype = HoloInfer::holoinfer_datatype::h_Int64;
          break;
        case nvidia::gxf::PrimitiveType::kUnsigned8:
          dtype = HoloInfer::holoinfer_datatype::h_UInt8;
          break;
        default: {
          HOLOSCAN_LOG_INFO("Incoming tensors must be of type: float, int32, int64, int8, uint8");
          return HoloInfer::report_error(module,
                                         "Data extraction, data type not supported in extraction.");
        }
      }

      auto data_map = data_per_input_tensor.find(in_tensors[i]);
      if (data_map == data_per_input_tensor.end()) {
        data_map = data_per_input_tensor
                       .insert({in_tensors[i], std::make_shared<HoloInfer::DataBuffer>(dtype)})
                       .first;
      }
      auto& db = data_map->second;

      std::vector<int> dims;
      for (unsigned int di = 0; di < in_tensor->shape().rank(); ++di) {
        dims.push_back(in_tensor->shape().dimension(di));
      }
      dims_per_tensor[in_tensors[i]] = std::move(dims);

      if (to == storage_type) {
        auto buffer =
            std::make_shared<GxfTensorBuffer>(messages[message_index], in_tensor);
        if (to == nvidia::gxf::MemoryStorageType::kDevice) {
          db->device_buffer_ = buffer;
        } else {
          db->host_buffer_ = buffer;
        }
      } else {
        gxf_result_t status = extract_data(db,
                                           to,
                                           storage_type,
                                           dtype,
                                           in_tensor->pointer(),
                                           in_tensor->element_count(),
                                           module,
                                           cstream);
        if (status != GXF_SUCCESS) {
          return HoloInfer::report_error(module, "Data extraction, In tensor extraction failed.");
        }
      }
    }

    HoloInfer::timer_init(e_time);
    HoloInfer::timer_check(s_time, e_time, module);
  } catch (std::exception& _ex) {
    return HoloInfer::report_error(module, "Data extraction, Message: " + std::string(_ex.what()));
  } catch (...) { return HoloInfer::report_error(module, "Data extraction, Unknown exception"); }
  return GXF_SUCCESS;
}

template <typename T>
gxf_result_t transmit_data(nvidia::gxf::MemoryStorageType from, nvidia::gxf::MemoryStorageType to,
                           nvidia::gxf::Expected<nvidia::gxf::Entity>& out_message,
                           const std::string& current_tensor, nvidia::gxf::Shape& output_shape,
                           size_t& buffer_size, HoloInfer::DataMap& input_data_map,
                           const nvidia::gxf::Handle<nvidia::gxf::Allocator>& allocator_,
                           const std::string& module, cudaStream_t cstream) {
  auto out_tensor = out_message.value().add<nvidia::gxf::Tensor>(current_tensor.c_str());
  if (!out_tensor)
    return HoloInfer::report_error(module, "Data transmission, Out tensor allocation.");

  if (from == nvidia::gxf::MemoryStorageType::kHost) {
    if (to == nvidia::gxf::MemoryStorageType::kHost) {
      out_tensor.value()->reshape<T>(
          output_shape, nvidia::gxf::MemoryStorageType::kHost, allocator_);
      if (!out_tensor.value()->pointer())
        return HoloInfer::report_error(module, "Data transmission, Out tensor buffer allocation.");

      nvidia::gxf::Expected<T*> out_tensor_data = out_tensor.value()->data<T>();
      if (!out_tensor_data)
        return HoloInfer::report_error(module, "Data transmission, Getting out tensor data.");

      auto current_model_output = input_data_map.at(current_tensor);
      memcpy(out_tensor_data.value(),
             current_model_output->host_buffer_->data(),
             buffer_size * sizeof(T));
    } else {  // to is on device
      out_tensor.value()->reshape<T>(
          output_shape, nvidia::gxf::MemoryStorageType::kDevice, allocator_);
      if (!out_tensor.value()->pointer())
        return HoloInfer::report_error(module, "Data transmission, Out tensor buffer allocation.");

      nvidia::gxf::Expected<T*> out_tensor_data = out_tensor.value()->data<T>();
      if (!out_tensor_data)
        return HoloInfer::report_error(module, "Data transmission, Getting out tensor data.");

      auto current_model_dev_buff = input_data_map.at(current_tensor)->host_buffer_->data();
      cudaError_t cuda_result = cudaMemcpyAsync(static_cast<void*>(out_tensor_data.value()),
                                                static_cast<const void*>(current_model_dev_buff),
                                                buffer_size * sizeof(T),
                                                cudaMemcpyHostToDevice,
                                                cstream);
      if (cuda_result != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("Data transmission (HtoD) failed: {}", cudaGetErrorString(cuda_result));
        return HoloInfer::report_error(module, "Data Transmission, HtoD cudaMemcpy.");
      }
      // When copying from pagable memory to device memory cudaMemcpyAsync() is copying the memory
      // to staging memory first and therefore is synchronous with the host execution. No need to
      // synchronize here.
    }
  } else {
    if (from == nvidia::gxf::MemoryStorageType::kDevice) {
      if (to == nvidia::gxf::MemoryStorageType::kDevice) {
        out_tensor.value()->reshape<T>(
            output_shape, nvidia::gxf::MemoryStorageType::kDevice, allocator_);
        if (!out_tensor.value()->pointer())
          return HoloInfer::report_error(module, "Data transmission, out tensor allocation.");

        nvidia::gxf::Expected<T*> out_tensor_data = out_tensor.value()->data<T>();
        if (!out_tensor_data)
          return HoloInfer::report_error(module, "Data Transmission, getting out tensor data.");

        void* current_model_dev_buff = input_data_map.at(current_tensor)->device_buffer_->data();
        cudaError_t cuda_result = cudaMemcpyAsync(static_cast<void*>(out_tensor_data.value()),
                                                  static_cast<const void*>(current_model_dev_buff),
                                                  buffer_size * sizeof(T),
                                                  cudaMemcpyDeviceToDevice,
                                                  cstream);
        if (cuda_result != cudaSuccess) {
          HOLOSCAN_LOG_ERROR("Data transmission (DtoD) failed: {}",
                             cudaGetErrorString(cuda_result));
          return HoloInfer::report_error(module, "Data extraction, DtoD cudaMemcpy.");
        }
      } else {  // to is on host
        out_tensor.value()->reshape<T>(
            output_shape, nvidia::gxf::MemoryStorageType::kHost, allocator_);
        if (!out_tensor.value()->pointer())
          return HoloInfer::report_error(module, "Data Transmission, out tensor allocation");

        nvidia::gxf::Expected<T*> out_tensor_data = out_tensor.value()->data<T>();
        if (!out_tensor_data)
          return HoloInfer::report_error(module, "Data Transmission, getting out tensor data");

        void* current_model_dev_buff = input_data_map.at(current_tensor)->device_buffer_->data();
        cudaError_t cuda_result = cudaMemcpyAsync(static_cast<void*>(out_tensor_data.value()),
                                                  static_cast<const void*>(current_model_dev_buff),
                                                  buffer_size * sizeof(T),
                                                  cudaMemcpyDeviceToHost,
                                                  cstream);
        if (cuda_result != cudaSuccess) {
          HOLOSCAN_LOG_ERROR("Data transmission (DtoH) failed: {}",
                             cudaGetErrorString(cuda_result));
          return HoloInfer::report_error(module, "Data transmission, DtoH cudaMemcpy");
        }
        // When copying from device memory to pagable memory the call is synchronous with the host
        // execution. No need to synchronize here.
      }
    }
  }
  return GXF_SUCCESS;
}

gxf_result_t transmit_data_per_model(gxf_context_t& cont,
                                     const HoloInfer::MultiMappings& model_to_tensor_map,
                                     HoloInfer::DataMap& input_data_map, OutputContext& op_output,
                                     std::vector<std::string>& out_tensors,
                                     HoloInfer::DimType& tensor_out_dims_map, bool cuda_buffer_in,
                                     bool cuda_buffer_out,
                                     const nvidia::gxf::Handle<nvidia::gxf::Allocator>& allocator_,
                                     const std::string& module,
                                     CudaStreamHandler& cuda_stream_handler) {
  HoloInfer::TimePoint s_time, e_time;
  HoloInfer::timer_init(s_time);
  try {
    nvidia::gxf::MemoryStorageType from = nvidia::gxf::MemoryStorageType::kHost;
    nvidia::gxf::MemoryStorageType to = nvidia::gxf::MemoryStorageType::kHost;

    if (cuda_buffer_in) { from = nvidia::gxf::MemoryStorageType::kDevice; }
    if (cuda_buffer_out) { to = nvidia::gxf::MemoryStorageType::kDevice; }

    HoloInfer::TimePoint s_time, e_time;
    HoloInfer::timer_init(s_time);

    // single transmitter used
    auto out_message = nvidia::gxf::Entity::New(cont);
    if (!out_message) {
      return HoloInfer::report_error(module, "Data transmission, Out message allocation");
    }

    // to combine static and dynamic I/O tensors, existing out_tensors must be checked for every
    // tensor in input_data_map
    for (const auto& dtensor : input_data_map) {
      if (std::find(out_tensors.begin(), out_tensors.end(), dtensor.first) == out_tensors.end()) {
        out_tensors.push_back(dtensor.first);
      }
    }

    for (unsigned int i = 0; i < out_tensors.size(); ++i) {
      if (input_data_map.find(out_tensors[i]) == input_data_map.end()) {
        return HoloInfer::report_error(
            module, "Data Transmission, Mapped data not found for " + out_tensors[i]);
      }
      auto current_out_tensor = out_tensors[i];

      // key_name is the model and tensor_index is the mapped tensor index in the
      // model_to_tensor_map.
      std::string key_name = "";
      unsigned int tensor_index = 0;

      // tensor_index always 0 for dynamic case. Correct index must be searched for in case where
      // tensor declarations are not in sequences or are more than 1.
      if (model_to_tensor_map.size() > 0) {
        // The right index of the tensor from the model_to_tensor_map is computed.
        // User populates model_to_tensor_map in the parameter set of the application
        //  - For inference:
        //      - key is model name, mapped to vector of output tensor names.
        //      - 'inference_map' in the parameter set
        //  - For processing: key is a tensor, mapped to processed tensors (from the operations)
        //      - 'processed_map' in the parameter set
        for (const auto& [key_to_tensor, tensor_names_vector] : model_to_tensor_map) {
          for (size_t a = 0; a < tensor_names_vector.size(); a++) {
            if (tensor_names_vector[a].compare(current_out_tensor) == 0) {
              key_name = key_to_tensor;
              tensor_index = a;
              break;
            }
          }

          if (key_name.length() != 0) { break; }
        }
      }

      if (key_name.length() == 0) {
        // this means key_name is not present in declared 'inference_map' or 'processed_map'
        // this is the case of dynamic messages when model_to_tensor_map is empty
        key_name = current_out_tensor;
      }

      if (tensor_out_dims_map.find(key_name) == tensor_out_dims_map.end())
        return HoloInfer::report_error(module,
                                       "Tensor mapping not found in dimension map for " + key_name);

      std::vector<int64_t> dims = tensor_out_dims_map.at(key_name)[tensor_index];

      nvidia::gxf::Shape output_shape;

      switch (dims.size()) {
        case 4: {
          std::array<int, 4> dimarray;
          for (size_t u = 0; u < 4; ++u) { dimarray[u] = (static_cast<int>(dims[u])); }
          nvidia::gxf::Shape out_shape(dimarray);
          output_shape = std::move(out_shape);
          break;
        }
        case 3: {
          std::array<int, 3> dimarray;
          for (size_t u = 0; u < 3; ++u) { dimarray[u] = (static_cast<int>(dims[u])); }
          nvidia::gxf::Shape out_shape(dimarray);
          output_shape = std::move(out_shape);
          break;
        }
        case 2: {
          nvidia::gxf::Shape out_shape{static_cast<int>(dims[0]), static_cast<int>(dims[1])};
          output_shape = std::move(out_shape);
          break;
        }
        case 1: {
          output_shape = nvidia::gxf::Shape{static_cast<int>(dims[0])};
          break;
        }
        default: {
          HOLOSCAN_LOG_INFO("Number of dimensions of each output tensor must be between 1 and 4.");
          return HoloInfer::report_error(
              module, "Output dimension size not supported. Size: " + std::to_string(dims.size()));
        }
      }

      size_t buffer_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
      auto tensor_dtype = input_data_map.at(current_out_tensor)->get_datatype();

      gxf_result_t stat = GXF_SUCCESS;
      switch (tensor_dtype) {
        case HoloInfer::holoinfer_datatype::h_Float32: {
          stat = transmit_data<float>(from,
                                      to,
                                      out_message,
                                      current_out_tensor,
                                      output_shape,
                                      buffer_size,
                                      input_data_map,
                                      allocator_,
                                      module,
                                      cuda_stream_handler.get_cuda_stream(cont));
          break;
        }
        case HoloInfer::holoinfer_datatype::h_Int32: {
          stat = transmit_data<int32_t>(from,
                                        to,
                                        out_message,
                                        current_out_tensor,
                                        output_shape,
                                        buffer_size,
                                        input_data_map,
                                        allocator_,
                                        module,
                                        cuda_stream_handler.get_cuda_stream(cont));
          break;
        }
        case HoloInfer::holoinfer_datatype::h_Int8: {
          stat = transmit_data<int8_t>(from,
                                       to,
                                       out_message,
                                       current_out_tensor,
                                       output_shape,
                                       buffer_size,
                                       input_data_map,
                                       allocator_,
                                       module,
                                       cuda_stream_handler.get_cuda_stream(cont));
          break;
        }
        case HoloInfer::holoinfer_datatype::h_UInt8: {
          stat = transmit_data<uint8_t>(from,
                                        to,
                                        out_message,
                                        current_out_tensor,
                                        output_shape,
                                        buffer_size,
                                        input_data_map,
                                        allocator_,
                                        module,
                                        cuda_stream_handler.get_cuda_stream(cont));
          break;
        }
        case HoloInfer::holoinfer_datatype::h_Int64: {
          stat = transmit_data<int64_t>(from,
                                        to,
                                        out_message,
                                        current_out_tensor,
                                        output_shape,
                                        buffer_size,
                                        input_data_map,
                                        allocator_,
                                        module,
                                        cuda_stream_handler.get_cuda_stream(cont));
          break;
        }
        default: {
          HOLOSCAN_LOG_INFO("Outgoing tensors must be of type: float, int32, int64, int8, uint8");
          HOLOSCAN_LOG_ERROR("Unsupported data type in HoloInfer data transmission.");
          stat = GXF_FAILURE;
        }
      }
      if (stat != GXF_SUCCESS) {
        return HoloInfer::report_error(
            module, "Data Transmission, Out tensor transmission failed for " + current_out_tensor);
      }
    }

    if (cuda_buffer_out) {
      // pass the CUDA stream to the output message
      gxf_result_t stream_handler_result = cuda_stream_handler.to_message(out_message);
      if (stream_handler_result != GXF_SUCCESS) {
        throw std::runtime_error("Failed to add the CUDA stream to the outgoing messages");
      }
    }

    // single transmitter used
    auto result = gxf::Entity(std::move(out_message.value()));
    op_output.emit(result);
    HoloInfer::timer_init(e_time);
  } catch (std::exception& _ex) {
    return HoloInfer::report_error(module,
                                   "Data transmission, Message: " + std::string(_ex.what()));
  } catch (...) { return HoloInfer::report_error(module, "Data transmission, Unknown exception"); }
  HoloInfer::timer_init(e_time);
  HoloInfer::timer_check(s_time, e_time, module + " Data transmission ");

  return GXF_SUCCESS;
}

}  // namespace holoscan::utils
