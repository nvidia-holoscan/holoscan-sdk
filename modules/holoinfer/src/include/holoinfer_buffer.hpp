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
#ifndef MODULES_HOLOINFER_SRC_INCLUDE_HOLOINFER_BUFFER_HPP
#define MODULES_HOLOINFER_SRC_INCLUDE_HOLOINFER_BUFFER_HPP

#include <cuda_runtime_api.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "holoinfer_constants.hpp"

#define _HOLOSCAN_EXTERNAL_API_ __attribute__((visibility("default")))

namespace holoscan {
namespace inference {

/**
 * @brief Get the element size
 *
 * @param element_type Input data type. Float32 is the only supported element type.
 *
 * @return Bytes used in storing element type
 */
uint32_t get_element_size(holoinfer_datatype t) noexcept;

/**
 * @brief Cuda memory allocator Functor
 */
class DeviceAllocator {
 public:
  bool operator()(void** ptr, size_t size) const;
};

/**
 * @brief Cuda memory de-allocator Functor
 */
class DeviceFree {
 public:
  void operator()(void* ptr) const;
};

/**
 * Base class for a buffer containing typed data.
 */
class Buffer {
 public:
  /**
   * @brief Construction with default type
   *
   * @param type Data type, defaults to float32
   * @param device_id GPU device ID, defaults to 0
   */
  explicit Buffer(holoinfer_datatype type = holoinfer_datatype::h_Float32, int device_id = 0)
      : type_(type), device_id_(device_id) {}

  virtual ~Buffer() = default;

  /**
   * @brief Get the data buffer
   *
   * @return Void pointer to the buffer
   */
  virtual void* data() = 0;

  /**
   * @brief Get the size of the allocated buffer in elements
   *
   * @return size in elements
   */
  virtual size_t size() const = 0;

  /**
   * @brief Get the bytes allocated
   *
   * @return allocated bytes
   */
  virtual size_t get_bytes() const = 0;

  /**
   * @brief Resize the underlying buffer, this is a no-op if the buffer is already large enough.
   *
   * @param number_of_elements Number of elements to be resized with
   */
  virtual void resize(size_t number_of_elements) = 0;

  /**
   * @brief Get the datatype
   *
   * @return datatype
   */
  holoinfer_datatype get_datatype() const { return type_; }

  /**
   * @brief Get the device ID
   *
   * @return device ID
   */
  int get_device() const { return device_id_; }

 protected:
  /// Datatype of the elements in the buffer
  holoinfer_datatype type_;
  /// Device ID
  int device_id_;
};

/**
 * @brief Cuda Device Buffer Class
 */
class DeviceBuffer : public Buffer {
 public:
  /**
   * @brief Construction with default type
   *
   * @param type Data type, defaults to float32
   * @param device_id GPU device ID, defaults to 0
   */
  explicit DeviceBuffer(holoinfer_datatype type = holoinfer_datatype::h_Float32, int device_id = 0);

  /**
   * @brief Construction with type and size
   *
   * @param size memory size to be allocated
   * @param type Data type to be allocated
   */
  DeviceBuffer(size_t size, holoinfer_datatype type);

  /**
   * @brief Destructor
   */
  ~DeviceBuffer();

  /// Buffer class virtual members implemented by this class
  ///@{
  void* data() override;
  size_t size() const override;
  size_t get_bytes() const override;
  void resize(size_t number_of_elements) override;
  ///@}

 private:
  size_t size_{0}, capacity_{0};
  void* buffer_ = nullptr;
  DeviceAllocator allocator_;
  DeviceFree free_;
};

class HostBuffer : public Buffer {
 public:
  /**
   * @brief Constructor
   *
   * @param data_type  data type of the buffer
   */
  explicit HostBuffer(holoinfer_datatype data_type = holoinfer_datatype::h_Float32)
      : Buffer(data_type, -1) {}

  /// Buffer class virtual members implemented by this class
  ///@{
  void* data() override;
  size_t size() const override;
  size_t get_bytes() const override;
  void resize(size_t number_of_elements) override;
  ///@}

  /// @brief Set the data type and resize the buffer
  /// @param in_type input data type
  void set_type(holoinfer_datatype in_type);

 private:
  /// @brief Data buffer on host, stored as a vector of bytes
  std::vector<byte> buffer_;
  /// @brief Number of elements in the buffer
  size_t number_of_elements_{0};
};

/**
 * @brief HoloInfer DataBuffer Class. Holds CPU based buffer and device buffer.
 */
class DataBuffer {
 public:
  /**
   * @brief Constructor
   *
   * @param type Data type, defaults to float32
   * @param device_id GPU device ID, defaults to 0
   */
  explicit DataBuffer(holoinfer_datatype data_type = holoinfer_datatype::h_Float32,
                      int device_id = 0);

  std::shared_ptr<Buffer> device_buffer_;
  std::shared_ptr<Buffer> host_buffer_;

  holoinfer_datatype get_datatype() const { return host_buffer_->get_datatype(); }

 private:
  /// Helper class for backwards compatibility, forwards calls to other buffer class.
  class BufferForward : public Buffer {
   public:
    explicit BufferForward(std::shared_ptr<Buffer>& buffer) : buffer_(buffer) {}
    BufferForward() = delete;

    /// Buffer class virtual members implemented by this class
    ///@{
    void* data() override { return buffer_->data(); }
    size_t size() const override { return buffer_->size(); }
    size_t get_bytes() const override { return buffer_->get_bytes(); }
    void resize(size_t number_of_elements) override { buffer_->resize(number_of_elements); }
    ///@}

   private:
    const std::shared_ptr<Buffer>& buffer_;
  };

 public:
  /// @deprecated since 2.6, use `device_buffer_` instead
  const std::shared_ptr<BufferForward> device_buffer{
      std::make_shared<BufferForward>(device_buffer_)};
  /// @deprecated since 2.6, use `host_buffer_` instead
  BufferForward host_buffer{host_buffer_};

 private:
  holoinfer_datatype data_type_;
};

using DataMap = std::map<std::string, std::shared_ptr<DataBuffer>>;
using Mappings = std::map<std::string, std::string>;
using DimType = std::map<std::string, std::vector<std::vector<int64_t>>>;
using MultiMappings = std::map<std::string, std::vector<std::string>>;

/**
 * @brief Struct that holds specifications related to inference, along with input and
 * output data buffer.
 */
struct InferenceSpecs {
  InferenceSpecs() = default;
  /**
   * @brief Constructor
   *
   * @param backend Backend inference (trt or onnxrt)
   * @param backend_map Backend inference map with model name as key, and backend as value
   * @param model_path_map Map with model name as key, path to model as value
   * @param pre_processor_map Map with model name as key, input tensor names in vector form as value
   * @param inference_map Map with model name as key, output tensor names in vector form as value
   * @param device_map Map with model name as key, GPU ID for inference as value
   * @param temporal_map Map with model name as key, frame number to skip for inference as value
   * @param activation_map Map with key as model name and activation state for inference as value
   * @param trt_opt_profile Vector of values for TensorRT optimization profile during engine
   * creation
   * @param is_engine_path Input path to model is trt engine
   * @param oncpu Perform inference on CPU
   * @param parallel_proc Perform parallel inference of multiple models
   * @param use_fp16 Use FP16 conversion, only supported for trt
   * @param cuda_buffer_in Input buffers on CUDA
   * @param cuda_buffer_out Output buffers on CUDA
   */
  InferenceSpecs(const std::string& backend, const Mappings& backend_map,
                 const Mappings& model_path_map, const MultiMappings& pre_processor_map,
                 const MultiMappings& inference_map, const Mappings& device_map,
                 const Mappings& temporal_map, const Mappings& activation_map,
                 const std::vector<int32_t>& trt_opt_profile, bool is_engine_path, bool oncpu,
                 bool parallel_proc, bool use_fp16, bool cuda_buffer_in, bool cuda_buffer_out)
      : backend_type_(backend),
        backend_map_(backend_map),
        model_path_map_(model_path_map),
        pre_processor_map_(pre_processor_map),
        inference_map_(inference_map),
        device_map_(device_map),
        temporal_map_(temporal_map),
        activation_map_(activation_map),
        trt_opt_profile_(trt_opt_profile),
        is_engine_path_(is_engine_path),
        oncuda_(!oncpu),
        parallel_processing_(parallel_proc),
        use_fp16_(use_fp16),
        cuda_buffer_in_(cuda_buffer_in),
        cuda_buffer_out_(cuda_buffer_out) {}

  /**
   * @brief Get the model data path map
   * @return Mappings data
   */
  Mappings get_path_map() const { return model_path_map_; }

  /**
   * @brief Get the model backend map
   * @return Mappings data
   */
  Mappings get_backend_map() const { return backend_map_; }

  /**
   * @brief Get the device map
   * @return Mappings data
   */
  Mappings get_device_map() const { return device_map_; }

  /**
   * @brief Get the Temporal map
   * @return Mappings data
   */
  Mappings get_temporal_map() const { return temporal_map_; }

  /**
   * @brief Get the Activation map
   * @return Mappings data
   */
  Mappings get_activation_map() const { return activation_map_; }

  /**
   * @brief Set the Activation map
   * @param activation_map Map that will be used to update the activation_map_ of specs.
   */
  void set_activation_map(const Mappings& activation_map) {
    for (const auto& [key, value] : activation_map) {
      if (activation_map_.find(key) != activation_map_.end()) { activation_map_.at(key) = value; }
    }
  }

  /// @brief Backend type (for all models)
  std::string backend_type_{""};

  /// @brief Backend map
  Mappings backend_map_;

  ///  @brief Map with key as model name and value as model file path
  Mappings model_path_map_;

  /// @brief Map with key as model name and value as vector of input tensor names
  MultiMappings pre_processor_map_;

  /// @brief Map with key as model name and value as inferred tensor name
  MultiMappings inference_map_;

  /// @brief Map with key as model name and value as GPU ID for inference
  Mappings device_map_;

  /// @brief Map with key as model name and frame number to skip for inference as value
  Mappings temporal_map_;

  /// @brief Map with key as model name and activation state for inference as value
  Mappings activation_map_;

  /// @brief TensorRT optimization profile during engine creation for dynamic inputs
  std::vector<int32_t> trt_opt_profile_;

  /// @brief Flag showing if input model path is path to engine files
  bool is_engine_path_ = false;

  ///  @brief Flag showing if inference on CUDA. Default is True.
  bool oncuda_ = true;

  ///  @brief Flag to enable parallel inference. Default is True.
  bool parallel_processing_ = false;

  ///  @brief Flag showing if trt engine file conversion will use FP16. Default is False.
  bool use_fp16_ = false;

  ///  @brief Flag showing if input buffers are on CUDA. Default is True.
  bool cuda_buffer_in_ = true;

  ///  @brief Flag showing if output buffers are on CUDA. Default is True.
  bool cuda_buffer_out_ = true;

  /// @brief Input Data Map with key as tensor name and value as DataBuffer
  DataMap data_per_tensor_;

  /// @brief Output Data Map with key as tensor name and value as DataBuffer
  DataMap output_per_model_;
};

/**
 * @brief Allocate buffer on host and device
 *
 * @param buffers Map with keyword as model name or tensor name, and value as DataBuffer. The map
 * is populated in this function.
 * @param dims Dimension of the allocation
 * @param datatype Data type of the buffer
 * @param keyname Storage name in the map against the created DataBuffer
 * @param allocate_cuda flag to allocate cuda buffer
 * @param device_id GPU ID to allocate buffers on
 * @return InferStatus with appropriate code and message
 */
InferStatus allocate_buffers(DataMap& buffers, std::vector<int64_t>& dims,
                             holoinfer_datatype datatype, const std::string& keyname,
                             bool allocate_cuda, int device_id);
}  // namespace inference
}  // namespace holoscan

#endif /* MODULES_HOLOINFER_SRC_INCLUDE_HOLOINFER_BUFFER_HPP */
