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
#ifndef _HOLOSCAN_INFER_BUFFER_H
#define _HOLOSCAN_INFER_BUFFER_H

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
 * @returns Bytes used in storing element type
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
 * @brief Cuda Device Buffer Class
 */
class DeviceBuffer {
 public:
  /**
   * @brief Construction with default type
   *
   * @param type Data type, defaults to float32
   */
  explicit DeviceBuffer(holoinfer_datatype type = holoinfer_datatype::hFloat);

  /**
   * @brief Construction with type and size
   *
   * @param size memory size to be allocated
   * @param type Data type to be allocated
   */
  DeviceBuffer(size_t size, holoinfer_datatype type);

  /**
   * @brief Get the data buffer
   *
   * @returns Void pointer to the buffer
   */
  void* data();

  /**
   * @brief Get the size of the allocated buffer
   *
   * @returns size
   */
  size_t size() const;

  /**
   * @brief Get the bytes allocated
   *
   * @returns allocated bytes
   */
  size_t get_bytes() const;

  /**
   * @brief Resize the underlying buffer
   *
   * @param element_size Size to be resized with
   */
  void resize(size_t element_size);

  /**
   * @brief Destructor
   */
  ~DeviceBuffer();

 private:
  size_t size_{0}, capacity_{0};
  holoinfer_datatype type_ = holoinfer_datatype::hFloat;
  void* buffer_ = nullptr;
  DeviceAllocator allocator_;
  DeviceFree free_;
};

/**
 * @brief Multi AI DataBuffer Class. Holds CPU based buffer as float32 vector and device buffer as a
 * shared pointer.
 */
class DataBuffer {
 public:
  /**
   * @brief Constructor
   */
  DataBuffer();
  std::shared_ptr<DeviceBuffer> device_buffer;
  std::vector<float> host_buffer;
};

using DataMap = std::map<std::string, std::shared_ptr<DataBuffer>>;
using Mappings = std::map<std::string, std::string>;
using DimType = std::map<std::string, std::vector<int64_t>>;
using MultiMappings = std::map<std::string, std::vector<std::string>>;

/**
 * @brief Struct that holds specifications related to Multi AI inference, along with input and
 * output data buffer.
 */
struct MultiAISpecs {
  MultiAISpecs() = default;
  /**
   * @brief Constructor
   *
   * @param backend Backend inference (trt or onnxrt)
   * @param model_path_map Map with model name as key, path to model as value
   * @param inference_map Map with model name as key, output tensor name as value
   * @param oncpu Perform inference on CPU
   * @param parallel_proc Perform parallel inference of multiple models
   * @param use_fp16 Use FP16 conversion, only supported for trt
   * @param cuda_buffer_in Input buffers on CUDA
   * @param cuda_buffer_out Output buffers on CUDA
   */
  MultiAISpecs(const std::string& backend, const Mappings& model_path_map,
               const Mappings& inference_map, bool is_engine_path, bool oncpu, bool parallel_proc,
               bool use_fp16, bool cuda_buffer_in, bool cuda_buffer_out)
      : backend_type_(backend),
        model_path_map_(model_path_map),
        inference_map_(inference_map),
        is_engine_path_(is_engine_path),
        oncuda_(!oncpu),
        parallel_processing_(parallel_proc),
        use_fp16_(use_fp16),
        cuda_buffer_in_(cuda_buffer_in),
        cuda_buffer_out_(cuda_buffer_out) {}

  /**
   * @brief Get the model data path map
   *
   * @return Mappings data
   */
  Mappings get_path_map() const { return model_path_map_; }

  std::string backend_type_{"trt"};

  ///  @brief Map with key as model name and value as model file path
  Mappings model_path_map_;

  /// @brief Map with key as model name and value as inferred tensor name
  Mappings inference_map_;

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

  /// @brief Input Data Map with key as model name and value as DataBuffer
  DataMap data_per_model_;

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
 * @param keyname Storage name in the map against the created DataBuffer
 * @returns InferStatus with appropriate code and message
 */
InferStatus allocate_host_device_buffers(DataMap& buffers, std::vector<int64_t>& dims_map,
                                         const std::string& mappings);

/**
 * @brief Allocate buffer on host
 *
 * @param buffers Map with keyword as model name or tensor name, and value as DataBuffer. The map
 * is populated in this function.
 * @param dims Dimension of the allocation
 * @param keyname Storage name in the map against the created DataBuffer
 * @returns InferStatus with appropriate code and message
 */
InferStatus allocate_host_buffers(DataMap& buffers, std::vector<int64_t>& dims,
                                  const std::string& keyname);
}  // namespace inference
}  // namespace holoscan

#endif
