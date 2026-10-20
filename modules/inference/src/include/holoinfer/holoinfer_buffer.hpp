/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "holoinfer_constants.hpp"

namespace holoscan {
namespace inference {

/**
 * @brief Get the element size
 *
 * @param element_type Input data type. Float32 is the only supported element type.
 *
 * @return Bytes used in storing element type
 */
uint32_t get_element_size(holoinfer_datatype element_type) noexcept;

/**
 * @brief Cuda memory allocator Functor
 */
class DeviceAllocator {
 public:
  /// @brief Allocate `size` bytes of CUDA device memory into `*ptr`; returns true on success.
  bool operator()(void** ptr, size_t size) const;
};

/**
 * @brief Cuda memory de-allocator Functor
 */
class DeviceFree {
 public:
  /// @brief Free CUDA device memory previously allocated by DeviceAllocator.
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

  /// @brief Virtual destructor for polymorphic ownership through Buffer*.
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

  /// @brief Return a raw pointer to the device buffer.
  void* data() override;
  /// @brief Return the current logical size of the buffer in elements.
  size_t size() const override;
  /// @brief Return the allocated byte size of the buffer.
  size_t get_bytes() const override;
  /// @brief Resize the device buffer to hold at least `number_of_elements`.
  void resize(size_t number_of_elements) override;

 private:
  size_t size_{0}, capacity_{0};
  void* buffer_ = nullptr;
  DeviceAllocator allocator_;
  DeviceFree free_;
};

/// @brief Buffer specialization storing bytes on the host as a std::vector<byte>.
class HostBuffer : public Buffer {
 public:
  /**
   * @brief Constructor
   *
   * @param data_type  data type of the buffer
   */
  explicit HostBuffer(holoinfer_datatype data_type = holoinfer_datatype::h_Float32)
      : Buffer(data_type, -1) {}

  /// @brief Return a raw pointer to the host buffer.
  void* data() override;
  /// @brief Return the current logical size of the buffer in elements.
  size_t size() const override;
  /// @brief Return the allocated byte size of the buffer.
  size_t get_bytes() const override;
  /// @brief Resize the host buffer to hold at least `number_of_elements`.
  void resize(size_t number_of_elements) override;

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
   * @param data_type Data type, defaults to float32
   * @param device_id GPU device ID, defaults to 0
   */
  explicit DataBuffer(holoinfer_datatype data_type = holoinfer_datatype::h_Float32,
                      int device_id = 0);

  DataBuffer(const DataBuffer&) = delete;
  DataBuffer& operator=(const DataBuffer&) = delete;
  DataBuffer(DataBuffer&&) = delete;
  DataBuffer& operator=(DataBuffer&&) = delete;

  std::shared_ptr<Buffer> device_buffer_;
  std::shared_ptr<Buffer> host_buffer_;

  /// @brief Return the datatype of the underlying host buffer.
  holoinfer_datatype get_datatype() const { return host_buffer_->get_datatype(); }

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
  /// @brief Default construct with all fields at their in-class defaults.
  InferenceSpecs() = default;
  /**
   * @brief Constructor
   *
   * @param backend Backend inference (trt)
   * @param backend_map Backend inference map with model name as key, and backend as value
   * @param model_path_map Map with model name as key, path to model as value
   * @param input_map Map with model name as key, input tensor names in vector form as value
   * @param output_map Map with model name as key, output tensor names in vector form as value
   * @param dla_core_map Map with model name as key, DLA core index for inference as value
   * @param trt_opt_profile Vector of values for TensorRT optimization profile during engine
   * creation
   * @param dynamic_input_dims Input dimensions to the model is dynamic
   * @param is_engine_path Input path to model is trt engine
   * @param parallel_proc Perform parallel inference of multiple models
   * @param use_fp16 Use FP16 conversion, only supported for trt
   * @param cuda_buffer_in Input buffers on CUDA
   * @param cuda_buffer_out Output buffers on CUDA
   * @param use_cuda_graphs Use CUDA graphs, only supported for trt
   * @param dla_core The DLA core index to execute the engine on, only supported for trt. Set to -1
   * to disable DLA.
   * @param dla_gpu_fallback If DLA is enabled, use the GPU if a layer cannot be executed on DLA.
   * @param gpu_resident_inference Flag to enable GPU resident inference.
   * @param allocate_cuda_stream Function to allocate a CUDA stream (optional)
   * @param max_batch_size TRT engine-build ceiling on batch dimension. Default 256.
   * @param max_memory TRT engine-build workspace-memory ceiling in bytes. Default 10 GB.
   */
  InferenceSpecs(std::string backend, Mappings backend_map, Mappings model_path_map,
                 MultiMappings input_map, MultiMappings output_map, Mappings dla_core_map,
                 MultiMappings trt_opt_profile, bool dynamic_input_dims, bool is_engine_path,
                 bool parallel_proc, bool use_fp16, bool cuda_buffer_in, bool cuda_buffer_out,
                 bool use_cuda_graphs, int32_t dla_core, bool dla_gpu_fallback,
                 bool gpu_resident_inference,
                 std::function<cudaStream_t(int32_t)> allocate_cuda_stream,
                 int32_t max_batch_size = 256, std::size_t max_memory = 10'000'000'000ULL)
      : backend_type_(std::move(backend)),
        backend_map_(std::move(backend_map)),
        model_path_map_(std::move(model_path_map)),
        input_map_(std::move(input_map)),
        output_map_(std::move(output_map)),
        dla_core_map_(std::move(dla_core_map)),
        trt_opt_profile_(std::move(trt_opt_profile)),
        dynamic_input_dims_(dynamic_input_dims),
        is_engine_path_(is_engine_path),
        parallel_processing_(parallel_proc),
        use_fp16_(use_fp16),
        cuda_buffer_in_(cuda_buffer_in),
        cuda_buffer_out_(cuda_buffer_out),
        use_cuda_graphs_(use_cuda_graphs),
        dla_core_(dla_core),
        dla_gpu_fallback_(dla_gpu_fallback),
        gpu_resident_inference_(gpu_resident_inference),
        allocate_cuda_stream_(std::move(allocate_cuda_stream)),
        max_batch_size_(max_batch_size),
        max_memory_(max_memory) {}

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
   * @brief Get the DLA core map
   * @return Mappings data
   */
  Mappings get_dla_core_map() const { return dla_core_map_; }

  /// @brief Backend type (for all models)
  std::string backend_type_{""};

  /// @brief Backend map
  Mappings backend_map_;

  ///  @brief Map with key as model name and value as model file path
  Mappings model_path_map_;

  /// @brief Map with key as model name and value as vector of input tensor names
  MultiMappings input_map_;

  /// @brief Map with key as model name and value as inferred tensor name
  MultiMappings output_map_;

  /// @brief Map with key as model name and value as DLA core index for inference
  Mappings dla_core_map_;

  /// Map holding dimensions per tensor. Key is tensor name and value is a vector with
  /// dimensions.
  std::map<std::string, std::vector<int>> dims_per_tensor_;

  /// @brief TensorRT optimization profile during engine creation for dynamic inputs
  MultiMappings trt_opt_profile_;

  /// @brief Flag showing if the the inputs to the models are dynamic
  bool dynamic_input_dims_ = false;

  /// @brief Flag showing if input model path is path to engine files
  bool is_engine_path_ = false;

  ///  @brief Flag showing if inference on CUDA. Always true; TRT is CUDA-only.
  bool oncuda_ = true;

  ///  @brief Flag to enable parallel inference. Default is True.
  bool parallel_processing_ = false;

  ///  @brief Flag showing if trt engine file conversion will use FP16. Default is False.
  bool use_fp16_ = false;

  ///  @brief Flag showing if input buffers are on CUDA. Default is True.
  bool cuda_buffer_in_ = true;

  ///  @brief Flag showing if output buffers are on CUDA. Default is True.
  bool cuda_buffer_out_ = true;

  ///  @brief Flag showing if using CUDA Graphs. Default is True.
  bool use_cuda_graphs_ = true;

  /// @brief Pointer to GPU resident input buffer.
  void* gpu_resident_input_ = nullptr;

  /// @brief Pointer to GPU resident output buffer.
  void* gpu_resident_output_ = nullptr;

  /// @brief The DLA core index to execute the engine on, starts at 0. Set to -1 (the default) to
  /// disable DLA.
  int32_t dla_core_ = -1;

  /// @brief If DLA is enabled, use the GPU if a layer cannot be executed on DLA. If the fallback is
  /// disabled, engine creation will fail if a layer cannot executed on DLA.
  bool dla_gpu_fallback_ = true;

  /// @brief Flag showing if GPU resident inference is enabled. Default is False.
  bool gpu_resident_inference_ = false;

  /// @brief TRT engine-build ceiling on the batch dimension. Default 256; matches
  /// `NetworkOptions::max_batch_size` and is only consulted when building a fresh engine from
  /// ONNX (ignored for `is_engine_path_ = true`).
  int32_t max_batch_size_ = 256;

  /// @brief TRT engine-build workspace-memory ceiling in bytes. Default 10 GB; passed to TRT as
  /// the `kWORKSPACE` memory-pool limit during engine construction. Only consulted when
  /// building a fresh engine from ONNX.
  std::size_t max_memory_ = 10'000'000'000ULL;

  /// @brief Flag showing if fast multi-model is enabled.
  bool fast_multi_model_ = false;

  /// @brief Input Data Map with key as tensor name and value as DataBuffer
  DataMap data_per_tensor_;

  /// @brief Output Data Map with key as tensor name and value as DataBuffer
  DataMap output_per_model_;

  /// @brief Function to allocate a CUDA stream
  std::function<cudaStream_t(int32_t device_id)> allocate_cuda_stream_;

  /// @brief CUDA context derived from a CudaGreenContext. When non-null, TRT engine
  /// building runs within this context so that tactic selection is constrained to
  /// the assigned SM partition. Set by InferenceOp when a CudaGreenContext resource
  /// is present; nullptr means use the default primary context (all SMs).
  CUcontext build_cuda_context_ = nullptr;

  /// @brief SM count of the green context partition. When non-zero, embedded in the
  /// engine filename so full-GPU and partitioned engine builds do not collide.
  int32_t build_sm_count_ = 0;

  /// @brief Per-partition CUcontext handles derived from a CudaGreenContextPool, indexed by
  /// partition. Empty when no pool is present. Set by InferenceOp when a CudaGreenContextPool
  /// resource is configured; used by ManagerInfer::set_inference_params() to bind each
  /// multi-model fast-path TrtInfer instance (model index `m`) to partition `m % size()` so
  /// its runtime dispatch (TrtInfer::do_inference_on_stream()) actually executes within that
  /// SM partition's CUDA context, not just the default/build-time context.
  std::vector<CUcontext> green_context_partition_contexts_;

  /// @brief True exactly when InferenceOp::start()'s decide_capture_protection() chose
  /// CaptureProtectionOutcome::kGreenContextIsolated for this operator's configuration. When
  /// true, ManagerInfer's runtime dispatch is executing under real per-partition CUDA-context
  /// isolation, so the process-wide capture_serialization_mutex_ safety net (designed for the
  /// shared-context kMutexProtected case) is redundant and is skipped for performance.
  bool green_context_isolated_ = false;
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
