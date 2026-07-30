/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_UTILS_CUDA_BUFFER_HPP
#define HOLOSCAN_UTILS_CUDA_BUFFER_HPP

#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>

#include <holoscan/utils/cuda_macros.hpp>

/// The code is partially copied from modules/holoinfer/src/include/holoinfer_buffer.hpp
/// When time comes, we will merge these two implementations, so that not only holoinfer, but also
/// any other Holoscan SDK components can use these classes.

namespace holoscan {
namespace utils {
namespace cuda {

/**
 * @brief Data types supported by the buffer
 */
enum class BufferDataType {
  Float32 = 0,
  Int8 = 1,
  Int32 = 2,
  Int64 = 3,
  UInt8 = 4,
  Float16 = 5,
  Unsupported = 6
};

/**
 * @brief Get the element size in bytes for a given data type
 *
 * @param data_type The data type
 * @return Number of bytes per element
 */
inline uint32_t get_element_size(BufferDataType data_type) noexcept {
  switch (data_type) {
    case BufferDataType::Float32:
    case BufferDataType::Int32:
      return 4;
    case BufferDataType::Int64:
      return 8;
    case BufferDataType::Int8:
    case BufferDataType::UInt8:
      return 1;
    case BufferDataType::Float16:
      return 2;
    case BufferDataType::Unsupported:
      return 0;
  }
  return 0;
}

/**
 * @brief CUDA memory allocator functor
 */
class CudaAllocator {
 public:
  bool operator()(void** ptr, size_t size) const {
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(ptr, size), "Failed to allocate CUDA memory");
    return true;
  }
};

/**
 * @brief CUDA memory de-allocator functor
 */
class CudaFree {
 public:
  void operator()(void* ptr) const {
    if (ptr) {
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaFree(ptr), "Failed to free CUDA memory");
    }
  }
};

/**
 * @brief CUDA Host Mapped Allocator. This allocator allocates memory on the
 * host, which is also mapped to the device.
 *
 */
class CudaHostMappedAllocator {
 public:
  bool operator()(void** ptr, size_t size) const {
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaHostAlloc(ptr, size, cudaHostAllocMapped),
                                   "Failed to allocate CUDA host mapped memory");
    return true;
  }
};

/**
 * @brief A deallocator that frees page-locked host memory.
 *
 */
class CudaHostFree {
 public:
  void operator()(void* ptr) const {
    if (ptr) {
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaFreeHost(ptr), "Failed to free CUDA host memory");
    }
  }
};

/**
 * @brief Base class for a buffer containing typed data
 */
class Buffer {
 public:
  /**
   * @brief Constructor with default type
   *
   * @param data_type Data type, defaults to Float32
   * @param device_id GPU device ID, defaults to 0
   */
  explicit Buffer(BufferDataType data_type = BufferDataType::Float32, int device_id = 0)
      : data_type_(data_type), device_id_(device_id) {}

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
   * @brief Resize the underlying buffer
   *
   * @param number_of_elements Number of elements to be resized with
   */
  virtual void resize(size_t number_of_elements) = 0;

  /**
   * @brief Get the datatype
   *
   * @return datatype
   */
  BufferDataType get_datatype() const { return data_type_; }

  /**
   * @brief Get the device ID
   *
   * @return device ID
   */
  int get_device() const { return device_id_; }

 protected:
  /// Datatype of the elements in the buffer
  BufferDataType data_type_;
  /// Device ID
  int device_id_;
};

/**
 * @brief CUDA Device Buffer Class
 */
class DeviceBuffer : public Buffer {
 public:
  /**
   * @brief Construction with size
   *
   * @param size memory size to be allocated in bytes
   * @param device_id GPU device ID, defaults to 0
   */
  explicit DeviceBuffer(size_t size, int device_id = 0);

  /**
   * @brief Destructor
   */
  ~DeviceBuffer();

  // Delete copy operations to prevent double-free errors
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  // Delete move operations
  DeviceBuffer(DeviceBuffer&&) = delete;
  DeviceBuffer& operator=(DeviceBuffer&&) = delete;

  /// Buffer class virtual members implemented by this class
  ///@{
  void* data() override;
  size_t size() const override;
  size_t get_bytes() const override;
  void resize(size_t number_of_elements) override;
  ///@}

  /// Copies data from device to host after allocating the host memory
  void* host_data(cudaStream_t stream = 0);

 private:
  size_t size_{0}, capacity_{0};
  void* buffer_ = nullptr;
  void* host_buffer_ = nullptr;
  CudaAllocator allocator_;
  CudaFree free_;
};

/**
 * @brief CUDA Host Mapped Buffer Class
 *
 */
class CudaHostMappedBuffer : public Buffer {
 public:
  /**
   * @brief Construct a new Cuda Host Mapped Buffer object
   *
   * @param size
   * @param device_id
   * @param size memory size to be allocated in bytes
   * @param device_id GPU device ID, defaults to 0
   */
  explicit CudaHostMappedBuffer(size_t size, int device_id = 0);
  ~CudaHostMappedBuffer();
  void* data() override;
  size_t size() const override;
  size_t get_bytes() const override;
  void resize(size_t number_of_elements) override;

  void* device_data() const { return device_buffer_; }

 private:
  size_t size_{0}, capacity_{0};
  void *buffer_ = nullptr, *device_buffer_ = nullptr;
  CudaHostMappedAllocator allocator_;
  CudaHostFree free_;
};

}  // namespace cuda
}  // namespace utils
}  // namespace holoscan

#endif  // HOLOSCAN_UTILS_CUDA_BUFFER_HPP
