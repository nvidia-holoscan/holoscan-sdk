/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <stdexcept>

#include "holoscan/utils/cuda_macros.hpp"
#include "holoscan/utils/cuda/buffer.hpp"

namespace holoscan {
namespace utils {
namespace cuda {

DeviceBuffer::DeviceBuffer(size_t size, int device_id)
    : Buffer(BufferDataType::UInt8, device_id), size_(size), capacity_(size) {
  if (!allocator_(&buffer_, size)) {
    throw std::bad_alloc();
  }
}

void* DeviceBuffer::data() {
  return buffer_;
}

size_t DeviceBuffer::size() const {
  return size_;
}

size_t DeviceBuffer::get_bytes() const {
  return size_;
}

void DeviceBuffer::resize(size_t number_of_elements) {
  size_ = number_of_elements;
  if (capacity_ < number_of_elements) {
    free_(buffer_);
    if (!allocator_(&buffer_, size_)) {
      throw std::bad_alloc{};
    }
    capacity_ = number_of_elements;
  }
}

DeviceBuffer::~DeviceBuffer() {
  free_(buffer_);
}

CudaHostMappedBuffer::CudaHostMappedBuffer(size_t size, int device_id)
    : Buffer(BufferDataType::UInt8, device_id), size_(size), capacity_(size) {
  if (!allocator_(&buffer_, size)) {
    throw std::bad_alloc();
  }
  // get the device buffer pointer ready
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaHostGetDevicePointer(&device_buffer_, buffer_, 0),
                                 "Failed to get device pointer for host mapped buffer");
}

CudaHostMappedBuffer::~CudaHostMappedBuffer() {
  free_(buffer_);
}

void* CudaHostMappedBuffer::data() {
  return buffer_;
}

size_t CudaHostMappedBuffer::size() const {
  return size_;
}

size_t CudaHostMappedBuffer::get_bytes() const {
  return size_;
}

void CudaHostMappedBuffer::resize(size_t number_of_elements) {
  size_ = number_of_elements;
  if (capacity_ < number_of_elements) {
    free_(buffer_);
    if (!allocator_(&buffer_, size_)) {
      throw std::bad_alloc{};
    }
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaHostGetDevicePointer(&device_buffer_, buffer_, 0),
                                   "Failed to get device pointer for host mapped buffer");
    capacity_ = number_of_elements;
  }
}

}  // namespace cuda
}  // namespace utils
}  // namespace holoscan
