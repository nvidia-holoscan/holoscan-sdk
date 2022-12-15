/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoinfer_buffer.hpp>

namespace holoscan {
namespace inference {

/*
 * @brief Get the element size
 *
 * @param element_type Input data type. Float32 is the only supported element type.
 *
 * @returns Bytes used in storing element type
 */
uint32_t get_element_size(holoinfer_datatype element_type) noexcept {
  switch (element_type) {
    case holoinfer_datatype::hFloat:
      return 4;
  }
  return 0;
}

/*
 * @brief Allocate buffer on host and device
 *
 * @param buffers Map with keyword as model name or tensor name, and value as DataBuffer. The map
 * is populated in this function.
 * @param dims Dimension of the allocation
 * @param keyname Storage name in the map against the created DataBuffer
 * @returns InferStatus with appropriate code and message
 */
InferStatus allocate_host_device_buffers(DataMap& buffers, std::vector<int64_t>& dims,
                                         const std::string& keyname) {
  size_t buffer_size = accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());

  auto data_buffer = std::make_shared<DataBuffer>();
  data_buffer->host_buffer.resize(buffer_size);
  data_buffer->device_buffer->resize(buffer_size);

  buffers.insert({keyname, std::move(data_buffer)});

  return InferStatus();
}

/*
 * @brief Allocate buffer on host
 *
 * @param buffers Map with keyword as model name or tensor name, and value as DataBuffer. The map
 * is populated in this function.
 * @param dims Dimension of the allocation
 * @param keyname Storage name in the map against the created DataBuffer
 * @returns InferStatus with appropriate code and message
 */
InferStatus allocate_host_buffers(DataMap& buffers, std::vector<int64_t>& dims,
                                  const std::string& keyname) {
  size_t buffer_size = accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());

  auto data_buffer = std::make_shared<DataBuffer>();
  data_buffer->host_buffer.resize(buffer_size);

  buffers.insert({keyname, std::move(data_buffer)});
  return InferStatus();
}

bool DeviceAllocator::operator()(void** ptr, size_t size) const {
  return cudaMalloc(ptr, size) == cudaSuccess;
}

void DeviceFree::operator()(void* ptr) const {
  cudaFree(ptr);
}

DataBuffer::DataBuffer() {
  device_buffer = std::make_shared<DeviceBuffer>();
}

DeviceBuffer::DeviceBuffer(holoinfer_datatype type)
    : size_(0), capacity_(0), type_(type), buffer_(nullptr) {}

DeviceBuffer::DeviceBuffer(size_t size, holoinfer_datatype type)
    : size_(size), capacity_(size), type_(type) {
  if (!allocator_(&buffer_, this->get_bytes())) { throw std::bad_alloc(); }
}

void* DeviceBuffer::data() {
  return buffer_;
}

size_t DeviceBuffer::size() const {
  return size_;
}

size_t DeviceBuffer::get_bytes() const {
  return this->size() * get_element_size(type_);
}

void DeviceBuffer::resize(size_t element_size) {
  size_ = element_size;
  if (capacity_ < element_size) {
    free_(buffer_);
    if (!allocator_(&buffer_, this->get_bytes())) { throw std::bad_alloc{}; }
    capacity_ = element_size;
  }
}

DeviceBuffer::~DeviceBuffer() {
  free_(buffer_);
}

}  // namespace inference
}  // namespace holoscan
