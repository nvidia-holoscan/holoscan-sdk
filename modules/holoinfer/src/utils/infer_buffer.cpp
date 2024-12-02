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

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoinfer_buffer.hpp>

namespace holoscan {
namespace inference {

/*
 * @brief Get the element size
 *
 * @param element_type Input data type. Float32 is the only supported element type.
 *
 * @return Bytes used in storing element type
 */
uint32_t get_element_size(holoinfer_datatype element_type) noexcept {
  switch (element_type) {
    case holoinfer_datatype::h_Float32:
    case holoinfer_datatype::h_Int32:
      return 4;
    case holoinfer_datatype::h_Int64:
      return 8;
    case holoinfer_datatype::h_Int8:
    case holoinfer_datatype::h_UInt8:
      return 1;
    case holoinfer_datatype::h_Float16:
      return 2;
  }
  return 0;
}

InferStatus allocate_buffers(DataMap& buffers, std::vector<int64_t>& dims,
                             holoinfer_datatype datatype, const std::string& keyname,
                             bool allocate_cuda, int device_id) {
  size_t buffer_size = accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());

  std::shared_ptr<DataBuffer> data_buffer;
  try {
    data_buffer = std::make_shared<DataBuffer>(datatype, device_id);
  } catch (std::exception& e) {
    InferStatus status = InferStatus(holoinfer_code::H_ERROR);
    status.set_message(
        fmt::format("Data buffer creation failed for {} with error {}", keyname, e.what()));
    return status;
  }
  data_buffer->host_buffer_->resize(buffer_size);
  if (allocate_cuda) { data_buffer->device_buffer_->resize(buffer_size); }
  buffers.insert({keyname, std::move(data_buffer)});
  return InferStatus();
}

bool DeviceAllocator::operator()(void** ptr, size_t size) const {
  return cudaMalloc(ptr, size) == cudaSuccess;
}

void DeviceFree::operator()(void* ptr) const {
  if (ptr) { cudaFree(ptr); }
}

DataBuffer::DataBuffer(holoinfer_datatype data_type, int device_id)
    : data_type_(data_type) {
  try {
    device_buffer_ = std::make_shared<DeviceBuffer>(data_type_, device_id);
  } catch (std::exception& e) {
    throw std::runtime_error(
        fmt::format("Device buffer creation failed in DataBuffer constructor with {}", e.what()));
  }
  try {
    host_buffer_ = std::make_shared<HostBuffer>(data_type_);
  } catch (std::exception& e) {
    throw std::runtime_error(
        fmt::format("Host buffer creation failed in DataBuffer constructor with {}", e.what()));
  }
}

DeviceBuffer::DeviceBuffer(holoinfer_datatype type, int device_id)
    : Buffer(type, device_id), size_(0), capacity_(0), buffer_(nullptr) {}

DeviceBuffer::DeviceBuffer(size_t size, holoinfer_datatype type)
    : Buffer(type), size_(size), capacity_(size) {
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

void DeviceBuffer::resize(size_t number_of_elements) {
  size_ = number_of_elements;
  if (capacity_ < number_of_elements) {
    free_(buffer_);
    if (!allocator_(&buffer_, this->get_bytes())) { throw std::bad_alloc{}; }
    capacity_ = number_of_elements;
  }
}

DeviceBuffer::~DeviceBuffer() {
  free_(buffer_);
}

void* HostBuffer::data() {
  return static_cast<void*>(buffer_.data());
}

size_t HostBuffer::size() const {
  return number_of_elements_;
}

size_t HostBuffer::get_bytes() const {
  return buffer_.size();
}

void HostBuffer::set_type(holoinfer_datatype in_type) {
  type_ = in_type;
  resize(size());
}

void HostBuffer::resize(size_t number_of_elements) {
  if (number_of_elements != number_of_elements_) {
    buffer_.clear();
    number_of_elements_ = number_of_elements;
    buffer_.resize(number_of_elements * get_element_size(type_));
  }
}

}  // namespace inference
}  // namespace holoscan
