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

#include "mock_serialization_buffer.hpp"

#include <cstdint>
#include <vector>

#include "gxf/core/expected.hpp"
#include "gxf/serialization/endpoint.hpp"
#include "holoscan/core/expected.hpp"

#include "common/memory_utils.hpp"  // AllocateArray
#include "gxf/std/unbounded_allocator.hpp"

namespace holoscan {

// based on UcxSerializationBuffer::write_abi
expected<size_t, RuntimeError> MockUcxSerializationBuffer::write(const void* data, size_t size) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (data == nullptr) {
    return make_unexpected<RuntimeError>(RuntimeError(ErrorCode::kCodecError, "null data"));
  }
  if (size > buffer_.size() - write_offset_) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "size exceeds preallocated buffer size"));
  }
  std::memcpy(buffer_.pointer() + write_offset_, data, size);
  write_offset_ += size;
  return size;
}

// based on UcxSerializationBuffer::read_abi
expected<size_t, RuntimeError> MockUcxSerializationBuffer::read(void* data, size_t size) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (data == nullptr) {
    return make_unexpected<RuntimeError>(RuntimeError(ErrorCode::kCodecError, "null data"));
  }
  if (size > buffer_.size() - read_offset_) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "size exceeds preallocated buffer size"));
  }
  std::memcpy(data, buffer_.pointer() + read_offset_, size);
  read_offset_ += size;
  return size;
}

// based on UcxSerializationBuffer::write_ptr_abi
expected<void, RuntimeError> MockUcxSerializationBuffer::write_ptr(const void* pointer, size_t size,
                                                                   MemoryStorageType type) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (pointer == nullptr) {
    return make_unexpected<RuntimeError>(RuntimeError(ErrorCode::kCodecError, "null data"));
  }
  DataBuffer d;
  d.buffer = const_cast<void*>(pointer);
  d.length = size;
  data_buffers_.push_back(d);
  storage_type_ = type;
  return expected<void, RuntimeError>();
}

// Resizes the buffer
expected<void, RuntimeError> MockUcxSerializationBuffer::resize(size_t size,
                                                                MemoryStorageType storage_type) {
  std::unique_lock<std::mutex> lock(mutex_);
  write_offset_ = 0;
  read_offset_ = 0;
  auto result = buffer_.resize(allocator_, size, storage_type);
  if (!result) {
    return make_unexpected<RuntimeError>(
        RuntimeError(ErrorCode::kCodecError, "error in MemoryBuffer resize"));
  }
  return expected<void, RuntimeError>();
}

// Returns the number of bytes written to the buffer
size_t MockUcxSerializationBuffer::size() const {
  std::unique_lock<std::mutex> lock(mutex_);
  return write_offset_;
}

// Resets buffer for sequential access
void MockUcxSerializationBuffer::reset() {
  std::unique_lock<std::mutex> lock(mutex_);
  write_offset_ = 0;
  read_offset_ = 0;
}
}  // namespace holoscan
