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

#include <cstddef>  // std::byte
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

#include "./memory_buffer.hpp"
#include "./mock_allocator.hpp"
#include "holoscan/core/endpoint.hpp"
#include "holoscan/core/expected.hpp"

#ifndef HOLOSCAN_TESTS_CODECS_MOCK_SERIALIZATION_BUFFER_HPP
#define HOLOSCAN_TESTS_CODECS_MOCK_SERIALIZATION_BUFFER_HPP

namespace holoscan {

// like ucp_dt_iov
struct DataBuffer {
  void* buffer;  // pointer to a data buffer
  size_t length;
};

constexpr int32_t kDefaultMemoryStorageType =
    static_cast<int32_t>(holoscan::Endpoint::MemoryStorageType::kSystem);

// mock version of UcxSerializationBuffer
class MockUcxSerializationBuffer : public Endpoint {
 public:
  // MockUcxSerializationBuffer() = default;
  // MockUcxSerializationBuffer(MockEndpoint&&) = default;
  ~MockUcxSerializationBuffer() override = default;

  explicit MockUcxSerializationBuffer(MockAllocator* allocator, size_t buffer_size,
                                      MemoryStorageType storage_type = MemoryStorageType::kSystem)
      : allocator_(allocator), buffer_size_(buffer_size), storage_type_(storage_type) {
    resize(buffer_size_, storage_type_);
  }

  explicit MockUcxSerializationBuffer(size_t buffer_size,
                                      MemoryStorageType storage_type = MemoryStorageType::kSystem)
      : buffer_size_(buffer_size), storage_type_(storage_type) {
    allocator_ = std::make_shared<MockAllocator>("mock_allocator");
    resize(buffer_size_, storage_type_);
  }

  // C++ API wrappers
  bool is_read_available() override { return true; }
  bool is_write_available() override { return true; }
  expected<size_t, RuntimeError> write(const void* data, size_t size) override;
  expected<size_t, RuntimeError> read(void* data, size_t size) override;
  expected<void, RuntimeError> write_ptr(const void* pointer, size_t size,
                                         MemoryStorageType type) override;

  // Resizes the buffer
  expected<void, RuntimeError> resize(size_t size, MemoryStorageType storage_type);

  // The type of memory where the data is stored
  MemoryStorageType storage_type() const { return buffer_.storage_type(); }
  // Returns a read-only pointer to buffer data
  const std::byte* data() const { return buffer_.pointer(); }
  // Returns the capacity of the buffer
  size_t capacity() const { return buffer_.size(); }
  // Returns the number of bytes written to the buffer
  size_t size() const;
  // Resets buffer for sequential access
  void reset();

 private:
  std::shared_ptr<MockAllocator> allocator_;
  size_t buffer_size_;
  MemoryStorageType storage_type_;

  // Data buffers used by write_ptr
  std::vector<DataBuffer> data_buffers_;

  // Data buffer used by read/write
  MemoryBuffer buffer_;
  // Offset for sequential writes
  size_t write_offset_;
  // Offset for sequential reads
  size_t read_offset_;
  // Mutex to guard concurrent access
  mutable std::mutex mutex_;
};

}  // namespace holoscan

#endif  // HOLOSCAN_TESTS_CODECS_MOCK_SERIALIZATION_BUFFER_HPP
