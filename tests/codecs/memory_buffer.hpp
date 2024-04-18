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

#include <cstddef>  // std::byte
#include <memory>
#include <utility>

#include "./mock_allocator.hpp"

#include "common/logger.hpp"  // GXF_LOG*
#include "gxf/core/gxf.h"     // GxfResultStr
#include "gxf/core/expected.hpp"  // nvidia::gxf::Expected, nvidia::gxf::ForwardError, nvidia::gxf::Success

#ifndef HOLOSCAN_TESTS_CODECS_MEMORY_BUFFER_HPP
#define HOLOSCAN_TESTS_CODECS_MEMORY_BUFFER_HPP

namespace holoscan {

// like nvidia::GXF::MemoryBuffer except
//   - uses std::shared_ptr<holoscan::MockAllocator> instead of
//         nvidia::gxf::Handle<nvidia::gxf::Allocator>
//   - unused methods like wrapMemory were omitted
class MemoryBuffer {
 public:
  MemoryBuffer() = default;
  MemoryBuffer(const MemoryBuffer&) = delete;
  MemoryBuffer& operator=(const MemoryBuffer&) = delete;

  MemoryBuffer(MemoryBuffer&& other) { *this = std::move(other); }

  MemoryBuffer& operator=(MemoryBuffer&& other) {
    size_ = other.size_;
    storage_type_ = other.storage_type_;
    pointer_ = other.pointer_;
    release_func_ = std::move(other.release_func_);

    other.pointer_ = nullptr;
    other.release_func_ = nullptr;

    return *this;
  }

  // Type of the callback function to release memory passed to the MemoryBuffer
  // using the wrapMemory method
  using release_function_t = std::function<nvidia::gxf::Expected<void>(void* pointer)>;

  nvidia::gxf::Expected<void> freeBuffer() {
    if (release_func_ && pointer_) {
      const nvidia::gxf::Expected<void> result = release_func_(pointer_);
      if (!result) { return nvidia::gxf::ForwardError(result); }

      release_func_ = nullptr;
      pointer_ = nullptr;
      size_ = 0;
    }

    return nvidia::gxf::Success;
  }

  ~MemoryBuffer() { freeBuffer(); }

  nvidia::gxf::Expected<void> resize(std::shared_ptr<MockAllocator> allocator, uint64_t size,
                                     nvidia::gxf::MemoryStorageType storage_type) {
    const auto result = freeBuffer();
    if (!result) {
      GXF_LOG_ERROR("Failed to free memory. Error code: %s", GxfResultStr(result.error()));
      return nvidia::gxf::ForwardError(result);
    }

    const auto maybe = allocator->allocate(size, storage_type);
    if (!maybe) {
      GXF_LOG_ERROR("%s Failed to allocate %zu size of memory of type %d. Error code: %s",
                    allocator->name(),
                    size,
                    static_cast<int>(storage_type),
                    GxfResultStr(maybe.error()));
      return nvidia::gxf::ForwardError(maybe);
    }

    storage_type_ = storage_type;
    pointer_ = maybe.value();
    size_ = size;

    release_func_ = [allocator](void* data) {
      return allocator->free(reinterpret_cast<std::byte*>(data));
    };

    return nvidia::gxf::Success;
  }

  // The type of memory where the data is stored.
  nvidia::gxf::MemoryStorageType storage_type() const { return storage_type_; }

  // Raw pointer to the first byte of elements stored in the buffer.
  std::byte* pointer() const { return pointer_; }

  // Size of buffer contents in bytes
  uint64_t size() const { return size_; }

 private:
  uint64_t size_ = 0;
  std::byte* pointer_ = nullptr;
  nvidia::gxf::MemoryStorageType storage_type_ = nvidia::gxf::MemoryStorageType::kHost;
  release_function_t release_func_ = nullptr;
};
}  // namespace holoscan

#endif  // HOLOSCAN_TESTS_CODECS_MEMORY_BUFFER_HPP
