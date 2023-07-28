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
#include <set>

#include "gxf/core/expected.hpp"  // nvidia::gxf::Expected
#include "gxf/std/allocator.hpp"  // nvidia::gxf::MemoryStorageType

#ifndef HOLOSCAN_TESTS_CODECS_MOCK_ALLOCATOR_HPP
#define HOLOSCAN_TESTS_CODECS_MOCK_ALLOCATOR_HPP

namespace holoscan {

// Implementation adapted from nvidia::gxf::UnboundedAllocator
class MockAllocator {
 public:
  virtual ~MockAllocator() = default;

  MockAllocator(const MockAllocator& component) = delete;
  MockAllocator(MockAllocator&& component) = delete;
  MockAllocator& operator=(const MockAllocator& component) = delete;
  MockAllocator& operator=(MockAllocator&& component) = delete;

  explicit MockAllocator(const char* name) : name_(name) {}

  // nvidia::gxf::MemoryBuffer class needs name, allocate and free methods

  const char* name() { return name_; }

  // Allocates a memory block with the given size.
  nvidia::gxf::Expected<std::byte*> allocate(uint64_t size, nvidia::gxf::MemoryStorageType type);

  // Frees the given memory block.
  nvidia::gxf::Expected<void> free(std::byte* pointer);

 private:
  const char* name_;
};

}  // namespace holoscan

#endif  // HOLOSCAN_TESTS_CODECS_MOCK_ALLOCATOR_HPP
