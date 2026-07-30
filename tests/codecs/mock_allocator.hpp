/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>  // std::byte
#include <cstdint>
#include <set>

#include <gxf/core/expected.hpp>  // nvidia::gxf::Expected
#include <gxf/std/allocator.hpp>  // nvidia::gxf::MemoryStorageType

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
