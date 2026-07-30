/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mock_allocator.hpp"

#include <cstddef>
#include <cstdint>

#include <gxf/core/gxf.h>
#include <gxf/core/expected.hpp>  // nvidia::gxf::Expected
#include <gxf/std/allocator.hpp>  // nvidia::gxf::MemoryStorageType
#include "common/logger.hpp"      // GXF_LOG_*

namespace holoscan {

// Allocates a memory block with the given size.
nvidia::gxf::Expected<std::byte*> MockAllocator::allocate(uint64_t size,
                                                          nvidia::gxf::MemoryStorageType type) {
  void* result;

  // We cannot allocate safely a block of size 0.
  // We can artificially increase the size of 1 to remove failure when free_abi is called.
  if (size == 0) {
    size = 1;
  }

  switch (static_cast<nvidia::gxf::MemoryStorageType>(type)) {
    case nvidia::gxf::MemoryStorageType::kHost: {
      GXF_LOG_ERROR("Failure in allocate, memory type kHost not supported by MockAllocator.");
      return nvidia::gxf::Unexpected{GXF_NOT_IMPLEMENTED};
    } break;
    case nvidia::gxf::MemoryStorageType::kDevice: {
      GXF_LOG_ERROR("Failure in allocate, memory type kDevice not supported by MockAllocator.");
      return nvidia::gxf::Unexpected{GXF_NOT_IMPLEMENTED};
    } break;
    case nvidia::gxf::MemoryStorageType::kCudaManaged: {
      GXF_LOG_ERROR(
          "Failure in allocate, memory type kCudaManaged not supported by MockAllocator.");
      return nvidia::gxf::Unexpected{GXF_NOT_IMPLEMENTED};
    }
    case nvidia::gxf::MemoryStorageType::kSystem: {
      result = static_cast<std::byte*>(::operator new(size * sizeof(std::byte), std::nothrow));
      if (result == nullptr) {
        return nvidia::gxf::Unexpected{GXF_OUT_OF_MEMORY};
      }
    } break;
    default:
      return nvidia::gxf::Unexpected{GXF_PARAMETER_OUT_OF_RANGE};
  }
  return static_cast<std::byte*>(result);
}

// Frees the given memory block.
nvidia::gxf::Expected<void> MockAllocator::free(std::byte* pointer) {
  void* vpointer = static_cast<void*>(pointer);
  ::operator delete(pointer);
  return nvidia::gxf::Expected<void>();
}
}  // namespace holoscan
