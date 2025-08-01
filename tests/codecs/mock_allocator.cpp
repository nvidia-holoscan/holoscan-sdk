/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mock_allocator.hpp"

#include <cstddef>
#include <cstdint>

#include "gxf/core/gxf.h"
#include "common/logger.hpp"      // GXF_LOG_*
#include "gxf/core/expected.hpp"  // nvidia::gxf::Expected
#include "gxf/std/allocator.hpp"  // nvidia::gxf::MemoryStorageType

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
