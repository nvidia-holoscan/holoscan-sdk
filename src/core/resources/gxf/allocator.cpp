/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/resources/gxf/allocator.hpp"

namespace holoscan {

Allocator::Allocator(const std::string& name, nvidia::gxf::Allocator* component)
    : GXFResource(name, component) {}

bool Allocator::is_available(uint64_t size) {
  if (gxf_cptr_) {
    nvidia::gxf::Allocator* allocator = static_cast<nvidia::gxf::Allocator*>(gxf_cptr_);
    return allocator->is_available(size);
  }

  return false;
}

nvidia::byte* Allocator::allocate(uint64_t size, MemoryStorageType type) {
  if (gxf_cptr_) {
    nvidia::gxf::Allocator* allocator = static_cast<nvidia::gxf::Allocator*>(gxf_cptr_);

    auto result = allocator->allocate(size, static_cast<nvidia::gxf::MemoryStorageType>(type));
    if (result) { return result.value(); }
  }

  HOLOSCAN_LOG_ERROR(
      "Failed to allocate memory of size {} with type {}", size, static_cast<int>(type));

  return nullptr;
}

void Allocator::free(nvidia::byte* pointer) {
  if (gxf_cptr_) {
    nvidia::gxf::Allocator* allocator = static_cast<nvidia::gxf::Allocator*>(gxf_cptr_);
    auto result = allocator->free(pointer);
    if (!result) { HOLOSCAN_LOG_ERROR("Failed to free memory at {}", static_cast<void*>(pointer)); }
  }
}

}  // namespace holoscan
