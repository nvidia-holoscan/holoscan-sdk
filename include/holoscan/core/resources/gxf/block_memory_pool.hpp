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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_BLOCK_MEMORY_POOL_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_BLOCK_MEMORY_POOL_HPP

#include <cstdint>
#include <string>
#include "gxf/std/allocator.hpp"
#include "gxf/std/block_memory_pool.hpp"

#include "./allocator.hpp"

namespace holoscan {

/**
 * @brief Block memory pool allocator.
 *
 * This is a memory pool which provides a user-specified number of equally sized blocks of memory.
 */
class BlockMemoryPool : public Allocator {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(BlockMemoryPool, Allocator)

  BlockMemoryPool() = default;
  BlockMemoryPool(int32_t storage_type, uint64_t block_size, uint64_t num_blocks)
      : storage_type_(storage_type), block_size_(block_size), num_blocks_(num_blocks) {}
  BlockMemoryPool(const std::string& name, nvidia::gxf::BlockMemoryPool* component);

  const char* gxf_typename() const override { return "nvidia::gxf::BlockMemoryPool"; }

  void setup(ComponentSpec& spec) override;

 private:
  Parameter<int32_t> storage_type_;
  Parameter<uint64_t> block_size_;
  Parameter<uint64_t> num_blocks_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_BLOCK_MEMORY_POOL_HPP */
