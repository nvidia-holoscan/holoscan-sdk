/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 *
 * ==Parameters==
 *
 * - **block_size** (uint64_t): The size of the individual memory blocks in the pool (in bytes).
 * - **num_blocks** (uint64_t): The number of memory blocks available in the pool.
 * - **storage_type** (int32_t, optional): The memory type allocated by the pool (0=Host, 1=Device,
 * 2=System, 3=CUDA Managed) will use (Default: 0). Here "host" and "system" are both CPU memory,
 * but "host" refers to pinned host memory (allocated via `cudaMallocHost`) while "system" is memory
 * allocated by standard C++ `new`. CUDA Managed memory is allocated via CUDA's managed memory APIs
 * and can be used from both host and device.
 * - **dev_id** (int32_t, optional): The CUDA device id specifying which device the memory pool
 * will use (Default: 0).
 */
class BlockMemoryPool : public Allocator {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(BlockMemoryPool, Allocator)

  BlockMemoryPool() = default;
  BlockMemoryPool(int32_t storage_type, uint64_t block_size, uint64_t num_blocks,
                  int32_t dev_id = 0)
      : storage_type_(storage_type),
        block_size_(block_size),
        num_blocks_(num_blocks),
        dev_id_(dev_id) {}
  BlockMemoryPool(const std::string& name, nvidia::gxf::BlockMemoryPool* component);

  const char* gxf_typename() const override { return "nvidia::gxf::BlockMemoryPool"; }

  void setup(ComponentSpec& spec) override;

  // Returns the storage type of the memory blocks
  nvidia::gxf::MemoryStorageType storage_type() const;

  // Returns the total number of blocks
  uint64_t num_blocks() const;

  nvidia::gxf::BlockMemoryPool* get() const;

 private:
  Parameter<int32_t> storage_type_;
  Parameter<uint64_t> block_size_;
  Parameter<uint64_t> num_blocks_;
  Parameter<int32_t> dev_id_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_BLOCK_MEMORY_POOL_HPP */
