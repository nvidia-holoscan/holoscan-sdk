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

#include "holoscan/core/resources/gxf/block_memory_pool.hpp"

#include "holoscan/core/component_spec.hpp"

namespace holoscan {

BlockMemoryPool::BlockMemoryPool(const std::string& name, nvidia::gxf::BlockMemoryPool* component)
    : Allocator(name, component) {
  int32_t storage_type = 0;
  GxfParameterGetInt32(gxf_context_, gxf_cid_, "storage_type", &storage_type);
  storage_type_ = storage_type;
  uint64_t block_size = 0;
  GxfParameterGetUInt64(gxf_context_, gxf_cid_, "block_size", &block_size);
  block_size_ = block_size;
  uint64_t num_blocks = 0;
  GxfParameterGetUInt64(gxf_context_, gxf_cid_, "num_blocks", &num_blocks);
  num_blocks_ = num_blocks;
}

void BlockMemoryPool::setup(ComponentSpec& spec) {
  spec.param(storage_type_,
             "storage_type",
             "Storage type",
             "The memory storage type used by this allocator. Can be kHost (0), kDevice (1) or "
             "kSystem (2)",
             0);
  spec.param(block_size_,
             "block_size",
             "Block size",
             "The size of one block of memory in byte. Allocation requests can only be "
             "fulfilled if they "
             "fit into one block. If less memory is requested still a full block is issued.");
  spec.param(
      num_blocks_,
      "num_blocks",
      "Number of blocks",
      "The total number of blocks which are allocated by the pool. If more blocks are requested "
      "allocation requests will fail.");
}

}  // namespace holoscan
