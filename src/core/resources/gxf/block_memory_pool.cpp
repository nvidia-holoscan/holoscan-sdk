/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>
#include <string>

#include "gxf/std/resources.hpp"  // for GPUDevice
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"

namespace holoscan {

namespace {
constexpr int32_t kDefaultDeviceId = 0;
}  // namespace

BlockMemoryPool::BlockMemoryPool(const std::string& name, nvidia::gxf::BlockMemoryPool* component)
    : Allocator(name, component) {
  auto maybe_storage_type = component->getParameter<int32_t>("storage_type");
  if (!maybe_storage_type) { throw std::runtime_error("Failed to get storage_type"); }
  storage_type_ = maybe_storage_type.value();

  auto maybe_block_size = component->getParameter<uint64_t>("block_size");
  if (!maybe_block_size) { throw std::runtime_error("Failed to get block_size"); }
  block_size_ = maybe_block_size.value();

  auto maybe_num_blocks = component->getParameter<uint64_t>("num_blocks");
  if (!maybe_num_blocks) { throw std::runtime_error("Failed to get num_blocks"); }
  num_blocks_ = maybe_num_blocks.value();

  auto maybe_gpu_device =
      component->getParameter<nvidia::gxf::Handle<nvidia::gxf::GPUDevice>>("dev_id");
  if (!maybe_gpu_device) { throw std::runtime_error("Failed to get dev_id"); }
  auto gpu_device_handle = maybe_gpu_device.value();
  dev_id_ = gpu_device_handle->device_id();
}

nvidia::gxf::BlockMemoryPool* BlockMemoryPool::get() const {
  return static_cast<nvidia::gxf::BlockMemoryPool*>(gxf_cptr_);
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
  spec.param(dev_id_,
             "dev_id",
             "Device Id",
             "Device on which to create the memory pool.",
             kDefaultDeviceId);
}

nvidia::gxf::MemoryStorageType BlockMemoryPool::storage_type() const {
  auto pool = get();
  if (pool) {
    return pool->storage_type();
  } else {
    // TODO(unknown): throw error or return Unexpected?
    HOLOSCAN_LOG_ERROR("BlockMemoryPool component not yet registered with GXF");
    return nvidia::gxf::MemoryStorageType::kSystem;
  }
}

uint64_t BlockMemoryPool::num_blocks() const {
  auto pool = get();
  if (!pool) {
    HOLOSCAN_LOG_ERROR("BlockMemoryPool component not yet registered with GXF");
    return 0;
  }
  return pool->num_blocks();
}

}  // namespace holoscan
