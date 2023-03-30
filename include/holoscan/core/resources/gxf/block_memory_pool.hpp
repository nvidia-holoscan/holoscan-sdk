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

#include <string>
// gxf/std/block_memory_pool.hpp is missing in the GXF SDK
// The following is a copy of the file from the GXF SDK until it is fixed
// TODO: Remove this file once the issue is fixed
////////////////////////////////////////////////////////////////////////////////////////////////////
#include <cstdint>
#include <memory>
#include <mutex>
#include "gxf/std/allocator.hpp"

namespace nvidia {
namespace gxf {

class FixedPoolUint64;

// A memory pools which provides a maximum number of equally sized blocks of
// memory.
class BlockMemoryPool : public Allocator {
 public:
  BlockMemoryPool();
  ~BlockMemoryPool();

  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t is_available_abi(uint64_t size) override;
  gxf_result_t allocate_abi(uint64_t size, int32_t type, void** pointer) override;
  gxf_result_t free_abi(void* pointer) override;
  gxf_result_t deinitialize() override;

 private:
  Parameter<int32_t> storage_type_;
  Parameter<uint64_t> block_size_;
  Parameter<uint64_t> num_blocks_;

  void* pointer_;
  std::unique_ptr<FixedPoolUint64> stack_;
  std::mutex stack_mutex_;
};

}  // namespace gxf
}  // namespace nvidia
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "./allocator.hpp"

namespace holoscan {

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
