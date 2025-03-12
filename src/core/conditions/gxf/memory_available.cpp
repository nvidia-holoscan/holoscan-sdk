/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/conditions/gxf/memory_available.hpp"

#include "holoscan/core/component_spec.hpp"

namespace holoscan {
void MemoryAvailableCondition::setup(ComponentSpec& spec) {
  spec.param(allocator_,
             "allocator",
             "Allocator",
             "The allocator whose memory availability will be checked.");
  spec.param(min_bytes_,
             "min_bytes",
             "Minimum number of bytes",
             "The minimum number of bytes that must be available in order for the associated "
             "operator to execute. Exclusive with min_blocks (only one can be set)",
             ParameterFlag::kOptional);
  spec.param(min_blocks_,
             "min_blocks",
             "Minimum number of memory blocks",
             "The minimum number of blocks that must be available in order for the associated "
             "operator to execute. Can only be used with allocators such as `BlockMemoryPool` "
             "that use memory blocks. Exclusive with min_bytes (only one can be set)",
             ParameterFlag::kOptional);
}

void MemoryAvailableCondition::initialize() {
  auto& current_args = args();

  bool has_min_bytes = std::find_if(current_args.begin(), current_args.end(), [](const auto& arg) {
                         return (arg.name() == "min_bytes");
                       }) != current_args.end();

  bool has_min_blocks = std::find_if(current_args.begin(), current_args.end(), [](const auto& arg) {
                          return (arg.name() == "min_blocks");
                        }) != current_args.end();

  if (!has_min_bytes && !has_min_blocks) {
    throw std::runtime_error(
        "MemoryAvailableCondition: either a `min_bytes` or `min_blocks` argument must be "
        "provided");
  }
  if (has_min_bytes && has_min_blocks) {
    throw std::runtime_error(
        "MemoryAvailableCondition: `min_bytes` or `min_blocks` cannot both be provided. "
        "Please provide only one of the two.");
  }

  GXFCondition::initialize();
}

nvidia::gxf::MemoryAvailableSchedulingTerm* MemoryAvailableCondition::get() const {
  return static_cast<nvidia::gxf::MemoryAvailableSchedulingTerm*>(gxf_cptr_);
}

}  // namespace holoscan
