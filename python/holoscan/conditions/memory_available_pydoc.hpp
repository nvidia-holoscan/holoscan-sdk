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

#ifndef PYHOLOSCAN_CONDITIONS_MEMORY_AVAILABLE_PYDOC_HPP
#define PYHOLOSCAN_CONDITIONS_MEMORY_AVAILABLE_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace MemoryAvailableCondition {

PYDOC(MemoryAvailableCondition, R"doc(
Condition that permits execution only when a specified allocator has sufficient memory available.

The memory is typically provided via the `min_bytes` parameter, but for allocators that use memory
blocks it is possible to specify the memory via `min_blocks` instead if desired.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment the condition will be associated with
min_bytes: int, optional
    The minimum number of bytes that must be available in order for the associated operator to
    execute. Exclusive with `min_blocks` (only one of the two can be set).
min_blocks: int, optional
    The minimum number of blocks that must be available in order for the associated operator to
    execute. Can only be used with allocators such as `BlockMemoryPool` that use memory blocks.
    Exclusive with `min_bytes` (only one of the two can be set).
allocator : holoscan.core.Allocator
    The allocator whose memory availability will be checked.
name : str, optional
    The name of the condition.
)doc")

PYDOC(allocator, R"doc(
The allocator associated with the condition.
)doc")

}  // namespace MemoryAvailableCondition

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_CONDITIONS_MEMORY_AVAILABLE_PYDOC_HPP */
