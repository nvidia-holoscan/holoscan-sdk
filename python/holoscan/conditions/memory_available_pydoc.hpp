/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment (or subgraph) the condition will be associated with
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
