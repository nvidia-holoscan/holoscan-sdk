/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_RESOURCES_ALLOCATORS_PYDOC_HPP
#define PYHOLOSCAN_RESOURCES_ALLOCATORS_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace Allocator {

PYDOC(Allocator, R"doc(
Base allocator class.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the resource.

Returns
-------
str
    The GXF type name of the resource
)doc")

PYDOC(is_available, R"doc(
Boolean representing whether the resource is available.

Returns
-------
bool
    Availability of the resource.
)doc")

PYDOC(allocate, R"doc(
Allocate the requested amount of memory.

Parameters
----------
size : int
    The amount of memory to allocate
type : holoscan.resources.MemoryStorageType
    Enum representing the type of memory to allocate.

Returns
-------
Opaque PyCapsule object representing a std::byte* pointer to the allocated memory.
)doc")

PYDOC(free, R"doc(
Free the allocated memory

Parameters
----------
pointer : PyCapsule
    Opaque PyCapsule object representing a std::byte* pointer to the allocated
    memory.
)doc")

}  // namespace Allocator

namespace BlockMemoryPool {

PYDOC(BlockMemoryPool, R"doc(
Block memory pool resource.

Provides a maximum number of equally sized blocks of memory.
)doc")

// Constructor
PYDOC(BlockMemoryPool_python, R"doc(
Block memory pool resource.

Provides a maximum number of equally sized blocks of memory.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
storage_type : int or holoscan.resources.MemoryStorageType
    The storage type (0=Host, 1=Device, 2=System).
block_size : int
    The size of each block in the memory pool (in bytes).
num_blocks : int
    The number of blocks in the memory pool.
dev_id : int
    CUDA device ID. Specifies the device on which to create the memory pool.
name : str, optional
    The name of the memory pool.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the resource.

Returns
-------
str
    The GXF type name of the resource
)doc")

PYDOC(setup, R"doc(
Define the component specification.

Parameters
----------
spec : holoscan.core.ComponentSpec
    Component specification associated with the resource.
)doc")

}  // namespace BlockMemoryPool

namespace CudaStreamPool {

PYDOC(CudaStreamPool, R"doc(
CUDA stream pool.
)doc")

// Constructor
PYDOC(CudaStreamPool_python, R"doc(
CUDA stream pool.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
dev_id : int
    CUDA device ID. Specifies the device on which to create the stream pool.
stream_flags : int
    Flag values used in creating CUDA streams.
stream_priority : int
    Priority values used in creating CUDA streams.
reserved_size : int
    TODO
max_size : int
    Maximum stream size.
name : str, optional
    The name of the stream pool.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the resource.

Returns
-------
str
    The GXF type name of the resource
)doc")

PYDOC(setup, R"doc(
Define the component specification.

Parameters
----------
spec : holoscan.core.ComponentSpec
    Component specification associated with the resource.
)doc")

}  // namespace CudaStreamPool

namespace UnboundedAllocator {

PYDOC(UnboundedAllocator, R"doc(
Unbounded allocator.

This allocator uses dynamic memory allocation without an upper bound.
)doc")

// Constructor
PYDOC(UnboundedAllocator_python, R"doc(
Unbounded allocator.

This allocator uses dynamic memory allocation without an upper bound.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
name : str, optional
    The name of the serializer.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the resource.

Returns
-------
str
    The GXF type name of the resource
)doc")

PYDOC(setup, R"doc(
Define the component specification.

Parameters
----------
spec : holoscan.core.ComponentSpec
    Component specification associated with the resource.
)doc")

}  // namespace UnboundedAllocator

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_RESOURCES_ALLOCATORS_PYDOC_HPP
