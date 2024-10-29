/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

PYDOC(is_available, R"doc(
Boolean representing whether the resource is available.

Returns
-------
bool
    Availability of the resource.
)doc")

PYDOC(block_size, R"doc(
Get the block size of the allocator.

Returns
-------
int
    The block size of the allocator. Returns 1 for byte-based allocators.
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

namespace CudaAllocator {

PYDOC(CudaAllocator, R"doc(
Base class for CUDA-based allocators.
)doc")

PYDOC(allocate_async, R"doc(
Allocate amount(s) of memory asynchronously.

Parameters
----------
size : int
    The amount of memory to allocate
stream : holoscan.CudaStream
    The CUDA stream to use for the allocation.

Returns
-------
Opaque PyCapsule object representing a std::byte* pointer to the allocated memory.
)doc")

PYDOC(free_async, R"doc(
Free CUDA-based memory asynchronously.

Parameters
----------
pointer : PyCapsule
    Opaque PyCapsule object representing a std::byte* pointer to the allocated
    memory.
stream : holoscan.CudaStream
    The CUDA stream to use for the allocation.
)doc")

PYDOC(block_size, R"doc(
Get the block size of the allocator.

Returns
-------
int
    The block size of the allocator. Returns 1 for byte-based allocators.
)doc")

PYDOC(pool_size, R"doc(
Return the memory pool size for the specified storage type.

Parameters
----------
storage_type : holoscan.resources.MemoryStorageType
    Enum representing the type of memory to allocate.

Returns
-------
size : int
    The size of the memory pool for the specified storage type.
)doc")

}  // namespace CudaAllocator

namespace BlockMemoryPool {

PYDOC(BlockMemoryPool, R"doc(
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

}  // namespace BlockMemoryPool

namespace CudaStreamPool {

PYDOC(CudaStreamPool, R"doc(
CUDA stream pool.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
dev_id : int
    CUDA device ID. Specifies the device on which to create the stream pool.
stream_flags : int
    Flags for CUDA streams in the pool. This will be passed to CUDA's
    cudaStreamCreateWithPriority [1]_ when creating the streams. The default value of 0 corresponds
    to ``cudaStreamDefault``. A value of 1 corresponds to ``cudaStreamNonBlocking``, indicating
    that the stream can run concurrently with work in stream 0 (default stream) and should not
    perform any implicit synchronization with it.
stream_priority : int
    Priority value for CUDA streams in the pool. This is an integer value passed to
    cudaSreamCreateWithPriority [1]_. Lower numbers represent higher priorities.
reserved_size : int
    The number of CUDA streams to initially reserve in the pool (prior to first request).
max_size : int
    The maximum number of streams that can be allocated, unlimited by default.
name : str, optional
    The name of the stream pool.

References
----------
.. [1] https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html

)doc")

}  // namespace CudaStreamPool

namespace UnboundedAllocator {

PYDOC(UnboundedAllocator, R"doc(
Unbounded allocator.

This allocator uses dynamic memory allocation without an upper bound.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
name : str, optional
    The name of the serializer.
)doc")

}  // namespace UnboundedAllocator

namespace RMMAllocator {

PYDOC(RMMAllocator, R"doc(
Device and Host allocator using RAPIDS memory manager (RMM).

Provides memory pools for asynchronously allocated CUDA device memory and pinned host memory.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
device_memory_initial_size : str, optional
    The initial size of the device memory pool. See the Notes section for the format accepted.
device_memory_max_size : str, optional
    The maximum size of the device memory pool. See the Notes section for the format accepted.
host_memory_initial_size : str, optional
    The initial size of the host memory pool. See the Notes section for the format accepted.
host_memory_max_size : str, optional
    The maximum size of the host memory pool. See the Notes section for the format accepted.
dev_id : int
    GPU device ID. Specifies the device on which to create the memory pool.
name : str, optional
    The name of the memory pool.

Notes
-----
The values for the memory parameters, such as `device_memory_initial_size` must be specified in the
form of a string containing a non-negative integer value followed by a suffix representing the
units. Supported units are B, KB, MB, GB and TB where the values are powers of 1024 bytes
(e.g. MB = 1024 * 1024 bytes). Examples of valid units are "512MB", "256 KB", "1 GB". If a floating
point number is specified that decimal portion will be truncated (i.e. the value is rounded down to
the nearest integer).

)doc")

}  // namespace RMMAllocator

namespace StreamOrderedAllocator {

PYDOC(StreamOrderedAllocator, R"doc(
Device and Host allocator using RAPIDS memory manager (StreamOrdered).

Provides memory pools for asynchronously allocated CUDA device memory and pinned host memory.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
device_memory_initial_size : str, optional
    The initial size of the device memory pool. See the Notes section for the format accepted.
device_memory_max_size : str, optional
    The maximum size of the device memory pool. See the Notes section for the format accepted.
release_threshold : str, optional
    The amount of reserved memory to hold onto before trying to release memory back to the OS.  See
    the Notes section for the format accepted.
dev_id : int, optional
    GPU device ID. Specifies the device on which to create the memory pool.
name : str, optional
    The name of the memory pool.

Notes
-----
The values for the memory parameters, such as `device_memory_initial_size` must be specified in the
form of a string containing a non-negative integer value followed by a suffix representing the
units. Supported units are B, KB, MB, GB and TB where the values are powers of 1024 bytes
(e.g. MB = 1024 * 1024 bytes). Examples of valid units are "512MB", "256 KB", "1 GB". If a floating
point number is specified that decimal portion will be truncated (i.e. the value is rounded down to
the nearest integer).
)doc")

}  // namespace StreamOrderedAllocator

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_RESOURCES_ALLOCATORS_PYDOC_HPP
