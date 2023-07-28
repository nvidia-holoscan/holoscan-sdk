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

#ifndef PYHOLOSCAN_RESOURCES_PYDOC_HPP
#define PYHOLOSCAN_RESOURCES_PYDOC_HPP

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
    CUDA device ID.
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

namespace SerializationBuffer {

PYDOC(SerializationBuffer, R"doc(
Serialization Buffer.
)doc")

// Constructor
PYDOC(SerializationBuffer_python, R"doc(
Serialization Buffer.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
allocator : holoscan.resource.Allocator
    The memory allocator for tensor components.
buffer_size : int, optional
    The size of the buffer in bytes.
name : str, optional
    The name of the serialization buffer
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

}  // namespace SerializationBuffer

namespace UcxSerializationBuffer {

PYDOC(UcxSerializationBuffer, R"doc(
UCX serialization buffer.
)doc")

// Constructor
PYDOC(UcxSerializationBuffer_python, R"doc(
UCX serialization buffer.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
allocator : holoscan.resource.Allocator
    The memory allocator for tensor components.
buffer_size : int, optional
    The size of the buffer in bytes.
name : str, optional
    The name of the serialization buffer
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

}  // namespace UcxSerializationBuffer

namespace UcxComponentSerializer {

PYDOC(UcxComponentSerializer, R"doc(
UCX component serializer.
)doc")

// Constructor
PYDOC(UcxComponentSerializer_python, R"doc(
UCX component serializer.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
allocator : holoscan.resource.Allocator
    The memory allocator for tensor components.
name : str, optional
    The name of the component serializer.
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

}  // namespace UcxComponentSerializer

namespace UcxHoloscanComponentSerializer {

PYDOC(UcxHoloscanComponentSerializer, R"doc(
UCX Holoscan component serializer.
)doc")

// Constructor
PYDOC(UcxHoloscanComponentSerializer_python, R"doc(
UCX Holoscan component serializer.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
allocator : holoscan.resource.Allocator
    The memory allocator for tensor components.
name : str, optional
    The name of the component serializer.
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

}  // namespace UcxHoloscanComponentSerializer

namespace UcxEntitySerializer {

PYDOC(UcxEntitySerializer, R"doc(
UCX entity serializer.
)doc")

// Constructor
PYDOC(UcxEntitySerializer_python, R"doc(
UCX entity serializer.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
component_serializer : list of holoscan.resource.Resource
    The component serializers used by the entity serializer.
verbose_warning : bool, optional
    Whether to use verbose warnings during serialization.
name : str, optional
    The name of the entity serializer.
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

}  // namespace UcxEntitySerializer

namespace DoubleBufferReceiver {

PYDOC(DoubleBufferReceiver, R"doc(
Receiver using a double-buffered queue.

New messages are first pushed to a back stage.
)doc")

// Constructor
PYDOC(DoubleBufferReceiver_python, R"doc(
Receiver using a double-buffered queue.

New messages are first pushed to a back stage.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
capacity : int, optional
    The capacity of the receiver.
policy : int, optional
    The policy to use (0=pop, 1=reject, 2=fault).
name : str, optional
    The name of the receiver.
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

}  // namespace DoubleBufferReceiver

namespace DoubleBufferTransmitter {

PYDOC(DoubleBufferTransmitter, R"doc(
Transmitter using a double-buffered queue.

Messages are pushed to a back stage after they are published.
)doc")

// Constructor
PYDOC(DoubleBufferTransmitter_python, R"doc(
Transmitter using a double-buffered queue.

Messages are pushed to a back stage after they are published.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
capacity : int, optional
    The capacity of the transmitter.
policy : int, optional
    The policy to use (0=pop, 1=reject, 2=fault).
name : str, optional
    The name of the transmitter.
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

}  // namespace DoubleBufferTransmitter

namespace UcxReceiver {

PYDOC(UcxReceiver, R"doc(
UCX network receiver using a double-buffered queue.

New messages are first pushed to a back stage.
)doc")

// Constructor
PYDOC(UcxReceiver_python, R"doc(
UCX network receiver using a double-buffered queue.

New messages are first pushed to a back stage.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
buffer : holoscan.resource.UcxSerializationBuffer
    The serialization buffer used by the transmitter.
capacity : int, optional
    The capacity of the receiver.
policy : int, optional
    The policy to use (0=pop, 1=reject, 2=fault).
address : str, optional
    The IP address used by the transmitter.
port : int, optional
    The network port used by the transmitter.
name : str, optional
    The name of the receiver.
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

}  // namespace UcxReceiver

namespace UcxTransmitter {

PYDOC(UcxTransmitter, R"doc(
UCX network transmitter using a double-buffered queue.

Messages are pushed to a back stage after they are published.
)doc")

// Constructor
PYDOC(UcxTransmitter_python, R"doc(
UCX network transmitter using a double-buffered queue.

Messages are pushed to a back stage after they are published.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
buffer : holoscan.resource.UcxSerializationBuffer
    The serialization buffer used by the transmitter.
capacity : int, optional
    The capacity of the transmitter.
policy : int, optional
    The policy to use (0=pop, 1=reject, 2=fault).
address : str, optional
    The IP address used by the transmitter.
port : int, optional
    The network port used by the transmitter.
maximum_connection_retries : int, optional
    The maximum number of times the transmitter will retry making a connection.
name : str, optional
    The name of the transmitter.
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

}  // namespace UcxTransmitter

namespace Receiver {

PYDOC(Receiver, R"doc(
Base GXF receiver class.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the resource.

Returns
-------
str
    The GXF type name of the resource
)doc")

}  // namespace Receiver

namespace StdComponentSerializer {

PYDOC(StdComponentSerializer, R"doc(
Serializer for GXF Timestamp and Tensor components.
)doc")

// Constructor
PYDOC(StdComponentSerializer_python, R"doc(
Serializer for GXF Timestamp and Tensor components.

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

PYDOC(initialize, R"doc(
Initialize the resource

This method is called only once when the resource is created for the first
time, and uses a light-weight initialization.
)doc")

}  // namespace StdComponentSerializer

namespace Transmitter {

PYDOC(Transmitter, R"doc(
Base GXF transmitter class.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the resource.

Returns
-------
str
    The GXF type name of the resource
)doc")

}  // namespace Transmitter

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

namespace VideoStreamSerializer {

PYDOC(VideoStreamSerializer, R"doc(
Serializer for video streams.
)doc")

// Constructor
PYDOC(VideoStreamSerializer_python, R"doc(
Serializer for video streams.

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

PYDOC(initialize, R"doc(
Initialize the resource

This method is called only once when the resource is created for the first
time, and uses a light-weight initialization.
)doc")

}  // namespace VideoStreamSerializer

namespace Clock {

PYDOC(Clock, R"doc(
Base clock class.
)doc")

PYDOC(time, R"doc(
The current time of the clock (in seconds).

Parameters
----------
time : double
    The current time of the clock (in seconds).
)doc")

PYDOC(timestamp, R"doc(
The current timestamp of the clock (in nanoseconds).

Parameters
----------
timestamp : int
    The current timestamp of the clock (in nanoseconds).
)doc")

PYDOC(sleep_for, R"doc(
Set the GXF scheduler to sleep for a specified duration.

Parameters
----------
duration_ns : int
    The duration to sleep (in nanoseconds).
)doc")

PYDOC(sleep_until, R"doc(
Set the GXF scheduler to sleep until a specified timestamp.

Parameters
----------
target_time_ns : int
    The target timestamp (in nanoseconds).
)doc")

}  // namespace Clock

namespace RealtimeClock {

PYDOC(RealtimeClock, R"doc(
Real-time clock class.
)doc")

// Constructor
PYDOC(RealtimeClock_python, R"doc(
Realtime clock.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
initial_timestamp : float, optional
    The initial time offset used until time scale is changed manually.
initial_time_scale : float, optional
    The initial time scale used until time scale is changed manually.
use_time_since_epoch : bool, optional
    If ``True``, clock time is time since epoch + `initial_time_offset` at ``initialize()``.
    Otherwise clock time is `initial_time_offset` at ``initialize()``.
name : str, optional
    The name of the clock.
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

PYDOC(set_time_scale, R"doc(
Adjust the time scaling used by the clock.

Parameters
----------
time_scale : float, optional
    Durations (e.g. for periodic condition or sleep_for) are reduced by this scale value. A scale
    of 1.0 represents real-time while a scale of 2.0 would represent a clock where time elapses
    twice as fast.
)doc")

}  // namespace RealtimeClock

namespace ManualClock {

PYDOC(ManualClock, R"doc(
Manual clock class.
)doc")

// Constructor
PYDOC(ManualClock_python, R"doc(
Manual clock.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
initial_timestamp : int, optional
    The initial timestamp on the clock (in nanoseconds).
name : str, optional
    The name of the clock.
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

}  // namespace ManualClock

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_RESOURCES_PYDOC_HPP
