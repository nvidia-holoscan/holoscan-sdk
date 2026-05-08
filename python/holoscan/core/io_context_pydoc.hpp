/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_CORE_IO_CONTEXT_PYDOC_HPP
#define PYHOLOSCAN_CORE_IO_CONTEXT_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace Message {

PYDOC(Message, R"doc(
Class representing a message.

A message is a data structure that is used to pass data between operators.
It wraps a ``std::any`` object and provides a type-safe interface to access the data.

This class is used by the `holoscan::gxf::GXFWrapper` to support the Holoscan native operator.
The `holoscan::gxf::GXFWrapper` will hold the object of this class and delegate the message to the
Holoscan native operator.
)doc")

}  // namespace Message

namespace InputContext {

PYDOC(InputContext, R"doc(
Class representing an input context.
)doc")

PYDOC(receive, R"doc(
Receive an object from the specified port.

Parameters
----------
name : str
    The name of the port to receive the object from.

Returns
-------
data : object
    The received Python object. If no entities were received on the port, `data` will be
    ``None``.

)doc")

PYDOC(receive_cuda_stream, R"doc(
Get the operator's internal CUDA stream, synchronizing any upstream streams to it.

This is the recommended method for stream handling in most operators. It performs several
operations:

1. **Synchronizes upstream streams** to the operator's internal stream using non-blocking
   CUDA events (``cudaEventRecord`` / ``cudaStreamWaitEvent``). This ensures upstream GPU work
   completes before this operator's work begins, without blocking the CPU.
2. **Sets the CUDA device** to match the internal stream's device.
3. **Configures all output ports** to automatically emit the internal stream ID when ``emit()``
   is called.
4. **Returns the internal stream pointer** for use in kernels and async memory operations.

.. note::
   The ``receive()`` method must be called for ``input_port_name`` **before** calling this
   method. The ``receive()`` call captures stream IDs from incoming messages.

Parameters
----------
input_port_name : str, optional
    The name of the input port. Can be omitted if the operator has only one input port.
allocate : bool, optional
    If True (default), allocates an internal stream if not already allocated. If False,
    the first received stream is used as the internal stream.
sync_to_default : bool, optional
    If True, also synchronizes the internal stream to ``cudaStreamDefault``. Default is False.

Returns
-------
stream_ptr : int
    The memory address of the operator's internal cudaStream_t (reused across all ``compute()``
    calls). Returns 0 (cudaStreamDefault) if no stream pool is available and no stream was found.

)doc")

PYDOC(receive_cuda_streams, R"doc(
Retrieve the raw CUDA streams found on an input port (advanced use).

Unlike ``receive_cuda_stream``, this method does **not** perform any synchronization, does not
allocate an internal stream, does not set the CUDA device, and does not configure output ports.
It simply returns the raw stream information found in the received messages.

This method is intended for advanced use cases where manual stream management is required.
For most operators, use ``receive_cuda_stream`` instead.

.. note::
   The ``receive()`` method must be called for ``input_port_name`` **before** calling this
   method. The ``receive()`` call captures stream IDs from incoming messages.

Parameters
----------
input_port_name : str, optional
    The name of the input port. Can be omitted if the operator has only one input port.

Returns
-------
stream_ptrs : list[int or None]
    The memory addresses of the cudaStream_t for each message. In normal operation, the list
    length matches the number of messages on the port, with ``None`` for messages without a
    stream. If stream handling is unavailable (e.g., CudaObjectHandler not initialized), an
    empty list is returned.

)doc")

PYDOC(get_acquisition_timestamp, R"doc(
Get the acquisition timestamp corresponding to a given input port.

Parameters
----------
input_port_name : str, optional
    The name of the input port to receive the object from. Can be left empty if there is only
    one input port on the operator.

Returns
-------
timestamp : int or None
    Returns the timestamp (in nanoseconds). If the upstream operator did not emit a timestamp or
    the input port name does not exist, this timestamp will be ``None``.

)doc")

PYDOC(get_acquisition_timestamps, R"doc(
Get the acquisition timestamsp corresponding to all messages received on a given input port.

Parameters
----------
input_port_name : str, optional
    The name of the input port to receive the object from. Can be left empty if there is only
    one input port on the operator.

Returns
-------
timestamps : list[int or None]
    Returns the timestamps (in nanoseconds). Values of None will be present for any of the
    received messages that did not contain a timestamp.

)doc")

}  // namespace InputContext

namespace OutputContext {

PYDOC(OutputContext, R"doc(
Class representing an output context.
)doc")

PYDOC(emit, R"doc(
Emit a Python or C++ object on the specified port.

Parameters
----------
data : object
    The Python object to emit. If it is a tensor-like object it will be transmitted as a C++
    holoscan::Tensor for compatibility with C++ operators expecting a holoscan::Tensor (no copy of
    the data is required when converting to the C++ tensor type). Similarly, if `data` is a
    dictionary where all keys are strings and all values are tensor-like objects then it will be
    transmitted as a holoscan::TensorMap for compatibility with Holoscan C++ operators. Similarly
    if it is detected that the output port is connected across fragments in a distributed
    application, then serialization of the data will automatically be performed so that it can be
    sent over the network via UCX.
name : str
    The name of the port to emit the object on.
emitter_name : str, optional
    This can be specified to force emitting as a different type than would be chosen by default.
    For example, if `data` is a Python `str` object it would normally be emitted as a Python
    string. However, to send the string as a `std::string` as expected by a downstream C++
    operator, one could set ``emitter_name="std::string"`` to make sure the data will be cast to
    this type. In general, any type that has been registered with the type registry can be
    specified here as long as the provided object can be cast to that type. To get a list of the
    currently registered type names, call ``holoscan.core.io_type_registry.registered_types()``.
)doc")

PYDOC(set_cuda_stream, R"doc(
Set a CUDA stream to be emitted on a given output port.

When using ``receive_cuda_stream``, output ports are automatically configured to emit the
operator's internal stream, so this method is typically not needed. Use this method when:

- Using ``allocate_cuda_stream`` to allocate a stream for a root operator
- Using ``receive_cuda_streams`` for manual stream handling

This method must be called **before** the corresponding ``emit()`` call for the port.

Parameters
----------
stream_ptr : int
    The memory address of the cudaStream_t to emit. Must be a Holoscan-managed stream (one
    returned by ``receive_cuda_stream``, ``receive_cuda_streams``, or ``allocate_cuda_stream``).
output_port_name : str, optional
    The name of the output port. Can be omitted if the operator has only one output port.
)doc")

}  // namespace OutputContext

namespace EmitterReceiverRegistry {

PYDOC(EmitterReceiverRegistry, R"doc(
Registry of methods to emit/receive different types.
)doc")

PYDOC(registered_types, R"doc(
List of types with an emitter and/or receiver registered

Returns
-------
names : list of str
    The list of registered emitter/receiver names.
)doc")

}  // namespace EmitterReceiverRegistry

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_CORE_IO_CONTEXT_PYDOC_HPP */
