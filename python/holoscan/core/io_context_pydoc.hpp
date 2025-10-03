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
Receive the CUDA stream associated with the specified input port.

Parameters
----------
input_port_name : str, optional
	The name of the input port to receive the object from.
allocate : bool, optional
	If True, the operator should allocate its own CUDA stream and synchronize any incoming streams
	to it. The stream returned by this function is then the internally, allocated stream.

Returns
-------
stream_ptr : int
	The memory address of the underlying cudaStream_t.

)doc")

PYDOC(receive_cuda_streams, R"doc(
Receive a list of CUDA streams associated with the specified input port.

The size of the list will be equal to the number of messages received on the port. For messages
not containing a CudaStream, the corresponding entry in the list will be ``None``.

Parameters
----------
input_port_name : str, optional
	The name of the input port to receive the object from.

Returns
-------
stream_ptrs : list[int]
	The memory addresses of the underlying cudaStream_t for each message. For any messages without
	a stream, the list will contain None.

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
Specify a CUDA stream to be emitted along with any data on the specified output port.

Parameters
----------
stream_ptr : int
	The memory address of the underlying cudaStream_t to be emitted.
output_port_name : str, optional
	The name of the output port to emit the stream on. Can be unspecified if there is only a single
	output port on the operator.
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
