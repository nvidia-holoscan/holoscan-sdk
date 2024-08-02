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

#endif  // PYHOLOSCAN_CORE_IO_CONTEXT_PYDOC_HPP
