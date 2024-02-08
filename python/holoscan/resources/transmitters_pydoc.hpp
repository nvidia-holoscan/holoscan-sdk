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

#ifndef PYHOLOSCAN_RESOURCES_COMPONENT_SERIALIZERS_PYDOC_HPP
#define PYHOLOSCAN_RESOURCES_COMPONENT_SERIALIZERS_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

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
receiver_address : str, optional
    The IP address used by the transmitter.
local_address : str, optional
    The local IP address to use for connection.
port : int, optional
    The network port used by the transmitter.
local_port : int, optional
    The local network port to use for connection.
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

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_RESOURCES_COMPONENT_SERIALIZERS_PYDOC_HPP
