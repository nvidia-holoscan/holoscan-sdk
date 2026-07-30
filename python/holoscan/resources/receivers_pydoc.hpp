/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_RESOURCES_RECEIVERS_PYDOC_HPP
#define PYHOLOSCAN_RESOURCES_RECEIVERS_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace Receiver {

PYDOC(Receiver, R"doc(
Base GXF receiver class.
)doc")

PYDOC(size, R"doc(
The size of the receiver queue's main stage.
)doc")

PYDOC(back_size, R"doc(
The size of the receiver queue's back stage.
)doc")

PYDOC(capacity, R"doc(
The capacity of the receiver queue's main stage.
)doc")

}  // namespace Receiver

namespace DoubleBufferReceiver {

PYDOC(DoubleBufferReceiver, R"doc(
Receiver using a double-buffered queue.

New messages are first pushed to a back stage.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment or subgraph to assign the resource to.
capacity : int, optional
    The capacity of the receiver.
policy : int, optional
    The policy to use (0=pop, 1=reject, 2=fault).
name : str, optional
    The name of the receiver.
)doc")

}  // namespace DoubleBufferReceiver

namespace UcxReceiver {

PYDOC(UcxReceiver, R"doc(
UCX network receiver using a double-buffered queue.

New messages are first pushed to a back stage.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment or subgraph to assign the resource to.
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

}  // namespace UcxReceiver

namespace AsyncBufferReceiver {

PYDOC(AsyncBufferReceiver, R"doc(
Receiver using an asynchronous buffer.

The latest message is received asynchronously.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment or subgraph to assign the resource to.
name : str, optional
    The name of the receiver.
)doc")

}  // namespace AsyncBufferReceiver

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_RESOURCES_RECEIVERS_PYDOC_HPP
