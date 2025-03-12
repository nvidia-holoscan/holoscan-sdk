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

#ifndef PYHOLOSCAN_CORE_IO_SPEC_PYDOC_HPP
#define PYHOLOSCAN_CORE_IO_SPEC_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace ConnectorType {

PYDOC(ConnectorType, R"doc(
Enum representing the receiver type (for input specs) or transmitter type (for output specs).
)doc")

}  // namespace ConnectorType

namespace QueuePolicy {

PYDOC(QueuePolicy, R"doc(
Enum representing the queue policy for a receiver (for input specs) or transmitter (for output
specs).
)doc")

}  // namespace QueuePolicy

namespace IOType {

PYDOC(IOType, R"doc(
Enum representing the I/O specification type (input or output).
)doc")

}  // namespace IOType

namespace IOSpec {

namespace IOSize {

//  Constructor
PYDOC(IOSize, R"doc(
I/O size class.

Parameters
----------
size : int
    The size of the input/output queue.
)doc")

PYDOC(size, R"doc(
The size of the I/O size class.

Returns
-------
size : int
)doc")
}  // namespace IOSize

//  Constructor
PYDOC(IOSpec, R"doc(
I/O specification class.

Parameters
----------
op_spec : holoscan.core.OperatorSpec
    Operator specification class of the associated operator.
name : str
    The name of the IOSpec object.
io_type : holoscan.core.IOSpec.IOType
    Enum indicating whether this is an input or output specification.
)doc")

PYDOC(name, R"doc(
The name of the I/O specification class.

Returns
-------
name : str
)doc")

PYDOC(io_type, R"doc(
The type (input or output) of the I/O specification class.

Returns
-------
io_type : holoscan.core.IOSpec.IOType
)doc")

PYDOC(connector_type, R"doc(
The receiver or transmitter type of the I/O specification class.

Returns
-------
connector_type : holoscan.core.IOSpec.ConnectorType
)doc")

PYDOC(conditions, R"doc(
List of Condition objects associated with this I/O specification.

Returns
-------
condition : list of holoscan.core.Condition
)doc")

PYDOC(condition, R"doc(
Add a condition to this input/output.

The following ConditionTypes are supported:

- `ConditionType.NONE`
- `ConditionType.MESSAGE_AVAILABLE`
- `ConditionType.DOWNSTREAM_MESSAGE_AFFORDABLE`
- `ConditionType.COUNT`
- `ConditionType.BOOLEAN`

Parameters
----------
kind : holoscan.core.ConditionType
    The type of the condition.
**kwargs
    Python keyword arguments that will be cast to an `ArgList` associated
    with the condition.

Returns
-------
obj : holoscan.core.IOSpec
    The self object.
)doc")

PYDOC(connector, R"doc(
Add a connector (transmitter or receiver) to this input/output.

The following ConditionTypes are supported:

- `IOSpec.ConnectorType.DEFAULT`
- `IOSpec.ConnectorType.DOUBLE_BUFFER`
- `IOSpec.ConnectorType.UCX`

If this method is not been called, the IOSpec's `connector_type` will be
`ConnectorType.DEFAULT` which will result in a DoubleBuffered receiver or
or transmitter being used (or their annotated variant if flow tracking is
enabled).

Parameters
----------
kind : holoscan.core.IOSpec.ConnectorType
    The type of the connector. For example for type `IOSpec.ConnectorType.DOUBLE_BUFFER`, a
    `holoscan.resources.DoubleBufferReceiver` will be used for an input port and a
    `holoscan.resources.DoubleBufferTransmitter` will be used for an output port.
**kwargs
    Python keyword arguments that will be cast to an `ArgList` associated
    with the resource (connector).

Returns
-------
obj : holoscan.core.IOSpec
    The self object.

Notes
-----
This is an overloaded function. Additional variants exist:

1.) A variant with no arguments will just return the `holoscan.core.Resource` corresponding to
the transmitter or receiver used by this `IOSpec` object. If None was explicitly set, it will
return ``None``.

2.) A variant that takes a single `holoscan.core.Resource` corresponding to a transmitter or
receiver as an argument. This sets the transmitter or receiver used by the `IOSpec` object.

)doc")

PYDOC(queue_size, R"doc(
The size of the input/output queue.

Notes
-----
This value is only used for initializing input ports. The queue size is set by the
'OperatorSpec.input()' method or this property.
If the queue size is set to 'any size' (IOSpec::kAnySize in C++ or IOSpec.ANY_SIZE in Python),
the connector/condition settings will be ignored.
If the queue size is set to other values, the default connector (DoubleBufferReceiver/UcxReceiver)
and condition (MessageAvailableCondition) will use the queue size for initialization
('capacity' for the connector and 'min_size' for the condition) if they are not set.
)doc")

PYDOC(queue_policy, R"doc(
The queue policy used by the input (or output) port's receiver (or transmitter).

Notes
-----
This value is only used for initializing input and output ports. The policy is set by the
`OperatorSpec.input`, `OperatorSpec.output` or `IOSpec.queue_policy` method.

The following IOSpec.QueuePolicy values are supported:

   - QueuePolicy.POP : If the queue is full, pop the oldest item, then add the new one.
   - QueuePolicy::REJECT : If the queue is full, reject (discard) the new item.
   - QueuePolicy.REJECT : If the queue is full, reject (discard) the new item.
   - QueuePolicy.FAULT : If the queue is full, log a warning and reject the new item.

)doc")

}  // namespace IOSpec

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CORE_IO_SPEC_PYDOC_HPP
