/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_CORE_CONDITION_PYDOC_HPP
#define PYHOLOSCAN_CORE_CONDITION_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace ConditionType {

//  Constructor
PYDOC(ConditionType, R"doc(
Enum class for Condition types.
)doc")

}  // namespace ConditionType

namespace SchedulingStatusType {

//  Constructor
PYDOC(SchedulingStatusType, R"doc(
Enum class for Condition scheduling status.
)doc")

}  // namespace SchedulingStatusType

namespace Condition {

PYDOC(Condition, R"doc(
Base class representing either a wrapped C++ condition or native Python condition.
)doc")

//  Constructor
PYDOC(Condition_args_kwargs, R"doc(
Base class representing either a wrapped C++ condition or native Python condition.

Can be initialized with any number of Python positional and keyword arguments.

If a `name` keyword argument is provided, it must be a `str` and will be
used to set the name of the condition.

If a `fragment` keyword argument is provided, it must be of type
`holoscan.core.Fragment` (or
`holoscan.core.Application`). A single `Fragment` object can also be
provided positionally instead.

Any other arguments will be cast from a Python argument type to a C++ `Arg`
and stored in ``self.args``. (For details on how the casting is done, see the
`py_object_to_arg` utility).

Parameters
----------
\*args
    Positional arguments.
\*\*kwargs
    Keyword arguments.

Raises
------
RuntimeError
    If `name` kwarg is provided, but is not of `str` type.
    If multiple arguments of type `Fragment` are provided.
    If any other arguments cannot be converted to `Arg` type via `py_object_to_arg`.
)doc")

PYDOC(name, R"doc(
The name of the condition.

Returns
-------
name : str
)doc")

PYDOC(fragment, R"doc(
Fragment that the condition belongs to.

Returns
-------
name : holoscan.core.Fragment
)doc")

PYDOC(spec, R"doc(
The condition's ComponentSpec.
)doc")

PYDOC(setup, R"doc(
setup method for the condition.
)doc")

PYDOC(initialize, R"doc(
initialization method for the condition.
)doc")

PYDOC(description, R"doc(
YAML formatted string describing the condition.
)doc")

PYDOC(condition_type, R"doc(
Condition type.

`holoscan.core.Condition.ConditionComponentType` enum representing the type of
the condition. The two types currently implemented are NATIVE and GXF.
)doc")

PYDOC(receiver, R"doc(
Get the receiver used by an input port of the operator this condition is associated with.

Parameters
----------
port_name : str
    The name of the input port.

Returns
-------
receiver : holoscan.resources.Receiver
    The receiver used by this input port. Will be None if the port does not exist.
)doc")

PYDOC(transmitter, R"doc(
Get the transmitter used by an output port of the operator this condition is associated with.

Parameters
----------
port_name : str
    The name of the output port.

Returns
-------
transmitter : holoscan.resources.Transmitter or None
    The transmitter used by this output port. Will be None if the port does not exist.
)doc")

PYDOC(notify_scheduler, R"doc(
Notify the scheduler that an asynchronous event has completed.

This method is used by event-based conditions (those returning
`SchedulingStatusType.WAIT_EVENT` from `check()`) to signal to the scheduler
that the condition is now ready to be re-evaluated.

This method can be called from any thread (e.g., a CUDA host callback or a
worker thread). It is thread-safe.

Returns
-------
success : bool
    True if the notification was successful, False otherwise.

Notes
-----
This method should be called after updating the condition's internal state
to indicate readiness. For example:

.. code-block:: python

    def my_callback(self):
        self.state = State.EVENT_COMPLETE
        self.notify_scheduler()

See Also
--------
holoscan.core.SchedulingStatusType.WAIT_EVENT
)doc")

}  // namespace Condition

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CORE_CONDITION_PYDOC_HPP
