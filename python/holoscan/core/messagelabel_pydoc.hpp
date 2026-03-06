/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_CORE_MESSAGELABEL_PYDOC_HPP
#define PYHOLOSCAN_CORE_MESSAGELABEL_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace OperatorTimestampLabel {

PYDOC(OperatorTimestampLabel, R"doc(
Timestamp label for a Holoscan Operator.

This struct stores information about the timestamps when an operator receives from
an input and when it publishes to an output. It also holds a reference to the
operator name.

This class is used by `MessageLabel` to create an array of operators representing
a path in the data flow.
)doc")

PYDOC(OperatorTimestampLabel_default, R"doc(
Construct a new OperatorTimestampLabel object with default values.
)doc")

PYDOC(OperatorTimestampLabel_name, R"doc(
Construct a new OperatorTimestampLabel object.

Parameters
----------
op_name : str
    The fully qualified name of the operator for which the timestamp label is created.
)doc")

PYDOC(OperatorTimestampLabel_full, R"doc(
Construct a new OperatorTimestampLabel object.

Parameters
----------
op_name : str
    The fully qualified name of the operator.
rec_timestamp : int
    The receive timestamp in microseconds.
pub_timestamp : int
    The publish timestamp in microseconds.
)doc")

PYDOC(operator_name, R"doc(
The fully qualified name of the operator.
)doc")

PYDOC(rec_timestamp, R"doc(
The timestamp when an operator receives from an input (in microseconds).
For a root operator, it is the start of the compute call.
)doc")

PYDOC(pub_timestamp, R"doc(
The timestamp when an operator publishes an output (in microseconds).
For a leaf operator, it is the end of the compute call.
)doc")

PYDOC(set_pub_timestamp_to_current, R"doc(
Set the publish timestamp to the current time.
)doc")

}  // namespace OperatorTimestampLabel

namespace MessageLabel {

PYDOC(MessageLabel, R"doc(
Class representing a message label for data flow tracking.

When data flow tracking is enabled, a `MessageLabel` is attached to every message being
communicated between operators. It contains a vector of paths, where each path is a vector
of operator timestamp labels. It also stores frame numbers for root operators.

Each path represents a unique trajectory that a message took through the application
graph, capturing timing information at each operator along the way.
)doc")

PYDOC(MessageLabel_default, R"doc(
Construct a new `MessageLabel` object with default values.
)doc")

PYDOC(MessageLabel_paths, R"doc(
Construct a new `MessageLabel` object from a list of paths.

Parameters
----------
paths : list of list of OperatorTimestampLabel
    The list of paths to create the `MessageLabel` from.
)doc")

PYDOC(num_paths, R"doc(
Get the number of paths in the `MessageLabel`.

Returns
-------
int
    The number of paths in the `MessageLabel`.
)doc")

PYDOC(get_all_path_names, R"doc(
Get all path names as formatted strings.

Each path name is a comma-separated list of operator names.

Returns
-------
list of str
    The list of path name strings.
)doc")

PYDOC(paths, R"doc(
Get the paths in the `MessageLabel`.

Returns
-------
list of list of OperatorTimestampLabel
    The list of timestamped paths.
)doc")

PYDOC(get_e2e_latency, R"doc(
Get the end-to-end latency of a path in microseconds.

Parameters
----------
index : int
    The index of the path for which to get the latency.

Returns
-------
int
    The end-to-end latency of the path in microseconds.
)doc")

PYDOC(get_e2e_latency_ms, R"doc(
Get the end-to-end latency of a path in milliseconds.

Parameters
----------
index : int
    The index of the path for which to get the latency.

Returns
-------
float
    The end-to-end latency of the path in milliseconds.
)doc")

PYDOC(get_path, R"doc(
Get the timestamped path at the given index.

Parameters
----------
index : int
    The index of the path to get.

Returns
-------
list of OperatorTimestampLabel
    The timestamped path at the given index.
)doc")

PYDOC(get_path_name, R"doc(
Get the path name string.

The path name is a comma-separated list of operator names.

Parameters
----------
index : int
    The index of the path.

Returns
-------
str
    The path name string.
)doc")

PYDOC(get_operator, R"doc(
Get the `OperatorTimestampLabel` at the given path and operator index.

Parameters
----------
path_index : int
    The path index of the `OperatorTimestampLabel` to get.
op_index : int
    The operator index of the `OperatorTimestampLabel` to get.

Returns
-------
OperatorTimestampLabel
    The operator timestamp label at the given path and operator index.
)doc")

PYDOC(has_operator, R"doc(
Check if an operator is present in the `MessageLabel`.

Parameters
----------
op_name : str
    The name of the operator to check.

Returns
-------
list of int
    List of path indexes where the operator is present. Returns an empty list if the
    operator is not present in any path.
)doc")

PYDOC(to_string, R"doc(
Convert the `MessageLabel` to a formatted string.

Returns
-------
str
    The formatted string representing the `MessageLabel` with all paths and operators
    with their publish and receive timestamps.
)doc")

PYDOC(print_all, R"doc(
Print the `MessageLabel` to standard output.

Prints the string representation with a heading for the `MessageLabel`.
)doc")

}  // namespace MessageLabel

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_CORE_MESSAGELABEL_PYDOC_HPP */
