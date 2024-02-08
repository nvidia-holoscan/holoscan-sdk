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

#ifndef PYHOLOSCAN_CORE_DATAFLOW_TRACKER_PYDOC_HPP
#define PYHOLOSCAN_CORE_DATAFLOW_TRACKER_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace DataFlowMetric {

//  Constructor
PYDOC(DataFlowMetric, R"doc(
Enum class for DataFlowMetric type.
)doc")

}  // namespace DataFlowMetric

namespace DataFlowTracker {

//  Constructor
PYDOC(DataFlowTracker, R"doc(
Data Flow Tracker class.

The DataFlowTracker class is used to track the data flow metrics for different paths
between the root and leaf operators. This class is used by developers to get data flow
metrics either during the execution of the application and/or as a summary after the
application ends.
)doc")

PYDOC(get_metric_with_pathstring, R"doc(
Return the value of a metric for a given path.

If `metric` is DataFlowMetric::NUM_SRC_MESSAGES, then the function returns -1.

Parameters
----------
pathstring : str
    The path name string for which the metric is being queried
metric : holoscan.core.DataFlowMetric
    The metric to be queried.

Returns
-------
val : float
    The value of the metric for the given path

Notes
-----
There is also an overloaded version of this function that takes only the `metric` argument.
)doc")

PYDOC(get_num_paths, R"doc(
The number of tracked paths

Returns
-------
num_paths : int
    The number of tracked paths
)doc")

PYDOC(get_path_strings, R"doc(
Return an array of strings which are path names. Each path name is a
comma-separated list of Operator names in a path. The paths are agnostic to the edges between
two Operators.

Returns
-------
paths : list[str]
    A list of the path names.
)doc")

PYDOC(enable_logging, R"doc(
Enable logging of frames at the end of the every execution of a leaf
Operator.

A path consisting of an array of tuples in the form of (an Operator name, message
receive timestamp, message publish timestamp) is logged in a file. The logging does not take
into account the number of message to skip or discard or the threshold latency.

This function buffers a number of lines set by `num_buffered_messages` before flushing the buffer
to the log file.

Parameters
----------
filename : str
    The name of the log file.
num_buffered_messages : int
    The number of messages to be buffered before flushing the buffer to the log file.
)doc")

PYDOC(end_logging, R"doc(
Write out any remaining messages from the log buffer and close the file
)doc")

PYDOC(print, R"doc(
Print the result of the data flow tracking in pretty-printed format to the standard output
)doc")

PYDOC(set_discard_last_messages, R"doc(
Set the number of messages to discard at the end of the execution.

This does not affect the log file or the number of source messages metric.

Parameters
----------
num : int
    The number of messages to discard.
)doc")

PYDOC(set_skip_latencies, R"doc(
Set the threshold latency for which the end-to-end latency calculations will be done.
Any latency strictly less than the threshold latency will be ignored.

This does not affect the log file or the number of source messages metric.

Parameters
----------
threshold : int
    The threshold latency in milliseconds.
)doc")

PYDOC(set_skip_starting_messages, R"doc(
Set the number of messages to skip at the beginning of the execution.

This does not affect the log file or the number of source messages metric.

Parameters
----------
num : int
    The number of messages to skip.
)doc")

}  // namespace DataFlowTracker

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CORE_DATAFLOW_TRACKER_PYDOC_HPP
