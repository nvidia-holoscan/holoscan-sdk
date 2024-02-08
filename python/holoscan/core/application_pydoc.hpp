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

#ifndef PYHOLOSCAN_CORE_APPLICATION_PYDOC_HPP
#define PYHOLOSCAN_CORE_APPLICATION_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace Application {

// Constructor
PYDOC(Application, R"doc(
Application class.

This constructor parses the command line for flags that are recognized by App Driver/Worker,
and removes all recognized flags so users can use the remaining flags for their own purposes.

If the arguments are not specified, the arguments are retrieved from ``sys.executable`` and
``sys.argv``.

The arguments after processing arguments (parsing Holoscan-specific flags and removing them)
are accessible through the ``argv`` attribute.

Parameters
----------
argv : List[str]
    The command line arguments to parse. The first item should be the path to the python executable.
    If not specified, ``[sys.executable, *sys.argv]`` is used.

Examples
--------
>>> from holoscan.core import Application
>>> import sys
>>> Application().argv == sys.argv
True
>>> Application([]).argv == sys.argv
True
>>> Application([sys.executable, *sys.argv]).argv == sys.argv
True
>>> Application(["python3", "myapp.py", "--driver", "--address=10.0.0.1", "my_arg1"]).argv
['myapp.py', 'my_arg1']
)doc")

PYDOC(name, R"doc(
The application's name.

Returns
-------
name : str
)doc")

PYDOC(description, R"doc(
The application's description.

Returns
-------
description : str
)doc")

PYDOC(version, R"doc(
The application's version.

Returns
-------
version : str
)doc")

PYDOC(argv, R"doc(
The command line arguments after processing flags.
This does not include the python executable like `sys.argv` does.

Returns
-------
argv : list of str
)doc")

PYDOC(options, R"doc(
The reference to the CLI options.

Returns
-------
options : holoscan.core.CLIOptions
)doc")

PYDOC(fragment_graph, R"doc(
Get the computation graph (Graph node is a Fragment) associated with the application.
)doc")

PYDOC(add_operator, R"doc(
Add an operator to the application.

Parameters
----------
op : holoscan.core.Operator
    The operator to add.
)doc")

PYDOC(add_fragment, R"doc(
Add a fragment to the application.

Parameters
----------
frag : holoscan.core.Fragment
    The fragment to add.
)doc")

PYDOC(compose, R"doc(
The compose method of the application.

This method should be called after `config`, but before `run` in order to
compose the computation graph.
)doc")

PYDOC(run, R"doc(
The run method of the application.

This method runs the computation. It must have first been initialized via
`config` and `compose`.
)doc")

PYDOC(track, R"doc(
The track method of the application.

This method enables data frame flow tracking and returns
a DataFlowTracker object which can be used to display metrics data
for profiling an application.

Parameters
----------
num_start_messages_to_skip : int
    The number of messages to skip at the beginning.
num_last_messages_to_discard : int
    The number of messages to discard at the end.
latency_threshold : int
    The minimum end-to-end latency in milliseconds to account for in the
    end-to-end latency metric calculations
)doc")
}  // namespace Application

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CORE_APPLICATION_PYDOC_HPP
