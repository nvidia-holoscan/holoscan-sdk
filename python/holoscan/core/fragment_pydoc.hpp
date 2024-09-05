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

#ifndef PYHOLOSCAN_CORE_FRAGMENT_PYDOC_HPP
#define PYHOLOSCAN_CORE_FRAGMENT_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace Config {

PYDOC(Config, R"doc(
Configuration class.

Represents configuration parameters as read from a YAML file.
)doc")

//  Constructor
PYDOC(Config_kwargs, R"doc(
Configuration class.

Represents configuration parameters as read from a YAML file.

Parameters
----------
config_file : str
    The path to the configuration file (in YAML format).
prefix : str, optional
    TODO
)doc")

PYDOC(config_file, R"doc(
The configuration file (in YAML format) associated with the Config object.
)doc")

PYDOC(prefix, R"doc(
TODO
)doc")

}  // namespace Config

namespace Fragment {

//  Constructor
PYDOC(Fragment, R"doc(
Fragment class.
)doc")

PYDOC(name, R"doc(
The fragment's name.

Returns
-------
name : str
)doc")

PYDOC(application, R"doc(
The application associated with the fragment.

Returns
-------
app : holoscan.core.Application
)doc")

//  Constructor
PYDOC(config_kwargs, R"doc(
Configuration class.

Represents configuration parameters as read from a YAML file.

Parameters
----------
config : str or holoscan.core.Config
    The path to the configuration file (in YAML format) or a `holoscan.core.Config`
    object.
prefix : str, optional
    Prefix path for the` config` file. Only available in the overloaded variant
    that takes a string for `config`.
)doc")

//  Constructor
PYDOC(config_keys, R"doc(
The set of keys present in the fragment's configuration file.
)doc")

PYDOC(graph, R"doc(
Get the computation graph (Graph node is an Operator) associated with the fragment.
)doc")

PYDOC(executor, R"doc(
Get the executor associated with the fragment.
)doc")

PYDOC(is_metadata_enabled, R"doc(
Property to get or set the boolean controlling whether operator metadata transmission is enabled.
)doc")

PYDOC(from_config, R"doc(
Retrieve parameters from the associated configuration.

Parameters
----------
key : str
    The key within the configuration file to retrieve. This can also be a specific
    component of the parameter via syntax `'key.sub_key'`.

Returns
-------
args : holoscan.core.ArgList
    An argument list associated with the key.
)doc")

PYDOC(kwargs, R"doc(
Retrieve a dictionary parameters from the associated configuration.

Parameters
----------
key : str
    The key within the configuration file to retrieve. This can also be a specific
    component of the parameter via syntax `'key.sub_key'`.

Returns
-------
kwargs : dict
    A Python dict containing the parameters in the configuration file under the
    specified key.
)doc")

PYDOC(add_operator, R"doc(
Add an operator to the fragment.

Parameters
----------
op : holoscan.core.Operator
    The operator to add.
)doc")

PYDOC(add_flow_pair, R"doc(
Connect two operators associated with the fragment.

Parameters
----------
upstream_op : holoscan.core.Operator
    Source operator.
downstream_op : holoscan.core.Operator
    Destination operator.
port_pairs : Sequence of (str, str) tuples
    Sequence of ports to connect. The first element of each 2-tuple is a port
    from `upstream_op` while the second element is the port of `downstream_op`
    to which it connects.

Notes
-----
This is an overloaded function. Additional variants exist:

1.) For the Application class there is a variant where the first two arguments are of type
`holoscan.core.Fragment` instead of `holoscan.core.Operator`. This variant is used in building
multi-fragment applications.
2.) There are also variants that omit the `port_pairs` argument that are applicable when there is
only a single output on the upstream operator/fragment and a single input on the downstream
operator/fragment.

)doc")

PYDOC(compose, R"doc(
The compose method of the Fragment.

This method should be called after `config`, but before `run` in order to
compose the computation graph.
)doc")

PYDOC(scheduler, R"doc(
Get the scheduler to be used by the Fragment.
)doc")

PYDOC(scheduler_kwargs, R"doc(
Assign a scheduler to the Fragment.

Parameters
----------
scheduler : holoscan.core.Scheduler
    A scheduler class instance to be used by the underlying GXF executor. If unspecified,
    the default is a `holoscan.gxf.GreedyScheduler`.
)doc")

PYDOC(network_context, R"doc(
Get the network context to be used by the Fragment
)doc")

PYDOC(network_context_kwargs, R"doc(
Assign a network context to the Fragment

Parameters
----------
network_context : holoscan.core.NetworkContext
    A network_context class instance to be used by the underlying GXF executor.
    If unspecified, no network context will be used.
)doc")

PYDOC(track, R"doc(
The track method of the fragment (or application).

This method enables data frame flow tracking and returns a DataFlowTracker object which can be
used to display metrics data for profiling an application.

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

PYDOC(run, R"doc(
The run method of the Fragment.

This method runs the computation. It must have first been initialized via
`config` and `compose`.
)doc")

}  // namespace Fragment

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CORE_FRAGMENT_PYDOC_HPP
