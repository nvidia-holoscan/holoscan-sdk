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

#ifndef PYHOLOSCAN_CORE_NETWORK_CONTEXT_PYDOC_HPP
#define PYHOLOSCAN_CORE_NETWORK_CONTEXT_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace NetworkContext {

PYDOC(NetworkContext, R"doc(
Class representing a network context.
)doc")

//  Constructor
PYDOC(NetworkContext_args_kwargs, R"doc(
Class representing a network context.

Parameters
----------
*args
    Positional arguments.
**kwargs
    Keyword arguments.

Raises
------
RuntimeError
    If `name` kwarg is provided, but is not of `str` type.
    If multiple arguments of type `Fragment` are provided.
    If any other arguments cannot be converted to `Arg` type via `py_object_to_arg`.
)doc")

PYDOC(name, R"doc(
The name of the network context.

Returns
-------
name : str
)doc")

PYDOC(fragment, R"doc(
Fragment that the network context belongs to.

Returns
-------
name : holoscan.core.Fragment
)doc")

PYDOC(spec, R"doc(
The network context's ComponentSpec.
)doc")

PYDOC(setup, R"doc(
setup method for the network context.
)doc")

PYDOC(initialize, R"doc(
initialization method for the network context.
)doc")

}  // namespace NetworkContext

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CORE_NETWORK_CONTEXT_PYDOC_HPP
