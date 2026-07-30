/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
