/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_CORE_RESOURCE_PYDOC_HPP
#define PYHOLOSCAN_CORE_RESOURCE_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace Resource {

PYDOC(Resource, R"doc(
Base class representing either a wrapped C++ resource or native Python resource.
)doc")

//  Constructor
PYDOC(Resource_args_kwargs, R"doc(
Base class representing either a wrapped C++ resource or native Python resource.

Can be initialized with any number of Python positional and keyword arguments.

If a `name` keyword argument is provided, it must be a `str` and will be
used to set the name of the resource.

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
The name of the resource.

Returns
-------
name : str
)doc")

PYDOC(fragment, R"doc(
Fragment that the resource belongs to.

Returns
-------
name : holoscan.core.Fragment
)doc")

PYDOC(spec, R"doc(
The condition's ComponentSpec.
)doc")

PYDOC(setup, R"doc(
setup method for the resource.
)doc")

PYDOC(initialize, R"doc(
initialization method for the resource.
)doc")

PYDOC(description, R"doc(
YAML formatted string describing the resource.
)doc")

PYDOC(resource_type, R"doc(
Resource type.

`holoscan.core.Resource.ResourceType` enum representing the type of
the resource. The two types currently implemented are NATIVE and GXF.
)doc")

}  // namespace Resource

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_CORE_RESOURCE_PYDOC_HPP */
