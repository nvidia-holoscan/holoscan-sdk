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

#ifndef PYHOLOSCAN_CORE_KWARG_HANDLING_PYDOC_HPP
#define PYHOLOSCAN_CORE_KWARG_HANDLING_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace KwargHandling {

PYDOC(py_object_to_arg, R"doc(
Utility that converts a single python argument to a corresponding `Arg` type.

Parameters
----------
value : Any
    The python value to convert.

Returns
-------
obj : holoscan.core.Arg
    `Arg` class corresponding to the provided value. For example a Python float
    will become an `Arg` containing a C++ double while a list of Python ints
    would become an `Arg` corresponding to a ``std::vector<uint64_t>``.
name : str, optional
    A name to assign to the argument.
)doc")

PYDOC(kwargs_to_arglist, R"doc(
Utility that converts a set of python keyword arguments to an `ArgList`.

Parameters
----------
**kwargs
    The python keyword arguments to convert.

Returns
-------
arglist : holoscan.core.ArgList
    `ArgList` class corresponding to the provided keyword values. The argument
    names will match the keyword names. Values will be converted as for
    `py_object_to_arg`.
)doc")

PYDOC(arg_to_py_object, R"doc(
Utility that converts an `Arg` to a corresponding Python object.

Parameters
----------
arg : holoscan.core.Arg
    The argument to convert.

Returns
-------
obj : Any
    Python object corresponding to the provided argument. For example,
    an argument of any integer type will become a Python `int` while
    `std::vector<double>` would become a list of Python floats.
)doc")

PYDOC(arglist_to_kwargs, R"doc(
Utility that converts an `ArgList` to a Python kwargs dictionary.

Parameters
----------
arglist : holoscan.core.ArgList
    The argument list to convert.

Returns
-------
kwargs : dict
    Python dictionary with keys matching the names of the arguments in
    `ArgList`. The values will be converted as for `arg_to_py_object`.
)doc")

}  // namespace KwargHandling

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CORE_KWARG_HANDLING_PYDOC_HPP
