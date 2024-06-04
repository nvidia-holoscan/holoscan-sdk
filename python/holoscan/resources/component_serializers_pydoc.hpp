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

#ifndef PYHOLOSCAN_RESOURCES_COMPONENT_SERIALIZERS_PYDOC_HPP
#define PYHOLOSCAN_RESOURCES_COMPONENT_SERIALIZERS_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace StdComponentSerializer {

PYDOC(StdComponentSerializer, R"doc(
Serializer for GXF Timestamp and Tensor components.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
name : str, optional
    The name of the serializer.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the resource.

Returns
-------
str
    The GXF type name of the resource
)doc")

PYDOC(setup, R"doc(
Define the component specification.

Parameters
----------
spec : holoscan.core.ComponentSpec
    Component specification associated with the resource.
)doc")

PYDOC(initialize, R"doc(
Initialize the resource

This method is called only once when the resource is created for the first
time, and uses a light-weight initialization.
)doc")

}  // namespace StdComponentSerializer

namespace UcxComponentSerializer {

PYDOC(UcxComponentSerializer, R"doc(
UCX component serializer.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
allocator : holoscan.resource.Allocator
    The memory allocator for tensor components.
name : str, optional
    The name of the component serializer.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the resource.

Returns
-------
str
    The GXF type name of the resource
)doc")

PYDOC(setup, R"doc(
Define the component specification.

Parameters
----------
spec : holoscan.core.ComponentSpec
    Component specification associated with the resource.
)doc")

}  // namespace UcxComponentSerializer

namespace UcxHoloscanComponentSerializer {

PYDOC(UcxHoloscanComponentSerializer, R"doc(
UCX Holoscan component serializer.
)doc")

// Constructor
PYDOC(UcxHoloscanComponentSerializer_python, R"doc(
UCX Holoscan component serializer.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
allocator : holoscan.resource.Allocator
    The memory allocator for tensor components.
name : str, optional
    The name of the component serializer.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the resource.

Returns
-------
str
    The GXF type name of the resource
)doc")

PYDOC(setup, R"doc(
Define the component specification.

Parameters
----------
spec : holoscan.core.ComponentSpec
    Component specification associated with the resource.
)doc")

}  // namespace UcxHoloscanComponentSerializer

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_RESOURCES_COMPONENT_SERIALIZERS_PYDOC_HPP
