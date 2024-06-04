/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_RESOURCES_GXF_COMPONENT_RESOURCE_PYDOC_HPP
#define PYHOLOSCAN_RESOURCES_GXF_COMPONENT_RESOURCE_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace GXFComponentResource {

PYDOC(GXFComponentResource, R"doc(
Class that wraps a GXF Component as a Holoscan Resource.

Parameters
----------
fragment : holoscan.core.Fragment (constructor only)
    The fragment that the resource belongs to.
gxf_typename : str
    The GXF type name that identifies the specific GXF Component being wrapped.
name : str, optional (constructor only)
    The name of the resource. Default value is ``"gxf_component"``.
**kwargs : dict
    The additional keyword arguments that can be passed depend on the underlying GXF Component.
    These parameters can provide further customization and functionality to the resource.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the resource.

Returns
-------
str
    The GXF type name of the resource
)doc")

PYDOC(initialize, R"doc(
Initialize the resource.

This method is called only once when the resource is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the resource specification.

Parameters
----------
spec : holoscan.core.ComponentSpec
    The resource specification.
)doc")

}  // namespace GXFComponentResource

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_RESOURCES_GXF_COMPONENT_RESOURCE_PYDOC_HPP */
