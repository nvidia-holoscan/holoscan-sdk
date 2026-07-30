/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
fragment : holoscan.core.Fragment or holoscan.core.Subgraph (constructor only)
    The fragment or subgraph that the resource belongs to.
gxf_typename : str
    The GXF type name that identifies the specific GXF Component being wrapped.
name : str, optional (constructor only)
    The name of the resource. Default value is ``"gxf_component"``.
kwargs : dict
    The additional keyword arguments that can be passed depend on the underlying GXF Component.
    These parameters can provide further customization and functionality to the resource.
)doc")

}  // namespace GXFComponentResource

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_RESOURCES_GXF_COMPONENT_RESOURCE_PYDOC_HPP */
