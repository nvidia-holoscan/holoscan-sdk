/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment or subgraph to assign the resource to.
name : str, optional
    The name of the serializer.
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
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment or subgraph to assign the resource to.
allocator : holoscan.resource.Allocator
    The memory allocator for tensor components.
name : str, optional
    The name of the component serializer.
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
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment or subgraph to assign the resource to.
allocator : holoscan.resource.Allocator
    The memory allocator for tensor components.
name : str, optional
    The name of the component serializer.
)doc")

}  // namespace UcxHoloscanComponentSerializer

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_RESOURCES_COMPONENT_SERIALIZERS_PYDOC_HPP
