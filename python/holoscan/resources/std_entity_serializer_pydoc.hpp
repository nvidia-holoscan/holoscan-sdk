/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_RESOURCES_COMPONENT_SERIALIZERS_PYDOC_HPP
#define PYHOLOSCAN_RESOURCES_COMPONENT_SERIALIZERS_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace StdEntitySerializer {

PYDOC(StdEntitySerializer, R"doc(
Default serializer for GXF entities.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment or subgraph to assign the resource to.
name : str, optional
    The name of the serializer.
)doc")

}  // namespace StdEntitySerializer

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_RESOURCES_COMPONENT_SERIALIZERS_PYDOC_HPP
