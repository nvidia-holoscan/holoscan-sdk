/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_RESOURCES_COMPONENT_SERIALIZERS_PYDOC_HPP
#define PYHOLOSCAN_RESOURCES_COMPONENT_SERIALIZERS_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace UcxEntitySerializer {

PYDOC(UcxEntitySerializer, R"doc(
UCX entity serializer.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment or subgraph to assign the resource to.
component_serializer : list of holoscan.resource.Resource
    The component serializers used by the entity serializer.
verbose_warning : bool, optional
    Whether to use verbose warnings during serialization.
name : str, optional
    The name of the entity serializer.
)doc")

}  // namespace UcxEntitySerializer

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_RESOURCES_COMPONENT_SERIALIZERS_PYDOC_HPP
