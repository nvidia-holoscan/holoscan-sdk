/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_RESOURCES_COMPONENT_SERIALIZERS_PYDOC_HPP
#define PYHOLOSCAN_RESOURCES_COMPONENT_SERIALIZERS_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace SerializationBuffer {

PYDOC(SerializationBuffer, R"doc(
Serialization Buffer.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment or subgraph to assign the resource to.
allocator : holoscan.resource.Allocator
    The memory allocator for tensor components.
buffer_size : int, optional
    The size of the buffer in bytes.
name : str, optional
    The name of the serialization buffer
)doc")

}  // namespace SerializationBuffer

namespace UcxSerializationBuffer {

PYDOC(UcxSerializationBuffer, R"doc(
UCX serialization buffer.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment or subgraph to assign the resource to.
allocator : holoscan.resource.Allocator
    The memory allocator for tensor components.
buffer_size : int, optional
    The size of the buffer in bytes.
name : str, optional
    The name of the serialization buffer
)doc")

}  // namespace UcxSerializationBuffer

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_RESOURCES_COMPONENT_SERIALIZERS_PYDOC_HPP
