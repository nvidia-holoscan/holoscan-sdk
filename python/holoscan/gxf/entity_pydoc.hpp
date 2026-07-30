/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_GXF_ENTITY_PYDOC_HPP
#define PYHOLOSCAN_GXF_ENTITY_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace Entity {

// Constructor
PYDOC(Entity, R"doc(
Base class representing a GXF entity.
)doc")

PYDOC(get, R"doc(
Get a resource by name.

Parameters
----------
name : str
    Name of the resource to get.
log_errors : bool
    Whether to log errors when the resource is not found.
    Default is True.

Returns
-------
resource : Tensor
    The resource with the given name.
)doc")

}  // namespace Entity

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_GXF_ENTITY_PYDOC_HPP
