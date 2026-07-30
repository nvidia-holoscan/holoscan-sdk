/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_GXF_OPERATOR_PYDOC_HPP
#define PYHOLOSCAN_GXF_OPERATOR_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace GXFOperator {

// Constructor
PYDOC(GXFOperator, R"doc(
Base GXF-based operator class.
)doc")

PYDOC(GXFOperator_kwargs, R"doc(
Base GXF-based operator class.

Parameters
----------
**kwargs : dict
    Keyword arguments to pass on to the parent operator class.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the operator.

Returns
-------
str
    The GXF type name of the operator.
)doc")

PYDOC(gxf_context, R"doc(
The GXF context of the component.
)doc")

PYDOC(gxf_eid, R"doc(
The GXF entity ID.
)doc")

PYDOC(gxf_cid, R"doc(
The GXF component ID.
)doc")

PYDOC(gxf_entity_group_name, R"doc(
The name of the GXF EntityGroup containing this operator.

Returns
-------
str
    The entity group name.
)doc")

PYDOC(description, R"doc(
YAML formatted string describing the operator.
)doc")

}  // namespace GXFOperator

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_GXF_OPERATOR_PYDOC_HPP
