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

PYDOC(initialize, R"doc(
Initialize the operator.
)doc")

}  // namespace GXFOperator

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_GXF_OPERATOR_PYDOC_HPP
