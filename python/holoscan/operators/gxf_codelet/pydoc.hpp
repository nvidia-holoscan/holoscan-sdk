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

#ifndef PYHOLOSCAN_OPERATORS_GXF_CODELET_PYDOC_HPP
#define PYHOLOSCAN_OPERATORS_GXF_CODELET_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc::GXFCodeletOp {

// PyGXFCodeletOp Constructor
PYDOC(GXFCodeletOp, R"doc(
GXF Codelet wrapper operator.

**==Named Inputs==**

    Input ports are automatically defined based on the parameters of the underlying GXF Codelet
    that include the ``nvidia::gxf::Receiver`` component handle.

    To view the information about the operator, refer to the `description` property of this object.

**==Named Outputs==**

    Output ports are automatically defined based on the parameters of the underlying GXF Codelet
    that include the ``nvidia::gxf::Transmitter`` component handle.

    To view the information about the operator, refer to the ``description`` property of this object.

Parameters
----------
fragment : holoscan.core.Fragment (constructor only)
    The fragment that the operator belongs to.
gxf_typename : str
    The GXF type name that identifies the specific GXF Codelet being wrapped.
*args : tuple
    Additional positional arguments (``holoscan.core.Condition`` or ``holoscan.core.Resource``).
name : str, optional (constructor only)
    The name of the operator. Default value is ``"gxf_codelet"``.
**kwargs : dict
   The additional keyword arguments that can be passed depend on the underlying GXF Codelet. The
   additional parameters are the parameters of the underlying GXF Codelet that are neither
   specifically part of the ``nvidia::gxf::Receiver`` nor the    ``nvidia::gxf::Transmitter``
   components. These parameters can provide further customization and functionality to the operator.
)doc")
}  // namespace holoscan::doc::GXFCodeletOp

#endif /* PYHOLOSCAN_OPERATORS_GXF_CODELET_PYDOC_HPP */
