/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_GXF_PYDOC_HPP
#define PYHOLOSCAN_GXF_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace GXFComponent {

// Constructor
PYDOC(GXFComponent, R"doc(
Base GXF-based component class.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the component.

Returns
-------
str
    The GXF type name of the component.
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

PYDOC(gxf_cname, R"doc(
The name of the component.
)doc")

PYDOC(gxf_initialize, R"doc(
Initialize the component.
)doc")

}  // namespace GXFComponent

namespace GXFCondition {

// Constructor
PYDOC(GXFCondition, R"doc(
Base GXF-based condition class.
)doc")

PYDOC(GXFCondition_kwargs, R"doc(
Base GXF-based condition class.

Parameters
----------
**kwargs : dict
    Keyword arguments to pass on to the parent condition class.
)doc")

}  // namespace GXFCondition

namespace GXFResource {

// Constructor
PYDOC(GXFResource, R"doc(
Base GXF-based resource class.
)doc")

PYDOC(GXFResource_kwargs, R"doc(
Base GXF-based resource class.

Parameters
----------
**kwargs : dict
    Keyword arguments to pass on to the parent resource class.
)doc")

}  // namespace GXFResource

namespace GXFSystemResourceBase {

// Constructor
PYDOC(GXFSystemResourceBase, R"doc(
A class considered a GXF nvidia::gxf::Resource.

This represents a resource such as a ThreadPool or GPUDevice that may be shared amongst multiple
entities in an nvidia::gxf::EntityGroup.

)doc")

PYDOC(GXFSystemResourceBase_kwargs, R"doc(
A class considered a GXF nvidia::gxf::Resource.

This represents a resource such as a ThreadPool or GPUDevice that may be shared amongst multiple
entities in an nvidia::gxf::EntityGroup.

Parameters
----------
**kwargs : dict
    Keyword arguments to pass on to the parent resource class.
)doc")

}  // namespace GXFSystemResourceBase

namespace GXFInputContext {

// Constructor
PYDOC(GXFInputContext, R"doc(
GXF input context.

Parameters
----------
op : holoscan.gxf.GXFOperator
    The GXF operator that owns this context.
)doc")

// PYDOC(receive_impl, R"doc(
//
// )doc")

}  // namespace GXFInputContext

namespace GXFOutputContext {

// Constructor
PYDOC(GXFOutputContext, R"doc(
GXF input context.

Parameters
----------
op : holoscan.gxf.GXFOperator
    The GXF operator that owns this context.
)doc")

// PYDOC(emit_impl, R"doc(
//
// )doc")

}  // namespace GXFOutputContext

namespace GXFExecutionContext {

// Constructor
PYDOC(GXFExecutionContext, R"doc(
Execution context for an operator using GXF.

Parameters
----------
op : holoscan.gxf.GXFOperator
    The GXF operator that owns this context.
)doc")

}  // namespace GXFExecutionContext

namespace GXFWrapper {

// Constructor
PYDOC(GXFWrapper, R"doc(
Class to wrap an Operator into a GXF Codelet
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(deinitialize, R"doc(
De-initialize the operator.
)doc")

PYDOC(start, R"doc(
Start method.
)doc")

PYDOC(tick, R"doc(
Tick method.
)doc")

PYDOC(stop, R"doc(
Stop method.
)doc")

PYDOC(set_operator, R"doc(
Set the Operator object to be wrapped.

Parameters
----------
op : holoscan.gxf.GXFOperator
    The GXF operator to wrap.
)doc")

PYDOC(registerInterface, R"doc(
GXF registrar to use.
)doc")

}  // namespace GXFWrapper

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_GXF_PYDOC_HPP
