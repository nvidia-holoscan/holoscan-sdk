/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_BAYER_DEMOSAIC_PYDOC_HPP
#define HOLOSCAN_OPERATORS_BAYER_DEMOSAIC_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc::BayerDemosaicOp {

// Constructor
PYDOC(BayerDemosaicOp, R"doc(
Bayer Demosaic operator.
)doc")

// PyBayerDemosaicOp Constructor
PYDOC(BayerDemosaicOp_python, R"doc(
Bayer Demosaic operator.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the operator belongs to.
pool : holoscan.resources.Allocator
    Memory pool allocator used by the operator.
cuda_stream_pool : holoscan.resources.CudaStreamPool
    CUDA Stream pool to create CUDA streams
in_tensor_name : str, optional
    The name of the input tensor.
out_tensor_name : str, optional
    The name of the output tensor.
interpolation_mode : int, optional
    The interpolation model to be used for demosaicing. Values available at:
    https://docs.nvidia.com/cuda/npp/group__typedefs__npp.html#ga2b58ebd329141d560aa4367f1708f191
bayer_grid_pos : int, optional
    The Bayer grid position (default of 2 = GBRG). Values available at:
    https://docs.nvidia.com/cuda/npp/group__typedefs__npp.html#ga5597309d6766fb2dffe155990d915ecb
generate_alpha : bool, optional
    Generate alpha channel.
alpha_value : int, optional
    Alpha value to be generated if `generate_alpha` is set to ``True``.
name : str, optional
    The name of the operator.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the resource.

Returns
-------
str
    The GXF type name of the resource
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc")

}  // namespace holoscan::doc::BayerDemosaicOp

#endif /* HOLOSCAN_OPERATORS_BAYER_DEMOSAIC_PYDOC_HPP */
