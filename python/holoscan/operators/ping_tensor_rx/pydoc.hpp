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

#ifndef PYHOLOSCAN_OPERATORS_PING_TENSOR_RX_PYDOC_HPP
#define PYHOLOSCAN_OPERATORS_PING_TENSOR_RX_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc::PingTensorRxOp {

// PyPingTensorRxOp Constructor
PYDOC(PingTensorRxOp, R"doc(
Example tensor receive operator.

**==Named Inputs==**

    in : nvidia::gxf::TensorMap
        A message containing any number of host or device tensors.

Parameters
----------
fragment : holoscan.core.Fragment (constructor only)
    The fragment that the operator belongs to.
name : str, optional (constructor only)
    The name of the operator. Default value is ``"ping_tensor_rx"``.
)doc")

}  // namespace holoscan::doc::PingTensorRxOp

#endif /* PYHOLOSCAN_OPERATORS_PING_TENSOR_RX_PYDOC_HPP */
