/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment or subgraph that the operator belongs to.
receive_as_tensormap : bool, optional
    Whether to receive the tensor as a TensorMap. If false, receive<std::shared_ptr<Tensor>> is
    used instead. Default value is ``True``.
name : str, optional
    The name of the operator. Default value is ``"ping_tensor_rx"``.
)doc")

}  // namespace holoscan::doc::PingTensorRxOp

#endif /* PYHOLOSCAN_OPERATORS_PING_TENSOR_RX_PYDOC_HPP */
