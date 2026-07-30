/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_OPERATORS_TEST_OPS_PYDOC_HPP
#define PYHOLOSCAN_OPERATORS_TEST_OPS_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc {

namespace DataTypeTxTestOp {
// DataTypeTxTestOp Constructor
PYDOC(DataTypeTxTestOp, R"doc(
C++ Data type transmitter operator intended for use in tests.

On each tick, it transmits a fixed value of the specified `data_type` on the output port.

**==Named Outputs==**

    out : <data_type>
        A fixed value corresponding to the chosen `data_type`.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph (constructor only)
    The fragment that the operator belongs to.
data_type : str, optional
    A string representing the data type for the generated tensor. Must be one of
    "int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "float", "double", "complex<float>", or "complex<double>", "bool" , "std::string" or
    "std::unordered_map<std::string, std::string>". Also supports  "std::vector<T>" and
    "std::vector<std::vector<T>>" for the types T above. Additionally supports
    "std::shared_ptr<T>" types for these types.
name : str, optional
    The name of the operator. Default value is ``"data_type_tx_test_op"``.
)doc")

}  // namespace DataTypeTxTestOp

namespace DataTypeRxTestOp {

// DataTypeRxTestOp Constructor
PYDOC(DataTypeRxTestOp, R"doc(
C++ Data type receiver operator intended for use in tests.

On each tick, it receives a std::any and prints the type name.

**==Named Inputs==**

    in : <data_type>
        Receives value as std::any type.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph (constructor only)
    The fragment that the operator belongs to.
)doc")

}  // namespace DataTypeRxTestOp

namespace PoseTreeManagerLookupOp {

// PoseTreeManagerLookupOp Constructor
PYDOC(PoseTreeManagerLookupOp, R"doc(
C++ PoseTreeManager service lookup operator intended for use in tests.

During initialization, it attempts to retrieve the PoseTreeManager service using the
"pose_tree_manager" id and throws if the service is not found.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph (constructor only)
    The fragment that the operator belongs to.
name : str, optional
    The name of the operator. Default value is ``"pose_tree_manager_lookup_op"``.
)doc")

}  // namespace PoseTreeManagerLookupOp

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_OPERATORS_TEST_OPS_PYDOC_HPP */
