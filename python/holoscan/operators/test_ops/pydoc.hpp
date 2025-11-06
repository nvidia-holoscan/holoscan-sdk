/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_OPERATORS_TEST_OPS_PYDOC_HPP */
