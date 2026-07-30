/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_CONDITIONS_COUNT_PYDOC_HPP
#define PYHOLOSCAN_CONDITIONS_COUNT_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace CountCondition {

PYDOC(CountCondition, R"doc(
Count condition.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment (or subgraph) the condition will be associated with
count : int
    The execution count value used by the condition.
name : str, optional
    The name of the condition.
)doc")

PYDOC(count, R"doc(
The execution count associated with the condition
)doc")

}  // namespace CountCondition

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CONDITIONS_COUNT_PYDOC_HPP
