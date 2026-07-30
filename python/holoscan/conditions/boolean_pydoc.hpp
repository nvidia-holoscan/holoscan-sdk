/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_CONDITIONS_BOOLEAN_PYDOC_HPP
#define PYHOLOSCAN_CONDITIONS_BOOLEAN_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace BooleanCondition {

PYDOC(BooleanCondition, R"doc(
Boolean condition.

This condition can be used as a kill switch for an operator. Once the condition is set to false,
the operator will enter the NEVER scheduling status and cannot be executed again. In other words
,this condition cannot currently be used to pause and resume an operator as the operator cannot
be restarted once it is in the NEVER state.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment (or subgraph) the condition will be associated with
enable_tick : bool, optional
    Boolean value for the condition.
name : str, optional
    The name of the condition.
)doc")

PYDOC(enable_tick, R"doc(
Set condition to ``True``.
)doc")

PYDOC(disable_tick, R"doc(
Set condition to ``False``.
)doc")

PYDOC(check_tick_enabled, R"doc(
Check whether the condition is ``True``.
)doc")

}  // namespace BooleanCondition
}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CONDITIONS_BOOLEAN_PYDOC_HPP
