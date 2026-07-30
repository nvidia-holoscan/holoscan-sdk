/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_RESOURCES_CONDITION_COMBINERS_PYDOC_HPP
#define PYHOLOSCAN_RESOURCES_CONDITION_COMBINERS_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace ConditionCombiners {

PYDOC(OrConditionCombiner, R"doc(
OR Condition Combiner

Will configure the associated conditions to be OR combined instead of
the default AND combination behavior.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment or subgraph to assign the resource to.
terms : list of holoscan.core.Condition
    The conditions to be OR combined.
name : str, optional
    The name of the serializer.
)doc")

}  // namespace ConditionCombiners

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_RESOURCES_CONDITION_COMBINERS_PYDOC_HPP
