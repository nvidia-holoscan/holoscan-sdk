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
