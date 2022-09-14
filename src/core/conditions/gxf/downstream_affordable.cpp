/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CORE_CONDITIONS_GXF_DOWNSTREAM_AFFORDABLE_CPP
#define CORE_CONDITIONS_GXF_DOWNSTREAM_AFFORDABLE_CPP

#include "holoscan/core/conditions/gxf/downstream_affordable.hpp"

#include "holoscan/core/component_spec.hpp"

namespace holoscan {

void DownstreamMessageAffordableCondition::setup(ComponentSpec& spec) {
  spec.param(
      transmitter_,
      "transmitter",
      "Transmitter",
      "The term permits execution if this transmitter can publish a message, i.e. if the receiver "
      "which is connected to this transmitter can receive messages.");
  spec.param(min_size_,
             "min_size",
             "Minimum size",
             "The term permits execution if the receiver connected to the transmitter has at least "
             "the specified number of free slots in its back buffer.",
             1UL);
}

}  // namespace holoscan

#endif /* CORE_CONDITIONS_GXF_DOWNSTREAM_AFFORDABLE_CPP */
