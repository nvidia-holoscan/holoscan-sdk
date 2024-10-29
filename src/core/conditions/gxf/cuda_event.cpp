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

#include "holoscan/core/conditions/gxf/cuda_event.hpp"

#include <string>

#include "holoscan/core/component_spec.hpp"

namespace holoscan {

CudaEventCondition::CudaEventCondition(const std::string& name,
                                       nvidia::gxf::CudaEventSchedulingTerm* term)
    : GXFCondition(name, term) {}

nvidia::gxf::CudaEventSchedulingTerm* CudaEventCondition::get() const {
  return static_cast<nvidia::gxf::CudaEventSchedulingTerm*>(gxf_cptr_);
}

void CudaEventCondition::setup(ComponentSpec& spec) {
  spec.param(receiver_,
             "receiver",
             "Queue channel",
             "The receiver on which data will be available oncethe stream completes.");
  spec.param(event_name_,
             "event_name",
             "Event name",
             "The event name on which the cudaEventQuery API is called to get the status",
             std::string(""));
}

}  // namespace holoscan
