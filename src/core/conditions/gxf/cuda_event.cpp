/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/conditions/gxf/cuda_event.hpp>

#include <string>

#include <holoscan/core/component_spec.hpp>

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
