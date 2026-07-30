/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/conditions/gxf/cuda_buffer_available.hpp>

#include <string>

#include <holoscan/core/component_spec.hpp>

namespace holoscan {

CudaBufferAvailableCondition::CudaBufferAvailableCondition(
    const std::string& name, nvidia::gxf::CudaBufferAvailableSchedulingTerm* term)
    : GXFCondition(name, term) {}

nvidia::gxf::CudaBufferAvailableSchedulingTerm* CudaBufferAvailableCondition::get() const {
  return static_cast<nvidia::gxf::CudaBufferAvailableSchedulingTerm*>(gxf_cptr_);
}

void CudaBufferAvailableCondition::setup(ComponentSpec& spec) {
  spec.param(receiver_,
             "receiver",
             "Queue channel",
             "The receiver on which data will be available oncethe stream completes.");
}

}  // namespace holoscan
