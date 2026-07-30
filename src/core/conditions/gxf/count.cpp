/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/conditions/gxf/count.hpp>

#include <holoscan/core/component_spec.hpp>

namespace holoscan {

void CountCondition::setup(ComponentSpec& spec) {
  spec.param(
      count_, "count", "Count", "The total number of times this term will permit execution.", 1L);
}

nvidia::gxf::CountSchedulingTerm* CountCondition::get() const {
  return static_cast<nvidia::gxf::CountSchedulingTerm*>(gxf_cptr_);
}

}  // namespace holoscan
