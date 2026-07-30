/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CORE_CONDITIONS_GXF_DOWNSTREAM_AFFORDABLE_CPP
#define CORE_CONDITIONS_GXF_DOWNSTREAM_AFFORDABLE_CPP

#include <holoscan/core/conditions/gxf/downstream_affordable.hpp>

#include <holoscan/core/component_spec.hpp>

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

nvidia::gxf::DownstreamReceptiveSchedulingTerm* DownstreamMessageAffordableCondition::get() const {
  return static_cast<nvidia::gxf::DownstreamReceptiveSchedulingTerm*>(gxf_cptr_);
}

void DownstreamMessageAffordableCondition::min_size(uint64_t min_size) {
  auto cond = get();
  if (cond) {
    auto maybe_set = cond->setMinSize(min_size);
    if (!maybe_set) {
      throw std::runtime_error(
          fmt::format("Failed to set min_size: {}", GxfResultStr(maybe_set.error())));
    }
  }
  min_size_ = min_size;
  return;
}

}  // namespace holoscan

#endif /* CORE_CONDITIONS_GXF_DOWNSTREAM_AFFORDABLE_CPP */
