/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/resources/gxf/clock.hpp>

#include <string>

#include <gxf/std/clock.hpp>

namespace holoscan {

namespace gxf {

Clock::Clock(const std::string& name, nvidia::gxf::Clock* component)
    : GXFResource(name, component) {}

nvidia::gxf::Clock* Clock::get() const {
  return static_cast<nvidia::gxf::Clock*>(gxf_cptr_);
}

}  // namespace gxf

}  // namespace holoscan
