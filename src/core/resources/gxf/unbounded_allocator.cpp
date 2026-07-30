/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/resources/gxf/unbounded_allocator.hpp>

#include <string>

namespace holoscan {

UnboundedAllocator::UnboundedAllocator(const std::string& name,
                                       nvidia::gxf::UnboundedAllocator* component)
    : Allocator(name, component) {}

nvidia::gxf::UnboundedAllocator* UnboundedAllocator::get() const {
  return static_cast<nvidia::gxf::UnboundedAllocator*>(gxf_cptr_);
}

}  // namespace holoscan
