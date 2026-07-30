/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_UNBOUNDED_ALLOCATOR_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_UNBOUNDED_ALLOCATOR_HPP

#include <string>

#include <gxf/std/unbounded_allocator.hpp>

#include "./allocator.hpp"

namespace holoscan {

/**
 * @brief Unbounded memory allocator.
 *
 * An allocator that uses dynamic host or device memory allocation without an upper bound.
 *
 * ==Parameters==
 *
 * None
 */
class UnboundedAllocator : public Allocator {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(UnboundedAllocator, Allocator)
  UnboundedAllocator() = default;
  UnboundedAllocator(const std::string& name, nvidia::gxf::UnboundedAllocator* component);

  const char* gxf_typename() const override { return "nvidia::gxf::UnboundedAllocator"; }

  nvidia::gxf::UnboundedAllocator* get() const;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_UNBOUNDED_ALLOCATOR_HPP */
