/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_ALLOCATOR_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_ALLOCATOR_HPP

#include <string>

#include <gxf/std/allocator.hpp>

#include "../../gxf/gxf_resource.hpp"

namespace holoscan {

/**
 * @brief Base class for all allocators.
 *
 * Allocators are used to allocate resources such as memory or CUDA threads.
 */
class Allocator : public gxf::GXFResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(Allocator, GXFResource)
  Allocator() = default;
  Allocator(const std::string& name, nvidia::gxf::Allocator* component);

  const char* gxf_typename() const override { return "nvidia::gxf::Allocator"; }

  virtual bool is_available(uint64_t size);

  // TODO(gbae): Introduce expected<> type
  virtual nvidia::byte* allocate(uint64_t size, MemoryStorageType type);

  virtual void free(nvidia::byte* pointer);

  // Get the block size of this allocator, defaults to 1 for byte-based allocators
  uint64_t block_size();

  nvidia::gxf::Allocator* get() const;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_ALLOCATOR_HPP */
