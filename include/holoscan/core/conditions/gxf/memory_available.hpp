/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_CONDITIONS_GXF_MEMORY_AVAILABLE_HPP
#define HOLOSCAN_CORE_CONDITIONS_GXF_MEMORY_AVAILABLE_HPP

#include <cinttypes>
#include <memory>
#include <utility>

#include <gxf/std/scheduling_terms.hpp>

#include "../../gxf/gxf_condition.hpp"
#include "../../resources/gxf/allocator.hpp"
#include "../../resources/gxf/receiver.hpp"

namespace holoscan {

/**
 * @brief Condition that permits execution only when a specified allocator has sufficient memory
 * available.
 *
 * The memory is typically provided via the `min_bytes` parameter, but for allocators that use
 * memory blocks it is possible to specify the memory via `min_blocks` instead if desired.
 *
 * ==Parameters==
 *
 * - **allocator** (holoscan::Allocator): The allocator whose memory availability should be checked.
 * - **min_bytes** (uint64_t, optional): The minimum number of bytes that must be available in order
 * for the associated operator to execute. Exclusive with min_blocks (only one can be set).
 * - **min_blocks** (uint64_t, optional): The minimum number of blocks that must be available in
 * order for the associated operator to execute. Can only be used with allocators such as
 * `BlockMemoryPool` that use memory blocks. Exclusive with min_bytes (only one can be set).
 */
class MemoryAvailableCondition : public gxf::GXFCondition {
 public:
  HOLOSCAN_CONDITION_FORWARD_ARGS_SUPER(MemoryAvailableCondition, GXFCondition)
  MemoryAvailableCondition() = default;

  const char* gxf_typename() const override { return "nvidia::gxf::MemoryAvailableSchedulingTerm"; }

  void allocator(std::shared_ptr<Allocator> allocator) { allocator_ = std::move(allocator); }
  std::shared_ptr<Allocator> allocator() { return allocator_.get(); }

  void setup(ComponentSpec& spec) override;

  void initialize() override;

  nvidia::gxf::MemoryAvailableSchedulingTerm* get() const;

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<uint64_t> min_bytes_;
  Parameter<uint64_t> min_blocks_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_CONDITIONS_GXF_MEMORY_AVAILABLE_HPP */
