/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_CUDA_ALLOCATOR_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_CUDA_ALLOCATOR_HPP

#include <cstdint>
#include <string>

#include <gxf/cuda/cuda_allocator.hpp>

#include "../../gxf/gxf_resource.hpp"
#include "./allocator.hpp"

namespace holoscan {

/**
 * @brief Base class for all CUDA allocators.
 *
 * CudaAllocators are allocators for CUDA memory that also support asynchronous allocation.
 */
class CudaAllocator : public Allocator {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(CudaAllocator, Allocator)

  CudaAllocator() = default;
  CudaAllocator(const std::string& name, nvidia::gxf::CudaAllocator* component);

  const char* gxf_typename() const override { return "nvidia::gxf::CudaAllocator"; }

  // the following async functions and get_pool_size are specific to CudaAllocator
  nvidia::byte* allocate_async(uint64_t size, cudaStream_t stream);
  void free_async(byte* pointer, cudaStream_t stream);
  size_t pool_size(MemoryStorageType type) const;

  nvidia::gxf::CudaAllocator* get() const;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_CUDA_ALLOCATOR_HPP */
