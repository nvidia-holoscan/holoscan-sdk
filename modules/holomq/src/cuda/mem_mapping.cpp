/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda/mem_mapping.hpp"

#include <cuda.h>

#include <utility>

namespace holomq {
namespace cuda {

MemMapping::MemMapping(CUdeviceptr ptr, std::size_t size_bytes,
                       CUmemGenericAllocationHandle alloc_handle) noexcept {
  if (cuMemMap(ptr, size_bytes, 0U, alloc_handle, 0ULL) != CUDA_SUCCESS) {
    return;
  }
  ptr_ = ptr;
  size_bytes_ = size_bytes;
}

MemMapping::MemMapping(MemMapping&& other) noexcept
    : ptr_(other.ptr_), size_bytes_(other.size_bytes_) {
  other.ptr_ = CUdeviceptr{};
  other.size_bytes_ = 0;
}

MemMapping& MemMapping::operator=(MemMapping&& other) noexcept {
  if (this != &other) {
    if (static_cast<bool>(*this)) {
      (void)cuMemUnmap(ptr_, size_bytes_);
    }
    ptr_ = other.ptr_;
    size_bytes_ = other.size_bytes_;
    other.ptr_ = CUdeviceptr{};
    other.size_bytes_ = 0;
  }
  return *this;
}

MemMapping::~MemMapping() noexcept {
  if (!static_cast<bool>(*this)) {
    return;
  }
  (void)cuMemUnmap(ptr_, size_bytes_);
}

void MemMapping::release() noexcept {
  ptr_ = CUdeviceptr{};
  size_bytes_ = 0;
}

}  // namespace cuda
}  // namespace holomq
