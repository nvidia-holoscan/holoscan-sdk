/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda/mem.hpp"

#include <cuda.h>

#include <cstdint>
#include <utility>

namespace holomq {
namespace cuda {

Mem::Mem(std::size_t size_bytes, const CUmemAllocationProp* prop,
         unsigned long long flags) noexcept {
  if (prop == nullptr) {
    return;
  }
  CUmemGenericAllocationHandle h{};
  if (cuMemCreate(&h, size_bytes, prop, flags) != CUDA_SUCCESS) {
    return;
  }
  handle_ = h;
}

Mem::Mem(ScopedFileDescriptor import_fd) noexcept {
  CUmemGenericAllocationHandle h{};
  // POSIX FD: osHandle is the fd value as void*, not &fd (see NVIDIA memMapIpc).
  void* const os_handle = reinterpret_cast<void*>(static_cast<std::uintptr_t>(import_fd.get()));
  if (cuMemImportFromShareableHandle(&h, os_handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) !=
      CUDA_SUCCESS) {
    return;
  }
  handle_ = h;
}

Mem::Mem(void* device_ptr) noexcept {
  if (device_ptr == nullptr) {
    return;
  }
  CUmemGenericAllocationHandle h{};
  if (cuMemRetainAllocationHandle(&h, device_ptr) != CUDA_SUCCESS) {
    return;
  }
  handle_ = h;
}

Mem::Mem(Mem&& other) noexcept : handle_(other.handle_) {
  other.handle_ = CUmemGenericAllocationHandle{};
}

Mem& Mem::operator=(Mem&& other) noexcept {
  if (this != &other) {
    if (static_cast<bool>(*this)) {
      (void)cuMemRelease(handle_);
    }
    handle_ = other.handle_;
    other.handle_ = CUmemGenericAllocationHandle{};
  }
  return *this;
}

Mem::~Mem() noexcept {
  if (!static_cast<bool>(*this)) {
    return;
  }
  (void)cuMemRelease(handle_);
}

void Mem::release() noexcept {
  handle_ = CUmemGenericAllocationHandle{};
}

}  // namespace cuda
}  // namespace holomq
