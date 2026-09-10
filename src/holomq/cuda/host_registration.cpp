/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda/host_registration.hpp"

#include <cuda.h>

#include <utility>

namespace holomq {
namespace cuda {

RegisteredHostMemory::RegisteredHostMemory(void* const base, const std::size_t size_bytes,
                                           const unsigned int flags) noexcept
    : base_(base) {
  if (base_ == nullptr || size_bytes == 0U) {
    base_ = nullptr;
    return;
  }
  register_result_ = cuMemHostRegister(base_, size_bytes, flags);
  registered_ = register_result_ == CUDA_SUCCESS;
  if (!registered_) {
    base_ = nullptr;
  }
}

RegisteredHostMemory::RegisteredHostMemory(RegisteredHostMemory&& other) noexcept
    : base_(other.base_), register_result_(other.register_result_), registered_(other.registered_) {
  other.base_ = nullptr;
  other.register_result_ = CUDA_SUCCESS;
  other.registered_ = false;
}

RegisteredHostMemory& RegisteredHostMemory::operator=(RegisteredHostMemory&& other) noexcept {
  if (this != &other) {
    unregister_if_registered();
    base_ = other.base_;
    register_result_ = other.register_result_;
    registered_ = other.registered_;
    other.base_ = nullptr;
    other.register_result_ = CUDA_SUCCESS;
    other.registered_ = false;
  }
  return *this;
}

RegisteredHostMemory::~RegisteredHostMemory() noexcept {
  unregister_if_registered();
}

void RegisteredHostMemory::unregister_if_registered() noexcept {
  if (!registered_ || base_ == nullptr) {
    return;
  }
  static_cast<void>(cuMemHostUnregister(base_));
  registered_ = false;
  base_ = nullptr;
}

}  // namespace cuda
}  // namespace holomq
