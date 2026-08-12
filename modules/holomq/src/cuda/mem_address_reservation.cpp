/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda/mem_address_reservation.hpp"

#include <cuda.h>

#include <utility>

namespace holomq {
namespace cuda {

MemAddressReservation::MemAddressReservation(std::size_t size_bytes,
                                             std::size_t granularity) noexcept {
  CUdeviceptr base{};
  if (cuMemAddressReserve(&base, size_bytes, granularity, 0U, 0ULL) != CUDA_SUCCESS) {
    return;
  }
  ptr_ = base;
  size_bytes_ = size_bytes;
}

MemAddressReservation::MemAddressReservation(MemAddressReservation&& other) noexcept
    : ptr_(other.ptr_), size_bytes_(other.size_bytes_) {
  other.ptr_ = CUdeviceptr{};
  other.size_bytes_ = 0;
}

MemAddressReservation& MemAddressReservation::operator=(MemAddressReservation&& other) noexcept {
  if (this != &other) {
    if (static_cast<bool>(*this)) {
      (void)cuMemAddressFree(ptr_, size_bytes_);
    }
    ptr_ = other.ptr_;
    size_bytes_ = other.size_bytes_;
    other.ptr_ = CUdeviceptr{};
    other.size_bytes_ = 0;
  }
  return *this;
}

MemAddressReservation::~MemAddressReservation() noexcept {
  if (!static_cast<bool>(*this)) {
    return;
  }
  (void)cuMemAddressFree(ptr_, size_bytes_);
}

void MemAddressReservation::release() noexcept {
  ptr_ = CUdeviceptr{};
  size_bytes_ = 0;
}

}  // namespace cuda
}  // namespace holomq
