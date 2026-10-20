/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>

#include <cstddef>

namespace holomq {
namespace cuda {

/// Reserves a virtual address range with \c cuMemAddressReserve in the constructor; calls \c
/// cuMemAddressFree on destruction unless \ref release transferred ownership to a longer-lived
/// object (for example a device \ref BufferPool). If reservation fails, the object is invalid (\c
/// operator bool() is \c false) and the destructor is a no-op.
///
/// The caller must keep the same \c CUcontext current that applies to \c cuMemAddressReserve
/// whenever this object may run \c cuMemAddressFree (destructor or move-assignment replacing an
/// owned range).
class MemAddressReservation {
 public:
  MemAddressReservation() noexcept = default;

  /// Calls \c cuMemAddressReserve for \p size_bytes with alignment \p granularity (CUDA recommended
  /// granularity). On failure the object remains invalid; on success \c operator bool() is \c true.
  MemAddressReservation(std::size_t size_bytes, std::size_t granularity) noexcept;

  MemAddressReservation(const MemAddressReservation&) = delete;
  MemAddressReservation& operator=(const MemAddressReservation&) = delete;

  MemAddressReservation(MemAddressReservation&& other) noexcept;
  MemAddressReservation& operator=(MemAddressReservation&& other) noexcept;

  ~MemAddressReservation() noexcept;

  [[nodiscard]] explicit operator bool() const noexcept {
    return ptr_ != CUdeviceptr{} && size_bytes_ != 0U;  // NOLINT(whitespace/braces)
  }

  [[nodiscard]] CUdeviceptr get() const noexcept { return ptr_; }

  [[nodiscard]] std::size_t size_bytes() const noexcept { return size_bytes_; }

  /// Stops owning the range; \c cuMemAddressFree will not run in the destructor (caller owns the VA
  /// range).
  void release() noexcept;

 private:
  CUdeviceptr ptr_{};
  std::size_t size_bytes_{};
};

}  // namespace cuda
}  // namespace holomq
