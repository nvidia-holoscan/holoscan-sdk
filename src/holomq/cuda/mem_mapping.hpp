/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>

#include <cstddef>

namespace holomq {
namespace cuda {

/// Maps \p alloc_handle into \p ptr / \p size_bytes with \c cuMemMap; calls \c cuMemUnmap on
/// destruction unless
/// \ref release transferred ownership (for example into a device \ref BufferPool). If mapping
/// fails, the object is invalid
/// (\c operator bool() is \c false) and the destructor is a no-op.
///
/// The caller must keep the appropriate \c CUcontext current for \c cuMemMap / \c cuMemUnmap
/// whenever this object may run those APIs (destructor or move-assignment replacing an owned
/// mapping).
class MemMapping {
 public:
  MemMapping() noexcept = default;

  /// Calls \c cuMemMap(\p ptr, \p size_bytes, 0, \p alloc_handle, 0). On failure the object remains
  /// invalid.
  MemMapping(CUdeviceptr ptr, std::size_t size_bytes,
             CUmemGenericAllocationHandle alloc_handle) noexcept;

  MemMapping(const MemMapping&) = delete;
  MemMapping& operator=(const MemMapping&) = delete;

  MemMapping(MemMapping&& other) noexcept;
  MemMapping& operator=(MemMapping&& other) noexcept;

  ~MemMapping() noexcept;

  [[nodiscard]] explicit operator bool() const noexcept {
    return ptr_ != CUdeviceptr{} && size_bytes_ != 0U;  // NOLINT(whitespace/braces)
  }

  [[nodiscard]] CUdeviceptr ptr() const noexcept { return ptr_; }

  [[nodiscard]] std::size_t size_bytes() const noexcept { return size_bytes_; }

  /// Stops owning the mapping; \c cuMemUnmap will not run in the destructor (caller owns the mapped
  /// range).
  void release() noexcept;

 private:
  CUdeviceptr ptr_{};
  std::size_t size_bytes_{};
};

}  // namespace cuda
}  // namespace holomq
