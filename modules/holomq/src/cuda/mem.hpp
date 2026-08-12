/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "scoped_file_descriptor.hpp"

#include <cuda.h>

#include <cstddef>

namespace holomq {
namespace cuda {

/// Owns a generic allocation from \c cuMemCreate, \c cuMemImportFromShareableHandle, or \c
/// cuMemRetainAllocationHandle; calls \c cuMemRelease on destruction unless \ref release
/// transferred ownership (for example into a device \ref BufferPool). If creation, import, or
/// retain fails, the object is invalid (\c operator bool() is \c false) and the destructor is a
/// no-op.
///
/// The caller must keep the appropriate \c CUcontext current for \c cuMemCreate / \c
/// cuMemImportFromShareableHandle /
/// \c cuMemRetainAllocationHandle / \c cuMemRelease whenever this object may run those APIs
/// (destructor or move-assignment replacing an owned handle).
class Mem {
 public:
  Mem() noexcept = default;

  /// Calls \c cuMemCreate(\p size_bytes, \p prop, \p flags). \p prop must be non-null. On failure
  /// the object remains invalid.
  Mem(std::size_t size_bytes, const CUmemAllocationProp* prop,
      unsigned long long flags = 0ULL) noexcept;

  /// Imports \p import_fd via \c cuMemImportFromShareableHandle (POSIX file descriptor). \p
  /// import_fd is closed when the constructor returns.
  explicit Mem(ScopedFileDescriptor import_fd) noexcept;

  /// Retains the allocation backing \p device_ptr via \c cuMemRetainAllocationHandle. On failure
  /// the object remains invalid.
  explicit Mem(void* device_ptr) noexcept;

  Mem(const Mem&) = delete;
  Mem& operator=(const Mem&) = delete;

  Mem(Mem&& other) noexcept;
  Mem& operator=(Mem&& other) noexcept;

  ~Mem() noexcept;

  [[nodiscard]] explicit operator bool() const noexcept {
    return handle_ != CUmemGenericAllocationHandle{};
  }

  [[nodiscard]] CUmemGenericAllocationHandle get() const noexcept { return handle_; }

  /// Stops owning the handle; \c cuMemRelease will not run in the destructor (caller owns the
  /// allocation object).
  void release() noexcept;

 private:
  CUmemGenericAllocationHandle handle_{};
};

}  // namespace cuda
}  // namespace holomq
