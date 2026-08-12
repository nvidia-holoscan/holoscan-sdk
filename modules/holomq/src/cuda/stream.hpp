/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>

struct CUstream_st;

namespace holomq {
namespace cuda {

/// Owns a CUDA stream from \c cuStreamCreate. The destructor synchronizes the stream (so enqueued
/// host functions, including holomq deferred ref releases, complete) before \c cuStreamDestroy.
class Stream {
 public:
  Stream() noexcept = default;

  /// Calls \c cuStreamCreate with \p flags. On failure the object remains invalid.
  explicit Stream(const unsigned int flags) noexcept;

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  Stream(Stream&& other) noexcept;
  Stream& operator=(Stream&& other) noexcept;

  ~Stream() noexcept;

  [[nodiscard]] explicit operator bool() const noexcept { return stream_ != nullptr; }

  [[nodiscard]] CUstream_st* get() const noexcept {
    return reinterpret_cast<CUstream_st*>(stream_);
  }

 private:
  void drain_and_destroy() noexcept;

  CUstream stream_{};
};

}  // namespace cuda
}  // namespace holomq
