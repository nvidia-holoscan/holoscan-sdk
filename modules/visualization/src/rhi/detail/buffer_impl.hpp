/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>

#include <cuda.h>

#include <holoviz/rhi/buffer.hpp>

namespace holoviz::rhi::detail {

/// Backend interface for Buffer; see the Buffer handle for API documentation.
class BufferImpl {
 public:
  BufferImpl(const BufferImpl&) = delete;
  BufferImpl& operator=(const BufferImpl&) = delete;
  BufferImpl(BufferImpl&&) = delete;
  BufferImpl& operator=(BufferImpl&&) = delete;
  virtual ~BufferImpl() = default;

  /// Release all owned backend resources now. \p timeout bounds how long to
  /// wait for the object to stop being in use: zero (the default) fails
  /// immediately with ErrorCode::kBackpressured if the object is in use, and a
  /// non-zero timeout waits up to that duration before returning
  /// ErrorCode::kBackpressured. Backends without in-use tracking ignore the
  /// timeout and destroy immediately. Idempotent: repeated calls are no-ops
  /// returning success. Called by the destructor if not already destroyed.
  [[nodiscard]] virtual expected<void> destroy(
      std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero()) = 0;

  [[nodiscard]] virtual BufferSlice whole() const = 0;
  [[nodiscard]] virtual expected<BufferSlice> slice(size_t offset_bytes,
                                                    size_t size_bytes) const = 0;
  [[nodiscard]] virtual expected<CUdeviceptr> cuda_device_ptr() const = 0;
  [[nodiscard]] virtual expected<NativeObject> native_object(NativeObjectType type) const = 0;

 protected:
  BufferImpl() = default;
};

}  // namespace holoviz::rhi::detail
