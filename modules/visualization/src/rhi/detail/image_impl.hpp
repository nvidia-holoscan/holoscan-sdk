/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>

#include <cuda.h>

#include <holoviz/rhi/image.hpp>

namespace holoviz::rhi::detail {

/// Backend interface for Image; see the Image handle for API documentation.
class ImageImpl {
 public:
  ImageImpl(const ImageImpl&) = delete;
  ImageImpl& operator=(const ImageImpl&) = delete;
  ImageImpl(ImageImpl&&) = delete;
  ImageImpl& operator=(ImageImpl&&) = delete;
  virtual ~ImageImpl() = default;

  /// Release all owned backend resources now. \p timeout bounds how long to
  /// wait for the object to stop being in use: zero (the default) fails
  /// immediately with ErrorCode::kBackpressured if the object is in use, and a
  /// non-zero timeout waits up to that duration before returning
  /// ErrorCode::kBackpressured. Backends without in-use tracking ignore the
  /// timeout and destroy immediately. Idempotent: repeated calls are no-ops
  /// returning success. Called by the destructor if not already destroyed.
  [[nodiscard]] virtual expected<void> destroy(
      std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero()) = 0;

  [[nodiscard]] virtual expected<CUmipmappedArray> cuda_mipmapped_array() const = 0;
  [[nodiscard]] virtual expected<CUarray> cuda_mipmapped_array_get_level(uint32_t level) const = 0;

  /// Infallible value getter (set at init), returned raw like the swapchain impl's
  /// getters; format() stays expected<> because the impl stores the native format and
  /// converts on read. The public Image handle wraps both in expected<> uniformly.
  [[nodiscard]] virtual Extent2D extent() const = 0;
  [[nodiscard]] virtual expected<Format> format() const = 0;
  [[nodiscard]] virtual expected<NativeObject> native_object(NativeObjectType type) const = 0;

 protected:
  ImageImpl() = default;
};

}  // namespace holoviz::rhi::detail
