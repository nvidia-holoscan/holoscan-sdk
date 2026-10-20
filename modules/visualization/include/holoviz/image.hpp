/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Image, sampler state, and the home-layout model. The API exposes zero
// image-layout vocabulary: every image lives "pinned" in one home state
// derived from its usage (sampled -> shader-read, render target -> color
// attachment). Operations that need another state (uploads, copies)
// transition internally and restore the home state before returning, so
// external observers — including future CUDA importers at fence boundaries —
// always find the image in a known state.

#pragma once

#include <chrono>
#include <memory>

#include <cuda.h>

#include <holoviz/holoviz_export.h>

#include "common.hpp"

namespace holoviz {

enum class ImageUsage : uint32_t {
  kNone = 0,
  kSampled = 1,
  kRenderTarget = 2,
};

struct ImageDesc {
  Extent2D extent;
  Format format = Format::kUnknown;
  ImageUsage usage = ImageUsage::kNone;  // Default to no usage; must be explicitly set by the user.
  uint32_t mip_levels = 1;    // Mip levels != 1 not supported yet, here for API completeness
  uint32_t array_layers = 1;  // Array layers != 1 not supported yet, here for API completeness
  ShareMode share = ShareMode::kPrivate;
  const char* debug_name = nullptr;
  /// Enable CUDA surface writes to the imported array; requires kExported sharing.
  bool cuda_surface_load_store = false;
};

enum class FilterMode : uint8_t { kNearest, kLinear };

enum class AddressMode : uint8_t {
  kClampToEdge,
  kClampToBorder,
  kRepeat,
  kMirroredRepeat,
};

/// Sampler state as a plain value — there is no sampler object to create or
/// destroy. The device dedupes identical descs into one backend sampler, so
/// passing SamplerDesc by value everywhere costs nothing.
struct SamplerDesc {
  FilterMode min_filter = FilterMode::kLinear;
  FilterMode mag_filter = FilterMode::kLinear;
  AddressMode address_u = AddressMode::kClampToEdge;
  AddressMode address_v = AddressMode::kClampToEdge;

  bool is_equal(const SamplerDesc& other) const {
    return min_filter == other.min_filter && mag_filter == other.mag_filter &&
           address_u == other.address_u && address_v == other.address_v;
  }
};

namespace detail {
class ImageImpl;
}  // namespace detail

/// A 2D device image, transitioned to its home state at creation. Swapchain
/// images are also handed out as Image pointers (owned by the Swapchain) and
/// obey the same model, except that their transition runs behind the image's
/// first acquire (Vulkan forbids touching a presentable image before then).
class Image {
 public:
  HOLOVIZ_EXPORT explicit Image(std::unique_ptr<detail::ImageImpl> impl);
  HOLOVIZ_EXPORT Image();
  HOLOVIZ_EXPORT Image(Image&&) noexcept;
  HOLOVIZ_EXPORT Image& operator=(Image&&) noexcept;
  Image(const Image&) = delete;
  Image& operator=(const Image&) = delete;
  HOLOVIZ_EXPORT ~Image();

  explicit operator bool() const { return impl_ != nullptr; }

  /// Release the underlying backend resources now instead of at destruction.
  /// \p timeout bounds how long to wait for the object to stop being in use:
  /// zero (the default) fails immediately with ErrorCode::kBackpressured if
  /// the object is in use; a non-zero timeout waits up to that duration before
  /// returning ErrorCode::kBackpressured. Only CommandList currently tracks
  /// in-use state; other objects destroy immediately. On success (and on any
  /// error other than kBackpressured) the handle becomes empty. Safe to call
  /// on an empty handle (no-op success). The destructor calls this
  /// automatically and logs any error.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> destroy(
      std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero());

  [[nodiscard]] HOLOVIZ_EXPORT expected<CUmipmappedArray> cuda_mipmapped_array() const;

  [[nodiscard]] HOLOVIZ_EXPORT expected<CUarray> cuda_mipmapped_array_get_level(
      uint32_t level) const;

  [[nodiscard]] HOLOVIZ_EXPORT expected<Extent2D> extent() const;
  /// Return the image view format. For swapchain images with an sRGB attachment view,
  /// this can differ from the scanout storage format's transfer function.
  [[nodiscard]] HOLOVIZ_EXPORT expected<Format> format() const;

  /// Vulkan: answers kVkDevice, kVkImage, kVkImageView.
  [[nodiscard]] HOLOVIZ_EXPORT expected<NativeObject> native_object(NativeObjectType type) const;

  /// Backend-internal access; not part of the stable API surface.
  /// Precondition: the handle is non-empty.
  detail::ImageImpl& impl() { return *impl_; }
  const detail::ImageImpl& impl() const { return *impl_; }

 private:
  std::unique_ptr<detail::ImageImpl> impl_;
};

}  // namespace holoviz
