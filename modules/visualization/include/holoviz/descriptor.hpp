/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Descriptor: one pre-baked set of shader resource bindings, created against
// a specific Pipeline (Pipeline::create_descriptor) and shaped by that
// pipeline's Slang-reflected interface. Populate during setup, then bind per
// draw with CommandList::bind_descriptor — this pre-baked set is the one
// binding mechanism, chosen over per-draw descriptor writes and
// buffer-device-address tricks that do not port to every target.

#pragma once

#include <chrono>
#include <memory>
#include <string>

#include <holoviz/holoviz_export.h>

#include "buffer.hpp"
#include "common.hpp"
#include "image.hpp"

namespace holoviz {

class Device;

namespace detail {
class DescriptorImpl;
}  // namespace detail

/// Binding slots can be addressed by reflected index or by the variable name
/// used in the shader source. Bound resources are borrowed — they must outlive
/// any use of the descriptor. Writes update the backend set immediately, so
/// do not rebind slots while a submit using this descriptor is still in
/// flight; the originating Pipeline must outlive the descriptor.
class Descriptor {
 public:
  HOLOVIZ_EXPORT explicit Descriptor(std::unique_ptr<detail::DescriptorImpl> impl);
  HOLOVIZ_EXPORT Descriptor();
  HOLOVIZ_EXPORT Descriptor(Descriptor&&) noexcept;
  HOLOVIZ_EXPORT Descriptor& operator=(Descriptor&&) noexcept;
  Descriptor(const Descriptor&) = delete;
  Descriptor& operator=(const Descriptor&) = delete;
  HOLOVIZ_EXPORT ~Descriptor();

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

  /// Bind a buffer range as a storage buffer.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> bind_buffer(const BufferSlice& slice,
                                                          uint32_t binding_index);
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> bind_buffer(const BufferSlice& slice,
                                                          const std::string& binding_name);

  /// Bind an image for sampling; pair it with a bind_sampler slot. The image must
  /// have been created with ImageUsage::kSampled — render targets and swapchain
  /// images live in a different home layout and are rejected.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> bind_image(const Image& image,
                                                         uint32_t binding_index);
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> bind_image(const Image& image,
                                                         const std::string& binding_name);

  /// Bind sampler state by value; the device dedupes identical descs into one
  /// backend sampler (see SamplerDesc in image.hpp).
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> bind_sampler(Device& device,
                                                           const SamplerDesc& sampler,
                                                           uint32_t binding_index);
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> bind_sampler(Device& device,
                                                           const SamplerDesc& sampler,
                                                           const std::string& binding_name);

  /// Vulkan: answers kVkDevice, kVkDescriptorSetLayout, kVkPipelineLayout,
  /// kVkDescriptorPool, kVkDescriptorSet.
  [[nodiscard]] HOLOVIZ_EXPORT expected<NativeObject> native_object(NativeObjectType type) const;

  /// Backend-internal access; not part of the stable API surface.
  /// Precondition: the handle is non-empty.
  detail::DescriptorImpl& impl() { return *impl_; }
  const detail::DescriptorImpl& impl() const { return *impl_; }

 private:
  std::unique_ptr<detail::DescriptorImpl> impl_;
};

}  // namespace holoviz
