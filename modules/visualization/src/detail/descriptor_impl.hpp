/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>

#include <string>

#include <holoviz/descriptor.hpp>

namespace holoviz::detail {

class DeviceImpl;
class ImageImpl;

/// Backend interface for Descriptor; see the Descriptor handle for API docs.
class DescriptorImpl {
 public:
  DescriptorImpl(const DescriptorImpl&) = delete;
  DescriptorImpl& operator=(const DescriptorImpl&) = delete;
  DescriptorImpl(DescriptorImpl&&) = delete;
  DescriptorImpl& operator=(DescriptorImpl&&) = delete;
  virtual ~DescriptorImpl() = default;

  /// Release all owned backend resources now. \p timeout bounds how long to
  /// wait for the object to stop being in use: zero (the default) fails
  /// immediately with ErrorCode::kBackpressured if the object is in use, and a
  /// non-zero timeout waits up to that duration before returning
  /// ErrorCode::kBackpressured. Backends without in-use tracking ignore the
  /// timeout and destroy immediately. Idempotent: repeated calls are no-ops
  /// returning success. Called by the destructor if not already destroyed.
  [[nodiscard]] virtual expected<void> destroy(std::chrono::nanoseconds timeout) = 0;

  [[nodiscard]] virtual expected<void> bind_buffer(const BufferSlice& slice,
                                                   uint32_t binding_index) = 0;
  [[nodiscard]] virtual expected<void> bind_buffer(const BufferSlice& slice,
                                                   const std::string& binding_name) = 0;
  [[nodiscard]] virtual expected<void> bind_image(const ImageImpl& image,
                                                  uint32_t binding_index) = 0;
  [[nodiscard]] virtual expected<void> bind_image(const ImageImpl& image,
                                                  const std::string& binding_name) = 0;
  [[nodiscard]] virtual expected<void> bind_sampler(DeviceImpl& device, const SamplerDesc& sampler,
                                                    uint32_t binding_index) = 0;
  [[nodiscard]] virtual expected<void> bind_sampler(DeviceImpl& device, const SamplerDesc& sampler,
                                                    const std::string& binding_name) = 0;
  [[nodiscard]] virtual expected<NativeObject> native_object(NativeObjectType type) const = 0;

 protected:
  DescriptorImpl() = default;
};

}  // namespace holoviz::detail
