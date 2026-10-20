/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "holoviz/descriptor.hpp"

#include <memory>
#include <string>
#include <utility>

#include "holoviz/device.hpp"

#include "detail/descriptor_impl.hpp"
#include "detail/handle_errors.hpp"

namespace holoviz {

Descriptor::Descriptor(std::unique_ptr<detail::DescriptorImpl> impl) : impl_(std::move(impl)) {}
Descriptor::Descriptor() = default;
Descriptor::Descriptor(Descriptor&&) noexcept = default;
Descriptor& Descriptor::operator=(Descriptor&&) noexcept = default;
Descriptor::~Descriptor() = default;

expected<void> Descriptor::destroy(std::chrono::nanoseconds timeout) {
  if (!impl_) {
    return {};  // empty handle: nothing to free
  }
  expected<void> result = impl_->destroy(timeout);
  if (result || result.error().code != ErrorCode::kBackpressured) {
    impl_.reset();
  }
  return result;
}

expected<void> Descriptor::bind_buffer(const BufferSlice& slice, uint32_t binding_index) {
  if (!impl_) {
    return detail::null_impl_error("Descriptor::bind_buffer");
  }
  if (slice.buffer == nullptr) {
    return detail::null_arg_error("Descriptor::bind_buffer", "slice");
  }
  return impl_->bind_buffer(slice, binding_index);
}

expected<void> Descriptor::bind_buffer(const BufferSlice& slice, const std::string& binding_name) {
  if (!impl_) {
    return detail::null_impl_error("Descriptor::bind_buffer");
  }
  if (slice.buffer == nullptr) {
    return detail::null_arg_error("Descriptor::bind_buffer", "slice");
  }
  return impl_->bind_buffer(slice, binding_name);
}

expected<void> Descriptor::bind_image(const Image& image, uint32_t binding_index) {
  if (!impl_) {
    return detail::null_impl_error("Descriptor::bind_image");
  }
  if (!image) {
    return detail::null_arg_error("Descriptor::bind_image", "image");
  }
  return impl_->bind_image(image.impl(), binding_index);
}

expected<void> Descriptor::bind_image(const Image& image, const std::string& binding_name) {
  if (!impl_) {
    return detail::null_impl_error("Descriptor::bind_image");
  }
  if (!image) {
    return detail::null_arg_error("Descriptor::bind_image", "image");
  }
  return impl_->bind_image(image.impl(), binding_name);
}

expected<void> Descriptor::bind_sampler(Device& device, const SamplerDesc& sampler,
                                        uint32_t binding_index) {
  if (!impl_) {
    return detail::null_impl_error("Descriptor::bind_sampler");
  }
  if (!device) {
    return detail::null_arg_error("Descriptor::bind_sampler", "device");
  }
  return impl_->bind_sampler(device.impl(), sampler, binding_index);
}

expected<void> Descriptor::bind_sampler(Device& device, const SamplerDesc& sampler,
                                        const std::string& binding_name) {
  if (!impl_) {
    return detail::null_impl_error("Descriptor::bind_sampler");
  }
  if (!device) {
    return detail::null_arg_error("Descriptor::bind_sampler", "device");
  }
  return impl_->bind_sampler(device.impl(), sampler, binding_name);
}

expected<NativeObject> Descriptor::native_object(NativeObjectType type) const {
  if (!impl_) {
    return detail::null_impl_error("Descriptor::native_object");
  }
  return impl_->native_object(type);
}

}  // namespace holoviz
