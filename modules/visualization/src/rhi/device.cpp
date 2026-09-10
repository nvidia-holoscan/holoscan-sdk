/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "holoviz/rhi/device.hpp"
#include "holoviz/expected.hpp"
#include "holoviz/rhi/adapter.hpp"

#include <memory>
#include <utility>

#include "detail/buffer_impl.hpp"
#include "detail/command_list_impl.hpp"
#include "detail/device_impl.hpp"
#include "detail/external_semaphore_impl.hpp"
#include "detail/handle_errors.hpp"
#include "detail/image_impl.hpp"
#include "detail/pipeline_impl.hpp"
#include "detail/query_impl.hpp"
#include "detail/render_pass_impl.hpp"
#include "detail/swapchain_impl.hpp"

namespace holoviz::rhi {

Device::Device(std::unique_ptr<detail::DeviceImpl> impl) : impl_(std::move(impl)) {}
Device::Device() = default;
Device::Device(Device&&) noexcept = default;
Device& Device::operator=(Device&&) noexcept = default;
Device::~Device() = default;

expected<void> Device::destroy(std::chrono::nanoseconds timeout) {
  if (!impl_) {
    return {};  // empty handle: nothing to free
  }
  expected<void> result = impl_->destroy(timeout);
  if (result || result.error().code != ErrorCode::kBackpressured) {
    impl_.reset();
  }
  return result;
}

expected<Adapter> Device::adapter() const {
  if (!impl_) {
    return detail::null_impl_error("Device::adapter");
  }
  return impl_->adapter();
}

expected<CapabilityReport> Device::capabilities() const {
  if (!impl_) {
    return detail::null_impl_error("Device::capabilities");
  }
  return impl_->capabilities();
}

expected<Backend> Device::backend() const {
  if (!impl_) {
    return detail::null_impl_error("Device::backend");
  }
  return impl_->backend();
}

expected<DevicePolicy> Device::policy() const {
  if (!impl_) {
    return detail::null_impl_error("Device::policy");
  }
  return impl_->policy();
}

expected<void> Device::seal() {
  if (!impl_) {
    return detail::null_impl_error("Device::seal");
  }
  return impl_->seal();
}

expected<bool> Device::sealed() const {
  if (!impl_) {
    return detail::null_impl_error("Device::sealed");
  }
  return impl_->sealed();
}

expected<void> Device::wait_idle() {
  if (!impl_) {
    return detail::null_impl_error("Device::wait_idle");
  }
  return impl_->wait_idle();
}

expected<void> Device::wait_on_submit(SubmitSyncToken sync_token,
                                      std::chrono::nanoseconds timeout) {
  if (!impl_) {
    return detail::null_impl_error("Device::wait_on_submit");
  }
  return impl_->wait_on_submit(sync_token, timeout);
}

expected<SubmitSyncToken> Device::submit(const SubmitDesc& batch) {
  if (!impl_) {
    return detail::null_impl_error("Device::submit");
  }
  return impl_->submit(batch);
}

expected<SurfaceCapabilities> Device::surface_capabilities(const Surface& surface) const {
  if (!impl_) {
    return detail::null_impl_error("Device::surface_capabilities");
  }
  if (!surface) {
    return detail::null_arg_error("Device::surface_capabilities", "surface");
  }
  return impl_->surface_capabilities(surface.impl());
}

expected<CUdevice> Device::cuda_device() const {
  if (!impl_) {
    return detail::null_impl_error("Device::cuda_device");
  }
  return impl_->cuda_device();
}

expected<CUcontext> Device::cuda_context() const {
  if (!impl_) {
    return detail::null_impl_error("Device::cuda_context");
  }
  return impl_->cuda_context();
}

expected<bool> Device::cuda_is_cig() const {
  if (!impl_) {
    return detail::null_impl_error("Device::cuda_is_cig");
  }
  return impl_->cuda_is_cig();
}

expected<Buffer> Device::create_buffer(const BufferDesc& desc) const {
  if (!impl_) {
    return detail::null_impl_error("Device::create_buffer");
  }
  return impl_->create_buffer(desc).map(
      [](std::unique_ptr<detail::BufferImpl> impl) { return Buffer{std::move(impl)}; });
}

expected<Image> Device::create_image(const ImageDesc& desc) const {
  if (!impl_) {
    return detail::null_impl_error("Device::create_image");
  }
  return impl_->create_image(desc).map(
      [](std::unique_ptr<detail::ImageImpl> impl) { return Image{std::move(impl)}; });
}

expected<RenderPass> Device::create_render_pass(const RenderPassDesc& desc) const {
  if (!impl_) {
    return detail::null_impl_error("Device::create_render_pass");
  }
  return impl_->create_render_pass(desc).map(
      [](std::unique_ptr<detail::RenderPassImpl> impl) { return RenderPass{std::move(impl)}; });
}

expected<Pipeline> Device::create_pipeline(const PipelineDesc& desc) const {
  if (!impl_) {
    return detail::null_impl_error("Device::create_pipeline");
  }
  return impl_->create_pipeline(desc).map(
      [](std::unique_ptr<detail::PipelineImpl> impl) { return Pipeline{std::move(impl)}; });
}

expected<CommandList> Device::create_command_list(const CommandListDesc& desc) const {
  if (!impl_) {
    return detail::null_impl_error("Device::create_command_list");
  }
  return impl_->create_command_list(desc).map(
      [](std::unique_ptr<detail::CommandListImpl> impl) { return CommandList{std::move(impl)}; });
}

expected<Swapchain> Device::create_swapchain(const SwapchainDesc& desc) const {
  if (!impl_) {
    return detail::null_impl_error("Device::create_swapchain");
  }
  return impl_->create_swapchain(desc).map(
      [](std::unique_ptr<detail::SwapchainImpl> impl) { return Swapchain{std::move(impl)}; });
}

expected<QueryPool> Device::create_query_pool(const QueryPoolDesc& desc) const {
  if (!impl_) {
    return detail::null_impl_error("Device::create_query_pool");
  }
  return impl_->create_query_pool(desc).map(
      [](std::unique_ptr<detail::QueryPoolImpl> impl) { return QueryPool{std::move(impl)}; });
}

expected<ExternalSemaphore> Device::create_external_semaphore(CUstream stream) const {
  if (!impl_) {
    return detail::null_impl_error("Device::create_external_semaphore");
  }
  return impl_->create_external_semaphore(stream).map(
      [](std::unique_ptr<detail::ExternalSemaphoreImpl> impl) {
        return ExternalSemaphore{std::move(impl)};
      });
}

expected<void> Device::upload_to_buffer(const BufferSlice& dst, std::span<const std::byte> bytes) {
  if (!impl_) {
    return detail::null_impl_error("Device::upload_to_buffer");
  }
  if (dst.buffer == nullptr) {
    return detail::null_arg_error("Device::upload_to_buffer", "dst");
  }
  return impl_->upload_to_buffer(dst, bytes);
}

expected<void> Device::download_from_buffer(const BufferSlice& src, std::span<std::byte> bytes) {
  if (!impl_) {
    return detail::null_impl_error("Device::download_from_buffer");
  }
  if (src.buffer == nullptr) {
    return detail::null_arg_error("Device::download_from_buffer", "src");
  }
  return impl_->download_from_buffer(src, bytes);
}

expected<void> Device::upload_to_image(const Image& dst, uint32_t mip,
                                       std::span<const std::byte> bytes, size_t row_pitch_bytes) {
  if (!impl_) {
    return detail::null_impl_error("Device::upload_to_image");
  }
  if (!dst) {
    return detail::null_arg_error("Device::upload_to_image", "dst");
  }
  return impl_->upload_to_image(dst.impl(), mip, bytes, row_pitch_bytes);
}

expected<void> Device::download_from_image(const Image& src, uint32_t mip,
                                           std::span<std::byte> bytes, size_t row_pitch_bytes) {
  if (!impl_) {
    return detail::null_impl_error("Device::download_from_image");
  }
  if (!src) {
    return detail::null_arg_error("Device::download_from_image", "src");
  }
  return impl_->download_from_image(src.impl(), mip, bytes, row_pitch_bytes);
}

expected<NativeObject> Device::native_object(NativeObjectType type) const {
  if (!impl_) {
    return detail::null_impl_error("Device::native_object");
  }
  return impl_->native_object(type);
}

}  // namespace holoviz::rhi
