/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>

#include <cuda.h>

#include <holoviz/adapter.hpp>
#include <holoviz/buffer.hpp>
#include <holoviz/command_list.hpp>
#include <holoviz/device.hpp>
#include <holoviz/external_semaphore.hpp>
#include <holoviz/image.hpp>
#include <holoviz/pipeline.hpp>
#include <holoviz/query.hpp>
#include <holoviz/surface.hpp>
#include <holoviz/swapchain.hpp>

namespace holoviz::detail {

/// Backend interface for Device; see the Device handle for API documentation.
/// The sibling *Impl interfaces referenced by the factory methods are only
/// forward-declared here (via the public headers); backends include the
/// concrete detail headers they implement.
class DeviceImpl {
 public:
  DeviceImpl(const DeviceImpl&) = delete;
  DeviceImpl& operator=(const DeviceImpl&) = delete;
  DeviceImpl(DeviceImpl&&) = delete;
  DeviceImpl& operator=(DeviceImpl&&) = delete;
  virtual ~DeviceImpl() = default;

  /// Release all owned backend resources now. \p timeout bounds how long to
  /// wait for the object to stop being in use: zero (the default) fails
  /// immediately with ErrorCode::kBackpressured if the object is in use, and a
  /// non-zero timeout waits up to that duration before returning
  /// ErrorCode::kBackpressured. Backends without in-use tracking ignore the
  /// timeout and destroy immediately. Idempotent: repeated calls are no-ops
  /// returning success. Called by the destructor if not already destroyed.
  [[nodiscard]] virtual expected<void> destroy(
      std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero()) = 0;

  virtual const Adapter& adapter() const = 0;
  virtual const CapabilityReport& capabilities() const = 0;
  virtual Backend backend() const = 0;
  virtual DevicePolicy policy() const = 0;
  [[nodiscard]] virtual expected<void> seal() = 0;
  virtual bool sealed() const = 0;
  [[nodiscard]] virtual expected<void> wait_idle() = 0;
  [[nodiscard]] virtual expected<void> wait_on_submit(SubmitSyncToken sync_token,
                                                      std::chrono::nanoseconds timeout) = 0;
  [[nodiscard]] virtual expected<SubmitSyncToken> submit(const SubmitDesc& batch) = 0;

  /// \p present_mode narrows the image-count/extent bounds to one mode; nullopt queries the
  /// mode-agnostic capabilities.
  [[nodiscard]] virtual expected<SurfaceCapabilities> surface_capabilities(
      const SurfaceImpl& surface, std::optional<PresentMode> present_mode) const = 0;

  [[nodiscard]] virtual expected<CUdevice> cuda_device() const = 0;
  [[nodiscard]] virtual expected<CUcontext> cuda_context() const = 0;
  [[nodiscard]] virtual expected<bool> cuda_is_cig() const = 0;

  [[nodiscard]] virtual expected<std::unique_ptr<BufferImpl>> create_buffer(
      const BufferDesc& desc) const = 0;
  [[nodiscard]] virtual expected<std::unique_ptr<ImageImpl>> create_image(
      const ImageDesc& desc) const = 0;
  [[nodiscard]] virtual expected<std::unique_ptr<RenderPassImpl>> create_render_pass(
      const RenderPassDesc& desc) const = 0;
  [[nodiscard]] virtual expected<std::unique_ptr<PipelineImpl>> create_pipeline(
      const PipelineDesc& desc) const = 0;
  [[nodiscard]] virtual expected<std::unique_ptr<CommandListImpl>> create_command_list(
      const CommandListDesc& desc) const = 0;
  [[nodiscard]] virtual expected<std::unique_ptr<SwapchainImpl>> create_swapchain(
      const SwapchainDesc& desc) const = 0;
  [[nodiscard]] virtual expected<std::unique_ptr<QueryPoolImpl>> create_query_pool(
      const QueryPoolDesc& desc) const = 0;
  [[nodiscard]] virtual expected<std::unique_ptr<ExternalSemaphoreImpl>> create_external_semaphore(
      CUstream stream) const = 0;

  [[nodiscard]] virtual expected<void> upload_to_buffer(const BufferSlice& dst,
                                                        std::span<const std::byte> bytes) = 0;
  [[nodiscard]] virtual expected<void> download_from_buffer(const BufferSlice& src,
                                                            std::span<std::byte> bytes) = 0;
  [[nodiscard]] virtual expected<void> upload_to_image(const ImageImpl& dst, uint32_t mip,
                                                       std::span<const std::byte> bytes,
                                                       size_t row_pitch_bytes) = 0;
  [[nodiscard]] virtual expected<void> download_from_image(const ImageImpl& src, uint32_t mip,
                                                           std::span<std::byte> bytes,
                                                           size_t row_pitch_bytes) = 0;
  [[nodiscard]] virtual expected<NativeObject> native_object(NativeObjectType type) const = 0;

  /// Terminal poison: a child object's teardown gave up on wedged pending work and leaked
  /// resources that still reference this device (see the Vulkan device for the concrete
  /// semantics — its destroy() then leaks the native device instead of destroying the
  /// parent of live children). Sticky for the device's lifetime; const because children
  /// hold a const device pointer. Default: no-op for backends without the mechanism.
  virtual void report_leaked_resources() const {}
  [[nodiscard]] virtual bool resources_leaked() const { return false; }

 protected:
  DeviceImpl() = default;
};

}  // namespace holoviz::detail
