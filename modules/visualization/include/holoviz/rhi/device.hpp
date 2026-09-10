/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Device: the root object of the RHI. One Device per adapter, with the single
// graphics queue folded in — submit() is a Device method and there is no
// separate queue class, because one device + one graphics queue is the only
// topology this module targets (and the one that maps to every backend).

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include <cuda.h>

#include <holoviz/holoviz_export.h>

#include <holoviz/expected.hpp>

#include "buffer.hpp"
#include "command_list.hpp"
#include "common.hpp"
#include "external_semaphore.hpp"
#include "image.hpp"
#include "pipeline.hpp"
#include "query.hpp"
#include "surface.hpp"
#include "swapchain.hpp"

namespace holoviz::rhi {

struct Adapter;
struct CapabilityReport;

struct SubmitDesc {
  /// Tokens with timeline_value == 0 are skipped (nothing to wait on).
  std::span<const SubmitSyncToken> wait_submit_sync_tokens;
  /// Default-constructed entries are skipped (no-op tokens); see ExternalSyncToken.
  std::span<const ExternalSyncToken> wait_external_sync_tokens;
  std::span<CommandList> command_lists;
  std::span<AcquireSignalGate> acquire_gates;
  std::span<PresentWaitGate> present_gates;
};

namespace detail {
class DeviceImpl;
}

/// A logical device plus its single graphics queue. All resources are created
/// through it and must not outlive it.
///
/// Every submit also waits on and advances an internal *device timeline*
/// (a monotonic counter signaled once per submit, in submission order).
/// submit() returns the value the timeline reaches when that batch completes;
/// it is the module's universal "this work is done" token — command-list
/// reset() blocks on it, and exported it is what CUDA consumers will wait on.
class Device {
 public:
  HOLOVIZ_EXPORT explicit Device(std::unique_ptr<detail::DeviceImpl> impl);
  HOLOVIZ_EXPORT Device();
  HOLOVIZ_EXPORT Device(Device&&) noexcept;
  HOLOVIZ_EXPORT Device& operator=(Device&&) noexcept;
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  HOLOVIZ_EXPORT ~Device();

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
  /// @warning Objects created from this device (buffers, images, swapchains,
  /// command lists, ...) reference it during their own teardown. Destroy all
  /// child objects before destroying the device; destroying the device first
  /// is undefined behavior.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> destroy(
      std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero());

  [[nodiscard]] HOLOVIZ_EXPORT expected<Adapter> adapter() const;

  /// Return the adapter limits snapshot taken at device creation (see
  /// CapabilityReport in adapter.hpp). Query once during setup and branch on
  /// capabilities up front instead of handling per-call failures.
  [[nodiscard]] HOLOVIZ_EXPORT expected<CapabilityReport> capabilities() const;

  [[nodiscard]] HOLOVIZ_EXPORT expected<Backend> backend() const;
  [[nodiscard]] HOLOVIZ_EXPORT expected<DevicePolicy> policy() const;

  /// End the setup phase. Not implemented yet — currently returns an error.
  /// Once kSealed lands it will make further create_* calls fail.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> seal();
  [[nodiscard]] HOLOVIZ_EXPORT expected<bool> sealed() const;

  /// Block the host until the graphics queue drains. Teardown/debug tool —
  /// never part of the frame loop.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> wait_idle();

  /// Block the host until the device timeline reaches \p timeline_value (one
  /// returned by submit() on this device); the host-side counterpart of
  /// cuda_wait(). Returns kBackpressured if \p timeout elapses first. This is
  /// the completion gate for host readback of work products — e.g. wait on
  /// the submit that resolved a query range before QueryPool::results_ns().
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> wait_on_submit(SubmitSyncToken sync_token,
                                                             std::chrono::nanoseconds timeout);

  /// Submit a batch to the graphics queue and return the device-timeline
  /// value that completes with it. Consumes every gate in the batch; passing
  /// an already-consumed gate is an error.
  [[nodiscard]] HOLOVIZ_EXPORT expected<SubmitSyncToken> submit(const SubmitDesc& batch);

  /// Report what this (device, surface) pair supports: image-count bounds, extent bounds,
  /// and the present modes actually usable (surface-reported, masked to enabled device
  /// extensions). Query during setup — this calls into the platform surface layer and is
  /// not frame-path work.
  [[nodiscard]] HOLOVIZ_EXPORT expected<SurfaceCapabilities> surface_capabilities(
      const Surface& surface) const;

  [[nodiscard]] HOLOVIZ_EXPORT expected<CUdevice> cuda_device() const;
  [[nodiscard]] HOLOVIZ_EXPORT expected<CUcontext> cuda_context() const;
  [[nodiscard]] HOLOVIZ_EXPORT expected<bool> cuda_is_cig() const;

  [[nodiscard]] HOLOVIZ_EXPORT expected<Buffer> create_buffer(const BufferDesc& desc) const;
  [[nodiscard]] HOLOVIZ_EXPORT expected<Image> create_image(const ImageDesc& desc) const;
  [[nodiscard]] HOLOVIZ_EXPORT expected<RenderPass> create_render_pass(
      const RenderPassDesc& desc) const;
  [[nodiscard]] HOLOVIZ_EXPORT expected<Pipeline> create_pipeline(const PipelineDesc& desc) const;
  [[nodiscard]] HOLOVIZ_EXPORT expected<CommandList> create_command_list(
      const CommandListDesc& desc) const;
  [[nodiscard]] HOLOVIZ_EXPORT expected<Swapchain> create_swapchain(
      const SwapchainDesc& desc) const;
  [[nodiscard]] HOLOVIZ_EXPORT expected<QueryPool> create_query_pool(
      const QueryPoolDesc& desc) const;
  [[nodiscard]] HOLOVIZ_EXPORT expected<ExternalSemaphore> create_external_semaphore(
      CUstream stream) const;

  // The upload/download helpers below are synchronous: each stages through an
  // internal host-visible buffer, submits a copy, and blocks until the GPU
  // finishes. They exist for setup-time initialization of static content;
  // per-frame data belongs in CUDA-written exported buffers or push data.

  [[nodiscard]] HOLOVIZ_EXPORT expected<void> upload_to_buffer(const BufferSlice& dst,
                                                               std::span<const std::byte> bytes);
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> download_from_buffer(const BufferSlice& src,
                                                                   std::span<std::byte> bytes);

  /// Upload pixel data into an image. Only `mip` 0 is currently supported (image
  /// creation rejects `mip_levels != 1`; the parameter is here for API completeness).
  /// `row_pitch_bytes` is the source row stride (0 = tightly packed); it must be a
  /// multiple of the texel size and at least one tightly packed row of the image.
  /// The image is transitioned for the copy and restored to its home layout before
  /// the call returns (see image.hpp).
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> upload_to_image(const Image& dst, uint32_t mip,
                                                              std::span<const std::byte> bytes,
                                                              size_t row_pitch_bytes);

  /// Read pixel data back from an image; mirrors the upload_to_image() contract,
  /// including the mip-0-only limit.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> download_from_image(const Image& src, uint32_t mip,
                                                                  std::span<std::byte> bytes,
                                                                  size_t row_pitch_bytes);

  // [[nodiscard]] expected<std::span<DisplayInfo>> enumerate_displays() const;

  /// Vulkan: answers kVkInstance, kVkPhysicalDevice, kVkDevice, kVkQueue.
  [[nodiscard]] HOLOVIZ_EXPORT expected<NativeObject> native_object(NativeObjectType type) const;

  /// Backend-internal access; not part of the stable API surface.
  /// Precondition: the handle is non-empty.
  detail::DeviceImpl& impl() { return *impl_; }
  const detail::DeviceImpl& impl() const { return *impl_; }

 private:
  std::unique_ptr<detail::DeviceImpl> impl_;
};

}  // namespace holoviz::rhi
