/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <functional>

#include <holoviz/rhi/swapchain.hpp>

namespace holoviz::rhi::detail {

/// Backend interface for AcquireSignalGate; see the handle for API docs.
class AcquireSignalGateImpl {
 public:
  AcquireSignalGateImpl(const AcquireSignalGateImpl&) = delete;
  AcquireSignalGateImpl& operator=(const AcquireSignalGateImpl&) = delete;
  AcquireSignalGateImpl(AcquireSignalGateImpl&&) = delete;
  AcquireSignalGateImpl& operator=(AcquireSignalGateImpl&&) = delete;
  virtual ~AcquireSignalGateImpl() = default;

  [[nodiscard]] virtual expected<void> wait(std::chrono::nanoseconds timeout) = 0;
  [[nodiscard]] virtual expected<bool> consumed() const = 0;
  [[nodiscard]] virtual expected<void> consume() = 0;
  [[nodiscard]] virtual expected<NativeObject> native_object(NativeObjectType type) const = 0;

  /// Stage the consuming submit blocks at; set before submitting.
  GateWaitStage wait_stage = GateWaitStage::kColorAttachmentOutput;

 protected:
  AcquireSignalGateImpl() = default;
};

/// Backend interface for PresentWaitGate; see the handle for API docs.
class PresentWaitGateImpl {
 public:
  PresentWaitGateImpl(const PresentWaitGateImpl&) = delete;
  PresentWaitGateImpl& operator=(const PresentWaitGateImpl&) = delete;
  PresentWaitGateImpl(PresentWaitGateImpl&&) = delete;
  PresentWaitGateImpl& operator=(PresentWaitGateImpl&&) = delete;
  virtual ~PresentWaitGateImpl() = default;

  [[nodiscard]] virtual expected<void> signal() = 0;
  [[nodiscard]] virtual expected<bool> consumed() const = 0;
  [[nodiscard]] virtual expected<void> consume() = 0;
  [[nodiscard]] virtual expected<NativeObject> native_object(NativeObjectType type) const = 0;

 protected:
  PresentWaitGateImpl() = default;
};

/// Backend interface for Swapchain; see the Swapchain handle for API docs.
class SwapchainImpl {
 public:
  SwapchainImpl(const SwapchainImpl&) = delete;
  SwapchainImpl& operator=(const SwapchainImpl&) = delete;
  SwapchainImpl(SwapchainImpl&&) = delete;
  SwapchainImpl& operator=(SwapchainImpl&&) = delete;
  virtual ~SwapchainImpl() = default;

  /// Release all owned backend resources now. \p timeout bounds how long to
  /// wait for the object to stop being in use: zero (the default) fails
  /// immediately with ErrorCode::kBackpressured if the object is in use, and a
  /// non-zero timeout waits up to that duration before returning
  /// ErrorCode::kBackpressured. Backends without in-use tracking ignore the
  /// timeout and destroy immediately. Idempotent: repeated calls are no-ops
  /// returning success. Called by the destructor if not already destroyed.
  [[nodiscard]] virtual expected<void> destroy(
      std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero()) = 0;

  /// See Swapchain::resize for API docs.
  [[nodiscard]] virtual expected<void> resize(Extent2D new_extent,
                                              std::chrono::nanoseconds timeout) = 0;

  virtual Extent2D extent() const = 0;
  virtual Format format() const = 0;
  virtual ColorSpace color_space() const = 0;
  virtual PresentMode present_mode() const = 0;
  /// See Swapchain::set_present_mode for API docs.
  [[nodiscard]] virtual expected<void> set_present_mode(PresentMode mode) = 0;
  [[nodiscard]] virtual expected<std::reference_wrapper<const Image>> image(
      uint32_t index) const = 0;
  virtual uint32_t image_count() const = 0;
  virtual uint64_t generation() const = 0;
  [[nodiscard]] virtual expected<AcquireSyncToken> acquire(std::chrono::nanoseconds timeout) = 0;
  [[nodiscard]] virtual expected<void> present(AcquireSyncToken&& acquired,
                                               SubmitSyncToken submit_sync_token,
                                               uint64_t present_id, PresentConsumed* consumed) = 0;
  [[nodiscard]] virtual expected<void> wait_for_present(uint64_t present_id,
                                                        std::chrono::nanoseconds timeout) = 0;
  [[nodiscard]] virtual expected<NativeObject> native_object(NativeObjectType type) const = 0;

 protected:
  SwapchainImpl() = default;
};

}  // namespace holoviz::rhi::detail
