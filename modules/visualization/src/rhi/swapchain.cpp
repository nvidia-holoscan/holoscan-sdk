/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "holoviz/rhi/swapchain.hpp"

#include <memory>
#include <utility>

#include "detail/handle_errors.hpp"
#include "detail/swapchain_impl.hpp"

namespace holoviz::rhi {

AcquireSignalGate::AcquireSignalGate(detail::AcquireSignalGateImpl* impl) : impl_(impl) {}

AcquireSignalGate::AcquireSignalGate(detail::AcquireSignalGateImpl* impl,
                                     const detail::SwapchainImpl* origin, uint64_t generation)
    : impl_(impl), origin_(origin), generation_(generation) {}

expected<void> AcquireSignalGate::wait(std::chrono::nanoseconds timeout) {
  if (!impl_) {
    return detail::null_impl_error("AcquireSignalGate::wait");
  }
  if (origin_ != nullptr && origin_->generation() != generation_) {
    return detail::stale_gate_error("AcquireSignalGate::wait");
  }
  return impl_->wait(timeout);
}

expected<bool> AcquireSignalGate::consumed() const {
  if (!impl_) {
    return detail::null_impl_error("AcquireSignalGate::consumed");
  }
  if (origin_ != nullptr && origin_->generation() != generation_) {
    return detail::stale_gate_error("AcquireSignalGate::consumed");
  }
  return impl_->consumed();
}

expected<void> AcquireSignalGate::consume() {
  if (!impl_) {
    return detail::null_impl_error("AcquireSignalGate::consume");
  }
  if (origin_ != nullptr && origin_->generation() != generation_) {
    return detail::stale_gate_error("AcquireSignalGate::consume");
  }
  return impl_->consume();
}

expected<NativeObject> AcquireSignalGate::native_object(NativeObjectType type) const {
  if (!impl_) {
    return detail::null_impl_error("AcquireSignalGate::native_object");
  }
  if (origin_ != nullptr && origin_->generation() != generation_) {
    return detail::stale_gate_error("AcquireSignalGate::native_object");
  }
  return impl_->native_object(type);
}

expected<GateWaitStage> AcquireSignalGate::wait_stage() const {
  if (!impl_) {
    return detail::null_impl_error("AcquireSignalGate::wait_stage");
  }
  if (origin_ != nullptr && origin_->generation() != generation_) {
    return detail::stale_gate_error("AcquireSignalGate::wait_stage");
  }
  return impl_->wait_stage;
}

expected<void> AcquireSignalGate::set_wait_stage(GateWaitStage stage) {
  if (!impl_) {
    return detail::null_impl_error("AcquireSignalGate::set_wait_stage");
  }
  if (origin_ != nullptr && origin_->generation() != generation_) {
    return detail::stale_gate_error("AcquireSignalGate::set_wait_stage");
  }
  impl_->wait_stage = stage;
  return {};
}

PresentWaitGate::PresentWaitGate(detail::PresentWaitGateImpl* impl) : impl_(impl) {}

PresentWaitGate::PresentWaitGate(detail::PresentWaitGateImpl* impl,
                                 const detail::SwapchainImpl* origin, uint64_t generation)
    : impl_(impl), origin_(origin), generation_(generation) {}

expected<void> PresentWaitGate::signal() {
  if (!impl_) {
    return detail::null_impl_error("PresentWaitGate::signal");
  }
  if (origin_ != nullptr && origin_->generation() != generation_) {
    return detail::stale_gate_error("PresentWaitGate::signal");
  }
  return impl_->signal();
}

expected<bool> PresentWaitGate::consumed() const {
  if (!impl_) {
    return detail::null_impl_error("PresentWaitGate::consumed");
  }
  if (origin_ != nullptr && origin_->generation() != generation_) {
    return detail::stale_gate_error("PresentWaitGate::consumed");
  }
  return impl_->consumed();
}

expected<void> PresentWaitGate::consume() {
  if (!impl_) {
    return detail::null_impl_error("PresentWaitGate::consume");
  }
  if (origin_ != nullptr && origin_->generation() != generation_) {
    return detail::stale_gate_error("PresentWaitGate::consume");
  }
  return impl_->consume();
}

expected<NativeObject> PresentWaitGate::native_object(NativeObjectType type) const {
  if (!impl_) {
    return detail::null_impl_error("PresentWaitGate::native_object");
  }
  if (origin_ != nullptr && origin_->generation() != generation_) {
    return detail::stale_gate_error("PresentWaitGate::native_object");
  }
  return impl_->native_object(type);
}

Swapchain::Swapchain(std::unique_ptr<detail::SwapchainImpl> impl) : impl_(std::move(impl)) {}
Swapchain::Swapchain() = default;
Swapchain::Swapchain(Swapchain&&) noexcept = default;
Swapchain& Swapchain::operator=(Swapchain&&) noexcept = default;
Swapchain::~Swapchain() = default;

expected<void> Swapchain::destroy(std::chrono::nanoseconds timeout) {
  if (!impl_) {
    return {};  // empty handle: nothing to free
  }
  expected<void> result = impl_->destroy(timeout);
  if (result || result.error().code != ErrorCode::kBackpressured) {
    impl_.reset();
  }
  return result;
}

expected<void> Swapchain::resize(Extent2D new_extent, std::chrono::nanoseconds timeout) {
  if (!impl_) {
    return detail::null_impl_error("Swapchain::resize");
  }
  return impl_->resize(new_extent, timeout);
}

expected<Extent2D> Swapchain::extent() const {
  if (!impl_) {
    return detail::null_impl_error("Swapchain::extent");
  }
  return impl_->extent();
}

expected<Format> Swapchain::format() const {
  if (!impl_) {
    return detail::null_impl_error("Swapchain::format");
  }
  return impl_->format();
}

expected<ColorSpace> Swapchain::color_space() const {
  if (!impl_) {
    return detail::null_impl_error("Swapchain::color_space");
  }
  return impl_->color_space();
}

expected<PresentMode> Swapchain::present_mode() const {
  if (!impl_) {
    return detail::null_impl_error("Swapchain::present_mode");
  }
  return impl_->present_mode();
}

expected<void> Swapchain::set_present_mode(PresentMode mode) {
  if (!impl_) {
    return detail::null_impl_error("Swapchain::set_present_mode");
  }
  return impl_->set_present_mode(mode);
}

expected<std::reference_wrapper<const Image>> Swapchain::image(uint32_t index) const {
  if (!impl_) {
    return detail::null_impl_error("Swapchain::image");
  }
  return impl_->image(index);
}

expected<uint32_t> Swapchain::image_count() const {
  if (!impl_) {
    return detail::null_impl_error("Swapchain::image_count");
  }
  return impl_->image_count();
}

expected<uint64_t> Swapchain::generation() const {
  if (!impl_) {
    return detail::null_impl_error("Swapchain::generation");
  }
  return impl_->generation();
}

expected<AcquireSyncToken> Swapchain::acquire(std::chrono::nanoseconds timeout) {
  if (!impl_) {
    return detail::null_impl_error("Swapchain::acquire");
  }
  return impl_->acquire(timeout);
}

expected<void> Swapchain::present(AcquireSyncToken&& acquired, SubmitSyncToken submit_sync_token,
                                  uint64_t present_id, PresentConsumed* consumed) {
  if (!impl_) {
    return detail::null_impl_error("Swapchain::present");
  }
  return impl_->present(std::move(acquired), submit_sync_token, present_id, consumed);
}

expected<void> Swapchain::wait_for_present(uint64_t present_id, std::chrono::nanoseconds timeout) {
  if (!impl_) {
    return detail::null_impl_error("Swapchain::wait_for_present");
  }
  return impl_->wait_for_present(present_id, timeout);
}

expected<NativeObject> Swapchain::native_object(NativeObjectType type) const {
  if (!impl_) {
    return detail::null_impl_error("Swapchain::native_object");
  }
  return impl_->native_object(type);
}

}  // namespace holoviz::rhi
