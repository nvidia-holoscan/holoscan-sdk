/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/wait_set.hpp>

#include <holomq/context.hpp>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>
#include <holomq/logger.hpp>
#include "context_impl.hpp"

#include <memory>
#include <utility>

namespace holomq {

WaitSetAttachment::WaitSetAttachment(WaitSetHandle handle,
                                     std::function<void(WaitSetHandle)> detach_fn) noexcept
    : handle_(handle), detach_fn_(std::move(detach_fn)) {}

WaitSetAttachment::WaitSetAttachment(WaitSetAttachment&& other) noexcept
    : handle_(other.handle_), detach_fn_(std::move(other.detach_fn_)) {
  other.handle_ = WaitSetHandle{};
  other.detach_fn_ = nullptr;
}

WaitSetAttachment& WaitSetAttachment::operator=(WaitSetAttachment&& other) noexcept {
  if (this != &other) {
    if (detach_fn_) {
      detach_fn_(handle_);
    }
    handle_ = other.handle_;
    detach_fn_ = std::move(other.detach_fn_);
    other.handle_ = WaitSetHandle{};
    other.detach_fn_ = nullptr;
  }
  return *this;
}

WaitSetAttachment::~WaitSetAttachment() {
  if (detach_fn_) {
    detach_fn_(handle_);
  }
}

WaitSet::WaitSet(std::shared_ptr<detail::transport::WaitSetBackend> backend) noexcept
    : backend_(std::move(backend)) {}

WaitSetAttachment WaitSet::make_attachment(const WaitSetHandle handle) const {
  const std::shared_ptr<detail::transport::WaitSetBackend> backend = backend_;
  return WaitSetAttachment(handle, [backend](const WaitSetHandle h) { backend->detach(h); });
}

auto WaitSet::attach_trigger() -> expected<WaitTriggerAttachment, Error> {
  auto trigger_or = WaitTrigger::create();
  if (!trigger_or.has_value()) {
    HOLOMQ_LOG_ERROR("WaitSet attach_trigger failed: {}", error_message(trigger_or.error()));
    return holomq::forward_error(std::move(trigger_or));
  }
  std::shared_ptr<WaitTrigger> trigger = std::move(trigger_or).value();

  // The backend receives the raw descriptor and a drain callback rather than the trigger type, so
  // the wake abstraction never has to befriend a transport backend. The retained owner keeps the
  // descriptor open for at least as long as the attachment.
  auto handle_or = backend_->attach_trigger(
      trigger, trigger->descriptor(), [raw = trigger.get()]() { raw->drain(); });
  if (!handle_or.has_value()) {
    return holomq::forward_error(std::move(handle_or));
  }

  const WaitSetHandle handle = handle_or.value();
  const std::shared_ptr<detail::transport::WaitSetBackend> backend = backend_;
  return WaitTriggerAttachment(std::move(trigger),
                               [backend, handle]() { backend->detach(handle); });
}

void WaitSet::set_wait_strategy(const WaitStrategy strategy) noexcept {
  default_strategy_ = strategy;
}

auto WaitSet::wait() noexcept -> expected<WaitResult, Error> {
  return wait_impl(default_strategy_);
}

auto WaitSet::wait(const WaitStrategy strategy) noexcept -> expected<WaitResult, Error> {
  return wait_impl(strategy);
}

auto WaitSet::wait_impl(const WaitStrategy& strategy) noexcept -> expected<WaitResult, Error> {
  auto waited = backend_->wait(strategy, true);
  if (!waited.has_value()) {
    return make_unexpected(Error{waited.error()});
  }
  return waited.value();
}

auto WaitSet::wait_impl_assuming_no_ready(const WaitStrategy& strategy) noexcept
    -> expected<WaitResult, Error> {
  auto waited = backend_->wait(strategy, false);
  if (!waited.has_value()) {
    return make_unexpected(Error{waited.error()});
  }
  return waited.value();
}

std::span<const WaitSetHandle> WaitSet::ready_handles() const noexcept {
  return backend_->ready_handles();
}

std::size_t WaitSet::capacity() const noexcept {
  return backend_->capacity();
}

std::size_t WaitSet::attachment_count() const noexcept {
  return backend_->attachment_count();
}

expected<WaitSet, Error> Context::create_waitset() {
  if (impl_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  auto backend_or = detail::transport::create_wait_set_backend(impl_->transport());
  if (!backend_or.has_value()) {
    HOLOMQ_LOG_ERROR("Context create_waitset failed: {}", error_message(backend_or.error()));
    return holomq::forward_error(std::move(backend_or));
  }
  return WaitSet(
      std::make_shared<detail::transport::WaitSetBackend>(std::move(backend_or).value()));
}

}  // namespace holomq
