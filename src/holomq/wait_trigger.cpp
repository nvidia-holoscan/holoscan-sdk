/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/wait_trigger.hpp>

#include <holomq/logger.hpp>
#include "scoped_file_descriptor.hpp"

#include <cerrno>
#include <cstdint>
#include <memory>
#include <utility>

#include <sys/eventfd.h>
#include <unistd.h>

namespace holomq {

namespace {

constexpr std::uint64_t k_wake_post = 1U;

}  // namespace

/// Private kernel wake object. An eventfd is used because it is the cheapest descriptor-backed
/// counter Linux offers and the iceoryx2 wait set accepts any FileDescriptorBased attachment.
/// EFD_SEMAPHORE is deliberately not set: a latch wants posts to coalesce, not queue.
class WaitTrigger::Impl {
 public:
  explicit Impl(ScopedFileDescriptor descriptor) noexcept : descriptor_(std::move(descriptor)) {}

  [[nodiscard]] int descriptor() const noexcept { return descriptor_.get(); }

 private:
  ScopedFileDescriptor descriptor_;
};

WaitTrigger::WaitTrigger(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

WaitTrigger::~WaitTrigger() = default;

auto WaitTrigger::create() -> expected<std::shared_ptr<WaitTrigger>, Error> {
  ScopedFileDescriptor descriptor{::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)};
  if (!descriptor) {
    HOLOMQ_LOG_ERROR("WaitTrigger create failed: eventfd errno {}", errno);
    return make_unexpected(Error{ErrorCode::kFailure, "failed to create wait trigger wake object"});
  }
  auto impl = std::make_unique<Impl>(std::move(descriptor));
  // Constructed through new because the constructor is private to this class.
  return std::shared_ptr<WaitTrigger>(new WaitTrigger(std::move(impl)));
}

int WaitTrigger::descriptor() const noexcept {
  return impl_ == nullptr ? -1 : impl_->descriptor();
}

expected<void, Error> WaitTrigger::fire() noexcept {
  // Publish the latch first. A waiting thread that checks is_set() before blocking must never be
  // able to observe an un-latched trigger whose descriptor has already been posted.
  if (latched_.exchange(true, std::memory_order_acq_rel)) {
    return {};  // already fired; the post is redundant and the counter stays at one
  }

  const int fd = descriptor();
  if (fd < 0) {
    return make_unexpected(Error{ErrorCode::kFailure, "wait trigger has no wake object"});
  }

  const std::uint64_t post = k_wake_post;
  while (::write(fd, &post, sizeof(post)) < 0) {
    if (errno == EINTR) {
      continue;
    }
    // The latch is already published, so a waiter that rechecks is_set() still observes the
    // request. Only the prompt kernel wake is lost.
    HOLOMQ_LOG_ERROR("WaitTrigger fire failed to post wake object: errno {}", errno);
    return make_unexpected(Error{ErrorCode::kFailure, "failed to post wait trigger wake object"});
  }
  return {};
}

void WaitTrigger::drain() noexcept {
  const int fd = descriptor();
  if (fd < 0) {
    return;
  }
  std::uint64_t sink = 0U;
  while (::read(fd, &sink, sizeof(sink)) > 0) {
    // Drain until the counter is empty; EFD_NONBLOCK ends the loop with EAGAIN.
  }
}

WaitTriggerAttachment::WaitTriggerAttachment(std::shared_ptr<WaitTrigger> trigger,
                                             std::function<void()> detach_fn) noexcept
    : trigger_(std::move(trigger)), detach_fn_(std::move(detach_fn)) {}

WaitTriggerAttachment::WaitTriggerAttachment(WaitTriggerAttachment&& other) noexcept
    : trigger_(std::move(other.trigger_)), detach_fn_(std::move(other.detach_fn_)) {
  other.trigger_ = nullptr;
  other.detach_fn_ = nullptr;
}

WaitTriggerAttachment& WaitTriggerAttachment::operator=(WaitTriggerAttachment&& other) noexcept {
  if (this != &other) {
    if (detach_fn_) {
      detach_fn_();
    }
    trigger_ = std::move(other.trigger_);
    detach_fn_ = std::move(other.detach_fn_);
    other.trigger_ = nullptr;
    other.detach_fn_ = nullptr;
  }
  return *this;
}

WaitTriggerAttachment::~WaitTriggerAttachment() {
  if (detach_fn_) {
    detach_fn_();
  }
}

}  // namespace holomq
