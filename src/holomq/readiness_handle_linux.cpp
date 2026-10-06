/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/readiness_handle.hpp>

#include <cerrno>
#include <cstdint>
#include <memory>
#include <new>
#include <utility>

#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>

namespace holomq {

namespace {

[[nodiscard]] Error descriptor_error() noexcept {
  return Error{errno == EMFILE || errno == ENFILE || errno == ENOMEM ? ErrorCode::kResourceExhausted
                                                                     : ErrorCode::kFailure};
}

}  // namespace

class ReadinessHandle::Impl {
 public:
  Impl(std::shared_ptr<void> owner, const int source_descriptor, const AcknowledgeFn acknowledge,
       const DetachFn detach) noexcept
      : owner_(std::move(owner)),
        descriptor_(source_descriptor),
        acknowledge_(acknowledge),
        detach_(detach) {}

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  ~Impl() {
    if (detach_ != nullptr && owner_ != nullptr) {
      detach_(owner_.get());
    }
    if (aggregate_descriptor_ >= 0) {
      static_cast<void>(::close(aggregate_descriptor_));
    }
    if (primer_descriptor_ >= 0) {
      static_cast<void>(::close(primer_descriptor_));
    }
  }

  [[nodiscard]] expected<void, Error> initialize(const bool initially_ready) noexcept {
    aggregate_descriptor_ = ::epoll_create1(EPOLL_CLOEXEC);
    if (aggregate_descriptor_ < 0) {
      return make_unexpected(descriptor_error());
    }

    epoll_event source_event{};
    source_event.events = EPOLLIN;
    source_event.data.fd = descriptor_;
    if (::epoll_ctl(aggregate_descriptor_, EPOLL_CTL_ADD, descriptor_, &source_event) != 0) {
      return make_unexpected(descriptor_error());
    }

    primer_descriptor_ = ::eventfd(initially_ready ? 1U : 0U, EFD_CLOEXEC | EFD_NONBLOCK);
    if (primer_descriptor_ < 0) {
      return make_unexpected(descriptor_error());
    }
    epoll_event primer_event{};
    primer_event.events = EPOLLIN;
    primer_event.data.fd = primer_descriptor_;
    if (::epoll_ctl(aggregate_descriptor_, EPOLL_CTL_ADD, primer_descriptor_, &primer_event) != 0) {
      return make_unexpected(descriptor_error());
    }
    return {};
  }

  [[nodiscard]] int descriptor() const noexcept { return aggregate_descriptor_; }

  [[nodiscard]] expected<void, Error> acknowledge() noexcept {
    auto acknowledged = acknowledge_(owner_.get());

    if (primer_descriptor_ >= 0) {
      std::uint64_t value = 0U;
      while (::read(primer_descriptor_, &value, sizeof(value)) < 0 && errno == EINTR) {
      }
    }
    if (!acknowledged) {
      return make_unexpected(std::move(acknowledged).error());
    }
    if (*acknowledged) {
      constexpr std::uint64_t kReady = 1U;
      ssize_t written = -1;
      do {
        written = ::write(primer_descriptor_, &kReady, sizeof(kReady));
      } while (written < 0 && errno == EINTR);
      if (written != static_cast<ssize_t>(sizeof(kReady))) {
        return make_unexpected(descriptor_error());
      }
    }
    return {};
  }

 private:
  std::shared_ptr<void> owner_;
  int descriptor_{-1};
  int aggregate_descriptor_{-1};
  int primer_descriptor_{-1};
  AcknowledgeFn acknowledge_{};
  DetachFn detach_{};
};

ReadinessHandle::ReadinessHandle(ReadinessHandle&& other) noexcept = default;
ReadinessHandle& ReadinessHandle::operator=(ReadinessHandle&& other) noexcept = default;
ReadinessHandle::~ReadinessHandle() = default;

expected<ReadinessHandle, Error> ReadinessHandle::create(
    std::shared_ptr<void> owner,  // NOLINT(performance-unnecessary-value-param)
    const int source_descriptor, const bool initially_ready, const AcknowledgeFn acknowledge,
    const DetachFn detach) {
  if (owner == nullptr || source_descriptor < 0 || acknowledge == nullptr || detach == nullptr) {
    if (owner != nullptr && detach != nullptr) {
      detach(owner.get());
    }
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  try {
    auto impl = std::make_unique<Impl>(owner, source_descriptor, acknowledge, detach);
    auto initialized = impl->initialize(initially_ready);
    if (!initialized) {
      return make_unexpected(std::move(initialized).error());
    }
    ReadinessHandle handle;
    handle.impl_ = std::move(impl);
    return handle;
  } catch (const std::bad_alloc&) {
    detach(owner.get());
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
}

int ReadinessHandle::descriptor() const noexcept {
  return impl_ == nullptr ? -1 : impl_->descriptor();
}

expected<void, Error> ReadinessHandle::acknowledge() noexcept {
  if (impl_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return impl_->acknowledge();
}

void ReadinessHandle::detach() noexcept {
  impl_.reset();
}

}  // namespace holomq
