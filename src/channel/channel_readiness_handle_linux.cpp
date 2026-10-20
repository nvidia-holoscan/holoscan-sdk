/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "channel_readiness_handle.hpp"

#include "channel_common.hpp"

#include <cerrno>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <utility>
#include <vector>

#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>

namespace holoscan {

namespace {

[[nodiscard]] Error descriptor_error() noexcept {
  return Error{errno == EMFILE || errno == ENFILE || errno == ENOMEM ? ErrorCode::kResourceExhausted
                                                                     : ErrorCode::kFailure};
}

}  // namespace

class ChannelReadinessHandle::Impl {
 public:
  Impl(std::vector<holomq::ReadinessHandle> holomq_handles,
       std::vector<ChannelReadinessHandle> channel_handles) noexcept
      : holomq_handles_(std::move(holomq_handles)), channel_handles_(std::move(channel_handles)) {}

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  ~Impl() {
    if (aggregate_descriptor_ >= 0) {
      static_cast<void>(::close(aggregate_descriptor_));
    }
    if (primer_descriptor_ >= 0) {
      static_cast<void>(::close(primer_descriptor_));
    }
  }

  [[nodiscard]] expected<void, Error> initialize(const bool initially_ready) noexcept {
    const std::size_t source_count = holomq_handles_.size() + channel_handles_.size();
    if (source_count == 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (source_count == 1U && !initially_ready) {
      descriptor_ = holomq_handles_.empty() ? channel_handles_.front().descriptor()
                                            : holomq_handles_.front().descriptor();
      return descriptor_ >= 0 ? expected<void, Error>{}
                              : make_unexpected(Error{ErrorCode::kFailure});
    }

    aggregate_descriptor_ = ::epoll_create1(EPOLL_CLOEXEC);
    if (aggregate_descriptor_ < 0) {
      return make_unexpected(descriptor_error());
    }
    for (const auto& handle : holomq_handles_) {
      if (auto added = add_source(handle.descriptor()); !added) {
        return added;
      }
    }
    for (const auto& handle : channel_handles_) {
      if (auto added = add_source(handle.descriptor()); !added) {
        return added;
      }
    }

    if (initially_ready) {
      primer_descriptor_ = ::eventfd(1U, EFD_CLOEXEC | EFD_NONBLOCK);
      if (primer_descriptor_ < 0) {
        return make_unexpected(descriptor_error());
      }
      if (auto added = add_source(primer_descriptor_); !added) {
        return added;
      }
    }
    descriptor_ = aggregate_descriptor_;
    return {};
  }

  [[nodiscard]] int descriptor() const noexcept { return descriptor_; }

  [[nodiscard]] expected<void, Error> acknowledge() noexcept {
    std::optional<Error> first_error;
    for (auto& handle : holomq_handles_) {
      auto acknowledged = handle.acknowledge();
      if (!acknowledged && !first_error.has_value()) {
        first_error.emplace(channel_internal::from_holomq_error(std::move(acknowledged).error()));
      }
    }
    for (auto& handle : channel_handles_) {
      auto acknowledged = handle.acknowledge();
      if (!acknowledged && !first_error.has_value()) {
        first_error.emplace(std::move(acknowledged).error());
      }
    }
    if (primer_descriptor_ >= 0) {
      // This primer represents only source-private work observed during handle acquisition. It is
      // intentionally one-shot: retaining a recheck callback would retain or borrow the facade
      // implementation that the handle is designed to outlive. Once acquired, resolving staged
      // work requires a receive-relevant lane operation, and each retained lane handle preserves
      // that operation's wake across acknowledgement.
      std::uint64_t value = 0U;
      while (::read(primer_descriptor_, &value, sizeof(value)) < 0 && errno == EINTR) {
      }
    }
    if (first_error.has_value()) {
      return make_unexpected(std::move(*first_error));
    }
    return {};
  }

 private:
  [[nodiscard]] expected<void, Error> add_source(const int descriptor) noexcept {
    if (descriptor < 0) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    epoll_event event{};
    event.events = EPOLLIN;
    event.data.fd = descriptor;
    if (::epoll_ctl(aggregate_descriptor_, EPOLL_CTL_ADD, descriptor, &event) != 0) {
      return make_unexpected(descriptor_error());
    }
    return {};
  }

  std::vector<holomq::ReadinessHandle> holomq_handles_;
  std::vector<ChannelReadinessHandle> channel_handles_;
  int descriptor_{-1};
  int aggregate_descriptor_{-1};
  int primer_descriptor_{-1};
};

ChannelReadinessHandle::ChannelReadinessHandle(ChannelReadinessHandle&& other) noexcept = default;
ChannelReadinessHandle& ChannelReadinessHandle::operator=(ChannelReadinessHandle&& other) noexcept =
    default;
ChannelReadinessHandle::~ChannelReadinessHandle() = default;

int ChannelReadinessHandle::descriptor() const noexcept {
  return impl_ == nullptr ? -1 : impl_->descriptor();
}

expected<void, Error> ChannelReadinessHandle::acknowledge() noexcept {
  if (impl_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return impl_->acknowledge();
}

void ChannelReadinessHandle::detach() noexcept {
  impl_.reset();
}

expected<ChannelReadinessHandle, Error> channel_internal::ChannelReadinessHandleAccess::create(
    std::vector<holomq::ReadinessHandle> holomq_handles,
    std::vector<ChannelReadinessHandle> channel_handles, const bool initially_ready) {
  try {
    auto impl = std::make_unique<ChannelReadinessHandle::Impl>(std::move(holomq_handles),
                                                               std::move(channel_handles));
    auto initialized = impl->initialize(initially_ready);
    if (!initialized) {
      return make_unexpected(std::move(initialized).error());
    }
    ChannelReadinessHandle handle;
    handle.impl_ = std::move(impl);
    return handle;
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
}

}  // namespace holoscan
